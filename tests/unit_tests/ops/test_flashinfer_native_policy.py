# SPDX-License-Identifier: Apache-2.0
"""Whole-operation native policy, independent of hardware availability."""

import inspect
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer_gaudi import clear_backend_policy_override, set_backend_policy, silu_and_mul
from flashinfer_gaudi._config import bridge_auto_enabled
from flashinfer_gaudi._dispatch import BackendUnavailableError
from flashinfer_gaudi._qualification import qualify
from flashinfer_gaudi._tactics import public_gdn_auto_promoted
from flashinfer_gaudi.gdn_decode import (
    _call_native_packed,
    gated_delta_rule_decode,
    gated_delta_rule_decode_packed,
    gated_delta_rule_decode_pretranspose,
    gated_delta_rule_mtp,
)
from flashinfer_gaudi.gdn_fused_decode import gdn_fused_decode_step
from flashinfer_gaudi.gdn_prefill import _chunk_gated_delta_rule_log_gate, chunk_gated_delta_rule


@pytest.fixture(autouse=True)
def reset_policy():
    clear_backend_policy_override()
    yield
    clear_backend_policy_override()


REFERENCE_ENTRIES = (
    _call_native_packed,
    gated_delta_rule_decode,
    gated_delta_rule_decode_packed,
    gated_delta_rule_decode_pretranspose,
    gated_delta_rule_mtp,
    gdn_fused_decode_step,
    _chunk_gated_delta_rule_log_gate,
    chunk_gated_delta_rule,
)


@pytest.mark.parametrize("policy", ("native", "public", "bridge"))
@pytest.mark.parametrize("entry", REFERENCE_ENTRIES)
def test_strict_rejects_before_any_tensor_access_or_reference(entry, policy):
    set_backend_policy(policy)
    # Opaque inputs prove the guard occurs before validation, allocations,
    # gating, casts, index reads, or state mutation, even on malformed input.
    arguments = {
        name: object()
        for name, p in inspect.signature(entry).parameters.items() if p.default is inspect.Parameter.empty
    }
    with pytest.raises(BackendUnavailableError, match="decomposition is forbidden"):
        entry(**arguments)


@pytest.mark.parametrize("policy", ("native", "public", "bridge"))
def test_activation_never_falls_back_on_cpu(policy):
    set_backend_policy(policy)
    with mock.patch("torch.nn.functional.silu", side_effect=AssertionError("reference ran")), \
         pytest.raises(BackendUnavailableError):
        silu_and_mul(torch.zeros(1, 256, dtype=torch.bfloat16))


def test_native_activation_invokes_only_registered_op():
    set_backend_policy("native")
    tensor = SimpleNamespace(ndim=2,
                             shape=(8, 34816),
                             dtype=torch.bfloat16,
                             device=torch.device("hpu"),
                             requires_grad=False,
                             is_contiguous=lambda: True)
    native = mock.Mock(return_value=mock.sentinel.output)
    with mock.patch("flashinfer_gaudi.activation.silu_and_mul_op", return_value=native), \
         mock.patch("torch.nn.functional.silu", side_effect=AssertionError("reference ran")):
        assert silu_and_mul(tensor) is mock.sentinel.output
    native.assert_called_once_with(tensor)


def test_native_failure_is_not_retried():
    set_backend_policy("native")
    tensor = SimpleNamespace(ndim=2,
                             shape=(1, 256),
                             dtype=torch.bfloat16,
                             device=torch.device("hpu"),
                             requires_grad=False,
                             is_contiguous=lambda: True)
    native = mock.Mock(side_effect=RuntimeError("device failure"))
    with mock.patch("flashinfer_gaudi.activation.silu_and_mul_op", return_value=native), \
         pytest.raises(RuntimeError, match="device failure"):
        silu_and_mul(tensor)
    assert native.call_count == 1


def test_auto_keeps_unqualified_activation_on_reference():
    x = torch.randn(2, 256, dtype=torch.bfloat16)
    out = torch.empty(2, 128, dtype=x.dtype)
    with mock.patch("flashinfer_gaudi.activation.silu_and_mul_op", side_effect=AssertionError("unqualified native")):
        assert silu_and_mul(x, out=out) is out
    torch.testing.assert_close(out, torch.nn.functional.silu(x[:, :128]) * x[:, 128:])


def test_override_flags_do_not_promote_prototypes(monkeypatch):
    monkeypatch.setenv("FLASHINFER_GAUDI_ENABLE_PUBLIC_AUTO", "1")
    monkeypatch.setenv("FLASHINFER_GAUDI_ENABLE_BRIDGE_AUTO", "1")
    assert not public_gdn_auto_promoted()
    assert not bridge_auto_enabled()


@pytest.mark.parametrize("policy", ("native", "public", "bridge"))
@pytest.mark.parametrize("name",
                         ("maybe_run_gdn_prefill", "maybe_run_gdn_decode_packed", "maybe_run_gdn_fused_decode_step"))
def test_enabled_vllm_adapter_cannot_escape_strict_policy(policy, name):
    from vllm_gaudi.ops import flashinfer_gaudi_adapter as adapter

    set_backend_policy(policy)
    entry = getattr(adapter, name)
    arguments = {
        key: object()
        for key, parameter in inspect.signature(entry).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    with mock.patch.object(adapter, "flashinfer_gdn_enabled", return_value=True), \
         pytest.raises(BackendUnavailableError):
        entry(**arguments)


def test_activation_missing_library_cannot_fall_back():
    set_backend_policy("native")
    tensor = SimpleNamespace(ndim=2,
                             shape=(1, 256),
                             dtype=torch.bfloat16,
                             requires_grad=False,
                             device=torch.device("hpu"),
                             is_contiguous=lambda: True)
    with mock.patch("flashinfer_gaudi.activation.silu_and_mul_op", return_value=None), \
         pytest.raises(BackendUnavailableError, match="no fallback"):
        silu_and_mul(tensor)


def sessions(speedup=1.2):
    return [{"session_id": i, "reference_ms": [1.0] * 15, "candidate_ms": [1.0 / speedup] * 15} for i in range(3)]


def test_qualification_requires_correctness_and_whole_native():
    for correct, native in ((False, True), (True, False)):
        assert not qualify([], correctness=correct, whole_operation_native=native)["qualified"]


@pytest.mark.parametrize("speedup,expected", ((1.2, True), (1.04, False), (.9, False)))
def test_qualification_threshold(speedup, expected):
    result = qualify(sessions(speedup), correctness=True, whole_operation_native=True, bootstrap_samples=1000)
    assert result["qualified"] is expected


def test_qualification_rejects_missing_sessions_and_invalid_samples():
    assert not qualify(sessions()[:1], correctness=True, whole_operation_native=True)["qualified"]
    values = sessions()
    values[0]["candidate_ms"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        qualify(values, correctness=True, whole_operation_native=True)
    values = sessions()
    values[1]["session_id"] = values[0]["session_id"]
    with pytest.raises(ValueError, match="distinct"):
        qualify(values, correctness=True, whole_operation_native=True)


def test_qualification_does_not_hide_a_slow_process_session():
    values = sessions(1.2)
    values[0]["candidate_ms"] = [1.3] * 15
    result = qualify(values, correctness=True, whole_operation_native=True, bootstrap_samples=1000)
    assert result["speedup"] > 1.05
    assert result["confidence_interval_95"][0] < 1.0
    assert not result["qualified"]
