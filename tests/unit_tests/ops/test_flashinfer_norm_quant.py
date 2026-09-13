# SPDX-License-Identifier: Apache-2.0
"""Policy, functional-output and native trace contracts for fused norm/quant."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer_gaudi import clear_backend_policy_override, fused_add_rmsnorm_quant, set_backend_policy
from flashinfer_gaudi._dispatch import BackendUnavailableError
from tools.benchmark_flashinfer_native_norm import audit_trace, qualify_case


@pytest.fixture(autouse=True)
def reset_policy():
    clear_backend_policy_override()
    yield
    clear_backend_policy_override()


def _inputs(batch=2, width=256):
    generator = torch.Generator().manual_seed(19)
    return (torch.randn(batch, width, generator=generator).bfloat16(), torch.randn(batch, width,
                                                                                   generator=generator).bfloat16(),
            torch.randn(width, generator=generator).bfloat16())


@pytest.mark.parametrize("policy", ["native", "public", "bridge"])
def test_strict_cpu_never_decomposes(policy):
    set_backend_policy(policy)
    with mock.patch("flashinfer_gaudi.norm._reference") as reference, pytest.raises(BackendUnavailableError):
        fused_add_rmsnorm_quant(*_inputs())
    reference.assert_not_called()


@pytest.mark.parametrize("epsilon", [-1., 0., 1e-40, 1e40, float("nan"), float("inf")])
def test_invalid_epsilon_rejected_before_computation(epsilon):
    with mock.patch("flashinfer_gaudi.norm._reference") as reference, pytest.raises(ValueError, match="epsilon"):
        fused_add_rmsnorm_quant(*_inputs(), eps=epsilon)
    reference.assert_not_called()


def test_invalid_scale_mode_rejected_before_computation():
    with mock.patch("flashinfer_gaudi.norm._reference") as reference, pytest.raises(ValueError, match="scale_mode"):
        fused_add_rmsnorm_quant(*_inputs(), scale_mode="automatic")
    reference.assert_not_called()


@pytest.mark.parametrize("scale_mode,inverse_range", [("bf16_reciprocal", 0.004180908203125),
                                                      ("fp32_divide", 1.0 / 240.0)])
def test_scale_mode_has_explicit_rounding_contract(scale_mode, inverse_range):
    set_backend_policy("pytorch")
    outputs = fused_add_rmsnorm_quant(*_inputs(), scale_mode=scale_mode)
    expected = ((outputs[2].abs().amax(-1, keepdim=True) + 1e-8).float() * inverse_range).bfloat16().float()
    torch.testing.assert_close(outputs[1], expected, rtol=0, atol=0)


@pytest.mark.parametrize("factor", [1e-5, .01, .1, .3, 1., 3., 10., 1000.])
def test_normalized_amax_can_be_reduced_before_positive_scaling(factor):
    values = torch.randn(32, 17408, generator=torch.Generator().manual_seed(257)).bfloat16()
    values[0] = 0
    values[:, :5] = torch.tensor([0., -0., 1., -1., .0078125]).bfloat16()
    actual = (values.float() * factor).bfloat16().abs().amax(-1, keepdim=True)
    reduced = (values.abs().amax(-1, keepdim=True).float() * factor).bfloat16()
    torch.testing.assert_close(actual, reduced, rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["residual_shape", "weight_shape", "dtype", "grad", "empty"])
def test_input_contract_rejected_without_mutation(bad):
    x, residual, weight = _inputs()
    if bad == "residual_shape":
        residual = residual[:1]
    elif bad == "weight_shape":
        weight = weight[:128]
    elif bad == "dtype":
        weight = weight.float()
    elif bad == "grad":
        x.requires_grad_(True)
    else:
        x, residual = x[:0], residual[:0]
    before = tuple(value.clone() for value in (x, residual, weight))
    with pytest.raises((ValueError, BackendUnavailableError)):
        fused_add_rmsnorm_quant(x, residual, weight)
    for value, expected in zip((x, residual, weight), before):
        torch.testing.assert_close(value, expected, atol=0, rtol=0)


@pytest.mark.parametrize("epsilon", [1e-6, 1e-4])
@pytest.mark.parametrize("amplitude", [0., 1e-5, 1., 8.])
def test_functional_reference_independent_norm_oracle(epsilon, amplitude):
    set_backend_policy("pytorch")
    x, residual, weight = _inputs()
    x, residual = (x * amplitude).bfloat16(), (residual * amplitude).bfloat16()
    before = tuple(value.clone() for value in (x, residual, weight))
    q, scale, normed, summed = fused_add_rmsnorm_quant(x, residual, weight, eps=epsilon)
    expected_sum = (x.float() + residual.float()).bfloat16()
    fp64 = expected_sum.double()
    weighted = (expected_sum * weight).double()
    expected_norm = (weighted / torch.sqrt(fp64.square().mean(-1, keepdim=True) + epsilon)).bfloat16()
    torch.testing.assert_close(summed, expected_sum, rtol=0, atol=0)
    torch.testing.assert_close(normed, expected_norm, rtol=.02, atol=.002)
    assert q.dtype == torch.float8_e4m3fn and q.shape == x.shape
    assert scale.dtype == torch.float32 and scale.shape == (2, 1) and (scale > 0).all()
    if amplitude == 0:
        assert not torch.count_nonzero(q.float()) and not torch.count_nonzero(normed)
    for value, expected in zip((x, residual, weight), before):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert all(output.data_ptr() not in {value.data_ptr()
                                         for value in (x, residual, weight)} for output in (q, scale, normed, summed))


@pytest.mark.parametrize("policy", ["native", "public"])
def test_missing_and_failed_native_never_retry_reference(policy):
    set_backend_policy(policy)
    device = SimpleNamespace(type="hpu")
    common = dict(ndim=2, dtype=torch.bfloat16, requires_grad=False, device=device, is_contiguous=lambda: True)
    x = SimpleNamespace(shape=(2, 256), **common)
    weight = SimpleNamespace(shape=(256, ), **common)
    with mock.patch("flashinfer_gaudi.norm._reference") as reference:
        with mock.patch("flashinfer_gaudi.norm.add_rmsnorm_quant_op",
                        return_value=None), pytest.raises(BackendUnavailableError, match="unavailable"):
            fused_add_rmsnorm_quant(x, x, weight)
        with mock.patch("flashinfer_gaudi.norm.add_rmsnorm_quant_op",
                        return_value=mock.Mock(side_effect=RuntimeError("native error"))), pytest.raises(
                            RuntimeError, match="native error"):
            fused_add_rmsnorm_quant(x, x, weight)
    reference.assert_not_called()


@pytest.mark.parametrize("violation", [None, "extra_kernel", "aten::mul", "hpu::cast_to_fp8_v2", "extra_launch"])
def test_native_trace_covers_whole_operation_and_recipe(violation):
    events = [{"cat": "kernel", "name": "flashinfer_gaudi_add_rmsnorm_quant_bf16_gaudi2"}]
    events.extend({"cat": "privateuse1_runtime", "name": "Launch"} for _ in range(5))
    if violation is None:
        assert audit_trace(events)["launch_events"] == 5
    else:
        if violation == "extra_kernel":
            events.append({"cat": "kernel", "name": "rms_norm"})
        elif violation == "extra_launch":
            events.append({"cat": "privateuse1_runtime", "name": "Launch"})
        else:
            events.append({"cat": "cpu_op", "name": violation})
        with pytest.raises(RuntimeError, match="audit failed"):
            audit_trace(events)


@pytest.mark.parametrize("missing", [None, "trace", "fx", "artifact", "correctness", "fallback", "worker"])
def test_qualification_requires_device_evidence_and_every_process(missing):
    sessions = [{
        "session_id": str(index),
        "artifacts_unchanged": True,
        "cross_shape_reentry": True,
        "sha256": {
            "kernel": "same-elf"
        },
        "use_eager_fallback": False,
        "native_fx_graphs": [["native"]],
        "cases": {
            "1x256": {
                "correctness": True,
                "whole_operation_native": index == 0,
                "traces": {
                    "native": {
                        "launch_events": 5
                    }
                } if index == 0 else {},
                "native_ms": [.1] * 15,
                "vendor_ms": [.2] * 15,
                "formula_ms": [.2] * 15,
                "cguid_ms": [.2] * 15
            }
        }
    } for index in range(3)]
    errors = []
    if missing == "trace":
        sessions[0]["cases"]["1x256"]["traces"] = {}
    elif missing == "fx":
        sessions[1]["native_fx_graphs"] = []
    elif missing == "artifact":
        sessions[2]["sha256"]["kernel"] = "changed"
    elif missing == "correctness":
        sessions[2]["cases"]["1x256"]["correctness"] = False
    elif missing == "fallback":
        sessions[1]["use_eager_fallback"] = True
    elif missing == "worker":
        errors = [{"returncode": 1}]
    assert qualify_case(sessions, "1x256", 3, errors)["qualified"] is (missing is None)
