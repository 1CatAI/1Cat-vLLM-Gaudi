# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from unittest.mock import Mock

from vllm_gaudi.distributed.tp2_fused_ar_norm import (
    _MAX_FUSED_DECODE_TOKENS,
    _exceeds_fused_decode_token_limit,
    _rejection_reason,
    _native_probe_valid,
)


def test_fused_ar_norm_rejects_prefill_before_device_checks():
    partial = torch.empty((1, 8), dtype=torch.bfloat16)
    residual = torch.empty_like(partial)
    weight = torch.empty(8, dtype=torch.bfloat16)

    assert _rejection_reason(
        partial,
        residual,
        weight,
        is_prompt=True,
        max_bytes=1024,
    ) == "prefill"


def test_fused_ar_norm_rejects_non_hpu_input():
    partial = torch.empty((1, 8), dtype=torch.bfloat16)
    residual = torch.empty_like(partial)
    weight = torch.empty(8, dtype=torch.bfloat16)

    assert _rejection_reason(
        partial,
        residual,
        weight,
        is_prompt=False,
        max_bytes=1024,
    ) == "non-HPU input"


def test_fused_ar_norm_decode_cost_gate_matches_profitable_bucket():
    assert _MAX_FUSED_DECODE_TOKENS == 20
    weight = torch.empty(8, dtype=torch.bfloat16)
    assert not _exceeds_fused_decode_token_limit(torch.empty((20, 8)), weight)
    assert _exceeds_fused_decode_token_limit(torch.empty((21, 8)), weight)


def test_explicit_stock_boundary_never_enters_native_fusion(monkeypatch):
    import vllm_gaudi.distributed.tp2_fused_ar_norm as fused_module

    partial = torch.empty((1, 8), dtype=torch.bfloat16)
    residual = torch.empty_like(partial)
    weight = torch.ones(8, dtype=torch.bfloat16)
    fallback = Mock(return_value=(partial, residual))
    runtime = Mock(side_effect=AssertionError("Native runtime must not be used"))
    monkeypatch.setattr(fused_module, "_fallback", fallback)
    monkeypatch.setattr(fused_module, "_resolve_runtime", runtime)
    monkeypatch.setattr(fused_module, "get_config", runtime)

    result = fused_module.tp2_allreduce_residual_rms_norm(partial,
                                                          residual,
                                                          weight,
                                                          1e-6,
                                                          is_prompt=False,
                                                          allow_fused=False)

    assert result[0] is partial
    assert result[1] is residual
    fallback.assert_called_once_with(partial, residual, weight, 1e-6)
    runtime.assert_not_called()


def test_direct_boundary_bypasses_collective_operator(monkeypatch):
    import vllm_gaudi.distributed.tp2_fused_ar_norm as fused_module

    partial = torch.empty((1, 8), dtype=torch.bfloat16)
    residual = torch.empty_like(partial)
    weight = torch.ones(8, dtype=torch.bfloat16)
    expected = (torch.full_like(partial, 3), torch.full_like(partial, 5))
    direct = Mock(return_value=expected)
    runtime = Mock(side_effect=AssertionError("CollectiveOperator runtime must not be used"))
    fallback = Mock(side_effect=AssertionError("Stock fallback must not be used"))
    monkeypatch.setattr(fused_module, "_rejection_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(fused_module, "_resolve_runtime", runtime)
    monkeypatch.setattr(fused_module, "_fallback", fallback)
    monkeypatch.setattr(fused_module, "get_config", lambda: Mock(tp2_fused_ar_norm_max_bytes=524288))
    monkeypatch.setattr(torch.ops.vllm_gaudi, "tp2_allreduce_residual_rms_norm", direct)

    result = fused_module.tp2_allreduce_residual_rms_norm(partial,
                                                          residual,
                                                          weight,
                                                          1e-6,
                                                          is_prompt=False,
                                                          prefer_direct=True)

    assert result is expected
    direct.assert_called_once_with(partial, residual, weight, 1e-6)
    runtime.assert_not_called()
    fallback.assert_not_called()


def test_multitoken_direct_boundary_uses_exchange_only_path(monkeypatch):
    import vllm_gaudi.distributed.tp2_fused_ar_norm as fused_module

    partial = torch.empty((2, 8), dtype=torch.bfloat16)
    residual = torch.empty_like(partial)
    weight = torch.ones(8, dtype=torch.bfloat16)
    expected = (torch.full_like(partial, 3), torch.full_like(partial, 5))
    exchange = Mock(return_value=expected)
    fused = Mock(side_effect=AssertionError("Multi-token input must not compile a second fused recipe"))
    monkeypatch.setattr(fused_module, "_rejection_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(fused_module, "_tp2_exchange_residual_rms_norm", exchange)
    monkeypatch.setattr(fused_module, "get_config", lambda: Mock(tp2_fused_ar_norm_max_bytes=524288))
    monkeypatch.setattr(torch.ops.vllm_gaudi, "tp2_allreduce_residual_rms_norm", fused)

    result = fused_module.tp2_allreduce_residual_rms_norm(
        partial,
        residual,
        weight,
        1e-6,
        is_prompt=False,
        prefer_direct=True,
    )

    assert result is expected
    exchange.assert_called_once_with(partial, residual, weight, 1e-6)
    fused.assert_not_called()


@pytest.mark.parametrize("failure", [
    None, "nan", "inf", "wrong_norm", "wrong_residual", "missing_launch", "extra_launch", "wrong_shape", "wrong_dtype"
])
def test_native_probe_fails_closed(failure):
    expected = (torch.ones((1, 2, 8), dtype=torch.bfloat16), torch.full((1, 2, 8), 2, dtype=torch.bfloat16))
    actual = [value.clone() for value in expected]
    launches = 1
    if failure == "nan":
        actual[0][0, 0, 0] = float("nan")
    elif failure == "inf":
        actual[1][0, 0, 0] = float("inf")
    elif failure == "wrong_norm":
        actual[0].zero_()
    elif failure == "wrong_residual":
        actual[1][0, 0, 0] += 0.015625
    elif failure == "missing_launch":
        launches = 0
    elif failure == "extra_launch":
        launches = 2
    elif failure == "wrong_shape":
        actual[0] = actual[0].reshape(2, 8)
    elif failure == "wrong_dtype":
        actual[0] = actual[0].float()
    assert _native_probe_valid(actual, expected, launches) is (failure is None)
