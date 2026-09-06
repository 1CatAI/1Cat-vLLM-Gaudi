# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from unittest.mock import Mock

from vllm_gaudi.distributed.tp2_fused_ar_norm import (
    _MAX_FUSED_DECODE_TOKENS,
    _exceeds_fused_decode_token_limit,
    _rejection_reason,
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
