# SPDX-License-Identifier: Apache-2.0
"""SAT qualification must inspect compact channels and exclude only proven padding."""
import numpy as np
import pytest

from vllm_gaudi.ops.deepseek_v41_expert_n256 import saturated_decode_eligible


def test_compact_offsets_and_padded_576_to_640_boundary():
    data = np.full((2, 21, 256), 120, dtype=np.uint8)
    data[:, :18] = 115
    data[:, 18:20] = 0
    planes = data.view("<i2").reshape(2, 2688)
    assert saturated_decode_eligible(planes, active_k=576)
    assert not saturated_decode_eligible(planes, active_k=640)
    data[0, 17, 0] = 114
    assert not saturated_decode_eligible(planes, active_k=576)


def test_full_plane_offsets_keep_original_sibling_contract():
    data = np.zeros((2, 4, 512), dtype=np.uint8)
    data[:, :, 256:] = np.array(-40, dtype=np.int8).view(np.uint8)
    planes = data.view("<i2").reshape(2, 1024)
    assert saturated_decode_eligible(planes)
    data[1, 3, 511] = 56
    assert not saturated_decode_eligible(planes)


@pytest.mark.parametrize("active_k", [0, 577, 672])
def test_invalid_active_prefix_rejected(active_k):
    with pytest.raises(ValueError):
        saturated_decode_eligible(np.zeros((2, 2688), dtype="<i2"), active_k=active_k)


def test_capability_refreshes_after_loading_and_is_cleared_on_retirement(monkeypatch):
    from types import SimpleNamespace
    import torch
    from vllm_gaudi.models.deepseek_v41_program import PreparedMoE
    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_expert_n256_moe_token_wide_sat_fp8_gaudi2",
                        lambda *args: None, raising=False)
    moe = PreparedMoE.__new__(PreparedMoE)
    torch.nn.Module.__init__(moe)
    moe.weights = SimpleNamespace(experts=SimpleNamespace(w13_q16=torch.empty(1), w2_q16=torch.empty(1)))
    moe.n256_fused, moe.shared_gate_up = True, False
    moe.refresh_sat_eligibility()
    assert not moe.c6_token_wide_sat
    for weight in (moe.weights.experts.w13_q16, moe.weights.experts.w2_q16):
        weight.dsv41_sat_eligible = True
    moe.prepare_shared_gate_up_weight()
    assert moe.c6_token_wide_sat
    moe.release_shared_gate_up_weight()
    assert not moe.c6_token_wide_sat
    moe.weights.experts.w2_q16.dsv41_sat_eligible = False
    moe.prepare_shared_gate_up_weight()
    assert not moe.c6_token_wide_sat
