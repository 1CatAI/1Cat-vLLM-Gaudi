# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm_gaudi.ops.deepseek_v41_recipe_constants import immutable_bindings


def branch(**buffers):
    result = torch.nn.Module()
    for name, value in buffers.items():
        result.register_buffer(name, value)
    return result


def fixture():
    tensor = lambda: torch.empty((8, 8), device="meta")
    weights = branch(hc_attn_fn=tensor())
    weights.ffn = branch()
    weights.ffn.gate = branch(weight=tensor())
    weights.ffn.experts = branch(w13_q16=tensor(), w2_s16=tensor())
    weights.ffn.shared_experts = branch()
    weights.ffn.shared_experts.w1 = branch(weight=tensor())
    weights.engram = branch(table=tensor(), wkv=tensor())
    attention = branch(fused_wqa_wkv=tensor(), swa=tensor(), decoded_main=tensor(), positions=tensor())
    return SimpleNamespace(weights=weights, attention=attention, hc_attn_fn_mme=tensor(),
                           hc_attn_fn_packed=None, hc_ffn_fn_packed=None, hc_ffn_fn_mme=None)


def test_no_request_cache_or_expert_bank_can_be_marked():
    bindings = immutable_bindings(fixture())
    names = {row[3] for row in bindings}
    assert names == {"weights.hc_attn_fn", "weights.ffn.gate.weight",
                     "weights.ffn.shared_experts.w1.weight", "hc_attn_fn_mme", "attention.fused_wqa_wkv"}


def test_enumeration_does_not_replace_or_mark_baseline_tensors():
    layer = fixture()
    original = layer.weights.ffn.gate.weight
    bindings = immutable_bindings(layer)
    gate = next(row for row in bindings if row[3] == "weights.ffn.gate.weight")
    assert gate[0] is layer.weights.ffn.gate and gate[1] == "weight" and gate[2] is original
    assert layer.weights.ffn.gate.weight is original


def test_shared_expert_is_distinct_from_routed_bank():
    layer = fixture()
    names = {row[3] for row in immutable_bindings(layer)}
    assert "weights.ffn.shared_experts.w1.weight" in names
    assert all(not name.startswith("weights.ffn.experts.") for name in names)
