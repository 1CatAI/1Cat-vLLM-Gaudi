# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the V4.1 Q/KV input projection fusion."""

import torch
from torch import nn
import torch.nn.functional as F

from vllm_gaudi.ops.deepseek_v41_attention import CSA2Attention
from vllm_gaudi.ops.deepseek_v41_math import quantize_activation


def _matrix(rows, cols, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator, dtype=torch.bfloat16)


def _weight(rows, cols, seed):
    module = nn.Module()
    module.register_buffer("weight", _matrix(rows, cols, seed), False)
    # Dense prepared weights carry the scale plane as a marker for the
    # existing block-quantized activation contract.
    module.register_buffer("scale", torch.ones((1, ), dtype=torch.uint8), False)
    return module


def _attention(weights):
    config = {
        "index_topk": 512,
        "kv_source_layer_ids": [],
        "index_source_layer_ids": [],
        "candidate_source_layer_id": 99,
        "candidate_topk_blocks": 8,
        "candidate_block_size": 64,
        "compress_ratios": [0],
        "num_hidden_layers": 40,
        "num_attention_heads": 4,
        "o_groups": 8,
        "rms_norm_eps": 1e-6,
        "sliding_window": 128,
        "qk_rope_head_dim": 4,
        "compress_rope_theta": 10000.0,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "original_max_position_embeddings": 512,
            "factor": 1.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0
        },
    }
    shared = nn.Module()
    shared.length = 512
    shared.layer_start = 0
    shared.decoded_kv_state = False
    shared.sources = nn.ModuleDict()
    shared.topk = nn.ModuleDict()
    return CSA2Attention(weights, config, 0, shared, _linear, lambda x: x, "cpu")


def _linear(value, layer):
    if hasattr(layer, "scale"):
        value = quantize_activation(value)
    return F.linear(value, layer.weight)


def _weights():
    weights = nn.Module()
    weights.wq_a = _weight(8, 32, 100)
    weights.wkv = _weight(4, 32, 200)
    return weights


def test_fused_qkv_matches_two_projection_contract():
    value = _matrix(1, 32, 300)
    separate = _attention(_weights())
    fused_weights = _weights()
    fused = _attention(fused_weights)
    fused.qkv_fused_input = True
    fused.prepare_qkv_input_weight()

    expected = separate._project_qkv_input(value)
    actual = fused._project_qkv_input(value)
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])
    assert fused.fused_wqa_wkv.shape == (12, 32)
    assert fused.weights.wq_a.weight.untyped_storage().data_ptr() == fused.fused_wqa_wkv.untyped_storage().data_ptr()
    assert fused.weights.wkv.weight.untyped_storage().data_ptr() == fused.fused_wqa_wkv.untyped_storage().data_ptr()
    assert fused.weights.wkv.weight.storage_offset() == 8 * 32


def test_fused_qkv_requires_bf16_and_matching_k():
    weights = _weights()
    weights.wkv.weight = weights.wkv.weight[:, :-1]
    attention = _attention(weights)
    attention.qkv_fused_input = True
    try:
        attention.prepare_qkv_input_weight()
    except ValueError as exc:
        assert "matching input K" in str(exc)
    else:
        raise AssertionError("shape mismatch must reject QKV fusion")


def test_fused_qkv_can_be_invalidated_for_reload():
    attention = _attention(_weights())
    attention.qkv_fused_input = True
    attention.prepare_qkv_input_weight()
    attention.invalidate_qkv_input_weight()
    assert attention._fused_qkv_weight is None
    assert "fused_wqa_wkv" not in attention._buffers
