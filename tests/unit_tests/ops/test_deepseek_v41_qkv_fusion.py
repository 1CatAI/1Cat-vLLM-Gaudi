# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the V4.1 Q/KV input projection fusion."""

import torch
from torch import nn
import torch.nn.functional as F

from vllm_gaudi.ops.deepseek_v41_attention import CSA2Attention
from vllm_gaudi.ops.deepseek_v41_math import quantize_activation
from vllm_gaudi.ops.deepseek_v41_qkv import FusedCompressorInput


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


class _CompressorProjection(FusedCompressorInput, nn.Module):

    def __init__(self, *, ratio=2, owns_kv=True):
        super().__init__()
        self.weights = nn.Module()
        self.weights.compressor = nn.Module()
        self.weights.compressor.wkv = _plain_weight(12, 32, 501)
        self.weights.compressor.wgate = _plain_weight(12, 32, 502)
        self.linear = lambda value, layer: F.linear(value, layer.weight)
        self.compressor_fused_input = True
        self._fused_compressor_weight = None
        self._fused_compressor_kv_width = 0
        self.ratio = ratio
        self.owns_kv = owns_kv


def _plain_weight(rows, cols, seed):
    module = nn.Module()
    module.register_buffer("weight", torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed)), False)
    return module


def test_fused_compressor_matches_two_fp32_projections():
    projection = _CompressorProjection()
    value = torch.randn(3, 32, generator=torch.Generator().manual_seed(503)).bfloat16()
    expected = projection._project_compressor_input(value)
    projection.prepare_compressor_input_weight()
    actual = projection._project_compressor_input(value)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    fused = projection.fused_compressor_wkv_wgate
    assert fused.shape == (24, 32)
    assert projection.weights.compressor.wkv.weight.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
    assert projection.weights.compressor.wgate.weight.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
    assert projection.weights.compressor.wgate.weight.storage_offset() == 12 * 32


def test_fused_compressor_skips_non_owner_and_invalidates_for_reload():
    projection = _CompressorProjection(owns_kv=False)
    projection.prepare_compressor_input_weight()
    assert projection._fused_compressor_weight is None

    projection = _CompressorProjection()
    projection.prepare_compressor_input_weight()
    projection.invalidate_compressor_input_weight()
    assert projection._fused_compressor_weight is None
    assert "fused_compressor_wkv_wgate" not in projection._buffers
