# SPDX-License-Identifier: Apache-2.0
"""The native descriptor consumes the original key layout without changing math."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.gqa_compact import compact_gqa_matmul, native_gqa_matmul


@pytest.fixture
def native_reference(monkeypatch):
    calls = []

    def multiply(x, y, transpose=False):
        calls.append((y, transpose))
        return torch.matmul(x, y.transpose(-2, -1) if transpose else y)

    monkeypatch.setattr(torch.ops.custom_op, "tp2_gqa_matmul", multiply, raising=False)
    return calls


@pytest.mark.parametrize("blocks", [1, 4, 32])
def test_interleaved_keys_remain_aliases_and_both_products_are_exact(native_reference, blocks):
    rng = torch.Generator().manual_seed(741)
    pool = torch.randn(blocks, 128, 2, 256, generator=rng).bfloat16()
    key = pool.transpose(1, 2).unsqueeze(2)
    query = torch.randn(blocks, 2, 6, 1, 256, generator=rng).bfloat16()
    score = native_gqa_matmul(query, key, transpose_rhs=True)
    assert torch.equal(score, compact_gqa_matmul(query, key.transpose(-2, -1)))
    assert native_reference[0][0].untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()
    assert native_reference[0][0].stride(-1) == 1 and native_reference[0][1]
    value_result = native_gqa_matmul(score, key)
    assert torch.equal(value_result, compact_gqa_matmul(score, key))
    assert not native_reference[1][1]


def test_invalid_key_rejected_before_operator(native_reference):
    query = torch.zeros(4, 2, 6, 1, 256, dtype=torch.bfloat16)
    key = torch.zeros(4, 2, 1, 128, 255, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="matching BF16"):
        native_gqa_matmul(query, key, transpose_rhs=True)
    assert not native_reference


def test_normal_attention_entry_preserves_scores_and_block_reductions(monkeypatch, native_reference):
    from vllm_gaudi.extension import ops

    monkeypatch.setattr(ops, "is_hpu_gaudi2", True)
    monkeypatch.setattr(
        ops, "get_config", lambda: SimpleNamespace(fp32_softmax=False,
                                                   fused_block_softmax=False,
                                                   fused_block_softmax_adjustment=False,
                                                   per_token_kv_scaling_support=False))
    monkeypatch.setenv("VLLM_HPU_TP2_GQA_COMPACT_KV", "1")
    monkeypatch.setenv("VLLM_HPU_TP2_SINGLE_BATCH_MAPPING", "1")
    rng = torch.Generator().manual_seed(46)
    key = torch.randn(8 * 128, 2, 256, generator=rng).bfloat16()
    value = torch.randn_like(key)
    for step in range(4):
        query = torch.randn(1, 1, 3072, generator=rng).bfloat16()
        blocks = (torch.arange(4) + step) % 8
        bias = torch.zeros(4, 128, dtype=torch.bfloat16)
        bias[-1, 64:] = -1000
        kwargs = dict(query=query,
                      key_cache=key,
                      value_cache=value,
                      block_list=blocks,
                      block_mapping=torch.ones(4, 1, dtype=torch.bfloat16),
                      block_bias=bias,
                      block_groups=torch.zeros(4, dtype=torch.int32),
                      block_size=128,
                      scale=.0625,
                      matmul_qk_op=torch.matmul,
                      matmul_av_op=torch.matmul,
                      batch2block_matmul_op=torch.matmul,
                      block2batch_matmul_op=torch.matmul,
                      position_bias=None,
                      sinks=None,
                      k_scales=None,
                      v_scales=None,
                      keys_fetch_func=lambda cache, blocks: cache.index_select(0, blocks),
                      values_fetch_func=lambda cache, blocks: cache.index_select(0, blocks))
        monkeypatch.setenv("VLLM_HPU_TP2_GQA_NATIVE_MATMUL", "0")
        expected = ops.flat_pa(**kwargs)
        monkeypatch.setenv("VLLM_HPU_TP2_GQA_NATIVE_MATMUL", "1")
        actual = ops.flat_pa(**kwargs)
        assert torch.equal(actual, expected)
    assert len(native_reference) == 8
