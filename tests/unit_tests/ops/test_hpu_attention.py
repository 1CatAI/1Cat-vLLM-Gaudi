# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backend import AttentionType

import vllm_gaudi.attention.backends.hpu_attn as hpu_attn


def test_merged_prefill_keeps_batched_decode_as_single_token_sequences(monkeypatch):
    impl = object.__new__(hpu_attn.HPUAttentionImpl)
    impl.attn_type = AttentionType.DECODER
    impl.use_merged_prefill = True
    impl.num_heads = 1
    impl.num_kv_heads = 1
    impl.head_size = 4
    impl.kv_sharing_target_layer_name = None
    impl.k_cache = None
    impl.v_cache = None
    impl.sliding_window = None
    impl.is_chunked_attention = False
    impl.alibi_slopes = None
    impl.common_attention_args = lambda *args, **kwargs: {}

    def unexpected_prompt_attention(**kwargs):
        pytest.fail("batched decode was incorrectly routed through prompt attention")

    monkeypatch.setattr(hpu_attn.ops, "prompt_attention", unexpected_prompt_attention)

    batch_size = 8
    hidden_size = impl.num_heads * impl.head_size
    query = torch.zeros((batch_size, hidden_size), dtype=torch.bfloat16)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    metadata = SimpleNamespace(
        is_prompt=False,
        seq_lens_tensor=torch.ones(batch_size, dtype=torch.int32),
        slot_mapping=None,
        block_list=None,
        block_groups=None,
        block_mapping=None,
        attn_bias=None,
        block_size=128,
    )

    output = impl.forward(
        layer=SimpleNamespace(),
        query=query,
        key=key,
        value=value,
        kv_cache=None,
        attn_metadata=metadata,
    )

    assert output.shape == query.shape
    assert torch.equal(output, torch.zeros_like(query))
