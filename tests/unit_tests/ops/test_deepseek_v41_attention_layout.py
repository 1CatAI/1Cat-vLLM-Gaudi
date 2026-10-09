# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_attention_layout import (
    group32_membership, interleave_output_weight, interleave_query_weight, pack_heads, unpack_heads,
)


@pytest.mark.parametrize('heads', [16, 32])
def test_permuted_weights_and_codec_groups(heads):
    groups = heads // 8
    query = torch.arange(heads * 512 * 3).reshape(heads * 512, 3)
    packed = interleave_query_weight(query, heads)
    assert torch.equal(packed.reshape(groups, 512, 8, 3).transpose(1, 2).reshape_as(query), query)
    output = torch.arange(groups * 3 * 4096).reshape(groups, 3, 4096)
    packed = interleave_output_weight(output)
    assert torch.equal(packed.reshape(groups, 3, 512, 8).transpose(2, 3).reshape_as(output), output)
    membership = group32_membership(heads)
    assert torch.equal(membership, membership[:, :, :1].expand_as(membership))
    assert membership.unique().numel() == heads * 16


@pytest.mark.parametrize('rows', [2, 6])
def test_head_bytes_survive_layout_roundtrip(rows):
    values = torch.arange(rows * 16 * 512).reshape(rows, 16, 512)
    assert torch.equal(unpack_heads(pack_heads(values)), values)
