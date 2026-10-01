# SPDX-License-Identifier: Apache-2.0
"""Rank/head order must survive packing across requests and slot permutations."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_index_packet import gather_index_query_packet


@pytest.mark.parametrize("batch", (1, 2, 6))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_packet_preserves_rank_head_and_slot_order(batch, dtype):
    generator = torch.Generator().manual_seed(8501)
    queries = [torch.randn(batch, 8, 128, generator=generator).to(dtype) for _ in range(4)]
    gains = [torch.randn(batch, 8, generator=generator).to(dtype) for _ in range(4)]
    order = torch.arange(batch - 1, -1, -1)
    for slot_order in (torch.arange(batch), order):
        q = [value[slot_order] for value in queries]
        g = [value[slot_order] for value in gains]
        for rank in range(4):
            calls = []

            def gather(packet, dimension, calls=calls, q=q, g=g, rank=rank):
                calls.append(dimension)
                assert torch.equal(packet, torch.cat((q[rank].flatten(1), g[rank]), -1))
                return torch.cat([torch.cat((a.flatten(1), b), -1) for a, b in zip(q, g, strict=True)], 1)

            actual_q, actual_g = gather_index_query_packet(q[rank], g[rank], gather)
            assert calls == [1]
            assert torch.equal(actual_q, torch.cat(q, 1))
            assert torch.equal(actual_g, torch.cat(g, 1))
            assert actual_q.is_contiguous() and actual_g.is_contiguous()


def test_packet_rejects_implicit_gain_promotion():
    with pytest.raises(ValueError, match="dtype"):
        gather_index_query_packet(torch.zeros(1, 8, 128, dtype=torch.bfloat16),
                                  torch.zeros(1, 8), lambda x, dim: x)
