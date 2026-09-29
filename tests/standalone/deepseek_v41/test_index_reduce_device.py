# SPDX-License-Identifier: Apache-2.0
"""Ordered head accumulation and SIMD candidate-lane mapping."""
from test_native_moe import HPU, torch


@HPU
def test_ordered_index_reduce_preserves_candidate_lanes_and_bf16_boundaries():
    op = torch.ops.custom_op.custom_deepseek_v41_index_reduce_gaudi2
    torch.manual_seed(623)
    for batch, columns in ((1, 128), (8, 2048), (64, 512)):
        for change in range(2):
            dots = (torch.randn(batch, 32, columns) * (1 if change else 8)).bfloat16()
            weights = (torch.randn(batch, 32) * .02).bfloat16()
            dots[..., ::17] = 0
            products = (dots.relu() * weights.unsqueeze(-1)).float()
            partial = []
            for shard in range(2):
                total = torch.zeros(batch, columns)
                for head in range(16):
                    total += products[:, shard * 16 + head]
                partial.append(total.bfloat16().float())
            expected = (partial[0] + partial[1]).bfloat16().float()
            actual = op(dots.to("hpu"), weights.to("hpu")).cpu()
            assert torch.equal(actual, expected), (batch, columns, change)
