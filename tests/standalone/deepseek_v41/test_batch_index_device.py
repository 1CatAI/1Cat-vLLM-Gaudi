# SPDX-License-Identifier: Apache-2.0
"""Independent request addressing for fixed-shape native index selection."""

from test_native_moe import HPU, torch

from vllm_gaudi.ops.deepseek_v41_indexer import runtime_index_select


@HPU
def test_batched_indexer_matches_c1_for_mixed_lengths_pages_and_padding():
    torch.manual_seed(957)
    capacity = 2048
    # Each request has a distinct page permutation into one shared KV pool.
    # Keep the pool small for the contract test; pages can alias read-only
    # fixtures, but the logical-to-physical maps must not cross request rows.
    packed = torch.randint(0, 256, (capacity * 2, 68), dtype=torch.uint8)
    packed[:, 64:] = torch.randint(119, 127, (capacity * 2, 4), dtype=torch.uint8)
    packed = packed.to("hpu")
    for ratio, batch in ((1, 1), (2, 2), (1, 8), (2, 64)):
        width = 128 // ratio
        pages = torch.stack([torch.randperm(capacity * 2 // width)[:capacity // width]
                             for _ in range(batch)]).int().to("hpu")
        positions = torch.tensor([(-1, 127, 511 * ratio, 512 * ratio, 900 * ratio, 1535 * ratio)[i % 6]
                                  for i in range(batch)],
                                 dtype=torch.int32).to("hpu")
        candidates = torch.full((batch, 2048), -1, dtype=torch.int32)
        for row in range(batch):
            candidates[row, :capacity // 8] = torch.randperm(capacity // 8).int()
        candidates[:, 17:23] = -1
        candidates = candidates.to("hpu")
        query = torch.randn(batch, 32, 128).bfloat16().to("hpu")
        weights = (torch.randn(batch, 32) * 0.1).bfloat16().to("hpu")
        for change in range(2):
            if change:
                positions.copy_(positions.cpu().roll(1))
                query.copy_(query.cpu().roll(1, 0))
                weights.copy_(weights.cpu().roll(1, 0))
            for reindex in (False, True):
                outputs, blocks = runtime_index_select(query,
                                                       weights,
                                                       packed,
                                                       pages,
                                                       positions,
                                                       candidates,
                                                       ratio=ratio,
                                                       capacity=capacity,
                                                       reindex=reindex,
                                                       publish_candidates=not reindex)
                expected, expected_blocks = [], []
                for row in range(batch):
                    selected, pool = runtime_index_select(query[row:row + 1].clone(),
                                                          weights[row:row + 1].clone(),
                                                          packed,
                                                          pages[row].clone(),
                                                          positions[row:row + 1].clone(),
                                                          candidates[row:row + 1].clone(),
                                                          ratio=ratio,
                                                          capacity=capacity,
                                                          reindex=reindex,
                                                          publish_candidates=not reindex)
                    expected.append(selected.cpu())
                    if pool is not None:
                        expected_blocks.append(pool.cpu())
                assert torch.equal(outputs.cpu(), torch.cat(expected))
                if blocks is not None:
                    assert torch.equal(blocks.cpu(), torch.cat(expected_blocks))
