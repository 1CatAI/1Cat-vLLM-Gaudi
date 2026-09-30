# SPDX-License-Identifier: Apache-2.0
"""Keep complete tile sort order and candidate publication across rank ranges."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention
from vllm_gaudi.ops.deepseek_v41_tp4_selection import (
    _rank_tiles, local_selection_packets, merge_selection_packets, supports_mirror_partition)


@pytest.mark.parametrize("ratio,columns,visible,blocks", [(1, 32768, 20480, True), (1, 32768, 24576, True),
                                                       (1, 32768, 8192, True), (2, 16384, 10240, False)])
@pytest.mark.parametrize("ties", [True, False])
def test_contiguous_rank_packets_retain_unsorted_ties_and_future_blocks(ratio, columns, visible, blocks, ties):
    scores = torch.zeros(1, columns) if ties else (torch.arange(columns).remainder(997).float() / 113).reshape(1, -1)
    positions = torch.tensor([min(16384, visible * ratio - 1)], dtype=torch.int32)
    scores[:, (int(positions[0]) + 1) // ratio:] = -torch.inf
    owner = SimpleNamespace(ratio=ratio, tensor_parallel_size=4,
                            _scores=lambda p, rows, q, weights: scores[:, rows.long()],
                            _merge_topk=PagedCSA2Attention._merge_topk)
    rows, q = torch.arange(columns, dtype=torch.int32), torch.ones(1, 32, 128)
    wanted = PagedCSA2Attention._stream_topk(owner, positions, rows, q, None,
                                            collect_blocks=blocks, visible_rows=visible)
    packets = []
    for rank in range(4):
        first, last = _rank_tiles(visible // 2048, rank)
        start, stop = first * 2048, last * 2048
        prefix = scores[:, start:stop] if start < stop else None
        packets.append(local_selection_packets(owner, positions, rows, q, None, rank=rank,
                                                collect_blocks=blocks, visible_rows=visible,
                                                contiguous=True, prefix_scores=prefix))
    observed = merge_selection_packets(owner, positions, rows, q, torch.cat(packets, 1),
                                        collect_blocks=blocks, visible_rows=visible, contiguous=True)
    for expected, actual in zip(wanted, observed, strict=True):
        if expected is None:
            assert actual is None
        else:
            assert torch.equal(actual, expected)


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("tp", [2, 4])
def test_partition_eligibility_preserves_wider_decode_and_tp2(batch, tp):
    owner = SimpleNamespace(tensor_parallel_size=tp, _uses_index_mirror=lambda q: True)
    assert supports_mirror_partition(owner, torch.arange(32768, dtype=torch.int32),
                                     torch.empty(batch, 32, 128), 20480, 512) == (batch == 1 and tp == 4)


@pytest.mark.parametrize("columns,visible,valid", [(65536, 20480, True), (32768, 4096, True),
                                                 (32768, 9088, True), (32768, 20480, False)])
def test_unqualified_prefixes_keep_the_original_path(columns, visible, valid):
    owner = SimpleNamespace(tensor_parallel_size=4, _uses_index_mirror=lambda q: valid)
    assert not supports_mirror_partition(owner, torch.arange(columns, dtype=torch.int32),
                                        torch.empty(1, 32, 128), visible, 512)


@pytest.mark.parametrize("tiles", range(4, 17))
def test_every_rank_keeps_real_query_dependencies_and_ranges_cover_once(tiles):
    ranges = [_rank_tiles(tiles, rank) for rank in range(4)]
    assert all(stop > start for start, stop in ranges)
    assert [tile for start, stop in ranges for tile in range(start, stop)] == list(range(tiles))
    lengths = [stop - start for start, stop in ranges]
    assert max(lengths) - min(lengths) <= 1
