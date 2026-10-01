# SPDX-License-Identifier: Apache-2.0
"""Check tile transport, cutoff ties and the published Reindex input order."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention
from vllm_gaudi.ops.deepseek_v41_tp4_selection import local_selection_packets, merge_selection_packets


class Scorer:
    _merge_topk = staticmethod(PagedCSA2Attention._merge_topk)

    def __init__(self, ratio):
        self.ratio = ratio
        self.tiles = 0

    def _scores(self, positions, rows, q, weights):
        del weights
        self.tiles += 1
        scores = (rows.remainder(23) - 4).float().expand(q.shape[0], -1)
        # Ensure the upper logical ID actually crosses the collective; merely
        # placing a low-scoring ID in the source would not test its encoding.
        scores = torch.where(rows == 1048575, 100.0, scores)
        count = ((positions + 1) // self.ratio).unsqueeze(-1)
        return scores.masked_fill((rows < 0) | (rows >= count), -torch.inf)


def check_partition(ratio, positions, rows, visible, collect_blocks):
    q = torch.empty(positions.numel(), 32, 128)
    reference = Scorer(ratio)
    expected = PagedCSA2Attention._stream_topk(reference, positions, rows, q, None,
                                             collect_blocks=collect_blocks, visible_rows=visible)
    owners = [Scorer(ratio) for _ in range(4)]
    local = [local_selection_packets(owner, positions, rows, q, None, rank=rank,
                                      collect_blocks=collect_blocks, visible_rows=visible)
             for rank, owner in enumerate(owners)]
    assert len({tuple(x.shape) for x in local}) == 1
    assert all(x.dtype == torch.float32 for x in local)
    assert sum(owner.tiles for owner in owners) == reference.tiles
    gathered = torch.cat(local, dim=1)
    actual = merge_selection_packets(owners[0], positions, rows, q, gathered,
                                      collect_blocks=collect_blocks, visible_rows=visible)
    for got, want in zip(actual, expected, strict=True):
        if want is None:
            assert got is None
        else:
            # Compare unsorted local IDs as well as final candidate order;
            # sorting outputs here could hide a changed tie contract.
            assert got.dtype == want.dtype
            assert torch.equal(got, want)
    return local


@pytest.mark.parametrize("tokens", [1, 6])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("position", [16384, 19366, 20474])
def test_source_ties_and_invisible_candidate_merges(tokens, ratio, position):
    positions = torch.arange(position, position + tokens, dtype=torch.int32)
    rows = torch.arange(32768 // ratio, dtype=torch.int32)
    check_partition(ratio, positions, rows, 20480 // ratio, True)


@pytest.mark.parametrize("tokens", [1, 6])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_reindex_preserves_duplicate_invalid_and_high_logical_ids(tokens, dtype):
    generator = torch.Generator().manual_seed(29)
    rows = torch.randint(0, 32768, (tokens, 16384), generator=generator, dtype=dtype)
    rows[:, ::7] = -1
    rows[:, ::11] = 1048575
    positions = torch.arange(1048576 - tokens, 1048576, dtype=torch.int32)
    packets = check_partition(1, positions, rows, None, False)
    assert any(torch.any(packet == 1048575) for packet in packets)


@pytest.mark.parametrize("visible", [1, 2048, 4096, 6144])
def test_padding_ranks_and_all_invalid_tiles(visible):
    rows = torch.arange(8192, dtype=torch.int32)
    check_partition(1, torch.tensor([0], dtype=torch.int32), rows, visible, True)


def test_packet_shape_and_topk_contract_guards():
    q = torch.empty(1, 32, 128)
    positions = torch.tensor([16384], dtype=torch.int32)
    rows = torch.arange(8192, dtype=torch.int32)
    owner = Scorer(1)
    with pytest.raises(ValueError, match="Top512"):
        local_selection_packets(owner, positions, rows, q, None, rank=0, width=256)
    with pytest.raises(ValueError, match="rank"):
        local_selection_packets(owner, positions, rows, q, None, rank=4)
    with pytest.raises(ValueError, match="prefix"):
        local_selection_packets(owner, positions, rows, q, None, rank=0, visible_rows=8193)
    with pytest.raises(ValueError, match="shape or dtype"):
        merge_selection_packets(owner, positions, rows, q, torch.zeros(1, 10))
