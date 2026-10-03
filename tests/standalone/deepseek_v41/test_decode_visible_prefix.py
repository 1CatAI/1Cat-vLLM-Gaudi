# SPDX-License-Identifier: Apache-2.0
"""Conservative decode pruning must preserve exact source and candidate IDs."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_config import decode_source_prefix_bound
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention


@pytest.mark.parametrize("end,search,wanted", [(1, 512, 512), (8193, 16384, 12288),
                                             (16384, 32768, 16384), (16385, 32768, 20480),
                                             (20480, 32768, 20480), (20481, 32768, 24576),
                                             (32768, 32768, 32768), (32769, 65536, 40960),
                                             (82945, 131072, 98304), (144385, 262144, 163840),
                                             (1048576, 1048576, 1048576)])
def test_bound_covers_transaction_without_reducing_context_capacity(end, search, wanted):
    assert decode_source_prefix_bound(end, search, 4) == wanted
    assert decode_source_prefix_bound(end, search, 2) == wanted
    assert decode_source_prefix_bound(end, search, 4, runtime_indexer=True) == wanted


@pytest.mark.parametrize("end", [0, -1, 32769])
def test_bound_rejects_a_transaction_outside_its_search_bucket(end):
    with pytest.raises(ValueError, match="outside"):
        decode_source_prefix_bound(end, 32768, 4)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("tokens", [1, 6])
@pytest.mark.parametrize("position", [16384, 19366, 20474])
def test_future_tile_pruning_preserves_cutoff_ties_and_published_candidate_order(ratio, tokens, position):
    class Scorer:
        _merge_topk = staticmethod(PagedCSA2Attention._merge_topk)

        def __init__(self):
            self.ratio = ratio

        def _scores(self, positions, rows, q, weights):
            del q, weights
            # Many ties at the selection cutoff, plus the causal invalid tail.
            scores = (rows.remainder(23) - 4).float().expand(tokens, -1)
            return scores.masked_fill(rows.unsqueeze(0) >= ((positions + 1) // ratio).unsqueeze(1), -torch.inf)

    positions = torch.arange(position, position + tokens, dtype=torch.int32)
    rows = torch.arange(32768 // ratio, dtype=torch.int32)
    q = torch.empty(tokens, 32, 128)
    end = position + tokens
    reference = PagedCSA2Attention._stream_topk(Scorer(), positions, rows, q, None, collect_blocks=True)
    candidate = PagedCSA2Attention._stream_topk(
        Scorer(), positions, rows, q, None, collect_blocks=True,
        visible_rows=decode_source_prefix_bound(end, 32768, 4) // ratio)
    # _select publishes source IDs after a final integer sort. Candidate
    # blocks retain their original unsorted order for later Reindex consumers.
    assert torch.equal(reference[0].sort(-1).values, candidate[0].sort(-1).values)
    assert torch.equal(reference[1], candidate[1])
    assert torch.equal(reference[2], candidate[2])


def test_reindex_and_prefill_clear_the_previous_decode_bound():
    owner = SimpleNamespace(ratio=1, layer=20, candidate_source=20, search_length=32768,
                            tensor_parallel_size=4, runtime_indexer=False, decode_visible_rows=None)
    PagedCSA2Attention.set_decode_visible_tokens(owner, 16385)
    assert owner.decode_visible_rows == 20480
    PagedCSA2Attention.set_decode_visible_tokens(owner, None)
    assert owner.decode_visible_rows is None
    owner.layer = 24
    PagedCSA2Attention.set_decode_visible_tokens(owner, 16385)
    assert owner.decode_visible_rows is None
