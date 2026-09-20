# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm_gaudi.ops.deepseek_v41_math import pack_swa, unpack_swa
from vllm_gaudi.ops.deepseek_v41_paged_attention import (
    PagedCSA2Attention,
    PREFILL_INDEX_ROWS,
    SWA_ROWS,
)


def test_flat_prefill_swa_workspace_preserves_prior_tail_and_causal_indices():
    start, tokens, width = 256, 4, 512
    prior = torch.arange(SWA_ROWS * width, dtype=torch.float32).reshape(SWA_ROWS, width).remainder(31).bfloat16()
    swa = pack_swa(prior)
    current = torch.arange(tokens * width, dtype=torch.float32).reshape(tokens, width).remainder(17).bfloat16()
    attention = SimpleNamespace(
        window=128,
        window_offsets=torch.arange(128, dtype=torch.int32),
        swa=swa.clone(),
    )
    positions = torch.arange(start, start + tokens, dtype=torch.int32)

    cache, indices = PagedCSA2Attention._prefill_swa_workspace(attention, current, positions)

    expected_prefix = unpack_swa(swa.index_select(0, torch.arange(start - 127, start).remainder(SWA_ROWS)))
    assert torch.equal(cache[:127], expected_prefix)
    assert torch.equal(cache[127:], current)
    assert torch.equal(indices[0], torch.arange(128, dtype=torch.int32))
    assert torch.equal(indices[-1], torch.arange(tokens - 1, tokens - 1 + 128, dtype=torch.int32))
    expected_tail = pack_swa(current)
    assert torch.equal(attention.swa.index_select(0, positions.long().remainder(SWA_ROWS)), expected_tail)


def test_first_prefill_swa_workspace_masks_positions_before_zero():
    attention = SimpleNamespace(
        window=128,
        window_offsets=torch.arange(128, dtype=torch.int32),
        swa=torch.zeros(SWA_ROWS, 528, dtype=torch.uint8),
    )
    current = torch.ones(3, 512, dtype=torch.bfloat16)
    positions = torch.arange(3, dtype=torch.int32)
    _, indices = PagedCSA2Attention._prefill_swa_workspace(attention, current, positions)
    assert (indices[0, :-1] == -1).all() and indices[0, -1] == 127
    assert (indices[1, :-2] == -1).all() and torch.equal(indices[1, -2:], torch.tensor([127, 128]))


def test_prefill_publishes_the_same_rows_to_the_decoded_c1_shadow():
    tokens, width, offset = 7, 512, 512
    current = torch.arange(tokens * width, dtype=torch.float32).reshape(tokens, width).remainder(13).bfloat16()
    attention = SimpleNamespace(
        window=128,
        window_offsets=torch.arange(128, dtype=torch.int32),
        swa=torch.zeros(SWA_ROWS, 528, dtype=torch.uint8),
        shared=SimpleNamespace(decoded_swa=torch.zeros(offset + SWA_ROWS, width, dtype=torch.bfloat16)),
        decoded_swa_offset=offset,
    )
    positions = torch.arange(20, 20 + tokens, dtype=torch.int32)

    PagedCSA2Attention._prefill_swa_workspace(attention, current, positions, decoded=True)

    expected = unpack_swa(pack_swa(current))
    assert torch.equal(attention.shared.decoded_swa[offset + positions[0]:offset + positions[-1] + 1], expected)


def test_streaming_topk_matches_one_shot_for_independent_source_rows():
    tokens, columns, width = 3, PREFILL_INDEX_ROWS + 173, 19
    positions = torch.arange(tokens, dtype=torch.int32)
    rows = torch.arange(columns, dtype=torch.int32)

    class Scorer:

        _merge_topk = staticmethod(PagedCSA2Attention._merge_topk)

        @staticmethod
        def _scores(current_positions, current_rows, q, weights):
            del q, weights
            # Unique scores avoid making this layout test depend on topk's tie ordering.
            return (current_rows.float().unsqueeze(0) * 0.01
                    + current_positions.float().unsqueeze(1) * 0.000001)

    actual, _, _ = PagedCSA2Attention._stream_topk(
        Scorer(), positions, rows, None, None, width=width)
    scores = Scorer._scores(positions, rows, None, None)
    expected = rows.expand_as(scores).gather(1, scores.topk(width, -1, sorted=False).indices)
    assert torch.equal(actual.sort(-1).values, expected.sort(-1).values)
