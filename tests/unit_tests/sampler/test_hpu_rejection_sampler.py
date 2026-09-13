# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for device-side greedy speculative rejection."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_gaudi.v1.sample.hpu_rejection_sampler import (
    _dflash2_greedy_fastpath_supported,
    dflash2_greedy_rejection_sample,
    dflash2_greedy_rejection_sample_packed,
    rejection_sample_pytorch,
)


def _greedy_metadata(**overrides):
    values = {
        "all_greedy": True,
        "max_num_logprobs": None,
        "logprob_token_ids": None,
        "no_penalties": True,
        "allowed_token_ids_mask": None,
        "bad_words_token_ids": {},
        "logitsprocs": SimpleNamespace(non_argmax_invariant=[]),
        "thinking_budget_state_holder": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_greedy_rejection_handles_mixed_prefix_lengths():
    draft = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, -1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    target = torch.tensor(
        [
            [1, 2, 30, 40],
            [50, 60, 70, 80],
            [9, 10, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    bonus = torch.tensor([[100], [101], [102], [103]], dtype=torch.int32)
    cumulative = torch.tensor([4, 8, 10, 10], dtype=torch.int32)

    output = rejection_sample_pytorch(draft, target, bonus, [4, 4, 2, 0], cumulative)

    assert output.tolist() == [
        [1, 2, 30, -1, -1],
        [50, -1, -1, -1, -1],
        [9, 10, 102, -1, -1],
        [103, -1, -1, -1, -1],
    ]


def test_greedy_rejection_keeps_tensor_on_input_device():
    draft = torch.tensor([[1, 2]], dtype=torch.int64)
    target = torch.tensor([[1, 2]], dtype=torch.int64)
    output = rejection_sample_pytorch(
        draft,
        target,
        torch.tensor([[3]], dtype=torch.int64),
        [2],
        torch.tensor([2], dtype=torch.int32),
    )
    assert output.device == draft.device
    assert output.dtype == torch.int32
    assert output.tolist() == [[1, 2, 3]]


def test_dflash2_greedy_fastpath_uses_one_packed_argmax():
    logits = torch.tensor([
        [0.0, 4.0, 1.0, 2.0],
        [0.0, 1.0, 5.0, 2.0],
        [0.0, 1.0, 2.0, 6.0],
    ])
    metadata = SimpleNamespace(
        draft_token_ids=torch.tensor([[1, 2]], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1], dtype=torch.int64),
        bonus_logits_indices=torch.tensor([2], dtype=torch.int64),
        num_draft_tokens=[2],
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int32),
    )

    output = dflash2_greedy_rejection_sample(logits, metadata, _greedy_metadata())

    assert output is not None
    assert output.tolist() == [[1, 2, 3]]

    packed_output = dflash2_greedy_rejection_sample_packed(
        logits,
        metadata.draft_token_ids,
        metadata.target_logits_indices,
        metadata.bonus_logits_indices,
        metadata.cu_num_draft_tokens,
    )
    torch.testing.assert_close(packed_output, output, rtol=0, atol=0)


def test_dflash2_greedy_fastpath_declines_semantic_constraints():
    assert not _dflash2_greedy_fastpath_supported(_greedy_metadata(no_penalties=False))
    assert not _dflash2_greedy_fastpath_supported(_greedy_metadata(max_num_logprobs=0))
    assert not _dflash2_greedy_fastpath_supported(
        _greedy_metadata(allowed_token_ids_mask=torch.zeros(1, 4, dtype=torch.bool)))
