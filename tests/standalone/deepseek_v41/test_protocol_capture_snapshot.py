# SPDX-License-Identifier: Apache-2.0
"""Cold protocol capture must restore its full write set with bounded KV copies."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_round_repair import C6LookaheadJournal, SampledRoundRepairFrame


def journal(target, indices):
    value = torch.nn.Module()
    value.specs = [("logical", 1)]
    value.register_buffer("target_0", target)
    value.register_buffer("indices_0", indices)
    return value


@pytest.mark.parametrize("offset", (0, 5, 11))
@pytest.mark.parametrize("lookahead", (False, True))
def test_exact_repair_restores_duplicate_rows_and_mutable_indices(offset, lookahead):
    target = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8)
    original = target.clone()
    indices = torch.tensor([offset, offset, offset + 1, offset + 2, offset + 2, offset + 3], dtype=torch.int32)
    original_indices = indices.clone()
    frame = SampledRoundRepairFrame.__new__(SampledRoundRepairFrame)
    torch.nn.Module.__init__(frame)
    first = journal(target, indices)
    if lookahead:
        pair = C6LookaheadJournal.__new__(C6LookaheadJournal)
        torch.nn.Module.__init__(pair)
        pair.first, pair.second = first, journal(target, indices + 2)
        frame.journal = pair
        owners = (pair.first, pair.second)
    else:
        frame.journal = first
        owners = (first,)
    control = torch.tensor([7], dtype=torch.int32)
    snapshot = frame.capture_snapshot((target, control, *(item.indices_0 for item in owners)), exact_repair=True)
    assert snapshot.bytes < target.numel() * target.element_size()
    for item in owners:
        target.index_fill_(0, item.indices_0.long(), -1)
        item.indices_0.zero_()
    control.zero_()
    snapshot.restore()
    assert torch.equal(target, original)
    assert control.item() == 7
    assert torch.equal(first.indices_0, original_indices)


def test_covered_protocol_does_not_clone_read_only_target():
    target = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8)
    frame = SampledRoundRepairFrame.__new__(SampledRoundRepairFrame)
    torch.nn.Module.__init__(frame)
    frame.journal = journal(target, torch.arange(6, dtype=torch.int32))
    saved = torch.ones(6, 8)
    snapshot = frame.capture_snapshot((target, saved), exact_repair=False)
    assert snapshot.bytes == saved.numel() * saved.element_size()
    saved.zero_()
    snapshot.restore()
    assert torch.equal(saved, torch.ones_like(saved))
