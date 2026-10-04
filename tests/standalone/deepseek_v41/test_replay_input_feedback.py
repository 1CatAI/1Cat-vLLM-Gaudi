# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_sampling import commit_replay_inputs


def test_feedback_retains_private_addresses_and_advances_exactly_once():
    ids = torch.tensor([11], dtype=torch.int32)
    positions = torch.tensor([16383], dtype=torch.int32)
    addresses = ids.data_ptr(), positions.data_ptr()
    for token, position in [(71, 16384), (82, 16385)]:
        output, next_position = commit_replay_inputs(ids, positions, torch.tensor([[token]], dtype=torch.int32))
        assert (output.data_ptr(), next_position.data_ptr()) == addresses
        assert output.tolist() == [[token]]
        assert next_position.tolist() == [position]


def test_full_sampling_repair_overwrites_the_next_input_alias():
    scheduler_ids = torch.tensor([11], dtype=torch.int32)
    scheduler_positions = torch.tensor([127], dtype=torch.int32)
    ids, positions = scheduler_ids.clone(), scheduler_positions.clone()
    output, next_position = commit_replay_inputs(ids, positions, torch.tensor([[71]], dtype=torch.int32))
    output.copy_(torch.tensor([[82]], dtype=torch.int32))
    assert ids.tolist() == [82]
    assert next_position.tolist() == [128]
    assert scheduler_ids.tolist() == [11]
    assert scheduler_positions.tolist() == [127]


def test_cold_snapshot_restores_the_private_input_generation():
    from vllm_gaudi.ops.deepseek_v41_replay import _InputFeedbackSnapshot, _Snapshot

    ids = torch.tensor([11], dtype=torch.int32)
    positions = torch.tensor([127], dtype=torch.int32)
    cache = torch.tensor([37], dtype=torch.int32)
    snapshot = _InputFeedbackSnapshot(_Snapshot((cache,)), (positions, ids))
    commit_replay_inputs(ids, positions, torch.tensor([[71]], dtype=torch.int32))
    cache.fill_(41)
    snapshot.restore()
    assert ids.tolist() == [11]
    assert positions.tolist() == [127]
    assert cache.tolist() == [37]


@pytest.mark.parametrize("rows", [2, 6])
def test_feedback_rejects_non_c1_without_mutating_roots(rows):
    ids = torch.arange(rows, dtype=torch.int32)
    positions = torch.arange(rows, dtype=torch.int32) + 127
    before = ids.clone(), positions.clone()
    with pytest.raises(ValueError, match="private C1 I32"):
        commit_replay_inputs(ids, positions, torch.ones(rows, 1, dtype=torch.int32))
    assert torch.equal(ids, before[0])
    assert torch.equal(positions, before[1])
