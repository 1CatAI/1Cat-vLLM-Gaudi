# SPDX-License-Identifier: Apache-2.0
"""Compact-state ownership tests: padding must never overwrite a live request."""
from types import SimpleNamespace

import pytest
import torch

import vllm_gaudi.envs as gaudi_envs
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner, _zero_compact_gdn_slot


@pytest.fixture
def runner():
    result = object.__new__(HPUModelRunner)
    result._direct_gdn_state_enabled = True
    result._padded_direct_gdn_state_enabled = True
    result._compact_gdn_enabled = True
    result.use_prefix_caching = False
    result._compact_gdn_group_ids = {1, 3}
    result._compact_gdn_group_offset = {1: 0, 3: 1}
    result._gdn_max_reqs = 8
    result._gdn_slot_free_list = list(range(7, 2, -1))
    result._gdn_req_to_base_slot = {f"request-{i}": i for i in range(3)}
    result.input_batch = SimpleNamespace(
        req_ids=list(result._gdn_req_to_base_slot),
        block_table=SimpleNamespace(block_tables=[None] * 4),
    )
    return result


def state_indices(runner, active, padded):
    indices = torch.zeros(4, padded, dtype=torch.int32)
    for group, offset in runner._compact_gdn_group_offset.items():
        indices[group, :active] = torch.arange(active, dtype=torch.int32) + offset * runner._gdn_max_reqs + 1
        indices[group, active:] = -1
    return indices


@pytest.mark.parametrize("active,padded", [(1, 1), (3, 4), (5, 8), (7, 8), (8, 8)])
def test_direct_state_accepts_only_owned_prefix_and_free_padding(runner, active, padded):
    runner._gdn_slot_free_list = list(range(7, active - 1, -1))
    indices = state_indices(runner, active, padded)
    before = indices.clone()
    assert runner._can_use_direct_gdn_state(indices, active, padded)
    assert torch.equal(indices, before)


def test_padded_opt_in_does_not_disable_full_buckets(runner):
    runner._padded_direct_gdn_state_enabled = False
    assert not runner._can_use_direct_gdn_state(state_indices(runner, 3, 4), 3, 4)
    assert runner._can_use_direct_gdn_state(state_indices(runner, 4, 4), 4, 4)


def test_padded_state_environment_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_GDN_PADDED_DIRECT_STATE", raising=False)
    assert not gaudi_envs.VLLM_HPU_GDN_PADDED_DIRECT_STATE
    for setting, enabled in (("1", True), (" TRUE ", True), ("0", False), ("false", False)):
        monkeypatch.setenv("VLLM_HPU_GDN_PADDED_DIRECT_STATE", setting)
        assert gaudi_envs.VLLM_HPU_GDN_PADDED_DIRECT_STATE is enabled


@pytest.mark.parametrize("flag,value", [
    ("_direct_gdn_state_enabled", False),
    ("_compact_gdn_enabled", False),
    ("use_prefix_caching", True),
    ("_compact_gdn_group_ids", set()),
])
def test_direct_state_keeps_existing_mode_guards(runner, flag, value):
    indices = state_indices(runner, 3, 4)
    setattr(runner, flag, value)
    assert not runner._can_use_direct_gdn_state(indices, 3, 4)


@pytest.mark.parametrize("active,padded,tokens", [(0, 4, 1), (-1, 4, 1), (5, 4, 1), (3, 16, 1), (3, 4, 0), (3, 4, 2)])
def test_direct_state_rejects_invalid_geometry(runner, active, padded, tokens):
    assert not runner._can_use_direct_gdn_state(state_indices(runner, 3, 4), active, padded, tokens)


@pytest.mark.parametrize("shape", [(4, ), (3, 4), (4, 3), (4, 4, 1)])
def test_direct_state_rejects_incomplete_metadata(runner, shape):
    assert not runner._can_use_direct_gdn_state(torch.zeros(shape, dtype=torch.int32), 3, 4)


@pytest.mark.parametrize("group", [1, 3])
@pytest.mark.parametrize("replacement", [[2, 1, 3], [1, 1, 3], [1, 3, 4]])
def test_direct_state_rejects_reordered_duplicate_or_fragmented_slots(runner, group, replacement):
    indices = state_indices(runner, 3, 4)
    indices[group, :3] = torch.tensor(replacement) + runner._compact_gdn_group_offset[group] * 8
    assert not runner._can_use_direct_gdn_state(indices, 3, 4)


@pytest.mark.parametrize("group", [1, 3])
@pytest.mark.parametrize("bad_padding", [0, 4, -2])
def test_direct_state_requires_padding_sentinels_in_every_group(runner, group, bad_padding):
    indices = state_indices(runner, 3, 4)
    indices[group, 3] = bad_padding
    assert not runner._can_use_direct_gdn_state(indices, 3, 4)


def test_paused_request_in_padding_keeps_its_state(runner):
    indices = state_indices(runner, 3, 4)
    runner._gdn_req_to_base_slot["paused-request"] = 3
    runner._gdn_slot_free_list.remove(3)
    assert not runner._can_use_direct_gdn_state(indices, 3, 4)
    # A retained request outside the padded span is safe.
    runner._gdn_req_to_base_slot["paused-request"] = 7
    runner._gdn_slot_free_list.remove(7)
    runner._gdn_slot_free_list.append(3)
    assert runner._can_use_direct_gdn_state(indices, 3, 4)


@pytest.mark.parametrize("groups,capacity", [(1, 4), (3, 2), (2, 8), (4, 1)])
def test_slot_reuse_clears_group_major_rows_without_touching_other_requests(groups, capacity):
    for slot in range(capacity):
        first = torch.arange((capacity * groups + 2) * 6, dtype=torch.float32).reshape(-1, 2, 3) + 1
        second = first[..., 0].clone().to(torch.bfloat16)
        expected = [state.clone() for state in (first, second)]
        rows = [0, *(group * capacity + slot + 1 for group in range(groups)), first.shape[0] - 1]
        for state in expected:
            state[rows] = 0
        _zero_compact_gdn_slot([first, second], slot, groups)
        for actual, reference in zip((first, second), expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("slot,groups,rows", [(-1, 2, 18), (8, 2, 18), (0, 0, 18), (0, 3, 18), (0, 2, 2), (0, 2, 1)])
def test_slot_clear_rejects_invalid_layout(slot, groups, rows):
    state = torch.ones(rows, 2)
    with pytest.raises(ValueError):
        _zero_compact_gdn_slot([state], slot, groups)
    assert torch.equal(state, torch.ones_like(state))


def test_slot_clear_validates_all_pools_before_any_mutation():
    good = torch.ones(18, 2)
    invalid = torch.ones(17, 2)
    with pytest.raises(ValueError):
        _zero_compact_gdn_slot([good, invalid], 1, 2)
    assert torch.equal(good, torch.ones_like(good))


def test_reuse_after_padded_decode_preserves_live_and_paused_request_states(runner):
    pool = torch.arange(18 * 2, dtype=torch.float32).reshape(18, 2)
    indices = state_indices(runner, 3, 4)
    assert runner._can_use_direct_gdn_state(indices, 3, 4)
    # Direct decode can write free padding rows; another request owns slot 7.
    pool[[4, 12]] = float("nan")
    before = pool.clone()
    _zero_compact_gdn_slot([pool], 3, 2)
    assert torch.isfinite(pool).all()
    for slot in (0, 1, 2, 4, 5, 6, 7):
        rows = [slot + 1, 8 + slot + 1]
        torch.testing.assert_close(pool[rows], before[rows], rtol=0, atol=0)
    runner._gdn_slot_free_list.remove(3)
    assert not runner._can_use_direct_gdn_state(indices, 3, 4)
