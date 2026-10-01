# SPDX-License-Identifier: Apache-2.0
"""Page growth retains existing logical rows; remaps retire their mirrors."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_state import PagedStageState


@pytest.fixture
def state():
    value = object.__new__(PagedStageState)
    value.program = SimpleNamespace(shared=SimpleNamespace(block_table=torch.zeros(16, dtype=torch.int32)))
    value.block_table_host = torch.zeros(16, dtype=torch.int32)
    value.block_table_host_values = value.block_table_host.numpy()
    value.published_block_ids = None
    value.identity_block_table_resident = False
    value.active, value.blocks = "a", 32
    value.invalidations = 0

    def invalidate():
        value.invalidations += 1

    value._invalidate_index_mirror = invalidate
    return value


def test_nonidentity_append_only_copies_unread_tail_and_retains_mirror(state, monkeypatch):
    state._publish_block_table([5, 3, 8])
    state.invalidations = 0
    calls, copy = [], torch.Tensor.copy_

    def record(destination, source, **kwargs):
        calls.append((destination.numel(), kwargs.get("non_blocking", False)))
        return copy(destination, source, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record)
    assert state.append_single_pages("a", [5, 3, 8], [7], 32)
    assert calls == [(1, True)]
    assert state.invalidations == 0
    assert state.program.shared.block_table.tolist() == [5, 3, 8, 7] + [0] * 12
    # The subsequent ordinary binding is already up to date.
    state._publish_block_table([5, 3, 8, 7])
    assert calls == [(1, True)]


def test_preinstalled_identity_append_needs_no_copy(state, monkeypatch):
    state.program.shared.block_table.copy_(torch.arange(1, 17, dtype=torch.int32))
    state.identity_block_table_resident = True
    state.published_block_ids = (1, 2, 3)
    monkeypatch.setattr(torch.Tensor, "copy_", lambda *_a, **_k: pytest.fail("Identity mapping is resident"))
    assert state.append_single_pages("a", [1, 2, 3], [4], 32)
    assert state.invalidations == 0
    assert state.published_block_ids == (1, 2, 3, 4)


@pytest.mark.parametrize("current,added,owner,pool", [([5, 3, 9], [7], "a", 32),
                                                    ([5, 3, 8], [7], "b", 32),
                                                    ([5, 3, 8], [], "a", 32),
                                                    ([5, 3, 8], [7], "a", 31)])
def test_append_cannot_replace_owner_mapping_or_pool(state, current, added, owner, pool):
    state._publish_block_table([5, 3, 8])
    before = state.program.shared.block_table.clone()
    assert not state.append_single_pages(owner, current, added, pool)
    assert torch.equal(before, state.program.shared.block_table)
    assert state.published_block_ids == (5, 3, 8)


def test_remap_still_invalidates_and_clears_unused_entries(state):
    state._publish_block_table([5, 3, 8])
    state._publish_block_table([5, 9])
    assert state.invalidations == 2
    assert state.program.shared.block_table.tolist() == [5, 9] + [0] * 14


@pytest.mark.parametrize("added", [[32], [7] * 14])
def test_invalid_append_is_rejected_before_pinned_or_device_write(state, added):
    state._publish_block_table([5, 3, 8])
    host, device = state.block_table_host.clone(), state.program.shared.block_table.clone()
    with pytest.raises(RuntimeError, match="invalid scheduler"):
        state.append_single_pages("a", [5, 3, 8], added, 32)
    assert torch.equal(host, state.block_table_host)
    assert torch.equal(device, state.program.shared.block_table)
