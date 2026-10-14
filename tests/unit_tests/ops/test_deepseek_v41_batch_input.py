# SPDX-License-Identifier: Apache-2.0
"""Page selection follows request ownership across reusable decode frames."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_batch_input import RequestInputFrame


def fixture(capacity):
    bank = SimpleNamespace(pages=torch.arange(8 * 19, dtype=torch.int32).reshape(8, 19), page_versions={})
    for slot in range(8):
        bank.page_versions[slot] = (1, tuple(bank.pages[slot].tolist()))
    frame = RequestInputFrame(bank, torch.zeros(3, capacity, dtype=torch.int32),
                              torch.zeros(3, capacity, dtype=torch.int32), torch.zeros(capacity, 19, dtype=torch.int32))
    return bank, frame


def prepare(frame, slots, step=0):
    owners = [SimpleNamespace(index=slot, generation=frame.bank.page_versions[slot][0]) for slot in slots]
    requests = [SimpleNamespace(num_computed_tokens=2048 + step + i) for i in range(len(slots))]
    ids = [17 + step + slot for slot in slots]
    inputs = frame.prepare(requests, owners, input_ids=ids)
    expected = torch.full_like(frame.metadata, -1)
    expected[0].zero_()
    if slots:
        expected[:, :len(slots)] = torch.tensor([ids, [r.num_computed_tokens for r in requests], slots])
    assert torch.equal(frame.metadata, expected)
    assert torch.equal(frame.pages, frame.bank.pages.index_select(0, expected[2].clamp_min(0).long()))
    return inputs


@pytest.mark.parametrize("capacity,slots", [(1, [3]), (2, [1, 5]), (4, [2, 4, 6]), (8, [7, 1, 0, 2, 5])])
def test_reused_frame_advances_values_without_reselecting_pages(monkeypatch, capacity, slots):
    bank, frame = fixture(capacity)
    first = prepare(frame, slots)
    pointers = [value.data_ptr() for value in first]

    def unexpected(*args, **kwargs):
        pytest.fail("Unchanged page ownership repeated device page selection")

    monkeypatch.setattr(torch, "index_select", unexpected)
    for step in range(1, 5):
        current = prepare(frame, slots, step)
        assert [value.data_ptr() for value in current] == pointers


def test_reorder_extend_and_reuse_slot_keep_exact_selected_pages():
    bank, frame = fixture(4)
    prepare(frame, [1, 2, 3])
    prepare(frame, [3, 1, 2], 1)
    # Scheduler adds a page to an existing request, then recycles the slot.
    bank.pages[1, 7] = 912
    bank.page_versions[1] = (1, tuple(bank.pages[1].tolist()))
    prepare(frame, [3, 1, 2], 2)
    bank.pages[1].zero_()
    bank.pages[1, :2] = torch.tensor([511, 512])
    bank.page_versions[1] = (2, (511, 512))
    prepare(frame, [1, 3], 3)
    prepare(frame, [3], 4)


def test_padding_tracks_row_zero_changes_from_another_lane():
    bank, frame = fixture(4)
    prepare(frame, [3, 4])
    other = RequestInputFrame(bank, torch.zeros_like(frame.host), torch.zeros_like(frame.metadata),
                              torch.zeros_like(frame.pages))
    bank.pages[0, 9] = 999
    bank.page_versions[0] = (2, tuple(bank.pages[0].tolist()))
    prepare(other, [0, 1, 2, 5])
    prepare(frame, [3, 4], 1)
    assert frame.pages[2, 9] == 999


def test_old_owner_cannot_publish_metadata_for_reused_slot():
    bank, frame = fixture(2)
    prepare(frame, [1, 2])
    metadata, pages = frame.metadata.clone(), frame.pages.clone()
    bank.page_versions[1] = (2, bank.page_versions[1][1])
    with pytest.raises(RuntimeError, match="current owner"):
        frame.prepare([SimpleNamespace(num_computed_tokens=9)], [SimpleNamespace(index=1, generation=1)],
                      input_ids=[11])
    assert torch.equal(frame.metadata, metadata)
    assert torch.equal(frame.pages, pages)
