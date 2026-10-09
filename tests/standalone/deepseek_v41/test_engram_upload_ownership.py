# SPDX-License-Identifier: Apache-2.0
"""Pinned upload ownership without a device lease; real consumers are tested on HPU."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_host import _TransferSlot


def test_compute_upload_retires_copy_with_consumer_without_stream_join(monkeypatch):
    calls = []
    slot = _TransferSlot.__new__(_TransferSlot)
    slot.host = torch.arange(6 * 3 * 264).reshape(6, 3, 264).to(torch.uint8)
    slot.device = torch.full_like(slot.host, 171)
    slot.inflight = False
    slot.compute_upload = False
    slot.consumer_done = SimpleNamespace(synchronize=lambda: calls.append("consumer"))
    slot.dma_done = SimpleNamespace(synchronize=lambda: calls.append("dma"))

    def unexpected_stream(*args):
        raise AssertionError("Compute FIFO uploads must not enter a different stream")

    monkeypatch.setattr(torch.hpu, "stream", unexpected_stream)
    for count in (6, 2, 1, 5, 6):
        before = slot.device[count:].clone()
        slot.host.add_(7)
        slot.upload(count)
        assert torch.equal(slot.device[:count], slot.host[:count])
        assert torch.equal(slot.device[count:], before)
        slot.inflight = True
        with pytest.raises(RuntimeError, match="occupied"):
            slot.upload(count)
        slot.reuse()
    assert calls == ["consumer"] * 5


def test_transfer_stream_still_owns_dma_event_and_consumer(monkeypatch):
    calls = []
    slot = _TransferSlot.__new__(_TransferSlot)
    slot.host = torch.ones(8, 3, 264, dtype=torch.uint8)
    slot.device = torch.zeros_like(slot.host)
    slot.inflight = False
    slot.compute_upload = True
    slot.dma_done = SimpleNamespace(record=lambda stream: calls.append("record_dma"),
                                    synchronize=lambda: calls.append("retire_dma"))
    slot.consumer_done = SimpleNamespace(synchronize=lambda: calls.append("retire_consumer"))

    @contextmanager
    def stream_context(stream):
        calls.append("enter")
        yield
        calls.append("leave")

    monkeypatch.setattr(torch.hpu, "stream", stream_context)
    monkeypatch.setattr(torch.hpu, "current_stream",
                        lambda: SimpleNamespace(wait_event=lambda event: calls.append("wait_dma")))
    slot.upload(8, object())
    slot.inflight = True
    slot.reuse()
    assert torch.equal(slot.host, slot.device)
    assert calls == ["enter", "record_dma", "leave", "wait_dma", "retire_consumer", "retire_dma"]
