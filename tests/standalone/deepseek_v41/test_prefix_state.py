# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
from vllm_gaudi.ops.deepseek_v41_batch_state import BatchLayerState
from vllm_gaudi.ops.deepseek_v41_prefix_state import BatchPrefixSnapshot, BatchPrefixStore, PrefixStateTicket


class Event:

    def __init__(self, ready=True):
        self.ready = ready

    def query(self):
        return self.ready


def make_bank():
    bank = SimpleNamespace(capacity=3,
                           slots=RequestSlots(3),
                           pending=None,
                           page_versions={},
                           program=SimpleNamespace(pp_rank=0, tp_rank=0, generation=5,
                                                   precision_fingerprint="quant-v1"),
                           layers={i: BatchLayerState(3, "cpu", compressor=i == 2)
                                   for i in (0, 1, 2)})
    for layer in bank.layers.values():
        for value in layer.buffers():
            values = torch.arange(value.numel()).reshape(value.shape)
            value.copy_(values.remainder(239).to(value.dtype))
    source, target, neighbour = (bank.slots.acquire(name) for name in ("source", "target", "neighbour"))
    for slot in (source, target, neighbour):
        bank.page_versions[slot.index] = slot.generation, tuple(range(1, 8193))
    return bank, source, target, neighbour


def slot_values(bank, slot):
    return [
        value[slot.index * (256 if name == "swa" else 8):(slot.index + 1) * (256 if name == "swa" else 8)].clone()
        for layer in bank.layers.values() for name, value in layer.named_buffers()
    ]


@pytest.mark.parametrize("tokens", [128, 2048, 8192, 1048448])
def test_prefix_checkpoint_survives_source_change_and_restores_only_target(tokens):
    bank, source, target, neighbour = make_bank()
    expected, untouched = slot_values(bank, source), slot_values(bank, neighbour)
    snapshot = BatchPrefixSnapshot.capture(bank, source, tokens, b"hash", Event(), record_done=Event)
    assert snapshot.allocated_bytes == sum(x.numel() * x.element_size() for x in expected)
    assert snapshot.allocated_bytes < 2 << 20
    for layer in bank.layers.values():
        layer.clear_slot(source.index)
        layer.clear_slot(target.index)
    done = snapshot.restore(bank, target, tokens, b"hash", Event(), record_done=Event)
    assert done.query()
    assert all(torch.equal(a, b) for a, b in zip(expected, slot_values(bank, target), strict=True))
    assert all(torch.equal(a, b) for a, b in zip(untouched, slot_values(bank, neighbour), strict=True))
    assert all(not x.any() for x in slot_values(bank, source))
    # The next decode writer changes its ring row, not the immutable checkpoint.
    bank.layers[0].swa[target.index * 256 + tokens % 256].fill_(255)
    assert torch.equal(snapshot.tensors[0, "swa"], expected[0])


@pytest.mark.parametrize("fault", ["hash", "tokens", "model", "rank", "generation", "pages", "layout", "stage"])
def test_invalid_checkpoint_rejected_before_any_destination_write(fault):
    bank, source, target, _ = make_bank()
    snapshot = BatchPrefixSnapshot.capture(bank, source, 2048, b"hash", Event(), record_done=Event)
    for layer in bank.layers.values():
        layer.clear_slot(target.index)
    before = slot_values(bank, target)
    tokens, digest = 2048, b"hash"
    if fault == "hash":
        digest = b"other"
    elif fault == "tokens":
        tokens = 1920
    elif fault == "model":
        bank.program.precision_fingerprint = "quant-v2"
    elif fault == "rank":
        bank.program.pp_rank = 1
    elif fault == "generation":
        bank.program.generation += 1
    elif fault == "pages":
        bank.page_versions[target.index] = target.generation, tuple(range(2, 8194))
    elif fault == "layout":
        snapshot.tensors[2, "score_history"] = snapshot.tensors[2, "score_history"].double()
    else:
        snapshot.tensors.pop((2, "score_history"))
    with pytest.raises(ValueError):
        snapshot.restore(bank, target, tokens, digest, Event(), record_done=Event)
    assert all(torch.equal(a, b) for a, b in zip(before, slot_values(bank, target), strict=True))


def test_copy_events_and_slot_generation_protect_sources_and_destinations():
    bank, source, target, _ = make_bank()
    pending = Event(False)
    with pytest.raises(RuntimeError, match="unfinished"):
        BatchPrefixSnapshot.capture(bank, source, 2048, b"hash", pending, record_done=Event)
    snapshot = BatchPrefixSnapshot.capture(bank, source, 2048, b"hash", Event(), record_done=lambda: pending)
    with pytest.raises(RuntimeError, match="copies"):
        snapshot.restore(bank, target, 2048, b"hash", Event(), record_done=Event)
    pending.ready = True
    bank.pending = (1, (target, ))
    with pytest.raises(RuntimeError, match="unfinished"):
        snapshot.restore(bank, target, 2048, b"hash", Event(), record_done=Event)
    bank.pending = None
    done = Event(False)
    snapshot.restore(bank, target, 2048, b"hash", Event(), record_done=lambda: done)
    with pytest.raises(RuntimeError, match="unfinished"):
        snapshot.restore(bank, target, 2048, b"hash", Event(), record_done=Event)
    bank.slots.release(target)
    with pytest.raises(RuntimeError, match="live consumers"):
        bank.slots.acquire("replacement")
    done.ready = True
    replacement = bank.slots.acquire("replacement")
    assert replacement.index == target.index and replacement.generation > target.generation
    with pytest.raises(RuntimeError, match="Stale"):
        snapshot.restore(bank, target, 2048, b"hash", Event(), record_done=Event)
    with pytest.raises(RuntimeError, match="current scheduler"):
        snapshot.restore(bank, replacement, 2048, b"hash", Event(), record_done=Event)


@pytest.mark.parametrize("tokens", [0, -128, 127, 2049, 1048704])
def test_capture_rejects_partial_or_unbounded_prefix(tokens):
    bank, source, _, _ = make_bank()
    with pytest.raises(ValueError, match="boundary"):
        BatchPrefixSnapshot.capture(bank, source, tokens, b"hash", Event(), record_done=Event)


def test_budget_failure_precedes_allocation_and_event_publication():
    bank, source, _, _ = make_bank()

    def unexpected_event():
        pytest.fail("Capture must not publish an over-budget checkpoint")

    with pytest.raises(ValueError, match="budget"):
        BatchPrefixSnapshot.capture(bank, source, 2048, b"hash", Event(), record_done=unexpected_event, maximum_bytes=1)


def test_store_waits_for_publication_and_all_restore_consumers_before_replacement():
    bank, source, target, neighbour = make_bank()
    store, ticket = BatchPrefixStore(2), PrefixStateTicket(0, 1)
    capture_done = Event(False)
    store.capture(ticket, bank, source, 1920, b"prefix", Event(), record_done=lambda: capture_done)
    with pytest.raises(RuntimeError, match="unfinished"):
        store.publish(ticket)
    with pytest.raises(RuntimeError, match="not been published"):
        store.restore(ticket, bank, target, 1920, b"prefix", Event(), record_done=Event)
    capture_done.ready = True
    assert store.publish(ticket) == (b"prefix", 1920, tuple(range(1, 16)))
    first, second = Event(False), Event(False)
    store.restore(ticket, bank, target, 1920, b"prefix", Event(), record_done=lambda: first)
    store.restore(ticket, bank, neighbour, 1920, b"prefix", Event(), record_done=lambda: second)
    first.ready = True
    with pytest.raises(RuntimeError, match="unfinished restore"):
        store.discard(ticket)
    with pytest.raises(RuntimeError, match="unfinished restore"):
        store.capture(PrefixStateTicket(0, 2), bank, source, 2048, b"other", Event(), record_done=Event)
    second.ready = True
    store.capture(PrefixStateTicket(0, 2), bank, source, 2048, b"other", Event(), record_done=Event)
    with pytest.raises(RuntimeError, match="Stale"):
        store.restore(ticket, bank, target, 1920, b"prefix", Event(), record_done=Event)


def test_store_budget_covers_replacement_peak_and_preserves_previous_entry_on_failure():
    bank, source, target, _ = make_bank()
    size = sum(value.numel() * value.element_size() for value in slot_values(bank, source))
    store, ticket = BatchPrefixStore(2, maximum_bytes=2 * size - 1), PrefixStateTicket(0, 1)
    store.capture(ticket, bank, source, 2048, b"first", Event(), record_done=Event)
    store.publish(ticket)
    for index in (0, 1):
        with pytest.raises(ValueError, match="budget"):
            store.capture(PrefixStateTicket(index, 2), bank, source, 2048, b"new", Event(), record_done=Event)
        assert store.allocated_bytes == size
        store.restore(ticket, bank, target, 2048, b"first", Event(), record_done=Event)
    store.discard(ticket)
    assert store.allocated_bytes == 0
    with pytest.raises(RuntimeError, match="advance"):
        store.capture(ticket, bank, source, 2048, b"new", Event(), record_done=Event)
    store.capture(PrefixStateTicket(0, 2), bank, source, 2048, b"new", Event(), record_done=Event)


def test_repeated_hits_retire_event_references_and_close_preserves_inflight_sources():
    bank, source, target, _ = make_bank()
    store, ticket = BatchPrefixStore(1), PrefixStateTicket(0, 1)
    store.capture(ticket, bank, source, 2048, b"prefix", Event(), record_done=Event)
    store.publish(ticket)
    for _ in range(8):
        store.restore(ticket, bank, target, 2048, b"prefix", Event(), record_done=Event)
    assert len(store.copies[0]) == 1
    pending = Event(False)
    store.restore(ticket, bank, target, 2048, b"prefix", Event(), record_done=lambda: pending)
    allocated = store.allocated_bytes
    with pytest.raises(RuntimeError, match="unfinished restore"):
        store.close()
    assert store.allocated_bytes == allocated
    pending.ready = True
    store.close()
    assert store.allocated_bytes == 0
    with pytest.raises(RuntimeError, match="Stale"):
        store.publish(ticket)
