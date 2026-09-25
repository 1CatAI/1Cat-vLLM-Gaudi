# SPDX-License-Identifier: Apache-2.0
"""Auxiliary state at a scheduler-owned, immutable prefix boundary.

Compressed KV remains in the scheduler page pool. These snapshots contain
only one request's SWA and Compressor state; the worker also checkpoints
committed Engram history. A snapshot alone does not authorize a cache hit:
the scheduler must retain the pages and obtain readiness from every rank.
"""

from dataclasses import dataclass

import torch


def _identity(bank):
    program = bank.program
    return program.pp_rank, program.tp_rank, program.generation, program.precision_fingerprint


def _quiescent(bank, slot, done):
    bank.slots.validate(slot)
    if bank.pending is not None or not done.query():
        raise RuntimeError("Prefix state still has an unfinished producer or consumer")
    previous = bank.slots.consumers.get(slot.index)
    if previous is not None and not previous.query():
        raise RuntimeError("Prefix destination still has an unfinished consumer")


def _pages(bank, slot, num_tokens):
    if not 0 < num_tokens <= 1048576 or num_tokens % 128:
        raise ValueError("Prefix state requires a nonempty complete 128-token page boundary")
    version = bank.page_versions.get(slot.index)
    if version is None or version[0] != slot.generation or len(version[1]) < num_tokens // 128:
        raise RuntimeError("Prefix state is not backed by current scheduler-owned pages")
    return tuple(version[1][:num_tokens // 128])


def _views(bank, slot):
    result = {}
    for layer, state in sorted(bank.layers.items()):
        for name, value in state.named_buffers(recurse=False):
            if name not in ("swa", "kv_history", "score_history"):
                raise ValueError(f"Unaccounted auxiliary prefix state: {layer}.{name}")
            rows = 256 if name == "swa" else 8
            if value.shape[0] != bank.capacity * rows:
                raise ValueError("Prefix state layout does not match request-slot capacity")
            result[layer, name] = value[slot.index * rows:(slot.index + 1) * rows]
    return result


@dataclass(frozen=True)
class BatchPrefixSnapshot:
    identity: tuple
    block_hash: bytes
    num_tokens: int
    block_ids: tuple[int, ...]
    tensors: dict
    ready: object

    @classmethod
    def capture(cls, bank, slot, num_tokens, block_hash, producer_done, *, record_done, maximum_bytes=2 << 30):
        """Copy after the complete stage producer, before the next state write.

        ``record_done`` records an event on the copy stream after all copies.
        That event must complete before this checkpoint may be advertised.
        The cache owner must retain the returned buffers until all restores
        have completed; it must not evict a leased checkpoint.
        """
        _quiescent(bank, slot, producer_done)
        if not isinstance(block_hash, bytes) or not block_hash:
            raise ValueError("Prefix state requires the scheduler's immutable block hash")
        pages = _pages(bank, slot, num_tokens)
        views = _views(bank, slot)
        size = sum(value.numel() * value.element_size() for value in views.values())
        if size > maximum_bytes:
            raise ValueError("Auxiliary prefix checkpoint exceeds its memory budget")
        with torch.inference_mode():
            saved = {key: value.clone() for key, value in views.items()}
        return cls(_identity(bank), block_hash, num_tokens, pages, saved, record_done())

    @property
    def allocated_bytes(self):
        return sum(value.numel() * value.element_size() for value in self.tensors.values())

    def restore(self, bank, slot, num_tokens, block_hash, consumer_done, *, record_done):
        """Restore only into the scheduler-validated request slot.

        Validate every binding before the first mutation. The returned event
        owns both the source snapshot and destination rows until copies finish;
        the real first model consumer must wait for it before reading state.
        Device failure after mutation is fatal to that execution, not a signal
        to retry through the uncached path.
        """
        _quiescent(bank, slot, consumer_done)
        if not self.ready.query():
            raise RuntimeError("Prefix checkpoint copies have not completed")
        if _identity(bank) != self.identity or block_hash != self.block_hash or num_tokens != self.num_tokens:
            raise ValueError("Prefix checkpoint model, hash or absolute position changed")
        if _pages(bank, slot, num_tokens) != self.block_ids:
            raise ValueError("Prefix checkpoint no longer owns the matched compressed KV pages")
        views = _views(bank, slot)
        if views.keys() != self.tensors.keys():
            raise ValueError("Prefix checkpoint does not cover the complete stage")
        for key, target in views.items():
            source = self.tensors[key]
            if target.shape != source.shape or target.dtype != source.dtype or target.device != source.device:
                raise ValueError("Prefix checkpoint tensor layout changed")
        with torch.inference_mode():
            for key, target in views.items():
                target.copy_(self.tensors[key])
        done = record_done()
        bank.slots.submitted(slot, done)
        return done


@dataclass(frozen=True)
class PrefixStateTicket:
    """Scheduler-owned cache slot and its non-reusable publication generation."""
    index: int
    generation: int


class BatchPrefixStore:
    """Bounded worker checkpoints with scheduler-controlled replacement.

    Workers never choose an independent LRU victim: every TP/PP rank must
    interpret the same ticket. Publication waits for the copy, and a restore
    keeps its source alive until the destination copy completes. Scheduler
    leases must cover the interval before a restore reaches this worker.
    """

    def __init__(self, capacity, maximum_bytes=2 << 30):
        if not 0 < capacity <= 128 or not 0 < maximum_bytes <= 2 << 30:
            raise ValueError("Prefix store requires a bounded slot and byte budget")
        self.capacity, self.maximum_bytes = capacity, maximum_bytes
        self.entries = {}
        self.generations = [0] * capacity
        self.copies = {}
        self.published = set()

    @property
    def allocated_bytes(self):
        return sum(snapshot.allocated_bytes for snapshot in self.entries.values())

    def _index(self, ticket):
        if not isinstance(ticket, PrefixStateTicket) or not 0 <= ticket.index < self.capacity or ticket.generation <= 0:
            raise ValueError("Invalid prefix cache ticket")

    def _entry(self, ticket):
        self._index(ticket)
        if self.generations[ticket.index] != ticket.generation or ticket.index not in self.entries:
            raise RuntimeError("Stale prefix cache ticket")
        return self.entries[ticket.index]

    def _idle(self, index):
        snapshot = self.entries.get(index)
        if snapshot is not None and not snapshot.ready.query():
            raise RuntimeError("Prefix capture still has an unfinished producer")
        pending = [event for event in self.copies.get(index, ()) if not event.query()]
        self.copies[index] = pending
        if pending:
            raise RuntimeError("Prefix checkpoint still has unfinished restore copies")

    def capture(self, ticket, bank, slot, num_tokens, block_hash, producer_done, *, record_done):
        self._index(ticket)
        if ticket.generation <= self.generations[ticket.index]:
            raise RuntimeError("Prefix capture must advance its publication generation")
        self._idle(ticket.index)
        # Retain the old checkpoint until validation and allocation succeed.
        # Account for both during replacement, not just the eventual live set.
        remaining = self.maximum_bytes - self.allocated_bytes
        snapshot = BatchPrefixSnapshot.capture(bank,
                                               slot,
                                               num_tokens,
                                               block_hash,
                                               producer_done,
                                               record_done=record_done,
                                               maximum_bytes=remaining)
        self.entries[ticket.index] = snapshot
        self.generations[ticket.index] = ticket.generation
        self.published.discard(ticket.index)
        self.copies.pop(ticket.index, None)

    def publish(self, ticket):
        snapshot = self._entry(ticket)
        if not snapshot.ready.query():
            raise RuntimeError("Cannot publish an unfinished prefix checkpoint")
        self.published.add(ticket.index)
        return snapshot.block_hash, snapshot.num_tokens, snapshot.block_ids

    def restore(self, ticket, bank, slot, num_tokens, block_hash, consumer_done, *, record_done):
        snapshot = self._entry(ticket)
        if ticket.index not in self.published:
            raise RuntimeError("Prefix checkpoint has not been published")
        done = snapshot.restore(bank, slot, num_tokens, block_hash, consumer_done, record_done=record_done)
        self.copies[ticket.index] = [event for event in self.copies.get(ticket.index, ()) if not event.query()] + [done]
        return done

    def discard(self, ticket):
        self._entry(ticket)
        self._idle(ticket.index)
        del self.entries[ticket.index]
        self.published.discard(ticket.index)
        self.copies.pop(ticket.index, None)

    def close(self):
        for index in self.entries:
            self._idle(index)
        self.entries.clear()
        self.published.clear()
        self.copies.clear()
