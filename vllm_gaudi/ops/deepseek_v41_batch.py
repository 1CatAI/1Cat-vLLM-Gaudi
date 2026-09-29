# SPDX-License-Identifier: Apache-2.0
"""Request ownership and explicit phase metadata for flat V4.1 batches.

Microbatch history uses absolute request positions, following the input
contract in vLLM #56440. Slots are retired after their last device consumer;
padding has no request owner. These contracts do not treat speculative
positions or a prompt's query rows as additional requests.
"""

from dataclasses import dataclass
from enum import IntEnum
import threading

import torch


class Phase(IntEnum):
    PAD = 0
    PREFILL = 1
    DECODE = 2


@dataclass(frozen=True)
class Slot:
    request_id: str
    index: int
    generation: int


class RequestSlots:
    """Bounded host ownership; no live slot can be overwritten by reuse."""

    def __init__(self, capacity):
        if not 1 <= capacity <= 64:
            raise ValueError("V4.1 request capacity must be in 1..64")
        self.capacity = capacity
        self.generations = [0] * capacity
        self.owners = {}
        self.consumers = {}
        self.retired = {}
        self.lock = threading.Lock()

    def _check(self, slot):
        if self.owners.get(slot.request_id) != slot:
            raise RuntimeError("Stale V4.1 request slot generation")

    def acquire(self, request_id):
        with self.lock:
            if request_id in self.owners:
                return self.owners[request_id]
            # A query is nonblocking. The scheduler must wait for capacity
            # rather than replacing a still-consumed state bank.
            self.retired = {index: event for index, event in self.retired.items() if not event.query()}
            used = {slot.index for slot in self.owners.values()} | self.retired.keys()
            index = next((i for i in range(self.capacity) if i not in used), None)
            if index is None:
                raise RuntimeError("V4.1 request slots still have live consumers")
            self.generations[index] += 1
            slot = Slot(request_id, index, self.generations[index])
            self.owners[request_id] = slot
            return slot

    def submitted(self, slot, done):
        """Record completion of the last consumer, not merely input upload."""
        with self.lock:
            self._check(slot)
            self.consumers[slot.index] = done

    def release(self, slot):
        with self.lock:
            self._check(slot)
            self.owners.pop(slot.request_id)
            event = self.consumers.pop(slot.index, None)
            if event is not None and not event.query():
                self.retired[slot.index] = event

    def validate(self, slot):
        with self.lock:
            self._check(slot)


@dataclass(frozen=True)
class QuerySpan:
    slot: Slot
    start: int
    tokens: tuple[int, ...]
    phase: Phase
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class FlatBatch:
    spans: tuple[QuerySpan, ...]
    offsets: tuple[int, ...]
    token_count: int
    decode_bucket: int | None

    @classmethod
    def create(cls, spans, *, maximum, budget, capacity, page_size=128):
        spans = tuple(spans)
        if not spans or len(spans) > capacity or not 1 <= capacity <= 64 or maximum > 1048576:
            raise ValueError("Invalid V4.1 batch/request capacity")
        offsets, owners, request_ids = [0], set(), set()
        for span in spans:
            if (span.slot.index in owners or span.slot.request_id in request_ids or not 0 <= span.slot.index < capacity
                    or span.slot.generation < 1):
                raise ValueError("Each batch request must own one distinct live slot")
            count = len(span.tokens)
            if (count < 1 or span.start < 0 or span.start + count > maximum
                    or span.phase not in (Phase.PREFILL, Phase.DECODE)):
                raise ValueError("Invalid absolute query interval/phase")
            if span.phase == Phase.DECODE and count != 1:
                raise ValueError("Ordinary decode requires one input per request")
            if any(token < 0 or token >= 2**31 for token in span.tokens):
                raise ValueError("Invalid input token ID")
            pages = (span.start + count + page_size - 1) // page_size
            if len(span.block_ids) < pages or any(block <= 0 for block in span.block_ids):
                raise ValueError("Query interval is not backed by scheduler pages")
            owners.add(span.slot.index)
            request_ids.add(span.slot.request_id)
            offsets.append(offsets[-1] + count)
        if offsets[-1] > budget:
            raise ValueError("Flat query batch exceeds its scheduler token budget")
        bucket = (1 << (len(spans) - 1).bit_length() if all(span.phase == Phase.DECODE for span in spans) else None)
        return cls(spans, tuple(offsets), offsets[-1], bucket)

    def tensors(self, token_capacity=None):
        """Create host metadata for binding to a persistent transfer slot.

        Padding carries slot -1 and Phase.PAD. Device consumers must mask
        state writes and sampling with this ownership, even when its token
        ID or position happens to be a valid integer.
        """
        capacity = self.token_count if token_capacity is None else token_capacity
        if capacity < self.token_count:
            raise ValueError("Transfer slot cannot truncate real batch rows")
        result = {
            "token_ids": torch.zeros(capacity, dtype=torch.int32),
            "positions": torch.full((capacity, ), -1, dtype=torch.int32),
            "request_slots": torch.full((capacity, ), -1, dtype=torch.int32),
            "generations": torch.zeros(capacity, dtype=torch.int64),
            "phases": torch.zeros(capacity, dtype=torch.int32),
            "query_offsets": torch.tensor(self.offsets, dtype=torch.int32),
            "sequence_lengths": torch.tensor([span.start + len(span.tokens) for span in self.spans], dtype=torch.int32),
        }
        for index, span in enumerate(self.spans):
            target = slice(self.offsets[index], self.offsets[index + 1])
            result["token_ids"][target] = torch.tensor(span.tokens, dtype=torch.int32)
            result["positions"][target] = torch.arange(span.start, span.start + len(span.tokens), dtype=torch.int32)
            result["request_slots"][target] = span.slot.index
            result["generations"][target] = span.slot.generation
            result["phases"][target] = int(span.phase)
        return result
