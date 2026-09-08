# SPDX-License-Identifier: Apache-2.0
"""Event-ordered recurrent-state writes between compiled decoder groups."""

from dataclasses import dataclass
from functools import lru_cache, partial

import torch


@lru_cache(maxsize=1)
def _native_state_copier():
    from vllm_gaudi.distributed.tp2_fused_ar_norm import _resolve_runtime

    bridge, backend, _ = _resolve_runtime()
    copy = getattr(bridge, "copy_gdn_states_current_stream", None)
    if not callable(copy):
        raise RuntimeError("GDN async DMA requires a TP2 bridge built with copy_gdn_states_current_stream")
    return partial(copy, backend)


def copy_states_native(updates):
    """Submit one native batch, retaining tensors until its completion callback."""
    _native_state_copier()([source for _, source in updates], [destination for destination, _ in updates])


@dataclass
class _PendingWrite:
    ready: object
    done: object
    tensors: tuple[tuple[torch.Tensor, torch.Tensor], ...]


class GDNStateDMAPipeline:
    """Write on a separate stream; wait only before the destination is reused.

    Sources remain strongly referenced until the next consumer is enqueued.
    The native callback owns allocations through completion; the diagnostic
    PyTorch copy backend uses record_stream for the same lifetime requirement.
    Event waits order devices; steady decode never calls synchronize here.
    """

    def __init__(self, *, asynchronous: bool = True, api=None, copy_function=None):
        self.asynchronous = asynchronous
        self._api = api
        self._copy_function = copy_function
        self._copy_stream = None
        self._pending: dict[int, _PendingWrite] = {}
        self.submitted_groups = 0
        self.submitted_bytes = 0
        self.consumer_waits = 0
        self.flush_waits = 0
        self.host_synchronizations = 0

    @property
    def api(self):
        return torch.hpu if self._api is None else self._api

    @property
    def copy_stream(self):
        if self._copy_stream is None:
            self._copy_stream = self.api.Stream()
        return self._copy_stream

    def before_read(self, group: int) -> None:
        pending = self._pending.pop(group, None)
        if pending is not None:
            self.api.current_stream().wait_event(pending.done)
            self.consumer_waits += 1

    def _validate_submission(self, group, updates):
        if group in self._pending:
            raise RuntimeError("A GDN group must consume its previous state write before submitting another")
        if not updates:
            return False
        for destination, source in updates:
            if destination.shape != source.shape or destination.dtype != source.dtype:
                raise ValueError("GDN state DMA requires matching shapes and dtypes")
            if destination.device != source.device or not destination.is_contiguous() or not source.is_contiguous():
                raise ValueError("GDN state DMA requires contiguous tensors on the same device")
        return True

    def submit(self, group: int, updates: tuple[tuple[torch.Tensor, torch.Tensor], ...]):
        if not self._validate_submission(group, updates):
            return
        compute = self.api.current_stream()
        copy = self.copy_stream if self.asynchronous else compute
        ready = compute.record_event()
        with self.api.stream(copy):
            if self.asynchronous:
                copy.wait_event(ready)
            if self._copy_function is not None:
                # A native backend must retain sources and destinations until
                # its device completion callback, not just Python submission.
                self._copy_function(updates)
            else:
                for destination, source in updates:
                    destination.copy_(source, non_blocking=True)
                    self.api.record_stream(source, copy)
                    self.api.record_stream(destination, copy)
            done = copy.record_event()
        self._pending[group] = _PendingWrite(ready, done, updates)
        self.submitted_groups += 1
        self.submitted_bytes += sum(source.numel() * source.element_size() for _, source in updates)
        return ready, done

    def wait_all(self) -> None:
        """Order cache reset, prefill or indexed consumers after all writes."""
        compute = self.api.current_stream()
        for pending in self._pending.values():
            compute.wait_event(pending.done)
            self.flush_waits += 1
        self._pending.clear()

    def synchronize(self) -> None:
        """Drain before destroying/replacing pools, or for an explicit timer."""
        if self._pending:
            if self.asynchronous:
                self.copy_stream.synchronize()
            else:
                self.api.current_stream().synchronize()
            self.host_synchronizations += 1
            self._pending.clear()


class GDNQueuedStateDMAPipeline(GDNStateDMAPipeline):
    """Queue dependencies with DMA, avoiding Python event/stream host joins."""

    def __init__(self, *, asynchronous=True, api=None, runtime=None, precise_events=False):
        super().__init__(asynchronous=asynchronous, api=api)
        self._runtime = runtime
        self._has_work = False
        self._bound_groups = {}
        self._bound_ranges = {}
        self.precise_events = precise_events

    @property
    def runtime(self):
        if self._runtime is None:
            from vllm_gaudi.distributed.tp2_fused_ar_norm import _resolve_runtime

            bridge, backend, _ = _resolve_runtime()
            if not all(
                    callable(getattr(bridge, name, None))
                    for name in ("queue_gdn_state_copies", "queue_gdn_state_waits")):
                raise RuntimeError("GDN async DMA requires a TP2 bridge with queued state copy/wait APIs")
            self._runtime = bridge, backend
        return self._runtime

    def before_read(self, group):
        pending = self._pending.pop(group, None)
        if pending is not None:
            self.runtime[0].queue_gdn_state_waits([pending.done])
            self.consumer_waits += 1

    def submit(self, group, updates):
        if not self._validate_submission(group, updates):
            return
        destinations = tuple(destination for destination, _ in updates)
        previous = self._bound_groups.get(group, ())
        if len(previous) != len(destinations) or any(a is not b for a, b in zip(previous, destinations)):
            ranges = [(tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size())
                      for tensor in destinations]
            for owner, existing in self._bound_ranges.items():
                if owner != group and any(a < d and c < b for a, b in ranges for c, d in existing):
                    raise ValueError("Queued GDN groups must own disjoint destination slices")
            self._bound_groups[group], self._bound_ranges[group] = destinations, ranges
        bridge, backend = self.runtime
        copy = self.copy_stream if self.asynchronous else self.api.current_stream()
        submit = bridge.queue_gdn_state_copies
        if self.precise_events:
            submit = getattr(bridge, "queue_gdn_state_copies_precise", None)
            if not callable(submit):
                raise RuntimeError("Rebuild the TP2 bridge with precise GDN DMA events")
        ticket = submit(backend, [source for _, source in updates], [destination for destination, _ in updates],
                        copy.hpu_stream)
        self._pending[group] = _PendingWrite(None, ticket, updates)
        self._has_work = True
        self.submitted_groups += 1
        self.submitted_bytes += sum(source.numel() * source.element_size() for _, source in updates)
        return ticket

    def wait_all(self):
        if self._pending:
            tickets = [pending.done for pending in self._pending.values()]
            if self.precise_events:
                self.runtime[0].queue_gdn_state_waits(tickets, all_consumers=True)
            else:
                self.runtime[0].queue_gdn_state_waits(tickets)
            self.flush_waits += len(self._pending)
            self._pending.clear()

    def synchronize(self):
        if self._has_work:
            self.wait_all()
            self.api.synchronize()
            self.host_synchronizations += 1
            self._has_work = False

    def clear_bindings(self):
        self.synchronize()
        self._bound_groups.clear()
        self._bound_ranges.clear()
