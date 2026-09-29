# SPDX-License-Identifier: Apache-2.0
"""Immutable device positions for the single-in-flight V4.1 runner."""

import torch


class PinnedDecodeInputs:
    """Reuse bounded host token/position frames until their DMA has completed."""

    def __init__(self, capacity, device):
        if not 1 <= capacity <= 128:
            raise ValueError("Decode input staging requires a bounded token bucket")
        self.capacity, self.device = capacity, torch.device(device)
        self.ids = torch.empty((2, capacity), dtype=torch.int64, device="cpu")
        self.positions = torch.empty((2, capacity), dtype=torch.int32, device="cpu")
        if self.device.type == "hpu":
            self.ids = self.ids.pin_memory("hpu")
            self.positions = self.positions.pin_memory("hpu")
        self.ids_array, self.positions_array = self.ids.numpy(), self.positions.numpy()
        self.views = {(slot, count): (self.ids[slot, :count], self.positions[slot, :count])
                      for slot in range(2) for count in range(1, capacity + 1)}
        self.events = [torch.hpu.Event() for _ in range(2)] if self.device.type == "hpu" else None
        self.pending = [False, False]
        self.calls = self.waits = self.bytes = 0

    def stage(self, tokens, start, ids, positions):
        count = len(tokens)
        if not 1 <= count <= self.capacity or not 0 <= start <= 2**31 - count:
            raise ValueError("Decode input staging range is outside its prepared capacity")
        if (ids.shape != (count,) or positions.shape != (count,) or ids.dtype != torch.int64
                or positions.dtype != torch.int32 or ids.device != self.device or positions.device != self.device
                or not ids.is_contiguous() or not positions.is_contiguous()):
            raise ValueError("Decode input destinations differ from their prepared layout")
        slot = self.calls % 2
        if self.events is not None and self.pending[slot] and not self.events[slot].query():
            self.events[slot].synchronize()
            self.waits += 1
        self.ids_array[slot, :count] = tokens
        self.positions_array[slot, :count] = range(start, start + count)
        source_ids, source_positions = self.views[slot, count]
        ids.copy_(source_ids, non_blocking=True)
        positions.copy_(source_positions, non_blocking=True)
        if self.events is not None:
            self.events[slot].record()
            self.pending[slot] = True
        self.calls += 1
        self.bytes += count * 12


def _copy_position_values(source, destination):
    destination.copy_(source)
    return destination


class PositionBank:

    def __init__(self, length, capacity, device):
        if not 1 <= capacity <= length <= 1_048_576:
            raise ValueError("V4.1 position bank requires capacity <= context <= 1048576")
        self.length, self.capacity = length, capacity
        self._native_copies = {}
        self._compiled_copy = None
        self._compiled_counts = frozenset()
        self.copy_preparations = 0
        self._values = torch.arange(length, dtype=torch.int32, device="cpu").to(device)
        # The bounded C1 profile reuses prepared Tensor objects to avoid even
        # descriptor construction in its steady-state loop.  Materializing
        # that dictionary for a 1M context would create roughly six million
        # Python/Tensor objects.  Long-context mode instead keeps the same
        # immutable 4 MiB device table and creates only the requested narrow
        # view; no position values cross PCIe per token.
        self._views = ({
            (start, count): self._values[start:start + count]
            for count in range(1, capacity + 1)
            for start in range(length - count + 1)
        } if length <= 512 else None)

    def _view(self, start, count):
        if not 0 <= start <= self.length - count or not 1 <= count <= self.capacity:
            return None
        if self._views is not None:
            return self._views.get((start, count))
        return self._values.narrow(0, start, count)

    def copy_into(self, destination, start):
        if destination.ndim != 1 or destination.dtype != torch.int32 or not destination.is_contiguous():
            raise ValueError("V4.1 positions require a contiguous int32 vector")
        prepared = self._native_copies.get(id(destination))
        if prepared is not None:
            bound, frame = prepared
            if destination is not bound:
                raise RuntimeError("Prepared position destination identity changed")
            frame.copy_positions(start)
            return
        source = self._view(start, destination.numel())
        if source is None or source.device != destination.device:
            raise ValueError("V4.1 position range/device differs from its prepared bank")
        # The runner admits one in-flight batch and completes its previous
        # token before reusing destination. The source bank is never modified.
        if self._compiled_copy is None or destination.numel() not in self._compiled_counts:
            destination.copy_(source)
        else:
            self._compiled_copy(source, destination)

    @torch.inference_mode()
    def prepare_compiled_copies(self, destinations):
        """Prepare the ordinary copy recipe once for each bounded input shape."""
        if self._compiled_copy is not None or self._native_copies:
            raise RuntimeError("Position copies are already prepared")
        from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend
        compile_backend = make_backend()

        def backend(graph, inputs, **kwargs):
            result = compile_backend(graph, inputs, **kwargs)
            self.copy_preparations += 1
            return result

        compiled = torch.compile(_copy_position_values, backend=backend, fullgraph=True, dynamic=False)
        for count, destination in destinations.items():
            if (destination.shape != (count,) or not 1 <= count <= self.capacity
                    or destination.dtype != torch.int32 or not destination.is_contiguous()
                    or destination.device != self._values.device):
                raise ValueError("Prepared position destination differs from its bank")
            compiled(self._view(0, count), destination)
        self._compiled_copy = compiled
        self._compiled_counts = frozenset(destinations)

    def prepare_native_copies(self, destinations, frames):
        if self._native_copies:
            raise RuntimeError("Position destinations are already prepared")
        for count, frame in frames.items():
            destination = destinations[count]
            if (destination.shape != (count,) or count > self.capacity
                    or destination.device != self._values.device):
                raise ValueError("Prepared position destinations differ from the position bank")
            frame.bind_position_bank(self._values)
            self._native_copies[id(destination)] = destination, frame

    def view(self, start, count):
        value = self._view(start, count)
        if value is None:
            raise ValueError("V4.1 position range differs from its prepared bank")
        return value
