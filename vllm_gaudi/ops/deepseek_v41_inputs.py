# SPDX-License-Identifier: Apache-2.0
"""Immutable device positions for the single-in-flight V4.1 runner."""

import torch


class PositionBank:
    def __init__(self, length, capacity, device):
        if not 1 <= capacity <= length <= 1_048_576:
            raise ValueError("V4.1 position bank requires capacity <= context <= 1048576")
        self.length, self.capacity = length, capacity
        self._values = torch.arange(length, dtype=torch.int32, device="cpu").to(device)
        # The bounded C1 profile reuses prepared Tensor objects to avoid even
        # descriptor construction in its steady-state loop.  Materializing
        # that dictionary for a 1M context would create roughly six million
        # Python/Tensor objects.  Long-context mode instead keeps the same
        # immutable 4 MiB device table and creates only the requested narrow
        # view; no position values cross PCIe per token.
        self._views = ({(start, count): self._values[start:start + count]
                        for count in range(1, capacity + 1)
                        for start in range(length - count + 1)}
                       if length <= 512 else None)

    def _view(self, start, count):
        if not 0 <= start <= self.length - count or not 1 <= count <= self.capacity:
            return None
        if self._views is not None:
            return self._views.get((start, count))
        return self._values.narrow(0, start, count)

    def copy_into(self, destination, start):
        if destination.ndim != 1 or destination.dtype != torch.int32 or not destination.is_contiguous():
            raise ValueError("V4.1 positions require a contiguous int32 vector")
        source = self._view(start, destination.numel())
        if source is None or source.device != destination.device:
            raise ValueError("V4.1 position range/device differs from its prepared bank")
        # The runner admits one in-flight batch and completes its previous
        # token before reusing destination. The source bank is never modified.
        destination.copy_(source)

    def view(self, start, count):
        value = self._view(start, count)
        if value is None:
            raise ValueError("V4.1 position range differs from its prepared bank")
        return value
