# SPDX-License-Identifier: Apache-2.0
"""Immutable device positions for the bounded single-in-flight V4.1 runner."""

import torch


class PositionBank:
    def __init__(self, length, capacity, device):
        if not 1 <= capacity <= length <= 512:
            raise ValueError("V4.1 position bank requires capacity <= context <= 512")
        self.length, self.capacity = length, capacity
        self._values = torch.arange(length, dtype=torch.int32, device="cpu").to(device)
        self._views = {(start, count): self._values[start:start + count]
                       for count in range(1, capacity + 1) for start in range(length - count + 1)}

    def copy_into(self, destination, start):
        if destination.ndim != 1 or destination.dtype != torch.int32 or not destination.is_contiguous():
            raise ValueError("V4.1 positions require a contiguous int32 vector")
        source = self._views.get((start, destination.numel()))
        if source is None or source.device != destination.device:
            raise ValueError("V4.1 position range/device differs from its prepared bank")
        # The runner admits one in-flight batch and completes its previous
        # token before reusing destination. The source bank is never modified.
        destination.copy_(source)

    def view(self, start, count):
        value = self._views.get((start, count))
        if value is None:
            raise ValueError("V4.1 position range differs from its prepared bank")
        return value
