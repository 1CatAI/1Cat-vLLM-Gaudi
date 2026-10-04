# SPDX-License-Identifier: Apache-2.0
"""Fixed device frames for one-step-ahead ordinary decode.

The native stage reuses its output allocations. Preserve the current draw,
logits and lookback before enqueueing another invocation, so asynchronous
certificate repair cannot read the following token's values.
"""

from dataclasses import dataclass

import torch


def copy_sampling_frame(destinations, payload, history):
    for destination, source in zip(destinations[:5], payload, strict=True):
        destination.copy_(source)
    destinations[5].copy_(history)
    return destinations


@dataclass(frozen=True)
class DeviceInputTransaction:
    """Host history mirror for a device-produced ordinary input."""
    batch: object


@dataclass
class DeviceStep:
    request_id: str
    start: int
    hidden: object
    completion: object
    inputs: object

    def drain(self):
        if self.completion is None:
            raise RuntimeError("Queued device step lacks its native completion")
        self.completion.synchronize()


class SamplingFrames:
    """Two fixed-address frames, alternated only after the prior certificate."""

    def __init__(self, payload, history, *, compile_copy=True):
        if len(payload) != 5:
            raise ValueError("Device continuation requires token and next-position outputs")
        self.frames = tuple(tuple(torch.empty_like(value) for value in (*payload, history)) for _ in range(2))
        self.copy = (torch.compile(copy_sampling_frame, backend="hpu_backend", fullgraph=True, dynamic=False)
                     if compile_copy else copy_sampling_frame)
        self.parity = 0

    def preserve(self, payload, history):
        frame = self.frames[self.parity]
        self.parity ^= 1
        self.copy(frame, payload, history)
        return frame
