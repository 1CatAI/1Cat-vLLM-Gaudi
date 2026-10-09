# SPDX-License-Identifier: Apache-2.0
"""Fixed device frames for one-step-ahead ordinary decode.

The native stage reuses its output allocations. Preserve the current draw,
logits and lookback before enqueueing another invocation, so asynchronous
certificate repair cannot read the following token's values.
"""

from dataclasses import dataclass

import torch


def repair_candidate_pool_live(program):
    """Whether this decode geometry has a consumer of published block IDs.

    The short-prefix branch in PagedCSA2Attention._select constructs ordered
    row IDs directly in every owner. Its candidate workspace has no reader
    and is not a persistent KV/history state. Keep the strict pool check for
    any longer Reindex geometry, and conservatively for unknown programs.
    """
    layers = getattr(program, 'layers', None)
    if layers is None:
        return True
    for block in layers:
        attention = getattr(block, 'attention', None)
        if attention is None:
            return True
        if not getattr(attention, 'owns_index', False):
            continue
        if attention.layer <= attention.candidate_source:
            continue
        ratio = getattr(attention, 'ratio', 0)
        if not ratio or attention.search_length // ratio > 512:
            return True
    return False


def bitwise_equal(left, right):
    """Compare rollback bytes, including unused floating cache rows.

    A paged snapshot includes unwritten scratch rows, whose bit patterns may
    encode NaNs. Numeric equality rejects identical NaNs and also treats the
    two signed zeros as equal; neither implements the rollback contract.
    """
    if left.dtype != right.dtype or left.shape != right.shape or left.device != right.device:
        return False
    return torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                       right.contiguous().reshape(-1).view(torch.uint8))


def copy_sampling_frame(destinations, payload, history):
    for destination, source in zip(destinations[:-1], payload, strict=True):
        destination.copy_(source)
    destinations[-1].copy_(history)
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

    def __init__(self, payload, history, *, compile_copy=True, expected_payload_size=5):
        if len(payload) != expected_payload_size:
            raise ValueError("Device continuation payload differs from its declared frame contract")
        self.frames = tuple(tuple(torch.empty_like(value) for value in (*payload, history)) for _ in range(2))
        self.copy = (torch.compile(copy_sampling_frame, backend="hpu_backend", fullgraph=True, dynamic=False)
                     if compile_copy else copy_sampling_frame)
        self.parity = 0

    def preserve(self, payload, history):
        frame = self.frames[self.parity]
        self.parity ^= 1
        self.copy(frame, payload, history)
        return frame
