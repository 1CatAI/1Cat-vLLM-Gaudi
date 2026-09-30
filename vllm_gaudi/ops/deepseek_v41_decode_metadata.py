# SPDX-License-Identifier: Apache-2.0
"""Read-only decode coordinates, produced once per current layer-group call."""
import torch


def prepare_decode_metadata(positions):
    # The 256-row SWA ring is a power of two. This retains remainder semantics
    # for integer positions without a per-layer division or literal cast.
    rows = torch.bitwise_and(positions, 255).long()
    lengths = torch.full((positions.numel(),), 640, dtype=torch.int32, device=positions.device)
    return rows, lengths
