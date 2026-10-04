# SPDX-License-Identifier: Apache-2.0
"""Reference contract for a shared token coordinate producer, not yet serving."""
import torch

COORDINATE_WIDTH = 192
WINDOW_OFFSET = 64


def decode_coordinates_reference(positions, input_ids, block_table):
    if (positions.dtype != torch.int32 or input_ids.dtype != torch.int32 or block_table.dtype != torch.int32
            or positions.ndim != 1 or input_ids.shape != positions.shape or block_table.ndim != 1):
        raise ValueError('Decode coordinates require matching I32 token rows and an I32 page table')
    if block_table.numel() == 0:
        raise ValueError('Decode coordinates require a nonempty page table')
    page_index = positions >> 7
    position_valid = (positions >= 0) & (page_index < block_table.numel())
    page = block_table.index_select(0, page_index.clamp(0, block_table.numel() - 1))
    active = position_valid & (page >= 0)
    compressed = positions >> 1
    fields = (positions, positions & 255, positions & 7, positions, compressed,
              page * 128 + (positions & 127), page * 64 + (compressed & 63), page,
              positions & 1, positions & -2, positions - 127,
              ((input_ids == 129264) | (input_ids == 129265)).to(torch.int32),
              (positions + 1).clamp_max(128), positions + 1, (positions + 1) >> 1,
              torch.full_like(positions, 640))
    absolute = positions[:, None] - 127 + torch.arange(128, dtype=torch.int32, device=positions.device)
    output = torch.zeros(positions.numel(), COORDINATE_WIDTH, dtype=torch.int32, device=positions.device)
    output[:, :16] = torch.stack(fields, -1)
    # Padded or unowned rows never publish usable cache coordinates. Keep the
    # original position for diagnostics, without indexing the page table at a
    # negative/out-of-range address in either the reference or native kernel.
    invalid = torch.full_like(output[:, :16], -1)
    invalid[:, 0] = positions
    invalid[:, 8] = 0
    invalid[:, 11:16] = 0
    output[:, :16] = torch.where(active[:, None], output[:, :16], invalid)
    output[:, WINDOW_OFFSET:] = torch.where(active[:, None] & (absolute >= 0), absolute & 255, -1)
    return output
