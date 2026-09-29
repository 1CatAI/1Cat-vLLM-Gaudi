# SPDX-License-Identifier: Apache-2.0
"""Bounded multi-request packed-KV consumption for ordinary C1 decode.

Each query row is an independent request, with its own absolute position,
SWA ring slot and page table. No historical-length-dependent graph shapes.
"""

import torch


def write_state_rows(cache, values, rows):
    return torch.ops.custom_op.custom_deepseek_v41_state_rows_write_gaudi2(cache, values.contiguous(),
                                                                           rows.contiguous())


def read_state_rows(cache, rows, completion):
    return torch.ops.custom_op.custom_deepseek_v41_state_rows_read_gaudi2(cache, rows.contiguous(), completion)


def batch_physical_rows(rows, pages, ratio):
    width = 128 // ratio
    logical = rows.clamp_min(0)
    page = pages.gather(1, (logical // width).long().clamp_max(pages.shape[1] - 1))
    valid = (rows >= 0) & (logical < pages.shape[1] * width) & (page > 0)
    return torch.where(valid, page * width + logical.remainder(width), -1).int()


def batch_packed_mla(query,
                     swa,
                     main,
                     selected,
                     pages,
                     positions,
                     slots,
                     sink,
                     scale,
                     swa_done,
                     main_done,
                     *,
                     ratio,
                     window=128,
                     tile=8,
                     fused_gather=False,
                     sram_gather=False,
                     vector_gather=False):
    """Packed KV -> shared BF16 rows -> QK/PV MME for independent requests.

    Completion values participate in row eligibility, connecting mutations to
    the first device consumer. -1 is a completed no-write (e.g. an unfinished
    compression pair), while padding is independently masked by ownership.
    The internal query tile bounds SRAM; it is not the scheduler batch size.
    """
    if (ratio not in (0, 1, 2) or tile not in (4, 8, 16) or window != 128 or query.ndim != 3
            or not 1 <= query.shape[0] <= 64 or query.shape[-1] != 512):
        raise ValueError("Invalid batched C1 MLA contract")
    batch = query.shape[0]
    if sram_gather and (not fused_gather or tile > 8):
        raise ValueError("SRAM MLA requires fused decoding and a tile of at most eight requests")
    if vector_gather and not sram_gather:
        raise ValueError("Vector MLA decoding requires the packed SRAM consumer")
    if (positions.shape != (batch, ) or slots.shape != (batch, ) or pages.ndim != 2 or pages.shape[0] != batch
            or swa_done.shape != (batch, ) or main_done.shape != (batch, ) or selected.shape != (batch, 512)
            or swa.shape[1] != 528 or swa.shape[0] % 256):
        raise ValueError("Batch MLA request metadata does not match query rows")
    active = (positions >= 0) & (slots >= 0) & (slots < swa.shape[0] // 256)
    ready = (swa_done >= -1) & (main_done >= -1) & active
    window_rows = positions[:, None] - window + 1 + torch.arange(window, device=query.device, dtype=torch.int32)
    swa_rows = slots[:, None] * 256 + window_rows.remainder(256)
    swa_rows = torch.where((window_rows >= 0) & ready[:, None], swa_rows, -1)
    width = window
    if ratio:
        physical = batch_physical_rows(selected, pages, ratio)
        visible = selected < (positions[:, None] + 1) // ratio
        physical = torch.where((physical >= 0) & visible & ready[:, None], physical + swa.shape[0], -1)
        rows = torch.cat((swa_rows, physical), -1).int()
        width += 512
    else:
        rows = swa_rows.int()
    outputs = []
    op = (torch.ops.custom_op.custom_deepseek_v41_batch_packed_vector_mla_mme_gaudi2
          if vector_gather else torch.ops.custom_op.custom_deepseek_v41_batch_packed_sram_mla_mme_gaudi2
          if sram_gather else torch.ops.custom_op.custom_deepseek_v41_batch_packed_mla_mme_gaudi2
          if fused_gather else torch.ops.custom_op.custom_deepseek_v41_batch_paged_mla_mme_gaudi2)
    for start in range(0, batch, tile):
        stop = min(batch, start + tile)
        current = rows[start:stop].contiguous()
        local = torch.arange((stop - start) * width, device=query.device,
                             dtype=torch.int32).reshape(stop - start, width)
        indices = torch.where(current >= 0, local, -1).contiguous()
        lengths = torch.where(active[start:stop], width, 0).int().contiguous()
        output = op(query[start:stop].contiguous(), swa, main, current.reshape(1, -1), indices, sink, scale, lengths)
        outputs.append(output)
    return torch.cat(outputs, 0)
