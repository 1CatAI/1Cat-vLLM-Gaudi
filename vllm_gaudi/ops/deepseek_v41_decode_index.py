# SPDX-License-Identifier: Apache-2.0
"""Use the exact paged key/MME/reducer chain for bounded decode index tiles."""

import torch


def supports_native_index_tile(query, rows):
    # Shared source rows support multiple queries. Per-query rows use this
    # graph only for one query; the existing batched scorer remains available.
    return (query.device.type == "hpu" and query.dtype == torch.bfloat16 and query.ndim == 3
            and query.shape[1:] == (32, 128) and 1 <= query.shape[0] <= 128 and 1 <= rows.shape[-1] <= 2048
            and (rows.ndim == 1 or (rows.ndim == 2 and query.shape[0] == rows.shape[0] == 1)))


def native_index_tile(query, weights, packed, pages, positions, rows, ratio, local_heads):
    """Preserve full K128, per-head BF16 products and TP partial-sum rounding.

    This function returns scores in the caller's original source order.
    Streamed TopK, candidate publication and final selected-ID ordering stay
    with the caller; no capacity or useful source rows are removed.
    """
    if not supports_native_index_tile(query, rows):
        raise ValueError("Native decode index requires a bounded shared source tile or one per-query row")
    columns = rows.shape[-1]
    ids = rows.reshape(-1).to(torch.int32).contiguous()
    if columns % 128:
        ids = torch.nn.functional.pad(ids, (0, (-columns) % 128), value=-1)
    scores = torch.ops.custom_op.custom_deepseek_v41_prefill_paged_index_scores_gaudi2(
        query.contiguous(), weights.contiguous(), packed, pages,
        positions.to(torch.int32).contiguous(), ids, ratio, local_heads)
    return scores[:, :columns]
