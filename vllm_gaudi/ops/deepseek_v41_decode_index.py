# SPDX-License-Identifier: Apache-2.0
"""Use the exact paged key/MME/reducer chain for bounded decode index tiles."""

from functools import lru_cache

import torch


@lru_cache(maxsize=2)
def supports_query_row_reduction(local_heads):
    """Probe the additive row-plane ABI without allocating device state.

    Older libraries accept only a shared row vector. Keep their maintained
    tensor implementation until the matching native library is installed.
    """
    if local_heads not in (8, 16):
        return False
    try:
        operation = torch.ops.custom_op.custom_deepseek_v41_prefill_index_reduce_gaudi2
        dots = torch.empty(2, 32, 128, dtype=torch.bfloat16, device="meta")
        weights = torch.empty(2, 32, dtype=torch.bfloat16, device="meta")
        positions = torch.empty(2, dtype=torch.int32, device="meta")
        rows = torch.empty(2, 128, dtype=torch.int32, device="meta")
        result = operation(dots, weights, positions, rows, 1, local_heads)
        return result.shape == (2, 128) and result.dtype == torch.float32
    except (AttributeError, RuntimeError, TypeError):
        return False


def supports_per_query_index_keys(query, rows):
    """Keep C2-C6 Reindex gathers within the existing native key ABI."""
    return (query.device.type == "hpu" and query.dtype == torch.bfloat16 and query.ndim == 3
            and query.shape[1:] == (32, 128) and 2 <= query.shape[0] <= 6 and rows.ndim == 2
            and rows.shape[0] == query.shape[0] and 1 <= rows.shape[1] <= 2048
            and hasattr(torch.ops.custom_op, "custom_deepseek_v41_index_keys_gaudi2"))


def per_query_index_keys(packed, pages, rows, ratio):
    """Reuse the prefill gather/codec without materializing packed key tiles."""
    return torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2(
        packed, pages, rows.to(torch.int32).contiguous(), ratio)


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


def can_score_query_rows_together(owner, query, rows):
    """Keep the wide score candidate inside the bounded per-query reducer ABI."""
    return (getattr(owner, "c6_index_wide_scores", False)
            and getattr(owner, "c6_index_reduce", False)
            and query.device.type == "hpu" and 2 <= query.shape[0] <= 6
            and rows.ndim == 2 and rows.shape[0] == query.shape[0]
            and 2048 < rows.shape[-1] <= 16384 and rows.shape[-1] % 2048 == 0
            and owner._uses_index_mirror(query))
