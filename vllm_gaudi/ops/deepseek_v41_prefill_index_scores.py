# SPDX-License-Identifier: Apache-2.0
"""Compile index K unpack, MME scores and weighted head reduction together."""
from functools import lru_cache
from types import FunctionType

import torch

from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4

# A transaction-local workspace, never a mirror of the full context cache.
SHARED_INDEX_MAX_ROWS = 32768
INDEX_KEY_TILE = 2048
_index_tp_audit = {
    "full_calls": 0,
    "reindex_calls": 0,
    "input_queries": 0,
    "local_queries": 0,
    "outgoing_choice_bytes": 0
}


def prefill_index_stats():
    return dict(_index_tp_audit)


def _record_query_partition(kind, tokens, local_tokens, packet):
    _index_tp_audit[kind + "_calls"] += 1
    _index_tp_audit["input_queries"] += tokens
    _index_tp_audit["local_queries"] += local_tokens
    _index_tp_audit["outgoing_choice_bytes"] += packet.numel() * packet.element_size()


def prefill_full_selection_step(scores, rows, positions, best_scores, best_rows, block_scores, block_rows, ratio,
                                publish_candidates):
    """Reuse one selection region while preserving the existing top-k order.

    Source tiles remain in increasing logical order. This does not replace
    streamed selection with a global top-k, whose cutoff ties can differ.
    """

    def merge(previous_scores, previous_rows, values, ids, width):
        values, offsets = values.topk(min(width, values.shape[-1]), dim=-1, sorted=False)
        selected = ids.expand(scores.shape[0], -1).gather(1, offsets)
        if previous_scores is not None:
            values = torch.cat((previous_scores, values), -1)
            selected = torch.cat((previous_rows, selected), -1)
            values, offsets = values.topk(min(width, values.shape[-1]), dim=-1, sorted=False)
            selected = selected.gather(1, offsets)
        return values, selected

    best_scores, best_rows = merge(best_scores, best_rows, scores, rows, 512)
    if publish_candidates:
        if scores.shape[-1] % 8:
            scores = torch.nn.functional.pad(scores, (0, 8 - scores.shape[-1] % 8), value=-torch.inf)
        grouped = scores.reshape(scores.shape[0], -1, 8).amax(-1)
        ids = rows[::8] // 8
        count = ((positions + 1) // ratio).unsqueeze(-1)
        grouped = grouped.masked_fill(ids == ((count - 1) // 8), torch.inf)
        block_scores, block_rows = merge(block_scores, block_rows, grouped, ids, 2048)
    return best_scores, best_rows, block_scores, block_rows


@lru_cache(maxsize=64)
def compiled_prefill_full_selection_step(signature):
    entry = FunctionType(prefill_full_selection_step.__code__.replace(co_name=f"prefill_full_select_{signature}"),
                         prefill_full_selection_step.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def native_prefill_index_scores(query, weights, packed, table, positions, rows, ratio):
    """Score one large-M shared-source tile without materialized head products.

    The native epilogue adapts the existing decode index reducer to explicit
    source rows and affine prompt slicing. Keep the original source-tile top-k
    order in the caller, including its candidate-block publication semantics.
    """
    keys = torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2(packed, table, rows.reshape(1, -1), ratio)[0]
    return torch.ops.custom_op.custom_deepseek_v41_prefill_index_scores_gaudi2(query, weights, keys, positions, rows,
                                                                               ratio)


@lru_cache(maxsize=64)
def compiled_native_prefill_index_scores(signature):
    entry = FunctionType(native_prefill_index_scores.__code__.replace(co_name=f"prefill_native_index_{signature}"),
                         native_prefill_index_scores.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def full_prefill_sram_scores(query, weights, packed, table, positions, rows, ratio):
    """Bind explicit runtime data to a bounded source-tile scoring region.

    Tail rows carry invalid IDs, so native key loads and score masking do not
    read past the source. Keep the original tile's length for streamed top-k.
    """
    if (query.ndim != 3 or query.shape[1:] != (32, 128) or not 1 <= query.shape[0] <= 8192
            or query.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16 or rows.ndim != 1
            or not 1 <= rows.numel() <= INDEX_KEY_TILE or ratio not in (1, 2)):
        raise ValueError("Full prefill SRAM scoring requires TP2 BF16 queries and a bounded source tile")
    columns = rows.numel()
    rows = rows.to(torch.int32)
    rows = (torch.nn.functional.pad(rows, (0, (-columns) % 128), value=-1) if columns % 128 else rows.clone())
    query, weights = query.contiguous(), weights.contiguous()
    positions = positions.to(torch.int32).contiguous()
    signature = (tuple(query.shape), tuple(rows.shape), tuple(packed.shape), tuple(table.shape), ratio)
    output = compiled_native_prefill_index_scores(signature)(query, weights, packed, table, positions, rows, ratio)
    return output[:, :columns]


def full_prefill_index_selection(query,
                                 weights,
                                 packed,
                                 table,
                                 positions,
                                 ratio,
                                 source_rows,
                                 publish_candidates,
                                 visible_rows=None):
    """Select independent query rows with the existing streamed tie order."""
    from types import SimpleNamespace
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention

    owner = SimpleNamespace(ratio=ratio)
    owner._merge_topk = PagedCSA2Attention._merge_topk
    owner._scores = lambda p, rows, q, w: full_prefill_sram_scores(q, w, packed, table, p, rows, ratio)
    rows = torch.arange(source_rows, device=query.device, dtype=torch.int32)
    selected, scores, blocks = PagedCSA2Attention._stream_topk(owner,
                                                               positions,
                                                               rows,
                                                               query,
                                                               weights,
                                                               collect_blocks=publish_candidates,
                                                               visible_rows=visible_rows)
    count = ((positions + 1) // ratio).unsqueeze(-1)
    selected = torch.where((selected >= 0) & (selected < count), selected, source_rows).sort(-1).values
    selected = torch.where(selected < source_rows, selected, -1).int()
    if publish_candidates:
        blocks = torch.where(scores > -torch.inf, blocks, -1).int()
        blocks = torch.nn.functional.pad(blocks, (0, 2048 - blocks.shape[1]), value=-1)
    return selected, blocks


def tp_full_prefill_index_selection(query,
                                    weights,
                                    packed,
                                    table,
                                    positions,
                                    ratio,
                                    source_rows,
                                    publish_candidates,
                                    tp_rank,
                                    all_gather,
                                    visible_rows=None):
    """Partition replicated index work; restore row order before consumption.

    Query heads have already been gathered. Each rank evaluates complete head
    reductions for its query rows, then exchanges only final integer choices.
    Padding is local to selection and never reaches a KV or history writer.
    """
    tokens = query.shape[0]
    if (tp_rank not in (0, 1) or tokens < 1 or positions.shape != (tokens, )
            or not 512 <= source_rows <= SHARED_INDEX_MAX_ROWS):
        raise ValueError("Index query partition requires TP2 and a complete query interval")
    local_tokens = (tokens + 1) // 2
    start, stop = tp_rank * local_tokens, min((tp_rank + 1) * local_tokens, tokens)
    local_q = query[start:stop].clone()
    local_w = weights[start:stop].clone()
    local_p = positions[start:stop].clone()
    padding = local_tokens - (stop - start)
    if padding:
        local_q = torch.nn.functional.pad(local_q, (0, 0, 0, 0, 0, padding))
        local_w = torch.nn.functional.pad(local_w, (0, 0, 0, padding))
        local_p = torch.nn.functional.pad(local_p, (0, padding), value=-1)
    selected, blocks = full_prefill_index_selection(local_q, local_w, packed, table, local_p, ratio, source_rows,
                                                    publish_candidates, visible_rows)
    packet = torch.cat((selected, blocks), -1) if publish_candidates else selected
    _record_query_partition("full", tokens, local_tokens, packet)
    gathered = all_gather(packet.contiguous(), 0)[:tokens]
    return gathered[:, :512], gathered[:, 512:] if publish_candidates else None


def prefill_reindex_selection(query,
                              weights,
                              packed,
                              table,
                              positions,
                              blocks,
                              ratio,
                              source_rows,
                              native_gather=True,
                              native_scores=True):
    """Evaluate a row interval with the production bounded Reindex recipes."""
    signature = (tuple(packed.shape), tuple(table.shape), ratio, source_rows)
    keys = compiled_decode_shared_index_keys(signature)(packed, table, ratio, source_rows)
    outputs = []
    for start in range(0, query.shape[0], 128):
        stop = min(start + 128, query.shape[0])
        current = blocks[start:stop].clone()
        signature = (stop - start, tuple(current.shape), source_rows, ratio, 16, native_gather, native_scores)
        outputs.append(
            compiled_decoded_reindex(signature)(query[start:stop].clone(), weights[start:stop].clone(), keys,
                                                positions[start:stop].clone(), current, ratio, 16, native_gather,
                                                native_scores))
    return torch.cat(outputs, 0)


def tp_prefill_reindex_selection(query,
                                 weights,
                                 packed,
                                 table,
                                 positions,
                                 blocks,
                                 ratio,
                                 source_rows,
                                 tp_rank,
                                 all_gather,
                                 native_gather=True,
                                 native_scores=True):
    """Split independent candidate-slot selection and gather only final IDs."""
    tokens = query.shape[0]
    if tp_rank not in (0, 1) or tokens < 1 or blocks.shape[0] != tokens:
        raise ValueError("Reindex query partition requires TP2 and matching candidate rows")
    local_tokens = (tokens + 1) // 2
    start, stop = tp_rank * local_tokens, min((tp_rank + 1) * local_tokens, tokens)
    local_q, local_w, local_p, local_b = (value[start:stop].clone() for value in (query, weights, positions, blocks))
    padding = local_tokens - (stop - start)
    if padding:
        local_q = torch.nn.functional.pad(local_q, (0, 0, 0, 0, 0, padding))
        local_w = torch.nn.functional.pad(local_w, (0, 0, 0, padding))
        local_p = torch.nn.functional.pad(local_p, (0, padding), value=-1)
        local_b = torch.nn.functional.pad(local_b, (0, 0, 0, padding), value=-1)
    selected = prefill_reindex_selection(local_q, local_w, packed, table, local_p, local_b, ratio, source_rows,
                                         native_gather, native_scores)
    _record_query_partition("reindex", tokens, local_tokens, selected)
    return all_gather(selected.contiguous(), 0)[:tokens]


def decode_shared_index_keys(packed, table, ratio, source_rows):
    """Decode a bounded paged prefix once for all queries in this transaction.

    Call after the source's cache write. The result belongs to this invocation:
    retaining it across requests or producer writes would read stale keys.
    """
    if not 1 <= source_rows <= SHARED_INDEX_MAX_ROWS or ratio not in (1, 2):
        raise ValueError("Shared prefill index keys exceed the bounded workspace")
    pieces = []
    for start in range(0, source_rows, INDEX_KEY_TILE):
        rows = torch.arange(start, min(start + INDEX_KEY_TILE, source_rows), device=packed.device, dtype=torch.int32)
        if packed.device.type == "hpu":
            keys = torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2(packed, table, rows.reshape(1, -1),
                                                                             ratio)[0]
        else:
            page_rows = 128 // ratio
            physical = table[(rows // page_rows).long()] * page_rows + rows % page_rows
            keys = unpack_fp4(packed.index_select(0, physical.long()), 128, 32)
        pieces.append(keys)
    return torch.cat(pieces, 0)


def _weighted_index_scores(query, weights, keys, local_heads, preserve_rounding):
    # Flatten query/head rows into M: every MME tile reuses the same K rows.
    scores = (torch.matmul(query, keys.transpose(0, 1)) if keys.ndim == 2 else torch.bmm(query, keys.transpose(1, 2)))

    def boundary(value):
        if preserve_rounding and value.device.type == "hpu":
            return torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(value.reshape(1,
                                                                                              -1)).reshape(value.shape)
        return value

    scores = boundary(scores)
    scores = boundary(scores.relu() * weights.unsqueeze(-1))
    partial = boundary(scores.reshape(query.shape[0], 2, local_heads, -1).sum(2))
    return boundary(partial.sum(1)).float()


def decoded_reindex(query,
                    weights,
                    keys,
                    positions,
                    blocks,
                    ratio,
                    local_heads,
                    native_gather=False,
                    native_scores=False):
    """Reuse common keys without changing candidate-slot or top-k merge order.

    Decode reuse follows the shared-source ownership used by SGLang #40436;
    the bounded MME scoring and two BF16 TP sums are Gaudi-specific. Unlike
    a global top-k, gathering scores back to candidate slots preserves ties.
    """
    source_rows = keys.shape[0]
    if not 1 <= source_rows <= SHARED_INDEX_MAX_ROWS or not 1 <= query.shape[0] <= 128:
        raise ValueError("Shared Reindex requires bounded query and source tiles")
    common = []
    for start in range(0, source_rows, INDEX_KEY_TILE):
        current = keys[start:start + INDEX_KEY_TILE]
        count = current.shape[0]
        # Keep native SIMD tails in-bounds even for a non-power-of-two fixture.
        if count % 128:
            current = torch.cat((current, current.new_zeros((128 - count % 128, 128))), 0)
        if native_scores:
            if local_heads != 16 or query.shape[1:] != (32, 128):
                raise ValueError("Native Reindex scoring requires the TP2 head contract")
            rows = torch.arange(start, start + current.shape[0], device=query.device, dtype=torch.int32)
            scores = torch.ops.custom_op.custom_deepseek_v41_prefill_index_scores_gaudi2(
                query, weights, current, positions, rows, ratio)
        else:
            scores = _weighted_index_scores(query, weights, current, local_heads, True)
        common.append(scores[:, :count])
    common = torch.cat(common, -1)
    return _candidate_slot_topk(common, positions, blocks, ratio, native_gather)


def _candidate_slot_topk(common, positions, blocks, ratio, native_gather=False):
    source_rows = common.shape[1]
    if native_gather:
        gathered, rows = torch.ops.custom_op.custom_deepseek_v41_candidate_gather_f32_gaudi2(
            common, blocks, positions, ratio)
    else:
        rows = blocks.unsqueeze(-1) * 8 + torch.arange(8, device=common.device, dtype=torch.int32)
        rows = torch.where(blocks.unsqueeze(-1) >= 0, rows, -1).flatten(1)
    count = ((positions + 1) // ratio).unsqueeze(-1)
    best_values = best_rows = None
    for start in range(0, rows.shape[-1], INDEX_KEY_TILE):
        current = rows[:, start:start + INDEX_KEY_TILE]
        if native_gather:
            scores = gathered[:, start:start + INDEX_KEY_TILE]
        else:
            scores = common.gather(1, current.clamp(0, source_rows - 1).long())
            scores = scores.masked_fill((current < 0) | (current >= source_rows) | (current >= count), -torch.inf)
        values, offsets = scores.topk(min(512, scores.shape[-1]), dim=-1, sorted=False)
        selected = current.gather(1, offsets)
        if best_values is not None:
            values = torch.cat((best_values, values), -1)
            selected = torch.cat((best_rows, selected), -1)
            values, offsets = values.topk(min(512, values.shape[-1]), dim=-1, sorted=False)
            selected = selected.gather(1, offsets)
        best_values, best_rows = values, selected
    selected = torch.where((best_rows >= 0) & (best_rows < count) & (best_rows < source_rows), best_rows, source_rows)
    selected = selected.sort(-1).values
    return torch.where(selected < source_rows, selected, -1).int()


@lru_cache(maxsize=32)
def compiled_decode_shared_index_keys(signature):
    entry = FunctionType(decode_shared_index_keys.__code__.replace(co_name=f"prefill_shared_keys_{signature}"),
                         decode_shared_index_keys.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@lru_cache(maxsize=64)
def compiled_decoded_reindex(signature):
    entry = FunctionType(decoded_reindex.__code__.replace(co_name=f"prefill_decoded_reindex_{signature}"),
                         decoded_reindex.__globals__,
                         argdefs=decoded_reindex.__defaults__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def index_scores(query,
                 weights,
                 packed,
                 table,
                 positions,
                 rows,
                 ratio,
                 local_heads,
                 preserve_rounding=True,
                 native_keys=False):
    if native_keys:
        key_rows = rows.reshape(1, -1) if rows.ndim == 1 else rows
        keys = torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2(packed, table, key_rows, ratio)
        if rows.ndim == 1:
            keys = keys[0]
    else:
        safe = rows.clamp_min(0)
        page_rows = 128 // ratio
        physical = table[(safe.flatten() // page_rows).long()].reshape(safe.shape) * page_rows + safe % page_rows
        keys = unpack_fp4(packed.index_select(0, physical.flatten().long()).reshape(*physical.shape, 68), 128, 32)
    scores = (torch.einsum("thd,nd->thn", query, keys) if keys.ndim == 2 else torch.einsum("thd,tnd->thn", query, keys))

    def boundary(value):
        if preserve_rounding:
            return torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(value.reshape(1,
                                                                                              -1)).reshape(value.shape)
        return value

    scores = boundary(scores)
    scores = boundary(scores.relu() * weights.unsqueeze(-1))
    partial = boundary(scores.reshape(query.shape[0], 2, local_heads, -1).sum(2))
    reduced = boundary(partial.sum(1)).float()
    visible = ((positions + 1) // ratio).unsqueeze(-1)
    valid = (rows >= 0) & (rows < visible)
    return reduced.masked_fill(~valid, -torch.inf)


@lru_cache(maxsize=64)
def compiled_index_scores(signature):
    entry = FunctionType(index_scores.__code__.replace(co_name=f"prefill_index_scores_{signature}"),
                         index_scores.__globals__,
                         argdefs=index_scores.__defaults__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def shared_index_scores(query, weights, packed, table, positions, ratio, local_heads, source_rows):
    """Decode a bounded common source once, retaining all score boundaries.

    This computes scores only. Reindex must gather them back into its original
    candidate order before top-k: sorting a common source directly changes tie
    resolution and is not equivalent to the candidate-pool contract.
    """
    if not 1 <= source_rows <= 16384 or not 1 <= query.shape[0] <= 128:
        raise ValueError("Shared index scores require a bounded query/source tile")
    outputs = []
    for start in range(0, source_rows, 2048):
        rows = torch.arange(start, min(start + 2048, source_rows), device=query.device, dtype=torch.int32)
        outputs.append(index_scores(query, weights, packed, table, positions, rows, ratio, local_heads, True, True))
    return torch.cat(outputs, -1)


@lru_cache(maxsize=64)
def compiled_shared_index_scores(signature):
    entry = FunctionType(shared_index_scores.__code__.replace(co_name=f"shared_index_scores_{signature}"),
                         shared_index_scores.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def shared_reindex(query, weights, packed, table, positions, blocks, ratio, local_heads, source_rows):
    """Bounded common scoring followed by the original candidate-slot top-k.

    Preserve both the candidate order and the source-tile merge order. In
    particular, neither source deduplication nor a global top-k is a valid
    replacement for the existing tie behavior.
    """
    common = shared_index_scores(query, weights, packed, table, positions, ratio, local_heads, source_rows)
    rows = blocks.unsqueeze(-1) * 8 + torch.arange(8, device=query.device, dtype=torch.int32)
    rows = torch.where(blocks.unsqueeze(-1) >= 0, rows, -1).flatten(1)
    best_values = best_rows = None
    for start in range(0, rows.shape[-1], 2048):
        current = rows[:, start:start + 2048]
        scores = common.gather(1, current.clamp(0, source_rows - 1).long())
        scores = scores.masked_fill((current < 0) | (current >= source_rows), -torch.inf)
        values, offsets = scores.topk(min(512, scores.shape[-1]), dim=-1, sorted=False)
        selected = current.gather(1, offsets)
        if best_values is not None:
            values = torch.cat((best_values, values), -1)
            selected = torch.cat((best_rows, selected), -1)
            values, offsets = values.topk(min(512, values.shape[-1]), dim=-1, sorted=False)
            selected = selected.gather(1, offsets)
        best_values, best_rows = values, selected
    count = ((positions + 1) // ratio).unsqueeze(-1)
    # Source capacity is a safe sentinel strictly beyond every eligible row.
    selected = torch.where((best_rows >= 0) & (best_rows < count), best_rows, source_rows)
    selected = selected.sort(-1).values
    return torch.where(selected < source_rows, selected, -1).int()


@lru_cache(maxsize=64)
def compiled_shared_reindex(signature):
    entry = FunctionType(shared_reindex.__code__.replace(co_name=f"shared_reindex_{signature}"),
                         shared_reindex.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)
