# SPDX-License-Identifier: Apache-2.0
"""Partition independent index tiles while preserving every ordered merge.

The caller supplies replicated query/cache state and the ordinary TP gather.
Local TopK retains its original tile geometry. Scores and logical IDs travel
as FP32 numbers; the supported 1M logical address space is exactly representable.
"""

import torch
import torch.nn.functional as F

from vllm_gaudi.ops.deepseek_v41_paged_attention import PREFILL_INDEX_ROWS


def invalidate_local_index_queries(owner):
    """Release cold replicas after the execution owner has retired its plans."""
    owner.tp4_local_index_queries = False
    for kind in ("query", "score"):
        for rank in range(4):
            setattr(owner, f"_tp4_index_{kind}_shard_{rank}", None)


def prepare_local_index_queries(owner, *, release_local=False):
    """Prepare immutable rank-ordered projection shards outside execution."""
    if owner.tensor_parallel_size != 4 or not owner.owns_index:
        raise ValueError("Replicated index projections require a TP4 index owner")
    if owner.tp4_local_index_queries or any(
            getattr(owner, f"_tp4_index_{kind}_shard_{rank}", None) is not None
            for kind in ("query", "score") for rank in range(4)):
        raise ValueError("Index projections must be invalidated before preparing again")
    indexer = owner.weights.indexer
    query, score = indexer.wq_b.weight, indexer.weights_proj.weight
    if (query.dtype != torch.bfloat16 or tuple(query.shape) != (1024, 1280)
            or score.dtype != torch.bfloat16 or tuple(score.shape) != (8, 5120)
            or owner.index_heads != 8 or hasattr(indexer.wq_b, "bias")
            or hasattr(indexer.weights_proj, "bias") or hasattr(indexer.weights_proj, "scale")):
        raise ValueError("Index projections differ from the qualified TP4 BF16 shard contract")
    query_shards = owner.gather(query.contiguous(), 0).reshape(4, 1024, 1280)
    score_shards = owner.gather(score.contiguous(), 0).reshape(4, 8, 5120)
    for rank in range(4):
        # Materialize each immutable shard outside execution. Linear's weight
        # transposes can still produce batch_as_strided in the compiled plan.
        setattr(owner, f"_tp4_index_query_shard_{rank}", query_shards[rank].clone())
        setattr(owner, f"_tp4_index_score_shard_{rank}", score_shards[rank].clone())
    if release_local:
        # Prefill and wider decode retain their original local shard geometry
        # through references, without keeping duplicate rank-local allocations.
        rank = owner.prefill_tp_rank
        indexer.wq_b.weight = getattr(owner, f"_tp4_index_query_shard_{rank}")
        indexer.weights_proj.weight = getattr(owner, f"_tp4_index_score_shard_{rank}")
    owner.tp4_local_index_queries = True


def local_index_query_projections(owner, value, qr):
    """Keep each original shard GEMM and activation rounding boundary."""
    from vllm_gaudi.ops.deepseek_v41_math import quantize_activation
    query_weight = [getattr(owner, f"_tp4_index_query_shard_{rank}") for rank in range(4)]
    score_weight = [getattr(owner, f"_tp4_index_score_shard_{rank}") for rank in range(4)]
    if any(value is None for value in (*query_weight, *score_weight)):
        raise ValueError("Local index query projections were not prepared")
    quantized = quantize_activation(qr) if hasattr(owner.weights.indexer.wq_b, "scale") else qr
    query = torch.cat([F.linear(quantized, query_weight[rank]) for rank in range(4)], dim=-1)
    scores = torch.cat([F.linear(value, score_weight[rank]) for rank in range(4)], dim=-1)
    return query.reshape(value.shape[0], 32, 128), scores


def _layout(rows, visible_rows, width, collect_blocks, tp_size):
    columns = rows.shape[-1]
    if tp_size != 4 or width != 512 or columns < PREFILL_INDEX_ROWS or columns % PREFILL_INDEX_ROWS:
        raise ValueError("TP4 selection requires complete index tiles and Top512")
    if rows.ndim not in (1, 2) or rows.dtype not in (torch.int32, torch.int64):
        raise ValueError("Selection rows must be integer shared or per-query logical IDs")
    if visible_rows is not None and (rows.ndim != 1 or not 0 < visible_rows <= columns):
        raise ValueError("Visible source bound requires a nonempty shared prefix")
    active = columns if visible_rows is None else visible_rows
    active_tiles = (active + PREFILL_INDEX_ROWS - 1) // PREFILL_INDEX_ROWS
    per_rank = (active_tiles + tp_size - 1) // tp_size
    packet_width = 2 * width + (PREFILL_INDEX_ROWS // 8 if collect_blocks else 0)
    return active_tiles, per_rank, packet_width


def _rank_tiles(active_tiles, rank, tp_size=4):
    base, extra = divmod(active_tiles, tp_size)
    start = rank * base + min(rank, extra)
    return start, start + base + int(rank < extra)


def local_selection_packets(owner, positions, rows, q, weights, *, rank, width=512,
                            collect_blocks=False, visible_rows=None, tp_size=4,
                            contiguous=False, prefix_scores=None):
    """Compute only this rank's original local sorts; do not merge them yet."""
    active_tiles, per_rank, packet_width = _layout(rows, visible_rows, width, collect_blocks, tp_size)
    if not 0 <= rank < tp_size:
        raise ValueError("Invalid TP selection rank")
    packets = []
    first, stop = _rank_tiles(active_tiles, rank, tp_size) if contiguous else (rank, active_tiles)
    for tile in range(first, stop, 1 if contiguous else tp_size):
        start = tile * PREFILL_INDEX_ROWS
        current = rows[..., start:start + PREFILL_INDEX_ROWS]
        local_start = (tile - first) * PREFILL_INDEX_ROWS
        scores = (owner._scores(positions, current, q, weights) if prefix_scores is None else
                  prefix_scores[:, local_start:local_start + PREFILL_INDEX_ROWS])
        values, offsets = scores.topk(width, dim=-1, sorted=False)
        selected = current.expand_as(scores).gather(1, offsets) if rows.ndim == 1 else current.gather(1, offsets)
        parts = [values, selected.to(torch.float32)]
        if collect_blocks:
            parts.append(scores.reshape(scores.shape[0], -1, 8).amax(-1))
        packets.append(torch.cat(parts, dim=-1))
    while len(packets) < per_rank:
        # Padding has no logical tile and is never consumed by the merge.
        packets.append(q.new_zeros((q.shape[0], packet_width), dtype=torch.float32))
    return torch.cat(packets, dim=-1)


def merge_selection_packets(owner, positions, rows, q, gathered, *, width=512,
                            collect_blocks=False, visible_rows=None, tp_size=4, contiguous=False):
    """Restore tile order and apply the unchanged sequential merge network."""
    active_tiles, per_rank, packet_width = _layout(rows, visible_rows, width, collect_blocks, tp_size)
    if gathered.shape != (q.shape[0], tp_size * per_rank * packet_width) or gathered.dtype != torch.float32:
        raise ValueError("Gathered selection packets have an incompatible shape or dtype")
    packets = gathered.reshape(q.shape[0], tp_size, per_rank, packet_width)
    best_scores = best_rows = None
    block_scores = block_ids = None
    for tile in range(rows.shape[-1] // PREFILL_INDEX_ROWS):
        if tile < active_tiles:
            if contiguous:
                # Every participating rank owns a real tile. An empty rank's
                # constant packet could otherwise move ahead of query gathers
                # during partition scheduling and change collective order.
                base, extra = divmod(active_tiles, tp_size)
                larger_tiles = (base + 1) * extra
                if tile < larger_tiles:
                    rank, local = divmod(tile, base + 1)
                else:
                    tail_rank, local = divmod(tile - larger_tiles, base)
                    rank = extra + tail_rank
            else:
                rank, local = tile % tp_size, tile // tp_size
            packet = packets[:, rank, local, :]
            values = packet[:, :width]
            selected = packet[:, width:2 * width].to(rows.dtype)
            if best_scores is not None:
                # The local TopK has already run. Applying it again would
                # reorder tied values before this existing 1024-wide merge.
                values = torch.cat((best_scores, values), -1)
                selected = torch.cat((best_rows, selected), -1)
                values, offsets = values.topk(width, dim=-1, sorted=False)
                selected = selected.gather(1, offsets)
            best_scores, best_rows = values, selected
        if collect_blocks:
            grouped = (packet[:, 2 * width:] if tile < active_tiles else
                       q.new_full((q.shape[0], PREFILL_INDEX_ROWS // 8), -torch.inf, dtype=torch.float32))
            start = tile * PREFILL_INDEX_ROWS
            ids = torch.arange(start // 8, (start + PREFILL_INDEX_ROWS) // 8,
                               device=positions.device, dtype=torch.int32)
            count = ((positions + 1) // owner.ratio).unsqueeze(-1)
            grouped = grouped.masked_fill(ids == ((count - 1) // 8), torch.inf)
            block_scores, block_ids = owner._merge_topk(block_scores, block_ids, grouped, ids, 2048)
    return best_rows, block_scores, block_ids


def tp4_stream_topk(owner, positions, rows, q, weights, *, width=512,
                    collect_blocks=False, visible_rows=None):
    local = local_selection_packets(owner, positions, rows, q, weights, rank=owner.prefill_tp_rank,
                                    width=width, collect_blocks=collect_blocks, visible_rows=visible_rows)
    gathered = owner.gather(local, 1)
    return merge_selection_packets(owner, positions, rows, q, gathered, width=width,
                                   collect_blocks=collect_blocks, visible_rows=visible_rows)


def supports_mirror_partition(owner, rows, q, active_columns, width):
    return (owner.tensor_parallel_size == 4 and q.shape[0] == 1 and rows.ndim == 1 and width == 512
            and 4 * PREFILL_INDEX_ROWS <= active_columns <= rows.shape[-1] <= 32768
            and active_columns % PREFILL_INDEX_ROWS == 0 and rows.shape[-1] % PREFILL_INDEX_ROWS == 0
            and rows.dtype == torch.int32 and owner._uses_index_mirror(q))


def tp4_mirror_stream_topk(owner, positions, rows, q, weights, *, width=512,
                          collect_blocks=False, visible_rows=None):
    """Score one contiguous mirror range per rank; keep original tile merges."""
    active_tiles, per_rank, _ = _layout(rows, visible_rows, width, collect_blocks, 4)
    if active_tiles < 4:
        raise ValueError("Mirror partition requires a real scoring tile on every rank")
    rank = owner.prefill_tp_rank
    if not 0 <= rank < 4:
        raise ValueError("Invalid TP4 index owner rank")
    first, last = _rank_tiles(active_tiles, rank)
    start, stop = first * PREFILL_INDEX_ROWS, last * PREFILL_INDEX_ROWS
    prefix_scores = None
    if start < stop:
        prefix_scores = torch.ops.custom_op.custom_deepseek_v41_prefill_index_scores_gaudi2(
            q.contiguous(), weights.contiguous(), owner.cache.index_mirror[start:stop].contiguous(),
            positions.to(torch.int32).contiguous(), rows[start:stop].contiguous(),
            owner.ratio, owner.index_heads)
    local = local_selection_packets(owner, positions, rows, q, weights, rank=rank, width=width,
                                    collect_blocks=collect_blocks, visible_rows=visible_rows,
                                    contiguous=True, prefix_scores=prefix_scores)
    gathered = owner.gather(local, 1)
    return merge_selection_packets(owner, positions, rows, q, gathered, width=width,
                                   collect_blocks=collect_blocks, visible_rows=visible_rows, contiguous=True)
