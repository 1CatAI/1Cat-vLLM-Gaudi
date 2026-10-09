# SPDX-License-Identifier: Apache-2.0
"""Actual C6 score operands against the accepted per-row C1 ordered selector."""
import torch


def install(owner, rows, columns):
    from vllm_gaudi.ops import deepseek_v41_decode_selection as selection

    owner._selection_oracle_blocks = owner.layer == owner.candidate_source
    specifications = (
        ('scores', (rows, columns), torch.float32), ('rows', (rows, columns), torch.int32),
        ('positions', (rows,), torch.int32), ('indices', (rows, 512), torch.int32),
    )
    if owner._selection_oracle_blocks:
        specifications += (('block_ids', (rows, 2048), torch.int32), ('block_scores', (rows, 2048), torch.float32))
    for name, shape, dtype in specifications:
        owner.register_buffer('selection_oracle_' + name, torch.empty(shape, dtype=dtype, device='hpu'), False)
    original = selection.threshold_decode_selection
    owner._selection_oracle_ratio = owner.ratio
    owner._selection_oracle_blocks = owner.layer == owner.candidate_source

    def captured(score_tiles, source, positions, ratio, *, collect_blocks=False):
        scores = torch.cat(score_tiles, -1)
        owner.selection_oracle_scores.copy_(scores)
        source = source.expand(scores.shape[0], -1) if source.ndim == 1 else source
        owner.selection_oracle_rows.copy_(source)
        owner.selection_oracle_positions.copy_(positions)
        result = original(score_tiles, source, positions, ratio, collect_blocks=collect_blocks)
        owner.selection_oracle_indices.copy_(result[0])
        if collect_blocks:
            owner.selection_oracle_block_ids.copy_(result[2])
            owner.selection_oracle_block_scores.copy_(result[1])
        return result

    selection.threshold_decode_selection = captured


def audit(owner):
    from vllm_gaudi.ops.deepseek_v41_indexer import ordered_index_ids

    score = owner.selection_oracle_scores.clone()
    source = owner.selection_oracle_rows.clone()
    positions = owner.selection_oracle_positions.clone()
    actual_ids = owner.selection_oracle_indices.cpu()
    block_ids = owner.selection_oracle_block_ids.cpu() if owner._selection_oracle_blocks else None
    block_scores = owner.selection_oracle_block_scores.cpu() if owner._selection_oracle_blocks else None
    expected, blocks, values = [], [], []
    rows, columns = score.shape
    for i in range(rows):
        extent = torch.full((1,), columns - 1, dtype=torch.int32, device='hpu')
        unused = torch.empty((1, 2048), dtype=torch.int32, device='hpu')
        offsets = ordered_index_ids(score[i:i + 1].contiguous(), extent, unused, 1)
        ids = source[i:i + 1].gather(1, offsets.clamp_min(0).long())
        expected.append(torch.where(offsets >= 0, ids, -1).cpu())
        if owner._selection_oracle_blocks:
            grouped = score[i:i + 1].reshape(1, -1, 8).amax(-1)
            bid = torch.arange(columns // 8, dtype=torch.int32, device='hpu').reshape(1, -1)
            newest = ((positions[i:i + 1] + 1) // owner._selection_oracle_ratio - 1) // 8
            grouped = grouped.masked_fill(bid == newest.reshape(1, 1), torch.inf).contiguous()
            if grouped.shape[-1] > 2048:
                bid = ordered_index_ids(grouped, extent, unused, 1, blocks=True)
                grouped = grouped.gather(1, bid.clamp_min(0).long())
                grouped = torch.where(bid >= 0, grouped, -torch.inf)
            blocks.append(bid.cpu())
            values.append(grouped.cpu())
    score_bf16_exact = torch.equal(score.cpu(), score.bfloat16().float().cpu())
    ids_exact = torch.equal(actual_ids, torch.cat(expected))
    blocks_exact = (not owner._selection_oracle_blocks or
                    (torch.equal(block_ids, torch.cat(blocks)) and torch.equal(block_scores, torch.cat(values))))
    return dict(reference='Same actual score/causal mask/logical row operands; accepted C1 ordered_index_ids per row',
                scores_have_bf16_boundary=score_bf16_exact, selected_ids_exact=ids_exact,
                candidate_blocks_exact=blocks_exact, passed=bool(score_bf16_exact and ids_exact and blocks_exact),
                full_service_quality_passed=False)
