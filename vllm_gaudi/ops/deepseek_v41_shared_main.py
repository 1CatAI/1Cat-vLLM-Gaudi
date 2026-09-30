# SPDX-License-Identifier: Apache-2.0
"""Selected main rows shared only by consumers in the current C1 layer group."""
import torch


def shared_main_attention(owner, query, positions, selected, lengths, workspace):
    # A selection owner can change without a KV owner change (reindex layers).
    # Neither ownership key alone identifies the reusable selected rows.
    key = (owner.kv_source, owner.index_source, owner.ratio)
    query = query.contiguous()
    positions = positions.to(torch.int32).contiguous()
    if key in workspace:
        rows, mask = workspace[key]
        return torch.ops.custom_op.custom_deepseek_v41_main_reuse_mla_gaudi2(
            query, owner.swa, rows, mask, positions, owner.weights.attn_sink, owner.scale, lengths)
    output, rows, mask = torch.ops.custom_op.custom_deepseek_v41_main_publish_mla_gaudi2(
        query, owner.swa, owner.cache.main, selected.contiguous(), positions, owner.shared.block_table,
        owner.weights.attn_sink, owner.scale, lengths, owner.ratio)
    workspace[key] = rows, mask
    return output
