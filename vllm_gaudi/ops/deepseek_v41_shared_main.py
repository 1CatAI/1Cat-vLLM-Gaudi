# SPDX-License-Identifier: Apache-2.0
"""Selected main rows shared only by consumers in the current C1 layer group."""
import torch

from vllm_gaudi import envs as gaudi_envs


def shared_main_attention(owner, query, positions, selected, lengths, workspace, *, projection=False):
    # A selection owner can change without a KV owner change (reindex layers).
    # Neither ownership key alone identifies the reusable selected rows.
    key = (owner.kv_source, owner.index_source, owner.ratio)
    query = query.contiguous()
    positions = positions.to(torch.int32).contiguous()
    tail = ()
    if projection:
        wa, wb = owner.weights.wo_a, owner.weights.wo_b
        tail = (wa.weight, wa.channel_scale, owner._rotary_native_table(), wb.weight, wb.channel_scale)
    if key in workspace:
        rows, mask = workspace[key]
        operation = (torch.ops.custom_op.custom_deepseek_v41_main_reuse_projection_gaudi2 if projection
                     else torch.ops.custom_op.custom_deepseek_v41_main_reuse_vector_mla_gaudi2
                     if gaudi_envs.VLLM_HPU_DSV41_MLA_VECTOR_CODEC else
                     torch.ops.custom_op.custom_deepseek_v41_main_reuse_mla_gaudi2)
        return operation(query, owner.swa, rows, mask, positions, owner.weights.attn_sink, owner.scale, lengths, *tail)
    operation = (torch.ops.custom_op.custom_deepseek_v41_main_publish_projection_gaudi2 if projection
                 else torch.ops.custom_op.custom_deepseek_v41_main_publish_vector_mla_gaudi2
                 if gaudi_envs.VLLM_HPU_DSV41_MLA_VECTOR_CODEC else
                 torch.ops.custom_op.custom_deepseek_v41_main_publish_mla_gaudi2)
    output, rows, mask = operation(
        query, owner.swa, owner.cache.main, selected.contiguous(), positions, owner.shared.block_table,
        owner.weights.attn_sink, owner.scale, lengths, owner.ratio, *tail)
    workspace[key] = rows, mask
    return output
