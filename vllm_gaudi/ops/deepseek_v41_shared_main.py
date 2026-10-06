# SPDX-License-Identifier: Apache-2.0
"""Selected main rows shared only by consumers in the current C1 layer group."""
import torch

from vllm_gaudi import envs as gaudi_envs


def shared_main_attention(owner, query, positions, selected, lengths, workspace, *, projection=False, decoded_swa=None):
    # A selection owner can change without a KV owner change (reindex layers).
    # Neither ownership key alone identifies the reusable selected rows.
    key = (owner.kv_source, owner.index_source, owner.ratio)
    query = query.contiguous()
    positions = positions.to(torch.int32).contiguous()
    if (gaudi_envs.VLLM_HPU_DSV41_MLA_VECTOR_MASK
            or gaudi_envs.VLLM_HPU_DSV41_MLA_PUBLISH_MASK) and not gaudi_envs.VLLM_HPU_DSV41_MLA_VECTOR_CODEC:
        raise ValueError("Vector MLA mask candidate requires the qualified vector codec parent")
    if gaudi_envs.VLLM_HPU_DSV41_MLA_REUSE_HW_CODEC and not (
            gaudi_envs.VLLM_HPU_DSV41_MLA_VECTOR_CODEC and gaudi_envs.VLLM_HPU_DSV41_MLA_VECTOR_MASK):
        raise ValueError("Hardware reuse codec requires the vector codec and vector mask parents")
    tail = (True,) if gaudi_envs.VLLM_HPU_DSV41_MLA_REGISTER_SOFTMAX else ()
    if projection:
        wa, wb = owner.weights.wo_a, owner.weights.wo_b
        tail = (wa.weight, wa.channel_scale, owner._rotary_native_table(), wb.weight, wb.channel_scale)
    if key in workspace:
        rows, mask = workspace[key]
        if projection:
            operation = torch.ops.custom_op.custom_deepseek_v41_main_reuse_projection_gaudi2
        elif gaudi_envs.VLLM_HPU_DSV41_MLA_DECODED_SWA and decoded_swa is not None and query.shape[0] == 1:
            if (decoded_swa.dtype != torch.bfloat16 or decoded_swa.shape != (512, 512)
                    or decoded_swa.device != query.device or not decoded_swa.is_contiguous()):
                raise ValueError("Decoded SWA reader requires the current publisher's contiguous layer slot")
            operation = torch.ops.custom_op.custom_deepseek_v41_main_reuse_decoded_swa_mla_gaudi2
            return operation(query, decoded_swa, rows, mask, positions,
                             owner.weights.attn_sink, owner.scale, lengths, *tail)
        elif gaudi_envs.VLLM_HPU_DSV41_MLA_REUSE_HW_CODEC and query.shape[0] == 1:
            operation = torch.ops.custom_op.custom_deepseek_v41_main_reuse_native_codec_mla_gaudi2
        elif gaudi_envs.VLLM_HPU_DSV41_MLA_VECTOR_MASK and query.shape[0] == 1:
            operation = torch.ops.custom_op.custom_deepseek_v41_main_reuse_vector_mask_mla_gaudi2
        elif gaudi_envs.VLLM_HPU_DSV41_MLA_VECTOR_CODEC:
            operation = torch.ops.custom_op.custom_deepseek_v41_main_reuse_vector_mla_gaudi2
        else:
            operation = torch.ops.custom_op.custom_deepseek_v41_main_reuse_mla_gaudi2
        return operation(query, owner.swa, rows, mask, positions, owner.weights.attn_sink, owner.scale, lengths, *tail)
    if projection:
        operation = torch.ops.custom_op.custom_deepseek_v41_main_publish_projection_gaudi2
    elif gaudi_envs.VLLM_HPU_DSV41_MLA_PUBLISH_MASK and query.shape[0] == 1:
        operation = torch.ops.custom_op.custom_deepseek_v41_main_publish_vector_mask_mla_gaudi2
    elif gaudi_envs.VLLM_HPU_DSV41_MLA_VECTOR_CODEC:
        operation = torch.ops.custom_op.custom_deepseek_v41_main_publish_vector_mla_gaudi2
    else:
        operation = torch.ops.custom_op.custom_deepseek_v41_main_publish_mla_gaudi2
    output, rows, mask = operation(query, owner.swa, owner.cache.main, selected.contiguous(), positions,
                                   owner.shared.block_table, owner.weights.attn_sink, owner.scale, lengths, owner.ratio,
                                   *tail)
    workspace[key] = rows, mask
    return output
