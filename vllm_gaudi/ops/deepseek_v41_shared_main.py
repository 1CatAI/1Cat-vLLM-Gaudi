# SPDX-License-Identifier: Apache-2.0
"""Selected main rows shared only by consumers in the current bounded layer group."""
import torch

from vllm_gaudi import envs as gaudi_envs


class DecodedSwaRing(torch.nn.Module):
    """A request-owned BF16 ring using the existing nested SWA state contract."""
    def __init__(self, device):
        super().__init__()
        self.register_buffer("swa", torch.zeros((256, 512), dtype=torch.bfloat16, device=device), False)


def shared_main_attention(owner, query, positions, selected, lengths, workspace, completion=None, *, projection=False, decoded_swa=None):
    if query.shape[0] == 1:
        return _shared_main_c1(owner, query, positions, selected, lengths, workspace,
                               projection=projection, decoded_swa=decoded_swa)
    # A selection owner can change without a KV owner change (reindex layers).
    # Neither ownership key alone identifies the reusable selected rows.
    key = (owner.kv_source, owner.index_source, owner.ratio)
    query = query.contiguous()
    positions = positions.to(torch.int32).contiguous()
    batched = query.shape[0] > 1
    flat_qk = 2 <= query.shape[0] <= 6 and getattr(owner, "dspark_qk_flat", False)
    direct_qk = 2 <= query.shape[0] <= 6 and getattr(owner, "dspark_qk_flat_direct", False)
    if 2 <= query.shape[0] <= 6 and getattr(owner, "dspark_stream_exp", False):
        phase = owner._rotary_native_table()
        if key in workspace:
            rows, mask = workspace[key]
            return torch.ops.custom_op.custom_deepseek_v41_main_stream_exp_reuse_mla_gaudi2(
                query, owner.swa, rows, mask, positions, owner.weights.attn_sink, owner.scale, lengths, phase)
        output, rows, mask = torch.ops.custom_op.custom_deepseek_v41_main_stream_exp_publish_mla_gaudi2(
            query, owner.swa, owner.cache.main, selected.contiguous(), positions, owner.shared.block_table,
            owner.weights.attn_sink, owner.scale, lengths, owner.ratio, phase)
        workspace[key] = rows, mask
        return output
    if batched and getattr(owner, "dspark_layer_main_split", False):
        if getattr(owner, "dspark_main_single_bank", False):
            if key in workspace:
                rows, mask = workspace[key]
                return torch.ops.custom_op.custom_deepseek_v41_main_single_bank_reuse_mla_gaudi2(
                    query, owner.swa, rows, mask, positions, owner.weights.attn_sink, owner.scale, lengths)
            output, rows, mask = torch.ops.custom_op.custom_deepseek_v41_main_single_bank_publish_mla_gaudi2(
                query, owner.swa, owner.cache.main, selected.contiguous(), positions, owner.shared.block_table,
                owner.weights.attn_sink, owner.scale, lengths, owner.ratio)
            workspace[key] = rows, mask
            return output
        if getattr(owner, "dspark_swa_cache", False):
            if completion is None:
                raise ValueError("SWA cache consumer requires the current writer completion")
            if key in workspace:
                rows, mask, values = workspace[key]
                return torch.ops.custom_op.custom_deepseek_v41_swa_cached_reuse_mla_gaudi2(
                    query, owner.swa_decoded.swa, rows, mask, positions, owner.weights.attn_sink, owner.scale,
                    lengths, values, completion)
            output, rows, mask, values = torch.ops.custom_op.custom_deepseek_v41_swa_cached_publish_mla_gaudi2(
                query, owner.swa_decoded.swa, owner.cache.main, selected.contiguous(), positions,
                owner.shared.block_table, owner.weights.attn_sink, owner.scale, lengths, owner.ratio,
                completion)
            workspace[key] = rows, mask, values
            return output
        if key in workspace:
            rows, mask, values = workspace[key]
            reuse = (torch.ops.custom_op.custom_deepseek_v41_main_qk_flat_direct_reuse_mla_gaudi2
                     if direct_qk else
                     torch.ops.custom_op.custom_deepseek_v41_main_qk_flat_reuse_mla_gaudi2
                     if flat_qk else
                     torch.ops.custom_op.custom_deepseek_v41_coherent_swa_mla_gaudi2
                     if getattr(owner, "dspark_coherent_swa", False) else
                     torch.ops.custom_op.custom_deepseek_v41_main_adjacent_reuse_mla_gaudi2
                     if getattr(owner, "dspark_main_adjacent_pv", False) else
                     torch.ops.custom_op.custom_deepseek_v41_swa_source_reuse_mla_gaudi2
                     if getattr(owner, "dspark_swa_source_reuse", False) else
                     torch.ops.custom_op.custom_deepseek_v41_main_split_reuse_mla_gaudi2)
            return reuse(
                query, owner.swa, rows, mask, positions, owner.weights.attn_sink, owner.scale, lengths, values)
        publish = (torch.ops.custom_op.custom_deepseek_v41_main_qk_flat_direct_publish_mla_gaudi2
                   if direct_qk else
                   torch.ops.custom_op.custom_deepseek_v41_main_qk_flat_publish_mla_gaudi2
                   if flat_qk else torch.ops.custom_op.custom_deepseek_v41_main_split_publish_mla_gaudi2)
        output, rows, mask, values = publish(
            query, owner.swa, owner.cache.main, selected.contiguous(), positions, owner.shared.block_table,
            owner.weights.attn_sink, owner.scale, lengths, owner.ratio)
        workspace[key] = rows, mask, values
        return output
    publish = (torch.ops.custom_op.custom_deepseek_v41_main_batch_publish_mla_gaudi2
               if batched else torch.ops.custom_op.custom_deepseek_v41_main_publish_mla_gaudi2)
    reuse = (torch.ops.custom_op.custom_deepseek_v41_main_batch_reuse_mla_gaudi2
             if batched else torch.ops.custom_op.custom_deepseek_v41_main_reuse_mla_gaudi2)
    if key in workspace:
        rows, mask = workspace[key]
        return reuse(
            query, owner.swa, rows, mask, positions, owner.weights.attn_sink, owner.scale, lengths)
    output, rows, mask = publish(
        query, owner.swa, owner.cache.main, selected.contiguous(), positions, owner.shared.block_table,
        owner.weights.attn_sink, owner.scale, lengths, owner.ratio)
    workspace[key] = rows, mask
    return output


def _shared_main_c1(owner, query, positions, selected, lengths, workspace, *, projection=False, decoded_swa=None):
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
    tail = (True,) if gaudi_envs.VLLM_HPU_DSV41_MLA_REGISTER_SOFTMAX and query.shape[0] == 1 else ()
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
