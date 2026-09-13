# SPDX-License-Identifier: Apache-2.0
"""Route upstream DFlash2 primitives through FlashInfer-Gaudi on HPU."""

from __future__ import annotations

import torch

from flashinfer_gaudi.dflash2 import grouped_conv, score_edges, top_k
from vllm_gaudi import envs as gaudi_envs


def _enabled_for(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "hpu" and gaudi_envs.VLLM_HPU_FLASHINFER_DFLASH2


def _rms_norm_reference(
    input_tensor: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    """Vectorized RMSNorm supporting DFlash's per-layer K-norm weights."""
    variance = input_tensor.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = input_tensor * torch.rsqrt(variance + epsilon).to(input_tensor.dtype)
    if weight is None:
        return normalized
    if weight.ndim == 1:
        return normalized * weight
    if weight.ndim == 2 and input_tensor.ndim >= 2:
        layer_weight = weight.reshape(weight.shape[0], *([1] * (input_tensor.ndim - 2)), weight.shape[1])
        return normalized * layer_weight
    raise ValueError("HPU DFlash2 RMSNorm expects a vector weight or a [layers, hidden] "
                     f"weight, got input={tuple(input_tensor.shape)}, weight={tuple(weight.shape)}")


def _hpu_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Apply the HPU RoPE kernel with the in-place vLLM custom-op contract."""
    from habana_frameworks.torch.hpex.kernels import (RotaryPosEmbeddingMode, apply_rotary_pos_emb)

    positions = positions.flatten()
    selected = cos_sin_cache.index_select(0, positions).view(positions.numel(), 1, -1)
    cos, sin = selected.chunk(2, dim=-1)
    if is_neox:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        mode = RotaryPosEmbeddingMode.BLOCKWISE
    else:
        cos = torch.repeat_interleave(cos, 2, dim=-1, output_size=head_size)
        sin = torch.repeat_interleave(sin, 2, dim=-1, output_size=head_size)
        mode = RotaryPosEmbeddingMode.PAIRWISE

    def apply(tensor: torch.Tensor) -> None:
        original_shape = tensor.shape
        view = tensor.view(positions.numel(), -1, head_size)
        rotated = apply_rotary_pos_emb(view, cos, sin, None, 0, mode)
        tensor.copy_(rotated.reshape(original_shape))

    apply(query)
    if key is not None:
        apply(key)


def register() -> None:
    """Install device-gated patches without changing CPU/CUDA behavior."""
    from vllm.model_executor.layers import logits_processor
    from vllm.model_executor.models import qwen3_dflash2
    from vllm import _custom_ops

    original_rms_norm = _custom_ops.rms_norm
    if not getattr(original_rms_norm, "_vllm_gaudi_dflash2", False):

        def _rms_norm(out, input_tensor, weight, epsilon):
            if _enabled_for(input_tensor):
                out.copy_(_rms_norm_reference(input_tensor, weight, epsilon))
                return None
            return original_rms_norm(out, input_tensor, weight, epsilon)

        _rms_norm._vllm_gaudi_dflash2 = True
        _custom_ops.rms_norm = _rms_norm

    original_rotary_embedding = _custom_ops.rotary_embedding
    if not getattr(original_rotary_embedding, "_vllm_gaudi_dflash2", False):

        def _rotary_embedding(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
            rope_dim_offset=0,
            inverse=False,
        ):
            if _enabled_for(query):
                if rope_dim_offset != 0 or inverse:
                    raise NotImplementedError("HPU DFlash2 context RoPE does not support offsets or inverse rotation")
                return _hpu_rotary_embedding(
                    positions,
                    query,
                    key,
                    head_size,
                    cos_sin_cache,
                    is_neox,
                )
            return original_rotary_embedding(
                positions,
                query,
                key,
                head_size,
                cos_sin_cache,
                is_neox,
                rope_dim_offset,
                inverse,
            )

        _rotary_embedding._vllm_gaudi_dflash2 = True
        _custom_ops.rotary_embedding = _rotary_embedding

    original_grouped_conv = qwen3_dflash2._grouped_conv
    if not getattr(original_grouped_conv, "_vllm_gaudi_dflash2", False):

        def _grouped_conv(
            hidden_states,
            delta,
            base,
            block_size,
            num_groups,
            group_size,
            taps,
        ):
            if _enabled_for(hidden_states):
                return grouped_conv(
                    hidden_states,
                    delta,
                    base,
                    block_size,
                    num_groups,
                    group_size,
                    taps,
                )
            return original_grouped_conv(
                hidden_states,
                delta,
                base,
                block_size,
                num_groups,
                group_size,
                taps,
            )

        _grouped_conv._vllm_gaudi_dflash2 = True
        qwen3_dflash2._grouped_conv = _grouped_conv

    original_score_edges = qwen3_dflash2._score_edges
    if not getattr(original_score_edges, "_vllm_gaudi_dflash2", False):

        def _score_edges(
            predecessor_table,
            successor_table,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            top_k_size,
        ):
            if _enabled_for(candidate_ids):
                return score_edges(
                    predecessor_table,
                    successor_table,
                    candidate_ids,
                    unary_logits,
                    hidden,
                    anchor_token_ids,
                )
            return original_score_edges(
                predecessor_table,
                successor_table,
                candidate_ids,
                unary_logits,
                hidden,
                anchor_token_ids,
                top_k_size,
            )

        _score_edges._vllm_gaudi_dflash2 = True
        qwen3_dflash2._score_edges = _score_edges

    original_topk = logits_processor._topk
    if not getattr(original_topk, "_vllm_gaudi_dflash2", False):

        def _topk(scores, k):
            if _enabled_for(scores):
                return top_k(scores, k, sorted=True, deterministic=True)
            return original_topk(scores, k)

        _topk._vllm_gaudi_dflash2 = True
        logits_processor._topk = _topk


register()

__all__ = ["register"]
