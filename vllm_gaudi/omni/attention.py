# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense packed attention for vLLM-Omni diffusion models on Gaudi."""

from __future__ import annotations

import os

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl

HPU_SDPA_SOFTMAX_MODE = os.environ.get("VLLM_GAUDI_OMNI_SDPA_SOFTMAX_MODE", "None")
if HPU_SDPA_SOFTMAX_MODE not in {"fp32", "None", "fast"}:
    raise ValueError("VLLM_GAUDI_OMNI_SDPA_SOFTMAX_MODE must be one of "
                     f"'fp32', 'None', or 'fast', got {HPU_SDPA_SOFTMAX_MODE!r}")


class HPUSDPABackend(SDPABackend):
    """Gaudi FusedSDPA backend with mask-free H3 suffix-padding support."""

    accept_output_buffer = False
    supports_prefix_kv_slicing = True
    supported_platforms = ("oot", )

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "HPU_SDPA"

    @staticmethod
    def get_impl_cls() -> type[HPUSDPAImpl]:
        return HPUSDPAImpl


class HPUSDPAImpl(SDPAImpl):
    """Invoke Habana FusedSDPA without materializing a full padding mask."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if query.device.type != "hpu" or key.device.type != "hpu" or value.device.type != "hpu":
            raise RuntimeError("HPU_SDPA received non-HPU tensors; implicit host fallback is disabled")

        original_q_length = query.shape[1]
        q_length = original_q_length
        kv_length = key.shape[1]
        attention_mask = None
        if attn_metadata is not None:
            if attn_metadata.packed_padding is not None:
                q_length = int(attn_metadata.packed_padding.q_length)
                kv_length = int(attn_metadata.packed_padding.kv_length)
            elif "valid_kv_length" in attn_metadata.extra:
                kv_length = int(attn_metadata.extra["valid_kv_length"])
                q_length = min(q_length, kv_length)
            attention_mask = attn_metadata.attn_mask

        if attention_mask is not None and attention_mask.device.type != "hpu":
            raise RuntimeError("HPU_SDPA received a non-HPU attention mask; implicit host fallback is disabled")

        query = query[:, :q_length].permute(0, 2, 1, 3).contiguous()
        key = key[:, :kv_length].permute(0, 2, 1, 3).contiguous()
        value = value[:, :kv_length].permute(0, 2, 1, 3).contiguous()

        if query.shape[1] != key.shape[1]:
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError("HPU FusedSDPA requires query heads to be a multiple of KV heads, "
                                 f"got {query.shape[1]} and {key.shape[1]}")
            repeat = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :kv_length]
            else:
                attention_mask = attention_mask[..., :q_length, :kv_length]

        from habana_frameworks.torch.hpex.kernels import FusedSDPA

        output = FusedSDPA.apply(
            query,
            key,
            value,
            attention_mask,
            0.0,
            self.causal,
            self.softmax_scale,
            HPU_SDPA_SOFTMAX_MODE,
            True,
        ).permute(0, 2, 1, 3)

        if q_length < original_q_length:
            padding = output.new_zeros(
                output.shape[0],
                original_q_length - q_length,
                output.shape[2],
                output.shape[3],
            )
            output = torch.cat((output, padding), dim=1)
        return output


__all__ = ["HPU_SDPA_SOFTMAX_MODE", "HPUSDPABackend", "HPUSDPAImpl"]
