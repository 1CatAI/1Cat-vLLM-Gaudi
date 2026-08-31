# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional Gaudi2-native Qwen3.8 post-convolution Q/K preparation."""

from __future__ import annotations

from pathlib import Path

import torch

import vllm_gaudi.envs as gaudi_envs

_EXPECTED_QKV_WIDTH = 10240
_EXPECTED_KEY_WIDTH = 2048
_EXPECTED_VALUE_WIDTH = 6144
_EXPECTED_HEAD_DIM = 128
_EXPECTED_QK_HEADS = 16
_EXPECTED_VALUE_HEADS = 48

_loaded_extension: Path | None = None
_compiled_conv_producer = None


def _resolve_post_conv_op():
    if gaudi_envs.VLLM_GDN_QWEN38_COMPACT_QK:
        return torch.ops.custom_op.qwen38_post_conv_qk_compact_bf16_gaudi2
    if gaudi_envs.VLLM_GDN_QWEN38_BF16_QK:
        return torch.ops.custom_op.qwen38_post_conv_qk_expanded_bf16_gaudi2
    return torch.ops.custom_op.qwen38_post_conv_qk_bf16_gaudi2


def _resolve_hpu_causal_conv_op():
    return torch.ops.hpu.causal_conv1d_fwd


def _qwen38_hpu_conv_producer_impl(
    packed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
):
    return _resolve_hpu_causal_conv_op()(
        packed_qkv,
        conv_state,
        weight.transpose(0, 1).contiguous(),
        bias,
        has_initial_state,
        query_start_loc,
        cache_indices,
        activation=True,
        pad_slot_id=-1,
    )


def _get_compiled_conv_producer():
    global _compiled_conv_producer
    if _compiled_conv_producer is None:
        _compiled_conv_producer = torch.compile(
            _qwen38_hpu_conv_producer_impl,
            backend="hpu_backend",
            fullgraph=True,
            options={"force_static_compile": True},
        )
    return _compiled_conv_producer


def validate_qwen38_native_qk_shape(
    *,
    tp_size: int,
    qkv_width: int,
    key_width: int,
    value_width: int,
    key_head_dim: int,
    value_head_dim: int,
) -> int:
    """Validate the fixed Qwen3.8 TP1 kernel contract and return head repeat."""
    actual = (
        tp_size,
        qkv_width,
        key_width,
        value_width,
        key_head_dim,
        value_head_dim,
    )
    expected = (
        1,
        _EXPECTED_QKV_WIDTH,
        _EXPECTED_KEY_WIDTH,
        _EXPECTED_VALUE_WIDTH,
        _EXPECTED_HEAD_DIM,
        _EXPECTED_HEAD_DIM,
    )
    if actual != expected:
        raise ValueError(
            "VLLM_GDN_QWEN38_NATIVE_QK_PREP currently requires the Qwen3.8 "
            "TP1 layout (qkv=10240, key=2048, value=6144, head_dim=128), "
            f"got {actual}."
        )
    return _EXPECTED_VALUE_HEADS // _EXPECTED_QK_HEADS


def load_qwen38_native_qk_prep() -> bool:
    """Load the PyTorch registration library for the opt-in native kernel."""
    global _loaded_extension

    if not gaudi_envs.VLLM_GDN_QWEN38_NATIVE_QK_PREP:
        return False

    from vllm_gaudi.patches import _patch_hpu_fx_stack_trace_parser

    _patch_hpu_fx_stack_trace_parser()

    configured_path = gaudi_envs.VLLM_GDN_QWEN38_NATIVE_QK_PREP_EXTENSION
    if not configured_path:
        raise RuntimeError(
            "VLLM_GDN_QWEN38_NATIVE_QK_PREP_EXTENSION must point to the "
            "built flashqla_pair_transform_pt2 extension when native Q/K "
            "preparation is enabled."
        )

    extension = Path(configured_path).expanduser().resolve()
    if not extension.is_file():
        raise RuntimeError(f"Native Q/K preparation extension does not exist: {extension}")
    if _loaded_extension is not None:
        if extension != _loaded_extension:
            raise RuntimeError(
                "Native Q/K preparation was already loaded from "
                f"{_loaded_extension}, cannot reload it from {extension}."
            )
        return True

    torch.ops.load_library(str(extension))
    try:
        _resolve_post_conv_op()
        if gaudi_envs.VLLM_GDN_QWEN38_NATIVE_COMPACT_KKT:
            torch.ops.custom_op.qwen38_compact_kkt_bf16_gaudi2
    except AttributeError as exc:
        raise RuntimeError(
            "The configured extension is missing a requested Qwen3.8 "
            "Gaudi2 custom op."
        ) from exc

    _loaded_extension = extension
    return True


def qwen38_native_qk_prep(packed_qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize packed Q/K into the selected expanded or compact layout."""
    return _resolve_post_conv_op()(packed_qkv)


@torch._dynamo.disable
def qwen38_hpu_conv_producer(
    packed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
):
    """Run compiled causal-conv outside the decoder's large HPU cluster."""
    return _get_compiled_conv_producer()(
        packed_qkv,
        conv_state,
        weight,
        bias,
        has_initial_state,
        query_start_loc,
        cache_indices,
    )
