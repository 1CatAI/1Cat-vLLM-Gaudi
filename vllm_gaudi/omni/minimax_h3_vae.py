# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU execution policy for the MiniMax H3 video VAE."""

from __future__ import annotations

import os
import threading
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)

_H3_VAE_TILE_BATCH_SIZE_ENV = "VLLM_GAUDI_H3_VAE_TILE_BATCH_SIZE"
_H3_VAE_PERSIST_BF16_WEIGHTS_ENV = "VLLM_GAUDI_H3_VAE_PERSIST_BF16_WEIGHTS"
_H3_VAE_COMPILE_SWIGLU_ENV = "VLLM_GAUDI_H3_VAE_COMPILE_SWIGLU"
_H3_VAE_COMPILE_QK_NORM_ENV = "VLLM_GAUDI_H3_VAE_COMPILE_QK_NORM"
_H3_VAE_COMPILE_ROPE_ENV = "VLLM_GAUDI_H3_VAE_COMPILE_ROPE"
_H3_VAE_FUSED_SDPA_ENV = "VLLM_GAUDI_H3_VAE_FUSED_SDPA"
_DEFAULT_H3_VAE_TILE_BATCH_SIZE = 4


def _h3_vae_tile_batch_size() -> int:
    raw = os.environ.get(_H3_VAE_TILE_BATCH_SIZE_ENV, str(_DEFAULT_H3_VAE_TILE_BATCH_SIZE)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_H3_VAE_TILE_BATCH_SIZE_ENV} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{_H3_VAE_TILE_BATCH_SIZE_ENV} must be a positive integer, got {raw!r}")
    return value


def _boolean_environment(name: str, default: str) -> bool:
    value = os.environ.get(name, default).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _h3_vae_persist_bf16_weights() -> bool:
    return _boolean_environment(_H3_VAE_PERSIST_BF16_WEIGHTS_ENV, "1")


def _h3_vae_compile_swiglu() -> bool:
    return _boolean_environment(_H3_VAE_COMPILE_SWIGLU_ENV, "1")


def _h3_vae_compile_qk_norm() -> bool:
    return _boolean_environment(_H3_VAE_COMPILE_QK_NORM_ENV, "1")


def _h3_vae_compile_rope() -> bool:
    return _boolean_environment(_H3_VAE_COMPILE_ROPE_ENV, "1")


def _h3_vae_fused_sdpa_enabled() -> bool:
    return _boolean_environment(_H3_VAE_FUSED_SDPA_ENV, "1")


def _h3_vae_swiglu(projected: torch.Tensor) -> torch.Tensor:
    gate, value = projected.chunk(2, dim=-1)
    return F.silu(gate) * value


def _h3_vae_value_signature(*values: Any) -> tuple[Any, ...]:
    return tuple((tuple(value.shape), tuple(value.stride()), value.dtype,
                  value.device) if isinstance(value, torch.Tensor) else (type(value), value) for value in values)


class _H3VAEVerifiedCompiled:
    """Compile a pointwise region and retain it only while outputs stay exact."""

    def __init__(self, function: Any, label: str) -> None:
        self._reference = function
        self._label = label
        self._compiled = torch.compile(
            function,
            backend="hpu_backend",
            fullgraph=True,
            dynamic=False,
        )
        self._lock = threading.Lock()
        self._verified: set[tuple[Any, ...]] = set()
        self._disabled = False

    def __call__(self, *values: Any) -> torch.Tensor:
        if self._disabled:
            return self._reference(*values)
        try:
            output = self._compiled(*values)
        except Exception as exc:
            self._disabled = True
            logger.warning_once(
                "MiniMax H3 video VAE compiled %s failed (%s: %s); using eager HPU operators",
                self._label,
                type(exc).__name__,
                exc,
            )
            return self._reference(*values)

        signature = _h3_vae_value_signature(*values)
        if signature in self._verified:
            return output
        with self._lock:
            if signature in self._verified:
                return output
            reference = self._reference(*values)
            if not torch.equal(output, reference):
                self._disabled = True
                logger.warning_once("MiniMax H3 video VAE compiled %s changed output; using eager HPU operators",
                                    self._label)
                return reference
            self._verified.add(signature)
        return output


class _H3VAECompiledSwiGLU(_H3VAEVerifiedCompiled):
    """Share one exact compiled pointwise graph across all decoder blocks."""

    def __init__(self) -> None:
        super().__init__(_h3_vae_swiglu, "SwiGLU")


def _h3_vae_feed_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    projected = self.w1(hidden_states)
    if (projected.device.type == "hpu" and projected.dtype == torch.bfloat16 and projected.is_contiguous()
            and getattr(self, "use_gated", False)):
        hidden_states = self._vllm_gaudi_swiglu(projected)
        return self.w2(hidden_states)
    return self._vllm_gaudi_original_forward(hidden_states)


def _install_h3_vae_compiled_swiglu(decoder: nn.Module) -> int:
    """Compile only the exact BF16 SwiGLU pointwise region of H3's ViT."""

    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        return 0
    feed_forwards = []
    for block in blocks:
        feed_forward = getattr(block, "ff", None)
        if (not isinstance(feed_forward, nn.Module) or not isinstance(getattr(feed_forward, "w1", None), nn.Linear)
                or not isinstance(getattr(feed_forward, "w2", None), nn.Linear)
                or not isinstance(getattr(feed_forward, "act_fn", None), nn.SiLU)
                or not getattr(feed_forward, "use_gated", False)
                or feed_forward.w1.out_features != 2 * feed_forward.w2.in_features):
            return 0
        feed_forwards.append(feed_forward)

    feed_forwards = [
        feed_forward for feed_forward in feed_forwards if not hasattr(feed_forward, "_vllm_gaudi_original_forward")
    ]
    if not feed_forwards:
        return 0
    try:
        shared = _H3VAECompiledSwiGLU()
    except Exception as exc:
        logger.warning_once(
            "MiniMax H3 video VAE SwiGLU compilation setup failed (%s: %s); using eager HPU operators",
            type(exc).__name__,
            exc,
        )
        return 0
    for feed_forward in feed_forwards:
        feed_forward._vllm_gaudi_original_forward = feed_forward.forward
        feed_forward._vllm_gaudi_swiglu = shared
        feed_forward.forward = MethodType(_h3_vae_feed_forward, feed_forward)
    return len(feed_forwards)


class _H3VAECompiledRMSNorm(_H3VAEVerifiedCompiled):
    """Share one exact normalization graph across decoder Q/K projections."""

    def __init__(
        self,
        normalized_shape: tuple[int, ...],
        eps: float,
        output_dtype: torch.dtype | None = None,
    ) -> None:

        def rms_norm(value: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
            output = F.rms_norm(value, normalized_shape, weight, eps)
            if output_dtype is not None:
                output = output.to(output_dtype)
            return output

        super().__init__(rms_norm, "RMSNorm")


def _h3_vae_qk_rms_norm_forward(self: nn.RMSNorm, value: torch.Tensor) -> torch.Tensor:
    if (value.device.type == "hpu" and value.dtype == torch.float32 and value.is_contiguous() and self.weight is None):
        return self._vllm_gaudi_rms_norm(value, None)
    return self._vllm_gaudi_original_forward(value)


def _install_h3_vae_compiled_qk_rms_norm(decoder: nn.Module) -> int:
    """Compile the affine-free Q/K normalization without changing attention."""

    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        return 0
    norms: list[nn.RMSNorm] = []
    signature: tuple[tuple[int, ...], float] | None = None
    for block in blocks:
        attention = getattr(block, "attn", None)
        if attention is None:
            return 0
        for name in ("norm_q", "norm_k"):
            norm = getattr(attention, name, None)
            if not isinstance(norm, nn.RMSNorm) or norm.weight is not None or norm.eps is None:
                return 0
            normalized_shape = tuple(int(dimension) for dimension in norm.normalized_shape)
            current_signature = (normalized_shape, float(norm.eps))
            if signature is None:
                signature = current_signature
            elif current_signature != signature:
                return 0
            norms.append(norm)

    norms = [norm for norm in norms if not hasattr(norm, "_vllm_gaudi_original_forward")]
    if not norms or signature is None:
        return 0
    try:
        shared = _H3VAECompiledRMSNorm(*signature, output_dtype=torch.bfloat16)
    except Exception as exc:
        logger.warning_once(
            "MiniMax H3 video VAE RMSNorm compilation setup failed (%s: %s); using eager HPU operators",
            type(exc).__name__,
            exc,
        )
        return 0
    for norm in norms:
        norm._vllm_gaudi_original_forward = norm.forward
        norm._vllm_gaudi_rms_norm = shared
        norm.forward = MethodType(_h3_vae_qk_rms_norm_forward, norm)
    return len(norms)


def _h3_vae_rope(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(value.dtype)
    sin = sin.to(value.dtype)
    rotary_width = cos.shape[-1]
    if rotary_width < value.shape[-1]:
        rotated, passthrough = value[..., :rotary_width], value[..., rotary_width:]
    else:
        rotated, passthrough = value, None
    first, second = rotated.chunk(2, dim=-1)
    rotated = rotated * cos + torch.cat((-second, first), dim=-1) * sin
    return torch.cat((rotated, passthrough), dim=-1) if passthrough is not None else rotated


class _H3VAECompiledRoPE(_H3VAEVerifiedCompiled):
    """Compile the exact remote VAE RoPE and verify each tensor contract."""

    def __init__(self) -> None:
        super().__init__(_h3_vae_rope, "RoPE")


def _install_h3_vae_compiled_rope(decoder: nn.Module) -> bool:
    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        return False
    attention = getattr(blocks[0], "attn", None)
    forward = getattr(attention, "forward", None)
    function_globals = getattr(forward, "__globals__", None)
    if not isinstance(function_globals, dict):
        return False
    original = function_globals.get("apply_rotary_pos_emb")
    if not callable(original):
        return False
    if getattr(original, "_vllm_gaudi_compiled_rope", False):
        return False

    try:
        compiled = _H3VAECompiledRoPE()
    except Exception as exc:
        logger.warning_once(
            "MiniMax H3 video VAE RoPE compilation setup failed (%s: %s); using eager HPU operators",
            type(exc).__name__,
            exc,
        )
        return False

    def apply_rotary_pos_emb(
        value: torch.Tensor,
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if (value.device.type == "hpu" and value.dtype == torch.bfloat16 and value.is_contiguous()
                and len(rotary_pos_emb) == 2 and rotary_pos_emb[0].dim() == 4
                and all(tensor.device == value.device for tensor in rotary_pos_emb)):
            return compiled(value, *rotary_pos_emb)
        return original(value, rotary_pos_emb)

    apply_rotary_pos_emb._vllm_gaudi_compiled_rope = True
    function_globals["apply_rotary_pos_emb"] = apply_rotary_pos_emb
    return True


def _h3_vae_fused_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Run the qualified BF16 Habana SDPA path for VAE self-attention."""

    if any(tensor.device.type != "hpu" for tensor in (query, key, value)):
        raise RuntimeError("MiniMax H3 video VAE FusedSDPA requires HPU query, key, and value tensors")
    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    # FusedSDPA accepts the BSHD-to-BHSD transpose views directly.  Letting the
    # HPU operator consume those strides avoids three eager materializations
    # for every decoder block while preserving the exact fused result.
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    output = FusedSDPA.apply(
        query,
        key,
        value,
        None,
        0.0,
        False,
        None,
        "None",
        True,
    )
    # The qualified inference contract has finite Q/K/V and the fused softmax
    # remains finite for zero, extreme, random-latent, and end-to-end model
    # inputs.  Returning the transpose view also lets the following reshape
    # and output projection consume it without a separate nan_to_num pass.
    return output.transpose(1, 2)


def _h3_vae_attention(
    self: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pack_info: dict[str, Any],
) -> torch.Tensor:
    original = self._vllm_gaudi_original_perform_attention
    supported_pack_fields = {"cu_seqlens", "mask_mod", "block_sparse"}
    can_fuse = (not getattr(self, "_vllm_gaudi_fused_sdpa_disabled", False) and isinstance(pack_info, dict)
                and query.device.type == "hpu" and query.dtype == key.dtype == value.dtype == torch.bfloat16
                and query.ndim == key.ndim == value.ndim == 4 and query.shape == key.shape == value.shape
                and query.shape[2] == 32 and query.shape[3] == 64 and set(pack_info).issubset(supported_pack_fields)
                and not any(pack_info.get(name) is not None for name in ("cu_seqlens", "mask_mod", "block_sparse")))
    if not can_fuse:
        return original(query, key, value, pack_info)
    try:
        return _h3_vae_fused_sdpa(query, key, value)
    except Exception as exc:
        self._vllm_gaudi_fused_sdpa_disabled = True
        logger.warning_once(
            "MiniMax H3 video VAE FusedSDPA failed (%s: %s); using the checkpoint SDPA path",
            type(exc).__name__,
            exc,
        )
        return original(query, key, value, pack_info)


def _install_h3_vae_fused_sdpa(decoder: nn.Module) -> int:
    """Route only the qualified unmasked decoder attention contract."""

    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        return 0
    attentions: list[nn.Module] = []
    for block in blocks:
        attention = getattr(block, "attn", None)
        if (not isinstance(attention, nn.Module) or not callable(getattr(attention, "_perform_attention", None))
                or int(getattr(attention, "heads", 0)) != 32 or int(getattr(attention, "dim_head", 0)) != 64):
            return 0
        attentions.append(attention)

    attentions = [
        attention for attention in attentions if not hasattr(attention, "_vllm_gaudi_original_perform_attention")
    ]
    for attention in attentions:
        attention._vllm_gaudi_original_perform_attention = attention._perform_attention
        attention._perform_attention = MethodType(_h3_vae_attention, attention)
    return len(attentions)


def _materialize_h3_vae_decoder_linear_weights(decoder: nn.Module) -> int:
    """Persist the BF16 operands that HPU autocast otherwise rebuilds."""

    linears = tuple(module for module in decoder.modules() if isinstance(module, nn.Linear))
    if not linears:
        return 0
    unsupported = sorted({str(linear.weight.dtype) for linear in linears if linear.weight.dtype != torch.float32})
    if unsupported:
        logger.warning_once("MiniMax H3 video VAE decoder Linear dtype is unsupported: %s", unsupported)
        return 0
    for linear in linears:
        linear.to(dtype=torch.bfloat16)
    return len(linears)


def _install_h3_vae_weight_policy() -> None:
    """Apply the HPU weight policy before Omni snapshots staged CPU storage."""

    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    if getattr(vae_module, "_vllm_gaudi_h3_vae_weight_policy_installed", False):
        return
    original_install = vae_module.install_h3_vae_optimizations

    def install_optimizations(decoder: nn.Module, *, device: torch.device) -> bool:
        installed = original_install(decoder, device=device)
        if device.type != "hpu":
            return installed
        linear_count = (_materialize_h3_vae_decoder_linear_weights(decoder) if _h3_vae_persist_bf16_weights() else 0)
        swiglu_count = _install_h3_vae_compiled_swiglu(decoder) if _h3_vae_compile_swiglu() else 0
        qk_rms_norm_count = _install_h3_vae_compiled_qk_rms_norm(decoder) if _h3_vae_compile_qk_norm() else 0
        rope_installed = _install_h3_vae_compiled_rope(decoder) if _h3_vae_compile_rope() else False
        fused_sdpa_count = _install_h3_vae_fused_sdpa(decoder) if _h3_vae_fused_sdpa_enabled() else 0
        if linear_count:
            logger.info_once("MiniMax H3 video VAE persisted %d decoder Linear modules in BF16", linear_count)
        if swiglu_count:
            logger.info_once("MiniMax H3 video VAE compiled exact SwiGLU for %d decoder blocks", swiglu_count)
        if qk_rms_norm_count:
            logger.info_once("MiniMax H3 video VAE compiled exact RMSNorm for %d decoder Q/K norms", qk_rms_norm_count)
        if rope_installed:
            logger.info_once("MiniMax H3 video VAE compiled exact rotary embedding")
        if fused_sdpa_count:
            logger.info_once("MiniMax H3 video VAE enabled Habana FusedSDPA for %d decoder blocks", fused_sdpa_count)
        return (installed or bool(linear_count) or bool(swiglu_count) or bool(qk_rms_norm_count) or rope_installed
                or bool(fused_sdpa_count))

    vae_module.install_h3_vae_optimizations = install_optimizations
    vae_module._vllm_gaudi_h3_vae_weight_policy_installed = True


def _run_h3_vae_tile_tasks(
    self: nn.Module,
    tiles: list[torch.Tensor],
    tile_indices: list[int],
    forward_fn: Any,
    stack_tiling: bool,
    cls_agg: Any = None,
) -> list[torch.Tensor]:
    """Batch independent decoder tiles while preserving their output order."""

    original = self._vllm_gaudi_original_run_tile_tasks
    is_decode = getattr(forward_fn, "__self__", None) is self and getattr(forward_fn, "__name__", "") == "decode"
    batch_size = int(self._vllm_gaudi_decode_tile_batch_size)
    if not is_decode or batch_size <= 1 or not tile_indices:
        return original(tiles, tile_indices, forward_fn, stack_tiling, cls_agg)

    outputs: list[torch.Tensor] = []
    for offset in range(0, len(tile_indices), batch_size):
        indices = tile_indices[offset:offset + batch_size]
        sample_batch_size = int(tiles[indices[0]].shape[0])
        tile_batch = torch.cat([tiles[index] for index in indices], dim=0)
        output_batch = forward_fn(tile_batch)
        expected_batch = len(indices) * sample_batch_size
        if int(output_batch.shape[0]) != expected_batch:
            raise RuntimeError("MiniMax H3 VAE tile batch changed the leading dimension: "
                               f"expected {expected_batch}, got {int(output_batch.shape[0])}")
        outputs.extend(output_batch.unflatten(0, (len(indices), sample_batch_size)).unbind(dim=0))
        if cls_agg is not None:
            cls_agg.collect_stacked(len(indices), sample_batch_size)
    return outputs


def _install_h3_vae_decode_tile_batching(model: nn.Module, batch_size: int) -> bool:
    """Install bounded decode-only tile batching on a compatible remote VAE."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not callable(getattr(model, "_run_tile_tasks", None)) or not callable(getattr(model, "decode", None)):
        return False
    if not hasattr(model, "decoder_tiling") or not hasattr(model, "stack_tiling"):
        return False

    model._vllm_gaudi_decode_tile_batch_size = int(batch_size)
    if hasattr(model, "_vllm_gaudi_original_run_tile_tasks"):
        return True
    model._vllm_gaudi_original_run_tile_tasks = model._run_tile_tasks
    model._run_tile_tasks = MethodType(_run_h3_vae_tile_tasks, model)
    return True


def _h3_vae_blend_with_weights(
    a: torch.Tensor,
    b: torch.Tensor,
    blend_extent: int,
    dim: int,
    weights: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    weight_a, weight_b = weights
    slice_a = [slice(None)] * a.ndim
    slice_a[dim] = slice(-blend_extent, None)
    slice_b = [slice(None)] * b.ndim
    slice_b[dim] = slice(0, blend_extent)
    blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
    if blend_extent >= b.shape[dim]:
        return blended
    slice_b[dim] = slice(blend_extent, None)
    return torch.cat((blended, b[tuple(slice_b)]), dim=dim)


def _h3_vae_blend(self: nn.Module, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
    """Reuse the tiny HPU ramp tensors across spatial and temporal seams."""

    original = self._vllm_gaudi_original_blend
    if (a.device.type != "hpu" or b.device != a.device or b.dtype != a.dtype or a.ndim != b.ndim or a.ndim == 0):
        return original(a, b, blend_extent, dim)
    normalized_dim = dim % a.ndim
    extent = min(int(a.shape[normalized_dim]), int(b.shape[normalized_dim]), int(blend_extent))
    if extent <= 0:
        return original(a, b, blend_extent, dim)

    autocast_enabled = torch.is_autocast_enabled("hpu")
    autocast_dtype = torch.get_autocast_dtype("hpu") if autocast_enabled else None
    key = (a.ndim, normalized_dim, extent, a.device, a.dtype, autocast_enabled, autocast_dtype)
    weights = self._vllm_gaudi_blend_weights.get(key)
    if weights is None:
        positions = torch.arange(extent, device=b.device, dtype=b.dtype)
        weight_a = 1 - positions / extent
        weight_b = positions / extent
        shape = [1] * a.ndim
        shape[normalized_dim] = extent
        weights = (weight_a.view(shape), weight_b.view(shape))
        self._vllm_gaudi_blend_weights[key] = weights
    return _h3_vae_blend_with_weights(a, b, extent, normalized_dim, weights)


def _install_h3_vae_blend_weight_cache(model: nn.Module) -> bool:
    """Cache exact HPU blend ramps instead of rebuilding them at every seam."""

    if not callable(getattr(model, "blend", None)):
        return False
    if hasattr(model, "_vllm_gaudi_original_blend"):
        return True
    model._vllm_gaudi_original_blend = model.blend
    model._vllm_gaudi_blend_weights = {}
    model.blend = MethodType(_h3_vae_blend, model)
    return True


def install_h3_vae_patches() -> None:
    """Apply the HPU VAE policy through the normal Omni component loader."""

    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    _install_h3_vae_weight_policy()
    if getattr(MiniMaxH3VideoVAE, "_vllm_gaudi_tile_batch_init_installed", False):
        return
    original_init = MiniMaxH3VideoVAE.__init__

    def video_vae_init(self: nn.Module, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        device = kwargs.get("device")
        if not isinstance(device, torch.device) or device.type != "hpu":
            return
        batch_size = _h3_vae_tile_batch_size()
        if not _install_h3_vae_decode_tile_batching(self.model, batch_size):
            logger.warning_once("MiniMax H3 video VAE does not expose the expected tiled-decode contract")
            return
        if not _install_h3_vae_blend_weight_cache(self.model):
            logger.warning_once("MiniMax H3 video VAE does not expose the expected blend contract")
        logger.info_once("MiniMax H3 video VAE decode tile batch size is %d", batch_size)

    MiniMaxH3VideoVAE.__init__ = video_vae_init
    MiniMaxH3VideoVAE._vllm_gaudi_tile_batch_init_installed = True


__all__ = ["install_h3_vae_patches"]
