# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU execution policy for the MiniMax H3 video VAE."""

from __future__ import annotations

import os
import queue
import threading
from contextlib import suppress
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
_H3_VAE_COMPILE_BLOCKS_ENV = "VLLM_GAUDI_H3_VAE_COMPILE_BLOCKS"
_H3_VAE_FUSED_SDPA_ENV = "VLLM_GAUDI_H3_VAE_FUSED_SDPA"
_H3_VAE_ASYNC_D2H_ENV = "VLLM_GAUDI_H3_VAE_ASYNC_D2H"
_H3_VAE_TEMPORAL_BATCH_SIZE_ENV = "VLLM_GAUDI_H3_VAE_TEMPORAL_BATCH_SIZE"
_DEFAULT_H3_VAE_TILE_BATCH_SIZE = 4
_DEFAULT_H3_VAE_TEMPORAL_BATCH_SIZE = 2


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


def _h3_vae_compile_blocks() -> bool:
    """Enable HPU block-level compilation for the qualified H3 decoder."""

    return _boolean_environment(_H3_VAE_COMPILE_BLOCKS_ENV, "1")


def _h3_vae_fused_sdpa_enabled() -> bool:
    return _boolean_environment(_H3_VAE_FUSED_SDPA_ENV, "1")


def _h3_vae_async_d2h_enabled() -> bool:
    """Overlap normalized frame DMA with the next decoder tile."""

    return _boolean_environment(_H3_VAE_ASYNC_D2H_ENV, "1")


def _h3_vae_temporal_batch_size() -> int:
    """Number of equal-shaped temporal clips submitted in one VAE call.

    ``2`` is the only production-qualified value on Gaudi 2.  Larger groups
    change the reduction order of the tiled decoder and are deliberately
    rejected instead of silently trading output fidelity for a benchmark win.
    """

    raw = os.environ.get(
        _H3_VAE_TEMPORAL_BATCH_SIZE_ENV,
        str(_DEFAULT_H3_VAE_TEMPORAL_BATCH_SIZE),
    ).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_H3_VAE_TEMPORAL_BATCH_SIZE_ENV} must be 1 or 2, got {raw!r}"
        ) from exc
    if value not in (1, 2):
        raise ValueError(f"{_H3_VAE_TEMPORAL_BATCH_SIZE_ENV} must be 1 or 2, got {raw!r}")
    return value


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
        # The complete TransformerBlock compiler may inline this helper.  Do
        # not enter the Python verification lock from inside a Dynamo graph:
        # the reference expression is traceable and the outer block performs
        # the one-time graph verification at its own boundary.
        if torch.compiler.is_compiling():
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


def _h3_vae_compiled_block_forward(
    self: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Dispatch a compiled H3 block and permanently fall back on failure.

    HPU graph compilation is intentionally best effort here.  A decoder block
    can encounter a shape or operator combination outside the qualified
    single-card contract (for example, a masked/packed request).  Retrying the
    same failing compiled graph on every block would turn a recoverable HPU
    capability miss into a large latency penalty, so the first failure disables
    only that block and runs the original implementation thereafter.
    """

    state = self._vllm_gaudi_block_compile_state
    if state["disabled"]:
        return self._vllm_gaudi_original_block_forward(*args, **kwargs)
    hidden_states = args[0] if args else kwargs.get("hidden_states")
    # The graph is qualified only for the HPU, unmasked decoder contract.  In
    # particular, do not ask the HPU backend to trace a CPU-staged utility call
    # or a packed/causal request; those cases retain the checkpoint block with
    # no failed compile attempt on the critical path.
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.device.type != "hpu":
        # CPU construction/staging is expected before the first HPU request;
        # keep the compiled graph eligible for the later device transition.
        return self._vllm_gaudi_original_block_forward(*args, **kwargs)
    pack_info = kwargs.get("pack_info")
    if pack_info is None and len(args) >= 3:
        pack_info = args[2]
    if isinstance(pack_info, dict) and any(value is not None for value in pack_info.values()):
        # A masked/packed request is outside the qualified graph contract, but
        # it must not poison the unmasked graph used by a subsequent request.
        return self._vllm_gaudi_original_block_forward(*args, **kwargs)
    try:
        return self._vllm_gaudi_compiled_block_forward(*args, **kwargs)
    except Exception as exc:
        state["disabled"] = True
        logger.warning_once(
            "MiniMax H3 VAE compiled TransformerBlock %d failed (%s: %s); "
            "using its original HPU block",
            state["index"],
            type(exc).__name__,
            exc,
        )
        return self._vllm_gaudi_original_block_forward(*args, **kwargs)


def _install_h3_vae_compiled_blocks(decoder: nn.Module) -> int:
    """Compile each complete decoder block after HPU attention is installed.

    Compiling the complete block lets the HPU backend schedule the two linear
    projections, RMSNorm/residual pointwise work, and the fused SDPA boundary as
    one stable shape-specialized region.  The qualified unmasked H3 decode path
    is fully traceable, so ``fullgraph=True`` avoids a graph break at every
    block.  The original bound method remains attached for an automatic
    per-block fallback when a request carries an unsupported mask or layout.
    """

    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks or not hasattr(torch, "compile"):
        return 0
    with suppress(Exception):
        # The H3 decode contract uses a fixed tile batch and sequence shape.  A
        # larger cache prevents unrelated block signatures from evicting one
        # another when the first request compiles all 36 blocks.
        torch._dynamo.config.cache_size_limit = max(128, len(blocks) * 4)

    installed = 0
    for index, block in enumerate(blocks):
        if not isinstance(block, nn.Module) or hasattr(block, "_vllm_gaudi_compiled_block_forward"):
            continue
        original = block.forward
        try:
            compiled = torch.compile(
                original,
                backend="hpu_backend",
                fullgraph=True,
                dynamic=False,
            )
        except Exception as exc:
            logger.warning_once(
                "MiniMax H3 VAE TransformerBlock %d compilation setup failed "
                "(%s: %s); using its original HPU block",
                index,
                type(exc).__name__,
                exc,
            )
            continue
        block._vllm_gaudi_original_block_forward = original
        block._vllm_gaudi_compiled_block_forward = compiled
        block._vllm_gaudi_block_compile_state = {"index": index, "disabled": False}
        block.forward = MethodType(_h3_vae_compiled_block_forward, block)
        installed += 1
    return installed


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
        block_compile_count = (
            _install_h3_vae_compiled_blocks(decoder)
            if fused_sdpa_count and _h3_vae_compile_blocks()
            else 0
        )
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
        if block_compile_count:
            logger.info_once(
                "MiniMax H3 video VAE compiled %d complete decoder TransformerBlocks with hpu_backend",
                block_compile_count,
            )
        return (installed or bool(linear_count) or bool(swiglu_count) or bool(qk_rms_norm_count) or rope_installed
                or bool(fused_sdpa_count) or bool(block_compile_count))

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


def _h3_vae_temporal_batch_eligible(
    model: nn.Module,
    latent: torch.Tensor,
    group_size: int,
) -> bool:
    """Check the single-card shape for which temporal batching was qualified."""

    return (
        group_size == 2
        and isinstance(latent, torch.Tensor)
        and latent.device.type == "hpu"
        and latent.ndim == 5
        and int(latent.shape[0]) == 1
        and bool(getattr(model, "use_3d_conv", False))
        and bool(getattr(model, "decoder_tiling", False))
        and not bool(getattr(model, "parallel_tiling", False))
        and int(getattr(model, "_vllm_gaudi_decode_tile_batch_size", 0)) >= 28
    )


def _h3_decode_temporal_chunks_batched(
    model: nn.Module,
    latent: torch.Tensor,
    callback: Any,
    *,
    group_size: int,
) -> torch.Tensor:
    """Batch adjacent equal-shaped H3 temporal clips without changing seams.

    The released loop has seven temporal clips for the qualified 5-second
    shape.  Each clip is independent until its overlap is blended into the
    preceding output, so two clips can share one tiled decoder submission.  We
    unbatch before applying the boundary rules and restore the original tile
    batch size in a ``finally`` block; this keeps the exact causal/isolated
    frame semantics and leaves subsequent requests on the normal geometry.
    """

    if latent.ndim != 5 or not bool(getattr(model, "use_3d_conv", False)):
        raise ValueError("MiniMax H3 temporal chunk decode requires a rank-5 3D latent")

    token_drop = int(model.token_drop)
    chunk_size = int(model.tokens_chunk_size)
    overlap_tokens = int(model.token_overlap)
    ratio_t = int(model.vae_ratio_t)
    pre_padding = int(model.frame_pre_padding)

    isolated_first = bool(model.isolated_first_frame and pre_padding == 0)
    isolated_last = bool(model.isolated_last_frame)
    z_head = latent[:, :, :1] if isolated_first else None
    z_tail = latent[:, :, -1:] if isolated_last else None
    start = int(isolated_first)
    stop = int(latent.shape[2]) - int(isolated_last)
    latent = latent[:, :, start:stop]

    pseudo_tokens = int(latent.shape[2]) + token_drop
    pad_tokens = (-pseudo_tokens) % chunk_size
    if pad_tokens:
        latent = torch.cat(
            (latent, latent[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)),
            dim=2,
        )
    num_chunks = (pseudo_tokens + pad_tokens) // chunk_size - int(token_drop > 0)
    if num_chunks <= 0:
        raise ValueError("MiniMax H3 temporal chunk plan is empty")

    _total_frames, _pad_frames, output_frames = model._decode_temporal_output_frame_plan(
        latent,
        z_head,
        z_tail,
        num_chunks,
        pad_tokens,
    )
    del _total_frames, _pad_frames
    output_frames = int(output_frames)
    if output_frames <= 0:
        raise ValueError("MiniMax H3 temporal chunk plan has no output frames")

    collected: list[torch.Tensor] = []
    overlap: torch.Tensor | None = None
    written = 0
    chunk_frames = chunk_size * ratio_t
    split_count = int(token_drop > 0) + 1

    def emit(part: torch.Tensor) -> None:
        nonlocal written
        frames = int(part.shape[2])
        if frames <= 0 or written >= output_frames:
            return
        frames = min(frames, output_frames - written)
        part = part[:, :, :frames]
        if callback is None:
            collected.append(part)
        else:
            callback(part)
        written += frames

    def run_group(items: list[tuple[torch.Tensor, int]]) -> torch.Tensor:
        original_batch = int(getattr(model, "_vllm_gaudi_decode_tile_batch_size", 1))
        effective_batch = max(1, original_batch // len(items))
        model._vllm_gaudi_decode_tile_batch_size = effective_batch
        try:
            value = model._adaptive_decode(torch.cat([item[0] for item in items], dim=0))
        finally:
            model._vllm_gaudi_decode_tile_batch_size = original_batch
        expected = (len(items), int(latent.shape[0]))
        if int(value.shape[0]) != expected[0] * expected[1]:
            raise RuntimeError(
                "MiniMax H3 temporal batch changed the decoder leading dimension: "
                f"expected {expected[0] * expected[1]}, got {int(value.shape[0])}"
            )
        return value.unflatten(0, expected)

    pending: list[tuple[torch.Tensor, int]] = []

    def flush_group() -> None:
        nonlocal overlap, pending
        if not pending:
            return
        current = pending
        pending = []
        decoded_group = run_group(current)
        for item_index, decoded in zip(
            (item[1] for item in current),
            decoded_group.unbind(dim=0),
            strict=True,
        ):
            if item_index == 0 and z_head is not None:
                emit(decoded[:, :, ratio_t - 1 : ratio_t])
                decoded = decoded[:, :, ratio_t:]

            decoded_tail = None
            if item_index == num_chunks - 1 and z_tail is not None:
                decoded_tail = decoded[:, :, -1:]
                decoded = decoded[:, :, :-ratio_t]

            for split in range(split_count):
                begin = split * chunk_frames
                end = min(begin + chunk_frames, int(decoded.shape[2]))
                part = decoded[:, :, begin:end]
                part = part[:, :, pre_padding:]
                if split == 0:
                    if overlap is not None:
                        part = model.blend(
                            overlap,
                            part,
                            int(model.frame_overlap),
                            dim=-3,
                        )
                        overlap = None
                    emit(part)
                else:
                    overlap = part.contiguous()

            if item_index == num_chunks - 1:
                if overlap is not None:
                    emit(overlap)
                    overlap = None
                if decoded_tail is not None:
                    emit(decoded_tail)

    for index in range(num_chunks):
        begin = index * chunk_size
        end = begin + chunk_size + overlap_tokens
        clip = latent[:, :, begin:end]
        if index == 0 and z_head is not None:
            clip = torch.cat((z_head, clip), dim=2)
        if index == num_chunks - 1 and z_tail is not None:
            clip = torch.cat((clip, z_tail), dim=2)
        if pending and (
            int(pending[-1][0].shape[2]) != int(clip.shape[2])
            or len(pending) >= group_size
        ):
            flush_group()
        pending.append((clip, index))
    flush_group()

    if written != output_frames:
        raise RuntimeError(f"MiniMax H3 temporal decode emitted {written}/{output_frames} frames")
    if callback is not None:
        return latent.new_empty((0,))
    return torch.cat(collected, dim=2) if collected else latent.new_empty((0,))


def _install_h3_vae_temporal_batching() -> bool:
    """Route the H3 chunk coordinator through the qualified pair-batch loop."""

    try:
        from vllm_omni.diffusion.models.minimax_h3 import chunked_decode
    except ImportError:
        return False
    if getattr(chunked_decode, "_vllm_gaudi_temporal_batching_installed", False):
        return True
    original = getattr(chunked_decode, "decode_temporal_chunks", None)
    if not callable(original):
        return False

    def decode_temporal_chunks(
        model: nn.Module,
        latent: torch.Tensor,
        callback: Any,
    ) -> torch.Tensor:
        # Parse the opt-in only for the HPU path.  CPU-only installs retain the
        # upstream function even if an HPU-specific variable is malformed.
        if not isinstance(latent, torch.Tensor) or latent.device.type != "hpu":
            return original(model, latent, callback)
        group_size = _h3_vae_temporal_batch_size()
        if group_size <= 1 or not _h3_vae_temporal_batch_eligible(model, latent, group_size):
            return original(model, latent, callback)
        logger.info_once(
            "MiniMax H3 VAE batching adjacent temporal clips in pairs with spatial tile budget %d",
            int(getattr(model, "_vllm_gaudi_decode_tile_batch_size", 0)),
        )
        return _h3_decode_temporal_chunks_batched(
            model,
            latent,
            callback,
            group_size=group_size,
        )

    chunked_decode._vllm_gaudi_original_decode_temporal_chunks = original
    chunked_decode.decode_temporal_chunks = decode_temporal_chunks
    chunked_decode._vllm_gaudi_temporal_batching_installed = True
    return True


class _H3AsyncVideoTransfer:
    """Queue prepared HPU frames while the decoder advances to the next tile.

    ``Tensor.to("cpu", non_blocking=True)`` submits the device-to-host copy,
    but converting the returned tensor to NumPy is still a synchronization
    point.  Keeping that conversion on a host worker lets the HPU submit the
    next decoder graph while the previous tile is copied and handed to the
    mux worker.  The queue is bounded so a slow encoder cannot grow host
    memory without limit.
    """

    _DONE = object()

    def __init__(self, encoder: Any, *, max_pending: int) -> None:
        self._encoder = encoder
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_pending)
        self._error: BaseException | None = None
        self._error_event = threading.Event()
        self._state_lock = threading.Lock()
        self._copy_stream: Any | None = None
        # ``ChunkedMP4Encoder.push`` transfers the NumPy object to its mux
        # queue and returns before that worker has consumed it. Keep the CPU
        # destination tensors alive until encoder.finish()/abort() joins that
        # worker; otherwise the HPU allocator can recycle a destination while
        # PyAV is still reading it. The H3 5-second contract has eight chunks,
        # so this is a bounded, sub-gigabyte ownership window.
        self._owned_cpu_tensors: list[torch.Tensor] = []
        # Keep source views alive until the transfer worker has consumed them.
        # ``record_stream`` informs the allocator, while this reference also
        # covers remote VAE implementations that return short-lived tile views.
        self._owned_device_tensors: list[torch.Tensor] = []
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="minimax-h3-vae-d2h",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is self._DONE:
                        return
                    if not isinstance(item, tuple) or len(item) != 2:
                        raise TypeError("MiniMax H3 asynchronous transfer received an invalid work item")
                    host_frames, ready_event = item
                    if not isinstance(host_frames, torch.Tensor):
                        raise TypeError("MiniMax H3 asynchronous transfer received a non-tensor chunk")
                    if ready_event is not None:
                        # The copy lives on a dedicated HPU stream. A plain
                        # ``numpy()`` call from this host thread does not
                        # reliably wait for that stream on all Gaudi builds.
                        ready_event.synchronize()
                    # PyAV may retain the ndarray backing a VideoFrame beyond
                    # the generator iteration. Give the mux worker an owned
                    # snapshot instead of exposing the non-blocking DMA
                    # destination to that lifetime. This prevents allocator
                    # reuse from changing pixels; threaded H.264 encoders can
                    # still vary encoded bytes across runs, so quality rather
                    # than bit identity is the acceptance criterion.
                    self._encoder.push(host_frames.numpy().copy())
                finally:
                    self._queue.task_done()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            self._error_event.set()
            # Let the producer finish its cleanup without leaking a daemon
            # thread.  Discard queued chunks after an encoder/transfer error;
            # the request will be failed by ``finish`` or ``abort``.
            while True:
                item = self._queue.get()
                try:
                    if item is self._DONE:
                        return
                finally:
                    self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def push(self, frames: torch.Tensor) -> None:
        if not isinstance(frames, torch.Tensor):
            raise TypeError("MiniMax H3 asynchronous transfer expects a torch.Tensor")
        self._raise_if_failed()
        ready_event = None
        if frames.device.type == "hpu":
            if self._copy_stream is None:
                self._copy_stream = torch.hpu.Stream()
            compute_stream = torch.hpu.current_stream()
            ready = compute_stream.record_event()
            with torch.hpu.stream(self._copy_stream):
                self._copy_stream.wait_event(ready)
                host_frames = frames.to(device="cpu", non_blocking=True)
                torch.hpu.record_stream(frames, self._copy_stream)
                ready_event = self._copy_stream.record_event()
        else:
            host_frames = frames.to(device="cpu", non_blocking=True)
        if frames.device.type == "hpu":
            self._owned_device_tensors.append(frames)
        self._owned_cpu_tensors.append(host_frames)
        while True:
            self._raise_if_failed()
            try:
                self._queue.put((host_frames, ready_event), timeout=0.05)
                break
            except queue.Full:
                if self._error_event.is_set():
                    self._raise_if_failed()
        self._raise_if_failed()

    def release(self) -> None:
        """Release DMA source and destination references after mux joins."""

        self._owned_cpu_tensors.clear()
        self._owned_device_tensors.clear()

    def finish(self) -> None:
        with self._state_lock:
            if self._closed:
                self._raise_if_failed()
                return
        self._queue.join()
        self._raise_if_failed()
        self._queue.put(self._DONE)
        self._queue.join()
        self._thread.join()
        with self._state_lock:
            self._closed = True
        self._raise_if_failed()

    def abort(self) -> None:
        # ``ChunkedMP4Encoder.abort`` is called by the owner before this
        # method, which unblocks a worker currently waiting in encoder.push.
        with self._state_lock:
            if self._closed and not self._thread.is_alive():
                return
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()
        try:
            self._queue.put(self._DONE, timeout=1.0)
        except queue.Full:
            # A worker that is still draining will eventually observe the
            # sentinel; no request should remain blocked on cleanup.
            self._queue.put(self._DONE)
        self._queue.join()
        self._thread.join()
        with self._state_lock:
            self._closed = True


def _h3_decode_to_mp4_async(
    pipeline: nn.Module,
    original_decode_to_mp4: Any,
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    *,
    height: int,
    width: int,
    max_pending: int,
    batch_frames: int,
    video_codec_options: dict[str, str] | None,
) -> bytes:
    """Run the released H3 MP4 path with overlapped HPU frame transfer."""

    # Keep the exact upstream implementation available for CPU, distributed,
    # or explicitly disabled requests.  The asynchronous contract is only
    # qualified for one HPU output owner, which is the single-card target.
    video_vae = getattr(pipeline, "video_vae", None)
    if (
        getattr(getattr(pipeline, "device", None), "type", None) != "hpu"
        or not _h3_vae_async_d2h_enabled()
        or video_vae is None
        or int(getattr(video_vae, "parallel_size", 1)) != 1
    ):
        return original_decode_to_mp4(
            pipeline,
            video_latent,
            audio_latent,
            height=height,
            width=width,
            max_pending=max_pending,
            batch_frames=batch_frames,
            video_codec_options=video_codec_options,
        )

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_AUDIO_SAMPLE_RATE,
        MINIMAX_H3_FPS,
        _prepare_minimax_h3_video_output,
    )
    from vllm_omni.diffusion.utils.media_utils import ChunkedMP4Encoder
    from vllm_omni.platforms import current_omni_platform

    if batch_frames <= 0:
        raise ValueError("batch_frames must be positive")

    with pipeline._component_on_device(pipeline.audio_vae):
        audio = pipeline.audio_vae.decode_latent(audio_latent)
    audio_np = audio.detach().float().cpu().numpy()
    if audio_np.ndim == 3 and audio_np.shape[0] == 1:
        audio_np = audio_np[0]
    encoder = ChunkedMP4Encoder(
        width=width,
        height=height,
        fps=MINIMAX_H3_FPS,
        audio_waveform=audio_np,
        audio_sample_rate=MINIMAX_H3_AUDIO_SAMPLE_RATE,
        max_pending=max_pending,
        video_codec_options=video_codec_options,
    )
    transfer: _H3AsyncVideoTransfer | None = None
    pending_chunks: list[torch.Tensor] = []
    pending_frames = 0

    def flush_pending() -> None:
        nonlocal pending_frames
        if not pending_chunks:
            return
        batched = torch.cat(pending_chunks, dim=1)
        # Keep the HPU tensor alive until the DMA is queued; the worker owns
        # the CPU destination and subsequently synchronizes at ``numpy``.
        assert transfer is not None
        transfer.push(batched[0])
        pending_chunks.clear()
        pending_frames = 0

    def on_chunk(frames: torch.Tensor) -> None:
        nonlocal pending_frames
        prepared = _prepare_minimax_h3_video_output(frames[..., :height, :width])
        if prepared.shape[0] != 1:
            raise ValueError("MiniMax H3 chunked MP4 encoding currently expects one output per decoder")
        pending_chunks.append(prepared)
        pending_frames += int(prepared.shape[1])
        if pending_frames >= batch_frames:
            flush_pending()

    transfer = _H3AsyncVideoTransfer(encoder, max_pending=max_pending)
    try:
        with pipeline._component_on_device(video_vae), current_omni_platform.create_autocast_context(
            device_type=pipeline.device.type,
            dtype=torch.float16,
            enabled=True,
        ):
            video_vae.decode_with_chunks(video_latent, on_chunk=on_chunk)
        flush_pending()
        transfer.finish()
        try:
            return encoder.finish()
        finally:
            # The encoder owns the NumPy views until its worker joins. Only
            # then may PyTorch recycle the non-blocking CPU destinations.
            transfer.release()
    except BaseException:
        # Stop the mux worker first: it may be the operation currently
        # blocking the transfer worker's encoder.push call.
        try:
            encoder.abort()
        finally:
            assert transfer is not None
            transfer.abort()
            transfer.release()
        raise


def _install_h3_vae_async_decode_to_mp4() -> bool:
    """Install the normal H3 pipeline entry point for asynchronous D2H."""

    try:
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3
    except ImportError:
        return False
    pipeline_class = pipeline_minimax_h3.MiniMaxH3Pipeline
    if getattr(pipeline_class, "_vllm_gaudi_async_d2h_installed", False):
        return True
    original = pipeline_class.decode_to_mp4

    def decode_to_mp4(
        self: nn.Module,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        *,
        height: int,
        width: int,
        max_pending: int = 2,
        batch_frames: int = 17,
        video_codec_options: dict[str, str] | None = None,
    ) -> bytes:
        return _h3_decode_to_mp4_async(
            self,
            original,
            video_latent,
            audio_latent,
            height=int(height),
            width=int(width),
            max_pending=int(max_pending),
            batch_frames=int(batch_frames),
            video_codec_options=video_codec_options,
        )

    pipeline_class.decode_to_mp4 = decode_to_mp4
    pipeline_class._vllm_gaudi_async_d2h_installed = True
    return True


def install_h3_vae_patches() -> None:
    """Apply the HPU VAE policy through the normal Omni component loader."""

    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    _install_h3_vae_weight_policy()
    _install_h3_vae_temporal_batching()
    _install_h3_vae_async_decode_to_mp4()
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
