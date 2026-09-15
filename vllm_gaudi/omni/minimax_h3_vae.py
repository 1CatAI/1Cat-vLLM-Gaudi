# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU execution policy for the MiniMax H3 video VAE."""

from __future__ import annotations

import os
from types import MethodType
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)

_H3_VAE_TILE_BATCH_SIZE_ENV = "VLLM_GAUDI_H3_VAE_TILE_BATCH_SIZE"
_H3_VAE_PERSIST_BF16_WEIGHTS_ENV = "VLLM_GAUDI_H3_VAE_PERSIST_BF16_WEIGHTS"
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
        if linear_count:
            logger.info_once("MiniMax H3 video VAE persisted %d decoder Linear modules in BF16", linear_count)
        return installed or bool(linear_count)

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
        logger.info_once("MiniMax H3 video VAE decode tile batch size is %d", batch_size)

    MiniMaxH3VideoVAE.__init__ = video_vae_init
    MiniMaxH3VideoVAE._vllm_gaudi_tile_batch_init_installed = True


__all__ = ["install_h3_vae_patches"]
