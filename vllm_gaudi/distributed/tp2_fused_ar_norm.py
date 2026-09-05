# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental TP2 all-reduce, residual-add, and RMSNorm fusion."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce

from vllm_gaudi.extension.kernels import rms_norm
from vllm_gaudi.extension.runtime import get_config

_RUNTIME_ATTR = "_vllm_gaudi_tp2_fused_ar_norm_runtime"
_BRIDGE_MODULE = "tp2_fused_ar_norm_bridge"
# The fused boundary wins through the 16-token decode bucket on Gaudi2, while
# the stock compiled path is faster for the 32-token bucket.
_MAX_FUSED_DECODE_TOKENS = 20
_library = torch.library.Library("vllm_gaudi", "FRAGMENT")
_library.define("tp2_allreduce_residual_rms_norm(Tensor partial, Tensor residual, "
                "Tensor weight, float epsilon) -> (Tensor, Tensor)")
_library.define("tp2_allreduce_residual_rms_norm_out(Tensor partial, Tensor residual, "
                "Tensor weight, Tensor(a!) reduced, Tensor(b!) normalized, "
                "Tensor(c!) residual_out, Tensor(d!) inverse_rms, float epsilon) -> ()")


def _exceeds_fused_decode_token_limit(partial: torch.Tensor, weight: torch.Tensor) -> bool:
    return partial.numel() // weight.numel() > _MAX_FUSED_DECODE_TOKENS


def _load_bridge(path: Path):
    loaded = sys.modules.get(_BRIDGE_MODULE)
    if loaded is not None:
        loaded_path = Path(loaded.__file__).resolve()
        if loaded_path != path:
            raise RuntimeError(f"{_BRIDGE_MODULE} is already loaded from {loaded_path}, not {path}")
        return loaded

    spec = importlib.util.spec_from_file_location(_BRIDGE_MODULE, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load TP2 fused bridge: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def initialize_tp2_fused_ar_norm_runtime() -> None:
    """Load the native bridge and bind it to the TP process group."""
    if getattr(torch, _RUNTIME_ATTR, None) is not None:
        return
    if not dist.is_initialized():
        raise RuntimeError("TP2 fused all-reduce requires initialized torch.distributed")

    tp_group = get_tp_group().device_group
    tp_size = dist.get_world_size(group=tp_group)
    if tp_size != 2 or dist.get_world_size() != 2:
        raise RuntimeError("TP2 fused all-reduce currently requires a two-rank, TP-only process world")
    required_environment = {
        "PT_HPU_LAZY_MODE": "0",
        "PT_HPU_ENABLE_LAZY_COLLECTIVES": "1",
        "PT_HPU_EAGER_PIPELINE_ENABLE": "1",
        "PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE": "1",
    }
    invalid = [
        f"{name}={os.environ.get(name)!r} (expected {expected!r})" for name, expected in required_environment.items()
        if os.environ.get(name) != expected
    ]
    if invalid:
        raise RuntimeError("TP2 fused all-reduce requires eager current-stream execution: " + ", ".join(invalid))

    bridge_value = os.environ.get("VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE")
    if not bridge_value:
        raise RuntimeError("VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE must point to the native bridge")
    bridge_path = Path(bridge_value).expanduser().resolve()
    if not bridge_path.is_file():
        raise RuntimeError(f"TP2 fused bridge does not exist: {bridge_path}")

    # Ensure ProcessGroupEagerHCCL owns an initialized communicator before the
    # bridge requests its current-stream handle.
    probe = torch.ones(128, dtype=torch.bfloat16, device="hpu")
    dist.all_reduce(probe, group=tp_group)
    torch.hpu.synchronize()
    if not torch.equal(probe.cpu(), torch.full((128, ), 2, dtype=torch.bfloat16)):
        raise RuntimeError("TP2 HCCL process-group initialization failed")

    bridge = _load_bridge(bridge_path)
    backend = tp_group._get_backend(torch.device("hpu"))
    communicator_id = bridge.communicator_id(backend)
    setattr(torch, _RUNTIME_ATTR, (bridge, backend, communicator_id))


def _resolve_runtime():
    runtime = getattr(torch, _RUNTIME_ATTR, None)
    if runtime is None:
        raise RuntimeError("TP2 fused all-reduce runtime is not initialized")
    return runtime


def _tp2_fused_ar_norm_impl(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    bridge, backend = _resolve_runtime()[:2]
    reduced = torch.empty_like(partial)
    normalized = torch.empty_like(partial)
    residual_out = torch.empty_like(partial)
    inverse_rms = torch.empty((*partial.shape[:-1], 1), dtype=torch.float32, device=partial.device)
    return bridge.allreduce_residual_rms_norm_current_stream(
        backend,
        partial,
        residual,
        weight,
        reduced,
        normalized,
        residual_out,
        inverse_rms,
        epsilon,
    )


def _tp2_fused_ar_norm_fake(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del residual, weight, epsilon
    return torch.empty_like(partial), torch.empty_like(partial)


def _tp2_fused_ar_norm_out_impl(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    reduced: torch.Tensor,
    normalized: torch.Tensor,
    residual_out: torch.Tensor,
    inverse_rms: torch.Tensor,
    epsilon: float,
) -> None:
    bridge, backend = _resolve_runtime()[:2]
    bridge.allreduce_residual_rms_norm_current_stream(
        backend,
        partial,
        residual,
        weight,
        reduced,
        normalized,
        residual_out,
        inverse_rms,
        epsilon,
    )


def _tp2_fused_ar_norm_out_fake(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    reduced: torch.Tensor,
    normalized: torch.Tensor,
    residual_out: torch.Tensor,
    inverse_rms: torch.Tensor,
    epsilon: float,
) -> None:
    del partial, residual, weight, reduced, normalized, residual_out, inverse_rms, epsilon


_library.impl(
    "tp2_allreduce_residual_rms_norm",
    _tp2_fused_ar_norm_impl,
    dispatch_key="HPU",
)
_library._register_fake(
    "tp2_allreduce_residual_rms_norm",
    _tp2_fused_ar_norm_fake,
)
_library.impl(
    "tp2_allreduce_residual_rms_norm_out",
    _tp2_fused_ar_norm_out_impl,
    dispatch_key="HPU",
)
_library._register_fake(
    "tp2_allreduce_residual_rms_norm_out",
    _tp2_fused_ar_norm_out_fake,
)


def _rejection_reason(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    is_prompt: bool,
    max_bytes: int,
) -> str | None:
    if is_prompt:
        return "prefill"
    if partial.device.type != "hpu":
        return "non-HPU input"
    if partial.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        return "activation dtype"
    if weight.dtype != torch.bfloat16:
        return "weight dtype"
    if partial.dim() < 2 or partial.numel() == 0:
        return "activation shape"
    if partial.shape != residual.shape:
        return "residual shape"
    if weight.dim() != 1 or partial.shape[-1] != weight.numel():
        return "weight shape"
    if _exceeds_fused_decode_token_limit(partial, weight):
        return "decode token count"
    if not partial.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
        return "non-contiguous tensor"
    if max_bytes <= 0 or partial.numel() * partial.element_size() > max_bytes:
        return "payload size"
    return None


def _fallback(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduced = tensor_model_parallel_all_reduce(partial)
    residual_out = residual + reduced.reshape(residual.shape)
    fused_rms_norm = rms_norm()
    if fused_rms_norm is None:
        variance = residual_out.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        normalized = residual_out * torch.rsqrt(variance + epsilon).to(residual_out.dtype)
        return normalized * weight, residual_out
    normalized = fused_rms_norm.apply(residual_out, weight, epsilon)
    return normalized.reshape(partial.shape), residual_out


def tp2_allreduce_residual_rms_norm(
    partial: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    *,
    is_prompt: bool,
    allow_fused: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the fused decode path, or preserve stock HCCL semantics."""
    if not allow_fused:
        return _fallback(partial, residual, weight, epsilon)
    max_bytes = get_config().tp2_fused_ar_norm_max_bytes
    reason = _rejection_reason(
        partial,
        residual,
        weight,
        is_prompt=is_prompt,
        max_bytes=max_bytes,
    )
    if reason is not None:
        return _fallback(partial, residual, weight, epsilon)
    communicator_id = _resolve_runtime()[2]
    packed, _inverse_rms = torch.ops.hccl.tp2_allreduce_residual_rms_norm(
        partial,
        residual,
        weight,
        epsilon,
        communicator_id,
    )
    return packed[0], packed[1]
