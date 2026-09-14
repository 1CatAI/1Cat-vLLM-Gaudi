# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gaudi implementation of the vLLM-Omni platform contract."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.kernel import IrOpPriorityConfig
from vllm.logger import init_logger

from vllm_gaudi.omni.compat import install_vllm_omni_compat
from vllm_gaudi.platform import HpuPlatform
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)

install_vllm_omni_compat()


class HPUOmniPlatform(OmniPlatform, HpuPlatform):
    """Single- and multi-process Omni runtime backed by Habana HPU APIs."""

    _omni_enum = OmniPlatformEnum.OOT
    dist_backend = "hccl"

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_ar_worker.GPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/deploy"

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
        allow_trtllm_default: bool = False,
    ) -> str:
        del head_size, allow_trtllm_default
        if selected_backend is not None and selected_backend.upper() not in {
            "TORCH_SDPA",
            "SDPA",
            "HPU_SDPA",
        }:
            raise ValueError(
                f"Diffusion attention backend {selected_backend!r} is unavailable on HPU; "
                "select TORCH_SDPA or omit the backend to use HPU_SDPA."
            )
        return "vllm_gaudi.omni.attention.HPUSDPABackend"

    @classmethod
    def supports_diffusion_dense_flash_attention(cls) -> bool:
        return False

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def supports_talker_mtp_graph_capture(cls) -> bool:
        return False

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("hpu")
        return torch.device("hpu", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.hpu.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        try:
            import habana_frameworks.torch.utils.experimental as htexp

            return str(htexp._get_device_type()).rsplit(".", 1)[-1]
        except Exception:
            return None

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        index = 0 if device.index is None else int(device.index)
        visible_modules = os.environ.get("HABANA_VISIBLE_MODULES", "").split(",")
        if visible_modules and visible_modules[0] and index < len(visible_modules):
            os.environ.setdefault("HLS_MODULE_ID", visible_modules[index])
        torch.hpu.set_device(index)

    @classmethod
    def synchronize(cls) -> None:
        torch.hpu.synchronize()

    @classmethod
    def empty_cache(cls) -> None:
        HpuPlatform.empty_cache()

    @classmethod
    def record_device_event(cls) -> Any | None:
        try:
            event = torch.hpu.Event()
            event.record()
            return event
        except Exception:
            logger.warning("Failed to record an HPU event for asynchronous output", exc_info=True)
            return None

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        del device
        free, _ = torch.hpu.mem_get_info()
        return int(free)

    @classmethod
    def get_device_memory(cls, device: torch.device | None = None) -> tuple[int, int]:
        del device
        free, total = torch.hpu.mem_get_info()
        return int(free), int(total)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        del device_id
        return cls.get_device_memory()[1]

    @classmethod
    def create_autocast_context(
        cls,
        *,
        device_type: str,
        dtype: torch.dtype,
        enabled: bool = True,
    ):
        if not enabled:
            return nullcontext()
        # H3's CUDA implementation asks for FP16 around the video VAE. Gaudi
        # autocast supports the same decoder path in BF16.
        if device_type == "hpu" and dtype == torch.float16:
            dtype = torch.bfloat16
        return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)

    @classmethod
    def get_default_ir_op_priority(cls, vllm_config: VllmConfig) -> IrOpPriorityConfig:
        del vllm_config
        return IrOpPriorityConfig.with_default(["native"])

    @classmethod
    def init_diffusion_worker_vllm_config(cls, vllm_config: Any) -> None:
        # H3 qualification starts in eager mode. The HPU plugin still applies
        # runtime-scale and parallel-compilation defaults used by native FP8.
        cls.set_compile_env_defaults()
        compilation_config = getattr(vllm_config, "compilation_config", None)
        if compilation_config is not None:
            try:
                from vllm.config.compilation import CompilationMode

                compilation_config.mode = CompilationMode.NONE
            except (AttributeError, ImportError):
                pass


__all__ = ["HPUOmniPlatform"]
