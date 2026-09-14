# SPDX-License-Identifier: Apache-2.0
"""Keep grouped query rows together without expanding the shared K/V operand."""
from pathlib import Path

import torch

import vllm_gaudi.envs as gaudi_envs


_loaded_extension: Path | None = None


def load_native_gqa_matmul() -> bool:
    """Load the opt-in native descriptor before attention graph capture."""
    global _loaded_extension
    if not gaudi_envs.VLLM_HPU_GQA_NATIVE_MATMUL:
        return False
    configured_path = gaudi_envs.VLLM_HPU_GQA_NATIVE_MATMUL_EXTENSION
    if not configured_path:
        raise RuntimeError("VLLM_HPU_GQA_NATIVE_MATMUL_EXTENSION must identify the built native GQA extension")
    extension = Path(configured_path).expanduser().resolve()
    if not extension.is_file():
        raise RuntimeError(f"Native GQA extension does not exist: {extension}")
    if _loaded_extension is not None:
        if extension != _loaded_extension:
            raise RuntimeError(f"Native GQA was already loaded from {_loaded_extension}, cannot reload {extension}")
        return True
    torch.ops.load_library(str(extension))
    try:
        _ = torch.ops.custom_op.hpu_gqa_matmul
    except AttributeError as exc:
        raise RuntimeError("The configured extension does not register custom_op::hpu_gqa_matmul") from exc
    _loaded_extension = extension
    return True


def native_gqa_matmul(lhs: torch.Tensor, rhs: torch.Tensor, *, transpose_rhs: bool = False):
    """Issue a GQA batch GEMM without materializing the shared K/V heads.

    The first two dimensions identify cache pages and KV heads.  The third
    dimension contains the query rows which share each KV head.  Keeping that
    group as the M dimension lets Synapse express QK's transpose in the MME
    descriptor instead of inserting a physical key-layout copy.
    """
    if (lhs.ndim != 5 or rhs.ndim != 5 or lhs.shape[:2] != rhs.shape[:2] or lhs.shape[1] not in (2, 4)
            or lhs.shape[2] != 6 or lhs.shape[-2] != 1 or rhs.shape[2] != 1
            or lhs.dtype != torch.bfloat16 or rhs.dtype != torch.bfloat16 or lhs.device != rhs.device
            or lhs.shape[-1] != rhs.shape[-1 if transpose_rhs else -2]):
        raise RuntimeError("Native GQA requires matching Qwen BF16 compact query and cache batches")
    return torch.ops.custom_op.hpu_gqa_matmul(lhs.squeeze(-2), rhs.squeeze(2), transpose_rhs).unsqueeze(-2)


def compact_gqa_matmul(lhs: torch.Tensor, rhs: torch.Tensor, matmul_op=torch.matmul):
    if (lhs.ndim != 5 or rhs.ndim != 5 or lhs.shape[:2] != rhs.shape[:2] or lhs.shape[-2] != 1 or rhs.shape[2] != 1
            or lhs.shape[-1] != rhs.shape[-2] or lhs.dtype != torch.bfloat16 or rhs.dtype != torch.bfloat16):
        raise RuntimeError("Compact GQA requires matching BF16 block/head batches and one query row")
    # The group axis becomes independent output rows of the same GEMM. The
    # reduction dimension, operand values and BF16 output boundary are intact.
    return matmul_op(lhs.squeeze(-2), rhs.squeeze(2)).unsqueeze(-2)
