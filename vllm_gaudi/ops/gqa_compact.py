# SPDX-License-Identifier: Apache-2.0
"""Keep grouped query rows together without expanding the shared K/V operand."""
import torch


def native_gqa_matmul(lhs: torch.Tensor, rhs: torch.Tensor, *, transpose_rhs: bool = False):
    """Preserve the cache layout and describe QK transposition in the MME node."""
    if (lhs.ndim != 5 or rhs.ndim != 5 or lhs.shape[:2] != rhs.shape[:2] or lhs.shape[1:3] != (2, 6)
            or lhs.shape[-2] != 1 or rhs.shape[2] != 1 or lhs.dtype != torch.bfloat16 or rhs.dtype != torch.bfloat16
            or lhs.device != rhs.device or lhs.shape[-1] != rhs.shape[-1 if transpose_rhs else -2]):
        raise RuntimeError("Native GQA requires matching BF16 compact query and cache batches")
    # Construct K in its original layout at the caller. Building a transpose
    # first and undoing it here can leave an unnecessary physical DMA node.
    return torch.ops.custom_op.tp2_gqa_matmul(lhs.squeeze(-2), rhs.squeeze(2), transpose_rhs).unsqueeze(-2)


def single_batch_to_blocks(tensor: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Express the single-product BF16 mapping without an MME reduction loop."""
    if (tensor.ndim < 2 or tensor.shape[0] != 1 or mapping.ndim != 2 or mapping.shape[1] != 1
            or tensor.dtype != torch.bfloat16 or mapping.dtype != torch.bfloat16 or tensor.device != mapping.device):
        raise RuntimeError("Single-batch mapping requires matching BF16 tensors and a one-column mapping")
    return mapping.reshape(-1, *([1] * (tensor.ndim - 1))) * tensor


def compact_gqa_matmul(lhs: torch.Tensor, rhs: torch.Tensor, matmul_op=torch.matmul):
    if (lhs.ndim != 5 or rhs.ndim != 5 or lhs.shape[:2] != rhs.shape[:2] or lhs.shape[-2] != 1 or rhs.shape[2] != 1
            or lhs.shape[-1] != rhs.shape[-2] or lhs.dtype != torch.bfloat16 or rhs.dtype != torch.bfloat16):
        raise RuntimeError("Compact GQA requires matching BF16 block/head batches and one query row")
    # The group axis becomes independent output rows of the same GEMM. The
    # reduction dimension, operand values and BF16 output boundary are intact.
    return matmul_op(lhs.squeeze(-2), rhs.squeeze(2)).unsqueeze(-2)
