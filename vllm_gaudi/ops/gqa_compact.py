# SPDX-License-Identifier: Apache-2.0
"""Keep grouped query rows together without expanding the shared K/V operand."""
import torch


def compact_gqa_matmul(lhs: torch.Tensor, rhs: torch.Tensor, matmul_op=torch.matmul):
    if (lhs.ndim != 5 or rhs.ndim != 5 or lhs.shape[:2] != rhs.shape[:2] or lhs.shape[-2] != 1 or rhs.shape[2] != 1
            or lhs.shape[-1] != rhs.shape[-2] or lhs.dtype != torch.bfloat16 or rhs.dtype != torch.bfloat16):
        raise RuntimeError("Compact GQA requires matching BF16 block/head batches and one query row")
    # The group axis becomes independent output rows of the same GEMM. The
    # reduction dimension, operand values and BF16 output boundary are intact.
    return matmul_op(lhs.squeeze(-2), rhs.squeeze(2)).unsqueeze(-2)
