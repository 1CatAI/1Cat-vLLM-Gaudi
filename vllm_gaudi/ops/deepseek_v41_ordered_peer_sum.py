# SPDX-License-Identifier: Apache-2.0
"""Fixed-rank BF16 peer reduction with a single FP32 accumulation boundary."""
import torch


def ordered_peer_sum_reference(shards):
    if shards.dtype != torch.bfloat16 or shards.ndim != 2 or not 2 <= shards.shape[0] <= 8:
        raise ValueError('Ordered peer sum requires BF16 [ranks, width]')
    total = shards[0].float()
    for rank in range(1, shards.shape[0]):
        total = total + shards[rank].float()
    return total.to(torch.bfloat16).reshape(1, -1)


def ordered_peer_sum(shards):
    return torch.ops.custom_op.custom_deepseek_v41_ordered_peer_sum_gaudi2(shards)
