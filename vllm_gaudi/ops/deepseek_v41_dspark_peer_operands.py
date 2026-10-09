# SPDX-License-Identifier: Apache-2.0
"""Use the maintained scheduled peer operands for the DSpark consumer."""

import torch


def make_peer_operands(tp_rank, tp_size):
    """Match stage_collectives' native transport without gather reconstruction.

    Keep the producer's ready outputs attached to the existing command. The
    consumer performs the rank-ordered sum at its BF16 boundary; communication
    and C1 default dispatch remain owned by the maintained native operators.
    """
    if tp_size not in (2, 4) or not 0 <= tp_rank < tp_size:
        raise ValueError("Scheduled peer operands require a supported TP group")

    def operands(value, ready_outputs=()):
        if value.dtype != torch.bfloat16 or value.numel() > 32768:
            raise ValueError("Deferred operands require the existing small BF16 transport")
        flat = value.reshape(1, -1).contiguous()
        if tp_size == 2:
            peer = (torch.ops.vllm_gaudi.tp2_exchange_peer_scheduled(flat, list(ready_outputs))
                    if ready_outputs else torch.ops.vllm_gaudi.tp2_exchange_peer(flat))
            ordered = (flat, peer) if tp_rank == 0 else (peer, flat)
            return torch.stack(ordered).reshape(tp_size, *value.shape)
        peers = (torch.ops.vllm_gaudi.tp_peer_allgather_scheduled(flat, tp_size, list(ready_outputs))
                 if ready_outputs else torch.ops.vllm_gaudi.tp_peer_allgather(flat, tp_size))
        return peers.reshape(tp_size, *value.shape)

    return operands
