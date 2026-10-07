# SPDX-License-Identifier: Apache-2.0
"""Retain native peer communication, fuse only its ordered arithmetic consumer."""
from pathlib import Path
import hashlib


def make_candidate(reference, clone):
    import torch
    from vllm_gaudi.ops import deepseek_v41_ordered_peer_sum, deepseek_v41_replay

    if not hasattr(torch.ops.custom_op, 'custom_deepseek_v41_ordered_peer_sum_gaudi2'):
        raise RuntimeError('Build and install ordered peer reduction before loading this candidate')
    result = clone(reference, recursive=True)
    reduce, _ = deepseek_v41_replay.stage_collectives(
        result.tp_rank, True, result.tensor_parallel_size, ordered_peer_sum=True)
    for module in result.modules():
        if hasattr(module, 'reduce'):
            module.reduce = reduce
    result.candidate_replay_tail = True
    sources = [Path(module.__file__).resolve()
               for module in (deepseek_v41_ordered_peer_sum, deepseek_v41_replay)]
    result.candidate_shared_sources = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                       for p in sources]
    return result
