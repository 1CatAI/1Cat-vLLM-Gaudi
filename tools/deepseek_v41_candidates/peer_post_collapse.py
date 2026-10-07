# SPDX-License-Identifier: Apache-2.0
"""Keep peer exchange, fuse rank-ordered summation into its mHC consumer."""
from pathlib import Path
import hashlib


def make_candidate(reference, clone):
    import torch
    from vllm_gaudi.models import deepseek_v41_program
    from vllm_gaudi.ops import deepseek_v41_paged_attention, deepseek_v41_replay

    result = clone(reference, recursive=True)
    for block in result.layers:
        if not (hasattr(torch.ops.custom_op, 'custom_deepseek_v41_mhc_post_collapse_gaudi2')
                and block.mhc_interlayer_bf16 and not block.draft):
            raise ValueError('Peer post/collapse requires the qualified BF16 mHC handoff')
        block.mhc_post_collapse = True
        block.peer_post_collapse = True
        block.attention.peer_post_collapse = True
        block.moe.peer_post_collapse = True
    result.candidate_replay_tail = True
    sources = [Path(module.__file__).resolve() for module in (
        deepseek_v41_program, deepseek_v41_paged_attention, deepseek_v41_replay)]
    result.candidate_shared_sources = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                       for p in sources]
    return result
