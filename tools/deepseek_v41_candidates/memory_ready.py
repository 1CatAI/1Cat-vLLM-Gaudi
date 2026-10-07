# SPDX-License-Identifier: Apache-2.0
"""Qualify acquiring readers on the shared segmented native stage owner."""
from pathlib import Path
import hashlib
import os


def make_candidate(reference, clone):
    import torch
    from vllm_gaudi.ops import deepseek_v41_replay, tp2_prepared_plan
    from vllm_gaudi.models import deepseek_v41_program
    from vllm_gaudi.compilation import deepseek_v41_memory_ready

    library = Path(os.environ['DSV41_MEMORY_READY_OPERATOR']).resolve(strict=True)
    torch.ops.load_library(str(library))
    result = clone(reference)
    result.native_memory_ready = True
    result.native_receive_prepost = getattr(reference, 'native_receive_prepost', False)
    result.candidate_replay_tail = getattr(reference, 'candidate_replay_tail', False)
    result.candidate_required_operators = ('private_memory_ready_post',)
    sources = [Path(module.__file__).resolve() for module in (
        deepseek_v41_replay, tp2_prepared_plan, deepseek_v41_program, deepseek_v41_memory_ready)]
    result.candidate_shared_sources = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                       for p in (*sources, library)]
    return result
