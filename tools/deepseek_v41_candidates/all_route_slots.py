# SPDX-License-Identifier: Apache-2.0
"""Qualify the shared all-route decoder without changing the reference owners."""
from types import MethodType
import hashlib
from pathlib import Path


def make_candidate(reference, clone):
    import importlib
    import torch
    from vllm_gaudi.models import deepseek_v41_program

    # Cold preparation only. Existing reference recipes remain untouched;
    # only the candidate's method receives the current shared implementation.
    implementation = importlib.reload(deepseek_v41_program).PreparedMoE._forward_n256_fp8
    result = clone(reference)
    source = Path(deepseek_v41_program.__file__).resolve()
    result.candidate_shared_sources = [dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())]
    result.layers = torch.nn.ModuleList([clone(block) for block in reference.layers])
    for block in result.layers:
        block.moe = clone(block.moe)
        block.moe.all_route_slots = True
        block.moe._forward_n256_fp8 = MethodType(implementation, block.moe)
    return result
