# SPDX-License-Identifier: Apache-2.0
"""Qualify the shared all-route decoder without changing the reference owners."""
from types import MethodType
import hashlib
from pathlib import Path


def make_candidate(reference, clone):
    import importlib
    import torch
    from vllm_gaudi.models import deepseek_v41_program
    from vllm_gaudi.compilation import deepseek_v41_overlap

    # Cold preparation only. Existing reference recipes remain untouched;
    # only the candidate's method receives the current shared implementation.
    model = importlib.reload(deepseek_v41_program)
    overlap = importlib.reload(deepseek_v41_overlap)
    # Existing native reference plans stay retained. Future candidate cold
    # entries use the same maintained compiler and activation gate.
    implementation = model.PreparedMoE
    result = clone(reference)
    source = Path(deepseek_v41_program.__file__).resolve()
    result.candidate_shared_sources = [dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())]
    result.candidate_shared_sources.append(dict(path=str(Path(overlap.__file__).resolve()),
        sha256=hashlib.sha256(Path(overlap.__file__).read_bytes()).hexdigest()))
    result.layers = torch.nn.ModuleList([clone(block) for block in reference.layers])
    for block in result.layers:
        block.moe = clone(block.moe)
        block.moe.all_route_slots = True
        block.moe.forward = MethodType(implementation.forward, block.moe)
        block.moe._forward_n256_fp8 = MethodType(implementation._forward_n256_fp8, block.moe)
    result.candidate_required_operators = [
        'custom_deepseek_v41_expert_n256_moe_prequant_direct_finalize_slots_fp8_gaudi2']
    return result
