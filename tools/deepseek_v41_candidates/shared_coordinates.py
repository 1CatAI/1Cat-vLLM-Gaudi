# SPDX-License-Identifier: Apache-2.0
"""Capture one coordinate producer for all groups in the same native stage."""
from pathlib import Path
import hashlib


def make_candidate(reference, clone):
    from vllm_gaudi.models import deepseek_v41_program
    from vllm_gaudi.ops import deepseek_v41_replay, deepseek_v41_decode_coordinates

    result = clone(reference)
    result.decode_shared_coordinates = True
    result.candidate_replay_tail = getattr(reference, 'candidate_replay_tail', False)
    sources = [Path(module.__file__).resolve() for module in
               (deepseek_v41_program, deepseek_v41_replay, deepseek_v41_decode_coordinates)]
    result.candidate_shared_sources = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                       for p in sources]
    return result
