# SPDX-License-Identifier: Apache-2.0
"""Measure the maintained sampling-frame and two-layer Engram input lifecycle."""
from pathlib import Path
import hashlib


def make_candidate(reference, clone):
    from vllm_gaudi.ops import deepseek_v41_device_engram, deepseek_v41_device_loop, deepseek_v41_replay

    if not (reference.device_sampling and reference.device_next_position):
        raise ValueError('Closed-loop micro requires the official device-sampling parent')
    result = clone(reference)
    result.device_input_feedback = True
    result.device_closed_loop = True
    result.candidate_device_loop = True
    result.candidate_replay_tail = True
    sources = [Path(module.__file__).resolve()
               for module in (deepseek_v41_device_engram, deepseek_v41_device_loop, deepseek_v41_replay)]
    result.candidate_shared_sources = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                       for p in sources]
    return result
