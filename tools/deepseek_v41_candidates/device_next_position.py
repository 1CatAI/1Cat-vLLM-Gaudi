# SPDX-License-Identifier: Apache-2.0
"""Reuse the bounded sampler and generate continuation coordinates in replay."""
import importlib
from pathlib import Path
import hashlib


def make_candidate(reference, clone):
    from vllm_gaudi.models import deepseek_v41_program

    model = importlib.reload(deepseek_v41_program)
    result = clone(reference)
    if not getattr(result, 'benchmark_official_sampling', False):
        raise ValueError('Device-position comparison requires official device controls')
    result.device_sampling = True
    result.device_next_position = True
    result.candidate_replay_tail = True
    source = Path(model.__file__).resolve()
    result.candidate_shared_sources = [dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())]
    # The fixture constructs a fresh common StageReplay from the current shared
    # PreparedGreedyTail class. No private arithmetic or reference mutation.
    return result
