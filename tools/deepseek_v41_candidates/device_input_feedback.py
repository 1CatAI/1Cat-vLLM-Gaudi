# SPDX-License-Identifier: Apache-2.0
"""Select the shared private-root feedback path for the resident C1 chain."""
from pathlib import Path
import hashlib


def make_candidate(reference, clone):
    from vllm_gaudi.models import deepseek_v41_program
    from vllm_gaudi.ops import deepseek_v41_replay, deepseek_v41_sampling

    result = clone(reference)
    if not (getattr(result, 'benchmark_official_sampling', False)
            and result.device_sampling and result.device_next_position):
        raise ValueError('Input feedback requires the official bounded sampler and device position baseline')
    result.device_input_feedback = True
    result.candidate_replay_tail = True
    sources = [Path(module.__file__).resolve()
               for module in (deepseek_v41_program, deepseek_v41_replay, deepseek_v41_sampling)]
    result.candidate_shared_sources = [dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                                       for source in sources]
    return result
