# SPDX-License-Identifier: Apache-2.0
"""Compare the shared epoch-aware policy on a retained real stage plan."""
from pathlib import Path
import hashlib


def make_candidate(reference, clone):
    from vllm_gaudi.ops import deepseek_v41_replay, tp2_prepared_plan

    result = clone(reference)
    result.native_receive_prepost = True
    result.candidate_replay_tail = getattr(reference, 'candidate_replay_tail', False)
    sources = [Path(module.__file__).resolve() for module in (deepseek_v41_replay, tp2_prepared_plan)]
    result.candidate_shared_sources = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                       for p in sources]
    return result
