# SPDX-License-Identifier: Apache-2.0
"""Keep independent mHC work before the peer wait, inside its producer recipe."""
import hashlib
from pathlib import Path

from tools.deepseek_v41_candidates.device_closed_loop import make_candidate as closed_loop


def make_candidate(reference, clone):
    from vllm_gaudi.compilation import deepseek_v41_mhc_producer_fusion, deepseek_v41_overlap

    result = closed_loop(reference, clone)
    result.decode_merge_mhc_partitions = False
    result.decode_mhc_producer_fusion = True
    result.candidate_shared_sources += [
        dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (Path(module.__file__).resolve()
                     for module in (deepseek_v41_mhc_producer_fusion, deepseek_v41_overlap))
    ]
    return result
