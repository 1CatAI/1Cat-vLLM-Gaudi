# SPDX-License-Identifier: Apache-2.0
"""Keep native batch consumption after the shared all-route decoder."""
from tools.deepseek_v41_candidates.all_route_slots import make_candidate as make_slots


def make_candidate(reference, clone):
    result = make_slots(reference, clone)
    result.candidate_compiler_config = {'ENABLE_BGEMM_FLATTEN_TO_GEMM_FOR_SLICING': 'false'}
    return result
