# SPDX-License-Identifier: Apache-2.0
"""Select the shared compiler's coarser local partitions; retain collectives."""


def make_candidate(reference, clone):
    result = clone(reference)
    result.decode_merge_mhc_partitions = True
    result.candidate_replay_tail = getattr(reference, 'benchmark_official_sampling', False)
    return result
