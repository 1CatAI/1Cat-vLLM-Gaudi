# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU preparation; device copies retain their original ordering."""
from concurrent.futures import ThreadPoolExecutor

from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert, saturated_decode_eligible
from vllm_gaudi.ops.deepseek_v41_fp8 import read_expert


def preparation_workers(source_q, source_s, requested, staging_bytes):
    if not isinstance(requested, int) or not 1 <= requested <= 4:
        raise ValueError("Expert loading supports one to four CPU preparation workers")
    if staging_bytes < 0 or staging_bytes > 128 << 20:
        raise ValueError("Expert loading staging exceeds its bounded buffer")
    q_bytes = source_q.nbytes // source_q.shape[0]
    s_bytes = source_s.nbytes // source_s.shape[0]
    channels = source_q.shape[1] * 128 * 2
    # Includes the source, output, nibble scans and simultaneously live NumPy
    # temporaries. Completed results can also remain in the bounded batch.
    per_worker = 16 * q_bytes + 32 * s_bytes + 64 * channels
    budget = (2 << 30) - 2 * staging_bytes
    if per_worker > budget:
        raise ValueError("Expert CPU preparation exceeds its temporary memory budget")
    return min(requested, max(1, budget // per_worker))


def prepare_expert_batch(shard, source_q, source_s, first, last, *, compact_scales, active_k, workers=1, pack_q16=None):
    if not 0 <= first < last <= source_q.shape[0] or last - first > 16:
        raise ValueError("Expert preparation requires a nonempty bounded rank-local batch")
    if source_q.shape[0] != source_s.shape[0]:
        raise ValueError("Expert weight and scale counts disagree")
    blocks, stream = source_q.shape[1:]
    scales = stream // 8 + 128 if compact_scales else source_s.shape[-1] * 2
    output_bytes = (last - first) * (blocks // 2) * (stream * 2 + scales + 256) * 2
    workers = preparation_workers(source_q, source_s, workers, output_bytes)

    def prepare(expert):
        shard.check_identity()
        q, scales, channel, _ = prepare_expert(
            read_expert(source_q, expert, keep_file_cache=True),
            read_expert(source_s, expert, keep_file_cache=True),
            compact_scales=compact_scales,
            pack_q16=pack_q16,
        )
        import numpy as np

        eligible = not np.any(q[:, active_k * 64:]) and saturated_decode_eligible(scales, active_k=active_k)
        shard.check_identity()
        return expert, q, scales, channel, bool(eligible)

    if workers == 1:
        yield from map(prepare, range(first, last))
    else:
        # map preserves expert order. Workers touch only rank-local files and
        # CPU arrays; HPU allocation, copies and publication stay on the caller.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(prepare, range(first, last))
