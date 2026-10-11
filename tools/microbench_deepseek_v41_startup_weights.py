# SPDX-License-Identifier: Apache-2.0
"""Compare bounded CPU loader preparation with three real expert batches.

No HPU is acquired. This measures reads, conversion, qualification and CPU
staging; it does not establish upload time or complete service startup time.
"""
import argparse
import json
import os
from pathlib import Path
from statistics import median
import time

import numpy as np

from vllm_gaudi.ops.deepseek_v41_expert_load import prepare_expert_batch
from vllm_gaudi.ops.deepseek_v41_fp8 import read_expert
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard


def stage_batch(shard, q, scales, first, *, workers, compact, active_k, pack_q16=None):
    blocks, stream = q.shape[1:]
    width = stream // 8 + 128 if compact else scales.shape[-1] * 2
    outputs = (np.empty((16, blocks // 2, stream * 2), dtype="<i2"), np.empty(
        (16, blocks // 2, width), dtype="<i2"), np.empty((16, blocks // 2, 256), dtype="<u2"))
    flags = []
    started = time.perf_counter_ns()
    for expert, packed, planes, channel, eligible in prepare_expert_batch(
            shard,
            q,
            scales,
            first,
            first + 16,
            compact_scales=compact,
            active_k=active_k,
            workers=workers,
            pack_q16=pack_q16,
    ):
        for destination, value in zip(outputs, (packed, planes, channel), strict=True):
            destination[expert - first] = value
        flags.append(eligible)
    return outputs, flags, (time.perf_counter_ns() - started) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--reference-workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--pack-library", type=Path, help="Optional exact native CPU layout candidate")
    parser.add_argument("--cpus", help="Comma-separated machine-local CPU allocation")
    args = parser.parse_args()
    pack_q16 = None
    if args.pack_library:
        from vllm_gaudi.ops.deepseek_v41_startup_pack import native_q16_packer

        pack_q16 = native_q16_packer(args.pack_library)
    if args.cpus:
        os.sched_setaffinity(0, {int(cpu) for cpu in args.cpus.split(",")})
    shard = PreparedV41Shard(args.prepared, 0, args.rank)
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    pairs, initialization = [], []
    for projection in ("w13", "w2"):
        prefix = f"layers.{args.layer}.ffn.experts.{projection}"
        q, scales = (shard.catalog[prefix + suffix] for suffix in ("_q16", "_s16"))
        active_k = config["moe_intermediate_size"] // shard.tensor_parallel_size if projection == "w2" else q.shape[
            -1] // 32
        if pack_q16 is not None:
            # Record first-use cost rather than mixing it into steady batch
            # throughput. Both arms exercise the same initialized projection.
            for workers, packer in ((args.reference_workers, None), (args.workers, pack_q16)):
                outputs, flags, elapsed = stage_batch(shard,
                                                      q,
                                                      scales,
                                                      0,
                                                      workers=workers,
                                                      compact=shard.tensor_parallel_size == 4,
                                                      active_k=active_k,
                                                      pack_q16=packer)
                initialization.append(dict(projection=projection, native=packer is not None, elapsed_ms=elapsed))
                del outputs, flags
        for first in (0, 160, 320):
            # Equal cache state for the CPU comparison. No file pages are
            # explicitly evicted and neither arm retains a whole checkpoint.
            for expert in range(first, first + 16):
                read_expert(q, expert, keep_file_cache=True)
                read_expert(scales, expert, keep_file_cache=True)
            reference, old_flags, old_ms = stage_batch(shard,
                                                       q,
                                                       scales,
                                                       first,
                                                       workers=args.reference_workers,
                                                       compact=shard.tensor_parallel_size == 4,
                                                       active_k=active_k)
            candidate, new_flags, new_ms = stage_batch(shard,
                                                       q,
                                                       scales,
                                                       first,
                                                       workers=args.workers,
                                                       pack_q16=pack_q16,
                                                       compact=shard.tensor_parallel_size == 4,
                                                       active_k=active_k)
            exact = old_flags == new_flags and all(
                np.array_equal(a, b) for a, b in zip(reference, candidate, strict=True))
            if not exact:
                raise RuntimeError("Parallel loader changed prepared weight bytes or qualification")
            pairs.append(
                dict(projection=projection,
                     first_expert=first,
                     experts=16,
                     serial_ms=old_ms,
                     candidate_ms=new_ms,
                     saving_ms=old_ms - new_ms,
                     exact=exact))
            del reference, candidate
    shard.check_identity()
    report = dict(scope="CPU reads, conversion, SAT checks and bounded staging; no HPU or service-startup claim",
                  cpus=sorted(os.sched_getaffinity(0)),
                  workers=args.workers,
                  reference_workers=args.reference_workers,
                  pack_library=str(args.pack_library) if args.pack_library else None,
                  initialization=initialization,
                  pairs=pairs,
                  source_plan_fingerprint=shard.manifest["plan_fingerprint"],
                  summaries={
                      projection:
                      dict(serial_median_ms=median(p["serial_ms"] for p in pairs if p["projection"] == projection),
                           candidate_median_ms=median(p["candidate_ms"] for p in pairs
                                                      if p["projection"] == projection),
                           all_three_faster=all(p["saving_ms"] > 0 for p in pairs if p["projection"] == projection))
                      for projection in ("w13", "w2")
                  })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summaries"]))


if __name__ == "__main__":
    main()
