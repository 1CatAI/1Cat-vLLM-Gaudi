# SPDX-License-Identifier: Apache-2.0
"""Real host rows -> pinned DMA -> TP gather -> Engram update -> mHC/norm.

Run under a two-module lease and an EngramResidency owner. This measures the
resident component, not an end-to-end speedup or a faulting-versus-resident AB.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[int(os.environ["LOCAL_RANK"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--microbatches", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    import numpy as np
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel, destroy_model_parallel
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree, linear
    from vllm_gaudi.ops.deepseek_v41_math import unpack_swa, engram_update, hc_pre, rms_norm
    from vllm_gaudi.ops.deepseek_v41_host import EngramHost
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend

    rank = int(os.environ["LOCAL_RANK"])
    if os.environ.get("DSV41_MICRO_RANK_CPUS"):
        os.sched_setaffinity(0, json.loads(os.environ["DSV41_MICRO_RANK_CPUS"])[rank])
    torch.set_num_threads(1)
    torch.manual_seed(23451)
    torch.hpu.set_device(rank)
    context = set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2)))
    context.__enter__()
    init_distributed_environment(world_size=2,
                                 rank=rank,
                                 distributed_init_method="env://",
                                 local_rank=rank,
                                 backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=2)
    initialize_tp2_fused_ar_norm_runtime()
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    _, gather = stage_collectives(rank, True)
    shard = PreparedV41Shard(args.prepared, 0, rank)
    specs = {
        k: v
        for k, v in shard.specs.items() if any(
            k.startswith(f"layers.{i}.{part}") for i in (1, 14) for part in ("engram.", "hc_attn_", "attn_norm."))
    }
    weights = _weight_tree(specs)
    load_weight_tree(shard, weights, "hpu", specs)
    cfg = json.loads((args.prepared / "config.json").read_text())["text_config"]
    host = EngramHost(args.prepared, rank, "hpu", max_tokens=64, checkpoint_audit=args.checkpoint_audit)
    reports = dict(scope=__doc__, rank=rank, ready_residency=host.residency(), cases=[])
    if os.environ.get("VLLM_HPU_DSV41_BATCH_DECODE") == "1":
        assert host.device_c1 is None, "Request batches must not pin the unused C1 device table"
        assert not host.audit["device_c1_enabled"]
    reports["device_c1_enabled"] = host.device_c1 is not None
    reports["host_memory"] = Path("/proc/self/smaps_rollup").read_text()
    output = args.output.with_name(f"rank{rank}-" + args.output.name)
    source = json.loads((args.prepared / f"engram-tp{rank}.json").read_text())
    arrays = {}
    for layer in (1, 14):
        arrays[layer] = []
        for suffix in ("weight", "scale"):
            item = source["tables"][f"layers.{layer}.engram.embed.{suffix}"]
            arrays[layer].append(
                np.memmap(item["file"],
                          dtype=np.uint8,
                          mode="r",
                          offset=item["shard_offset"],
                          shape=(item["row_stop"] - item["row_start"], item["row_bytes"])))

    def consumer(layer):
        w = weights.layers.get_submodule(str(layer))

        def chain(packed, residual, pre, active):
            rows = gather(packed if packed.dtype == torch.bfloat16 else unpack_swa(packed, 256), dim=1)
            kv = linear(rows.flatten(1), w.engram.wkv)
            updated = engram_update(residual, kv, w.engram.q_weight, w.engram.k_weight, active, cfg["rms_norm_eps"])
            value, _, _, _ = hc_pre(updated,
                                    pre,
                                    w.hc_attn_fn,
                                    w.hc_attn_scale,
                                    w.hc_attn_base,
                                    cfg["rms_norm_eps"],
                                    cfg["hc_eps"],
                                    cfg["hc_sinkhorn_iters"],
                                    request_batch=True)
            return rms_norm(value, w.attn_norm.weight, cfg["rms_norm_eps"], request_batch=True)

        return chain

    try:
        for b in args.batches:
            microbatches = min(args.microbatches, b)
            if b % microbatches:
                raise ValueError("Engram chain requires equally sized ordinary microbatches")
            width = b // microbatches
            compiled = [
                torch.compile(consumer(layer), backend=make_backend(), fullgraph=True, dynamic=False)
                for layer in (1, 14)
            ]
            residual = torch.randn(b, 4, 5120).bfloat16().to("hpu")
            pre = torch.ones(b, 4, device="hpu")
            active = torch.ones(b, dtype=torch.bool, device="hpu")
            begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
            samples = []
            faults_before = host.audit["major_faults"]
            for step in range(args.steps + 3):
                spans = [(f"b{b}-r{i}", step, [(step * 733 + i * 173 + 199) % 128000], [False]) for i in range(b)]
                residual.copy_(torch.randn(b, 4, 5120).bfloat16())
                torch.hpu.synchronize()
                started = time.perf_counter_ns()
                begin.record()
                transactions = []
                for first in range(0, b, width):
                    stop = first + width
                    ticket = host.prepare_batch(spans[first:stop], capacity=width, defer_wait=True)
                    host.wait(ticket)
                    values = [
                        fn(rows, residual[first:stop], pre[first:stop], active[first:stop])
                        for fn, rows in zip(compiled, ticket.buffers)
                    ]
                    host.complete_batch(ticket, [1] * width)
                    transactions.append((ticket, values, first, stop))
                end.record()
                end.synchronize()
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if step >= 3:
                    samples.append(dict(wall_ms=elapsed, device_ms=begin.elapsed_time(end)))
                if step in (2, args.steps + 2):
                    for ticket, values, first, stop in transactions:
                        for index, layer in enumerate((1, 14)):
                            split = host.shards[layer]
                            ids = ticket.batch.hash_ids[:, index, split["head_start"]:split["head_stop"]]
                            w, s = arrays[layer]
                            expected = np.concatenate((w[ids - split["row_start"]], s[ids - split["row_start"]]),
                                                      axis=-1)
                            reference_rows = torch.from_numpy(expected).to("hpu")
                            if ticket.buffers[index].dtype == torch.bfloat16:
                                reference_rows = unpack_swa(reference_rows, 256)
                            assert torch.equal(ticket.buffers[index].cpu().view(torch.uint8),
                                               reference_rows.cpu().view(torch.uint8))
                            reference = compiled[index](reference_rows, residual[first:stop], pre[first:stop],
                                                        active[first:stop])
                            assert torch.equal(values[index].cpu(), reference.cpu())
                            assert all(host.histories[f"b{b}-r{i}"].position == step + 1 for i in range(first, stop))
            reports["cases"].append(
                dict(batch=b,
                     samples=samples,
                     exact=True,
                     microbatches=microbatches,
                     major_faults=host.audit["major_faults"] - faults_before,
                     median_wall_ms=statistics.median(s["wall_ms"] for s in samples),
                     median_device_ms=statistics.median(s["device_ms"] for s in samples)))
            output.write_text(json.dumps(reports, indent=2))
            print(json.dumps(reports["cases"][-1]), flush=True)
        reports.update(final_residency=host.residency(), audit=host.audit)
        output.write_text(json.dumps(reports, indent=2))
    finally:
        host.close()
        destroy_model_parallel()
        torch.distributed.destroy_process_group()
        context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
