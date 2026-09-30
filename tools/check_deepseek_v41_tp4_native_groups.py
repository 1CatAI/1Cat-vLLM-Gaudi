# SPDX-License-Identifier: Apache-2.0
"""Check compiled TP4 replay, both collectives and mutable ownership; no timing claim."""
import argparse
import copy
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--timing-probe", action="store_true",
                        help="After exact capture, measure this small consumer chain before wider integration")
    parser.add_argument("--reuse-timing-reference", type=Path)
    parser.add_argument("--diagnose-stages", action="store_true",
                        help="Retain post-collective values to localize the first replay mismatch")
    args = parser.parse_args()
    if args.timing_probe and not args.capture_only:
        parser.error("The initial timing probe uses only the short capture gate")
    if args.reuse_timing_reference and not args.timing_probe:
        parser.error("An archived timing reference requires the timing probe")
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    if "{rank}" in os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", ""):
        os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ["PT_HPU_RECIPE_CACHE_CONFIG"].replace("{rank}", str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import torch.nn.functional as F
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel,
                                  init_distributed_environment, initialize_model_parallel)
    from vllm_gaudi.compilation.deepseek_v41_prepared import PreparedTP4Group
    from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats, shutdown_prepared_group_plans
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method="env://", local_rank=rank,
                                 backend="hccl")
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=1))
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(rank=rank, status="running", scope="synthetic compiled MME/AR/AG/state gate; no speed credit",
                  cases=[])
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            reduce, gather = stage_collectives(rank, False)
            reduce(torch.ones(1, device="hpu", dtype=torch.bfloat16)).cpu()
            bind_worker_helpers(rank)

            class Chain(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.generation, self.search_length = 1, 32768
                    torch.manual_seed(217 + rank)
                    for name, shape in (("w1", (128, 5120)), ("w2", (1024, 128)),
                                        ("w3", (8, 128)), ("wo", (5120, 4096))):
                        self.register_buffer(name, (torch.randn(shape) / 32).bfloat16().to("hpu"))
                    self.register_buffer("state", torch.zeros(32, 5120, device="hpu", dtype=torch.bfloat16))

                def forward(self, value, slots):
                    partial = F.linear(value, self.w1)
                    reduced = reduce(partial)
                    query = F.linear(reduced, self.w2) / 128
                    gathered = gather(query.reshape(value.shape[0], 8, 128), dim=1)
                    weights = gather(F.linear(reduced, self.w3) / 128, dim=1)
                    output = F.linear(gathered.flatten(1), self.wo) + weights.sum(-1, keepdim=True)
                    output = reduce(output) / 256 + value / 2
                    self.state.index_copy_(0, slots, output)
                    state_output = self.state.index_select(0, slots)
                    if args.diagnose_stages:
                        # Lossless observation buffers avoid exporting aliased
                        # views that alter the prepared host-graph vocabulary.
                        return output, state_output, reduced.float(), gathered.float(), weights.float()
                    return output, state_output

            model = Chain()
            oracle_model = copy.deepcopy(model)
            oracle = torch.compile(oracle_model, backend=make_backend(), fullgraph=True, dynamic=False)
            candidate = PreparedTP4Group(model, model, make_backend(native_group_owner=model))
            report["loaded_native_libraries"] = sorted({line.split()[-1]
                                                       for line in Path("/proc/self/maps").read_text().splitlines()
                                                       if "/libhcl.so" in line or "/libSynapse.so" in line})
            previous = None
            for step, count in enumerate([1] * 3 if args.capture_only else [1] * 20 + [2, 6] + [1] * 20):
                if step == 22:
                    retired, saved = model.state, model.state.cpu()
                    model.state = torch.zeros_like(model.state)
                    oracle_model.state = torch.zeros_like(oracle_model.state)
                    model.generation += 1
                torch.manual_seed(993 + step)
                value = (torch.randn(count, 5120) / 8).bfloat16().to("hpu")
                if count == 1 and previous is not None:
                    value = value + previous / 4
                slots = (torch.arange(count, dtype=torch.int64) + step % 16).to("hpu")
                wanted = oracle(value, slots)
                observed = candidate(value, slots)
                actual_cpu = [tensor.cpu() for tensor in observed]
                expected_cpu = [tensor.cpu() for tensor in wanted]
                if any(not torch.equal(actual, expected)
                       for actual, expected in zip(actual_cpu, expected_cpu, strict=True)):
                    roles = ("output", "state_output", "first_allreduce", "query_allgather", "weight_allgather")
                    mismatch = [{"role": role, "shape": list(actual.shape),
                                 "different": int((actual != expected).sum()),
                                 "max_absolute": float((actual.float() - expected.float()).abs().max())}
                                for role, actual, expected in zip(roles, actual_cpu, expected_cpu)]
                    report["first_mismatch"] = {"step": step, "values": mismatch}
                    torch.save({"input": value.cpu(), "actual": actual_cpu, "expected": expected_cpu,
                                "slots": slots.cpu()}, root / f"first-replay-difference-rank{rank}.pt")
                    print(json.dumps({"rank": rank, "mismatch": mismatch}), flush=True)
                for actual, expected in zip(actual_cpu, expected_cpu, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(model.state.cpu(), oracle_model.state.cpu(), rtol=0, atol=0)
                if step >= 22:
                    torch.testing.assert_close(retired.cpu(), saved, rtol=0, atol=0)
                previous = observed[0] if count == 1 else None
                report["cases"].append(dict(step=step, count=count, exact=True))
                print(json.dumps(dict(rank=rank, step=step, exact=True)), flush=True)
            stats = prepared_group_stats()
            assert stats["native_replays"] >= (1 if args.capture_only else 8), stats
            assert stats["native_collectives"] >= 4, stats
            report.update(status="exact", native=stats)
            if args.timing_probe:
                # Reuse an existing ordinary measurement. Its changing-output
                # oracle remains untimed; this is not a model speed claim.
                archived = None
                if args.reuse_timing_reference:
                    archived = json.loads((args.reuse_timing_reference / f"native-group-rank{rank}.json").read_text())
                    assert archived['status'] == 'exact' and archived['timing_final_exact']
                    report['timing_reference_reused'] = str(args.reuse_timing_reference)
                ordinary = PreparedTP4Group(oracle_model, oracle_model, make_backend())
                slots = torch.zeros(1, dtype=torch.int64, device="hpu")
                torch.manual_seed(6151)
                initial = (torch.randn(1, 5120) / 8).bfloat16().to("hpu")
                bias = (torch.randn(1, 5120) / 128).bfloat16().to("hpu")
                report["timings"] = {}
                finals = []
                for name, function in (("ordinary", ordinary), ("native", candidate)):
                    value = initial.clone()
                    for _ in range(32):
                        value = function(value, slots)[0] + bias
                    torch.hpu.synchronize()
                    measure = name != 'ordinary' or archived is None
                    if measure:
                        start = torch.hpu.Event(enable_timing=True)
                        stop = torch.hpu.Event(enable_timing=True)
                        start.record()
                        wall = time.perf_counter_ns()
                    for _ in range(64):
                        value = function(value, slots)[0] + bias
                    if measure:
                        stop.record()
                    torch.hpu.synchronize()
                    if measure:
                        wall = (time.perf_counter_ns() - wall) / 1e6 / 64
                        report["timings"][name] = {"steps": 64, "host_ms_per_step": wall,
                                                   "device_ms_per_step": start.elapsed_time(stop) / 64}
                    else:
                        report['timings'][name] = archived['timings'][name]
                    finals.append(value.cpu())
                torch.testing.assert_close(finals[0], finals[1], rtol=0, atol=0)
                report["timing_final_exact"] = True
                report['native_after_timing'] = prepared_group_stats()
                report["timing_scope"] = "Small synthetic submission chain only; no model gain-ledger credit."
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (root / f"native-group-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        shutdown_prepared_group_plans()
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
