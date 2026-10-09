# SPDX-License-Identifier: Apache-2.0
"""Measure selection publications between peer points in shared native replay."""
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace


def main():
    rank, tp = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG",
                                                              "").replace("{rank}", str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(tensor_parallel_size=tp, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel, init_distributed_environment,
                                  initialize_model_parallel)
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_replay import _Snapshot, stage_collectives
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
    from vllm_gaudi.ops.tp2_prepared_plan import (collect_prepared_group_replays, prepared_group_stats,
                                                  record_native_decoder_outputs, replay_native_decoder,
                                                  shutdown_prepared_group_plans)
    from check_deepseek_v41_selection_state import update_selection

    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(rank=rank,
                  status="running",
                  measurements=[],
                  formal_qualification=False,
                  scope="native update/consume between two peer reductions; no index scoring or model TPOT")
    owners, cases = [], []
    with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp))), \
            torch.inference_mode():
        try:
            init_distributed_environment(world_size=tp,
                                         rank=rank,
                                         distributed_init_method="env://",
                                         local_rank=rank,
                                         backend="hccl")
            initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            bind_worker_helpers(rank)
            reduce, _ = stage_collectives(rank, True, tp)
            for capacity in (8192, 128):
                owner = torch.nn.Module()
                owners.append(owner)
                owner.register_buffer("indices", torch.full((capacity, 512), -1, dtype=torch.int32, device="hpu"))
                owner.register_buffer("candidate_pool", torch.full((capacity, 2048),
                                                                   -1,
                                                                   dtype=torch.int32,
                                                                   device="hpu"))
                states = tuple(owner.buffers())
                metadata = SimpleNamespace(native_completion=None)
                adapter = DecoderTopology("deepseek_v41_selection_probe", (1, ), 2, False)

                def body(value, owner=owner):
                    selected = reduce(value).to(torch.int32)
                    blocks = torch.cat((selected, selected, selected, selected), -1)
                    emitted = update_selection(owner.indices, owner.candidate_pool, selected, blocks)
                    result = reduce(emitted.to(torch.bfloat16))
                    return result, result.float().sum(-1)

                compiled = torch.compile(body, backend="hpu_backend", fullgraph=True, dynamic=False)
                seed_cpu = (torch.arange(6 * 512).reshape(6, 512) % 17).bfloat16()
                fixed = seed_cpu.to("hpu")
                roots = dict(hidden_states=fixed,
                             pre_mix=None,
                             positions=None,
                             input_ids=None,
                             attention_inputs=(),
                             metadata=metadata,
                             state_generation=1,
                             state_tensors=states)

                def call(value,
                         owner=owner,
                         roots=roots,
                         compiled=compiled,
                         fixed=fixed,
                         adapter=adapter,
                         states=states):
                    result = replay_native_decoder(owner, **dict(roots, hidden_states=value))
                    if result is not None:
                        return result[0]
                    fixed.copy_(value)
                    with collect_prepared_group_replays(owner=owner,
                                                        adapter=adapter,
                                                        snapshot=lambda: _Snapshot(states),
                                                        **roots) as context:
                        context["group_index"] = 0
                        result = compiled(fixed)
                        record_native_decoder_outputs(*result)
                    return result[0]

                for offset in (0, 1):
                    value = (seed_cpu + offset).to("hpu")
                    actual = call(value)
                    expected = (seed_cpu + offset) * (2 * tp * tp)
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                    selected = ((seed_cpu + offset) * tp).int()
                    torch.testing.assert_close(owner.indices[:6].cpu(), selected, rtol=0, atol=0)
                    torch.testing.assert_close(owner.candidate_pool[:6].cpu(), selected.repeat(1, 4), rtol=0, atol=0)
                    assert (owner.indices[6:].cpu() == -1).all()
                    assert (owner.candidate_pool[6:].cpu() == -1).all()
                for _ in range(64):
                    call(value)
                torch.hpu.synchronize()
                before = prepared_group_stats()
                assert before["native_graphs"] == len(owners)
                samples = []
                for _ in range(3):
                    torch.distributed.barrier()
                    start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    start.record()
                    host = time.perf_counter_ns()
                    for _ in range(256):
                        call(value)
                    stop.record()
                    stop.synchronize()
                    samples.append(
                        dict(device_us=start.elapsed_time(stop) * 1000 / 256,
                             host_us=(time.perf_counter_ns() - host) / 1000 / 256))
                after = prepared_group_stats()
                assert before["native_captures"] == after["native_captures"]
                assert before["prepares"] == after["prepares"]
                report["measurements"].append(
                    dict(capacity=capacity, rows=6, exact=True, samples=samples, before=before, after=after))
                cases.append((capacity, call, value))
                (root / f"selection-replay-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
            report["interleaved"] = []
            for _, call, value in cases:
                for _ in range(256):
                    call(value)
            torch.hpu.synchronize()
            before = prepared_group_stats()
            for case in (0, 1, 1, 0, 1, 0, 0, 1):
                capacity, call, value = cases[case]
                torch.distributed.barrier()
                start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                start.record()
                host = time.perf_counter_ns()
                for _ in range(512):
                    call(value)
                stop.record()
                stop.synchronize()
                report["interleaved"].append(
                    dict(capacity=capacity,
                         device_us=start.elapsed_time(stop) * 1000 / 512,
                         host_us=(time.perf_counter_ns() - host) / 1000 / 512))
            after = prepared_group_stats()
            assert before["native_captures"] == after["native_captures"]
            assert before["prepares"] == after["prepares"]
            report["interleaved_stats"] = dict(before=before, after=after)
            report["status"] = "passed"
        except Exception as exc:
            report.update(status="failed", error=repr(exc))
            raise
        finally:
            (root / f"selection-replay-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
            if torch.distributed.is_initialized():
                shutdown_prepared_group_plans()
                destroy_model_parallel()
                destroy_distributed_environment()


if __name__ == "__main__":
    main()
