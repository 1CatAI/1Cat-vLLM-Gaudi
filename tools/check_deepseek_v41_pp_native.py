# SPDX-License-Identifier: Apache-2.0
"""Isolate PP communication from native TP replay without loading model weights."""

import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

rank = int(os.environ["LOCAL_RANK"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment  # noqa: E402

prepare_environment()

import torch  # noqa: E402
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime  # noqa: E402
from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology  # noqa: E402
from vllm_gaudi.ops.tp2_prepared_plan import (  # noqa: E402
    collect_prepared_group_replays,
    prepared_group_stats,
    record_native_decoder_outputs,
    recapture_native_decoder_programs,
    replay_native_decoder,
    shutdown_prepared_group_plans,
)
from vllm_gaudi.v1.worker.deepseek_v41_runner import PPBuffers  # noqa: E402


def program(hidden, pre, positions):
    value = hidden[:, 0, :] + positions.to(torch.bfloat16).unsqueeze(-1)
    value = (value + torch.ops.vllm_gaudi.tp2_exchange_peer(value)) * 0.5
    value = (value + torch.ops.vllm_gaudi.tp2_exchange_peer(value)) * 0.5
    return value, pre + 1, value + 2


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-receive", action="store_true")
    parser.add_argument("--local-input", action="store_true",
                        help="Complete unrelated PP traffic, then use independent CPU-generated replay inputs")
    parser.add_argument("--restore-tp-context", action="store_true",
                        help="Diagnostic only: insert an ordinary TP exchange after PP")
    parser.add_argument("--steps", type=int, default=34)
    parser.add_argument("--profile", action="store_true", help="Capture only steps after cold/capture/hot warmup")
    parser.add_argument("--profile-cycle-steps", type=int,
                        help="Export and restart profiling every N hot steps to verify repeated acquisitions")
    parser.add_argument("--verify-commit", action="store_true", help="Exercise the real PP int64 verify commit")
    args = parser.parse_args()
    torch.hpu.set_device(rank)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    bind_worker_cpu(rank)
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method="env://",
                                 local_rank=rank, backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
    initialize_tp2_fused_ar_norm_runtime()
    pp = PPBuffers(torch.device("hpu"))
    owner = torch.nn.Identity()
    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    topology = DecoderTopology("deepseek_v41_pp1", (1,), 2, False)
    fixed, records, profiler = None, [], None
    evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
    for step in range(args.steps):
        if args.profile and (step == 4 or (args.profile_cycle_steps and step > 4
                                         and (step - 4) % args.profile_cycle_steps == 0)):
            if profiler is not None:
                profiler.stop()
                cycle = (step - 4) // args.profile_cycle_steps
                profiler.export_chrome_trace(str(evidence / f"rank{rank}.cycle{cycle}.trace.json.gz"))
                recapture_native_decoder_programs()
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.HPU],
                                              record_shapes=True, with_stack=False)
            profiler.start()
            recapture_native_decoder_programs()
        value = step % 32
        print(f"RANK {rank} STEP {step} PP start", flush=True)
        if pp.group.is_first_rank:
            hidden = torch.full((1, 4, 5120), value + rank, dtype=torch.bfloat16, device="cpu").to("hpu")
            pre = torch.full((1, 4), float(value), dtype=torch.float32, device="cpu").to("hpu")
            pp.exchange({"hidden_states": hidden, "pre_mix": pre}, 1)
        else:
            received = pp.exchange(None, 1)
            if args.wait_receive or args.local_input:
                torch.hpu.synchronize()
            if args.local_input:
                received = dict(
                    hidden_states=torch.full((1, 4, 5120), value + rank % 2,
                                             dtype=torch.bfloat16, device="cpu").to("hpu"),
                    pre_mix=torch.full((1, 4), float(value), dtype=torch.float32, device="cpu").to("hpu"))
            live = (received["hidden_states"], received["pre_mix"],
                    torch.tensor([value], dtype=torch.int32, device="cpu").to("hpu"))
            if args.restore_tp_context:
                probe = torch.zeros(1, 5120, dtype=torch.bfloat16, device="hpu")
                torch.ops.vllm_gaudi.tp2_exchange_peer(probe)
                torch.hpu.synchronize()
            if fixed is None:
                fixed = tuple(value.clone() for value in live)
            roots = dict(hidden_states=live[0], pre_mix=live[1], positions=live[2], residual=None,
                         metadata=SimpleNamespace(is_prompt=False, native_completion=None))
            result = replay_native_decoder(owner, **roots)
            if result is None:
                for destination, source in zip(fixed, live, strict=True):
                    destination.copy_(source)
                fixed_roots = dict(roots, hidden_states=fixed[0], pre_mix=fixed[1], positions=fixed[2])
                with collect_prepared_group_replays(
                        owner=owner, adapter=topology,
                        snapshot=lambda: SimpleNamespace(restore=lambda: None), **fixed_roots):
                    result = compiled(*fixed)
                    record_native_decoder_outputs(*result)
            print(f"RANK {rank} STEP {step} native submitted " + json.dumps(prepared_group_stats()), flush=True)
            ticket = roots["metadata"].native_completion
            if ticket is not None:
                deadline = time.monotonic() + 15
                while not ticket.query():
                    if time.monotonic() >= deadline:
                        diagnostic = dict(rank=rank, step=step, status="native-completion-timeout",
                                          native=prepared_group_stats())
                        evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
                        (evidence / f"timeout-rank{rank}.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
                        raise RuntimeError(f"Native PP diagnostic completion timed out: {diagnostic}")
                    time.sleep(0.01)
            actual = [value.cpu() for value in result]
            expected = (2 * value + 0.5, value + 1, 2 * value + 2.5)
            for value, scalar in zip(actual, expected, strict=True):
                assert torch.equal(value, torch.full_like(value, scalar)), (rank, step, value, scalar)
            records.append({"step": step, "native": prepared_group_stats()})
            print(f"RANK {rank} STEP {step} outputs exact", flush=True)
        if args.verify_commit:
            output = [100 + step % 32]
            drafts = [200 + step % 32 + index for index in range(5)]
            committed, actual_output, actual_drafts = pp.finish(1, output, drafts)
            assert (committed, actual_output, actual_drafts) == (1, output, drafts)
        pp.drain()
        torch.hpu.synchronize()
        pp.group.barrier()
    if profiler is not None:
        profiler.stop()
        profiler.export_chrome_trace(str(evidence / f"rank{rank}.trace.json.gz"))
        recapture_native_decoder_programs()
    from vllm_gaudi.ops.tp2_runtime_profile import verify_loaded_profile_libraries
    (evidence / f"rank{rank}.json").write_text(json.dumps(dict(
        status="exact", diagnostic="PP/native synchronization only; no model qualification",
        wait_receive=args.wait_receive, local_input=args.local_input,
        profile=args.profile, verify_commit=args.verify_commit,
        profile_libraries=verify_loaded_profile_libraries(),
        restore_tp_context=args.restore_tp_context, records=records,
        pp_sends=pp.sends, pp_receives=pp.receives, native=prepared_group_stats()), indent=2) + "\n")
    print(f"RANK {rank} shutdown native plans", flush=True)
    shutdown_prepared_group_plans()
    print(f"RANK {rank} shutdown model parallel groups", flush=True)
    destroy_model_parallel()
    print(f"RANK {rank} shutdown distributed environment", flush=True)
    destroy_distributed_environment()
    print(f"RANK {rank} shutdown complete", flush=True)


if __name__ == "__main__":
    with set_current_vllm_config(VllmConfig(
            parallel_config=ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2))):
        main()
