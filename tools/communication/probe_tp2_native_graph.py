#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qualify native compute -> TP2 exchange -> dependent compute replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def _pin_torchrun_rank() -> tuple[int, int]:
    modules = [int(value) for value in os.environ["HABANA_VISIBLE_MODULES"].split(",")]
    if len(modules) != 2:
        raise RuntimeError("HABANA_VISIBLE_MODULES must contain exactly two module IDs")
    local_rank = int(os.environ["LOCAL_RANK"])
    module_to_device = {}
    for module_path in Path("/sys/class/accel").glob("accel*/device/module_id"):
        module_id = int(module_path.read_text(encoding="ascii").strip())
        device_id = int(module_path.parents[1].name.removeprefix("accel"))
        module_to_device[module_id] = device_id
    module_id = modules[local_rank]
    os.environ["HLS_MODULE_ID"] = str(module_id)
    return module_id, module_to_device[module_id]


MODULE_ID, DEVICE_ID = _pin_torchrun_rank()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import habana_frameworks.torch.core as htcore  # noqa: E402, F401
from native_graph_validation import assert_exact, validate_snapshots  # noqa: E402


def _load_bridge(path: Path):
    spec = importlib.util.spec_from_file_location("tp2_fused_ar_norm_bridge", path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load TP2 bridge: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cpu_inputs(rank: int, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.arange(5120, dtype=torch.int64)
    partial = (((index * 17 + rank * 29 + step * 7) % 127) - 63).float().div_(64).to(torch.bfloat16)
    residual = (((index * 11 + step * 5) % 113) - 56).float().div_(32).to(torch.bfloat16)
    weight = (1 + (((index * 3) % 31) - 15).float().div_(128)).to(torch.bfloat16)
    return partial.view(1, 5120), residual.view(1, 5120), weight


def _rms_norm(value: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    value = value.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon) * weight.float()).to(torch.bfloat16)


def _reference(step: int, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    rank_inputs = [_cpu_inputs(rank, step) for rank in range(2)]
    pre = []
    for partial, residual, weight in rank_inputs:
        pre_residual = (partial.float() + residual.float()).to(torch.bfloat16)
        pre.append(_rms_norm(pre_residual, weight, epsilon))
    reduced = (pre[0].float() + pre[1].float()).to(torch.bfloat16)
    residual = rank_inputs[0][1]
    weight = rank_inputs[0][2]
    output_residual = (reduced.float() + residual.float()).to(torch.bfloat16)
    return _rms_norm(output_residual, weight, epsilon), output_residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=4096)
    parser.add_argument("--validation-replays", type=int, default=4)
    parser.add_argument("--sync-interval", type=int, default=32)
    parser.add_argument("--skew-every", type=int, default=1024)
    parser.add_argument("--skew-seconds", type=float, default=0.02)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--reference-directory", type=Path)
    parser.add_argument("--reference-cycle", type=int, default=0)
    parser.add_argument("--boundary-timing-replays", type=int, default=0)
    parser.add_argument("--candidate-timing-only", action="store_true")
    parser.add_argument("--interleave-hcl-context", action="store_true")
    args = parser.parse_args()
    if args.replays <= 0 or args.validation_replays <= 0 or args.sync_interval <= 0 or args.progress_interval < 0:
        parser.error("replay and synchronization counts must be positive")
    if not math.isfinite(args.skew_seconds) or args.skew_seconds < 0:
        parser.error("skew-seconds must be finite and nonnegative")

    dist.init_process_group("hccl")
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise RuntimeError("Native TP2 graph qualification requires exactly two ranks")
    warmup = torch.ones(128, dtype=torch.bfloat16, device="hpu")
    dist.all_reduce(warmup)
    torch.hpu.synchronize()

    bridge = _load_bridge(args.bridge)
    if not bridge.native_decode_graph_available():
        raise RuntimeError("Native Synapse/HCL graph symbols are unavailable")
    backend = dist.group.WORLD._get_backend(torch.device("hpu"))
    partial_cpu, residual_cpu, weight_cpu = _cpu_inputs(rank, 0)
    partial = partial_cpu.to("hpu")
    residual = residual_cpu.to("hpu")
    weight = weight_cpu.to("hpu")

    # Generate the same-math HPU oracle before capture. Reference collectives
    # must not be interleaved with replay of an already captured communicator.
    reference_path = (args.reference_directory or Path.cwd()) / f"rank{rank}-exact-hpu-reference.pt"
    if args.reference_cycle < 0 or (args.reference_cycle and args.reference_cycle < 4096):
        parser.error("A repeated reference set must contain at least 4096 changing inputs")
    reference_steps = min(args.replays, args.reference_cycle or args.replays)
    required_steps = 1 + args.validation_replays + reference_steps
    if args.reference_directory:
        manifest = json.loads((args.reference_directory / "exact-reference-manifest.json").read_text())
        if (manifest["epsilon"] != args.epsilon or manifest["steps"] < required_steps
                or manifest["input_generator"] != "mod127-113-v1"
                or manifest["sha256"][reference_path.name] != hashlib.sha256(reference_path.read_bytes()).hexdigest()):
            raise RuntimeError("Archived exact HPU reference contract or fingerprint differs")
        references = torch.load(reference_path, map_location="cpu", weights_only=True)
        if len(references) < required_steps:
            raise RuntimeError("Archived HPU reference is incomplete")
        print(json.dumps({
            "rank": rank,
            "reused_exact_reference_steps": required_steps,
            "reference": str(reference_path)
        }),
              flush=True)
    else:
        reference_probe = bridge.NativeTp2GraphProbe(backend, partial, residual, weight, args.epsilon)
        references = []
        for step in range(required_steps):
            for destination, source in zip((partial, residual, weight), _cpu_inputs(rank, step), strict=True):
                destination.copy_(source)
            reference_probe.reference()
            references.append(tuple(value.cpu() for value in reference_probe.outputs()))
            if args.progress_interval and (step + 1) % args.progress_interval == 0:
                print(json.dumps({"rank": rank, "exact_reference_steps": step + 1}), flush=True)
        reference_probe.close()
        torch.save(references, reference_path)
    for destination, source in zip((partial, residual, weight), _cpu_inputs(rank, 0), strict=True):
        destination.copy_(source)
    torch.hpu.synchronize()
    timing = {
        "ordinary_hpu": [],
        "native": [],
        "scope": "device_drained_norm_exchange_norm_wall",
        "candidate_only": args.candidate_timing_only,
        "inputs_drained_before_timer": True,
        "validation_sync_included": True,
        "timing_is_not_production_throughput": True
    }
    timing_count = args.boundary_timing_replays
    if timing_count < 0 or timing_count + 8 > len(references):
        raise RuntimeError("Timing count exceeds retained reference")
    timing_group = dist.new_group(backend="gloo") if timing_count else None
    if timing_count and not args.candidate_timing_only:
        oracle = bridge.NativeTp2GraphProbe(backend, partial, residual, weight, args.epsilon)
        for sample in range(timing_count + 8):
            for destination, source in zip((partial, residual, weight), _cpu_inputs(rank, sample), strict=True):
                destination.copy_(source)
            torch.hpu.synchronize()
            dist.barrier(group=timing_group)
            wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
            oracle.reference()
            elapsed_cpu, elapsed_wall = time.thread_time_ns() - cpu, time.perf_counter_ns() - wall
            assert_exact(tuple(value.cpu() for value in oracle.outputs()), references[sample], step=sample)
            if sample >= 8:
                timing["ordinary_hpu"].append({"wall_ns": elapsed_wall, "caller_cpu_ns": elapsed_cpu})
        oracle.close()
        for destination, source in zip((partial, residual, weight), _cpu_inputs(rank, 0), strict=True):
            destination.copy_(source)
        torch.hpu.synchronize()
    probe = bridge.NativeTp2GraphProbe(backend, partial, residual, weight, args.epsilon)
    probe.capture()
    capture_outputs = tuple(value.cpu() for value in probe.outputs())
    capture_errors = assert_exact(capture_outputs, references[0], step=0)
    cpu_diagnostic = [{
        "max_abs": float((actual.float() - expected.float()).abs().max())
    } for actual, expected in zip(capture_outputs, _reference(0, args.epsilon), strict=True)]

    validation_errors = []
    for step in range(1, args.validation_replays + 1):
        partial_cpu, residual_cpu, weight_cpu = _cpu_inputs(rank, step)
        partial.copy_(partial_cpu)
        residual.copy_(residual_cpu)
        weight.copy_(weight_cpu)
        probe.replay()
        probe.synchronize()
        actual = tuple(value.cpu() for value in probe.outputs())
        validation_errors.append(assert_exact(actual, references[step], step=step))

    context_interference = torch.empty_like(partial, dtype=torch.float32) if args.interleave_hcl_context else None
    snapshot_pool = [(torch.empty_like(partial), torch.empty_like(residual)) for _ in range(args.sync_interval)]
    async_producer = os.environ.get("TP2_PROBE_ASYNC_PRODUCER_STREAM", "0") == "1"
    producer = torch.hpu.Stream() if async_producer else None
    producer_scratch = (torch.empty(8 * 1024 * 1024, device="hpu", dtype=torch.bfloat16) if async_producer else None)
    pending = []
    checked_steps = 0
    caller_submit_ns = []
    caller_cpu_ns = []
    started = time.perf_counter()
    for replay in range(args.replays):
        step = 1 + args.validation_replays + replay % reference_steps
        if context_interference is not None:
            # Distinct WORLD communicator, dtype and NIC collective operation.
            # It must not change any decoder input or expected graph output.
            context_interference.fill_(rank + 1 + replay % 8)
            dist.all_reduce(context_interference)
            torch.hpu.synchronize()
            if float(context_interference.flatten()[0].cpu()) != 3 + 2 * (replay % 8):
                raise RuntimeError("Context interference control failed")
        partial_cpu, residual_cpu, weight_cpu = _cpu_inputs(rank, step)
        if async_producer:
            with torch.hpu.stream(producer):
                # Delay a real device producer; the graph must consume the
                # queued event through its logical COMPUTE stream job.
                if replay % 64 == 0:
                    producer_scratch.fill_(replay % 17)
                partial.copy_(partial_cpu, non_blocking=True)
                residual.copy_(residual_cpu, non_blocking=True)
                weight.copy_(weight_cpu, non_blocking=True)
        else:
            partial.copy_(partial_cpu)
            residual.copy_(residual_cpu)
            weight.copy_(weight_cpu)
        if rank == 1 and args.skew_every > 0 and replay > 0 and replay % args.skew_every == 0:
            time.sleep(args.skew_seconds)
        submit_start, cpu_start = time.perf_counter_ns(), time.thread_time_ns()
        probe.replay()
        caller_cpu_ns.append(time.thread_time_ns() - cpu_start)
        caller_submit_ns.append(time.perf_counter_ns() - submit_start)
        normalized, residual_out = probe.outputs()
        snapshot = snapshot_pool[len(pending)]
        snapshot[0].copy_(normalized)
        snapshot[1].copy_(residual_out)
        pending.append((step, snapshot))
        if len(pending) == args.sync_interval or replay + 1 == args.replays:
            probe.synchronize()
            checked_steps += validate_snapshots(pending, references)
            pending.clear()
            if args.progress_interval and (replay + 1) % args.progress_interval == 0:
                print(json.dumps({"rank": rank, "completed_stress_replays": replay + 1}), flush=True)
    elapsed = time.perf_counter() - started
    if checked_steps != args.replays:
        raise RuntimeError(f"Only {checked_steps}/{args.replays} replay outputs were checked")

    if timing_count:
        for sample in range(timing_count + 8):
            for destination, source in zip((partial, residual, weight), _cpu_inputs(rank, sample), strict=True):
                destination.copy_(source)
            torch.hpu.synchronize()
            dist.barrier(group=timing_group)
            wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
            probe.replay()
            probe.synchronize()
            elapsed_cpu, elapsed_wall = time.thread_time_ns() - cpu, time.perf_counter_ns() - wall
            assert_exact(tuple(value.cpu() for value in probe.outputs()), references[sample], step=sample)
            execution_wall, execution_cpu = probe.last_execution_timing()
            if sample >= 8:
                timing["native"].append({
                    "wall_ns": elapsed_wall,
                    "caller_cpu_ns": elapsed_cpu,
                    "execute_wall_ns": execution_wall,
                    "execute_cpu_ns": execution_cpu
                })
        dist.destroy_process_group(timing_group)

    info = tuple(probe.info())
    expected_replays = args.validation_replays + args.replays + (timing_count + 8 if timing_count else 0)
    if (info[0] != 1 or info[1] != expected_replays or info[2] != 2 or info[3] == 0 or info[4] == 0 or info[5] == 0
            or info[6] == 0 or info[7] == 0 or info[8] == 0 or info[9] == 0 or info[10] == 0 or info[12] == 0
            or info[13] < info[3] or info[14] < 2):
        raise RuntimeError(f"Native graph structure mismatch: info={info}, expected_replays={expected_replays}")
    uses_batch = os.environ.get("HCL_TP2_NATIVE_BATCH", "0") == "1"
    uses_joint = os.environ.get("VLLM_HPU_TP2_NATIVE_JOINT_PLAN", "0") == "1"
    joint_info = tuple(probe.joint_info())
    if uses_joint and (joint_info[0] != expected_replays or joint_info[11] != expected_replays or joint_info[8] != 1
                       or joint_info[6] == 0):
        raise RuntimeError(f"Joint plan execution counters mismatch: {joint_info}")
    if info[16] != (expected_replays if uses_batch else 0):
        raise RuntimeError(f"Native batch invocation count differs: {info[16]}")
    wrapped_hcl_ccb = info[11] > 0
    if args.replays >= 4096 and not wrapped_hcl_ccb:
        raise RuntimeError(
            f"Stress run did not cross an actual HCL CCB boundary: native_bytes={info[10]} ccb={info[9]}")
    if expected_replays >= info[14] and info[15] == 0:
        raise RuntimeError("Native stress did not execute a completion-ring boundary template")
    loaded = sorted({
        line.rsplit(maxsplit=1)[-1]
        for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
        if any(name in line for name in ("libSynapse.so", "libhcl.so", "libhabana_pytorch_backend"))
    })
    result = {
        "rank": rank,
        "boundary_timing": timing,
        "module_id": MODULE_ID,
        "device_id": DEVICE_ID,
        "capture_errors": capture_errors,
        "reference": {
            "path": str(reference_path),
            "kind": "same_recipe_ordinary_dedicated_hpu",
            "reused": args.reference_directory is not None,
            "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "steps": len(references),
            "cpu_diagnostic_only": cpu_diagnostic
        },
        "validation_errors": validation_errors,
        "stress_replays": args.replays,
        "different_reference_inputs": reference_steps,
        "interleaved_hcl_context_changes": args.replays if args.interleave_hcl_context else 0,
        "native_batch_calls": info[16],
        "per_node_hcl_stage_calls": 0 if uses_batch else expected_replays,
        "native_compute_segment_calls": 0 if uses_joint else 2 * expected_replays,
        "joint_plan_info": joint_info,
        "synapse_joint_replays": joint_info[0],
        "compute_scal_submissions": joint_info[6],
        "compute_command_pages_per_replay": joint_info[1],
        "compute_completion_targets": {
            "first": info[17],
            "last": info[18],
            "crossed_15bit_carry": (info[18] >> 15) > (info[17] >> 15)
        },
        "delayed_consumer_checks": checked_steps,
        "rank_skew_events": (args.replays - 1) // args.skew_every if rank == 1 and args.skew_every else 0,
        "elapsed_seconds": elapsed,
        "diagnostic_replays_per_second": args.replays / elapsed,
        "elapsed_includes_input_updates_snapshots_checks_and_injected_skew": True,
        "caller_submit_ns": caller_submit_ns,
        "caller_cpu_ns": caller_cpu_ns,
        "caller_timer_excludes_execute_thread_work": True,
        "native_info": info,
        "program_memory": {
            "global_bytes": info[5],
            "arc_bytes": info[6]
        },
        "hcl_command_bytes": {
            "total_per_replay": info[7],
            "max_stream_per_replay": info[8],
            "stream_ccb_bytes": info[9],
            "actual_native_replay_bytes": info[10],
            "actual_ccb_wrap_count": info[11],
            "actual_submission_count": info[12],
            "wrap_template_commands": info[13],
            "completion_ring_size": info[14],
            "completion_ring_wrap_replays": info[15],
            "wrapped_during_stress": wrapped_hcl_ccb,
        },
        "loaded_libraries": loaded,
        "passed": True,
        "exact_outputs_passed": True,
        "full_producer_state_chain_qualified": False,
        "scope": "stateless_same_recipe_compute_exchange_compute; no MME/state qualification yet",
    }
    # stdout from two ranks can concatenate large JSON writes. Preserve one
    # complete file per rank before emitting a compact status line.
    result_path = Path.cwd() / f"rank{rank}-result.json"
    pending_path = result_path.with_suffix(".json.pending")
    pending_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    pending_path.replace(result_path)
    print(json.dumps({"rank": rank, "passed": True, "result_path": str(result_path), "native_info": info}), flush=True)
    probe.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
