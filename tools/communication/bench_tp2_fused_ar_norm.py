#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and time the TP2 all-reduce/residual/RMSNorm prototype."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


def _pin_torchrun_rank() -> tuple[int | None, int | None]:
    visible_modules = os.environ.get("HABANA_VISIBLE_MODULES", "")
    if "," not in visible_modules:
        return None, None
    module_ids = [int(value) for value in visible_modules.split(",")]
    if len(module_ids) != 2:
        raise RuntimeError("HABANA_VISIBLE_MODULES must contain exactly two module IDs")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    module_to_device = {}
    for module_path in Path("/sys/class/accel").glob("accel*/device/module_id"):
        module_id = int(module_path.read_text(encoding="ascii").strip())
        device_id = int(module_path.parents[1].name.removeprefix("accel"))
        module_to_device[module_id] = device_id
    module_id = module_ids[local_rank]
    if module_id not in module_to_device:
        raise RuntimeError(f"Unknown Gaudi module ID {module_id}")
    os.environ["HLS_MODULE_ID"] = str(module_id)
    return module_id, module_to_device[module_id]


MODULE_ID, DEVICE_ID = _pin_torchrun_rank()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import habana_frameworks.torch.core as htcore  # noqa: E402, F401
from tp2_validation import validate_outputs  # noqa: E402


def _load_bridge(path: Path):
    spec = importlib.util.spec_from_file_location("tp2_fused_ar_norm_bridge", path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load TP2 fused bridge: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cpu_inputs(elements: int, hidden_size: int, rank: int, replay: int):
    index = torch.arange(elements, dtype=torch.int64)
    partial = (((index * 17 + rank * 29 + replay * 7) % 127) - 63).to(torch.float32) / 64
    residual = (((index * 11 + replay * 5) % 113) - 56).to(torch.float32) / 32
    weight_index = torch.arange(hidden_size, dtype=torch.int64)
    weight = 1 + (((weight_index * 3) % 31) - 15).to(torch.float32) / 128
    shape = (elements // hidden_size, hidden_size)
    return (
        partial.reshape(shape).to(torch.bfloat16),
        residual.reshape(shape).to(torch.bfloat16),
        weight.to(torch.bfloat16),
    )


def _reference(
    elements: int,
    hidden_size: int,
    replay: int,
    calls: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    partial0, residual, weight = _cpu_inputs(elements, hidden_size, 0, replay)
    partial1, _, _ = _cpu_inputs(elements, hidden_size, 1, replay)
    partial = partial0.float() + partial1.float()
    residual = residual.float()
    weight = weight.float()
    for _ in range(calls):
        residual = (partial + residual).to(torch.bfloat16).float()
        normalized = residual * torch.rsqrt(residual.square().mean(-1, keepdim=True) + epsilon) * weight
        partial = (normalized.to(torch.bfloat16).float()) * 2
    return normalized.to(torch.bfloat16), residual.to(torch.bfloat16)


def _allocate_outputs(shape: tuple[int, ...], calls: int):
    return [(
        torch.empty(shape, dtype=torch.bfloat16, device="hpu"),
        torch.empty(shape, dtype=torch.bfloat16, device="hpu"),
        torch.empty(shape, dtype=torch.bfloat16, device="hpu"),
        torch.empty((shape[0], 1), dtype=torch.float32, device="hpu"),
    ) for _ in range(calls)]


def _run_direct_chain(bridge, backend, partial, residual, weight, outputs, epsilon):
    for reduced, normalized, residual_out, inverse_rms in outputs:
        normalized, residual_out = bridge.allreduce_residual_rms_norm_current_stream(
            backend,
            partial,
            residual,
            weight,
            reduced,
            normalized,
            residual_out,
            inverse_rms,
            epsilon,
        )
        partial = normalized
        residual = residual_out
    return normalized, residual_out


class _CustomOpChain(torch.nn.Module):

    def __init__(self, calls: int, epsilon: float, out_variant: bool):
        super().__init__()
        self.calls = calls
        self.epsilon = epsilon
        self.out_variant = out_variant

    def forward(self, partial, residual, weight):
        for _ in range(self.calls):
            if self.out_variant:
                reduced = torch.empty_like(partial)
                normalized = torch.empty_like(partial)
                residual_out = torch.empty_like(partial)
                inverse_rms = torch.empty(
                    (*partial.shape[:-1], 1),
                    dtype=torch.float32,
                    device=partial.device,
                )
                torch.ops.vllm_gaudi.tp2_allreduce_residual_rms_norm_out(
                    partial,
                    residual,
                    weight,
                    reduced,
                    normalized,
                    residual_out,
                    inverse_rms,
                    self.epsilon,
                )
                partial, residual = normalized, residual_out
            else:
                partial, residual = torch.ops.vllm_gaudi.tp2_allreduce_residual_rms_norm(
                    partial, residual, weight, self.epsilon)
        return partial, residual


class _NativeCollectiveChain(torch.nn.Module):

    def __init__(self, calls: int, epsilon: float, communicator_id: int):
        super().__init__()
        self.calls = calls
        self.epsilon = epsilon
        self.communicator_id = communicator_id

    def forward(self, partial, residual, weight):
        outputs = None
        for _ in range(self.calls):
            packed, inverse_rms = torch.ops.hccl.tp2_allreduce_residual_rms_norm(partial, residual, weight,
                                                                                 self.epsilon, self.communicator_id)
            partial, residual = packed[0], packed[1]
            outputs = partial, residual, packed, inverse_rms
        assert outputs is not None
        return outputs


class _BaselineChain(torch.nn.Module):

    def __init__(self, calls: int, epsilon: float):
        super().__init__()
        self.calls = calls
        self.epsilon = epsilon

    def forward(self, partial, residual, weight):
        from habana_frameworks.torch.hpex.normalization import FusedRMSNorm

        partial = partial.clone()
        for _ in range(self.calls):
            dist.all_reduce(partial)
            residual = residual + partial
            partial = FusedRMSNorm.apply(residual.unsqueeze(0), weight, self.epsilon).squeeze(0)
        return partial, residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=5120)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=(
            "direct",
            "custom-op",
            "custom-op-out",
            "compile",
            "compile-out",
            "native",
            "compile-native",
            "baseline",
            "compile-baseline",
        ),
        default="direct",
    )
    parser.add_argument("--validation-replays", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--debug-values", action="store_true")
    parser.add_argument("--validation-atol", type=float, default=0.015625)
    parser.add_argument("--validation-rtol", type=float, default=0.015625)
    args = parser.parse_args()
    if args.elements <= 0 or args.elements % args.hidden_size:
        parser.error("elements must be a positive multiple of hidden-size")
    if args.calls <= 0:
        parser.error("calls must be positive")
    if args.validation_replays <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("validation-replays and iterations must be positive; warmup must be nonnegative")
    if not all(math.isfinite(value) and value >= 0 for value in (args.validation_atol, args.validation_rtol)):
        parser.error("validation tolerances must be finite and nonnegative")

    dist.init_process_group("hccl")
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise RuntimeError("This benchmark requires exactly two ranks")
    warmup_collective = torch.ones(128, dtype=torch.bfloat16, device="hpu")
    dist.all_reduce(warmup_collective)
    torch.hpu.synchronize()
    bridge = _load_bridge(args.bridge)
    backend = dist.group.WORLD._get_backend(torch.device("hpu"))
    shape = (args.elements // args.hidden_size, args.hidden_size)
    outputs = _allocate_outputs(shape, args.calls)
    if args.mode in ("baseline", "compile-baseline"):
        run_chain = _BaselineChain(args.calls, args.epsilon)
        if args.mode == "compile-baseline":
            from vllm_gaudi.utils import HPUCompileConfig

            compile_config = HPUCompileConfig(fullgraph=True, dynamic=False)
            run_chain = torch.compile(run_chain, **compile_config.get_compile_args())
    elif args.mode == "direct":

        def run_chain(partial, residual, weight):
            return _run_direct_chain(
                bridge,
                backend,
                partial,
                residual,
                weight,
                outputs,
                args.epsilon,
            )

    elif args.mode in ("native", "compile-native"):
        run_chain = _NativeCollectiveChain(args.calls, args.epsilon, bridge.communicator_id(backend))
        if args.mode == "compile-native":
            from vllm_gaudi.utils import HPUCompileConfig

            compile_config = HPUCompileConfig(fullgraph=True, dynamic=False)
            run_chain = torch.compile(run_chain, **compile_config.get_compile_args())
    else:
        import vllm_gaudi.distributed.tp2_fused_ar_norm as fused_module

        setattr(torch, fused_module._RUNTIME_ATTR, (bridge, backend))
        run_chain = _CustomOpChain(args.calls, args.epsilon, args.mode.endswith("-out"))
        if args.mode.startswith("compile"):
            from vllm_gaudi.utils import HPUCompileConfig

            compile_config = HPUCompileConfig(fullgraph=True, dynamic=False)
            run_chain = torch.compile(run_chain, **compile_config.get_compile_args())
    weight = None
    partial = None
    residual = None

    max_norm_error = 0.0
    max_residual_error = 0.0
    for replay in range(args.validation_replays):
        partial_cpu, residual_cpu, weight_cpu = _cpu_inputs(args.elements, args.hidden_size, rank, replay)
        if partial is None:
            partial = partial_cpu.to("hpu")
            residual = residual_cpu.to("hpu")
            weight = weight_cpu.to("hpu")
        else:
            partial.copy_(partial_cpu)
            residual.copy_(residual_cpu)
            weight.copy_(weight_cpu)
        launch_count_before = bridge.collective_launch_count()
        chain_result = run_chain(partial, residual, weight)
        normalized, residual_out = chain_result[:2]
        torch.hpu.synchronize()
        expected_norm, expected_residual = _reference(args.elements, args.hidden_size, replay, args.calls, args.epsilon)
        valid, errors = validate_outputs(
            (normalized.cpu(), residual_out.cpu()), (expected_norm, expected_residual),
            atol=args.validation_atol, rtol=args.validation_rtol,
            launches=bridge.collective_launch_count() - launch_count_before,
            expected_launches=args.calls if args.mode in ("native", "compile-native") else None)
        failed = torch.tensor([int(not valid)], dtype=torch.int32, device="hpu")
        dist.all_reduce(failed)
        if failed.cpu().item():
            dist.destroy_process_group()
            raise RuntimeError(f"TP2 validation failed before timing: replay={replay}, errors={errors}")
        if args.debug_values and replay == 0:
            print(
                json.dumps({
                    "rank":
                    rank,
                    "collective_debug_counts":
                    bridge.collective_debug_counts(),
                    "actual_norm":
                    normalized.cpu().flatten()[:8].float().tolist(),
                    "expected_norm":
                    expected_norm.flatten()[:8].float().tolist(),
                    "actual_residual":
                    residual_out.cpu().flatten()[:8].float().tolist(),
                    "expected_residual":
                    expected_residual.flatten()[:8].float().tolist(),
                    "native_output_2":
                    chain_result[2].cpu().flatten()[:8].float().tolist() if args.mode in ("native",
                                                                                          "compile-native") else None,
                    "native_output_3":
                    chain_result[3].cpu().flatten()[:8].float().tolist() if args.mode in ("native",
                                                                                          "compile-native") else None,
                }),
                flush=True,
            )
        max_norm_error = max(
            max_norm_error,
            (normalized.cpu().float() - expected_norm.float()).abs().max().item(),
        )
        max_residual_error = max(
            max_residual_error,
            (residual_out.cpu().float() - expected_residual.float()).abs().max().item(),
        )

    assert partial is not None and residual is not None and weight is not None
    for _ in range(args.warmup):
        chain_result = run_chain(partial, residual, weight)
    torch.hpu.synchronize()
    samples_us = []
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        chain_result = run_chain(partial, residual, weight)
        torch.hpu.synchronize()
        samples_us.append((time.perf_counter_ns() - start) / 1_000)

    result = {
        "rank": rank,
        "module_id": MODULE_ID,
        "device_id": DEVICE_ID,
        "elements": args.elements,
        "bytes": args.elements * 2,
        "calls": args.calls,
        "mode": args.mode,
        "validation_replays": args.validation_replays,
        "validation_passed": True,
        "validation_atol": args.validation_atol,
        "validation_rtol": args.validation_rtol,
        "max_norm_error": max_norm_error,
        "max_residual_error": max_residual_error,
        "host_sync_median_us": statistics.median(samples_us),
        "host_sync_p95_us": sorted(samples_us)[max(0,
                                                   int(len(samples_us) * 0.95) - 1)],
        "per_call_median_us": statistics.median(samples_us) / args.calls,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
