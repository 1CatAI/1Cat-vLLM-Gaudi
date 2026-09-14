#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate H3's real-shape FP8 linear and packed-attention HPU paths."""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm_gaudi.extension import ops as hpu_ops
from vllm_gaudi.omni.attention import HPUSDPAImpl
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
)


def _tensor_path(component_dir: Path, name: str) -> Path:
    index_path = component_dir / "diffusion_pytorch_model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    return component_dir / weight_map[name]


def _load_tensor(component_dir: Path, name: str) -> torch.Tensor:
    path = _tensor_path(component_dir, name)
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _device_wave(function, iterations: int = 1) -> tuple[float, float]:
    begin = torch.hpu.Event(enable_timing=True)
    end = torch.hpu.Event(enable_timing=True)
    begin.record()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    end.record()
    torch.hpu.synchronize()
    return begin.elapsed_time(end) / iterations, (time.perf_counter_ns() - started) / 1e6 / iterations


def _stats(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    candidate = candidate.float()
    reference = reference.float()
    difference = candidate - reference
    reference_norm = torch.linalg.vector_norm(reference)
    relative_l2 = torch.linalg.vector_norm(difference) / reference_norm.clamp_min(1e-12)
    max_abs_reference = reference.abs().max()
    result = {
        "finite": bool(torch.isfinite(candidate).all().cpu()),
        "max_abs_error": float(difference.abs().max().cpu()),
        "max_abs_reference": float(max_abs_reference.cpu()),
        "max_error_over_reference_peak": float((difference.abs().max() / max_abs_reference.clamp_min(1e-12)).cpu()),
        "mean_abs_error": float(difference.abs().mean().cpu()),
        "relative_l2": float(relative_l2.cpu()),
    }
    return result


def _quantized_activation_reference(
    value: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Replay FP8 GEMM arithmetic from the exact quantized operands.

    Comparing a dynamic-activation FP8 GEMM directly with an unquantized BF16
    GEMM mixes activation quantization drift with the MME implementation error.
    Multiplying dequantized BF16 operands introduces another rounding step, so
    this reference multiplies the exactly representable FP8 values first and
    applies the row/channel inverse scales to the accumulator afterwards.
    """
    value_fp8, value_scale = hpu_ops.dynamic_quant(value)
    unscaled = F.linear(value_fp8.float(), weight.t().float())
    return (unscaled.float() * value_scale.float() * weight_scale.float()).to(value.dtype)


def _native_mlp_chain(
    value: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    projected = hpu_ops.apply_fp8_linear_hpu(
        input=value,
        weight=weight,
        weight_scale=weight_scale,
        input_scale=None,
        trans_B=False,
    )
    gate, up = projected.chunk(2, dim=-1)
    return F.silu(gate) * up


def _reference_mlp_chain(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gate, up = F.linear(value, weight).chunk(2, dim=-1)
    return F.silu(gate) * up


def _profile_native_chain(
    output_dir: Path,
    value: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> dict[str, Any]:
    trace_path = output_dir / "native-fp8-chain.trace.json"
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
            record_shapes=True,
    ) as profiler:
        _native_mlp_chain(value, weight, weight_scale)
        torch.hpu.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text(encoding="utf-8"))["traceEvents"]
    selected = [{
        "name": event.get("name"),
        "cat": event.get("cat"),
        "args": event.get("args", {})
    } for event in events
                if any(token in str(event.get("name", "")).lower() for token in ("fp8", "gemm", "cast", "silu"))]
    selected_path = output_dir / "native-fp8-events.json"
    selected_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    names = {str(event["name"]) for event in selected}
    has_native_op = any("fp8_gemm_v2" in name for name in names)
    has_fp8_kernel = any("gemm" in name.lower() and ("f8" in name.lower() or "fp8" in name.lower()) for name in names)
    return {
        "trace": trace_path.name,
        "selected_events": selected_path.name,
        "event_names": sorted(names),
        "hpu_fp8_gemm_v2_observed": has_native_op,
        "fp8_gemm_kernel_observed": has_fp8_kernel,
    }


def _run_fp8_linear(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    component = args.checkpoint / "transformer"
    prefix = "transformer_blocks.0.ff.net.0.proj"
    source_weight = _load_tensor(component, prefix + ".weight")
    source_scale = _load_tensor(component, prefix + ".weight_scale")
    if source_weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected E4M3 checkpoint weight, got {source_weight.dtype}")
    if source_scale.dtype != torch.float32 or source_scale.shape != (source_weight.shape[0], ):
        raise ValueError("checkpoint does not carry one FP32 scale per fc1 output channel")

    # ModelOpt emits E4M3FN values with the wider CUDA range. Gaudi2 uses the
    # OCP E4M3 range; divide encoded values by two and multiply inverse scales
    # by two, preserving the represented weight while retaining FP8 storage.
    converted_weight = (source_weight.float() * 0.5).to(torch.float8_e4m3fn)
    converted_scale = source_scale * 2.0
    reference_weight = (converted_weight.float() * converted_scale[:, None]).to(torch.bfloat16)
    weight = converted_weight.t().contiguous().to("hpu")
    weight_scale = converted_scale.contiguous().to("hpu")
    reference_weight = reference_weight.to("hpu")
    del source_weight, source_scale, converted_weight, converted_scale

    if weight.dtype != torch.float8_e4m3fn or weight_scale.dtype != torch.float32:
        raise AssertionError("native operand dtypes changed during HPU transfer")

    generator = torch.Generator().manual_seed(args.seed)
    correctness: list[dict[str, Any]] = []
    for label, rows, k_width, n_width, multiplier in (
        ("zero", 1, weight.shape[0], weight.shape[1], 0.0),
        ("extreme", 3, weight.shape[0], weight.shape[1], 256.0),
        ("non_block", 65, weight.shape[0] - 1, weight.shape[1] - 2, 1.0),
        ("changing", 64, weight.shape[0], weight.shape[1], 1.0),
    ):
        cpu = torch.randn(rows, k_width, generator=generator, dtype=torch.bfloat16) * multiplier
        value = cpu.to("hpu")
        local_weight = weight[:k_width, :n_width].contiguous()
        local_scale = weight_scale[:n_width].contiguous()
        local_reference_weight = reference_weight[:n_width, :k_width].contiguous()
        actual = hpu_ops.apply_fp8_linear_hpu(
            input=value,
            weight=local_weight,
            weight_scale=local_scale,
            input_scale=None,
            trans_B=False,
        )
        expected = _quantized_activation_reference(value, local_weight, local_scale)
        unquantized = F.linear(value, local_reference_weight)
        torch.hpu.synchronize()
        case = {
            "case": label,
            "shape": [rows, k_width, n_width],
            "quantized_operands_reference": _stats(actual, expected),
            "drift_from_unquantized_bf16": _stats(actual, unquantized),
        }
        quantized_stats = case["quantized_operands_reference"]
        peak_tolerance = 0.5 + 0.008 * quantized_stats["max_abs_reference"]
        if (quantized_stats["relative_l2"] > 0.005 or quantized_stats["max_abs_error"] > peak_tolerance
                or not quantized_stats["finite"]):
            raise AssertionError(f"FP8 linear correctness failed: {case}")
        correctness.append(case)
        del cpu, value, local_weight, local_scale, local_reference_weight, actual, expected, unquantized

    rows = args.rows
    value = torch.randn(rows, weight.shape[0], generator=generator, dtype=torch.bfloat16).to("hpu")
    changed = (value * 0.75 + 0.125).to(torch.bfloat16)
    sample_rows = min(64, rows)
    actual_sample = _native_mlp_chain(value[:sample_rows], weight, weight_scale)
    quantized_projection = _quantized_activation_reference(value[:sample_rows], weight, weight_scale)
    quantized_gate, quantized_up = quantized_projection.chunk(2, dim=-1)
    expected_sample = F.silu(quantized_gate) * quantized_up
    unquantized_sample = _reference_mlp_chain(value[:sample_rows], reference_weight)
    torch.hpu.synchronize()
    chain_correctness = {
        "quantized_operands_reference": _stats(actual_sample, expected_sample),
        "drift_from_unquantized_bf16": _stats(actual_sample, unquantized_sample),
    }
    quantized_chain_stats = chain_correctness["quantized_operands_reference"]
    if quantized_chain_stats["relative_l2"] > 0.01 or not quantized_chain_stats["finite"]:
        raise AssertionError(f"FP8 MLP chain correctness failed: {chain_correctness}")
    del actual_sample, expected_sample, unquantized_sample, quantized_projection

    native = lambda: _native_mlp_chain(value, weight, weight_scale)
    reference = lambda: _reference_mlp_chain(value, reference_weight)
    for _ in range(args.warmups):
        native()
        reference()
    torch.hpu.synchronize()
    samples: dict[str, list[dict[str, float]]] = {"native_fp8": [], "bf16_reference": []}
    functions = {"native_fp8": native, "bf16_reference": reference}
    for wave in range(args.waves):
        labels = ("native_fp8", "bf16_reference") if wave % 2 == 0 else ("bf16_reference", "native_fp8")
        for label in labels:
            device_ms, host_ms = _device_wave(functions[label], args.iterations)
            samples[label].append({"device_ms": device_ms, "synchronized_host_ms": host_ms})

    first_checksum = float(_native_mlp_chain(value, weight, weight_scale).float().mean().cpu())
    changed_checksum = float(_native_mlp_chain(changed, weight, weight_scale).float().mean().cpu())
    if first_checksum == changed_checksum:
        raise AssertionError("changing production input did not change the MLP result")
    execution = _profile_native_chain(output_dir, value, weight, weight_scale)
    if not execution["hpu_fp8_gemm_v2_observed"]:
        raise AssertionError("profiler did not observe hpu.fp8_gemm_v2")

    medians = {
        label: {
            "device_ms": statistics.median(item["device_ms"] for item in values),
            "synchronized_host_ms": statistics.median(item["synchronized_host_ms"] for item in values),
        }
        for label, values in samples.items()
    }
    return {
        "checkpoint_tensor": prefix,
        "production_shape": [rows, int(weight.shape[0]), int(weight.shape[1])],
        "operand_dtypes": {
            "activation": str(value.dtype),
            "weight": str(weight.dtype),
            "activation_scale": "torch.float32 (dynamic per row)",
            "weight_scale": str(weight_scale.dtype) + " per output channel",
            "output": "torch.bfloat16",
        },
        "gaudi2_range_conversion": "weight / 2, inverse scale * 2",
        "correctness_cases": correctness,
        "chain_correctness": chain_correctness,
        "changed_input_checksums": [first_checksum, changed_checksum],
        "samples": samples,
        "medians": medians,
        "native_over_bf16_device_ratio": medians["native_fp8"]["device_ms"] / medians["bf16_reference"]["device_ms"],
        "execution_evidence": execution,
    }


def _run_attention(args: argparse.Namespace) -> dict[str, Any]:
    implementation = HPUSDPAImpl(num_heads=56, head_size=128, softmax_scale=128**-0.5)
    results: list[dict[str, Any]] = []
    for label, length, used in (("suffix_padding", 2048, 2003), ("production", args.attention_rows, args.used_rows)):
        query = torch.randn(1, length, 56, 128, dtype=torch.bfloat16, device="hpu")
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        cu = torch.tensor([0, used], dtype=torch.int32, device="hpu")
        metadata = AttentionMetadata(packed_padding=PackedPaddingMetadata(
            q_length=used,
            kv_length=used,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
        ))
        function = functools.partial(implementation.forward, query, key, value, metadata)
        function()
        torch.hpu.synchronize()
        device_ms, host_ms = _device_wave(function)
        output = function()
        finite = bool(torch.isfinite(output[:, :used]).all().cpu())
        suffix_zero = bool((output[:, used:] == 0).all().cpu()) if used < length else True
        if not finite or not suffix_zero:
            raise AssertionError(f"HPU packed attention failed for {label}")
        results.append({
            "case": label,
            "shape_bshd": [1, length, 56, 128],
            "used_rows": used,
            "mask_tensor": None,
            "finite": finite,
            "suffix_zero": suffix_zero,
            "device_ms": device_ms,
            "synchronized_host_ms": host_ms,
        })
        del query, key, value, output, metadata
    return {"backend": "Habana FusedSDPA", "causal": False, "cases": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="local FL2VA partition")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=38272, help="aligned packed rows for 124-frame 1344x768 T2VA")
    parser.add_argument("--attention-rows", type=int, default=38272)
    parser.add_argument("--used-rows", type=int, default=38222)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "visible_modules": os.environ.get("HABANA_VISIBLE_MODULES"),
        "hls_module_id": os.environ.get("HLS_MODULE_ID"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_version": torch.__version__,
    }
    report_path = args.output_dir / "result.json"
    try:
        import habana_frameworks.torch  # noqa: F401

        torch.hpu.init()
        torch.hpu.reset_peak_memory_stats()
        report["fp8_mlp"] = _run_fp8_linear(args, args.output_dir)
        report["attention"] = _run_attention(args)
        free, total = torch.hpu.mem_get_info()
        report["memory"] = {
            "peak_allocated_bytes": int(torch.hpu.max_memory_allocated()),
            "free_bytes_after": int(free),
            "total_bytes": int(total),
        }
        report["status"] = "pass"
    except BaseException as exc:
        report.update(status="fail", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
