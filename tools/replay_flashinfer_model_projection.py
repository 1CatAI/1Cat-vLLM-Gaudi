# SPDX-License-Identifier: Apache-2.0
"""Validate and time complete FP8 projections using captured model tensors.

This replays real checkpoint weights and activations, not a complete model.
The originating model's outputs are an additional independent numerical check.
No qualification result enables a model route automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

from tools.benchmark_flashinfer_native_norm import write_report
from tools.benchmark_flashinfer_native_projection import BASELINES, audit_trace, make_functions, qualify_case, validate


def worker(args):
    import torch
    import habana_frameworks.torch  # noqa: F401
    from habana_frameworks.torch.dynamo.compile_backend import config as backend_config
    from flashinfer_gaudi import load_native_extensions, set_backend_policy

    backend_config.use_eager_fallback = False
    libraries = load_native_extensions()
    if not libraries:
        raise RuntimeError("Native extension is not loaded")
    set_backend_policy("native")
    torch.hpu.init()
    torch._dynamo.config.cache_size_limit = 256
    graphs, case_key, shape_graphs = {}, [None], {}
    functions = make_functions("bf16_reciprocal", graphs, case_key)
    root = Path(__file__).resolve().parents[1]
    paths = [
        *map(Path, libraries), root / "flashinfer_gaudi/lib/libflashinfer_gaudi_kernels.so", *sorted(
            (root / "csrc/flashinfer_gaudi").rglob("*norm*.*")), root / "flashinfer_gaudi/norm.py",
        root / "flashinfer_gaudi/_native.py", root / "flashinfer_gaudi/_qualification.py",
        root / "vllm_gaudi/extension/ops.py", root / "vllm_gaudi/ops/hpu_layernorm.py",
        root / "tools/benchmark_flashinfer_native_projection.py", root / "tools/benchmark_flashinfer_native_norm.py",
        root / "tools/capture_flashinfer_model_projection.py",
        Path(__file__).resolve()
    ]
    fixtures = [args.capture / "report.json", args.capture / "inventory.json", *sorted(args.capture.glob("*.pt"))]

    def fingerprints():
        return {
            **{
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
            **{
                "capture/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in fixtures
            }
        }

    capture_report = json.loads((args.capture / "report.json").read_text())
    inventory = json.loads((args.capture / "inventory.json").read_text())
    report = {
        "schema_version": 1,
        "session_id": str(uuid.uuid4()),
        "pid": os.getpid(),
        "sha256": fingerprints(),
        "contract": {
            "device_module": os.environ.get("HLS_MODULE_ID"),
            "torch_version": torch.__version__,
            "scale_mode": "bf16_reciprocal",
            "model_config_sha256": capture_report["model_config_sha256"],
            "measurement": "paired device queue waves including submission gaps; captured model projections",
            "iterations": args.iterations,
            "waves": args.waves,
            "lazy_mode": os.environ.get("PT_HPU_LAZY_MODE")
        },
        "use_eager_fallback": False,
        "cases": {},
        "production_promoted": False,
        "model_route_enabled": False,
        "end_to_end_benchmark": False
    }
    for layer in inventory:
        if layer["epsilon"] != 1e-6:
            raise RuntimeError("The replay reference requires epsilon=1e-6")
        prefix = layer["prefix"]
        weights = torch.load(args.capture / f"{prefix}-weights.pt", map_location="cpu", weights_only=True)
        weight_inputs = tuple(weights[key].to("hpu") for key in ("gamma", "weight", "weight_scale"))
        saved = []
        for path in sorted(args.capture.glob(f"{prefix}-*.pt")):
            if path.name.endswith("-weights.pt"):
                continue
            key = path.stem
            case_key[0] = key
            tensors = torch.load(path, map_location="cpu", weights_only=True)
            case = {
                "correctness": False,
                "shape": [*tensors["x"].shape, weights["weight"].shape[0]],
                "phase": tensors["phase"],
                "baseline_correctness": dict.fromkeys(BASELINES, False),
                "reentry": {},
                "errors": {},
                "numerics": {},
                "samples_ms": {
                    name: []
                    for name in functions
                },
                "host_ms": {
                    name: []
                    for name in functions
                },
                "fixture_sha256": {
                    "activation": report["sha256"]["capture/" + path.name],
                    "weight": report["sha256"][f"capture/{prefix}-weights.pt"]
                }
            }
            report["cases"][key] = case
            try:
                inputs = tensors["x"].to("hpu"), tensors["residual"].to("hpu"), *weight_inputs
                actual = functions["native"](*inputs)
                shape = tuple(case["shape"])
                if key in graphs:
                    shape_graphs[shape] = graphs[key]
                case["native_fx"] = shape_graphs.get(shape, [])
                for label in BASELINES:
                    try:
                        case["numerics"][label] = validate(actual, functions[label](*inputs))
                        case["baseline_correctness"][label] = True
                    except AssertionError as exc:
                        case["errors"][label] = str(exc)
                case["correctness"] = all(case["baseline_correctness"].values())
                expected = tensors["projected"], tensors["summed"]
                # Eager capture is an extra oracle, not the compiled baseline.
                for label in ("native", "cguid", "vendor"):
                    try:
                        case["numerics"][label + "_vs_capture"] = validate(functions[label](*inputs), expected)
                        case[label + "_capture_compatible"] = True
                    except AssertionError as exc:
                        case[label + "_capture_compatible"] = False
                        case["errors"][label + "_vs_capture"] = str(exc)
                compatible = [label for label in BASELINES if case["baseline_correctness"][label]]
                if compatible:
                    labels = (*compatible, "native")
                    for _ in range(30):
                        for label in labels:
                            functions[label](*inputs)
                    torch.hpu.synchronize()
                    for index in range(args.waves):
                        order = labels[index % len(labels):] + labels[:index % len(labels)]
                        if index % 2:
                            order = tuple(reversed(order))
                        for label in order:
                            start, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
                            start.record()
                            started = time.perf_counter_ns()
                            for _ in range(args.iterations):
                                functions[label](*inputs)
                            end.record()
                            torch.hpu.synchronize()
                            case["samples_ms"][label].append(start.elapsed_time(end) / args.iterations)
                            case["host_ms"][label].append((time.perf_counter_ns() - started) / 1e6 / args.iterations)
                    case["median_ms"] = {
                        label: statistics.median(values)
                        for label, values in case["samples_ms"].items() if values
                    }
                    case["speedup"] = {
                        label: case["median_ms"][label] / case["median_ms"]["native"]
                        for label in compatible
                    }
                    if args.trace:
                        path = args.output.with_name(f"{args.output.stem}-{key}-trace.json")
                        with torch.profiler.profile(
                                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU
                                            ]) as profiler:
                            for _ in range(5):
                                functions["native"](*inputs)
                            torch.hpu.synchronize()
                        profiler.export_chrome_trace(str(path))
                        audit = audit_trace(json.loads(path.read_text())["traceEvents"])
                        case["traces"] = {"native": {"path": path.name, "fx_audited": bool(case["native_fx"]), **audit}}
                for value, expected in zip(
                        inputs,
                    (tensors["x"], tensors["residual"], weights["gamma"], weights["weight"], weights["weight_scale"])):
                    torch.testing.assert_close(value.cpu().view(torch.uint8),
                                               expected.view(torch.uint8),
                                               rtol=0,
                                               atol=0)
                case["inputs_unchanged"] = True
                saved.append((key, inputs))
            except Exception as exc:
                case.update(correctness=False, inputs_unchanged=False, error=f"{type(exc).__name__}: {exc}")
            write_report(args.output, report)
            print(key, case.get("error", case.get("speedup", case["baseline_correctness"])), flush=True)
        for key, inputs in reversed(saved):
            case = report["cases"][key]
            samples = inputs[0] * .375, inputs[1] * .375, *inputs[2:]
            for label in BASELINES:
                try:
                    validate(functions["native"](*samples), functions[label](*samples))
                    case["reentry"][label] = True
                except Exception as exc:
                    case["reentry"][label] = False
                    case["errors"][label + ":reentry"] = str(exc)
        del saved
    report["artifacts_unchanged"] = fingerprints() == report["sha256"]
    write_report(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--waves", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.sessions, args.iterations) < 1 or args.waves < 15:
        parser.error("Require positive counts and at least 15 paired waves")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    result = {
        "schema_version": 1,
        "production_promoted": False,
        "model_route_enabled": False,
        "end_to_end_benchmark": False,
        "cases": {},
        "worker_errors": []
    }
    sessions = []
    for index in range(args.sessions):
        output = args.output.with_name(f"{args.output.stem}-session-{index}.json")
        if output.exists():
            raise FileExistsError(f"Refusing to reuse previous evidence: {output}")
        command = [
            sys.executable,
            str(Path(__file__).resolve()), "--worker", "--capture",
            str(args.capture), "--output",
            str(output), "--waves",
            str(args.waves), "--iterations",
            str(args.iterations)
        ]
        if args.trace and index == 0:
            command.append("--trace")
        print(f"Starting captured model replay {index + 1}/{args.sessions}", flush=True)
        with output.with_suffix(".log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            result["worker_errors"].append({"session": index, "returncode": completed.returncode})
        if output.is_file():
            sessions.append(json.loads(output.read_text()))
    keys = sorted(set().union(*(session.get("cases", {}).keys() for session in sessions)))
    for key in keys:
        gate = qualify_case(sessions, key, args.sessions, result["worker_errors"])
        gate["native_capture_compatible"] = all(
            session.get("cases", {}).get(key, {}).get("native_capture_compatible") for session in sessions)
        gate["model_route_enabled"] = False
        result["cases"][key] = gate
    result["session_files"] = [f"{args.output.stem}-session-{i}.json" for i in range(args.sessions)]
    write_report(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
