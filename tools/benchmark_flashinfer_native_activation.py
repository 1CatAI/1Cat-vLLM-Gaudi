# SPDX-License-Identifier: Apache-2.0
"""Qualify the complete native SiLU operation, without changing dispatch defaults."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="1,8,32,256,1024")
    parser.add_argument("--widths", default="5120,17408")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--waves", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.batches = [int(value) for value in args.batches.split(",")]
    args.widths = [int(value) for value in args.widths.split(",")]
    if (not args.batches or not args.widths or min(args.batches) < 1 or min(args.widths) < 128
            or any(width % 128 for width in args.widths)):
        parser.error("Require positive batches and widths divisible by 128.")
    if min(args.sessions, args.waves, args.iterations, args.warmups) < 1:
        parser.error("All measurement counts must be positive.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def write_report(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def worker(args):
    import torch
    import torch.nn.functional as F
    import habana_frameworks.torch  # noqa: F401
    from flashinfer_gaudi import get_capabilities, load_native_extensions, set_backend_policy, silu_and_mul

    libraries = load_native_extensions()
    if not libraries:
        raise RuntimeError("No native library loaded.")
    kernel_library = Path(__file__).resolve().parents[1] / "flashinfer_gaudi/lib/libflashinfer_gaudi_kernels.so"
    artifact_paths = [*libraries, str(kernel_library)]
    set_backend_policy("native")
    hpu_backend = torch._dynamo.lookup_backend("hpu_backend")
    graphs = []

    def audited_backend(graph, inputs):
        calls = [node for node in graph.graph.nodes if node.op not in ("placeholder", "output")]
        targets = [str(node.target) for node in calls]
        if len(calls) != 1 or targets != ["custom_op.flashinfer_gaudi_silu_and_mul"]:
            raise RuntimeError(f"Native graph contains unexpected computation: {targets}")
        graphs.append(targets)
        return hpu_backend(graph, inputs)

    def reference(x):
        gate, value = x.chunk(2, dim=-1)
        return F.silu(gate) * value

    native = torch.compile(silu_and_mul, backend=audited_backend, fullgraph=True, dynamic=False)
    baseline = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch._dynamo.config.cache_size_limit = max(64, len(args.batches) * len(args.widths) + 4)
    report = {
        "schema_version": 1,
        "session_id": str(uuid.uuid4()),
        "pid": os.getpid(),
        "device_module": os.environ.get("HLS_MODULE_ID"),
        "capabilities": get_capabilities(),
        "torch_version": torch.__version__,
        "cases": {},
        "native_fx_graphs": graphs,
        "measurement": "paired_device_queue_waves_including_submission_gaps",
        "library_sha256": {
            Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in artifact_paths
        }
    }
    # Keep the same process for the entire shape sweep, then revisit its first
    # shape. Isolation alone would hide recipe cache/reentry regressions.
    first = None
    for batch in args.batches:
        for width in args.widths:
            key = f"{batch}x{width}"
            case = {
                "correctness": False,
                "whole_operation_native": True,
                "implementation": "flashinfer_gaudi_silu_and_mul_bf16_gaudi2",
                "reference_ms": [],
                "candidate_ms": [],
                "reference_host_ms": [],
                "candidate_host_ms": []
            }
            report["cases"][key] = case
            try:
                generator = torch.Generator().manual_seed(31 + batch + width)
                cpu = torch.randn(batch, 2 * width, generator=generator, dtype=torch.bfloat16)
                x = cpu.to("hpu")
                for sample in (x, (cpu.float() * 8).to(torch.bfloat16).to("hpu")):
                    expected, actual = baseline(sample), native(sample)
                    torch.hpu.synchronize()
                    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-3, rtol=2e-2)
                torch.testing.assert_close(x.cpu(), cpu, atol=0, rtol=0)
                case["correctness"] = True
                for _ in range(args.warmups):
                    baseline(x)
                    native(x)
                torch.hpu.synchronize()

                def wave(function, sample=x):
                    start, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
                    start.record()
                    started = time.perf_counter_ns()
                    for _ in range(args.iterations):
                        function(sample)
                    end.record()
                    torch.hpu.synchronize()
                    return start.elapsed_time(end) / args.iterations, (time.perf_counter_ns() -
                                                                       started) / 1e6 / args.iterations

                for wave_index in range(args.waves):
                    order = (("reference", baseline), ("candidate", native))
                    if wave_index % 2:
                        order = tuple(reversed(order))
                    for label, function in order:
                        device_ms, host_ms = wave(function)
                        case[f"{label}_ms"].append(device_ms)
                        case[f"{label}_host_ms"].append(host_ms)
                if first is None:
                    first = (x, baseline(x).cpu())
                if args.trace:
                    trace_path = args.output.with_name(f"{args.output.stem}-{key}-trace.json")
                    with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU
                                        ]) as profiler:
                        with torch.profiler.record_function("flashinfer_native_silu"):
                            for _ in range(5):
                                native(x)
                        torch.hpu.synchronize()
                    profiler.export_chrome_trace(str(trace_path))
                    case["trace"] = trace_path.name
            except Exception as exc:
                case["correctness"] = False
                case["error"] = f"{type(exc).__name__}: {exc}"
                case["reference_ms"], case["candidate_ms"] = [], []
            write_report(args.output, report)
    if first is not None:
        x, expected = first
        try:
            for _ in range(20):
                actual = native(x)
            torch.hpu.synchronize()
            torch.testing.assert_close(actual.cpu(), expected, atol=2e-3, rtol=2e-2)
            report["cross_shape_reentry"] = True
        except Exception as exc:
            report["cross_shape_reentry"] = False
            report["reentry_error"] = str(exc)
    write_report(args.output, report)


def main():
    args = arguments()
    if args.worker:
        worker(args)
        return
    from flashinfer_gaudi._qualification import qualify

    sessions = []
    for number in range(args.sessions):
        output = args.output.with_name(f"{args.output.stem}-session-{number}.json")
        command = [
            sys.executable,
            str(Path(__file__).resolve()), "--worker", "--output",
            str(output), "--batches", ",".join(map(str, args.batches)), "--widths", ",".join(map(str, args.widths)),
            "--waves",
            str(args.waves), "--iterations",
            str(args.iterations), "--warmups",
            str(args.warmups)
        ]
        if args.trace and number == 0:
            command.append("--trace")
        with output.with_suffix(".log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"Benchmark worker failed; inspect {output.with_suffix('.log')}")
        sessions.append(json.loads(output.read_text()))
    result = {
        "schema_version": 1,
        "production_promoted": False,
        "cases": {},
        "sessions": [session["session_id"] for session in sessions]
    }
    for key in sessions[0]["cases"]:
        values = [session["cases"][key] for session in sessions]
        paired = [{
            "session_id": session["session_id"],
            "reference_ms": value["reference_ms"],
            "candidate_ms": value["candidate_ms"]
        } for session, value in zip(sessions, values)]
        result["cases"][key] = qualify(paired,
                                       correctness=all(value["correctness"] for value in values)
                                       and all(session.get("cross_shape_reentry", False) for session in sessions),
                                       whole_operation_native=all(value["whole_operation_native"] for value in values))
    write_report(args.output, result)
    print(json.dumps(result, indent=2))
    if not all(case["correctness"] for case in result["cases"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
