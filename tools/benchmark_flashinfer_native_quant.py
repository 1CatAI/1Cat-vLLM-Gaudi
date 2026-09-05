# SPDX-License-Identifier: Apache-2.0
"""Qualify native gated FP8 quantization against both compiled HPU reference paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import uuid


def write_report(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def audit_native_kernels(kernels):
    expected = "flashinfer_gaudi_silu_mul_quant_bf16_gaudi2"
    vendor = [name for name in kernels if name != expected]
    if (expected not in kernels or len(kernels) != 2 or len(vendor) != 1
            or not (vendor[0] == "silu_fwd_bf16"
                    or re.fullmatch(r"fused_kernel_0x[0-9A-Fa-f]+_[0-9A-Fa-f]+_bf16", vendor[0]))):
        raise RuntimeError(f"Unexpected native kernels: {kernels}")


def audit_native_trace(events, replays=5):
    kernels = sorted({event["name"] for event in events if event.get("cat") == "kernel"})
    audit_native_kernels(kernels)
    cpu_ops = sorted({event["name"] for event in events if event.get("cat") == "cpu_op"})
    forbidden = [
        name for name in cpu_ops
        if name.startswith(("aten::", "hpu::")) and name not in ("aten::empty", "aten::empty_strided")
    ]
    launches = sum(event.get("cat") == "privateuse1_runtime" and event.get("name") == "Launch" for event in events)
    if forbidden or launches != replays:
        raise RuntimeError(f"Native trace decomposition/launch audit failed: {forbidden}, launches={launches}")
    return {"kernels": kernels, "cpu_ops": cpu_ops, "launch_events": launches}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="1,8,32,256")
    parser.add_argument("--widths", default="2048,3584,4096,5120,17408")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--waves", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.batches = [int(value) for value in args.batches.split(",")]
    args.widths = [int(value) for value in args.widths.split(",")]
    if (not args.batches or not args.widths or min(args.batches) < 1
            or any(width < 256 or width > 17408 or width % 128 for width in args.widths)):
        parser.error("Require B>0 and widths in [256,17408], divisible by 128")
    if min(args.sessions, args.iterations, args.warmups) < 1 or args.waves < 15:
        parser.error("Require positive counts and at least 15 paired waves")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def audit_fx(graph):
    calls = [node for node in graph.graph.nodes if node.op not in ("placeholder", "output")]
    if (not calls or str(calls[0].target) != "custom_op.flashinfer_gaudi_silu_mul_quant"
            or any(node.op != "call_function" or node.target is not operator.getitem or node.args[0] is not calls[0]
                   or node.args[1] not in (0, 1) for node in calls[1:])):
        raise RuntimeError(f"Unexpected native FX computation: {[str(node.target) for node in calls]}")
    return [str(node.target) for node in calls]


def worker(args):
    import torch
    import habana_frameworks.torch  # noqa: F401
    from flashinfer_gaudi import load_native_extensions, set_backend_policy, silu_and_mul_quant
    from flashinfer_gaudi.quantization import _silu_and_mul_quant_reference
    from flashinfer_gaudi._bridge import load_bridge_adapter

    def cguid(x):
        gate, up = x.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
        scale = torch.ops.hpu.calculate_scale_for_cast(activated, 2, 0, -1, True, 240., 1.)
        scale = scale + (1e-8 / 240.)
        output = torch.ops.hpu.cast_to_fp8_v2(activated, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
        return output, scale.float()

    libraries = load_native_extensions()
    bridge_identity = load_bridge_adapter()
    if not libraries:
        raise RuntimeError("Native extension is not loaded")
    set_backend_policy("native")
    graphs = []
    hpu_backend = torch._dynamo.lookup_backend("hpu_backend")

    def audited_backend(graph, inputs):
        graphs.append(audit_fx(graph))
        return hpu_backend(graph, inputs)

    functions = {
        "ordinary": torch.compile(_silu_and_mul_quant_reference, backend="hpu_backend", fullgraph=True, dynamic=False),
        "cguid": torch.compile(cguid, backend="hpu_backend", fullgraph=True, dynamic=False),
        "native": torch.compile(silu_and_mul_quant, backend=audited_backend, fullgraph=True, dynamic=False),
    }
    paths = [
        *map(Path, libraries),
        Path(__file__).resolve().parents[1] / "flashinfer_gaudi/lib/flashinfer_gaudi_bridge_ops.so",
        Path(__file__).resolve().parents[1] / "flashinfer_gaudi/lib/libflashinfer_gaudi_kernels.so"
    ]
    report = {
        "schema_version": 1,
        "session_id": str(uuid.uuid4()),
        "pid": os.getpid(),
        "device_module": os.environ.get("HLS_MODULE_ID"),
        "torch_version": torch.__version__,
        "library_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "measurement": "paired_device_queue_waves_including_submission_gaps",
        "bridge_artifact": bridge_identity,
        "cases": {},
        "native_fx_graphs": graphs,
        "production_promoted": False
    }
    torch._dynamo.config.cache_size_limit = max(128, len(args.batches) * len(args.widths) + 8)

    def validate(actual, expected):
        q, scale = (value.cpu() for value in actual)
        reference, expected_scale = (value.cpu() for value in expected)
        torch.testing.assert_close(scale, expected_scale, rtol=.02, atol=1e-8)
        torch.testing.assert_close(q.float() * scale, reference.float() * expected_scale, rtol=.08, atol=.02)
        return (q.view(torch.uint8) == reference.view(torch.uint8)).float().mean().item()

    first = None
    for batch in args.batches:
        for width in args.widths:
            key = f"{batch}x{width}"
            case = {
                "correctness": False,
                "whole_operation_native": True,
                "equal_fraction": {},
                **{
                    label + suffix: []
                    for label in functions
                    for suffix in ("_ms", "_host_ms")
                }
            }
            report["cases"][key] = case
            try:
                generator = torch.Generator().manual_seed(31 + batch + width)
                cpu = torch.randn((batch, 2 * width), generator=generator, dtype=torch.bfloat16)
                x = cpu.to("hpu")
                for magnitude in (.25, 1., 8.):
                    sample = (cpu.float() * magnitude).to(torch.bfloat16).to("hpu")
                    actual = functions["native"](sample)
                    for label in ("ordinary", "cguid"):
                        expected = functions[label](sample)
                        torch.hpu.synchronize()
                        case["equal_fraction"][f"{label}:{magnitude}"] = validate(actual, expected)
                    torch.testing.assert_close(sample.cpu(), (cpu.float() * magnitude).to(torch.bfloat16),
                                               rtol=0,
                                               atol=0)
                case["correctness"] = True
                for _ in range(args.warmups):
                    for fn in functions.values():
                        fn(x)
                torch.hpu.synchronize()

                def wave(fn, sample=x):
                    start, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
                    start.record()
                    started = time.perf_counter_ns()
                    for _ in range(args.iterations):
                        fn(sample)
                    end.record()
                    torch.hpu.synchronize()
                    return start.elapsed_time(end) / args.iterations, (time.perf_counter_ns() -
                                                                       started) / 1e6 / args.iterations

                labels = tuple(functions)
                for index in range(args.waves):
                    order = labels[index % 3:] + labels[:index % 3]
                    if index % 2:
                        order = tuple(reversed(order))
                    for label in order:
                        device_ms, host_ms = wave(functions[label])
                        case[label + "_ms"].append(device_ms)
                        case[label + "_host_ms"].append(host_ms)
                if args.trace:
                    case["traces"] = {}
                    for label, fn in functions.items():
                        path = args.output.with_name(f"{args.output.stem}-{key}-{label}-trace.json")
                        with torch.profiler.profile(
                                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU
                                            ]) as profiler:
                            with torch.profiler.record_function(label + "_silu_quant"):
                                for _ in range(5):
                                    fn(x)
                            torch.hpu.synchronize()
                        profiler.export_chrome_trace(str(path))
                        events = json.loads(path.read_text())["traceEvents"]
                        kernels = sorted({event["name"] for event in events if event.get("cat") == "kernel"})
                        audit = audit_native_trace(events) if label == "native" else {"kernels": kernels}
                        case["traces"][label] = {"path": path.name, **audit}
                if first is None:
                    first = (x, functions["ordinary"](x))
            except Exception as exc:
                case.update(correctness=False, error=f"{type(exc).__name__}: {exc}")
            write_report(args.output, report)
    try:
        x, expected = first
        actual = functions["native"](x)
        torch.hpu.synchronize()
        validate(actual, expected)
        report["cross_shape_reentry"] = True
    except Exception as exc:
        report.update(cross_shape_reentry=False, reentry_error=str(exc))
    write_report(args.output, report)


def main():
    args = arguments()
    if args.worker:
        worker(args)
        return
    from flashinfer_gaudi._qualification import qualify

    result = {"schema_version": 1, "production_promoted": False, "cases": {}, "worker_errors": []}
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
            result["worker_errors"].append({
                "session": number,
                "returncode": completed.returncode,
                "log": output.with_suffix(".log").name
            })
        else:
            sessions.append(json.loads(output.read_text()))
        write_report(args.output, result)
    result["sessions"] = [session["session_id"] for session in sessions]
    if len({json.dumps(session["library_sha256"], sort_keys=True) for session in sessions}) > 1:
        result["worker_errors"].append({"error": "Native artifacts changed between sessions"})
    if not result["worker_errors"]:
        for key in sessions[0]["cases"]:
            correct = all(session["cases"][key]["correctness"] and session["cross_shape_reentry"]
                          for session in sessions)
            gates = {}
            for label in ("ordinary", "cguid"):
                gates[label] = qualify([{
                    "session_id": session["session_id"],
                    "reference_ms": session["cases"][key][label + "_ms"],
                    "candidate_ms": session["cases"][key]["native_ms"]
                } for session in sessions],
                                       correctness=correct,
                                       whole_operation_native=True)
            result["cases"][key] = {
                "correctness": correct,
                "qualified": all(gate["qualified"] for gate in gates.values()),
                "baseline_gates": gates
            }
    write_report(args.output, result)
    print(json.dumps(result, indent=2))
    if result["worker_errors"] or not all(case["correctness"] for case in result["cases"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
