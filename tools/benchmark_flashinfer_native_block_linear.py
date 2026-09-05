# SPDX-License-Identifier: Apache-2.0
"""Qualify exact-scale TPC/BF16-MME linear against compiled block-weight dequant/GEMM."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2) + "\n")


def audit_fx(graph):
    calls = [node for node in graph.graph.nodes if node.op not in ("placeholder", "output")]
    targets = [str(node.target) for node in calls]
    if len(calls) != 1 or calls[0].op != "call_function" or targets != ["custom_op.flashinfer_gaudi_block_fp8_linear"]:
        raise RuntimeError(f"Unexpected native block-FP8 graph: {targets}")
    return targets


def audit_trace(events):
    cpu = sorted({e["name"] for e in events if e.get("cat") == "cpu_op"})
    kernels = sorted({e["name"] for e in events if e.get("cat") == "kernel"})
    expected = {"gemm", "flashinfer_gaudi_block_fp8_dequant_gaudi2"}
    forbidden = [
        name for name in cpu
        if name.startswith(("aten::", "hpu::")) and name not in ("aten::empty", "aten::empty_strided")
    ]
    launches = sum(e.get("cat") == "privateuse1_runtime" and e.get("name") == "Launch" for e in events)
    if forbidden or set(name.lower() for name in kernels) != expected or launches != 5:
        raise RuntimeError(f"Native block-FP8 trace audit failed: CPU={cpu}, kernels={kernels}, launches={launches}")
    return {"cpu_ops": cpu, "kernels": kernels, "launch_events": launches}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="1,8,32,256,1024")
    parser.add_argument("--weights",
                        default="34816x5120,17408x5120,5120x17408,5120x8704",
                        help="Comma-separated NxK weights; defaults cover MLP TP1/TP2 local shapes")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--waves", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        args.batches = [int(x) for x in args.batches.split(",")]
        args.weights = [tuple(int(x) for x in shape.split("x")) for shape in args.weights.split(",")]
        valid = (all(0 < m <= 2**31 - 1 for m in args.batches) and all(
            len(shape) == 2 and all(0 < d <= 2**31 - 1 and d % 128 == 0 for d in shape) for shape in args.weights))
    except ValueError:
        valid = False
    if not valid or min(args.sessions, args.iterations, args.warmups) < 1 or args.waves < 15:
        parser.error("Require positive counts, >=15 waves and aligned 128x128 NxK weights")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def worker(args):
    import torch
    import habana_frameworks.torch  # noqa: F401
    from flashinfer_gaudi import block_fp8_dequant, block_fp8_linear, set_backend_policy
    from flashinfer_gaudi._bridge import load_bridge_adapter, runtime_files, sha256
    from vllm_gaudi.extension.ops import dequant_block_fp8_weight_naive

    def reference(x, weight, scale):
        return torch.nn.functional.linear(x, dequant_block_fp8_weight_naive(weight, scale, [128, 128]))

    def bf16_ceiling(x, weight):
        return torch.nn.functional.linear(x, weight)

    identity = load_bridge_adapter()
    set_backend_policy("native")
    graphs = []
    hpu_backend = torch._dynamo.lookup_backend("hpu_backend")

    def audited_backend(graph, inputs):
        graphs.append(audit_fx(graph))
        return hpu_backend(graph, inputs)

    native = torch.compile(block_fp8_linear, backend=audited_backend, fullgraph=True, dynamic=False)
    baseline = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    ceiling = torch.compile(bf16_ceiling, backend="hpu_backend", fullgraph=True, dynamic=False)
    dequant = torch.compile(block_fp8_dequant, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch._dynamo.config.cache_size_limit = 128
    directory = Path(__file__).resolve().parents[1] / "flashinfer_gaudi/lib"
    measured_files = {
        **runtime_files(directory), "runner": Path(__file__),
        "reference_source": Path(sys.modules[dequant_block_fp8_weight_naive.__module__].__file__)
    }
    report = {
        "schema_version": 1,
        "session_id": str(uuid.uuid4()),
        "pid": os.getpid(),
        "device_module": os.environ.get("HLS_MODULE_ID"),
        "artifact": identity,
        "library_sha256": {
            name: sha256(path)
            for name, path in measured_files.items()
        },
        "native_fx_graphs": graphs,
        "cases": {},
        "production_promoted": False,
        "measurement": "paired_device_queue_waves_including_submission_gaps",
        "baseline": "compiled existing block-weight BF16 cast/multiply then BF16 GEMM",
        "ceiling": "predequantized BF16 weight; different persistent-memory contract, diagnostic only"
    }
    first = None
    for n, k in args.weights:
        generator = torch.Generator().manual_seed(n + k)
        cpu_weight = torch.randn(n, k, generator=generator).clamp(-240, 240).to(torch.float8_e4m3fn)
        cpu_scale = (torch.rand(n // 128, k // 128, generator=generator) * .7 + .1) / k**.5
        weight, scale = cpu_weight.to("hpu"), cpu_scale.to("hpu")
        for m in args.batches:
            key = f"{m}x{n}x{k}"
            case = {
                "correctness": False,
                **{
                    label + suffix: []
                    for label in ("reference", "native", "ceiling")
                    for suffix in ("_ms", "_host_ms")
                }
            }
            report["cases"][key] = case
            try:
                cpu_x = torch.randn(m, k, generator=generator).bfloat16()
                x = cpu_x.to("hpu")
                for factor in (1., .25, 8.):
                    scale.copy_(cpu_scale * factor)
                    expected_weight = dequant_block_fp8_weight_naive(weight, scale, [128, 128])
                    native_weight = dequant(weight, scale)
                    expected, actual = baseline(x, weight, scale), native(x, weight, scale)
                    torch.hpu.synchronize()
                    torch.testing.assert_close(native_weight.cpu(), expected_weight.cpu(), rtol=0, atol=0)
                    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=.02, atol=.002)
                scale.copy_(cpu_scale)
                dequantized = dequant_block_fp8_weight_naive(weight, scale, [128, 128])
                expected, actual = baseline(x, weight, scale), native(x, weight, scale)
                torch.hpu.synchronize()
                torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=.02, atol=.002)
                torch.testing.assert_close(x.cpu(), cpu_x, rtol=0, atol=0)
                assert torch.equal(weight.cpu().view(torch.uint8), cpu_weight.view(torch.uint8))
                torch.testing.assert_close(scale.cpu(), cpu_scale, rtol=0, atol=0)
                case["correctness"] = True
                functions = {
                    "reference": (baseline, (x, weight, scale)),
                    "native": (native, (x, weight, scale)),
                    "ceiling": (ceiling, (x, dequantized))
                }
                for _ in range(args.warmups):
                    for fn, tensors in functions.values():
                        fn(*tensors)
                torch.hpu.synchronize()

                def wave(fn, tensors):
                    begin, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
                    begin.record()
                    start = time.perf_counter_ns()
                    for _ in range(args.iterations):
                        fn(*tensors)
                    end.record()
                    torch.hpu.synchronize()
                    return begin.elapsed_time(end) / args.iterations, (time.perf_counter_ns() -
                                                                       start) / 1e6 / args.iterations

                labels = tuple(functions)
                for index in range(args.waves):
                    order = labels[index % 3:] + labels[:index % 3]
                    if index % 2:
                        order = tuple(reversed(order))
                    for label in order:
                        device_ms, host_ms = wave(*functions[label])
                        case[label + "_ms"].append(device_ms)
                        case[label + "_host_ms"].append(host_ms)
                if args.trace:
                    case["traces"] = {}
                    for label, (fn, tensors) in functions.items():
                        path = args.output.with_name(f"{args.output.stem}-{key}-{label}-trace.json")
                        with torch.profiler.profile(
                                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU
                                            ]) as profiler:
                            for _ in range(5):
                                fn(*tensors)
                            torch.hpu.synchronize()
                        profiler.export_chrome_trace(str(path))
                        events = json.loads(path.read_text())["traceEvents"]
                        audit = audit_trace(events) if label == "native" else {
                            "kernels": sorted({e["name"]
                                               for e in events if e.get("cat") == "kernel"})
                        }
                        case["traces"][label] = {"path": path.name, **audit}
                if first is None:
                    first = (x, weight, scale, expected.cpu())
            except Exception as exc:
                case.update(correctness=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            write_report(args.output, report)
    try:
        x, weight, scale, expected = first
        actual = native(x, weight, scale)
        torch.hpu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=.02, atol=.002)
        report["cross_shape_reentry"] = True
    except Exception as exc:
        report.update(cross_shape_reentry=False, reentry_error=str(exc))
    if report["library_sha256"] != {name: sha256(path) for name, path in measured_files.items()}:
        report.update(cross_shape_reentry=False, reentry_error="Sources or artifacts changed during measurement")
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
            str(output), "--batches", ",".join(map(str, args.batches)), "--weights",
            ",".join(f"{n}x{k}" for n, k in args.weights), "--waves",
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
    result["sessions"] = [s["session_id"] for s in sessions]
    if len({json.dumps(s["library_sha256"], sort_keys=True) for s in sessions}) > 1:
        result["worker_errors"].append({"error": "Native artifacts changed between sessions"})
    if not result["worker_errors"]:
        for key in sessions[0]["cases"]:
            result["cases"][key] = qualify([{
                "session_id": s["session_id"],
                "reference_ms": s["cases"][key]["reference_ms"],
                "candidate_ms": s["cases"][key]["native_ms"]
            } for s in sessions],
                                           correctness=all(s["cases"][key]["correctness"] and s["cross_shape_reentry"]
                                                           for s in sessions),
                                           whole_operation_native=True)
    write_report(args.output, result)
    print(json.dumps(result, indent=2))
    if result["worker_errors"] or not all(c["correctness"] for c in result["cases"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
