# SPDX-License-Identifier: Apache-2.0
"""Paired native GEMM-SiLU qualification against fullgraph compiled HPU math."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2) + "\n")


def audit_trace(path, mode):
    events = json.loads(path.read_text())["traceEvents"]
    cpu = sorted({event["name"] for event in events if event.get("cat") == "cpu_op"})
    kernels = sorted({event["name"] for event in events if event.get("cat") == "kernel"})
    forbidden = [name for name in cpu if name in ("aten::copy_", "aten::mm", "aten::matmul", "aten::silu", "aten::mul")]
    if mode == "out":
        forbidden.extend(name for name in cpu if name in ("aten::empty", "aten::empty_strided"))
    expected = {"gemm", "flashinfer_gaudi_silu_and_mul_bf16_gaudi2"}
    if forbidden or not expected.issubset(name.lower() for name in kernels):
        raise RuntimeError(f"Native trace audit failed: CPU={cpu}, kernels={kernels}")
    return {
        "cpu_ops":
        cpu,
        "kernel_names":
        kernels,
        "launch_events":
        sum(event.get("cat") == "privateuse1_runtime" and event.get("name") == "Launch" for event in events)
    }


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="1,8,32,256")
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--d", type=int, default=17408)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--waves", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.batches = [int(value) for value in args.batches.split(",")]
    if (not args.batches
            or min(*args.batches, args.k, args.d, args.sessions, args.waves, args.iterations, args.warmups) < 1
            or args.d % 128):
        parser.error("Require positive shapes/counts and D divisible by 128")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def worker(args):
    import torch
    import habana_frameworks.torch  # noqa: F401
    from flashinfer_gaudi._bridge import load_bridge_adapter
    from flashinfer_gaudi.gemm import GemmSiluPlan

    def reference(x, weight):
        gate, up = (x @ weight).chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    graphs = []
    hpu_backend = torch._dynamo.lookup_backend("hpu_backend")

    def audited_backend(graph, inputs):
        targets = [str(node.target) for node in graph.graph.nodes if node.op not in ("placeholder", "output")]
        if targets != ["custom_op.flashinfer_gaudi_gemm_silu"]:
            raise RuntimeError(f"Unexpected native computation: {targets}")
        graphs.append(targets)
        return hpu_backend(graph, inputs)

    report = {
        "schema_version": 1,
        "session_id": str(uuid.uuid4()),
        "pid": os.getpid(),
        "device_module": os.environ.get("HLS_MODULE_ID"),
        "artifact": load_bridge_adapter(),
        "cases": {},
        "native_fx_graphs": graphs,
        "measurement": "paired_device_queue_waves_including_submission_gaps",
        "baseline": "fullgraph compiled BF16 matmul, BF16 SiLU, BF16 mul; allocating output"
    }
    baseline = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch._dynamo.config.cache_size_limit = max(64, len(args.batches) * 4)
    first = None
    for m in args.batches:
        plan = GemmSiluPlan(m, args.k, args.d)
        native = torch.compile(plan.functional_op, backend=audited_backend, fullgraph=True, dynamic=False)
        generator = torch.Generator().manual_seed(31 + m)
        cpu_x = torch.randn(m, args.k, generator=generator).to(torch.bfloat16)
        cpu_w = (torch.randn(args.k, 2 * args.d, generator=generator) / args.k**.5).to(torch.bfloat16)
        x, weight = cpu_x.to("hpu"), cpu_w.to("hpu")
        out = torch.empty((m, args.d), dtype=x.dtype, device=x.device)
        pointer = out.data_ptr()

        def replay(a, b, selected_plan=plan, output=out):
            return selected_plan.run(a, b, out=output)

        for mode, candidate in (("functional", native), ("out", replay)):
            key = f"{m}x{args.k}x{args.d}:{mode}"
            case = {
                "correctness": False,
                "whole_operation_native": True,
                "reference_ms": [],
                "candidate_ms": [],
                "reference_host_ms": [],
                "candidate_host_ms": [],
                "graph_artifact": plan.artifact.to_json()
            }
            report["cases"][key] = case
            try:
                for sample in (x, (cpu_x.float() * 8).to(torch.bfloat16).to("hpu")):
                    expected, actual = baseline(sample, weight), candidate(sample, weight)
                    torch.hpu.synchronize()
                    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-3, rtol=2e-2)
                torch.testing.assert_close(x.cpu(), cpu_x, atol=0, rtol=0)
                torch.testing.assert_close(weight.cpu(), cpu_w, atol=0, rtol=0)
                if mode == "out":
                    assert actual is out and out.data_ptr() == pointer
                case["correctness"] = True
                for _ in range(args.warmups):
                    baseline(x, weight)
                    candidate(x, weight)
                torch.hpu.synchronize()

                def wave(fn, sample=x, matrix=weight):
                    start, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
                    start.record()
                    started = time.perf_counter_ns()
                    for _ in range(args.iterations):
                        fn(sample, matrix)
                    end.record()
                    torch.hpu.synchronize()
                    return start.elapsed_time(end) / args.iterations, (time.perf_counter_ns() -
                                                                       started) / 1e6 / args.iterations

                for index in range(args.waves):
                    order = (("reference", baseline), ("candidate", candidate))
                    for label, fn in (order if index % 2 == 0 else tuple(reversed(order))):
                        device_ms, host_ms = wave(fn)
                        case[f"{label}_ms"].append(device_ms)
                        case[f"{label}_host_ms"].append(host_ms)
                if args.trace:
                    trace = args.output.with_name(f"{args.output.stem}-{m}-{mode}-trace.json")
                    with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU
                                        ]) as profiler:
                        with torch.profiler.record_function(f"native_gemm_silu_{mode}"):
                            for _ in range(5):
                                candidate(x, weight)
                        torch.hpu.synchronize()
                    profiler.export_chrome_trace(str(trace))
                    case["trace"] = trace.name
                    case["trace_audit"] = audit_trace(trace, mode)
                    if mode == "functional":
                        reference_trace = trace.with_name(trace.stem + "-reference.json")
                        with torch.profiler.profile(
                                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU
                                            ]) as profiler:
                            with torch.profiler.record_function("compiled_reference_gemm_silu"):
                                for _ in range(5):
                                    baseline(x, weight)
                            torch.hpu.synchronize()
                        profiler.export_chrome_trace(str(reference_trace))
                        case["reference_trace"] = reference_trace.name
            except Exception as exc:
                case.update(correctness=False, error=f"{type(exc).__name__}: {exc}", reference_ms=[], candidate_ms=[])
            write_report(args.output, report)
        if first is None:
            first = (plan, native, x, weight, out, baseline(x, weight).cpu())
    try:
        plan, native, x, weight, out, expected = first
        plan.run(x, weight, out=out)
        actual = native(x, weight)
        torch.hpu.synchronize()
        for value in (out, actual):
            torch.testing.assert_close(value.cpu(), expected, atol=2e-3, rtol=2e-2)
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

    result = {"schema_version": 1, "production_promoted": False, "cases": {}, "sessions": [], "worker_errors": []}
    sessions = []
    for number in range(args.sessions):
        output = args.output.with_name(f"{args.output.stem}-session-{number}.json")
        command = [
            sys.executable,
            str(Path(__file__).resolve()), "--worker", "--output",
            str(output), "--batches", ",".join(map(str, args.batches)), "--k",
            str(args.k), "--d",
            str(args.d), "--waves",
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
        elif output.exists():
            sessions.append(json.loads(output.read_text()))
        write_report(args.output, result)
    result["sessions"] = [session["session_id"] for session in sessions]
    if len({session["artifact"]["artifact_id"] for session in sessions}) > 1:
        result["worker_errors"].append({"error": "Adapter artifact changed between process sessions"})
    if not result["worker_errors"]:
        for key in sessions[0]["cases"]:
            values = [session["cases"][key] for session in sessions]
            result["cases"][key] = qualify([{
                "session_id": session["session_id"],
                "reference_ms": value["reference_ms"],
                "candidate_ms": value["candidate_ms"]
            } for session, value in zip(sessions, values)],
                                           correctness=all(value["correctness"] for value in values)
                                           and all(session["cross_shape_reentry"] for session in sessions),
                                           whole_operation_native=True)
    write_report(args.output, result)
    print(json.dumps(result, indent=2))
    if result["worker_errors"] or not all(value["correctness"] for value in result["cases"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
