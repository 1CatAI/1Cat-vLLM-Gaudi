# SPDX-License-Identifier: Apache-2.0
"""Qualify complete residual/norm/row-FP8 fusion against compiled HPU baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def write_report(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def audit_fx(graph):
    calls = [node for node in graph.graph.nodes if node.op not in ("placeholder", "output")]
    if (not calls or str(calls[0].target) != "custom_op.flashinfer_gaudi_add_rmsnorm_quant"
            or any(node.op != "call_function" or node.target is not operator.getitem or node.args[0] is not calls[0]
                   or node.args[1] not in (0, 1, 2, 3) for node in calls[1:])):
        raise RuntimeError(f"Unexpected native FX computation: {[str(node.target) for node in calls]}")
    return [str(node.target) for node in calls]


def audit_trace(events, replays=5):
    kernels = sorted({event["name"] for event in events if event.get("cat") == "kernel"})
    cpu_ops = sorted({event["name"] for event in events if event.get("cat") == "cpu_op"})
    forbidden = [
        name for name in cpu_ops
        if name.startswith(("aten::", "hpu::")) and name not in ("aten::empty", "aten::empty_strided")
    ]
    launches = sum(event.get("cat") == "privateuse1_runtime" and event.get("name") == "Launch" for event in events)
    if kernels != ["flashinfer_gaudi_add_rmsnorm_quant_bf16_gaudi2"] or forbidden or launches != replays:
        raise RuntimeError(
            f"Norm native trace audit failed: kernels={kernels}, forbidden={forbidden}, launches={launches}")
    return {"kernels": kernels, "cpu_ops": cpu_ops, "launch_events": launches}


def qualify_case(sessions, key, expected_sessions, worker_errors):
    """Fail closed unless matching processes and the first device trace exist."""
    from flashinfer_gaudi._qualification import qualify
    correct = (len(sessions) == expected_sessions and not worker_errors and all(
        session.get("artifacts_unchanged") and session.get("cross_shape_reentry")
        and session.get("use_eager_fallback") is False and session.get("cases", {}).get(key, {}).get("correctness")
        for session in sessions) and len({json.dumps(session.get("sha256"), sort_keys=True)
                                          for session in sessions}) == 1)
    native = bool(correct and sessions and sessions[0].get("cases", {}).get(key, {}).get("whole_operation_native")
                  and sessions[0].get("cases", {}).get(key, {}).get("traces", {}).get("native")
                  and all(session.get("native_fx_graphs") for session in sessions))
    gates = {}
    for reference in ("formula", "vendor", "cguid"):
        pairs = [{
            "session_id": session["session_id"],
            "reference_ms": session.get("cases", {}).get(key, {}).get(reference + "_ms", []),
            "candidate_ms": session.get("cases", {}).get(key, {}).get("native_ms", [])
        } for session in sessions]
        gates[reference] = qualify(pairs, correctness=bool(correct), whole_operation_native=native)
    return {"qualified": all(gate["qualified"] for gate in gates.values()), "baselines": gates}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="1,8,32,256,2048")
    parser.add_argument("--widths", default="256,2048,5120,17408")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--waves", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.batches = [int(value) for value in args.batches.split(",")]
    args.widths = [int(value) for value in args.widths.split(",")]
    if (not args.batches or not args.widths or min(args.batches) < 1 or max(args.batches) > 2**31 - 1
            or any(width < 256 or width > 17408 or width % 128 for width in args.widths)):
        parser.error("Require B>0 and widths in [256,17408], divisible by 128")
    if min(args.sessions, args.iterations, args.warmups) < 1 or args.waves < 15:
        parser.error("Require positive counts and at least 15 paired waves")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def validate(actual, expected):
    import torch
    q, scale, normed, residual = (value.cpu() for value in actual)
    rq, rs, rn, rr = (value.cpu() for value in expected)
    assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert normed.dtype == residual.dtype == torch.bfloat16
    torch.testing.assert_close(residual, rr, rtol=0, atol=0)
    torch.testing.assert_close(normed, rn, rtol=.02, atol=.002)
    torch.testing.assert_close(scale, rs, rtol=.02, atol=1e-8)
    torch.testing.assert_close(q.float() * scale, rq.float() * rs, rtol=.08, atol=.02)
    return {
        "quant_equal_fraction": (q.view(torch.uint8) == rq.view(torch.uint8)).float().mean().item(),
        "norm_equal_fraction": (normed == rn).float().mean().item(),
        "norm_max_abs": (normed.float() - rn.float()).abs().max().item(),
    }


def worker(args):
    import torch
    import habana_frameworks.torch  # noqa: F401
    from habana_frameworks.torch.dynamo.compile_backend import config as backend_config
    from flashinfer_gaudi import fused_add_rmsnorm_quant, load_native_extensions, set_backend_policy
    from flashinfer_gaudi.norm import _reference

    backend_config.use_eager_fallback = False
    libraries = load_native_extensions()
    if not libraries:
        raise RuntimeError("Native extension is not loaded")
    set_backend_policy("native")
    torch.hpu.init()

    def vendor(x, residual, weight, cguid=False):
        summed = x + residual
        # The public HPEX wrapper calls this same vendor op. Use its supported
        # rank-2 form to avoid an unnecessary rank-3 view fallback in the baseline.
        normalized = torch.ops.hpu.rms_norm(summed, weight, 1e-6, None, False)[0]
        if cguid:
            scale = torch.ops.hpu.calculate_scale_for_cast(normalized, 2, 0, -1, True, 240., 1.)
            scale = scale + (1e-8 / 240.)
        else:
            scale = (normalized.abs().amax(-1, keepdim=True) + 1e-8) / 240.
        q = torch.ops.hpu.cast_to_fp8_v2(normalized, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
        return q, scale.float(), normalized, summed

    def formula(x, residual, weight):
        return _reference(x, residual, weight, 1e-6)

    def cguid(x, residual, weight):
        return vendor(x, residual, weight, True)

    graphs = []
    hpu_backend = torch._dynamo.lookup_backend("hpu_backend")

    def audited_backend(graph, inputs):
        graphs.append(audit_fx(graph))
        return hpu_backend(graph, inputs)

    functions = {
        name: torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
        for name, fn in (("formula", formula), ("vendor", vendor), ("cguid", cguid))
    }
    functions["native"] = torch.compile(fused_add_rmsnorm_quant, backend=audited_backend, fullgraph=True, dynamic=False)
    root = Path(__file__).resolve().parents[1]
    paths = [
        *map(Path, libraries), root / "flashinfer_gaudi/lib/libflashinfer_gaudi_kernels.so",
        root / "csrc/flashinfer_gaudi/kernels/add_rmsnorm_quant_bf16_gaudi2.c",
        root / "csrc/flashinfer_gaudi/kernels/add_rmsnorm_quant_bf16_gaudi2_small.c",
        root / "csrc/flashinfer_gaudi/host/add_rmsnorm_quant_bf16_gaudi2.cpp",
        root / "csrc/flashinfer_gaudi/host/add_rmsnorm_quant_bf16_gaudi2.hpp",
        root / "csrc/flashinfer_gaudi/norm_quant_params.h",
        root / "csrc/flashinfer_gaudi/pytorch/hpu_flashinfer_norm.cpp", root / "flashinfer_gaudi/norm.py",
        root / "flashinfer_gaudi/_native.py", root / "flashinfer_gaudi/_qualification.py",
        Path(__file__).resolve()
    ]

    def fingerprints():
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    report = {
        "schema_version": 1,
        "session_id": str(uuid.uuid4()),
        "pid": os.getpid(),
        "device_module": os.environ.get("HLS_MODULE_ID"),
        "torch_version": torch.__version__,
        "sha256": fingerprints(),
        "measurement": "paired_device_queue_waves_including_submission_gaps",
        "native_fx_graphs": graphs,
        "use_eager_fallback": backend_config.use_eager_fallback,
        "cases": {},
        "production_promoted": False
    }
    torch._dynamo.config.cache_size_limit = max(128, len(args.batches) * len(args.widths) * 4 + 8)
    first = None
    for batch in args.batches:
        for width in args.widths:
            key = f"{batch}x{width}"
            case = {
                "correctness": False,
                "whole_operation_native": False,
                "numerics": {},
                **{
                    label + suffix: []
                    for label in functions
                    for suffix in ("_ms", "_host_ms")
                }
            }
            report["cases"][key] = case
            try:
                generator = torch.Generator().manual_seed(911 + batch + width)
                source = torch.randn((batch, width), generator=generator, dtype=torch.bfloat16)
                residual_cpu = torch.randn(source.shape, generator=generator, dtype=torch.bfloat16)
                weight_cpu = (torch.randn(width, generator=generator) * .1 + 1.).bfloat16()
                inputs = tuple(value.to("hpu") for value in (source, residual_cpu, weight_cpu))
                for magnitude in (0., 1e-5, .25, 1., 8.):
                    samples = ((source.float() * magnitude).bfloat16().to("hpu"),
                               (residual_cpu.float() * magnitude).bfloat16().to("hpu"), inputs[2])
                    actual = functions["native"](*samples)
                    for label in ("formula", "vendor", "cguid"):
                        expected = functions[label](*samples)
                        torch.hpu.synchronize()
                        case["numerics"][f"{label}:{magnitude}"] = validate(actual, expected)
                    for value, cpu in zip(samples, ((source.float() * magnitude).bfloat16(),
                                                    (residual_cpu.float() * magnitude).bfloat16(), weight_cpu)):
                        torch.testing.assert_close(value.cpu(), cpu, rtol=0, atol=0)
                case["correctness"] = True
                for _ in range(args.warmups):
                    for fn in functions.values():
                        fn(*inputs)
                torch.hpu.synchronize()

                def wave(fn, samples=inputs):
                    start, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
                    start.record()
                    started = time.perf_counter_ns()
                    for _ in range(args.iterations):
                        fn(*samples)
                    end.record()
                    torch.hpu.synchronize()
                    return start.elapsed_time(end) / args.iterations, (time.perf_counter_ns() -
                                                                       started) / 1e6 / args.iterations

                labels = tuple(functions)
                for index in range(args.waves):
                    order = labels[index % len(labels):] + labels[:index % len(labels)]
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
                            for _ in range(5):
                                fn(*inputs)
                            torch.hpu.synchronize()
                        profiler.export_chrome_trace(str(path))
                        events = json.loads(path.read_text())["traceEvents"]
                        audit = audit_trace(events) if label == "native" else {
                            "kernels": sorted({event["name"]
                                               for event in events if event.get("cat") == "kernel"})
                        }
                        case["traces"][label] = {"path": path.name, **audit}
                        if label == "native":
                            case["whole_operation_native"] = True
                if first is None:
                    first = (inputs, tuple(value.cpu() for value in functions["vendor"](*inputs)))
            except Exception as exc:
                case.update(correctness=False, error=f"{type(exc).__name__}: {exc}")
            write_report(args.output, report)
            print(key, case.get("error", "correctness passed"), flush=True)
    try:
        if first is None:
            raise RuntimeError("No successful shape to revisit")
        inputs, expected = first
        validate(functions["native"](*inputs), expected)
        report["cross_shape_reentry"] = True
    except Exception as exc:
        report.update(cross_shape_reentry=False, reentry_error=str(exc))
    report["artifacts_unchanged"] = fingerprints() == report["sha256"]
    write_report(args.output, report)


def main():
    args = arguments()
    if args.worker:
        worker(args)
        return
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
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            result["worker_errors"].append({"session": number, "returncode": completed.returncode})
        if output.is_file():
            sessions.append(json.loads(output.read_text()))
    for batch in args.batches:
        for width in args.widths:
            key = f"{batch}x{width}"
            result["cases"][key] = qualify_case(sessions, key, args.sessions, result["worker_errors"])
    result["session_files"] = [f"{args.output.stem}-session-{i}.json" for i in range(args.sessions)]
    write_report(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
