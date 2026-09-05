# SPDX-License-Identifier: Apache-2.0
"""Cross-process qualification of residual/norm/row-FP8 plus a complete FP8 GEMM.

The ordinary and CGUID baselines preserve the serving dynamic_quant expressions.
Their numerical contracts are checked independently, never silently exchanged.
Reports are offline evidence, not a production routing allowlist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

from tools.benchmark_flashinfer_native_norm import write_report

BASELINES = ("formula", "vendor", "cguid")
NORM = "custom_op.flashinfer_gaudi_add_rmsnorm_quant"
GEMM = "hpu.fp8_gemm_v2"


def target_name(node):
    return str(node.target).removesuffix(".default")


def audit_fx(graph):
    """Accept exactly the native producer and its public FP8 MME consumer."""
    nodes = [node for node in graph.graph.nodes if node.op not in ("placeholder", "output")]
    norms = [node for node in nodes if node.op == "call_function" and target_name(node) == NORM]
    gemms = [node for node in nodes if node.op == "call_function" and target_name(node) == GEMM]
    if len(norms) != 1 or len(gemms) != 1:
        raise RuntimeError("Projection FX must contain exactly one native norm and one FP8 GEMM")
    norm, gemm = norms[0], gemms[0]
    extracts = {}
    for node in nodes:
        if node in (norm, gemm):
            continue
        if (node.op != "call_function" or node.target is not operator.getitem or len(node.args) != 2
                or node.args[0] is not norm or node.args[1] not in (0, 1, 2, 3)):
            raise RuntimeError(f"Unexpected projection FX computation: {node.op} {node.target}")
        extracts.setdefault(node.args[1], []).append(node)
    if len(gemm.args) != 10 or gemm.kwargs or gemm.args[0] not in extracts.get(0,
                                                                               ()) or gemm.args[6] not in extracts.get(
                                                                                   1, ()):
        raise RuntimeError("FP8 GEMM must consume the native FP8 values and row scales directly")
    outputs = next(node for node in graph.graph.nodes if node.op == "output").args[0]
    if len(outputs) != 2 or outputs[0] is not gemm or outputs[1] not in extracts.get(3, ()):
        raise RuntimeError("Projection must return the GEMM result and functional residual")
    return [str(node.target) for node in nodes]


def audit_trace(events, replays=5):
    kernels = sorted({event["name"] for event in events if event.get("cat") == "kernel"})
    required = {"flashinfer_gaudi_add_rmsnorm_quant_bf16_gaudi2", "GEMM"}
    extras = set(kernels) - required
    cpu_ops = sorted({event["name"] for event in events if event.get("cat") == "cpu_op"})
    forbidden = [
        name for name in cpu_ops
        if name.startswith(("aten::", "hpu::")) and name not in ("aten::empty", "aten::empty_strided")
    ]
    launches = sum(event.get("cat") == "privateuse1_runtime" and event.get("name") == "Launch" for event in events)
    if (not required.issubset(kernels) or any(not name.startswith("fused_kernel_") or not name.endswith("_f32")
                                              for name in extras) or forbidden or launches != replays):
        raise RuntimeError(f"Projection trace audit failed: {kernels=}, {forbidden=}, {launches=}")
    # The public vendor GEMM can lower to an additional FP32 scaling kernel.
    # Count and report it; the complete projection is not a single TPC kernel.
    return {
        "kernels": kernels,
        "vendor_generated_kernels": sorted(extras),
        "cpu_ops": cpu_ops,
        "launch_events": launches,
        "replays": replays
    }


def qualify_case(sessions, key, expected_sessions, worker_errors):
    from flashinfer_gaudi._qualification import qualify

    common = bool(
        len(sessions) == expected_sessions and not worker_errors and sessions and all(
            session.get("artifacts_unchanged") and session.get("use_eager_fallback") is False
            and session.get("cases", {}).get(key, {}).get("inputs_unchanged")
            and session.get("cases", {}).get(key, {}).get("native_fx") for session in sessions)
        and len({session.get("session_id")
                 for session in sessions}) == expected_sessions and len({session.get("pid")
                                                                         for session in sessions}) == expected_sessions
        and len({json.dumps(session.get("sha256"), sort_keys=True)
                 for session in sessions}) == 1
        and len({json.dumps(session.get("contract"), sort_keys=True)
                 for session in sessions}) == 1 and len({
                     json.dumps(session.get("cases", {}).get(key, {}).get("fixture_sha256"), sort_keys=True)
                     for session in sessions
                 }) == 1)
    trace = sessions[0].get("cases", {}).get(key, {}).get("traces", {}).get("native", {}) if sessions else {}
    native = bool(common and trace.get("launch_events") == trace.get("replays") == 5 and trace.get("fx_audited"))
    gates = {}
    for label in BASELINES:
        correct = bool(common and all(session["cases"][key].get("baseline_correctness", {}).get(label)
                                      and session["cases"][key].get("reentry", {}).get(label) for session in sessions))
        pairs = [{
            "session_id": session["session_id"],
            "reference_ms": session.get("cases", {}).get(key, {}).get("samples_ms", {}).get(label, []),
            "candidate_ms": session.get("cases", {}).get(key, {}).get("samples_ms", {}).get("native", [])
        } for session in sessions]
        try:
            gates[label] = qualify(pairs, correctness=correct, whole_operation_native=native)
        except (KeyError, TypeError, ValueError) as exc:
            gates[label] = {"qualified": False, "reason": "invalid_session_evidence", "error": str(exc)}
    return {
        "qualified": all(gate["qualified"] for gate in gates.values()),
        "baselines": gates,
        "production_promoted": False
    }


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="1,8,32,256,2048")
    parser.add_argument("--projections", default="10240,34816")
    parser.add_argument("--width", type=int, default=5120)
    parser.add_argument("--scale-mode", choices=("bf16_reciprocal", "fp32_divide"), default="bf16_reciprocal")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--waves", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.batches = [int(value) for value in args.batches.split(",")]
    args.projections = [int(value) for value in args.projections.split(",")]
    if (not args.batches or not args.projections or min(*args.batches, *args.projections) < 1
            or len(set(args.batches)) != len(args.batches) or len(set(args.projections)) != len(args.projections)
            or not 256 <= args.width <= 17408 or args.width % 128):
        parser.error("Require unique positive batches/projections and width in [256,17408], divisible by 128")
    if min(args.sessions, args.iterations, args.warmups) < 1 or args.waves < 15:
        parser.error("Require positive counts and at least 15 paired waves")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def validate(actual, expected):
    import torch
    actual, residual = (value.cpu() for value in actual)
    expected, reference_residual = (value.cpu() for value in expected)
    assert actual.dtype == expected.dtype == residual.dtype == reference_residual.dtype == torch.bfloat16
    torch.testing.assert_close(residual, reference_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
    return {
        "equal_fraction": (actual == expected).float().mean().item(),
        "max_abs": (actual.float() - expected.float()).abs().max().item()
    }


def make_functions(scale_mode, graphs, case_key):
    import torch
    from flashinfer_gaudi import fused_add_rmsnorm_quant
    from flashinfer_gaudi.norm import _reference

    def project(q, scale, weight, weight_scale):
        return torch.ops.hpu.fp8_gemm_v2(q, False, weight, True, None, torch.bfloat16, scale, weight_scale, None, False)

    def native(x, residual, gamma, weight, weight_scale):
        q, scale, _, summed = fused_add_rmsnorm_quant(x, residual, gamma, scale_mode=scale_mode)
        return project(q, scale, weight, weight_scale), summed

    def formula(x, residual, gamma, weight, weight_scale):
        q, scale, _, summed = _reference(x, residual, gamma, 1e-6, scale_mode=scale_mode)
        return project(q, scale, weight, weight_scale), summed

    def vendor(x, residual, gamma, weight, weight_scale, use_cguid=False):
        summed = x + residual
        normed = torch.ops.hpu.rms_norm(summed, gamma, 1e-6, None, False)[0]
        if use_cguid:
            scale = torch.ops.hpu.calculate_scale_for_cast(normed, 2, 0, -1, True, 240., 1.) + (1e-8 / 240.)
        else:
            # Match serving's max(dim).values and reciprocal expression exactly;
            # replacing them with amax/reciprocal may change compiler rounding.
            scale = (normed.abs().max(dim=-1).values + 1e-8) / 240.
            scale = scale.unsqueeze(-1)
        q = torch.ops.hpu.cast_to_fp8_v2(normed, 1.0 / scale, False, False, torch.float8_e4m3fn)[0]
        return project(q, scale.float(), weight, weight_scale), summed

    def cguid(x, residual, gamma, weight, weight_scale):
        return vendor(x, residual, gamma, weight, weight_scale, True)

    hpu_backend = torch._dynamo.lookup_backend("hpu_backend")

    def audited_backend(graph, inputs):
        graphs.setdefault(case_key[0], []).append(audit_fx(graph))
        return hpu_backend(graph, inputs)

    functions = {
        name: torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
        for name, fn in (("formula", formula), ("vendor", vendor), ("cguid", cguid))
    }
    functions["native"] = torch.compile(native, backend=audited_backend, fullgraph=True, dynamic=False)
    return functions


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
    torch._dynamo.config.cache_size_limit = max(128, len(args.batches) * len(args.projections) * 4 + 8)
    graphs, case_key = {}, [None]
    functions = make_functions(args.scale_mode, graphs, case_key)
    root = Path(__file__).resolve().parents[1]
    paths = [
        *map(Path, libraries), root / "flashinfer_gaudi/lib/libflashinfer_gaudi_kernels.so", *sorted(
            (root / "csrc/flashinfer_gaudi").rglob("*norm*.*")), root / "flashinfer_gaudi/norm.py",
        root / "flashinfer_gaudi/_native.py", root / "flashinfer_gaudi/_qualification.py",
        root / "vllm_gaudi/extension/ops.py",
        Path(__file__).resolve(), root / "tools/benchmark_flashinfer_native_norm.py"
    ]

    def fingerprints():
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    def tensor_hash(value):
        return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    report = {
        "schema_version": 1,
        "session_id": str(uuid.uuid4()),
        "pid": os.getpid(),
        "sha256": fingerprints(),
        "contract": {
            "device_module": os.environ.get("HLS_MODULE_ID"),
            "torch_version": torch.__version__,
            "scale_mode": args.scale_mode,
            "width": args.width,
            "batches": args.batches,
            "projections": args.projections,
            "iterations": args.iterations,
            "waves": args.waves,
            "weights": "synthetic BF16 to per-channel E4M3, outside timing",
            "activations": "synthetic BF16; per-shape seed independent of traversal",
            "measurement": "paired device queue waves including submission gaps",
            "lazy_mode": os.environ.get("PT_HPU_LAZY_MODE"),
            "triton_mode": os.environ.get("VLLM_HPU_TRITON_MODE")
        },
        "use_eager_fallback": False,
        "cases": {},
        "production_promoted": False
    }
    for projection in args.projections:
        generator = torch.Generator().manual_seed(2341 + projection)
        raw_weight = (torch.randn(projection, args.width, generator=generator, dtype=torch.bfloat16) * .02).to("hpu")
        weight_scale = (raw_weight.float().abs().amax(-1) + 1e-8) / 240.
        weight = torch.ops.hpu.cast_to_fp8_v2(raw_weight, weight_scale[:, None].reciprocal(), False, False,
                                              torch.float8_e4m3fn)[0]
        weight_cpu, weight_scale_cpu = weight.cpu(), weight_scale.cpu()
        weight_hash = tensor_hash(weight_cpu)
        del raw_weight
        saved = []
        for batch in args.batches:
            key = f"{batch}x{args.width}x{projection}"
            case_key[0] = key
            case = {
                "correctness": False,
                "baseline_correctness": dict.fromkeys(BASELINES, True),
                "comparison_errors": {},
                "numerics": {},
                "reentry": {},
                "samples_ms": {
                    name: []
                    for name in functions
                },
                "host_ms": {
                    name: []
                    for name in functions
                }
            }
            report["cases"][key] = case
            try:
                generator = torch.Generator().manual_seed(9137 + projection + batch * 65537 + args.width)
                x = torch.randn(batch, args.width, generator=generator).bfloat16()
                residual = torch.randn(x.shape, generator=generator).bfloat16()
                gamma = (torch.randn(args.width, generator=generator) * .1 + 1.).bfloat16()
                cpu_inputs = x, residual, gamma, weight_cpu, weight_scale_cpu
                inputs = x.to("hpu"), residual.to("hpu"), gamma.to("hpu"), weight, weight_scale
                case["fixture_sha256"] = {
                    "weight": weight_hash,
                    **{
                        str(i): tensor_hash(value)
                        for i, value in enumerate(cpu_inputs) if i != 3
                    }
                }
                for magnitude in (0., 1e-5, .25, 1., 8.):
                    samples = (x * magnitude).to("hpu"), (residual * magnitude).to("hpu"), *inputs[2:]
                    actual = functions["native"](*samples)
                    for label in BASELINES:
                        try:
                            case["numerics"][f"{label}:{magnitude}"] = validate(actual, functions[label](*samples))
                        except AssertionError as exc:
                            case["baseline_correctness"][label] = False
                            case["comparison_errors"][f"{label}:{magnitude}"] = str(exc)
                    for value, expected in zip(samples[:3], (x * magnitude, residual * magnitude, gamma)):
                        torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)
                case["native_fx"] = graphs.get(key, [])
                case["correctness"] = all(case["baseline_correctness"].values())
                compatible = [label for label in BASELINES if case["baseline_correctness"][label]]
                if compatible:
                    labels = (*compatible, "native")
                    for _ in range(args.warmups):
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
                        case["traces"] = {}
                        for label in labels:
                            path = args.output.with_name(f"{args.output.stem}-{key}-{label}-trace.json")
                            with torch.profiler.profile(activities=[
                                    torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU
                            ]) as profiler:
                                for _ in range(5):
                                    functions[label](*inputs)
                                torch.hpu.synchronize()
                            profiler.export_chrome_trace(str(path))
                            events = json.loads(path.read_text())["traceEvents"]
                            audit = audit_trace(events) if label == "native" else {
                                "kernels": sorted({event["name"]
                                                   for event in events if event.get("cat") == "kernel"})
                            }
                            case["traces"][label] = {"path": path.name, "fx_audited": bool(case["native_fx"]), **audit}
                for value, expected in zip(inputs, cpu_inputs):
                    torch.testing.assert_close(value.cpu().view(torch.uint8),
                                               expected.view(torch.uint8),
                                               rtol=0,
                                               atol=0)
                case["inputs_unchanged"] = True
                saved.append((key, inputs))
            except Exception as exc:
                case.update(correctness=False, error=f"{type(exc).__name__}: {exc}", inputs_unchanged=False)
            write_report(args.output, report)
            print(key, case.get("error", case.get("speedup", case["baseline_correctness"])), flush=True)
        # Revisit every specialization after compiling all batches, with changed
        # values. Keep each baseline's failure visible instead of replacing it.
        for key, inputs in reversed(saved):
            case = report["cases"][key]
            samples = inputs[0] * .375, inputs[1] * .375, *inputs[2:]
            for label in BASELINES:
                try:
                    validate(functions["native"](*samples), functions[label](*samples))
                    case["reentry"][label] = True
                except Exception as exc:
                    case["reentry"][label] = False
                    case["comparison_errors"][f"{label}:reentry"] = str(exc)
        del saved
        report["artifacts_unchanged"] = fingerprints() == report["sha256"]
        write_report(args.output, report)
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
        if output.exists():
            raise FileExistsError(f"Refusing to reuse previous session evidence: {output}")
        command = [
            sys.executable,
            str(Path(__file__).resolve()), "--worker", "--output",
            str(output), "--batches", ",".join(map(str, args.batches)), "--projections",
            ",".join(map(str, args.projections)), "--width",
            str(args.width), "--scale-mode", args.scale_mode, "--waves",
            str(args.waves), "--iterations",
            str(args.iterations), "--warmups",
            str(args.warmups)
        ]
        if args.trace and number == 0:
            command.append("--trace")
        print(f"Starting process session {number + 1}/{args.sessions}: {output.name}", flush=True)
        with output.with_suffix(".log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            result["worker_errors"].append({"session": number, "returncode": completed.returncode})
        if output.is_file():
            sessions.append(json.loads(output.read_text()))
        write_report(args.output, result)
    for projection in args.projections:
        for batch in args.batches:
            key = f"{batch}x{args.width}x{projection}"
            result["cases"][key] = qualify_case(sessions, key, args.sessions, result["worker_errors"])
    result["session_files"] = [f"{args.output.stem}-session-{i}.json" for i in range(args.sessions)]
    write_report(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
