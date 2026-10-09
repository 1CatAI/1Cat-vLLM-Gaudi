# SPDX-License-Identifier: Apache-2.0
"""Benchmark fused mHC control projection/RRMS through all gate consumers.

The chain uses four real V4.1 layers (attention and FFN mHC boundaries), real
weights, changing BF16 residuals and the production fused gate operator.  The
identity sublayer between mHC boundaries keeps the real dependency topology
without mixing attention or expert time into this component decision.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import habana_frameworks.torch.core  # noqa: F401
from safetensors import safe_open


def pack_control_weight(weight):
    """The linear-lane kernel consumes the original checkpoint K order."""
    return weight.contiguous()


def gate(projection, rrms, scale, base):
    value = torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(projection.contiguous(), rrms.contiguous(),
                                                                         scale.contiguous(), base.contiguous())
    return value[:, :4], value[:, 4:8], value[:, 8:].reshape(-1, 4, 4)


def update(residual, previous_pre, projection, rrms, scale, base):
    pre, post, comb = gate(projection, rrms, scale, base)
    collapsed = (residual.float() * previous_pre.unsqueeze(-1)).sum(1).to(residual.dtype)
    mixed = (comb.unsqueeze(-1) * residual.float().unsqueeze(2)).sum(1)
    # An identity sublayer output preserves the production hc_post dataflow.
    residual = (collapsed.float().unsqueeze(1) * post.unsqueeze(-1) + mixed).to(residual.dtype)
    return residual, pre


def reference(residual, previous_pre, weights, scales, bases, linear=False, epsilon=1e-20):
    for index in range(8):
        flat = residual.flatten(1).float()
        projection = (torch.nn.functional.linear(flat, weights[index]) if linear else
                      torch.ops.custom_op.custom_deepseek_v41_control_gemv_f32_gaudi2(flat, weights[index]))
        rrms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + epsilon)
        residual, previous_pre = update(residual, previous_pre, projection, rrms, scales[index], bases[index])
    return residual, previous_pre.contiguous()


def candidate(residual, previous_pre, weights, scales, bases, epsilon=1e-20):
    for index in range(8):
        control = (torch.ops.custom_op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(
            residual.flatten(1).contiguous(), weights[index], epsilon))
        projection, rrms = control[:, :24], control[:, 24:]
        residual, previous_pre = update(residual, previous_pre, projection, rrms, scales[index], bases[index])
    return residual, previous_pre.contiguous()


def transposed_candidate(residual, previous_pre, weights, scales, bases, epsilon):
    """Retain FP32 GEMM and its consumers with a prepared K-by-N weight."""
    for index in range(8):
        flat = residual.flatten(1).float()
        projection = torch.matmul(flat, weights[index])
        rrms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + epsilon)
        residual, previous_pre = update(residual, previous_pre, projection, rrms, scales[index], bases[index])
    return residual, previous_pre.contiguous()


def batch_control_candidate(residual, previous_pre, weights, scales, bases, epsilon):
    """Reuse each FP32 weight vector across rows through the existing control kernel."""
    for index in range(8):
        flat = residual.flatten(1).float()
        projection = torch.ops.custom_op.custom_deepseek_v41_control_batch4_f32_gaudi2(flat, weights[index])
        rrms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + epsilon)
        residual, previous_pre = update(residual, previous_pre, projection, rrms, scales[index], bases[index])
    return residual, previous_pre.contiguous()


def explicit_mme_candidate(residual, previous_pre, weights, scales, bases, epsilon):
    for index in range(8):
        flat = residual.flatten(1)
        operands = flat.float() if weights.dtype == torch.float32 else flat
        projection = torch.ops.custom_op.custom_deepseek_v41_control_mme_f32_gaudi2(operands, weights[index])
        if weights.shape[1] == 48:
            projection = projection[:, :24] + projection[:, 24:]
        rrms = torch.rsqrt(flat.float().square().mean(-1, keepdim=True) + epsilon)
        residual, previous_pre = update(residual, previous_pre, projection, rrms, scales[index], bases[index])
    return residual, previous_pre.contiguous()


def measure(function, fixtures, iterations):
    residual = torch.empty_like(fixtures[0][0])
    previous = torch.empty_like(fixtures[0][1])
    device, host = [], []
    for iteration in range(iterations + 8):
        source, source_pre = fixtures[iteration % len(fixtures)]
        residual.copy_(source)
        previous.copy_(source_pre)
        torch.hpu.synchronize()
        begin, end = (torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True))
        started = time.perf_counter_ns()
        begin.record()
        output = function(residual, previous)
        end.record()
        # Drain both real consumers instead of timing enqueue only.
        output[0].cpu()
        output[1].cpu()
        torch.hpu.synchronize()
        if iteration >= 8:
            device.append(begin.elapsed_time(end))
            host.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "iterations": iterations,
        "device_ms": device,
        "device_median_ms": statistics.median(device),
        "device_mean_ms": statistics.fmean(device),
        "host_ms": host,
        "host_median_ms": statistics.median(host),
    }


def measure_native(replay, buffers, fixtures, iterations):
    """Stage changing inputs before the event, then replay to both consumers."""
    residual, previous = buffers
    device, host = [], []
    for iteration in range(iterations + 8):
        source, source_pre = fixtures[iteration % len(fixtures)]
        residual.copy_(source)
        previous.copy_(source_pre)
        torch.hpu.synchronize()
        begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
        started = time.perf_counter_ns()
        begin.record()
        replay()
        end.record()
        end.synchronize()
        if iteration >= 8:
            device.append(begin.elapsed_time(end))
            host.append((time.perf_counter_ns() - started) / 1e6)
    return dict(iterations=iterations,
                device_ms=device,
                device_median_ms=statistics.median(device),
                device_mean_ms=statistics.fmean(device),
                host_ms=host,
                host_median_ms=statistics.median(host))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--linear-reference",
                        action="store_true",
                        help="Use the generic FP32 C6 control projection as the comparison arm")
    parser.add_argument("--diagnose-stages", action="store_true", help="Projection/RRMS/hybrid gate outputs, no timing")
    parser.add_argument("--transposed-gemm",
                        action="store_true",
                        help="Compare an immutable K-by-N FP32 weight layout, preserving generic GEMM arithmetic")
    parser.add_argument("--native-replay",
                        action="store_true",
                        help="Retain the compiler recipes through the existing compute-only diagnostic replay API")
    parser.add_argument("--batch-control", action="store_true",
                        help="Screen the existing FP32 control weight reuse through unchanged norm/gate consumers")
    parser.add_argument("--explicit-mme", choices=("f32", "bf16", "bf16-pair"),
                        help="Screen one explicit MME projection; BF16 operands retain FP32 accumulation/output")
    parser.add_argument("--control-addon", type=Path)
    parser.add_argument("--packed-reference", action="store_true",
                        help="Use the current packed TPC projection/RRMS as the comparison parent")
    args = parser.parse_args()
    if args.explicit_mme and (not args.control_addon or args.batch_control or args.transposed_gemm
                             or args.diagnose_stages):
        parser.error("Explicit MME requires its additive registration and a single candidate")
    if args.packed_reference and args.linear_reference:
        parser.error("Select exactly one reference projection")
    if args.batch_control and (args.transposed_gemm or args.diagnose_stages):
        parser.error("--batch-control cannot combine with other candidates")
    if args.transposed_gemm and (not args.linear_reference or args.diagnose_stages):
        parser.error("--transposed-gemm requires --linear-reference and cannot be combined with --diagnose-stages")
    if args.native_replay and args.diagnose_stages:
        parser.error("Stage diagnosis does not measure a native replay")
    epsilon = json.loads((args.prepared / "config.json").read_text())["text_config"]["rms_norm_eps"]
    if not 1 <= args.tokens <= 6:
        parser.error("The decode control fixture supports C1-C6")
    extension = next(args.native_dir.glob("hpu_dsv4_sparse_attn_pt2*.so"))
    torch.ops.load_library(str(extension))
    if args.control_addon:
        torch.ops.load_library(str(args.control_addon))

    rank_file = args.prepared / "pp0-tp0.safetensors"
    weights, scales, bases = [], [], []
    with safe_open(rank_file, framework="pt", device="cpu") as handle:
        for layer in range(4):
            for kind in ("attn", "ffn"):
                prefix = f"layers.{layer}.hc_{kind}_"
                weights.append(handle.get_tensor(prefix + "fn").float())
                scales.append(handle.get_tensor(prefix + "scale").float())
                bases.append(handle.get_tensor(prefix + "base").float())
    original = torch.stack(weights).to("hpu")
    packed = torch.stack(
        [value.T.contiguous() if args.transposed_gemm else pack_control_weight(value) for value in weights]).to("hpu")
    if args.explicit_mme == "bf16":
        packed = packed.to(torch.bfloat16)
    elif args.explicit_mme == "bf16-pair":
        high = packed.bfloat16()
        low = (packed - high.float()).bfloat16()
        packed = torch.cat((high, low), dim=1).contiguous()
    scales = torch.stack(scales).to("hpu")
    bases = torch.stack(bases).to("hpu")
    recorder = None
    if args.native_replay:
        from deepseek_v41_micro_replay import RecipeRecorder
        args.output.parent.mkdir(parents=True, exist_ok=True)
        recorder = RecipeRecorder(args.output.parent)

    def reference_fn(residual, previous):
        if args.packed_reference:
            return candidate(residual, previous, original, scales, bases, epsilon)
        return reference(residual, previous, original, scales, bases, args.linear_reference, epsilon)

    compiled_reference = torch.compile(reference_fn,
                                       backend="hpu_backend",
                                       fullgraph=True,
                                       dynamic=False)
    candidate_fn = (explicit_mme_candidate if args.explicit_mme else batch_control_candidate if args.batch_control else
                    transposed_candidate if args.transposed_gemm else candidate)
    compiled_candidate = torch.compile(
        lambda residual, previous: candidate_fn(residual, previous, packed, scales, bases, epsilon),
        backend="hpu_backend",
        fullgraph=True,
        dynamic=False)
    fixtures = []
    for seed in range(4):
        generator = torch.Generator().manual_seed(4100 + seed)
        residual = (torch.randn(args.tokens, 4, 5120, generator=generator, dtype=torch.bfloat16) * 0.25).to("hpu")
        previous = torch.tensor([[1.0 - 0.01 * seed, 0.01 * seed, 0, 0]], dtype=torch.float32,
                                device="hpu").expand(args.tokens, -1).contiguous()
        fixtures.append((residual, previous))

    if args.diagnose_stages:

        def stages(residual, previous):
            outputs = []
            for index in range(8):
                flat = residual.flatten(1).float()
                projection = torch.nn.functional.linear(flat, original[index])
                rrms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + epsilon)
                control = torch.ops.custom_op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(
                    residual.flatten(1).contiguous(), packed[index], epsilon)
                same = update(residual, previous, projection, rrms, scales[index], bases[index])
                only_projection = update(residual, previous, control[:, :24], rrms, scales[index], bases[index])
                only_rrms = update(residual, previous, projection, control[:, 24:], scales[index], bases[index])
                outputs.extend((projection, control[:, :24], rrms, control[:,
                                                                           24:], *same, *only_projection, *only_rrms))
                residual, previous = same
            return tuple(outputs)

        diagnostic = torch.compile(stages, backend="hpu_backend", fullgraph=True, dynamic=False)
        rows = []
        for fixture, (residual, previous) in enumerate(fixtures):
            values = tuple(v.cpu() for v in diagnostic(residual, previous))
            for index in range(8):
                p, np_, r, nr, h, pre, hp, prep, hr, prer = values[index * 10:(index + 1) * 10]
                rows.append(
                    dict(fixture=fixture,
                         boundary=index,
                         projection_exact=torch.equal(p, np_),
                         projection_max_abs=float((p - np_).abs().max()),
                         rrms_exact=torch.equal(r, nr),
                         rrms_max_abs=float((r - nr).abs().max()),
                         projection_only_residual_exact=torch.equal(h, hp),
                         projection_only_pre_exact=torch.equal(pre, prep),
                         rrms_only_residual_exact=torch.equal(h, hr),
                         rrms_only_pre_exact=torch.equal(pre, prer)))
        args.output.write_text(json.dumps(dict(status="diagnostic_only", epsilon=epsilon, rows=rows), indent=2) + "\n")
        return

    checks = []
    for residual, previous in fixtures:
        expected = compiled_reference(residual, previous)
        actual = compiled_candidate(residual, previous)
        torch.hpu.synchronize()
        expected = tuple(value.cpu() for value in expected)
        actual = tuple(value.cpu() for value in actual)
        checks.append({
            "residual_bitwise": torch.equal(expected[0].view(torch.int16), actual[0].view(torch.int16)),
            "residual_max_abs": (expected[0].float() - actual[0].float()).abs().max().item(),
            "pre_max_abs": (expected[1] - actual[1]).abs().max().item(),
            "pre_bitwise": torch.equal(expected[1].view(torch.int32), actual[1].view(torch.int32)),
            "finite": all(torch.isfinite(value).all().item() for value in actual),
        })
    if not all(item["finite"] for item in checks):
        raise RuntimeError("fused control/RRMS produced non-finite output")
    expect_exact = args.transposed_gemm or (
        args.batch_control and not args.linear_reference and not args.packed_reference)
    if expect_exact and not all(item["residual_bitwise"] and item["pre_bitwise"] for item in checks):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(dict(status="rejected_exactness", epsilon=epsilon, checks=checks), indent=2) + "\n")
        raise AssertionError("The FP32 arithmetic-preserving candidate changed the mHC consumer output")

    replays = {}
    if recorder is not None:
        buffers = torch.empty_like(fixtures[0][0]), torch.empty_like(fixtures[0][1])
        buffers[0].copy_(fixtures[0][0])
        buffers[1].copy_(fixtures[0][1])
        for arm, function in (("A", compiled_reference), ("B", compiled_candidate)):
            replays[arm] = recorder.prepare(function, [buffers[0]], [(buffers[1], )])
            for residual, previous in fixtures:
                buffers[0].copy_(residual)
                buffers[1].copy_(previous)
                expected = tuple(value.cpu() for value in function(*buffers))
                replays[arm]()
                # The diagnostic recorder does not publish output tensor
                # producer events. Drain its complete device work before a
                # correctness readback; this stays outside all timed samples.
                torch.hpu.synchronize()
                actual = tuple(value.cpu() for value in replays[arm].outputs)
                if not all(
                        torch.equal(lhs.contiguous().view(torch.uint8),
                                    rhs.contiguous().view(torch.uint8))
                        for lhs, rhs in zip(expected, actual, strict=True)):
                    args.output.write_text(json.dumps(dict(
                        status="rejected_native_consumer", arm=arm,
                        errors=[dict(max_abs=float((lhs.float() - rhs.float()).abs().max()),
                                     changed=int((lhs != rhs).sum()))
                                for lhs, rhs in zip(expected, actual, strict=True)],
                        native_info=replays[arm].native.info(), timing_samples=0), indent=2) + "\n")
                    raise AssertionError("Native mHC replay changed a compiled consumer")
        timings = [
            dict(arm=arm, **measure_native(replays[arm], buffers, fixtures, args.iterations))
            for arm in ("A", "B", "A", "B", "A", "B")
        ]
        reference_result, candidate_result = timings[:2]
    elif args.transposed_gemm or args.batch_control:
        timings = [
            dict(arm=arm, **measure(fn, fixtures, args.iterations))
            for arm, fn in (("A", compiled_reference), ("B", compiled_candidate), ("A", compiled_reference),
                            ("B", compiled_candidate), ("A", compiled_reference), ("B", compiled_candidate))
        ]
        reference_result, candidate_result = timings[:2]
    else:
        reference_result = measure(compiled_reference, fixtures, args.iterations)
        candidate_result = measure(compiled_candidate, fixtures, args.iterations)
    result = {
        "epsilon": epsilon,
        "status": "microbench_complete",
        "tokens": args.tokens,
        "reference_projection": ("Packed TPC control/RRMS" if args.packed_reference else
                                 "FP32 linear" if args.linear_reference else "TPC control GEMV"),
        "candidate_projection": (f"Explicit {args.explicit_mme} MME operands / FP32 output" if args.explicit_mme else
                                 "FP32 control weight reuse" if args.batch_control else
                                 "FP32 transposed GEMM" if args.transposed_gemm else "BF16-input TPC control/RRMS"),
        "boundary": "four real layers / eight mHC boundaries through fused gates and hc_post consumer",
        "real_weight_bytes": int(original.numel() * original.element_size()),
        "extra_prepared_weight_bytes": int(packed.numel() * packed.element_size()),
        "checks": checks,
        "reference": reference_result,
        "candidate": candidate_result,
        "timing_mode": "native compute replay" if recorder is not None else "compiled host submission",
    }
    result["device_saved_ms"] = (reference_result["device_median_ms"] - candidate_result["device_median_ms"])
    result["device_saved_percent"] = (100 * result["device_saved_ms"] / reference_result["device_median_ms"])
    if recorder is not None:
        result["ababab"] = timings
        result["native_compute_info"] = {arm: replay.native.info() for arm, replay in replays.items()}
        for replay in replays.values():
            replay.close()
        torch.distributed.destroy_process_group()
    elif args.transposed_gemm or args.batch_control:
        result["ababab"] = timings
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
