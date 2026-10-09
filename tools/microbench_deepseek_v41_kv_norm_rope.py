# SPDX-License-Identifier: Apache-2.0
"""Check and time the exact, batch-generic V4.1 KV norm/RoPE fusion."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
import habana_frameworks.torch.core  # noqa: F401


def _consume(fn, *, attention_consumer, swa, main_cache, selected, pages, query, sink, scale, lengths):
    """Bind each fixture's state before compilation and the next batch."""

    def chain(value, norm_weight, positions, rope_phase):
        rotated = fn(value, norm_weight, positions, rope_phase)
        if not attention_consumer:
            return rotated
        from vllm_gaudi.ops.deepseek_v41_math import pack_swa
        swa.index_copy_(0, positions.remainder(256).long(), pack_swa(rotated))
        output = torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(query, swa, main_cache, selected, positions,
                                                                            pages, sink, scale, lengths, 1, True)
        return torch.cat((rotated.flatten(), output.flatten()))

    return chain


def _measure(fn, args, repeats):
    for _ in range(8):
        fn(*args)
    torch.hpu.synchronize()
    start = torch.hpu.Event(enable_timing=True)
    end = torch.hpu.Event(enable_timing=True)
    host_start = time.perf_counter_ns()
    start.record()
    for _ in range(repeats):
        fn(*args)
    end.record()
    torch.hpu.synchronize()
    host_ms = (time.perf_counter_ns() - host_start) / 1.0e6
    return {
        "device_ms_per_call": start.elapsed_time(end) / repeats,
        "host_ms_per_call": host_ms / repeats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 6, 32, 512])
    parser.add_argument("--repeats", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=1e-20, help="Use the prepared model's RMS epsilon")
    parser.add_argument("--generic-reference", action="store_true", help="Match C6's current generic FP32 RMSNorm")
    parser.add_argument("--attention-consumer", action="store_true", help="Include SWA publication and logical MLA")
    args = parser.parse_args()
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("RMS epsilon must be finite and positive")
    # Both arithmetic arms share one factory code object across shape cases.
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 2 * len(args.batches) + 2)
    args.output = Path(os.environ.get("DSV41_RUN_EVIDENCE", str(args.output)))
    args.output.mkdir(parents=True, exist_ok=True)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.manual_seed(20260920)
    phase = torch.randn(32768, 64, dtype=torch.float32, device="hpu")
    weight = torch.randn(512, dtype=torch.bfloat16, device="hpu")
    epsilon = args.epsilon

    def reference(value, norm_weight, positions, rope_phase):
        if args.generic_reference:
            from vllm_gaudi.ops.deepseek_v41_math import rms_norm
            normalized = rms_norm(value, norm_weight, epsilon)
        else:
            normalized = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2(value, norm_weight, epsilon)
        if normalized.shape[0] <= 6:
            return torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(
                normalized.reshape(normalized.shape[0], 1, 512), positions, rope_phase).reshape(normalized.shape)
        pairs = normalized[..., -64:].float().unflatten(-1, (-1, 2))
        native_phase = rope_phase.index_select(0, positions.long())
        real, imag = pairs[..., 0], pairs[..., 1]
        cosine, sine = native_phase[..., :32], native_phase[..., 32:]
        rotated = torch.stack((real * cosine - imag * sine, real * sine + imag * cosine), -1)
        return torch.cat((normalized[..., :-64], rotated.flatten(-2).to(normalized.dtype)), -1)

    def candidate(value, norm_weight, positions, rope_phase):
        return torch.ops.custom_op.custom_deepseek_v41_kv_norm_rope_bf16_gaudi2(value, norm_weight, positions,
                                                                                rope_phase, epsilon)

    result = {
        "scope": "exact KV RMSNorm BF16 boundary plus forward RoPE",
        "batches": [],
        "repeats": args.repeats,
        "epsilon": epsilon,
    }
    with torch.inference_mode():
        for batch in args.batches:
            value = torch.randn(batch, 512, dtype=torch.bfloat16, device="hpu")
            positions = torch.arange(16384, 16384 + batch, dtype=torch.int32, device="hpu")
            swa = torch.zeros(256, 528, dtype=torch.uint8, device="hpu")
            main_cache = torch.zeros(32768, 288, dtype=torch.uint8, device="hpu")
            selected = torch.arange(512, dtype=torch.int32, device="hpu").reshape(1, -1).repeat(batch, 1)
            pages = torch.arange(256, dtype=torch.int32, device="hpu")
            query = torch.randn(batch, 16, 512, dtype=torch.bfloat16, device="hpu")
            sink = torch.zeros(16, device="hpu")
            scale = torch.tensor([512**-.5], device="hpu")
            lengths = torch.full((batch, ), 640, dtype=torch.int32, device="hpu")

            state = dict(attention_consumer=args.attention_consumer,
                         swa=swa,
                         main_cache=main_cache,
                         selected=selected,
                         pages=pages,
                         query=query,
                         sink=sink,
                         scale=scale,
                         lengths=lengths)
            ref = torch.compile(_consume(reference, **state), backend="hpu_backend", fullgraph=True, dynamic=False)
            cand = torch.compile(_consume(candidate, **state), backend="hpu_backend", fullgraph=True, dynamic=False)
            expected = ref(value, weight, positions, phase).cpu()
            actual = cand(value, weight, positions, phase).cpu()
            changed = value.clone()
            changed[:, 0].add_(torch.tensor(0.5, dtype=torch.bfloat16, device="hpu"))
            expected_changed = ref(changed, weight, positions, phase).cpu()
            actual_changed = cand(changed, weight, positions, phase).cpu()
            row = {
                "batch":
                batch,
                "bitwise_equal":
                bool(torch.equal(expected.view(torch.int16), actual.view(torch.int16))),
                "changed_bitwise_equal":
                bool(torch.equal(expected_changed.view(torch.int16), actual_changed.view(torch.int16))),
                "changed_input_consumed":
                bool(not torch.equal(actual.view(torch.int16), actual_changed.view(torch.int16))),
                "different":
                int((expected.view(torch.int16) != actual.view(torch.int16)).sum()),
                "reference":
                _measure(ref, (value, weight, positions, phase), args.repeats),
                "candidate":
                _measure(cand, (value, weight, positions, phase), args.repeats),
            }
            if args.attention_consumer:
                ref(value, weight, positions, phase)
                state_a = swa.cpu()
                cand(value, weight, positions, phase)
                row["packed_state_exact"] = bool(torch.equal(state_a, swa.cpu()))
                arms = (("A", ref), ("B", cand), ("A", ref), ("B", cand), ("A", ref), ("B", cand))
                row["ababab"] = [
                    dict(arm=arm, **_measure(fn, (value, weight, positions, phase), args.repeats)) for arm, fn in arms
                ]
            row["device_gain_percent"] = 100.0 * (
                row["reference"]["device_ms_per_call"] -
                row["candidate"]["device_ms_per_call"]) / row["reference"]["device_ms_per_call"]
            result["batches"].append(row)
            print(json.dumps(row), flush=True)
    if not all(row["bitwise_equal"] and row["changed_bitwise_equal"] and row["changed_input_consumed"]
               and row.get("packed_state_exact", True) for row in result["batches"]):
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        raise AssertionError("KV norm/RoPE fusion contract failed")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
