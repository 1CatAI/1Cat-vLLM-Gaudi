# SPDX-License-Identifier: Apache-2.0
"""Complete C6 main-derived operators, compiled and natively replayed."""
import argparse
import json
import os
from pathlib import Path
import time

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402,F401
from tools.deepseek_v41_component_replay import ComponentReplay  # noqa: E402
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_fp8 import read_expert  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa, unpack_fp4, unpack_swa  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402
from vllm_gaudi.ops.tp2_runtime_profile import verify_loaded_profile_libraries  # noqa: E402


def timing(native, iterations=64):
    for _ in range(3):
        native.replay()
    torch.hpu.synchronize()
    start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
    started = time.monotonic_ns()
    start.record()
    for _ in range(iterations):
        native.replay()
    end.record()
    end.synchronize()
    return {
        "replays": iterations,
        "synchronized_host_ms": (time.monotonic_ns() - started) / 1e6 / iterations,
        "device_event_ms": start.elapsed_time(end) / iterations,
        "native_info": native.info(),
        "native_statistics": list(native.statistics)
    }


def compare(actual, expected):
    error = (actual.float() - expected.float()).abs()
    return {
        "bf16_mismatches": int((actual.view(torch.int16) != expected.view(torch.int16)).sum()),
        "max_absolute_error": float(error.max()),
        "rmse": float(error.square().mean().sqrt()),
        "finite": bool(actual.isfinite().all())
    }


def moe(args, out):
    shard = PreparedV41Shard(args.checkpoint, 0, 0)
    weights = {}
    preparation = []
    for projection in ("w13", "w2"):
        prefix = "layers.0.ffn.experts." + projection
        arrays = [[], [], []]
        for expert in range(36):
            q = read_expert(shard.catalog[prefix + "_q16"], expert)
            s = read_expert(shard.catalog[prefix + "_s16"], expert)
            nq, ns, channel, record = prepare_expert(q, s)
            preparation.append({"projection": projection, "expert": expert, **record})
            for rows, value in zip(arrays, (nq, ns, channel), strict=True):
                rows.append(value)
        for suffix, rows in zip(("_q16", "_s16", "_channel"), arrays, strict=True):
            value = torch.from_numpy(np.stack(rows).view(np.int16))
            if suffix == "_channel":
                value = value.view(torch.bfloat16)
            weights[projection + suffix] = value.to("hpu")
        del arrays
    shard.check_identity()
    (out / "weight-preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
    lookup = mxfp4_bf16_lut(torch.device("hpu"))
    normal = shard.manifest["normal_scales"]["layers.0.ffn.experts"][0]
    name = {
        "bf16": "custom_deepseek_v41_expert_n256_moe_bf16_gaudi2",
        "fp8": "custom_deepseek_v41_expert_n256_moe_fp8_gaudi2",
        "fp8-fused": "custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2"
    }[args.precision]
    op = getattr(torch.ops.custom_op, name)
    extra = () if args.precision == "bf16" else (weights["w13_channel"], weights["w2_channel"])

    def complete(x, ids, routing):
        return (op(x, ids, routing, weights["w13_q16"], weights["w2_q16"], weights["w13_s16"], weights["w2_s16"],
                   lookup, *extra, normal), )

    compiled = torch.compile(complete, backend="hpu_backend", fullgraph=True, dynamic=False)
    paths = sorted(args.references.glob("real-weight-*.pt"))
    if len(paths) != 3:
        raise ValueError("Expected the three archived complete-MoE reference cases")
    first = torch.load(paths[0], weights_only=True)
    inputs = [first[name].to("hpu") for name in ("value", "ids", "routing")]
    compiled(*inputs)
    torch.hpu.synchronize()
    native = ComponentReplay(compiled, inputs)
    records = []
    try:
        for path in paths:
            saved = torch.load(path, weights_only=True)
            for destination, name in zip(inputs, ("value", "ids", "routing"), strict=True):
                destination.copy_(saved[name])
            ordinary = compiled(*inputs)[0]
            torch.hpu.synchronize()
            ordinary = ordinary.cpu()
            native.replay()
            torch.hpu.synchronize()
            replayed = native.outputs[0].cpu()
            consistency = compare(replayed, ordinary)
            reference = compare(replayed, saved["expected"])
            record = {
                "case": str(path),
                "precision": args.precision,
                "ordinary_vs_replay": consistency,
                "frozen_bf16_reference": reference,
                "scope": "Complete routed MoE; excludes other model modules and input staging, not full C6"
            }
            if consistency["bf16_mismatches"] or not reference["finite"]:
                records.append(record)
                torch.save({"replay": replayed, "ordinary": ordinary, "saved": saved}, out / "moe-failure.pt")
                raise RuntimeError("Changed-input complete MoE failed the execution gate")
            record.update(timing(native))
            records.append(record)
            (out / "moe-result.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            if args.precision == "bf16" and reference["bf16_mismatches"]:
                torch.save({"actual": replayed, "saved": saved}, out / "bf16-reference-difference.pt")
                raise RuntimeError("N256 BF16 changed the frozen arithmetic result")
    finally:
        (out / "moe-result.json").write_text(json.dumps(records, indent=2) + "\n")
        native.close()


def mla_reference(q, cache, ids, sink, scale, lengths):
    valid = (ids >= 0) & (ids < cache.shape[0]) & (torch.arange(ids.shape[1])[None, :] < lengths[:, None])
    selected = cache[ids.clamp(0, cache.shape[0] - 1).long()].float() * valid[:, :, None]
    scores = torch.bmm(q.float(), selected.transpose(-1, -2)) * scale
    scores.masked_fill_(~valid[:, None, :], -float("inf"))
    extended = torch.cat((scores, sink[None, :, None].expand(q.shape[0], -1, 1)), -1)
    probabilities = extended.softmax(-1)[..., :-1]
    return torch.bmm(probabilities, selected).bfloat16()


def mla(args, out):
    torch.manual_seed(307)
    tokens, heads, width = 6, 32, 640
    q = torch.randn(tokens, heads, 512).bfloat16()
    swa = pack_swa(torch.randn(256, 512).bfloat16())
    main = pack_fp4(torch.randn(1024, 512).bfloat16())
    # Matching retained selected-KV layout: first SWA slots, then main slots.
    main_rows = torch.arange(512) * 7 % 1024
    row_ids = torch.cat((torch.arange(256), 256 + main_rows)).to(torch.int32)[None, :]
    cache = torch.cat((unpack_swa(swa), unpack_fp4(main)[main_rows]), 0).contiguous()
    ids = torch.randint(0, 768, (tokens, width), dtype=torch.int32)
    ids[:, 7] = -1
    ids[:, 11] = 768
    lengths = torch.tensor([0, 1, 127, 512, 639, 640], dtype=torch.int32)
    sink, scale = torch.randn(heads), torch.tensor([512**-0.5])
    host_inputs = [q, swa, main, row_ids, ids, sink, scale, lengths]
    inputs = [x.to("hpu") for x in host_inputs]
    op = torch.ops.custom_op.custom_deepseek_v41_paged_mla_mme_gaudi2
    compiled = torch.compile(lambda *x: (op(*x), ), backend="hpu_backend", fullgraph=True, dynamic=False)
    compiled(*inputs)
    torch.hpu.synchronize()
    native = ComponentReplay(compiled, inputs)
    records = []
    try:
        for generation in range(2):
            if generation:
                q = -q
                ids = ids.flip(0).contiguous()
                lengths = lengths.flip(0).contiguous()
                for index, value in ((0, q), (4, ids), (7, lengths)):
                    inputs[index].copy_(value)
            ordinary = compiled(*inputs)[0]
            torch.hpu.synchronize()
            ordinary = ordinary.cpu()
            native.replay()
            torch.hpu.synchronize()
            result = native.outputs[0].cpu()
            expected = mla_reference(q, cache, ids, sink, scale, lengths)
            record = {
                "generation": generation,
                "ordinary_vs_replay": compare(result, ordinary),
                "fp32_reference": compare(result, expected),
                "scope": "Selected KV decode, gather, QK, softmax, PV and BF16 output; not full attention/C6"
            }
            torch.save(
                {
                    "inputs": host_inputs,
                    "q": q,
                    "ids": ids,
                    "lengths": lengths,
                    "actual": result,
                    "expected": expected
                }, out / f"mla-{generation}.pt")
            records.append(record)
            if record["ordinary_vs_replay"]["bf16_mismatches"] or not record["fp32_reference"]["finite"]:
                raise RuntimeError("Selected MLA changed-input replay contract failed")
            torch.testing.assert_close(result.float(), expected.float(), atol=0.01, rtol=0.02)
            record.update(timing(native))
            (out / "mla-result.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    finally:
        (out / "mla-result.json").write_text(json.dumps(records, indent=2) + "\n")
        native.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("moe", "mla"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--precision", choices=("bf16", "fp8", "fp8-fused"), default="bf16")
    args = parser.parse_args()
    out = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    (out / "loaded-runtime.json").write_text(json.dumps(verify_loaded_profile_libraries(), indent=2) + "\n")
    (moe if args.component == "moe" else mla)(args, out)


if __name__ == "__main__":
    main()
