# SPDX-License-Identifier: Apache-2.0
"""Isolate mHC/FFN arithmetic before qualifying the real batched TP chain."""
import argparse
import json
import os
from pathlib import Path

rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
if os.environ.get("DSV41_MICRO_RANK_CPUS"):
    os.sched_setaffinity(0, json.loads(os.environ["DSV41_MICRO_RANK_CPUS"])[rank])

import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from vllm_gaudi.ops.deepseek_v41_math import hc_pre, rms_norm, row_mean_square  # noqa: E402


def summary(expected, actual):
    expected, actual = expected.cpu(), actual.cpu()
    if expected.dtype == torch.float8_e4m3fn:
        expected, actual = expected.view(torch.uint8), actual.view(torch.uint8)
    diff = (expected.float() - actual.float()).abs()
    return dict(exact=torch.equal(expected, actual),
                changed=int((expected != actual).sum()),
                total=expected.numel(),
                max_abs=float(diff.max()))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prepared", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.hpu.set_device(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    ops = torch.ops.custom_op
    cfg = json.loads((args.prepared / "config.json").read_text())["text_config"]
    eps = cfg["rms_norm_eps"]
    with safe_open(args.prepared / f"pp0-tp{rank}.safetensors", framework="pt", device="cpu") as f:
        fn = f.get_tensor("layers.2.hc_attn_fn").float().to("hpu")
        scale = f.get_tensor("layers.2.hc_attn_scale").float().to("hpu")
        base = f.get_tensor("layers.2.hc_attn_base").float().to("hpu")
        weight = f.get_tensor("layers.2.ffn_norm.weight").to("hpu")

    def controls(x):
        flat = x.flatten(1).float()
        old = ops.custom_deepseek_v41_control_gemv_f32_gaudi2(flat, fn)
        rrms = torch.rsqrt(row_mean_square(flat, request_batch=True) + eps)
        fused = ops.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(x.flatten(1).contiguous(), fn, eps)
        return old, rrms, fused[:, :24], fused[:, 24:]

    def mhc(x, previous, fused):
        return hc_pre(x, previous, fn, scale, base, eps, request_batch=True, packed_fn=fn if fused else None)

    def ffn(x, fused):
        if fused:
            return ops.custom_deepseek_v41_ffn_norm_quant_gaudi2(x, weight, eps)
        normalized = rms_norm(x, weight, eps, request_batch=True)
        quantized, scales = ops.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(normalized)
        return normalized, quantized, scales

    compile_fn = lambda f: torch.compile(f, backend="hpu_backend", fullgraph=True, dynamic=False)
    control = compile_fn(controls)
    old_mhc, new_mhc = compile_fn(lambda x, y: mhc(x, y, False)), compile_fn(lambda x, y: mhc(x, y, True))
    old_ffn, new_ffn = compile_fn(lambda x: ffn(x, False)), compile_fn(lambda x: ffn(x, True))
    report = []
    for batch in (1, 8, 32):
        for seed in (1188, 1189):
            torch.manual_seed(seed)
            x = torch.randn(batch, 4, 5120).bfloat16().to("hpu")
            previous = torch.rand(batch, 4).to("hpu")
            a, b, c, d = control(x)
            mo, mn = old_mhc(x, previous), new_mhc(x, previous)
            fo, fe = old_ffn(mo[0]), new_ffn(mo[0])
            record = dict(batch=batch,
                          seed=seed,
                          projection=summary(a, c),
                          rrms=summary(b, d),
                          mhc=[summary(a, b) for a, b in zip(mo, mn, strict=True)],
                          ffn=[summary(a, b) for a, b in zip(fo, fe, strict=True)])
            report.append(record)
            args.output.with_name(f"rank{rank}-" + args.output.name).write_text(json.dumps(report, indent=2))
            print(json.dumps(record), flush=True)
    torch.hpu.synchronize()


if __name__ == "__main__":
    main()
