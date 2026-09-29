# SPDX-License-Identifier: Apache-2.0
"""Isolate request pair arithmetic using real compressor projections."""
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

from vllm_gaudi.ops.deepseek_v41_batch_attention import read_state_rows, write_state_rows  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_math import hc_pre, rms_norm  # noqa: E402


def comparison(a, b):
    a, b = a.cpu(), b.cpu()
    return dict(exact=torch.equal(a, b),
                changed=int((a != b).sum()),
                total=a.numel(),
                max_abs=float((a.float() - b.float()).abs().max()))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prepared", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.hpu.set_device(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    with safe_open(args.prepared / f"pp0-tp{rank}.safetensors", framework="pt", device="cpu") as f:
        weight = torch.cat((f.get_tensor("layers.2.attn.compressor.wkv.weight"),
                            f.get_tensor("layers.2.attn.compressor.wgate.weight")), 0).float().to("hpu")
        fn = f.get_tensor("layers.2.hc_attn_fn").float().to("hpu")
        scale = f.get_tensor("layers.2.hc_attn_scale").float().to("hpu")
        base = f.get_tensor("layers.2.hc_attn_base").float().to("hpu")
        norm = f.get_tensor("layers.2.attn_norm.weight").to("hpu")

    def projection(hidden, previous):
        x, _, _, _ = hc_pre(hidden, previous, fn, scale, base, config["rms_norm_eps"], request_batch=True)
        x = rms_norm(x, norm, config["rms_norm_eps"], request_batch=True)
        result = torch.nn.functional.linear(x.float(), weight)
        return result[:, :512].contiguous(), result[:, 512:].contiguous()

    def pair(kh, sh, kv, score, positions, slots, fused):
        if fused:
            return torch.ops.custom_op.custom_deepseek_v41_compressor_batch_bf16_gaudi2(
                kh, sh, kv, score, positions, slots)
        active = (slots >= 0) & (positions >= 0)
        first = positions - positions.remainder(2)
        rows = torch.where(active, slots * 8 + positions.remainder(8), -1).int()
        kd, sd = write_state_rows(kh, kv, rows), write_state_rows(sh, score, rows)
        a = torch.where(active, slots * 8 + first.remainder(8), -1).int()
        b = torch.where(active, slots * 8 + (first + 1).remainder(8), -1).int()
        ka, kb = read_state_rows(kh, a, kd), read_state_rows(kh, b, kd)
        sa, sb = read_state_rows(sh, a, sd), read_state_rows(sh, b, sd)
        gates = torch.stack((sa, sb), 1).softmax(1)
        return (ka * gates[:, 0] + kb * gates[:, 1]).bfloat16()

    project = torch.compile(projection, backend="hpu_backend", fullgraph=True, dynamic=False)
    compute = torch.compile(pair, backend="hpu_backend", fullgraph=True, dynamic=False)
    report = []
    for batch in (1, 8, 32):
        torch.manual_seed(1188)
        positions = (torch.arange(batch, dtype=torch.int32) * 127 + 126).to("hpu")
        slots = torch.randperm(64)[:batch].int().to("hpu")
        for seed in (1188, 1189):
            torch.manual_seed(seed)
            hidden = torch.randn(batch, 4, 5120).bfloat16().to("hpu")
            previous = torch.rand(batch, 4).to("hpu")
            kv, score = project(hidden, previous)
            kh, sh = torch.randn(512, 512).to("hpu"), torch.randn(512, 512).to("hpu")
            original = kh.clone(), sh.clone()
            a = compute(kh, sh, kv, score, positions, slots, False).cpu()
            expected = kh.cpu(), sh.cpu()
            kh.copy_(original[0])
            sh.copy_(original[1])
            b = compute(kh, sh, kv, score, positions, slots, True).cpu()
            record = dict(batch=batch,
                          seed=seed,
                          pair=comparison(a, b),
                          history=[comparison(x, y) for x, y in zip(expected, (kh, sh), strict=True)])
            report.append(record)
            args.output.with_name(f"rank{rank}-" + args.output.name).write_text(json.dumps(report, indent=2))
            torch.save(dict(reference=a, candidate=b, kv=kv.cpu(), score=score.cpu()),
                       args.output.parent / f"rank{rank}-b{batch}-seed{seed}.pt")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
