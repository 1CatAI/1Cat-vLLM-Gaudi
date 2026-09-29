# SPDX-License-Identifier: Apache-2.0
"""Numerical bisection of FFN norm, prequant binding and routed consumers.

This diagnostic does not qualify performance or change production boundaries.
Each rank consumes its real checkpoint shard without TP communication so that
local expert differences can be separated from communication and state reuse.
"""
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

from vllm_gaudi.models.deepseek_v41_program import (  # noqa: E402
    PreparedMoE, _weight_tree, load_weight_tree,
)
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_math import rms_norm  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402


def compare(a, b):
    a, b = a.cpu(), b.cpu()
    if a.dtype == torch.float8_e4m3fn:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return dict(exact=torch.equal(a, b),
                changed=int((a != b).sum()),
                total=a.numel(),
                max_abs=float((a.float() - b.float()).abs().max()))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prepared", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--input-fixture",
                   type=Path,
                   help="Directory with saved rankN-sublayers.pt; use its real FFN collapsed input")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.hpu.set_device(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    ops = torch.ops.custom_op
    cfg = json.loads((args.prepared / "config.json").read_text())["text_config"]
    if not 0 <= args.layer < cfg["num_hidden_layers"]:
        p.error("Layer is outside the target model")
    shard = PreparedV41Shard(args.prepared, args.layer // 20, rank)
    specs = {
        k: v
        for k, v in shard.specs.items() if k.startswith((f"layers.{args.layer}.ffn.", f"layers.{args.layer}.ffn_norm."))
    }
    tree = _weight_tree(specs)
    load_weight_tree(shard, tree, "hpu", specs)
    weights = tree.layers.get_submodule(str(args.layer))
    moe = PreparedMoE(weights.ffn, 6, True, mxfp4_bf16_lut("hpu"), lambda x, **kwargs: x)
    moe.prepare_shared_gate_up_weight()
    b, eps = args.batch, cfg["rms_norm_eps"]

    def preparation(x, fused):
        if fused:
            return ops.custom_deepseek_v41_ffn_norm_quant_gaudi2(x, weights.ffn_norm.weight, eps)
        normalized = rms_norm(x, weights.ffn_norm.weight, eps, request_batch=True)
        quantized, scale = ops.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(normalized)
        return normalized, quantized, scale

    def expert(x, ids, route, external):
        prequant = ops.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(x) if external else None
        return moe._forward_n256_fp8(x, ids, route, ordinary_decode=True, prequant=prequant)

    def full(x, mask, fused):
        normalized, quantized, scale = preparation(x, fused)
        return moe(normalized, mask, ordinary_decode=True, prequant=(quantized, scale) if fused else None)

    def compiled(fn):
        return torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)

    prep = compiled(preparation)
    local = compiled(expert)
    chain = compiled(full)
    output = args.output.with_name(f"rank{rank}-" + args.output.name)
    report = []
    fixture = None
    if args.input_fixture is not None:
        saved = torch.load(args.input_fixture / f"rank{rank}-sublayers.pt", weights_only=True)
        fixture = saved["actual"][7]
        if fixture.shape != (b, 5120) or fixture.dtype != torch.bfloat16:
            raise ValueError("Saved FFN collapsed input has an incompatible shape or dtype")
        if not torch.equal(fixture, saved["reference"][7]):
            raise ValueError("Saved FFN fixture already differs before normalization")
    for seed in ((1188, ) if fixture is not None else (1188, 1189)):
        torch.manual_seed(seed)
        x = (fixture if fixture is not None else
             (torch.randn(b, 5120) * torch.linspace(0.01, 8, b)[:, None]).bfloat16()).to("hpu")
        ids = torch.stack([torch.randperm(384)[:6] for _ in range(b)]).int().to("hpu")
        route = torch.rand(b, 6)
        route = (1.5 * route / route.sum(-1, keepdim=True)).to("hpu")
        mask = (torch.zeros(b, dtype=torch.bool) if fixture is not None else (torch.arange(b) % 2).bool()).to("hpu")
        old, new = prep(x, False), prep(x, True)
        internal = local(old[0], ids, route, False)
        external = local(old[0], ids, route, True)
        norm_changed = local(new[0], ids, route, False)
        original, fused = chain(x, mask, False), chain(x, mask, True)
        result = dict(seed=seed,
                      batch=b,
                      preparation=[compare(a, z) for a, z in zip(old, new, strict=True)],
                      same_input_expert=compare(internal, external),
                      normalized_input_expert=compare(internal, norm_changed),
                      full_local_moe=compare(original, fused))
        if fixture is not None:
            serial = torch.cat([chain(x[row:row + 1], mask[row:row + 1], True).clone() for row in range(b)])
            result.update(layer=args.layer,
                          fixture=str(args.input_fixture),
                          c1_vs_generic_batch=compare(serial, original),
                          c1_vs_fused_batch=compare(serial, fused))
            torch.save(
                dict(input=x.cpu(),
                     generic=original.cpu(),
                     fused=fused.cpu(),
                     serial=serial.cpu(),
                     normalized=old[0].cpu(),
                     fused_normalized=new[0].cpu()), args.output.parent / f"rank{rank}-ffn-contract.pt")
        report.append(result)
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(result), flush=True)
    torch.hpu.synchronize()


if __name__ == "__main__":
    main()
