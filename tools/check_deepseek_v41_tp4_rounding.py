# SPDX-License-Identifier: Apache-2.0
"""Locate the first grouped-prefill BF16 boundary differing from eager math."""
import json
import os
from pathlib import Path


def main():
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepared = Path("/opt/optane/prepared/DeepSeek-V4.1-Flash-dba1be0-tp4-pp1-q16v2")
    prepare_environment(prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import torch.nn.functional as F
    import habana_frameworks.torch.core  # noqa: F401
    torch.hpu.set_device(0)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_expert_n256 import load_projection
    from vllm_gaudi.ops.deepseek_v41_grouped_prefill import compiled_project
    from vllm_gaudi.ops.deepseek_v41_route_blocks import device_route_blocks
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from tools.check_deepseek_v41_tp4_components import read_experts, decode_reference
    shard = PreparedV41Shard(prepared, 0, 1)
    q13, s13, _ = load_projection(shard, "layers.0.ffn.experts.w13", "hpu")
    q2, s2, _ = load_projection(shard, "layers.0.ffn.experts.w2", "hpu")
    lookup = mxfp4_bf16_lut("hpu")
    cpu_weights = [decode_reference(read_experts(shard, f"layers.0.ffn.experts.{p}_q16"),
                                    read_experts(shard, f"layers.0.ffn.experts.{p}_s16")) for p in ("w13", "w2")]
    with torch.inference_mode():
        torch.manual_seed(4107)
        value = (torch.randn(128, 5120) * 8).bfloat16()
        ids = (torch.arange(768).reshape(128, 6) + 3).remainder(2).int()
        routing = torch.rand(128, 6)
        routing = routing / routing.sum(-1, keepdim=True) * 1.5
        experts, slots, _, counts = device_route_blocks(ids, 384, 128)
        active = int(((counts + 127) // 128).sum())
        experts, slots = experts[:, :active].clone(), slots[:active].clone()
        selected = value[(slots.clamp(min=0) // 6)].to("hpu")
        route = routing.flatten()[slots.clamp(min=0)].masked_fill(slots < 0, 0).to("hpu")
        expert_ids = experts.to("hpu")
        weights = [w[experts.flatten().long()].transpose(-1, -2).contiguous().to("hpu") for w in cpu_weights]

        def stages(selected, route, expert_ids, q13, q2, s13, s2, lookup):
            w13 = torch.ops.custom_op.custom_deepseek_v41_expert_n256_bf16_gaudi2(
                expert_ids, q13, s13, lookup, True)
            projected = torch.bmm(selected, w13)
            gate, up = projected.float().chunk(2, -1)
            middle = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10) * route[..., None]).bfloat16()
            rounded = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(
                middle.reshape(1, -1)).reshape_as(middle)
            w2 = torch.ops.custom_op.custom_deepseek_v41_expert_n256_bf16_gaudi2(
                expert_ids, q2, s2, lookup, True)
            return w13, projected, rounded, torch.bmm(rounded, w2)

        args = selected, route, expert_ids, q13, q2, s13, s2, lookup
        eager = stages(*args)
        compiled = torch.compile(stages, backend="hpu_backend", fullgraph=True, dynamic=False)(*args)
        canonical_projected = torch.bmm(selected, weights[0])
        gate, up = canonical_projected.float().chunk(2, -1)
        canonical_middle = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10) * route[..., None]).bfloat16()
        canonical = weights[0], canonical_projected, canonical_middle, torch.bmm(canonical_middle, weights[1])
        original = compiled_project((active, 128, 384, True))(*args, True).reshape(active, 128, 5120)
        record = {}
        saved = {"original": original.cpu()}
        for name, sequence in (("eager", eager), ("compiled", compiled), ("canonical", canonical)):
            for label, tensor in zip(("weight13", "projected", "middle", "down"), sequence):
                saved[name + "_" + label] = tensor.cpu()
        for label in ("weight13", "projected", "middle", "down"):
            base = saved["canonical_" + label].float()
            for name in ("eager", "compiled"):
                actual = saved[name + "_" + label].float()
                record[name + "_" + label] = {"different": int((actual != base).sum()),
                                                "max_abs": float((actual - base).abs().max())}
        for name in ("eager", "compiled", "canonical"):
            diff = saved["original"].float() - saved[name + "_down"].float()
            record["original_vs_" + name] = {"different": int((diff != 0).sum()), "max_abs": float(diff.abs().max())}
        root = Path(os.environ["DSV41_RUN_EVIDENCE"])
        torch.save(saved, root / "boundaries.pt")
        (root / "boundaries.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
