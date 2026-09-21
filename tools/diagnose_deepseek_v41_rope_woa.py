# SPDX-License-Identifier: Apache-2.0
"""Isolate inverse-RoPE and fused inverse-RoPE/wo_a numerical parity."""

import argparse
import json
import os
from pathlib import Path

import torch

from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
from vllm_gaudi.ops.deepseek_v41_math import rotary_table
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar


def comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int | bool]:
    delta = candidate.float() - reference.float()
    return {
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "different": int((reference != candidate).sum()),
        "elements": reference.numel(),
        "maximum_absolute_difference": float(delta.abs().max()),
        "rmse": float(delta.square().mean().sqrt()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("woa_sidecar", type=Path)
    parser.add_argument("--pp-rank", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    bind_worker_cpu(0)
    torch.manual_seed(67143)
    shard = PreparedV41Shard(args.prepared, args.pp_rank, 0)
    sidecar = WoaFP8Sidecar(args.woa_sidecar, shard)
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    layer = args.pp_rank * 20
    prefix = f"layers.{layer}.attn."
    weight = sidecar.tensor(prefix + "wo_a.weight", "hpu")
    scale = sidecar.tensor(prefix + "wo_a.channel_scale", "hpu")
    scaling = config["rope_scaling"]
    table = rotary_table(64, 512, config["rope_theta"],
                         scaling["original_max_position_embeddings"], scaling["factor"],
                         scaling["beta_fast"], scaling["beta_slow"])
    forward = torch.cat((table[..., 0], table[..., 1]), -1).contiguous().to("hpu")
    signed_inverse = torch.cat((table[..., 0], -table[..., 1]), -1).contiguous().to("hpu")
    records = []
    with torch.inference_mode():
        for position_value in (0, 63, 191, 511):
            position = torch.tensor([position_value], dtype=torch.int32, device="hpu")
            value = torch.randn(1, 32, 512, dtype=torch.bfloat16, device="hpu")
            reference_rope = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(
                value, position, signed_inverse)
            native_inverse = torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(
                value, position, forward)
            reference_quant, reference_scale = torch.ops.custom_op.custom_deepseek_v41_woa_quant_gaudi2(
                reference_rope.reshape(1, 4, 4096))
            fused_quant, fused_scale = torch.ops.custom_op.custom_deepseek_v41_rope_woa_quant_gaudi2(
                value, position, forward)
            reference = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2(
                reference_rope.reshape(1, 4, 4096), weight, scale)
            fused_forward = torch.ops.custom_op.custom_deepseek_v41_rope_woa_fp8_roundtrip_gaudi2(
                value, weight, scale, position, forward)
            fused_signed_inverse = torch.ops.custom_op.custom_deepseek_v41_rope_woa_fp8_roundtrip_gaudi2(
                value, weight, scale, position, signed_inverse)
            (reference_rope, native_inverse, reference_quant, fused_quant, reference_scale, fused_scale,
             reference, fused_forward, fused_signed_inverse) = (
                tensor.cpu() for tensor in
                (reference_rope, native_inverse, reference_quant, fused_quant, reference_scale, fused_scale,
                 reference, fused_forward, fused_signed_inverse))
            quant_mask = reference_quant != fused_quant
            quant_coordinates = quant_mask.nonzero()
            logical_offsets = quant_coordinates[:, -1]
            tail_mask = logical_offsets.remainder(512) >= 448
            sample_coordinates = quant_coordinates[:16]
            sample_reference = reference_quant[quant_mask][:16].float()
            sample_fused = fused_quant[quant_mask][:16].float()
            records.append({
                "position": position_value,
                "standalone_inverse": comparison(reference_rope, native_inverse),
                "fused_quant": comparison(reference_quant, fused_quant),
                "fused_quant_location": {
                    "different_in_rope_tail": int(tail_mask.sum()),
                    "different_outside_rope_tail": int((~tail_mask).sum()),
                    "sample_coordinates": sample_coordinates.tolist(),
                    "sample_reference": sample_reference.tolist(),
                    "sample_fused": sample_fused.tolist(),
                },
                "fused_scale": comparison(reference_scale, fused_scale),
                "fused_with_forward_phase": comparison(reference, fused_forward),
                "fused_with_signed_inverse_phase": comparison(reference, fused_signed_inverse),
            })
    result = {
        "layer": layer,
        "input": "seeded changing BF16 [1,32,512]",
        "reference": "forward RoPE kernel with signed inverse table, then wo_a roundtrip",
        "records": records,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
