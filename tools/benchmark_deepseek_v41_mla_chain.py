# SPDX-License-Identifier: Apache-2.0
"""C1 cache writer -> matrix attention -> real output projections.

Local compute diagnostic: no TP or PP communication. Includes every changed
gather, mask, cast and consumer; layer weights exceed on-chip cache capacity.
"""
import argparse
import json
import os
from pathlib import Path

from benchmark_deepseek_v41_projection_chains import benchmark
import torch
import torch.nn.functional as F

from deepseek_v41_micro_replay import RecipeRecorder
from vllm_gaudi.ops.deepseek_v41_math import quantize_activation, rotary_table
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu


def program(matrix, diagnostics=False):

    def chain(q, value, swa, main, packed, compressed, position, sink, scale, inverse, woa, woa_scale, wob):
        # main.shape[0] identifies the fixed per-layer ratio; no host read of
        # token metadata, scale or candidate indices occurs inside this graph.
        rows = main.shape[0] if main.shape[0] > 1 else 0
        ratio = 512 // rows if rows else 0
        done = torch.ops.custom_op.custom_deepseek_v41_swa_decoded_write_bf16_gaudi2(packed, value, position, swa, 0)
        ids, lengths = torch.ops.custom_op.custom_deepseek_v41_c1_indices_i32_gaudi2(position, compressed, ratio)
        args = (q, swa, main, ids, sink, scale, lengths, done, done, 0, rows, 512)
        if matrix:
            output = torch.ops.custom_op.custom_deepseek_v41_mla_mme_gaudi2(*args)
        else:
            output = torch.ops.custom_op.custom_deepseek_v41_decoded_attn_bf16_gaudi2(*args)[0]
        mla = output
        output = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(output, position, inverse)
        rope = output
        output = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_gaudi2(output.reshape(1, 4, 4096), woa, woa_scale)
        quant = quantize_activation(output)
        result = F.linear(quant, wob)
        return (mla, rope, output, quant, result) if diagnostics else result

    return chain


def validate_unchanged_consumer(arm, q, weights, actual, ordinary):
    # The unchanged wo_b BF16 MME can round a tie differently in eager and
    # compiled recipes. Verify every changed stage exactly, then isolate that
    # existing consumer at identical input bytes instead of relaxing MLA checks.
    fn = program(arm == "candidate", diagnostics=True)
    compiled = torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
    left = tuple(t.cpu() for t in fn(q, *weights))
    right = tuple(t.cpu() for t in compiled(q, *weights))
    if not all(torch.equal(a, b) for a, b in zip(left[:-1], right[:-1], strict=True)):
        raise AssertionError("Changed MLA/RoPE/wo_a/quant stages differ between ordinary and compile")
    if not torch.equal(right[-1], actual[0]) or not torch.equal(left[-1], ordinary[0]):
        raise AssertionError("Exposing stages changed the full consumer result; cause not isolated")
    hpu_input = left[-2].to("hpu")
    eager = F.linear(hpu_input, weights[-1]).cpu()
    replayed = torch.compile(F.linear, backend="hpu_backend", fullgraph=True, dynamic=False)(hpu_input,
                                                                                             weights[-1]).cpu()
    if not torch.equal(eager, ordinary[0]) or not torch.equal(replayed, actual[0]):
        raise AssertionError("Difference is not explained by unchanged wo_b at identical inputs")
    error = (actual[0].float() - ordinary[0].float()).abs()
    torch.testing.assert_close(actual[0], ordinary[0], rtol=.008, atol=1e-7)
    return {
        "arm": arm,
        "changed_stages_bitwise_equal": True,
        "unchanged_wob_eager_compiled_difference_reproduced": True,
        "different_outputs": int((actual[0] != ordinary[0]).sum()),
        "max_abs": error.max().item(),
        "quality_not_qualified": True
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--position", type=int, required=True, choices=(63, 191, 511))
    parser.add_argument("--pp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--diagnostic-input", type=Path)
    args = parser.parse_args()
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    bind_worker_cpu(0)
    torch.manual_seed(71933)
    recorder = RecipeRecorder(output)
    shard = PreparedV41Shard(args.prepared, args.pp_rank, 0)
    sidecar = WoaFP8Sidecar(args.sidecar, shard)
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    inputs, weights = [], []
    with torch.inference_mode():
        for layer in range(args.pp_rank * 20, (args.pp_rank + 1) * 20):
            ratio = config["compress_ratios"][layer]
            prefix = f"layers.{layer}.attn."
            rows = 512 // ratio if ratio else 0
            swa = torch.randn(512, 512).bfloat16().to("hpu")
            main_kv = torch.randn(max(1, rows), 512).bfloat16().to("hpu")
            packed = torch.zeros(512, 528, dtype=torch.uint8, device="hpu")
            compressed = torch.arange(512, dtype=torch.int32).reshape(1, -1)
            visible = (args.position + 1) // ratio if ratio else 0
            compressed[compressed >= visible] = -1
            scaling = config["rope_scaling"]
            table = rotary_table(64, 512, config["compress_rope_theta"] if ratio else config["rope_theta"],
                                 scaling["original_max_position_embeddings"] if ratio else 0, scaling["factor"],
                                 scaling["beta_fast"], scaling["beta_slow"])
            inverse = torch.cat((table[..., 0], -table[..., 1]), -1).contiguous().to("hpu")
            weights.append(
                (torch.randn(1, 512).bfloat16().to("hpu"), swa, main_kv, packed, compressed.to("hpu"),
                 torch.tensor([args.position], dtype=torch.int32,
                              device="hpu"), shard.tensor(prefix + "attn_sink", "hpu").float(),
                 torch.tensor([512**-.5], device="hpu"), inverse, sidecar.tensor(prefix + "wo_a.weight", "hpu"),
                 sidecar.tensor(prefix + "wo_a.channel_scale", "hpu"), shard.dense(prefix + "wo_b.weight", "hpu")))
            inputs.append(torch.randn(1, 32, 512).bfloat16().to("hpu"))
        if args.diagnose:
            q, w = inputs[0], weights[0]
            if args.diagnostic_input:
                q.copy_(torch.load(args.diagnostic_input, weights_only=True)["input"].to("hpu"))
            else:
                q.mul_(1.25).mul_(1.25)
            full = torch.compile(program(True), backend="hpu_backend", fullgraph=True, dynamic=False)
            full_output = full(q, *w).cpu()
            fn = program(True, True)
            compiled = torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
            ordinary = tuple(t.cpu() for t in fn(q, *w))
            actual = tuple(t.cpu() for t in compiled(q, *w))
            rows = []
            for stage, left, right in zip(("mla", "rope", "woa", "quant", "wob"), ordinary, actual, strict=True):
                error = (left.float() - right.float()).abs()
                rows.append({
                    "stage": stage,
                    "different": int((left != right).sum()),
                    "max_abs": error.max().item(),
                    "rmse": error.square().mean().sqrt().item()
                })
            full_error = (full_output.float() - ordinary[-1].float()).abs()
            rows.append({
                "stage": "full_vs_ordinary",
                "different": int((full_output != ordinary[-1]).sum()),
                "max_abs": full_error.max().item()
            })
            (output / "stage-comparison.json").write_text(json.dumps(rows, indent=2))
            torch.save({"ordinary": ordinary, "compiled": actual}, output / "stages.pt")
            print(rows, flush=True)
            torch.distributed.destroy_process_group()
            return
        result = benchmark(f"mla-output-p{args.pp_rank}-pos{args.position}",
                           program(False),
                           program(True),
                           inputs,
                           weights,
                           weights,
                           output,
                           3,
                           recorder,
                           not args.include_reference,
                           ordinary_validator=validate_unchanged_consumer)
        result["state_scope"] = ("real SWA writer and C1 index preparation; representative decoded history; "
                                 "static compressed publication, no compressor/TP/PP in this component")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
