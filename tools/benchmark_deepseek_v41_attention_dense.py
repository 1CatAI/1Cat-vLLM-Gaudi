# SPDX-License-Identifier: Apache-2.0
"""Real Attention projection/MLA chain ending at the materialized TP operand.

No TP/PP is timed here. The terminal BF16 output is persistent, matching the
production collective boundary; compiler operand placement must also be checked.
"""
import argparse
from functools import partial
import json
import os
from pathlib import Path

from benchmark_deepseek_v41_projection_chains import benchmark
import torch
import torch.nn.functional as F

from deepseek_v41_micro_replay import RecipeRecorder
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
from vllm_gaudi.ops.deepseek_v41_math import quantize_activation, rms_norm, rotary_table
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar


def program(fp8, diagnostics=False, kv_first=False, fused_norm=False):

    def project(value, weight, scale):
        value = quantize_activation(value)
        if fp8:
            return torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(value, weight, scale)
        return F.linear(value, weight)

    def chain(value, wqa, qnorm, wqb, wqb_scale, wkv, kvnorm, swa, main, packed, compressed, position,
              sink, scale, forward, inverse, woa, woa_scale, wob, wob_scale):
        norm = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2 if fused_norm else rms_norm
        latent = norm(F.linear(quantize_activation(value), wqa), qnorm, 1e-20)
        if not kv_first:
            expanded = project(latent, wqb, wqb_scale).reshape(1, 32, 512)
            q = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(expanded, position, forward)
        kv = norm(F.linear(quantize_activation(value), wkv), kvnorm, 1e-20)
        kv = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(
            kv.reshape(1, 1, 512), position, forward).reshape(1, 512)
        rows = main.shape[0] if main.shape[0] > 1 else 0
        ratio = 512 // rows if rows else 0
        done = torch.ops.custom_op.custom_deepseek_v41_swa_decoded_write_bf16_gaudi2(packed, kv, position, swa, 0)
        ids, lengths = torch.ops.custom_op.custom_deepseek_v41_c1_indices_i32_gaudi2(position, compressed, ratio)
        if kv_first:
            expanded = project(latent, wqb, wqb_scale).reshape(1, 32, 512)
            q = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(expanded, position, forward)
        output = torch.ops.custom_op.custom_deepseek_v41_mla_mme_gaudi2(
            q, swa, main, ids, sink, scale, lengths, done, done, 0, rows)
        mla = output
        output = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(output, position, inverse)
        output = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_gaudi2(output.reshape(1, 4, 4096), woa, woa_scale)
        result = project(output, wob, wob_scale)
        return (latent, expanded, q, kv, mla, output, quantize_activation(output), result) if diagnostics else result

    return chain


def validate_terminal_mme(arm, value, weights, actual, ordinary, candidate_fp8=True):
    use_fp8 = arm == "candidate" and candidate_fp8
    fn = program(use_fp8, diagnostics=True,
                 kv_first=arm == "candidate" and os.environ.get("VLLM_HPU_DSV41_ATTN_KV_FIRST") == "1",
                 fused_norm=arm == "candidate" and os.environ.get("VLLM_HPU_DSV41_ATTN_FUSED_NORM") == "1")
    compiled = torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
    eager = tuple(t.cpu() for t in fn(value, *weights))
    graph = tuple(t.cpu() for t in compiled(value, *weights))
    if not all(torch.equal(a, b) for a, b in zip(eager[:-1], graph[:-1], strict=True)):
        raise AssertionError("Pre-consumer stages differ; terminal MME does not explain the discrepancy")
    assert torch.equal(eager[-1], ordinary[0]) and torch.equal(graph[-1], actual[0])
    x = eager[-2].to("hpu")

    def consumer(x, w, scale):
        if use_fp8:
            return torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(x, w, scale)
        return F.linear(x, w)

    standalone = consumer(x, *weights[-2:]).cpu()
    prepared = torch.compile(consumer, backend="hpu_backend", fullgraph=True,
                             dynamic=False)(x, *weights[-2:]).cpu()
    assert torch.equal(standalone, ordinary[0]) and torch.equal(prepared, actual[0])
    return {"arm": arm, "terminal_mme_difference_reproduced_at_identical_input": True,
            "all_preconsumer_stages_equal": True, "different": int((actual[0] != ordinary[0]).sum()),
            "maximum_absolute_difference": float((actual[0].float() - ordinary[0].float()).abs().max()),
            "quality_not_qualified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("woa_sidecar", type=Path)
    parser.add_argument("dense_sidecar", type=Path)
    parser.add_argument("--pp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--position", type=int, choices=(63, 191, 511), default=191)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--candidate-bf16", action="store_true",
                        help="Isolate normalization fusion with the retained BF16 dense projections")
    parser.add_argument("--diagnostic-input", type=Path)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    candidate_fp8 = not args.candidate_bf16
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    bind_worker_cpu(0)
    torch.manual_seed(67143)
    recorder = RecipeRecorder(output)
    shard = PreparedV41Shard(args.prepared, args.pp_rank, 0)
    woa, dense = WoaFP8Sidecar(args.woa_sidecar, shard), DenseFP8Sidecar(args.dense_sidecar, shard)
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    assert config["rms_norm_eps"] == 1e-20
    inputs, old, new = [], [], []
    with torch.inference_mode():
        for layer in range(args.pp_rank * 20, (args.pp_rank + 1) * 20):
            ratio = config["compress_ratios"][layer]
            prefix = f"layers.{layer}.attn."
            rows = 512 // ratio if ratio else 0
            compressed = torch.arange(512, dtype=torch.int32).reshape(1, -1)
            compressed[compressed >= ((args.position + 1) // ratio if ratio else 0)] = -1
            scaling = config["rope_scaling"]
            table = rotary_table(64, 512, config["compress_rope_theta"] if ratio else config["rope_theta"],
                                 scaling["original_max_position_embeddings"] if ratio else 0, scaling["factor"],
                                 scaling["beta_fast"], scaling["beta_slow"])
            before = [shard.dense(prefix + "wq_a.weight", "hpu"), shard.tensor(prefix + "q_norm.weight", "hpu")]
            common = [shard.dense(prefix + "wkv.weight", "hpu"), shard.tensor(prefix + "kv_norm.weight", "hpu"),
                      torch.randn(512, 512).bfloat16().to("hpu"),
                      torch.randn(max(1, rows), 512).bfloat16().to("hpu"),
                      torch.zeros(512, 528, dtype=torch.uint8, device="hpu"), compressed.to("hpu"),
                      torch.tensor([args.position], dtype=torch.int32, device="hpu"),
                      shard.tensor(prefix + "attn_sink", "hpu").float(), torch.tensor([512**-.5], device="hpu"),
                      torch.cat((table[..., 0], table[..., 1]), -1).contiguous().to("hpu"),
                      torch.cat((table[..., 0], -table[..., 1]), -1).contiguous().to("hpu"),
                      woa.tensor(prefix + "wo_a.weight", "hpu"), woa.tensor(prefix + "wo_a.channel_scale", "hpu")]
            q_scale = dense.tensor(prefix + "wq_b.channel_scale", "hpu")
            o_scale = dense.tensor(prefix + "wo_b.channel_scale", "hpu")
            candidate_weight = dense.tensor if candidate_fp8 else shard.dense
            new.append(tuple(before + [candidate_weight(prefix + "wq_b.weight", "hpu"), q_scale] + common +
                             [candidate_weight(prefix + "wo_b.weight", "hpu"), o_scale]))
            if args.include_reference:
                old.append(tuple(before + [shard.dense(prefix + "wq_b.weight", "hpu"), q_scale] + common +
                                 [shard.dense(prefix + "wo_b.weight", "hpu"), o_scale]))
            inputs.append(torch.randn(1, 5120).bfloat16().to("hpu"))
        if args.diagnostic_input:
            saved = torch.load(args.diagnostic_input, weights_only=True)
            inputs[0].copy_(saved["input"].to("hpu"))
            audit = []
            for arm, weights in ((False, old), (True, new)):
                fn = program(arm and candidate_fp8, diagnostics=True,
                             kv_first=arm and os.environ.get("VLLM_HPU_DSV41_ATTN_KV_FIRST") == "1",
                             fused_norm=arm and os.environ.get("VLLM_HPU_DSV41_ATTN_FUSED_NORM") == "1")
                compiled = torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
                ordinary = tuple(t.cpu() for t in fn(inputs[0], *weights[0]))
                actual = tuple(t.cpu() for t in compiled(inputs[0], *weights[0]))
                torch.save({"ordinary": ordinary, "compiled": actual}, output / f"stages-{arm}.pt")
                for name, a, b in zip(("latent", "expanded", "rope_q", "kv", "mla", "woa", "quant", "wob"),
                                      ordinary, actual, strict=True):
                    audit.append({"candidate": arm, "stage": name, "different": int((a != b).sum()),
                                  "max_abs": float((a.float()-b.float()).abs().max())})
            (output / "stage-comparison.json").write_text(json.dumps(audit, indent=2))
            print(json.dumps(audit, indent=2), flush=True)
            torch.distributed.destroy_process_group()
            return
        candidate = program(candidate_fp8, kv_first=os.environ.get("VLLM_HPU_DSV41_ATTN_KV_FIRST") == "1",
                            fused_norm=os.environ.get("VLLM_HPU_DSV41_ATTN_FUSED_NORM") == "1")
        result = benchmark(f"attention-dense-p{args.pp_rank}-pos{args.position}", program(False), candidate,
                           inputs, old, new, output, 3, recorder, not args.include_reference,
                           ordinary_validator=partial(validate_terminal_mme, candidate_fp8=candidate_fp8),
                           profile_only=args.profile_only)
        result["candidate_precision"] = "fp8" if candidate_fp8 else "bf16"
        result["candidate_fused_norm"] = os.environ.get("VLLM_HPU_DSV41_ATTN_FUSED_NORM") == "1"
        result["state_scope"] = ("real Q/KV input projections/norms, SWA writer, C1 indices, shared-KV MME MLA, "
                                 "inverse RoPE and wo_a/wo_b to persistent BF16 TP operand; "
                                 "seeded decoded history/static compressed publication; no compressor, TP or PP")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
