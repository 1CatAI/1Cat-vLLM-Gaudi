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
from vllm_gaudi.ops.deepseek_v41_qkv import concatenate_static_weights
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
            q, swa, main, ids, sink, scale, lengths, done, done, 0, rows, 512)
        mla = output
        output = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(output, position, inverse)
        output = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_gaudi2(output.reshape(1, 4, 4096), woa, woa_scale)
        result = project(output, wob, wob_scale)
        return (latent, expanded, q, kv, mla, output, quantize_activation(output), result) if diagnostics else result

    return chain


def production_program(fused_rope_woa=False, diagnostics=False, fused_qnorm=False,
                       fused_kvnorm=False, bf16_pv=False, fused_mla_woa=False,
                       fused_mla_woa_wob=False):
    """Mirror the qualified paged C1 Attention projection/MLA chain.

    The older comparison program intentionally keeps its historical split
    Q/KV and projection variants.  Production has since moved to one static
    QKV input matrix, the Q projection/RoPE compound op, the exact wo_a
    group-32 roundtrip, and its FP8 wo_b consumer.  Keep this as a separate
    mode so profiling the retained path does not silently measure obsolete
    work.
    """

    def chain(value, qkv_weight, qnorm, kvnorm, wqb, wqb_scale, swa, main,
              packed, compressed, position, sink, scale, forward, inverse,
              woa, woa_scale, wob, wob_scale):
        projected = F.linear(quantize_activation(value), qkv_weight)
        query_input = projected[..., :1280].contiguous()
        kv_input = projected[..., 1280:].contiguous()
        norm = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2
        if fused_qnorm:
            q = torch.ops.custom_op.custom_deepseek_v41_q_norm_projection_rope_gaudi2(
                query_input, qnorm, wqb, wqb_scale, position, forward, 1e-20).reshape(1, 32, 512)
        else:
            latent = norm(query_input, qnorm, 1e-20)
            q = torch.ops.custom_op.custom_deepseek_v41_q_projection_rope_gaudi2(
                latent, wqb, wqb_scale, position, forward).reshape(1, 32, 512)
        if fused_kvnorm:
            kv = torch.ops.custom_op.custom_deepseek_v41_kv_norm_rope_bf16_gaudi2(
                kv_input, kvnorm, position, forward, 1e-20)
        else:
            kv = norm(kv_input, kvnorm, 1e-20)
            kv = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(
                kv.reshape(1, 1, 512), position, forward).reshape(1, 512)
        rows = main.shape[0] if main.shape[0] > 1 else 0
        ratio = 512 // rows if rows else 0
        done = torch.ops.custom_op.custom_deepseek_v41_swa_decoded_write_bf16_gaudi2(
            packed, kv, position, swa, 0)
        ids, lengths = torch.ops.custom_op.custom_deepseek_v41_c1_indices_i32_gaudi2(
            position, compressed, ratio)
        if fused_mla_woa_wob:
            output = torch.ops.custom_op.custom_deepseek_v41_mla_woa_wob_fp8_roundtrip_gaudi2(
                q, swa, main, ids, sink, scale, lengths, done, done,
                0, rows, 512, woa, woa_scale, position, forward, wob,
                wob_scale)
        elif fused_mla_woa:
            output = torch.ops.custom_op.custom_deepseek_v41_mla_woa_fp8_roundtrip_gaudi2(
                q, swa, main, ids, sink, scale, lengths, done, done,
                0, rows, 512, woa, woa_scale, position, forward)
        else:
            mla = (torch.ops.custom_op.custom_deepseek_v41_mla_bf16_pv_gaudi2
                   if bf16_pv else
                   torch.ops.custom_op.custom_deepseek_v41_mla_mme_gaudi2)
            output = mla(q, swa, main, ids, sink, scale, lengths, done, done,
                         0, rows, 512)
        if fused_mla_woa_wob:
            return output
        if fused_mla_woa:
            pass
        elif fused_rope_woa:
            output = torch.ops.custom_op.custom_deepseek_v41_rope_woa_fp8_roundtrip_gaudi2(
                output, woa, woa_scale, position, forward)
        else:
            output = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(
                output, position, inverse)
            output = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2(
                output.reshape(1, 4, 4096), woa, woa_scale)
        result = torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(
            output, wob, wob_scale)
        return (q, kv, output, result) if diagnostics else result

    return chain


def validate_terminal_mme(arm,
                          value,
                          weights,
                          actual,
                          ordinary,
                          candidate_fp8=True,
                          force_fp8=False,
                          compare_kv_order=False):
    use_fp8 = force_fp8 or (arm == "candidate" and candidate_fp8)
    fn = program(use_fp8, diagnostics=True,
                 kv_first=(arm == "candidate" and
                           (compare_kv_order or os.environ.get("VLLM_HPU_DSV41_ATTN_KV_FIRST") == "1")),
                 fused_norm=(force_fp8 or arm == "candidate")
                 and os.environ.get("VLLM_HPU_DSV41_ATTN_FUSED_NORM") == "1")
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
    parser.add_argument("--position", type=int, choices=(0, 63, 191, 511), default=191)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--candidate-bf16", action="store_true",
                        help="Isolate normalization fusion with the retained BF16 dense projections")
    parser.add_argument("--compare-kv-order", action="store_true",
                        help="Compare the accepted FP8 chain with KV/cache production scheduled before Q")
    parser.add_argument("--production-parity", action="store_true",
                        help="Profile the retained paged C1 production chain instead of historical variants")
    parser.add_argument("--compare-production-rope-woa", action="store_true",
                        help="Compare the production chain with exact inverse-RoPE/wo_a fusion")
    parser.add_argument("--compare-production-qnorm", action="store_true",
                        help="Compare production with exact QNorm/dynamic-quant fusion")
    parser.add_argument("--compare-production-kvnorm", action="store_true",
                        help="Compare production with exact KVNorm/forward-RoPE fusion")
    parser.add_argument("--compare-production-bf16-pv", action="store_true",
                        help="Compare production FP32 PV against shared-KV BF16 PV through wo_b")
    parser.add_argument("--compare-production-mla-woa", action="store_true",
                        help="Compare production with FP32-PV/BF16-boundary/inverse-RoPE/wo_a fusion")
    parser.add_argument("--compare-production-mla-woa-wob", action="store_true",
                        help="Compare retained Attention compounds with an exact internal wo_a to wo_b handoff")
    parser.add_argument("--compare-production-combined", action="store_true",
                        help="Compare retained KVNorm production with exact QNorm and MLA/wo_a fusions")
    parser.add_argument("--diagnostic-input", type=Path)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    if args.production_parity and (args.include_reference or args.candidate_bf16 or args.compare_kv_order
                                   or args.compare_production_rope_woa or args.compare_production_qnorm
                                   or args.compare_production_kvnorm
                                   or args.compare_production_bf16_pv
                                   or args.compare_production_mla_woa
                                   or args.compare_production_mla_woa_wob
                                   or args.compare_production_combined
                                   or args.diagnostic_input):
        parser.error("--production-parity is a candidate-only mode")
    if args.compare_production_rope_woa and (args.include_reference or args.candidate_bf16
                                             or args.compare_kv_order or args.compare_production_qnorm
                                             or args.compare_production_kvnorm
                                             or args.compare_production_bf16_pv
                                             or args.compare_production_mla_woa
                                             or args.compare_production_mla_woa_wob
                                             or args.compare_production_combined
                                             or args.diagnostic_input):
        parser.error("--compare-production-rope-woa cannot be combined with historical comparison modes")
    if args.compare_production_qnorm and (args.include_reference or args.candidate_bf16
                                          or args.compare_kv_order or args.compare_production_kvnorm
                                          or args.compare_production_bf16_pv
                                          or args.compare_production_mla_woa
                                          or args.compare_production_mla_woa_wob
                                          or args.compare_production_combined
                                          or args.diagnostic_input):
        parser.error("--compare-production-qnorm cannot be combined with historical comparison modes")
    if args.compare_production_kvnorm and (args.include_reference or args.candidate_bf16
                                           or args.compare_kv_order or args.compare_production_bf16_pv
                                           or args.compare_production_mla_woa
                                           or args.compare_production_mla_woa_wob
                                           or args.compare_production_combined
                                           or args.diagnostic_input):
        parser.error("--compare-production-kvnorm cannot be combined with historical comparison modes")
    if args.compare_production_bf16_pv and (args.include_reference or args.candidate_bf16
                                            or args.compare_kv_order or args.compare_production_mla_woa
                                            or args.compare_production_mla_woa_wob
                                            or args.compare_production_combined
                                            or args.diagnostic_input):
        parser.error("--compare-production-bf16-pv cannot be combined with historical comparison modes")
    if args.compare_production_mla_woa and (args.include_reference or args.candidate_bf16
                                             or args.compare_kv_order
                                             or args.compare_production_mla_woa_wob
                                             or args.compare_production_combined
                                             or args.diagnostic_input):
        parser.error("--compare-production-mla-woa cannot be combined with historical comparison modes")
    if args.compare_production_mla_woa_wob and (args.include_reference or args.candidate_bf16
                                                 or args.compare_kv_order
                                                 or args.compare_production_combined
                                                 or args.diagnostic_input):
        parser.error("--compare-production-mla-woa-wob cannot be combined with historical comparison modes")
    if args.compare_production_combined and (args.include_reference or args.candidate_bf16
                                              or args.compare_kv_order or args.diagnostic_input):
        parser.error("--compare-production-combined cannot be combined with historical comparison modes")
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
            wqa = shard.dense(prefix + "wq_a.weight", "hpu")
            wkv = shard.dense(prefix + "wkv.weight", "hpu")
            before = [wqa, shard.tensor(prefix + "q_norm.weight", "hpu")]
            common = [wkv, shard.tensor(prefix + "kv_norm.weight", "hpu"),
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
            if (args.production_parity or args.compare_production_rope_woa or
                    args.compare_production_qnorm or args.compare_production_kvnorm or
                    args.compare_production_bf16_pv or args.compare_production_mla_woa or
                    args.compare_production_mla_woa_wob or
                    args.compare_production_combined):
                # Production prepares this immutable matrix once while loading
                # weights.  Do not leave a concat node or duplicate source
                # matrices in the timed recipe/weight tuple.
                fused_qkv = concatenate_static_weights(wqa, wkv).contiguous()
                new.append((fused_qkv, before[1], common[1],
                            dense.tensor(prefix + "wq_b.weight", "hpu"), q_scale,
                            *common[2:],
                            dense.tensor(prefix + "wo_b.weight", "hpu"), o_scale))
                old.append(new[-1])
            else:
                new.append(tuple(before + [candidate_weight(prefix + "wq_b.weight", "hpu"), q_scale] + common +
                                 [candidate_weight(prefix + "wo_b.weight", "hpu"), o_scale]))
            if (args.production_parity or args.compare_production_rope_woa or
                    args.compare_production_qnorm or args.compare_production_kvnorm or
                    args.compare_production_bf16_pv or args.compare_production_mla_woa or
                    args.compare_production_mla_woa_wob or
                    args.compare_production_combined):
                pass
            elif args.compare_kv_order:
                old.append(new[-1])
            elif args.include_reference:
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
        fused_norm = os.environ.get("VLLM_HPU_DSV41_ATTN_FUSED_NORM") == "1"
        cross_arm_exact = None
        if args.production_parity:
            reference = candidate = production_program()
        elif args.compare_production_rope_woa:
            reference = production_program(False)
            candidate = production_program(True)
            diagnostic_reference = production_program(False, True)
            diagnostic_candidate = production_program(True, True)
            stages = [{"stage": "q", "different": 0, "maximum_absolute_difference": 0.0},
                      {"stage": "kv", "different": 0, "maximum_absolute_difference": 0.0},
                      {"stage": "wo_a", "different": 0, "maximum_absolute_difference": 0.0},
                      {"stage": "wo_b", "different": 0, "maximum_absolute_difference": 0.0}]
            for value, old_weight, new_weight in zip(inputs, old, new, strict=True):
                expected = tuple(x.cpu() for x in diagnostic_reference(value, *old_weight))
                actual = tuple(x.cpu() for x in diagnostic_candidate(value, *new_weight))
                for stage, wanted, got in zip(stages, expected, actual, strict=True):
                    if stage["stage"] == "kv" and stage["different"] == 0:
                        torch.save({"reference": wanted, "candidate": got}, output / "kv-first-layer.pt")
                    delta = (got.float() - wanted.float()).abs()
                    stage["different"] += int((got != wanted).sum())
                    stage["maximum_absolute_difference"] = max(stage["maximum_absolute_difference"],
                                                                  float(delta.max()))
            cross_arm_exact = {"layers": len(inputs), "stages": stages,
                               "bitwise_equal": all(not stage["different"] for stage in stages)}
            (output / "cross-arm-check.json").write_text(json.dumps(cross_arm_exact, indent=2) + "\n")
            if not cross_arm_exact["bitwise_equal"]:
                raise AssertionError(("production inverse-RoPE/wo_a fusion changed output", stages))
        elif args.compare_production_qnorm:
            reference = production_program(False)
            candidate = production_program(False, fused_qnorm=True)
            diagnostic_reference = production_program(False, True)
            diagnostic_candidate = production_program(False, True, True)
            stages = [{"stage": "q", "different": 0, "maximum_absolute_difference": 0.0},
                      {"stage": "kv", "different": 0, "maximum_absolute_difference": 0.0},
                      {"stage": "wo_a", "different": 0, "maximum_absolute_difference": 0.0},
                      {"stage": "wo_b", "different": 0, "maximum_absolute_difference": 0.0}]
            for value, old_weight, new_weight in zip(inputs, old, new, strict=True):
                expected = tuple(x.cpu() for x in diagnostic_reference(value, *old_weight))
                actual = tuple(x.cpu() for x in diagnostic_candidate(value, *new_weight))
                for stage, wanted, got in zip(stages, expected, actual, strict=True):
                    delta = (got.float() - wanted.float()).abs()
                    stage["different"] += int((got != wanted).sum())
                    stage["maximum_absolute_difference"] = max(stage["maximum_absolute_difference"],
                                                                  float(delta.max()))
            cross_arm_exact = {"layers": len(inputs), "stages": stages,
                               "bitwise_equal": all(not stage["different"] for stage in stages)}
            (output / "cross-arm-check.json").write_text(json.dumps(cross_arm_exact, indent=2) + "\n")
            if not cross_arm_exact["bitwise_equal"]:
                raise AssertionError(("production QNorm/quant fusion changed output", stages))
        elif args.compare_production_kvnorm:
            reference = production_program(False)
            candidate = production_program(False, fused_kvnorm=True)
            diagnostic_reference = production_program(False, True)
            diagnostic_candidate = production_program(False, True, False, True)
            stages = [{"stage": "q", "different": 0, "maximum_absolute_difference": 0.0},
                      {"stage": "kv", "different": 0, "maximum_absolute_difference": 0.0},
                      {"stage": "wo_a", "different": 0, "maximum_absolute_difference": 0.0},
                      {"stage": "wo_b", "different": 0, "maximum_absolute_difference": 0.0}]
            for layer_index, (value, old_weight, new_weight) in enumerate(
                    zip(inputs, old, new, strict=True)):
                expected = tuple(x.cpu() for x in diagnostic_reference(value, *old_weight))
                actual = tuple(x.cpu() for x in diagnostic_candidate(value, *new_weight))
                if layer_index == 0:
                    projected = F.linear(quantize_activation(value), new_weight[0])
                    kv_input = projected[..., 1280:].contiguous()
                    normalized = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2(
                        kv_input, new_weight[2], 1e-20)
                    torch.save({
                        "kv_input": kv_input.cpu(),
                        "normalized": normalized.cpu(),
                        "reference": expected[1],
                        "candidate": actual[1],
                    }, output / "kv-first-layer.pt")
                for stage, wanted, got in zip(stages, expected, actual, strict=True):
                    delta = (got.float() - wanted.float()).abs()
                    stage["different"] += int((got != wanted).sum())
                    stage["maximum_absolute_difference"] = max(stage["maximum_absolute_difference"],
                                                                  float(delta.max()))
            cross_arm_exact = {"layers": len(inputs), "stages": stages,
                               "bitwise_equal": all(not stage["different"] for stage in stages)}
            (output / "cross-arm-check.json").write_text(json.dumps(cross_arm_exact, indent=2) + "\n")
            if not cross_arm_exact["bitwise_equal"]:
                raise AssertionError(("production KVNorm/RoPE fusion changed output", stages))
        elif args.compare_production_mla_woa:
            reference = production_program()
            candidate = production_program(fused_mla_woa=True)
            diagnostic_reference = production_program(diagnostics=True)
            diagnostic_candidate = production_program(diagnostics=True, fused_mla_woa=True)
            stages = [{"stage": stage, "different": 0, "maximum_absolute_difference": 0.0}
                      for stage in ("q", "kv", "wo_a", "wo_b")]
            for value, old_weight, new_weight in zip(inputs, old, new, strict=True):
                expected = tuple(x.cpu() for x in diagnostic_reference(value, *old_weight))
                actual = tuple(x.cpu() for x in diagnostic_candidate(value, *new_weight))
                for stage, wanted, got in zip(stages, expected, actual, strict=True):
                    delta = (got.float() - wanted.float()).abs()
                    stage["different"] += int((got != wanted).sum())
                    stage["maximum_absolute_difference"] = max(
                        stage["maximum_absolute_difference"], float(delta.max()))
            cross_arm_exact = {"layers": len(inputs), "stages": stages,
                               "bitwise_equal": all(not stage["different"] for stage in stages)}
            (output / "cross-arm-check.json").write_text(json.dumps(cross_arm_exact, indent=2) + "\n")
            if not cross_arm_exact["bitwise_equal"]:
                raise AssertionError(("production MLA/wo_a fusion changed output", stages))
        elif args.compare_production_mla_woa_wob:
            # Compare against the already-qualified QNorm/KVNorm/MLA-wo_a
            # compounds so this measurement isolates only the internal BF16
            # wo_a -> dense quant -> wo_b handoff.
            reference = production_program(fused_qnorm=True,
                                           fused_kvnorm=True,
                                           fused_mla_woa=True)
            candidate = production_program(fused_qnorm=True,
                                           fused_kvnorm=True,
                                           fused_mla_woa_wob=True)
            different = 0
            maximum = 0.0
            for value, old_weight, new_weight in zip(inputs, old, new,
                                                      strict=True):
                expected = reference(value, *old_weight).cpu()
                actual = candidate(value, *new_weight).cpu()
                delta = (actual.float() - expected.float()).abs()
                different += int((actual != expected).sum())
                maximum = max(maximum, float(delta.max()))
            cross_arm_exact = {
                "layers": len(inputs),
                "stages": [{
                    "stage": "wo_b",
                    "different": different,
                    "maximum_absolute_difference": maximum,
                }],
                "bitwise_equal": different == 0,
            }
            (output / "cross-arm-check.json").write_text(
                json.dumps(cross_arm_exact, indent=2) + "\n")
            if different:
                raise AssertionError(("internal wo_a/wo_b handoff changed output",
                                      cross_arm_exact))
        elif args.compare_production_combined:
            # KVNorm/RoPE is already part of the V2 parent. Measure the two
            # newly connected exact producers together at their real first
            # downstream consumer rather than adding separate micro results.
            reference = production_program(fused_kvnorm=True)
            candidate = production_program(fused_qnorm=True,
                                           fused_kvnorm=True,
                                           fused_mla_woa=True)
            diagnostic_reference = production_program(diagnostics=True,
                                                      fused_kvnorm=True)
            diagnostic_candidate = production_program(diagnostics=True,
                                                      fused_qnorm=True,
                                                      fused_kvnorm=True,
                                                      fused_mla_woa=True)
            stages = [{"stage": stage, "different": 0,
                       "maximum_absolute_difference": 0.0}
                      for stage in ("q", "kv", "wo_a", "wo_b")]
            for value, old_weight, new_weight in zip(inputs, old, new, strict=True):
                expected = tuple(x.cpu() for x in diagnostic_reference(value, *old_weight))
                actual = tuple(x.cpu() for x in diagnostic_candidate(value, *new_weight))
                for stage, wanted, got in zip(stages, expected, actual, strict=True):
                    delta = (got.float() - wanted.float()).abs()
                    stage["different"] += int((got != wanted).sum())
                    stage["maximum_absolute_difference"] = max(
                        stage["maximum_absolute_difference"], float(delta.max()))
            cross_arm_exact = {"layers": len(inputs), "stages": stages,
                               "bitwise_equal": all(not stage["different"] for stage in stages)}
            (output / "cross-arm-check.json").write_text(
                json.dumps(cross_arm_exact, indent=2) + "\n")
            if not cross_arm_exact["bitwise_equal"]:
                raise AssertionError(("combined exact Attention fusions changed output", stages))
        elif args.compare_production_bf16_pv:
            reference = production_program()
            candidate = production_program(bf16_pv=True)
            diagnostic_reference = production_program(diagnostics=True)
            diagnostic_candidate = production_program(diagnostics=True, bf16_pv=True)
            stage_names = ("q", "kv", "wo_a", "wo_b")
            stages = [{"stage": stage, "different": 0,
                       "maximum_absolute_difference": 0.0,
                       "sum_squared_error": 0.0, "elements": 0}
                      for stage in stage_names]
            for value, old_weight, new_weight in zip(inputs, old, new, strict=True):
                expected = tuple(x.cpu() for x in diagnostic_reference(value, *old_weight))
                actual = tuple(x.cpu() for x in diagnostic_candidate(value, *new_weight))
                for stage, wanted, got in zip(stages, expected, actual, strict=True):
                    if not bool(got.isfinite().all()):
                        raise AssertionError(("BF16 PV produced non-finite output", stage["stage"]))
                    delta = got.float() - wanted.float()
                    stage["different"] += int((got != wanted).sum())
                    stage["maximum_absolute_difference"] = max(
                        stage["maximum_absolute_difference"], float(delta.abs().max()))
                    stage["sum_squared_error"] += float(delta.square().sum())
                    stage["elements"] += delta.numel()
            for stage in stages:
                stage["rmse"] = (stage.pop("sum_squared_error") /
                                  stage["elements"])**0.5
            cross_arm_exact = {
                "layers": len(inputs),
                "stages": stages,
                "bitwise_equal": all(not stage["different"] for stage in stages),
                "expected_numeric_change": True,
                "quality_not_qualified": True,
            }
            (output / "cross-arm-check.json").write_text(
                json.dumps(cross_arm_exact, indent=2) + "\n")
        elif args.compare_kv_order:
            reference = program(True, kv_first=False, fused_norm=fused_norm)
            candidate = program(True, kv_first=True, fused_norm=fused_norm)
        else:
            reference = program(False)
            candidate = program(candidate_fp8,
                                kv_first=os.environ.get("VLLM_HPU_DSV41_ATTN_KV_FIRST") == "1",
                                fused_norm=fused_norm)
        result = benchmark(f"attention-dense-p{args.pp_rank}-pos{args.position}", reference, candidate,
                           inputs, old, new, output, 3, recorder,
                           args.production_parity or (False if (args.compare_kv_order or
                                                               args.compare_production_rope_woa or
                                                               args.compare_production_qnorm or
                                                               args.compare_production_kvnorm or
                                                               args.compare_production_bf16_pv or
                                                               args.compare_production_mla_woa or
                                                               args.compare_production_mla_woa_wob or
                                                               args.compare_production_combined)
                                                       else not args.include_reference),
                           ordinary_validator=(None if (args.production_parity or
                                                        args.compare_production_rope_woa or
                                                        args.compare_production_qnorm or
                                                        args.compare_production_kvnorm or
                                                        args.compare_production_bf16_pv or
                                                        args.compare_production_mla_woa or
                                                        args.compare_production_mla_woa_wob or
                                                        args.compare_production_combined) else
                                               partial(validate_terminal_mme,
                                                       candidate_fp8=candidate_fp8,
                                                       force_fp8=args.compare_kv_order,
                                                       compare_kv_order=args.compare_kv_order)),
                           profile_only=args.profile_only)
        result["candidate_precision"] = "fp8" if candidate_fp8 else "bf16"
        result["candidate_fused_norm"] = os.environ.get("VLLM_HPU_DSV41_ATTN_FUSED_NORM") == "1"
        result["compare_kv_order"] = args.compare_kv_order
        result["production_parity"] = args.production_parity
        result["compare_production_rope_woa"] = args.compare_production_rope_woa
        result["compare_production_qnorm"] = args.compare_production_qnorm
        result["compare_production_kvnorm"] = args.compare_production_kvnorm
        result["compare_production_bf16_pv"] = args.compare_production_bf16_pv
        result["compare_production_mla_woa"] = args.compare_production_mla_woa
        result["compare_production_mla_woa_wob"] = args.compare_production_mla_woa_wob
        result["compare_production_combined"] = args.compare_production_combined
        if cross_arm_exact is not None:
            result["cross_arm_exact"] = cross_arm_exact
        result["state_scope"] = ("real Q/KV input projections/norms, SWA writer, C1 indices, shared-KV MME MLA, "
                                 "inverse RoPE and wo_a/wo_b to persistent BF16 TP operand; "
                                 "seeded decoded history/static compressed publication; no compressor, TP or PP")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
