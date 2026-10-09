# SPDX-License-Identifier: Apache-2.0
"""Complete routed MoE micro with real weights and persistent native recipes.

Only explicitly selected real experts are loaded into production-size tensors.
Every measured ID is checked against that set. This bounded diagnostic neither
loads a serving model nor qualifies its end-to-end timing.
"""
import argparse
import json
import os
from pathlib import Path
import time

from benchmark_deepseek_v41_projection_chains import summarize
from deepseek_v41_micro_replay import RecipeRecorder
import numpy as np
import torch

from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert, FINGERPRINT, COMPACT_FINGERPRINT
from vllm_gaudi.ops.deepseek_v41_fp8 import read_expert
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--include-missing-reference", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--post-graphs-only", action="store_true",
                        help="Cold-compile the real producer/consumer and retain physical allocation metadata")
    parser.add_argument("--fused-quant", action="store_true")
    parser.add_argument("--fused-reduce", action="store_true")
    parser.add_argument("--direct-finalize", action="store_true")
    parser.add_argument("--prefetch-w2", action="store_true",
                        help="Compare early W2 decode scheduling against accepted direct-finalize")
    parser.add_argument("--router-logits", action="store_true",
                        help="Compare the fused logits-to-top6 kernel through the real expert consumer")
    parser.add_argument("--ffn-norm-quant", action="store_true",
                        help="Compare fused FFN RMSNorm/quant through the accepted W13/W2 consumer")
    parser.add_argument("--slot-fused", action="store_true",
                        help="Compare one six-slot TPC decode launch against the accepted per-slot access map")
    parser.add_argument("--fused-shared-tail", action="store_true",
                        help="Keep the exact prequant routed/shared finalize inside the compound graph")
    parser.add_argument("--resident-fp8", action="store_true",
                        help="Compare a decoded-FP8 cache hit through the complete W13/W2 consumer chain")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 14, 19])
    parser.add_argument("--tokens",
                        type=int,
                        choices=range(1, 7),
                        default=1,
                        help="Decode batch used by the full consumer chain")
    parser.add_argument("--compact-scales",
                        action="store_true",
                        help="Use the production compact scale planes and channel row")
    parser.add_argument("--selected-experts", type=int, choices=(6, 12, 21, 36), default=12)
    parser.add_argument("--baseline-only",
                        action="store_true",
                        help="Measure only the current consumer, retaining ordinary/native correctness checks")
    parser.add_argument("--token-wide-sat", action="store_true", help="Qualified C2-C6 route-N packing and SAT codec")
    parser.add_argument("--token-wide-load16", action="store_true")
    parser.add_argument("--token-wide-k-first", action="store_true", help="Same SAT instructions, K128-first work grid")
    parser.add_argument("--sram-capacity-bytes", type=int, default=0,
                        help="Diagnostic compiler tiling cap; same SAT ELF, isolated graph alias")
    parser.add_argument("--silu-fixed640", action="store_true",
                        help="Same expert consumer; register-resident fixed-width activation/quant")
    parser.add_argument("--packed-byte-input", action="store_true",
                        help="Identical FP4 storage through a U8 descriptor, unchanged SAT and MME")
    parser.add_argument("--input-fixtures", type=Path,
                        help="CPU torch.save list of 3-5 real layer inputs: hidden/ids/router/shared/layer")
    parser.add_argument("--active-k", action="store_true", help="Omit qualified zero K padding from W2")
    parser.add_argument("--token-wide-prefetch", action="store_true",
                        help="Compare early W2 decode within the same qualified C2-C6 SAT body")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--event-samples", type=int, default=64)
    parser.add_argument("--warm-steps", type=int, default=32)
    args = parser.parse_args()
    if args.silu_fixed640 and (not args.token_wide_sat or any((args.token_wide_k_first,
                                   args.token_wide_load16, args.token_wide_prefetch, args.active_k,
                                   args.sram_capacity_bytes, args.packed_byte_input))):
        parser.error("Fixed-width Silu changes only the ordinary SAT activation/quant")
    if args.packed_byte_input and (not args.token_wide_sat or any((args.token_wide_k_first,
                                     args.token_wide_load16, args.token_wide_prefetch,
                                     args.active_k, args.sram_capacity_bytes))):
        parser.error("Byte descriptor comparison changes only ordinary SAT weight dtype")
    if args.sram_capacity_bytes and (not args.token_wide_sat or args.token_wide_k_first or
                                     args.token_wide_load16 or args.token_wide_prefetch or args.active_k):
        parser.error("Small SRAM requires the unchanged ordinary SAT consumer")
    if args.sram_capacity_bytes and os.environ.get("PT_HPU_ENABLE_JIT_GRAPH_NAME_HASH") != "1":
        parser.error("Compiler-capacity arms require graph-name cache isolation before initialization")
    if args.token_wide_k_first and (not args.token_wide_sat or args.token_wide_load16 or args.token_wide_prefetch):
        parser.error("K-first requires ordinary SAT consumer; no load16/early-W2 changes")
    if args.token_wide_load16 and (not args.token_wide_sat or args.token_wide_prefetch):
        parser.error("Load prefetch16 requires the same SAT body, excluding early W2")
    if args.token_wide_prefetch and not args.token_wide_sat:
        parser.error("Token-wide prefetch requires the qualified SAT comparison")
    if args.token_wide_sat and (not args.ffn_norm_quant or args.tokens < 2 or
                               (args.baseline_only and not args.post_graphs_only)):
        parser.error("Token-wide SAT requires the C2-C6 fused-norm comparison")
    if min(args.rounds, args.event_samples, args.warm_steps) < 1:
        parser.error("Measurement counts must be positive")
    if args.baseline_only and args.include_missing_reference:
        parser.error("A baseline-only run cannot include a missing reference")
    if args.fused_reduce and not args.fused_quant:
        parser.error("--fused-reduce requires --fused-quant")
    if args.direct_finalize and (args.fused_reduce or not args.fused_quant):
        parser.error("--direct-finalize requires --fused-quant and excludes --fused-reduce")
    if args.prefetch_w2 and not args.direct_finalize:
        parser.error("--prefetch-w2 requires --direct-finalize")
    if args.router_logits and not args.direct_finalize:
        parser.error("--router-logits requires the accepted --direct-finalize consumer")
    if args.ffn_norm_quant and not args.prefetch_w2:
        parser.error("--ffn-norm-quant requires the accepted direct-finalize/prefetch-W2 consumer")
    if args.slot_fused and not args.ffn_norm_quant:
        parser.error("--slot-fused requires --ffn-norm-quant so both arms consume identical prequantized input")
    if args.slot_fused and args.tokens != 1:
        parser.error("--slot-fused currently qualifies the top-6 single-token decode contract")
    if args.fused_shared_tail and (not args.ffn_norm_quant or not args.prefetch_w2
                                   or args.slot_fused or args.tokens != 1):
        parser.error("--fused-shared-tail requires C1 FFN-norm/prequant and prefetch-W2, without slot fusion")
    if args.resident_fp8 and (not args.ffn_norm_quant or args.slot_fused or args.tokens != 1):
        parser.error("--resident-fp8 requires the C1 FFN-norm/prequant reference and excludes slot fusion")
    output = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    if any((args.token_wide_load16, args.active_k, args.token_wide_k_first,
            args.sram_capacity_bytes, args.packed_byte_input, args.silu_fixed640)):
        torch.ops.load_library(os.environ["DSV41_UNIQUE_OPERATOR_LIBRARY"])
    bind_worker_cpu(0)
    torch.manual_seed(202641)
    torch._dynamo.config.recompile_limit = 32
    shard = PreparedV41Shard(args.prepared, 0, 0)
    epsilon = json.loads((args.prepared / "config.json").read_text())["text_config"]["rms_norm_eps"]
    logical_intermediate = (json.loads((args.prepared / "config.json").read_text())["text_config"]
                            ["moe_intermediate_size"] // shard.manifest["tensor_parallel_size"])
    if args.active_k and (not args.token_wide_sat or args.token_wide_prefetch or args.token_wide_load16):
        parser.error("Active K compares the qualified ordinary SAT chain without early W2/load16")
    if args.post_graphs_only:
        from vllm_gaudi.distributed import tp2_fused_ar_norm  # noqa: F401
        recorder = None
    else:
        recorder = RecipeRecorder(output)
    fixtures = torch.load(args.input_fixtures, map_location="cpu", weights_only=True) if args.input_fixtures else None
    if fixtures is not None and not 3 <= len(fixtures) <= 5:
        parser.error("Correctness requires 3-5 saved real input sets")
    lookup = mxfp4_bf16_lut("hpu")
    selected = list(range(args.selected_experts))
    inputs, weights = [], {"candidate": []}
    if not args.baseline_only and (args.include_missing_reference or args.fused_reduce or args.direct_finalize):
        weights["reference"] = []
    record = {
        "layout_fingerprint": COMPACT_FINGERPRINT if args.compact_scales else FINGERPRINT,
        "fused_quant": args.fused_quant,
        "fused_reduce": args.fused_reduce,
        "direct_finalize": args.direct_finalize,
        "prefetch_w2": args.prefetch_w2,
        "router_logits": args.router_logits,
        "ffn_norm_quant": args.ffn_norm_quant,
        "slot_fused": args.slot_fused,
        "fused_shared_tail": args.fused_shared_tail,
        "resident_fp8": args.resident_fp8,
        "tokens": args.tokens,
        "compact_scales": args.compact_scales,
        "baseline_only": args.baseline_only,
        "token_wide_sat": args.token_wide_sat,
        "token_wide_prefetch": args.token_wide_prefetch,
        "token_wide_load16": args.token_wide_load16,
        "token_wide_k_first": args.token_wide_k_first,
        "sram_capacity_bytes": args.sram_capacity_bytes,
        "packed_byte_input": args.packed_byte_input,
        "silu_fixed640": args.silu_fixed640,
        "input_fixtures": str(args.input_fixtures) if args.input_fixtures else None,
        "correctness_inputs": "saved-real" if fixtures is not None else "synthetic-screening-only",
        "active_k": args.active_k,
        "layers": args.layers,
        "selected_real_experts": selected,
        "uninitialized_experts_never_addressed": True,
        "scope": "full W13/decode/activation/quant/W2/scale/ordered sum/BF16 consumer; no TP/PP",
        "timing": "native compute command replay; one synchronized four-layer sweep per event pair",
        "source_plan_fingerprint": shard.manifest["plan_fingerprint"],
        "rounds": [],
        "checks": []
    }
    with torch.inference_mode():
        for layer in args.layers:
            old, new = {}, {}
            for name in ("w13", "w2"):
                prefix = f"layers.{layer}.ffn.experts.{name}"
                qs, ss = (shard.catalog[prefix + suffix] for suffix in ("_q16", "_s16"))
                e, blocks, stream = qs.shape
                nq = torch.empty(e, blocks // 2, stream * 2, dtype=torch.int16, device="hpu")
                scale_words = stream // 8 + 128 if args.compact_scales else ss.shape[2] * 2
                ns = torch.empty(e, blocks // 2, scale_words, dtype=torch.int16, device="hpu")
                nc = torch.empty(e, blocks // 2, 256, dtype=torch.bfloat16, device="hpu")
                if args.include_missing_reference:
                    oq = torch.empty(qs.shape, dtype=torch.int16, device="hpu")
                    oscale = torch.empty(ss.shape, dtype=torch.bfloat16, device="hpu")
                for expert in selected:
                    shard.check_identity()
                    q, s = read_expert(qs, expert), read_expert(ss, expert)
                    pq, ps, pc, qualification = prepare_expert(q, s, compact_scales=args.compact_scales)
                    if args.token_wide_sat:
                        from vllm_gaudi.ops.deepseek_v41_expert_n256 import saturated_decode_eligible
                        active_k = 5120 if name == "w13" else (json.loads(
                            (args.prepared / "config.json").read_text())["text_config"]["moe_intermediate_size"] //
                                                               shard.manifest["tensor_parallel_size"])
                        if np.any(pq[:, active_k * 64:]):
                            raise ValueError("Cannot exclude a nonzero padded K suffix from SAT qualification")
                        if not saturated_decode_eligible(ps, active_k=active_k):
                            raise ValueError(f"SAT offset qualification failed: layer{layer}/{name}/expert{expert}")
                    nq[expert].copy_(torch.from_numpy(pq))
                    ns[expert].copy_(torch.from_numpy(ps))
                    nc[expert].copy_(torch.from_numpy(pc.view(np.int16)).view(torch.bfloat16))
                    if args.include_missing_reference:
                        oq[expert].copy_(torch.from_numpy(q))
                        oscale[expert].copy_(torch.from_numpy(s.view(np.int16)).view(torch.bfloat16))
                new[name] = nq, ns, nc
                if args.include_missing_reference:
                    old[name] = oq, oscale
                print("prepared",
                      layer,
                      name,
                      "temporary_bound",
                      qualification["temporary_upper_bound_bytes"],
                      flush=True)
            inputs.append(torch.randn(args.tokens, 5120).bfloat16().to("hpu"))
            ids = (torch.arange(args.tokens * 6, dtype=torch.int32, device="hpu")
                   .reshape(args.tokens, 6) % len(selected))
            route = torch.softmax(torch.randn(args.tokens, 6), -1).mul_(1.5).to("hpu")
            shared = torch.randn(args.tokens, 5120).bfloat16().to("hpu")
            base_weights = (ids, route, new["w13"][0], new["w2"][0],
                            new["w13"][1], new["w2"][1], lookup,
                            new["w13"][2], new["w2"][2], shared)
            if args.ffn_norm_quant:
                base_weights += (shard.tensor(f"layers.{layer}.ffn_norm.weight", "hpu"), )
            if args.resident_fp8:
                resident_w13 = torch.ops.custom_op.custom_deepseek_v41_expert_n256_fp8_gaudi2(
                    ids, new["w13"][0], new["w13"][1], lookup, True)
                resident_w2 = torch.ops.custom_op.custom_deepseek_v41_expert_n256_fp8_gaudi2(
                    ids, new["w2"][0], new["w2"][1], lookup, True)
                torch.hpu.synchronize()
                candidate_weights = (ids, route, resident_w13, resident_w2,
                                     new["w13"][2], new["w2"][2], shared,
                                     base_weights[-1])
                record.setdefault("resident_fp8_bytes", 0)
                record["resident_fp8_bytes"] += resident_w13.numel() + resident_w2.numel()
            else:
                candidate_weights = base_weights
            if args.packed_byte_input:
                candidate_weights = (*base_weights[:2], base_weights[2].view(torch.uint8),
                                     base_weights[3].view(torch.uint8), *base_weights[4:])
                assert candidate_weights[2].data_ptr() == base_weights[2].data_ptr()
                assert candidate_weights[3].data_ptr() == base_weights[3].data_ptr()
            weights["candidate"].append(candidate_weights)
            if args.router_logits:
                prefix = f"layers.{layer}.ffn.gate."
                text_bias = shard.tensor(prefix + "bias", "hpu").clone()
                image_bias = shard.tensor(prefix + "bias_vl", "hpu").clone()
                # The bounded diagnostic materializes only these real experts.
                # Keep gate logits and biases real while making every measured
                # route provably address an initialized expert tensor.
                text_bias[len(selected):].fill_(-10000.0)
                image_bias[len(selected):].fill_(-10000.0)
                router = (shard.tensor(prefix + "weight", "hpu"),
                          text_bias,
                          image_bias,
                          torch.full((args.tokens,), layer % 3 == 0,
                                     dtype=torch.bool, device="hpu"))
                weights["candidate"][-1] += router
            if "reference" in weights and (args.fused_reduce or args.direct_finalize):
                # Isolate only the finalize tail. Both arms consume identical
                # prepared N256 weights and changing IDs/activations.
                weights["reference"].append(base_weights)
            elif args.include_missing_reference:
                weights["reference"].append(
                    (ids, route, old["w13"][0], old["w2"][0], old["w13"][1], old["w2"][1], lookup, shared))

        def candidate(x, ids, route, q13, q2, s13, s2, lut, c13, c2, shared):
            op = (torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_direct_finalize_prefetch_w2_fp8_gaudi2
                  if args.prefetch_w2 else
                  torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_direct_finalize_fp8_gaudi2
                  if args.direct_finalize else
                  torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_reduce_fp8_gaudi2
                  if args.fused_reduce else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2
                  if args.fused_quant else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fp8_gaudi2)
            value = op(x, ids, route, q13, q2, s13, s2, lut, c13, c2, True)
            return (value.float() + shared.float()).bfloat16()

        def norm_quant_candidate(x, ids, route, q13, q2, s13, s2, lut,
                                 c13, c2, shared, norm_weight):
            normalized, quantized, activation_scale = (
                torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
                    x, norm_weight, epsilon))
            op = (
                torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_silu640_fp8_gaudi2
                if args.silu_fixed640 else
                torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_bytes_fp8_gaudi2
                if args.packed_byte_input else
                torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_small_sram_fp8_gaudi2
                if args.sram_capacity_bytes else
                torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_k_first_fp8_gaudi2
                if args.token_wide_k_first else
                torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_active_k_fp8_gaudi2
                if args.active_k else
                torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_sat16_fp8_gaudi2
                if args.token_wide_load16 else
                torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_sat_prefetch_w2_fp8_gaudi2
                if args.token_wide_prefetch else
                torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_sat_fp8_gaudi2
                if args.token_wide_sat else torch.ops.custom_op.
                custom_deepseek_v41_expert_n256_moe_prequant_direct_finalize_shared_prefetch_w2_fp8_gaudi2
                if args.fused_shared_tail else torch.ops.custom_op.
                custom_deepseek_v41_expert_n256_moe_prequant_direct_finalize_slots_fp8_gaudi2
                if args.slot_fused else torch.ops.custom_op.
                custom_deepseek_v41_expert_n256_moe_prequant_direct_finalize_prefetch_w2_fp8_gaudi2
                if args.tokens == 1 else
                torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_prequant_fused_fp8_gaudi2)
            operands = (normalized, ids, route, q13, q2, s13, s2, lut,
                        c13, c2, quantized, activation_scale)
            if args.fused_shared_tail:
                return op(*operands, shared, True)
            value = (op(*operands, logical_intermediate, True) if args.active_k else op(*operands, True))
            return (value.float() + shared.float()).bfloat16()

        def norm_quant_reference(x, ids, route, q13, q2, s13, s2, lut,
                                 c13, c2, shared, norm_weight):
            if args.token_wide_sat:
                normalized, quantized, activation_scale = (
                    torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(x, norm_weight, epsilon))
                expert_op = (
                    torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_token_wide_sat_fp8_gaudi2
                    if any((args.token_wide_prefetch, args.token_wide_load16, args.active_k,
                            args.token_wide_k_first, args.sram_capacity_bytes, args.packed_byte_input,
                            args.silu_fixed640)) else
                    torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_prequant_fused_fp8_gaudi2
                )
                routed = expert_op(
                    normalized, ids, route, q13, q2, s13, s2, lut, c13, c2, quantized, activation_scale, True)
                return (routed.float() + shared.float()).bfloat16()
            if args.fused_shared_tail:
                normalized, quantized, activation_scale = (
                    torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
                        x, norm_weight, epsilon))
                routed = (
                    torch.ops.custom_op.
                    custom_deepseek_v41_expert_n256_moe_prequant_direct_finalize_prefetch_w2_fp8_gaudi2(
                        normalized, ids, route, q13, q2, s13, s2, lut,
                        c13, c2, quantized, activation_scale, True))
                return (routed.float() + shared.float()).bfloat16()
            if args.slot_fused:
                normalized, quantized, activation_scale = (
                    torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
                        x, norm_weight, epsilon))
                routed = (
                    torch.ops.custom_op.
                    custom_deepseek_v41_expert_n256_moe_prequant_direct_finalize_prefetch_w2_fp8_gaudi2(
                        normalized, ids, route, q13, q2, s13, s2, lut,
                        c13, c2, quantized, activation_scale, True))
                return (routed.float() + shared.float()).bfloat16()
            value = x.float()
            normalized = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
                          * norm_weight.float()).to(torch.bfloat16).contiguous()
            op = (torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_direct_finalize_prefetch_w2_fp8_gaudi2
                  if args.tokens == 1 else
                  torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2)
            routed = op(normalized, ids, route, q13, q2, s13, s2, lut,
                        c13, c2, True)
            return (routed.float() + shared.float()).bfloat16()

        def resident_candidate(x, ids, route, w13, w2, c13, c2, shared,
                               norm_weight):
            normalized, quantized, activation_scale = (
                torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
                    x, norm_weight, epsilon))
            routed = (
                torch.ops.custom_op.
                custom_deepseek_v41_expert_n256_moe_prequant_resident_direct_finalize_fp8_gaudi2(
                    normalized, ids, route, w13, w2, c13, c2,
                    quantized, activation_scale, True))
            return (routed.float() + shared.float()).bfloat16()

        def reference(x, ids, route, q13, q2, s13, s2, lut, shared):
            value = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_moe_k128_bf16_gaudi2(
                x, ids, route, q13, q2, s13, s2, lut, True)
            return (value.float() + shared.float()).bfloat16()

        def fused_reference(x, ids, route, q13, q2, s13, s2, lut, c13, c2, shared):
            op = (torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_reduce_fp8_gaudi2
                  if args.direct_finalize else
                  torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2)
            value = op(
                x, ids, route, q13, q2, s13, s2, lut, c13, c2, True)
            return (value.float() + shared.float()).bfloat16()

        def router_candidate(x, ids, route, q13, q2, s13, s2, lut, c13, c2,
                             shared, gate, text, image, mask):
            del ids, route
            logits = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
                x.contiguous(), gate)
            selected, routing = torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2(
                logits, text, image, mask)
            value = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_direct_finalize_fp8_gaudi2(
                x, selected, routing, q13, q2, s13, s2, lut, c13, c2, True)
            return (value.float() + shared.float()).bfloat16()

        def router_reference(x, ids, route, q13, q2, s13, s2, lut, c13, c2,
                             shared, gate, text, image, mask):
            del ids, route
            logits = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
                x.contiguous(), gate)
            scores = torch.nn.functional.softplus(logits).sqrt()
            selected, routing = torch.ops.custom_op.custom_deepseek_v41_router_top6_gaudi2(
                scores, text, image, mask)
            value = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_direct_finalize_fp8_gaudi2(
                x, selected, routing, q13, q2, s13, s2, lut, c13, c2, True)
            return (value.float() + shared.float()).bfloat16()

        replays, compiled = {}, {}
        for arm in weights:
            fn = ((resident_candidate if arm == "candidate" else norm_quant_reference)
                  if args.resident_fp8 else
                  (norm_quant_candidate if arm == "candidate" else norm_quant_reference)
                  if args.ffn_norm_quant else
                  (router_candidate if arm == "candidate" else router_reference)
                  if args.router_logits else candidate if arm == "candidate" else
                  fused_reference if (args.fused_reduce or args.direct_finalize) else reference)
            compiled[arm] = torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
            for x, w in zip(inputs, weights[arm], strict=True):
                ordinary = fn(x, *w).cpu()
                if args.post_graphs_only or args.sram_capacity_bytes:
                    from deepseek_v41_resident_ab import compiler_settings
                    directory = output / "physical-graphs" / arm
                    directory.mkdir(parents=True, exist_ok=True)
                    settings = {"DUMP_POST_GRAPHS": str(directory)}
                    if args.sram_capacity_bytes and arm == "candidate":
                        settings["SRAM_SLICER_MAX_CAPACITY_BYTES"] = str(args.sram_capacity_bytes)
                    with compiler_settings(settings):
                        actual = compiled[arm](x, *w).cpu()
                else:
                    actual = compiled[arm](x, *w).cpu()
                assert torch.equal(ordinary.view(torch.int16), actual.view(torch.int16)), (arm, "ordinary/compiled")
            if args.post_graphs_only:
                continue
            replays[arm] = recorder.prepare(compiled[arm], inputs, weights[arm])

        if args.post_graphs_only:
            record.update(status="allocation_capture_only", performance_qualified=False)
            (output / "ALLOCATION_CAPTURE.json").write_text(json.dumps(record, indent=2) + "\n")
            return
        if "reference" in weights and (args.fused_reduce or args.direct_finalize):
            for layer, (x, candidate_weights, reference_weights) in enumerate(
                    zip(inputs, weights["candidate"], weights["reference"], strict=True)):
                actual = compiled["candidate"](x, *candidate_weights).cpu()
                expected = compiled["reference"](x, *reference_weights).cpu()
                if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                    raise AssertionError((layer, "Non-finite expert consumer output"))
                if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
                    delta = actual.float() - expected.float()
                    mismatched = actual.view(torch.int16) != expected.view(torch.int16)
                    record["finalize_mismatch"] = {
                        "layer_index": layer,
                        "mismatched_elements": int(mismatched.sum()),
                        "total_elements": actual.numel(),
                        "maximum_absolute_error": float(delta.abs().max()),
                        "mean_absolute_error": float(delta.abs().mean()),
                        "candidate_first_16": actual.flatten()[:16].float().tolist(),
                        "reference_first_16": expected.flatten()[:16].float().tolist(),
                    }
                    np.save(output / "finalize_candidate.npy", actual.float().numpy())
                    np.save(output / "finalize_reference.npy", expected.float().numpy())
                    shared_index = -2 if args.ffn_norm_quant else -1
                    np.save(output / "finalize_shared.npy",
                            candidate_weights[shared_index].cpu().float().numpy())
                    if args.router_logits:
                        (output / "result.json").write_text(json.dumps(record, indent=2))
                        raise AssertionError((layer, "complete MoE consumer mismatch", record["finalize_mismatch"]))
                    if args.resident_fp8:
                        (output / "result.json").write_text(json.dumps(record, indent=2))
                        raise AssertionError((layer, "resident FP8 complete consumer mismatch",
                                              record["finalize_mismatch"]))
                    if args.fused_shared_tail:
                        (output / "result.json").write_text(json.dumps(record, indent=2))
                        raise AssertionError((layer, "prequant shared-tail mismatch",
                                              record["finalize_mismatch"]))
                    if args.token_wide_sat:
                        (output / "result.json").write_text(json.dumps(record, indent=2))
                        raise AssertionError((layer, "SAT complete consumer mismatch", record["finalize_mismatch"]))
                    candidate_core = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_reduce_fp8_gaudi2(
                        x, *candidate_weights[:-1], True).cpu()
                    reference_op = (torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_reduce_fp8_gaudi2
                                    if args.direct_finalize else
                                    torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2)
                    reference_core = reference_op(
                        x, *reference_weights[:-1], True).cpu()
                    np.save(output / "finalize_candidate_core.npy", candidate_core.float().numpy())
                    np.save(output / "finalize_reference_core.npy", reference_core.float().numpy())
                    (output / "result.json").write_text(json.dumps(record, indent=2))
                    raise AssertionError((layer, "finalize mismatch", record["finalize_mismatch"]))
            record["checks"].append({
                "candidate_reference_bitwise_equal":
                True,
                "comparison": ("same SAT C2-C6 weight decode, W13/W2, activation, quantization and BF16 consumer; "
                               "early versus late W2 weight producer"
                               if args.token_wide_prefetch else
                               "resident decoded-FP8 cache hit versus compressed MXFP4 decode "
                               "through identical N256 W13/W2, SwiGLU, direct-finalize and BF16 consumer"
                               if args.resident_fp8 else
                               "same prequant W13/W2 routed result; exact BF16/FP32 shared-expert "
                               "finalize inside versus outside the compound graph"
                               if args.fused_shared_tail else
                               "production score chain versus fused sqrt/top6 through identical direct-finalize MoE"
                               if args.router_logits else
                               "same N256 W13/W2 chain; direct FP32 scale/reduce versus two-TPC fused finalize"
                               if args.direct_finalize else
                               "same N256 W13/W2 chain; fused finalize versus materialized six-row tail"),
            })
        bind_worker_helpers(0)
        record["peak_hpu_allocated_bytes"] = torch.hpu.max_memory_allocated()
        record["thread_affinity"] = {
            p.name: sorted(os.sched_getaffinity(int(p.name)))
            for p in Path(f"/proc/{os.getpid()}/task").iterdir()
        }
        if args.profile_only:
            if args.include_missing_reference:
                raise ValueError("Profile the changed candidate only")
            replay = replays["candidate"]
            for _ in range(32):
                replay()
            torch.hpu.synchronize()
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
                    record_shapes=True,
                    with_stack=False) as profiler:
                for generation in range(32):
                    with torch.profiler.record_function(f"expert_sweep/generation={generation}"):
                        replay()
                torch.hpu.synchronize()
            profiler.export_chrome_trace(str(output / "profile.json"))
            record["profile_sweeps"] = 32
            record["native_compute_info"] = {arm: value.native.info() for arm, value in replays.items()}
            (output / "result.json").write_text(json.dumps(record, indent=2))
            for value in replays.values():
                value.close()
            torch.distributed.destroy_process_group()
            return
        events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True))
                  for _ in range(args.event_samples)]
        for round_id in range(args.rounds):
            for layer, x, w in zip(args.layers, inputs, weights["candidate"], strict=True):
                if fixtures is not None:
                    fixture = fixtures[round_id % len(fixtures)][layer]
                    assert fixture["hidden"].shape == x.shape
                    assert set(fixture["ids"].flatten().tolist()) <= set(selected)
                    x.copy_(fixture["hidden"])
                    w[0].copy_(fixture["ids"])
                    w[1].copy_(fixture["router"])
                    w[9].copy_(fixture["shared"])
                    continue
                x.mul_(1.01 if round_id % 2 else 0.99)
                if not args.router_logits and not args.resident_fp8:
                    current_ids = ((torch.arange(args.tokens * 6).reshape(args.tokens, 6)
                                    + round_id * 3) % len(selected)).int()
                    assert set(current_ids.flatten().tolist()) <= set(selected)
                    w[0].copy_(current_ids)
            torch.hpu.synchronize()
            arms = ("reference", "candidate") if args.token_wide_sat else tuple(replays)
            if args.token_wide_sat:
                for x, cw, rw in zip(inputs, weights["candidate"], weights["reference"], strict=True):
                    assert torch.equal(compiled["candidate"](x, *cw).cpu().view(torch.int16),
                                       compiled["reference"](x, *rw).cpu().view(torch.int16))
            for arm in arms:
                replay = replays[arm]
                expected = [compiled[arm](x, *w).cpu() for x, w in zip(inputs, weights[arm], strict=True)]
                replay()
                torch.hpu.synchronize()
                assert all(
                    torch.equal(a.view(torch.int16),
                                b.cpu().view(torch.int16))
                    for a, b in zip(expected, replay.outputs, strict=True)), (arm, "live replay mismatch")
                record["checks"].append({"round": round_id, "arm": arm, "input_and_ids_consumed": True})
                for _ in range(args.warm_steps):
                    replay()
                torch.hpu.synchronize()
                device, host = [], []
                for start, end in events:
                    begin = time.perf_counter_ns()
                    start.record()
                    replay()
                    end.record()
                    end.synchronize()
                    host.append((time.perf_counter_ns() - begin) / 1e6)
                    device.append(start.elapsed_time(end))
                item = {
                    "round": round_id,
                    "arm": arm,
                    "device_sweep": summarize(device),
                    "synchronized_host_sweep": summarize(host),
                    "device_samples_ms": device,
                    "median_device_per_layer_ms": float(np.median(device)) / len(inputs),
                    "mean_device_per_layer_ms": float(np.mean(device)) / len(inputs)
                }
                record["rounds"].append(item)
                (output / "result.json").write_text(json.dumps(record, indent=2))
                print(arm, round_id, item["mean_device_per_layer_ms"], "ms/layer", flush=True)
        if len(record["rounds"]) == 6 and args.rounds == 3:
            savings = [(a["median_device_per_layer_ms"] - b["median_device_per_layer_ms"]) * 40
                       for a, b in zip(record["rounds"][::2], record["rounds"][1::2], strict=True)]
            record["micro_gate"] = {
                "full_target_forecast_ms": float(np.median(savings)),
                "three_pair_savings_ms": savings,
                "three_directions_positive": all(value > 0 for value in savings),
                "real_input_correctness_passed": fixtures is not None,
                "qualified": fixtures is not None and all(value > 0 for value in savings)
                             and float(np.median(savings)) >= 0.3,
                "end_to_end_gain_credit": False,
            }
        record["native_compute_info"] = {arm: replay.native.info() for arm, replay in replays.items()}
        (output / "result.json").write_text(json.dumps(record, indent=2))
        for replay in replays.values():
            replay.close()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
