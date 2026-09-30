# SPDX-License-Identifier: Apache-2.0
"""Real TP4 FFN -> collective -> mHC consumer admission check.

Uses the production weight loader and PreparedMoE, with all 384 experts in
one layer. This is a component check, not full-model serving qualification.
"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--max-prefill-tokens", type=int, choices=(128, 8192), default=128)
    parser.add_argument("--native-prefill-plan", action="store_true",
                        help="Qualify the compute-only prepared-plan candidate before default promotion")
    parser.add_argument("--hybrid-rows", action="store_true")
    parser.add_argument("--bucket-transition", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    if args.native_prefill_plan:
        os.environ["VLLM_HPU_DSV41_PREFILL_NATIVE_PLAN"] = "1"
    if args.hybrid_rows:
        if not args.native_prefill_plan:
            raise ValueError("Hybrid rows require the prepared-plan candidate")
        os.environ["VLLM_HPU_DSV41_PREFILL_HYBRID_ROWS"] = "1"
        os.environ["VLLM_HPU_DSV41_PREFILL_SKIP_EMPTY"] = "1"
    import torch
    import torch.nn.functional as F
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel,
                                  init_distributed_environment, initialize_model_parallel)
    from vllm_gaudi.models.deepseek_v41_program import PreparedMoE, _prefill_hc_post, _weight_tree, load_weight_tree
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.deepseek_v41_math import hc_post, hc_pre, rms_norm
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from tools.check_deepseek_v41_tp4_components import validate_bf16_reduction
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method="env://", local_rank=rank,
                                 backend="hccl")
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=1))
    output = Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"ffn-rank{rank}.json"
    report = {"rank": rank, "tp": 4, "pp": 1, "status": "running", "cases": [],
              "boundary": "mHC pre -> RMSNorm -> router -> routed/shared experts -> TP reduce -> mHC post"}
    try:
        with set_current_vllm_config(config):
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            reduce, gather = stage_collectives(rank, False)
            shard = PreparedV41Shard(args.prepared, 0, rank)
            specs = {name: spec for name, spec in shard.specs.items()
                     if name.startswith(("layers.0.ffn.", "layers.0.ffn_norm.", "layers.0.hc_ffn_"))}
            tree = _weight_tree(specs)
            load_weight_tree(shard, tree, "hpu", specs)
            w = tree.layers.get_submodule("0")
            lookup = mxfp4_bf16_lut(torch.device("hpu"))
            normal = shard.manifest["normal_scales"]["layers.0.ffn.experts"][rank]
            moe = PreparedMoE(w.ffn, 6, normal, lookup, reduce, tensor_parallel_size=4)
            moe.prepare_shared_gate_up_weight()
            model_config = json.loads((args.prepared / "config.json").read_text())["text_config"]
            eps, hc_eps, iterations = (model_config[key] for key in
                                       ("rms_norm_eps", "hc_eps", "hc_sinkhorn_iters"))
            with torch.inference_mode():
                counts = ((1, 6, args.max_prefill_tokens, 128, args.max_prefill_tokens, 1)
                          if args.bucket_transition else (1, 6, args.max_prefill_tokens, 1))
                for step, count in enumerate(counts):
                    torch.manual_seed(7410 + step)
                    residual = torch.randn(count, 4, 5120).bfloat16().to("hpu")
                    previous = torch.full((count, 4), .25, device="hpu")
                    image_mask = (torch.arange(count, device="hpu") % 5 == 1)

                    def inputs(native=True):
                        value, pre, post, comb = hc_pre(residual, previous, w.hc_ffn_fn, w.hc_ffn_scale,
                                                       w.hc_ffn_base, eps, hc_eps, iterations,
                                                       packed_fn=w.hc_ffn_fn, prefill=native and count > 6)
                        return rms_norm(value, w.ffn_norm.weight, eps), pre, post, comb

                    def execute():
                        if count <= 2:
                            collapsed, pre, post, comb = hc_pre(
                                residual, previous, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base,
                                eps, hc_eps, iterations, packed_fn=w.hc_ffn_fn)
                            value, quantized, scale = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
                                collapsed.contiguous(), w.ffn_norm.weight, eps)
                            projected = moe(value, image_mask, decode=True, prequant=(quantized, scale))
                        else:
                            value, pre, post, comb = inputs()
                            projected = moe(value, image_mask, decode=count <= 6)
                        post_update = _prefill_hc_post if count > 6 else hc_post
                        return post_update(projected, residual, post, comb), pre

                    # Production wraps C1-C6 inside compiled decoder groups;
                    # prefill keeps the bounded expert recipe boundary.
                    if count <= 6:
                        execute = torch.compile(execute, backend="hpu_backend", fullgraph=True, dynamic=False)
                    value, _, post, comb = inputs(False)
                    gate = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
                        value.contiguous(), w.ffn.gate.weight)
                    scores = F.softplus(gate).sqrt()
                    bias = torch.where(image_mask[:, None], w.ffn.gate.bias_vl, w.ffn.gate.bias)
                    ids = torch.topk(scores + bias, 6, dim=-1, sorted=True).indices.int()
                    routing = scores.gather(1, ids.long())
                    routing = routing / (routing.sum(-1, keepdim=True) + 1e-20) * 1.5
                    experts = w.ffn.experts
                    operands = (value, ids, routing, experts.w13_q16, experts.w2_q16,
                                experts.w13_s16, experts.w2_s16, lookup)
                    if count <= 6:
                        routed = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fp8_gaudi2(
                            *operands, experts.w13_fp8_channel, experts.w2_fp8_channel, normal)
                    else:
                        from vllm_gaudi.ops.deepseek_v41_grouped_prefill import run_grouped_prefill
                        routed = run_grouped_prefill(*operands, normal, experts.w13_fp8_channel,
                                                     experts.w2_fp8_channel)
                    partial = (routed.float() + moe.shared_expert(value).float()).bfloat16()
                    copies = gather(partial, dim=0).cpu().reshape(4, count, 5120)
                    combined = reduce(partial.clone())
                    validate_bf16_reduction(combined.cpu(), copies)
                    expected = hc_post(combined, residual, post, comb).cpu()
                    actual, _ = execute()
                    host = actual.cpu()
                    torch.save({"actual": host, "expected": expected, "ids": ids.cpu()},
                               output.with_name(f"ffn-rank{rank}-step{step}.pt"))
                    torch.testing.assert_close(host.float(), expected.float(), atol=.03125, rtol=.03)
                    assert torch.isfinite(host).all()
                    # Values change between invocations, including the second C1
                    # after prefill. The timer includes the real downstream mHC
                    # consumer and the TP collective, with a drained endpoint.
                    del routed, partial, combined, copies, actual, value
                    torch.hpu.synchronize()
                    resident = torch.hpu.memory_allocated()
                    torch.hpu.reset_peak_memory_stats()
                    timings = []
                    for _ in range(5):
                        residual.add_(.001)
                        torch.hpu.synchronize()
                        started = time.perf_counter()
                        result = execute()
                        torch.hpu.synchronize()
                        timings.append((time.perf_counter() - started) * 1000)
                        del result
                    report["cases"].append({"tokens": count, "unique_experts": ids.cpu().unique().numel(),
                                             "host_sync_ms": timings, "resident_bytes": resident,
                                             "peak_allocated_bytes": torch.hpu.max_memory_allocated(),
                                             "max_abs_error": float((host.float() - expected.float()).abs().max()),
                                             "correctness": "passed"})
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    print(f"TP{rank} complete FFN C{count}: passed", flush=True)
            from vllm_gaudi.ops.deepseek_v41_prefill_plan import prefill_plan_stats
            report["prefill_plan"] = prefill_plan_stats()
            report["status"] = "passed"
            destroy_model_parallel()
            destroy_distributed_environment()
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        import traceback
        traceback.print_exc()
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
        if report["status"] != "passed":
            os._exit(1)


if __name__ == "__main__":
    main()
