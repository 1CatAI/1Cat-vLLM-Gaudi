# SPDX-License-Identifier: Apache-2.0
"""TP4 real-weight Attention/mHC admission, including 16K and the next decode bucket."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--tokens", type=int, choices=(128, 8192), default=128)
    parser.add_argument("--physical-tokens", type=int, default=257 * 128)
    args = parser.parse_args()
    if args.physical_tokens < 257 * 128 or args.physical_tokens % 128:
        parser.error("physical tokens must cover 257 whole pages")
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    if "{rank}" in os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", ""):
        os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ["PT_HPU_RECIPE_CACHE_CONFIG"].replace("{rank}", str(rank))
    if os.getenv("GRAPH_VISUALIZATION") == "1":
        graph_dir = Path(os.environ["GRAPH_VISUALIZATION_DIR"]) / f"rank{rank}"
        graph_dir.mkdir(parents=True, exist_ok=True)
        os.environ["GRAPH_VISUALIZATION_DIR"] = str(graph_dir)
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree, linear, _prefill_hc_post
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    from vllm_gaudi.ops.deepseek_v41_math import hc_pre, hc_post, rms_norm, prefill_hc_input
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention, PagedCSA2SharedState
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method="env://", local_rank=rank,
                                 backend="hccl")
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=1))
    output = Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"attention-rank{rank}.json"
    report = {"rank": rank, "tp": 4, "pp": 1, "status": "running", "cases": [], "physical_tokens": args.physical_tokens,
              "boundary": "mHC/RMS -> QKV -> cache/index -> MLA -> wo_a/wo_b -> HCCL -> mHC"}
    try:
        with set_current_vllm_config(config):
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            reduce, gather = stage_collectives(rank, False)
            shard = PreparedV41Shard(args.prepared, 0, rank)
            text = json.loads((args.prepared / "config.json").read_text())["text_config"]
            layers = (0, 2, 8, 14, 20, 24)
            specs = {name: spec for name, spec in shard.specs.items()
                     if any(name.startswith(f"layers.{layer}.{part}") for layer in layers
                            for part in ("attn.", "attn_norm.", "hc_attn_"))}
            tree = _weight_tree(specs)
            load_weight_tree(shard, tree, "hpu", specs,
                             woa_sidecar=WoaFP8Sidecar(args.prepared / "sidecars/wo_a_fp8", shard),
                             woa_layers=layers,
                             dense_sidecar=DenseFP8Sidecar(args.prepared / "sidecars/attention_dense_fp8", shard),
                             dense_config={"wq_b": layers, "wo_b": layers})
            shared = PagedCSA2SharedState(text, 0, 40, "hpu", 1048576)
            # Use real, distinct pages for every reachable row and leave the
            # unused 1M address space unmapped, exactly as the runner does.
            page_end = args.physical_tokens // 128
            shared.block_table[:256].copy_(torch.arange(page_end - 256, page_end, dtype=torch.int32, device="hpu"))
            for cache in shared.sources.values():
                cache.main = torch.zeros(args.physical_tokens // cache.ratio, 288, dtype=torch.uint8, device="hpu")
                cache.index = torch.zeros(args.physical_tokens // cache.ratio, 68, dtype=torch.uint8, device="hpu")
            attention = {}
            for layer in layers:
                w = tree.layers.get_submodule(str(layer))
                op = PagedCSA2Attention(w.attn, text, layer, shared, linear, reduce, gather, "hpu", 4)
                op.prefill_tp_rank = rank
                op.prepare_qkv_input_weight()
                op.prepare_compressor_input_weight()
                op.woa_fp8 = op.woa_output_roundtrip = True
                attention[layer] = op
            eps, hc_eps, iterations = (text[key] for key in ("rms_norm_eps", "hc_eps", "hc_sinkhorn_iters"))
            with torch.inference_mode():
                for step, (start, count, bucket) in enumerate(((0, args.tokens, 8192),
                                                              (args.tokens, args.tokens, 16384),
                                                              (2 * args.tokens, 1, 32768))):
                    shared.prefill_kv_generation += 1
                    shared.prefill_main_workspace.begin(shared.prefill_kv_generation)
                    torch.manual_seed(5810 + step)
                    residual = torch.randn(count, 4, 5120).bfloat16().to("hpu")
                    previous = torch.full((count, 4), .25, device="hpu")
                    positions = torch.arange(start, start + count, dtype=torch.int32, device="hpu")
                    for layer, op in attention.items():
                        op.set_search_length(bucket)
                        op.prefill_token_end = start + count
                        w = tree.layers.get_submodule(str(layer))

                        def execute(residual, previous, positions):
                            if count > 6:
                                collapsed, pre, post, comb, value = prefill_hc_input(
                                    residual, previous, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base,
                                    w.attn_norm.weight, eps, hc_eps, iterations, w.hc_attn_fn)
                            else:
                                value, pre, post, comb = hc_pre(
                                    residual, previous, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base,
                                    eps, hc_eps, iterations, packed_fn=w.hc_attn_fn)
                                value = rms_norm(value, w.attn_norm.weight, eps)
                            projected = op(value, positions, decode=count <= 6)
                            consumer = _prefill_hc_post if count > 6 else hc_post
                            return consumer(projected, residual, post, comb), pre

                        program = torch.compile(execute, backend="hpu_backend", fullgraph=True,
                                                dynamic=False) if count == 1 else execute
                        torch.hpu.synchronize()
                        started = time.perf_counter()
                        result, pre = program(residual, previous, positions)
                        host = result.cpu()
                        elapsed = (time.perf_counter() - started) * 1000
                        assert torch.isfinite(host).all(), (layer, step)
                        # All ranks consume the same all-reduced projection.
                        check = gather(result[:1].flatten(), 0).cpu().reshape(4, -1)
                        assert torch.equal(check, check[:1].expand_as(check)), (layer, "rank divergence")
                        selections = op.selection.indices[:count].cpu() if op.ratio else None
                        if selections is not None:
                            valid = selections >= 0
                            assert ((selections < ((positions.cpu() + 1) // op.ratio)[:, None]) | ~valid).all()
                        report["cases"].append({"layer": layer, "start": start, "tokens": count,
                                                 "bucket": bucket, "cold_host_sync_ms": elapsed,
                                                 "allocated_bytes": torch.hpu.memory_allocated(),
                                                 "peak_bytes": torch.hpu.max_memory_allocated(),
                                                 "finite": True, "rank_agreement": True})
                        output.write_text(json.dumps(report, indent=2) + "\n")
                        print(f"TP{rank} Attention L{layer} start={start} C{count}: passed", flush=True)
                        del result, host, pre, check
            report["status"] = "passed"
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
        if report["status"] != "passed":
            os._exit(1)


if __name__ == "__main__":
    main()
