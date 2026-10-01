# SPDX-License-Identifier: Apache-2.0
"""Qualify the actual four-layer compiled TP4 chain against one-layer boundaries."""
import argparse
import json
import os
from pathlib import Path
import time
from types import MethodType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--profile-first-decode", action="store_true")
    parser.add_argument("--raw-profile-steps", type=int, default=0)
    args = parser.parse_args()
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
    from vllm_gaudi.models.deepseek_v41_program import (
        PreparedStage, PreparedDecoderLayer, CompiledStage, _weight_tree, load_weight_tree)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives, stage_state_tensors, _Snapshot
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_engram_fp8 import EngramFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method="env://", local_rank=rank,
                                 backend="hccl")
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=1))
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    path = root / f"group-rank{rank}.json"
    report = dict(rank=rank, status="running", cases=[],
                  boundary="Engram/mHC -> four real Attention/FFN layers -> final collapse/RMS consumer")
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            reduce, gather = stage_collectives(rank, False)
            shard = PreparedV41Shard(args.prepared, 0, rank)
            text = json.loads((args.prepared / "config.json").read_text())["text_config"]
            specs = {name: spec for name, spec in shard.specs.items()
                     if name == "norm.weight" or name.startswith(tuple(f"layers.{i}." for i in range(4)))}
            tree = _weight_tree(specs)
            load_weight_tree(shard, tree, "hpu", specs,
                             woa_sidecar=WoaFP8Sidecar(args.prepared / "sidecars/wo_a_fp8", shard),
                             woa_layers=range(4),
                             dense_sidecar=DenseFP8Sidecar(args.prepared / "sidecars/attention_dense_fp8", shard),
                             dense_config={"wq_b": range(4), "wo_b": range(4)},
                             engram_sidecar=EngramFP8Sidecar(args.prepared / "sidecars/engram_fp8", shard))
            stage = torch.nn.Module()
            stage.weights, stage.config = tree, {"text_config": text}
            stage.length, stage.search_length = 1048576, 32768
            stage.fp8_decode, stage.expert_n256 = True, True
            stage.tensor_parallel_size, stage.pp_rank, stage.tp_rank = 4, 0, rank
            stage.dspark, stage.is_last_stage = False, True
            stage.reduce, stage.all_gather = reduce, gather
            stage.shared = PagedCSA2SharedState(text, 0, 4, "hpu", stage.length)
            stage.shared.block_table[:256].copy_(torch.arange(1, 257, dtype=torch.int32, device="hpu"))
            for cache in stage.shared.sources.values():
                cache.main = torch.zeros(257 * 128 // cache.ratio, 288, dtype=torch.uint8, device="hpu")
                cache.index = torch.zeros(257 * 128 // cache.ratio, 68, dtype=torch.uint8, device="hpu")
            lookup = mxfp4_bf16_lut(torch.device("hpu"))
            stage.layers = torch.nn.ModuleList()
            for layer in range(4):
                normal = shard.manifest["normal_scales"][f"layers.{layer}.ffn.experts"][rank]
                block = PreparedDecoderLayer(tree.layers.get_submodule(str(layer)), text, layer, stage.shared,
                                             normal, lookup, reduce, gather, "hpu", tensor_parallel_size=4)
                block.moe.prepare_shared_gate_up_weight()
                block.prepare_mhc_control_weights()
                block.attention.prefill_tp_rank = rank
                block.attention.prepare_qkv_input_weight()
                block.attention.prepare_compressor_input_weight()
                block.attention.woa_fp8 = block.attention.woa_output_roundtrip = True
                stage.layers.append(block)
            stage._forward_impl = MethodType(PreparedStage._forward_impl, stage)
            stage.forward = MethodType(PreparedStage.forward, stage)
            candidate, reference = CompiledStage(stage), CompiledStage(stage, group_size=1)
            # Fill the complete 16K state through the real large-M producer.
            for start in (0, 8192):
                torch.manual_seed(3310 + start)
                residual = torch.randn(8192, 4, 5120).bfloat16().to("hpu")
                pre = torch.full((8192, 4), .25, device="hpu")
                positions = torch.arange(start, start + 8192, dtype=torch.int32, device="hpu")
                ids = torch.arange(8192, device="hpu")
                rows = torch.randn(8192, 6, 256).bfloat16().to("hpu")
                for block in stage.layers:
                    block.attention.set_search_length(start + 8192)
                    block.attention.prefill_token_end = start + 8192
                output = stage(residual, pre, positions, ids, (rows, rows))[0].cpu()
                assert torch.isfinite(output).all()
                print(f"TP{rank} group prefill {start}+8192 passed", flush=True)
            for step, (position, count) in enumerate(((16384, 1), (16385, 1), (16386, 6), (16392, 1))):
                for block in stage.layers:
                    block.attention.set_search_length(32768)
                    block.attention.prefill_token_end = None
                torch.manual_seed(5511 + step)
                residual = torch.randn(count, 4, 5120).bfloat16().to("hpu")
                pre = torch.full((count, 4), .25, device="hpu")
                positions = torch.arange(position, position + count, dtype=torch.int32, device="hpu")
                ids = torch.arange(count, device="hpu") + 7
                rows = torch.randn(count, 6, 256).bfloat16().to("hpu")
                arguments = residual, pre, positions, ids, (rows, rows)
                states = stage_state_tensors(stage)
                snapshot = _Snapshot(states)
                expected = reference(*arguments)[0].cpu()
                state_reference = [value.cpu() for value in states]
                snapshot.restore()
                actual = candidate(*arguments)[0].cpu()
                torch.save(dict(actual=actual, expected=expected), root / f"group-rank{rank}-step{step}.pt")
                names = {id(value): name for name, value in stage.named_buffers()}
                state_diffs = []
                for value, wanted in zip(states, state_reference):
                    observed = value.cpu()
                    changed = int((observed != wanted).sum())
                    if changed:
                        state_diffs.append(dict(name=names.get(id(value)), changed=changed,
                                                maximum=float((observed.float() - wanted.float()).abs().max())))
                (root / f"state-diff-rank{rank}-step{step}.json").write_text(json.dumps(state_diffs, indent=2))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert not state_diffs, state_diffs
                rank_values = gather(actual.to("hpu"), 0).cpu().reshape(4, count, 5120)
                assert torch.equal(rank_values, rank_values[:1].expand_as(rank_values))
                del snapshot, state_reference
                timings = []
                for repeat in range(3):
                    residual.add_(.001)
                    torch.hpu.synchronize()
                    started = time.perf_counter()
                    consumer = candidate(*arguments)[0].cpu()
                    timings.append((time.perf_counter() - started) * 1000)
                    assert torch.isfinite(consumer).all()
                report["cases"].append(dict(position=position, tokens=count, exact_output=True, exact_state=True,
                                            host_sync_ms=timings, peak_bytes=torch.hpu.max_memory_allocated()))
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(f"TP{rank} compiled group C{count} P{position} passed", flush=True)
                if args.raw_profile_steps:
                    from vllm_gaudi.ops.deepseek_v41_native_trace import NativeTrace, scope
                    profiler = NativeTrace(cpu_trace_dir=root / f"cpu-rank{rank}")
                    profiler.start()
                    for repeat in range(args.raw_profile_steps):
                        with scope(f"TP4-group::iteration{repeat}"):
                            residual.add_(.001)
                            consumer = candidate(*arguments)[0].cpu()
                            assert torch.isfinite(consumer).all()
                    profiler.stop()
                    report["raw_profile"] = profiler.metadata
                    break
                if args.profile_first_decode:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                             torch.profiler.ProfilerActivity.HPU],
                                                record_shapes=True) as profiler:
                        for repeat in range(3):
                            with torch.profiler.record_function(f"TP4-group::iteration{repeat}"):
                                residual.add_(.001)
                                consumer = candidate(*arguments)[0].cpu()
                    profiler.export_chrome_trace(str(root / f"group-trace-rank{rank}.json.gz"))
                    break
            report.update(status="passed", compiled_dispatch=candidate.audit)
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status="failed", error=repr(error))
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
        if report["status"] != "passed":
            os._exit(1)


if __name__ == "__main__":
    main()
