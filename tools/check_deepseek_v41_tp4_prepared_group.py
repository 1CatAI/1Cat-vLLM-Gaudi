# SPDX-License-Identifier: Apache-2.0
"""Qualify prepared TP4 groups against the original compiled four-layer chain."""
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
    parser.add_argument("--measure-missing-reference", action="store_true")
    parser.add_argument("--reuse-reference-entry", action="store_true")
    parser.add_argument("--native-inputs", action="store_true")
    parser.add_argument("--state-contracts", action="store_true")
    parser.add_argument("--packed-engram", action="store_true")
    parser.add_argument("--native-binding-probe", action="store_true",
                        help="Check native recipe bindings on initialized zero KV; no prefill or performance claim")
    parser.add_argument("--native-binding-layers", type=int, choices=(4, 16), default=4)
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
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
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
            if args.native_binding_probe:
                reduce(torch.ones(1, device='hpu', dtype=torch.bfloat16)).cpu()
                bind_worker_helpers(rank)
            shard = PreparedV41Shard(args.prepared, 0, rank)
            layers = range(args.native_binding_layers if args.native_binding_probe else 4)
            text = json.loads((args.prepared / "config.json").read_text())["text_config"]
            specs = {name: spec for name, spec in shard.specs.items()
                     if name == "norm.weight" or name.startswith(tuple(f"layers.{i}." for i in layers))}
            tree = _weight_tree(specs)
            load_weight_tree(shard, tree, "hpu", specs,
                             woa_sidecar=WoaFP8Sidecar(args.prepared / "sidecars/wo_a_fp8", shard),
                             woa_layers=layers,
                             dense_sidecar=DenseFP8Sidecar(args.prepared / "sidecars/attention_dense_fp8", shard),
                             dense_config={"wq_b": layers, "wo_b": layers},
                             engram_sidecar=EngramFP8Sidecar(args.prepared / "sidecars/engram_fp8", shard))
            stage = torch.nn.Module()
            stage.weights, stage.config = tree, {"text_config": text}
            stage.length, stage.search_length = 1048576, 32768
            stage.fp8_decode, stage.expert_n256 = True, True
            stage.tensor_parallel_size, stage.pp_rank, stage.tp_rank = 4, 0, rank
            stage.dspark, stage.is_last_stage = False, True
            stage.generation = 1
            stage.reduce, stage.all_gather = reduce, gather
            stage.shared = PagedCSA2SharedState(text, 0, len(layers), "hpu", stage.length,
                                               tensor_parallel_size=4 if args.native_binding_probe else 2)
            stage.shared.block_table[:256].copy_(torch.arange(1, 257, dtype=torch.int32, device="hpu"))
            for cache in stage.shared.sources.values():
                cache.main = torch.zeros(257 * 128 // cache.ratio, 288, dtype=torch.uint8, device="hpu")
                cache.index = torch.zeros(257 * 128 // cache.ratio, 68, dtype=torch.uint8, device="hpu")
            lookup = mxfp4_bf16_lut(torch.device("hpu"))
            stage.layers = torch.nn.ModuleList()
            for layer in layers:
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
            candidate = CompiledStage(stage, prepared_tp4=True, native_tp4=args.native_binding_probe)
            reference = CompiledStage(stage, prepared_tp4=False)
            if args.native_binding_probe:
                stage.decode_token_bound = 20480
                stage.shared.prepare_index_mirror(16384)
                for block in stage.layers:
                    block.attention.set_search_length(stage.search_length)
                    block.attention.set_decode_visible_tokens(stage.decode_token_bound)
                report['scope'] = f'Zero KV {len(layers)}-layer native binding correctness; no timing or gain credit'
            control_frames = {}
            from vllm_gaudi.ops.deepseek_v41_math import pack_swa

            def engram_inputs(count):
                value = torch.randn(count, 6, 256).bfloat16()
                if not args.packed_engram:
                    rows = value.to('hpu')
                    return rows, rows
                packed = pack_swa(value)
                packet = torch.stack((packed, packed)).to('hpu')
                # Normal C1 packets expose two disjoint, contiguous views of
                # one byte allocation. Retain their storage-alias contract.
                return packet[0], packet[1]

            report['engram_layout'] = 'packed_uint8_packet' if args.packed_engram else 'bf16_reference_fixture'
            if args.native_inputs:
                from vllm_gaudi.ops.deepseek_v41_completion import prepare_token_readback
                from vllm_gaudi.distributed.tp2_fused_ar_norm import _load_bridge
                prepare_token_readback()  # validate artifact/runtime identity before loading input ABI
                bridge = _load_bridge(Path(os.environ['VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE']).resolve())
                all_ids = torch.empty(128, dtype=torch.int64, device='hpu')
                all_positions = torch.empty(128, dtype=torch.int32, device='hpu')
                for count in (1, 2, 6):
                    ids, positions = all_ids[:count], all_positions[:count]
                    control_frames[count] = ids, positions, bridge.PreparedControlInputs(ids, positions)
            # Fill the complete 16K state through the real large-M producer.
            for start in (() if args.native_binding_probe else (0, 8192)):
                torch.manual_seed(3310 + start)
                residual = torch.randn(8192, 4, 5120).bfloat16().to("hpu")
                pre = torch.full((8192, 4), .25, device="hpu")
                positions = torch.arange(start, start + 8192, dtype=torch.int32, device="hpu")
                ids = torch.arange(8192, device="hpu")
                rows = engram_inputs(8192)
                for block in stage.layers:
                    block.attention.set_search_length(start + 8192)
                    block.attention.prefill_token_end = start + 8192
                output = stage(residual, pre, positions, ids, rows)[0].cpu()
                assert torch.isfinite(output).all()
                print(f"TP{rank} group prefill {start}+8192 passed", flush=True)
            cases = [(16384, 1), (16385, 1), (16386, 2), (16388, 6), (16394, 1)]
            if args.native_binding_probe:
                cases = [(16384 + step, 1) for step in range(3)]
            if args.state_contracts:
                cases.extend(((16395, 2), (16397, 6), (0, 2)))
            for step, (position, count) in enumerate(cases):
                retired = []
                contract = 'ordinary'
                if step == 5:
                    # Keep logical history fixed while its physical pages move.
                    for cache in stage.shared.sources.values():
                        width = 128 // cache.ratio
                        for name in ('main', 'index'):
                            value = getattr(cache, name)
                            saved = value[width:2*width].clone()
                            value[width:2*width].copy_(value[2*width:3*width])
                            value[2*width:3*width].copy_(saved)
                    stage.shared.block_table[:2].copy_(torch.tensor([2, 1], dtype=torch.int32, device='hpu'))
                    contract = 'physical_page_reorder'
                elif step == 6:
                    # Allocation rebinding must invalidate prepared static
                    # arguments, including aliases shared by several layers.
                    old = {id(value): value for value in stage_state_tensors(stage)}
                    fresh = {key: value.clone() for key, value in old.items()}
                    retired = [(value, value.cpu()) for value in old.values()]
                    for name, value in list(stage.named_buffers(remove_duplicate=False)):
                        if id(value) in fresh:
                            parent, _, leaf = name.rpartition('.')
                            setattr(stage.get_submodule(parent), leaf, fresh[id(value)])
                    stage.generation += 1
                    contract = 'physical_state_rebind'
                elif step == 7:
                    state_names = {id(value): name for name, value in stage.named_buffers()}
                    for value in stage_state_tensors(stage):
                        if value is not stage.shared.block_table:
                            value.fill_(-1 if state_names[id(value)].rsplit('.', 1)[-1]
                                        in ('indices', 'candidate_pool') else 0)
                    contract = 'request_slot_reuse'
                for block in stage.layers:
                    block.attention.set_search_length(32768)
                    block.attention.prefill_token_end = None
                torch.manual_seed(5511 + step)
                residual = torch.randn(count, 4, 5120).bfloat16().to("hpu")
                pre = torch.full((count, 4), .25, device="hpu")
                if args.native_inputs:
                    ids, positions, control = control_frames[count]
                    control.upload(list(range(7, 7 + count)), position)
                else:
                    positions = torch.arange(position, position + count, dtype=torch.int32, device="hpu")
                    ids = torch.arange(count, device="hpu") + 7
                rows = engram_inputs(count)
                arguments = residual, pre, positions, ids, rows
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
                for value, saved in retired:
                    torch.testing.assert_close(value.cpu(), saved, rtol=0, atol=0)
                del retired
                rank_values = gather(actual.to("hpu"), 0).cpu().reshape(4, count, 5120)
                assert torch.equal(rank_values, rank_values[:1].expand_as(rank_values))
                del snapshot, state_reference
                # Match normal serving after all compiler/runtime pools have
                # been created; the older component harness only pinned main.
                bind_worker_helpers(rank)
                masks = {}
                for task in Path(f"/proc/{os.getpid()}/task").iterdir():
                    try:
                        mask = next(line.split(":", 1)[1].strip() for line in (task / "status").read_text().splitlines()
                                    if line.startswith("Cpus_allowed_list:"))
                        masks[mask] = masks.get(mask, 0) + 1
                    except (FileNotFoundError, StopIteration):
                        pass
                report["runtime_thread_affinity"] = masks
                matched_entry = None
                if step == 1 and args.measure_missing_reference and args.packed_engram:
                    matched_entry = _Snapshot(stage_state_tensors(stage)), residual.clone()
                replay_steps = 12 if step == 1 and args.reuse_reference_entry else 0
                for _ in range(replay_steps):
                    # Reproduce the saved reference's changing-input entry
                    # state using only the new candidate, without timing B.
                    residual.add_(.001)
                    consumer = candidate(*arguments)[0].cpu()
                    assert torch.isfinite(consumer).all()
                timings = []
                for repeat in range(12 if step < 5 and not args.native_binding_probe else 0):
                    residual.add_(.001)
                    torch.hpu.synchronize()
                    started = time.perf_counter()
                    consumer = candidate(*arguments)[0].cpu()
                    timings.append((time.perf_counter() - started) * 1000)
                    assert torch.isfinite(consumer).all()
                matched_reference = []
                if step == 1 and args.measure_missing_reference:
                    # One missing component reference under corrected runtime
                    # affinity. This does not repeat the serving baseline.
                    if matched_entry is not None:
                        matched_entry[0].restore()
                        residual.copy_(matched_entry[1])
                    for repeat in range(12):
                        residual.add_(.001)
                        torch.hpu.synchronize()
                        started = time.perf_counter()
                        consumer = reference(*arguments)[0].cpu()
                        matched_reference.append((time.perf_counter() - started) * 1000)
                        assert torch.isfinite(consumer).all()
                report["cases"].append(dict(position=position, tokens=count, contract=contract,
                                            exact_output=True, exact_state=True,
                                            host_sync_ms=timings, reference_host_sync_ms=matched_reference,
                                            reference_entry_replay_steps=replay_steps,
                                            reference_same_state_and_inputs=matched_entry is not None,
                                            peak_bytes=torch.hpu.max_memory_allocated()))
                if args.native_binding_probe:
                    from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats
                    report['native_groups'] = prepared_group_stats()
                    if step == 2:
                        assert report['native_groups']['native_replays'] > 0
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
                        for repeat in range(12):
                            with torch.profiler.record_function(f"TP4-group::iteration{repeat}"):
                                residual.add_(.001)
                                consumer = candidate(*arguments)[0].cpu()
                    profiler.export_chrome_trace(str(root / f"group-trace-rank{rank}.json.gz"))
                    break
            report.update(status="passed", compiled_dispatch=candidate.audit,
                          prepared_exports=[chunk.preparations for chunk in candidate.chunks])
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status="failed", error=repr(error))
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
        if args.native_binding_probe:
            from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans
            shutdown_prepared_group_plans()
        if report["status"] != "passed":
            os._exit(1)


if __name__ == "__main__":
    main()
