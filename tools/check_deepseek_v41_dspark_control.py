# SPDX-License-Identifier: Apache-2.0
"""Compare native speculative control through the next real target input GEMMs."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warm-steps", type=int, default=6)
    parser.add_argument("--timing-barrier", type=Path)
    parser.add_argument("--real16", action="store_true",
                        help="Include a real sixteen-layer Target before control and as its next consumer")
    args = parser.parse_args()
    rank, tp = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    if "PT_HPU_RECIPE_CACHE_CONFIG" in os.environ:
        os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ["PT_HPU_RECIPE_CACHE_CONFIG"].replace(
            "{rank}", str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=tp, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from types import SimpleNamespace
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel,
                                init_distributed_environment, initialize_model_parallel)
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.models.deepseek_v41_program import (
        PreparedDecoderLayer, PreparedDraft, PreparedInput, PreparedStage, _weight_tree, linear, load_weight_tree)
    from vllm_gaudi.ops.deepseek_v41_draft_replay import NativeDraftProtocol
    from vllm_gaudi.ops.deepseek_v41_math import hc_pre, rms_norm
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
    from vllm_gaudi.ops.deepseek_v41_qkv import FusedQKVInput
    from vllm_gaudi.ops.deepseek_v41_replay import StageReplay, _Snapshot, stage_collectives, stage_state_tensors
    from vllm_gaudi.ops.deepseek_v41_round_inputs import DeviceRoundInputs
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats, shutdown_prepared_group_plans

    torch.hpu.set_device(rank)
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(status="loading", rank=rank, formal_qualified=False,
                  endpoint="C6 verify/commit -> C5 draft/Markov -> device cursor -> next Target embedding/mHC/QKV",
                  prefix_history="synthetic 128-row MTP context; real checkpoint weights")
    plan = None

    def save():
        (root / f"control-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp))
    with set_current_vllm_config(config), torch.inference_mode():
        try:
            torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
            bind_worker_cpu(rank)
            init_distributed_environment(world_size=tp, rank=rank, local_rank=rank,
                                         distributed_init_method="env://", backend="hccl")
            initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            bind_worker_helpers(rank)
            if os.environ.get("GRAPH_VISUALIZATION") == "1":
                from vllm_gaudi.ops.deepseek_v41_native_trace import configure_post_graph_directory
                configure_post_graph_directory(root / "graphs" / f"rank{rank}")
            reduce, gather = stage_collectives(rank, True, tp)
            shard = PreparedV41Shard(args.prepared, 0, rank)
            text = json.loads((args.prepared / "config.json").read_text())["text_config"]
            specs = {name: spec for name, spec in shard.specs.items()
                     if name.startswith(("mtp.", "layers.0.attn.", "layers.0.hc_attn_", "layers.0.attn_norm."))
                     or name in ("head.weight", "embed.weight", "norm.weight")
                     or args.real16 and name.startswith(tuple(f"layers.{i}." for i in range(20, 36)))}
            tree = _weight_tree(specs)
            load_weight_tree(shard, tree, "hpu", specs)
            shared = PagedCSA2SharedState(text, 20 if args.real16 else 0, 36 if args.real16 else 40,
                                         "hpu", 1048576, tensor_parallel_size=tp)
            stage = SimpleNamespace(weights=tree, config={"text_config": text}, shard=shard, tp_rank=rank,
                                    tensor_parallel_size=tp, reduce=reduce, all_gather=gather, shared=shared,
                                    bf16_head=False)
            draft = PreparedDraft(stage, mxfp4_bf16_lut(torch.device("hpu")), "hpu")
            prefix = torch.compile(draft.verify_prefix, backend="hpu_backend", fullgraph=True, dynamic=False)
            propose = torch.compile(draft.draft_from_prefix, backend="hpu_backend", fullgraph=True, dynamic=False)
            plan = NativeDraftProtocol(draft, generation=1)
            target_stage = target_replay = target_embedding = None
            if args.real16:
                from types import MethodType
                target_stage = torch.nn.Module()
                target_stage.weights, target_stage.config = tree, {"text_config": text}
                target_stage.length, target_stage.search_length, target_stage.generation = 1048576, 32768, 1
                target_stage.pp_rank, target_stage.tp_rank, target_stage.tensor_parallel_size = 0, rank, tp
                target_stage.dspark, target_stage.is_last_stage = True, True
                target_stage.fp8_decode, target_stage.expert_n256, target_stage.bf16_head = True, True, False
                target_stage.precision_fingerprint = ("control-real16", "six-compute-RRMS1")
                target_stage.reduce, target_stage.all_gather, target_stage.shared = reduce, gather, shared
                target_stage.runtime_indexer = shared.runtime_indexer
                shared.block_table[:256].copy_(torch.arange(1, 257, dtype=torch.int32, device="hpu"))
                for cache in shared.sources.values():
                    cache.main = torch.zeros(257 * 128 // cache.ratio, 288, dtype=torch.uint8, device="hpu")
                    cache.index = torch.zeros(257 * 128 // cache.ratio, 68, dtype=torch.uint8, device="hpu")
                target_stage.layers = torch.nn.ModuleList()
                lookup = mxfp4_bf16_lut(torch.device("hpu"))
                for index in range(20, 36):
                    layer = PreparedDecoderLayer(tree.layers.get_submodule(str(index)), text, index, shared,
                                                 shard.manifest["normal_scales"][f"layers.{index}.ffn.experts"][rank],
                                                 lookup, reduce, gather, "hpu", tensor_parallel_size=tp)
                    layer.attention.set_search_length(32768)
                    layer.attention.prepare_qkv_input_weight()
                    layer.attention.prepare_compressor_input_weight()
                    layer.attention.prefill_tp_rank = rank
                    if layer.attention.prepared_output:
                        layer.attention.prepare_output_weight()
                    layer.prepare_mhc_control_weights()
                    target_stage.layers.append(layer)
                for name in ("_forward_impl", "forward"):
                    setattr(target_stage, name, MethodType(getattr(PreparedStage, name), target_stage))
                shared.prepare_index_mirror(16384)
                target_replay = StageReplay(target_stage)
                target_embedding = torch.compile(PreparedInput(tree.embed, rank, reduce), backend="hpu_backend",
                                                 fullgraph=True, dynamic=False)
                report.update(real_target_layers=list(range(20, 36)),
                              auxiliary_fixture="partial Target hidden repeated across the three MTP context fields",
                              endpoint="real16 Target -> verify/draft/Markov -> device cursor -> next real16 Target")

            class NextInput(FusedQKVInput, torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = PreparedInput(tree.embed, rank, reduce)
                    self.block = tree.layers.get_submodule("0")
                    self.weights, self.linear = self.block.attn, linear
                    self.qkv_fused_input, self._fused_qkv_weight = True, None
                    self.prepare_qkv_input_weight()

                def forward(self, ids):
                    residual, pre = self.embedding(ids)
                    value, _, _, _ = hc_pre(
                        residual, pre, self.block.hc_attn_fn, self.block.hc_attn_scale, self.block.hc_attn_base,
                        text["rms_norm_eps"], text["hc_eps"], text["hc_sinkhorn_iters"],
                        packed_fn=self.block.hc_attn_fn.contiguous(), decode=True)
                    return self._project_qkv_input(rms_norm(value, self.block.attn_norm.weight,
                                                          text["rms_norm_eps"]))

            consumer = torch.compile(NextInput(), backend="hpu_backend", fullgraph=True, dynamic=False)
            cursor = DeviceRoundInputs(torch.empty(6, dtype=torch.int64, device="hpu"),
                                       torch.empty(6, dtype=torch.int32, device="hpu"))
            histories = torch.zeros(7, 3, dtype=torch.int32, device="hpu")
            generator = torch.Generator().manual_seed(1806)
            hidden = torch.randn(6, 5120, generator=generator).bfloat16().to("hpu")
            auxiliary = torch.randn(6, 15360, generator=generator).bfloat16().to("hpu")
            context_positions = torch.arange(16256, 16384, dtype=torch.int32, device="hpu")
            context = torch.randn(128, 15360, generator=generator).bfloat16().to("hpu")
            draft.insert_context(context, context_positions)
            torch.hpu.synchronize()
            initial = _Snapshot((*tuple(layer.attention.swa for layer in draft.layers),
                                 *(stage_state_tensors(target_stage) if target_stage is not None else ())))
            argmax = torch.compile(lambda x: draft._global_argmax(draft._head_projection(x)),
                                   backend="hpu_backend", fullgraph=True, dynamic=False)
            target = argmax(hidden).cpu().tolist()

            def prepare(index):
                proposals = target[:5].copy()
                if index % 6 < 5:
                    slot = index % 6
                    proposals[slot] = (proposals[slot] + 1) % 129280
                cursor.retire("fixture") if cursor.owner is not None else None
                cursor.seed("fixture", [target[0], *proposals], 16384, 1024, 1048576, index + 1, [-1] * 3)
                torch.hpu.synchronize()

            def invoke(native):
                target_hidden, target_auxiliary = hidden, auxiliary
                if args.real16:
                    target_hidden = target_replay(*target_embedding(cursor.ids), cursor.positions, cursor.ids, ())[0]
                    target_auxiliary = target_hidden.repeat(1, 3)
                control, proposals = cursor.control[:7], cursor.control[7:]
                if native:
                    values = plan(target_hidden, proposals, control, target_auxiliary, cursor.positions)
                else:
                    verify = prefix(target_hidden, proposals, control, target_auxiliary, cursor.positions)
                    record, wire, confidence = propose(control, cursor.positions, *verify[1:7])
                    values = record, wire, verify[0], confidence
                cursor.next(values[0], histories)
                next_values = (target_replay(*target_embedding(cursor.ids), cursor.positions, cursor.ids, ())[:2]
                               if args.real16 else consumer(cursor.ids))
                return *values, *next_values

            expected = []
            for native in (False, True):
                initial.restore()
                output = []
                for index in range(6):
                    prepare(index)
                    output.append(tuple(x.cpu() for x in invoke(native)))
                expected.append(output)
            for old, new in zip(*expected, strict=True):
                for lhs, rhs in zip(old, new, strict=True):
                    torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
            plan.require_ready()
            report.update(status="correctness_passed", fixture_cases=list(range(6)),
                          forced_prefix_counts=None if args.real16 else list(range(1, 7)),
                          committed_counts=[int(values[0].reshape(-1)[1]) for values in expected[0]],
                          exact=True, blocks=[])
            save()
            while args.timing_barrier is not None and not args.timing_barrier.exists():
                time.sleep(1)
            for native in (False, True, False, True, False, True):
                initial.restore()
                for index in range(args.warm_steps):
                    prepare(index)
                    invoke(native)
                torch.hpu.synchronize()
                before = prepared_group_stats()
                from torch._dynamo.utils import counters
                compile_before = dict(counters["stats"])
                samples = []
                for index in range(args.steps):
                    prepare(index)
                    begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    begin.record()
                    started = time.perf_counter_ns()
                    invoke(native)
                    end.record()
                    end.synchronize()
                    samples.append(dict(device_ms=begin.elapsed_time(end),
                                        drained_host_ms=(time.perf_counter_ns() - started) / 1e6))
                report["blocks"].append(dict(native=native, samples=samples, before=before,
                                             after=prepared_group_stats(), compile_before=compile_before,
                                             compile_after=dict(counters["stats"])))
                save()
            report["status"] = "completed_component_ab"
            save()
        except Exception as error:
            report.update(status="failed", error=repr(error))
            save()
            raise
        finally:
            torch.hpu.synchronize()
            if plan is not None:
                plan.close()
            shutdown_prepared_group_plans()
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
