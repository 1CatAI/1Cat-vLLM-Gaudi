# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Real four-layer stages through ordinary TP2/PP2 batch transport.

This component excludes embedding, Engram and the vocabulary head. Its final
consumer samples the real PP1 residual; it does not qualify service quality.
"""
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

rank = int(os.environ["LOCAL_RANK"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
if os.environ.get("DSV41_MICRO_RANK_CPUS"):
    rank_cpus = json.loads(os.environ["DSV41_MICRO_RANK_CPUS"])[rank]
else:
    # The normal launcher records current NUMA-local leases under the worker
    # names. Do not inherit all four ranks' CPUs or hard-code old module IDs.
    main_cpu = int(os.environ["VLLM_HPU_DSV4_WORKER_CPUS"].split(",")[rank])
    helpers = os.environ["VLLM_HPU_DSV4_WORKER_HELPER_CPUS"].split(";")[rank]
    rank_cpus = [main_cpu, *(int(cpu) for cpu in helpers.split(",") if cpu)]
os.sched_setaffinity(0, rank_cpus)

if os.environ.get("GRAPH_VISUALIZATION") == "1":
    # This Synapse build writes .graph_dumps relative to cwd and ignores
    # GRAPH_VISUALIZATION_DIR. Isolate ranks and runs before any compilation.
    graph_cwd = Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"compiler-rank{rank}"
    graph_cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(graph_cwd)

import torch
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel, get_pp_group,
                              init_distributed_environment, initialize_model_parallel)
from vllm.sequence import IntermediateTensors
from vllm_gaudi import envs
from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
from vllm_gaudi.models.deepseek_v41_batch_program import PreparedBatchLayerGroup
from vllm_gaudi.models.deepseek_v41_program import PreparedDecoderLayer, _weight_tree, load_weight_tree
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
from vllm_gaudi.ops.deepseek_v41_batch_state import BatchStageState
from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar, precision_config
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar, layer_selection
from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats, shutdown_prepared_group_plans
from vllm_gaudi.v1.worker.deepseek_v41_batch_runner import BatchExecution
from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState
from deepseek_v41_native_fragment import NativeBatchFragment
from deepseek_v41_micro_contexts import micro_contexts


class Model:

    def __init__(self, prepared):
        self.pp_rank, self.engram_host = rank // 2, None
        first = 2 if self.pp_rank == 0 else 24
        stop, capacity = first + 4, 64
        self.context_profile = os.environ.get("CONCURRENT_CONTEXT_PROFILE", "legacy")
        self.page_count = 32 if self.context_profile == "2k-decode" else 16
        if self.context_profile not in ("legacy", "2k-decode"):
            raise ValueError("Unknown PP component context profile")
        shard = PreparedV41Shard(prepared, self.pp_rank, rank % 2)
        config = json.loads((prepared / "config.json").read_text())["text_config"]
        specs = {
            k: v
            for k, v in shard.specs.items() if k.startswith(tuple(f"layers.{i}." for i in range(first, stop)))
        }
        weights = _weight_tree(specs)
        dc = precision_config(envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG)
        wc = layer_selection(envs.VLLM_HPU_DSV41_WO_A_FP8_CONFIG)
        load_weight_tree(shard,
                         weights,
                         "hpu",
                         specs,
                         woa_sidecar=WoaFP8Sidecar(envs.VLLM_HPU_DSV41_WO_A_FP8_SIDECAR, shard),
                         woa_layers=wc["layers"],
                         dense_sidecar=DenseFP8Sidecar(envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR, shard),
                         dense_config=dc)
        source = max(i for i in config["kv_source_layer_ids"] if i <= first)
        shared = PagedCSA2SharedState(config, source, stop, "hpu", 1048576)
        cache = shared.sources[str(source)]
        self.ratio = cache.ratio
        rows = (capacity * self.page_count + 1) * (128 // cache.ratio)
        cache.main = torch.zeros(rows, 288, dtype=torch.uint8, device="hpu")
        cache.index = torch.zeros(rows, 68, dtype=torch.uint8, device="hpu")
        reduce, gather = stage_collectives(rank % 2, True)
        lookup = mxfp4_bf16_lut("hpu")
        layers = torch.nn.ModuleList([
            PreparedDecoderLayer(weights.layers.get_submodule(str(i)), config, i, shared, True, lookup, reduce, gather,
                                 "hpu") for i in range(first, stop)
        ])
        # These are interior layer fragments, so omit the decoder-tail norm.
        self.program = SimpleNamespace(layers=layers,
                                       shared=shared,
                                       config={"text_config": config},
                                       dspark=False,
                                       pp_rank=0,
                                       length=1048576,
                                       sample_greedy_token=lambda x: x[:, 0, :].float().argmax(-1, keepdim=True).int())
        self.batch_state = BatchStageState(self.program, capacity)
        for layer in layers:
            at = layer.attention
            at.woa_fp8 = layer.layer in wc["layers"]
            at.prepare_output_weight()
            at.prepare_qkv_input_weight()
            at.prepare_compressor_input_weight()
            at.set_search_length(1048576)
            layer.moe.prepare_shared_gate_up_weight()
            at.batch_state.swa.copy_(pack_swa(torch.randn(capacity * 256, 512).bfloat16()).to("hpu"))
            if hasattr(at.batch_state, "kv_history"):
                at.batch_state.kv_history.copy_(torch.randn(capacity * 8, 512))
                at.batch_state.score_history.copy_(torch.randn(capacity * 8, 512))
        for start in range(0, rows, 4096):
            count = min(4096, rows - start)
            cache.main[start:start + count].copy_(pack_fp4(torch.randn(count, 512).bfloat16(), 16).to("hpu"))
            cache.index[start:start + count].copy_(pack_fp4(torch.randn(count, 128).bfloat16(), 32).to("hpu"))
        self.states = [v for layer in layers for v in layer.attention.batch_state.buffers()] + [cache.main, cache.index]
        self.original = [v.clone() for v in self.states]
        self.seed = torch.randn(5120, 4, 5120).bfloat16().to("hpu") if self.pp_rank == 0 else None
        self.entries, self.last_outputs = {}, {}
        self.completed = 0

    def restore(self):
        for destination, value in zip(self.states, self.original, strict=True):
            destination.copy_(value)

    def complete_request_batch(self, counts):
        if any(c != 1 for c in counts):
            raise RuntimeError("Ordinary pipeline must consume one input per request")
        self.completed += len(counts)

    def forward_request_batch(self, ids, positions, slots, pages, spans, intermediate_tensors=None, *, lane=0):
        b = ids.numel()
        if self.pp_rank == 0:
            hidden = self.seed.index_select(0, ids.long())
            pre = torch.full((b, 4), 0.5, dtype=torch.float32, device="hpu")
        else:
            hidden, pre = intermediate_tensors["hidden_states"], intermediate_tensors["pre_mix"]
        key = lane, b
        host_positions = tuple(s[1] for s in spans) + (-1, ) * (b - len(spans))
        if key not in self.entries:
            from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
            group = PreparedBatchLayerGroup(self.program, 0, 4)
            compiled = torch.compile(group, backend=make_backend(), fullgraph=True, dynamic=False)
            fragment = NativeBatchFragment(group, compiled, self.states, self.restore,
                                           sum(2 for layer in self.program.layers if layer.attention.owns_index),
                                           host_positions)
            selected = tuple(
                torch.full((b, 512), -1, dtype=torch.int32, device="hpu") for _ in self.program.shared.topk)
            pool = torch.full((b, 2048), -1, dtype=torch.int32, device="hpu")
            if self.pp_rank == 1:
                if self.context_profile == "legacy":
                    pool[:, :256] = torch.arange(256, dtype=torch.int32, device="hpu")
                else:
                    # Match the short-context Full layer's ordered candidate
                    # prefix, including its -1 tail and inactive rows. This
                    # is immutable fixture setup, outside timed submissions.
                    cpu_pool = torch.full((b, 2048), -1, dtype=torch.int32)
                    for row, position in enumerate(host_positions):
                        visible = (position + 1) // self.ratio
                        blocks = (visible + 7) // 8
                        cpu_pool[row, :blocks] = torch.arange(blocks, dtype=torch.int32)
                    pool.copy_(cpu_pool)
            ready = tuple(torch.zeros(b, dtype=torch.int32, device="hpu") for _ in self.program.shared.sources)
            self.entries[key] = fragment, selected, pool, ready
        fragment, selected, pool, ready = self.entries[key]
        fragment.host_positions = host_positions
        result = fragment(hidden, pre, positions, ids, (), slots, pages, selected, pool, ready, ready)
        self.last_outputs[lane] = tuple(result)
        return (IntermediateTensors({
            "hidden_states": result[0],
            "pre_mix": result[1]
        }) if self.pp_rank == 0 else result[0])

    def close(self):
        for fragment, *_ in self.entries.values():
            fragment.close()
            fragment.metadata.native_completion = None
        self.entries.clear()


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(1191)
    torch.hpu.set_device(rank)
    init_distributed_environment(world_size=4,
                                 rank=rank,
                                 distributed_init_method="env://",
                                 local_rank=rank,
                                 backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
    initialize_tp2_fused_ar_norm_runtime()
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    model = Model(Path(os.environ["DSV41_PREPARED"]))
    runner = SimpleNamespace(model=model,
                             device=torch.device("hpu"),
                             state=SimpleNamespace(blocks=64 * model.page_count + 1),
                             pp=SimpleNamespace(group=get_pp_group()),
                             audit=dict(decode_steps=0, target_steps=0, target_tokens=0))
    batch = BatchExecution(runner, 64)
    pipeline = batch.pipeline
    count = int(os.environ.get("CONCURRENT_MICRO_BATCH", "32"))
    positions = (micro_contexts(count, 24, model.context_profile)[0]
                 if model.context_profile == "2k-decode" else [767] * count)
    requests = [
        RequestState(f"request{i}", [i + 1] * (positions[i] + 1), [], None,
                     (list(range(1 + i * model.page_count, 1 + (i + 1) * model.page_count)), )) for i in range(count)
    ]
    for request, position in zip(requests, positions, strict=True):
        request.num_computed_tokens = position
        model.batch_state.acquire(request.req_id)
    report = dict(batch=count,
                  scope="4 real layers per PP stage; no embedding/Engram/vocabulary head",
                  context_profile=model.context_profile,
                  initial_positions=positions,
                  cache_pages_per_request=model.page_count,
                  candidate_pool_layout="full-producer" if model.context_profile == "2k-decode" else "legacy",
                  timings={},
                  checks={})
    evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
    outputs, states = {}, {}
    for name, selected_pipeline in (("whole", None), ("pipeline", pipeline)):
        batch.pipeline = selected_pipeline
        for _ in range(4):
            model.restore()
            batch.execute(requests)
        for fragment, *_ in model.entries.values():
            fragment.require_ready()
        model.restore()
        model.last_outputs.clear()
        batch.execute(requests)
        outputs[name] = tuple(
            torch.cat([values[i].cpu() for _, values in sorted(model.last_outputs.items())])
            for i in range(len(next(iter(model.last_outputs.values())))))
        states[name] = [v.cpu() for v in model.states]
        (evidence / f"rank{rank}.json").write_text(json.dumps(report, indent=2))
    report["checks"] = dict(
        outputs=[
            dict(exact=torch.equal(a, b),
                 max_abs=float((a.float() - b.float()).abs().max()),
                 different=int((a != b).sum())) for a, b in zip(outputs["whole"], outputs["pipeline"], strict=True)
        ],
        states=[torch.equal(a, b) for a, b in zip(states["whole"], states["pipeline"], strict=True)])
    (evidence / f"rank{rank}.json").write_text(json.dumps(report, indent=2))
    torch.save(outputs, evidence / f"rank{rank}-outputs.pt")
    local_ok = all(row["exact"] for row in report["checks"]["outputs"]) and all(report["checks"]["states"])
    all_ok = torch.tensor(int(local_ok), dtype=torch.int32, device="hpu")
    torch.distributed.all_reduce(all_ok, op=torch.distributed.ReduceOp.MIN)
    if not all_ok.item():
        raise RuntimeError("PP split changed a real output/state; performance sampling is not authorized")
    for name, selected_pipeline in (("whole", None), ("pipeline", pipeline)):
        batch.pipeline = selected_pipeline
        samples = []
        # Reuse a saved component parent whenever its context and full chain
        # match. A new profile can measure its missing component reference.
        repetitions = 0 if name == "whole" and os.environ.get("CONCURRENT_CANDIDATE_ONLY") == "1" else 7
        if repetitions:
            model.restore()
            for _ in range(3):
                batch.execute(requests)
            for _ in range(repetitions):
                start = time.perf_counter()
                batch.execute(requests)
                samples.append((time.perf_counter() - start) * 1000)
        report["timings"][name] = dict(wall_ms=samples, median_ms=statistics.median(samples) if samples else None)
    report["native"] = prepared_group_stats()
    report["audit"] = runner.audit
    (evidence / f"rank{rank}.json").write_text(json.dumps(report, indent=2))
    model.close()
    shutdown_prepared_group_plans()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    try:
        with set_current_vllm_config(
                VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2))):
            main()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
