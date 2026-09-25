# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Request metadata and Engram preparation through native TP2 consumers.

No full model is loaded. The endpoint includes real checkpoint FP8 Engram
projections, residual updates and mHC/norm on both Engram layers. Two B16
transactions represent C32 preparation; this is not a complete PP2 token.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

rank = int(os.environ["LOCAL_RANK"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
if "PT_HPU_RECIPE_CACHE_CONFIG" in os.environ:
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ["PT_HPU_RECIPE_CACHE_CONFIG"].format(rank=rank)
main_cpu = int(os.environ["VLLM_HPU_DSV4_WORKER_CPUS"].split(",")[rank])
helpers = os.environ["VLLM_HPU_DSV4_WORKER_HELPER_CPUS"].split(";")[rank]
os.sched_setaffinity(0, [main_cpu, *(int(cpu) for cpu in helpers.split(",") if cpu)])

import numpy as np
import torch
import habana_frameworks.torch.core  # noqa: F401

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel, get_tp_group,
                              init_distributed_environment, initialize_model_parallel)
from vllm_gaudi import envs
from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
from vllm_gaudi.models.deepseek_v41_program import _compile_group, _weight_tree, linear, load_weight_tree
from vllm_gaudi.ops.deepseek_v41_batch_input import fill_request_metadata
from vllm_gaudi.ops.deepseek_v41_engram_fp8 import EngramFP8Sidecar
from vllm_gaudi.ops.deepseek_v41_host import EngramHost
from vllm_gaudi.ops.deepseek_v41_math import engram_update, hc_pre, rms_norm, unpack_swa
from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
from vllm_gaudi.ops.tp2_prepared_plan import (collect_prepared_group_replays, invalidate_prepared_group_plans,
                                              prepared_group_stats, record_native_decoder_outputs,
                                              replay_native_decoder, shutdown_prepared_group_plans)


class Consumer(torch.nn.Module):

    def __init__(self, weights, config, gather):
        super().__init__()
        self.weights, self.config, self.gather = weights, config, gather

    def forward(self, residual, pre, positions, ids, first, second, slots):
        active = (positions >= 0) & (slots >= 0) & (ids != 129264) & (ids != 129265)
        outputs = []
        cfg = self.config
        for layer, packed in zip((1, 14), (first, second), strict=True):
            w = self.weights.layers.get_submodule(str(layer))
            rows = self.gather(unpack_swa(packed, 256), dim=1)
            kv = linear(rows.flatten(1), w.engram.wkv)
            updated = engram_update(residual, kv, w.engram.q_weight, w.engram.k_weight, active, cfg["rms_norm_eps"])
            value, _, _, _ = hc_pre(updated,
                                    pre,
                                    w.hc_attn_fn,
                                    w.hc_attn_scale,
                                    w.hc_attn_base,
                                    cfg["rms_norm_eps"],
                                    cfg["hc_eps"],
                                    cfg["hc_sinkhorn_iters"],
                                    request_batch=True)
            outputs.append(rms_norm(value, w.attn_norm.weight, cfg["rms_norm_eps"], request_batch=True))
        return tuple(outputs)


class Replay(torch.nn.Module):

    def __init__(self, body):
        super().__init__()
        self.compiled = _compile_group(body, native=False, backend=make_backend())
        self.adapter = DecoderTopology("deepseek_v41_batch_gap_engram", (1, ), 0, False, 2)
        self.metadata = SimpleNamespace(native_completion=None)

    def forward(self, residual, pre, positions, ids, rows, slots):
        roots = dict(hidden_states=residual,
                     pre_mix=pre,
                     positions=positions,
                     input_ids=ids,
                     attention_inputs=(*rows, slots),
                     metadata=self.metadata,
                     state_generation=(1, ),
                     state_tensors=())
        result = replay_native_decoder(self, **roots)
        if result is not None:
            return result
        with collect_prepared_group_replays(owner=self,
                                            adapter=self.adapter,
                                            snapshot=lambda: SimpleNamespace(restore=lambda: None),
                                            **roots) as context:
            context["group_index"] = 0
            result = self.compiled(residual, pre, positions, ids, *rows, slots)
            record_native_decoder_outputs(*result)
        return result

    def close(self):
        invalidate_prepared_group_plans(owner=self, reason="batch_gap_component_end")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--checkpoint-audit", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--request-state-inputs",
                        action="store_true",
                        help="Include production RequestState token/span preparation in the timed chain")
    parser.add_argument("--paired-token-inputs",
                        action="store_true",
                        help="Alternate legacy and point token reads in one resident C32 component process")
    args = parser.parse_args()
    if args.paired_token_inputs:
        args.request_state_inputs = True
    evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.set_num_threads(1)
    torch.manual_seed(23451)
    torch.hpu.set_device(rank)
    context = set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2)))
    context.__enter__()
    init_distributed_environment(world_size=2,
                                 rank=rank,
                                 distributed_init_method="env://",
                                 local_rank=rank,
                                 backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=2)
    initialize_tp2_fused_ar_norm_runtime()
    from vllm_gaudi.ops.deepseek_v41_residency import EngramResidency, table_regions
    residency = EngramResidency(table_regions(args.prepared)) if rank == 0 else None
    if residency is not None:
        residency.start()
        (evidence / "engram-residency.json").write_text(json.dumps(residency.reports, indent=2))
    torch.distributed.barrier(group=get_tp_group().cpu_group)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    _, gather = stage_collectives(rank, True)
    shard = PreparedV41Shard(args.prepared, 0, rank)
    specs = {
        k: v
        for k, v in shard.specs.items() if any(
            k.startswith(f"layers.{i}.{part}") for i in (1, 14) for part in ("engram.", "hc_attn_", "attn_norm."))
    }
    weights = _weight_tree(specs)
    sidecar = EngramFP8Sidecar(envs.VLLM_HPU_DSV41_ENGRAM_FP8_SIDECAR, shard)
    load_weight_tree(shard, weights, "hpu", specs, engram_sidecar=sidecar)
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    host = EngramHost(args.prepared,
                      rank,
                      "hpu",
                      max_tokens=8192,
                      checkpoint_audit=args.checkpoint_audit,
                      force_lock=False)
    report = dict(scope=__doc__,
                  rank=rank,
                  cpus=sorted(os.sched_getaffinity(0)),
                  weights=sidecar.fingerprint,
                  context_capacity=1048576,
                  maximum_request_slots=32,
                  max_staging_tokens=8192,
                  request_state_inputs=args.request_state_inputs,
                  paired_token_inputs=args.paired_token_inputs,
                  cases=[],
                  reference=str(args.reference) if args.reference else None)
    output = evidence / f"rank{rank}-result.json"
    reference = torch.load(args.reference / f"rank{rank}-outputs.pt", weights_only=True) if args.reference else None
    saved = {}
    pages = torch.arange(64 * 8192, dtype=torch.int32, device="hpu").reshape(64, 8192)
    try:
        # B1 is a state/consumer smoke only; timed B2 and B16x2 cover the changed preparation.
        # Retain preceding shapes' input initialization so the saved C32
        # residual/FP8-consumer reference has the identical RNG sequence.
        for batch, width in ((1, 1), (2, 2), (32, 16)):
            lanes = batch // width
            replays = [Replay(Consumer(weights, config, gather)) for _ in range(lanes)]
            frames = []
            for lane in range(lanes):
                pinned = torch.zeros(3, width, dtype=torch.int32).pin_memory("hpu")
                metadata = torch.zeros_like(pinned, device="hpu")
                selected_pages = torch.empty(width, 8192, dtype=torch.int32, device="hpu")
                residual = torch.randn(width, 4, 5120).bfloat16().to("hpu")
                pre = torch.ones(width, 4, device="hpu")
                frames.append((pinned, metadata, selected_pages, residual, pre))
            names = [f"b{batch}-r{i}" for i in range(batch)]
            from vllm_gaudi.ops.deepseek_v41_engram import EngramTokenHistory
            for i, name in enumerate(names):
                history = EngramTokenHistory(host.layout, host.history.token_map)
                history.reset(name)
                history.restore_prefix(name, [(j * 17 + i * 23) % 128000 for j in range(2048)])
                host.histories[name] = history
            samples = []
            before_faults = host.audit["major_faults"]
            measured_steps = args.steps * (2 if args.paired_token_inputs else 1) if batch > 1 else 0
            if args.paired_token_inputs and batch != 32:
                measured_steps = 0
            reference_keys = []
            begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
            for step in range(4 + measured_steps):
                # Alternate AB/BA within successive pairs. Both arms retain
                # all real gathers, copies, native consumers and completion.
                point_inputs = not args.paired_token_inputs or step % 2 == (step // 2) % 2
                # Permute ownership and alter values on every step. Every request advances once.
                order = np.roll(np.arange(batch), step % batch).tolist()
                requests = [
                    SimpleNamespace(tokens=[(step * 733 + i * 173 + 199) % 128000],
                                    num_computed_tokens=0,
                                    absolute_position=2048 + step,
                                    req_id=names[i]) for i in order
                ]
                owners = [SimpleNamespace(index=i) for i in order]
                spans = [(request.req_id, request.absolute_position, request.tokens, [False]) for request in requests]
                # Make metadata preparation use exactly the production request interface.
                for request in requests:
                    request.tokens = [0] * request.absolute_position + request.tokens
                    request.num_computed_tokens = request.absolute_position
                if args.request_state_inputs:
                    from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState
                    requests = [
                        RequestState(request.req_id, request.tokens[:2048], [], None, ([1], ),
                                     request.num_computed_tokens, request.tokens[2048:]) for request in requests
                    ]
                phase = dict(metadata_ms=0.0, prepare_ms=0.0, wait_ms=0.0, submit_ms=0.0, commit_ms=0.0)
                torch.hpu.synchronize()
                started = time.perf_counter_ns()
                begin.record()
                if args.request_state_inputs:
                    mark = time.perf_counter_ns()
                    # These are the batch runner's committed-input checks and
                    # span producer. The legacy source lacks point access and
                    # copies the prefix here and again during metadata fill.
                    spans = []
                    for request in requests:
                        start = request.num_computed_tokens
                        fast = point_inputs and hasattr(request, "token_at")
                        count = request.token_count if fast else len(request.tokens)
                        if start >= count or start >= 1048576 or len(request.block_ids) != 1:
                            raise RuntimeError("Request decode lacks its committed input or scheduler pages")
                        token = request.token_at(start) if fast else request.tokens[start]
                        spans.append((request.req_id, start, [token], [token in (129264, 129265)]))
                    phase["metadata_ms"] += (time.perf_counter_ns() - mark) / 1e6
                snapshots = []
                for lane, (frame, replay) in enumerate(zip(frames, replays, strict=True)):
                    pinned, metadata, selected_pages, residual, pre = frame
                    a, b = lane * width, (lane + 1) * width
                    mark = time.perf_counter_ns()
                    if args.request_state_inputs and point_inputs and hasattr(requests[0], "token_at"):
                        fill_request_metadata(pinned,
                                              requests[a:b],
                                              owners[a:b],
                                              input_ids=(span[2][0] for span in spans[a:b]))
                    else:
                        fill_request_metadata(pinned, requests[a:b], owners[a:b])
                    metadata.copy_(pinned, non_blocking=True)
                    ids, positions, slots = metadata.unbind(0)
                    torch.index_select(pages, 0, slots.clamp_min(0).long(), out=selected_pages)
                    phase["metadata_ms"] += (time.perf_counter_ns() - mark) / 1e6
                    mark = time.perf_counter_ns()
                    ticket = host.prepare_batch(spans[a:b], capacity=width, defer_wait=True)
                    phase["prepare_ms"] += (time.perf_counter_ns() - mark) / 1e6
                    mark = time.perf_counter_ns()
                    rows = host.wait(ticket)
                    phase["wait_ms"] += (time.perf_counter_ns() - mark) / 1e6
                    mark = time.perf_counter_ns()
                    values = replay(residual, pre, positions, ids, rows, slots)
                    phase["submit_ms"] += (time.perf_counter_ns() - mark) / 1e6
                    if step >= 3:
                        from vllm_gaudi.ops.tp2_prepared_plan import _native_entries
                        if replay not in _native_entries:
                            raise RuntimeError("Engram component did not capture a native joined plan")
                    mark = time.perf_counter_ns()
                    host.complete_batch(ticket, [1] * width)
                    phase["commit_ms"] += (time.perf_counter_ns() - mark) / 1e6
                    snapshots.append((values, ticket.batch.hash_ids, metadata, selected_pages))
                end.record()
                mark = time.perf_counter_ns()
                end.synchronize()
                phase["drain_ms"] = (time.perf_counter_ns() - mark) / 1e6
                phase["wall_ms"] = (time.perf_counter_ns() - started) / 1e6
                phase["device_ms"] = begin.elapsed_time(end)
                if args.paired_token_inputs:
                    phase["point_inputs"] = point_inputs
                    phase["pair"] = (step - 4) // 2
                if step >= 4:
                    samples.append(phase)
                if step in (3, 24, 4 + measured_steps - 1):
                    for lane, (values, hashes, metadata, selected_pages) in enumerate(snapshots):
                        key = f"b{batch}-step{step}-lane{lane}"
                        tensors = [
                            *(v.cpu().clone() for v in values),
                            torch.from_numpy(hashes.copy()),
                            metadata.cpu(),
                            selected_pages.cpu()
                        ]
                        if reference is not None and (not args.paired_token_inputs or key in reference):
                            for index, (actual, expected) in enumerate(zip(tensors, reference[key], strict=True)):
                                if not torch.equal(actual, expected):
                                    raise AssertionError(f"Changed downstream output {key} item {index}")
                            reference_keys.append(key)
                        saved[key] = tensors
                assert all(host.histories[name].position == 2049 + step for name in names)
            native_stats = prepared_group_stats()
            for replay in replays:
                replay.close()
            case = dict(batch=batch,
                        lane_width=width,
                        samples=samples,
                        exact_saved_reference=reference is not None,
                        exact_reference_keys=reference_keys,
                        native_stats=native_stats,
                        median={
                            name: statistics.median(row[name] for row in samples)
                            for name in samples[0] if name not in ("point_inputs", "pair")
                        } if samples else {},
                        major_faults=host.audit["major_faults"] - before_faults)
            report["cases"].append(case)
            report["native_stats"] = prepared_group_stats()
            output.write_text(json.dumps(report, indent=2) + "\n")
            torch.save(saved, evidence / f"rank{rank}-outputs.pt")
            print(json.dumps({"rank": rank, "batch": batch, "median": case["median"]}), flush=True)
        report["audit"] = host.audit
        report["completed"] = True
        output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        host.close()
        shutdown_prepared_group_plans()
        torch.distributed.barrier(group=get_tp_group().cpu_group)
        if residency is not None:
            residency.close()
        destroy_model_parallel()
        destroy_distributed_environment()
        context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
