# SPDX-License-Identifier: Apache-2.0
"""Compare complete PP0 input-to-stage chains using maintained source paths."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

rank = int(os.environ["LOCAL_RANK"])
evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").replace("{rank}", str(rank))
working = evidence / f"rank{rank}-working"
working.mkdir()
os.chdir(working)
graphs = evidence / "graphs" / f"rank{rank}"
graphs.mkdir(parents=True)
os.environ["GRAPH_VISUALIZATION_DIR"] = str(graphs)
os.environ["PT_HPU_GRAPH_DUMP_PREFIX"] = str(graphs)

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment  # noqa: E402
prepare_environment()

import torch  # noqa: E402
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment)
from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime  # noqa: E402
from vllm_gaudi.models.deepseek_v41_program import PreparedStage, PreparedInput  # noqa: E402
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_host import EngramHost  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_inputs import PositionBank  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_replay import StageReplay, stage_collectives, stage_state_tensors  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_state import StageStateBlocks  # noqa: E402
from vllm_gaudi.ops.tp2_prepared_plan import _native_entries, prepared_group_stats  # noqa: E402


def summary(values):
    ordered = sorted(values)
    return dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                p90_ms=ordered[int(len(ordered) * .9)], max_ms=max(values))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--checkpoint-audit", required=True, type=Path)
    parser.add_argument("--reference-request", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--component-reference", type=Path)
    args = parser.parse_args()
    if not 32 <= args.steps <= 192:
        parser.error("Use 32..192 changing inputs within the production context bucket")
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    init_distributed_environment(world_size=2, rank=rank, distributed_init_method="env://",
                                 local_rank=rank, backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    initialize_tp2_fused_ar_norm_runtime()
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    reduce, gather = stage_collectives(rank, True)
    stage = PreparedStage(args.prepared, 0, rank, reduce, gather, "hpu", dspark=False)
    stage.replay_owner = None
    stage.load_prepared("hpu")
    state = StageStateBlocks(stage)
    state.allocate(2, "hpu")
    state.bind(1)
    host = EngramHost(args.prepared, rank, "hpu", checkpoint_audit=args.checkpoint_audit)
    external_input = torch.compile(PreparedInput(stage.weights.embed, rank, reduce),
                                   backend="hpu_backend", fullgraph=True, dynamic=False)
    first = json.loads((args.reference_request / "stream.jsonl").read_text().splitlines()[0])
    prompt = json.loads(first["data"])["choices"][0]["prompt_token_ids"]
    reference_ids = json.loads((args.reference_request / "token_ids.json").read_text())
    assert len(prompt) == 1
    tokens = (prompt + reference_ids)[:args.steps]
    # Keep a fixed ID offset: production uses offset zero initially and a
    # fixed commit view afterward. A growing offset bank compiles new recipes.
    ids = tuple(torch.tensor([token], dtype=torch.int32, device="hpu") for token in tokens)
    position_bank = PositionBank(512, 1, "hpu")
    positions = tuple(position_bank.view(index, 1) for index in range(args.steps))
    assert all(value.storage_offset() == 0 for value in ids)
    expected_outputs, expected_states, results = [], [], {}
    if args.component_reference:
        saved = torch.load(args.component_reference / f"rank{rank}-reference.pt", weights_only=True)
        if saved["tokens"] != tokens or saved["protocol"] != "zero-offset-ids-v2":
            raise RuntimeError("Saved component inputs or measurement protocol differ")
        expected_outputs, expected_states = saved["outputs"], saved["states"]
        results["external_input_reference"] = json.loads(
            (args.component_reference / f"rank{rank}-partial.json").read_text())["external_input_reference"]
    device_start, device_end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
    try:
        arms = ("native_input_candidate", ) if args.component_reference else (
            "external_input_reference", "native_input_candidate")
        for arm in arms:
            native_input = arm == "native_input_candidate"
            stage.replay_owner = StageReplay(stage)
            owner = stage.replay_owner

            def invoke(index, request_id, *, native_input=native_input, owner=owner):
                ticket = host.prepare(request_id, [tokens[index]])
                if native_input:
                    output = owner.from_input_ids(positions[index], ids[index], ticket.buffers)
                else:
                    residual, pre = external_input(ids[index])
                    output = owner(residual, pre, positions[index], ids[index], ticket.buffers)
                host.complete(ticket, 1)
                return output

            state.clear()
            host.reset(arm + "-warm")
            for index in range(32):
                invoke(index, arm + "-warm")
                torch.hpu.synchronize()
            bind_worker_helpers(rank)
            state.clear()
            host.reset(arm)
            torch.hpu.synchronize()
            warm_graphs = set(graphs.glob("*-PreGraph-symbol.pbtxt"))
            if not warm_graphs:
                raise RuntimeError("Graph dumps are required to check warm measurement coverage")
            warm_native = prepared_group_stats()
            wall, device_times = [], []
            for index in range(args.steps):
                started = time.perf_counter_ns()
                device_start.record()
                output = invoke(index, arm)
                device_end.record()
                device_end.synchronize()
                wall.append((time.perf_counter_ns() - started) / 1e6)
                device_times.append(device_start.elapsed_time(device_end))
                actual = tuple(value.cpu().clone() for value in output[:2])
                if native_input:
                    if not all(torch.equal(a, b) for a, b in zip(actual, expected_outputs[index], strict=True)):
                        torch.save(dict(expected=expected_outputs[index], actual=actual, index=index),
                                   evidence / f"rank{rank}-mismatch.pt")
                        raise RuntimeError(f"Native input changed PP0 output at step {index}")
                else:
                    expected_outputs.append(actual)
            actual_states = tuple(value.cpu().clone() for value in stage_state_tensors(stage))
            if native_input:
                if not all(torch.equal(a, b) for a, b in zip(actual_states, expected_states, strict=True)):
                    raise RuntimeError("Native input changed the final PP0 state pool")
            else:
                expected_states = actual_states
            new_graphs = set(graphs.glob("*-PreGraph-symbol.pbtxt")) - warm_graphs
            if new_graphs:
                raise RuntimeError(f"Measurement compiled {len(new_graphs)} new graphs after warmup")
            variant = owner.variants[(1, "input") if native_input else 1]
            graph, bindings, _, _ = _native_entries[variant]
            binding_names = [(binding.source, binding.name) for binding in bindings.bindings]
            expected_bindings = 4 if native_input else 6
            if len(binding_names) != expected_bindings or graph.collective_count() != (43 if native_input else 42):
                raise RuntimeError(f"Unexpected input/native topology: {binding_names}, {graph.collective_count()}")
            results[arm] = dict(wall=summary(wall), device_event=summary(device_times),
                                wall_ms=wall, device_event_ms=device_times, bindings=binding_names,
                                native=prepared_group_stats(), input_copies=bindings.input_copies,
                                state_tensors=len(actual_states), warm_native=warm_native,
                                measurement_new_graphs=len(new_graphs), token_storage_offset=0)
            (evidence / f"rank{rank}-partial.json").write_text(json.dumps(results, indent=2) + "\n")
            if not native_input:
                torch.save(dict(tokens=tokens, outputs=expected_outputs, states=expected_states,
                                protocol="zero-offset-ids-v2"), evidence / f"rank{rank}-reference.pt")
            print(f"rank{rank} {arm}: {results[arm]['wall']}", flush=True)
            owner.close()
        reference, candidate = results.values()
        result = dict(arms=results, steps=args.steps, outputs_exact=True, final_states_exact=True,
                      complete_chain=("changing token/position + CPU Engram gather/H2D + embedding TP + "
                                      "20-layer PP0 + ownership completion"),
                      timing_boundary=("host drained wall and current-stream HPU event elapsed; "
                                       "output/state CPU comparisons outside timer"),
                      reference_reason=("No archived comparable full PP0 ingress component; "
                                        "measure missing component reference once, no full-model B"),
                      component_reference=str(args.component_reference) if args.component_reference else None,
                      median_gain_ms=reference["wall"]["median_ms"] - candidate["wall"]["median_ms"],
                      p90_gain_ms=reference["wall"]["p90_ms"] - candidate["wall"]["p90_ms"])
        (evidence / f"rank{rank}-result.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if stage.replay_owner is not None:
            stage.replay_owner.close()
        host.close()
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2))):
        main()
