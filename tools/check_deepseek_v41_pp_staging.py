# SPDX-License-Identifier: Apache-2.0
"""Four-device diagnostic for PP receive to fixed native inputs to DSpark."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

rank = int(os.environ["LOCAL_RANK"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
if os.environ.get("GRAPH_VISUALIZATION") == "1":
    os.environ["GRAPH_VISUALIZATION_DIR"] += f"/rank{rank}"
    Path(os.environ["GRAPH_VISUALIZATION_DIR"]).mkdir(parents=True, exist_ok=True)

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment  # noqa: E402
prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment)
from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime  # noqa: E402
from vllm_gaudi.models.deepseek_v41_program import PreparedDraft, _weight_tree, load_weight_tree  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_attention import CSA2SharedState  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_replay import _Snapshot, stage_collectives, stage_state_tensors  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut  # noqa: E402
from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology  # noqa: E402
from vllm_gaudi.ops.tp2_prepared_plan import (  # noqa: E402
    collect_prepared_group_replays, record_native_decoder_outputs, replay_native_decoder,
    prepared_group_stats, recapture_native_decoder_programs, shutdown_prepared_group_plans)
from vllm_gaudi.v1.worker.deepseek_v41_runner import PPBuffers  # noqa: E402
from check_deepseek_v41_draft_tp2 import TargetGroup  # noqa: E402


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--wait-receive", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="Additional real target/draft calls profiled after the existing six warmup calls")
    parser.add_argument("--native-c6", action="store_true",
                        help="Capture both C1 and C6 native plans, matching target verification in the full runner")
    args = parser.parse_args()
    torch.hpu.set_device(rank)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    bind_worker_cpu(rank)
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method="env://",
                                 local_rank=rank, backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
    initialize_tp2_fused_ar_norm_runtime()
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    pp = PPBuffers(torch.device("hpu"))
    outputs, fixed_by_count, profiler = [], {}, None
    evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
    if pp.group.is_last_rank:
        config = json.loads((args.prepared / "config.json").read_text())
        shard = PreparedV41Shard(args.prepared, 1, rank % 2)
        prefixes = tuple(f"layers.{index}." for index in range(20, 24)) + ("mtp.",)
        specs = {name: spec for name, spec in shard.specs.items() if name.startswith(prefixes) or name == "head.weight"}
        weights = _weight_tree(specs)
        load_weight_tree(shard, weights, "hpu", specs)
        reduce, gather = stage_collectives(rank % 2, True)
        stage = SimpleNamespace(weights=weights, config=config, tp_rank=rank % 2, shard=shard,
            shared=CSA2SharedState(config["text_config"], 20, 24, "hpu"), reduce=reduce, all_gather=gather)
        target = TargetGroup(stage, aux_outputs=True)
        draft = PreparedDraft(stage, mxfp4_bf16_lut(torch.device("hpu")), "hpu")
        ordinary = torch.compile(target, backend="hpu_backend", fullgraph=True, dynamic=False)
        compiled = torch.compile(target.native_forward, backend="hpu_backend", fullgraph=True, dynamic=False)
        insert = torch.compile(draft.insert_context, backend="hpu_backend", fullgraph=True, dynamic=False)
        forward = torch.compile(draft, backend="hpu_backend", fullgraph=True, dynamic=False)
        sample = torch.compile(draft.sample_greedy, backend="hpu_backend", fullgraph=True, dynamic=False)
        topology = DecoderTopology("deepseek_v41_pp1", (4,), 2, False)
        targets = {count: torch.nn.Identity() for count in (1, 6)}
    counts = (6, 1, 1, 1, 1, 6) + tuple(6 if step % 8 == 0 else 1 for step in range(args.profile_steps))
    for step, count in enumerate(counts):
        if args.profile_steps and step == 6:
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                         torch.profiler.ProfilerActivity.HPU],
                                              record_shapes=True, with_stack=False)
            profiler.start()
            recapture_native_decoder_programs()
            print(f"RANK {rank} profiler started", flush=True)
        torch.manual_seed(41 + step)
        print(f"RANK {rank} STEP {step} C{count} receive/send", flush=True)
        positions = torch.arange(count, dtype=torch.int32, device="hpu")
        if pp.group.is_first_rank:
            values = torch.randn(count, 4, 5120, dtype=torch.bfloat16, device="cpu").to("hpu")
            pre = torch.tensor([[1., 0., 0., 0.]] * count, dtype=torch.float32, device="hpu")
            pp.exchange({"hidden_states": values, "pre_mix": pre}, count)
        else:
            received = pp.exchange(None, count)
            if args.wait_receive:
                torch.hpu.synchronize()
            residual, pre = received["hidden_states"], received["pre_mix"]
            if count != 1 and not args.native_c6:
                result = ordinary(residual, pre, positions)
            else:
                if count not in fixed_by_count:
                    fixed_by_count[count] = tuple(value.clone() for value in (residual, pre, positions))
                fixed = fixed_by_count[count]
                states = stage_state_tensors(target)
                roots = dict(hidden_states=residual, pre_mix=pre, positions=positions, residual=None,
                    state_tensors=states, state_generation=0,
                    metadata=SimpleNamespace(is_prompt=False, native_completion=None))
                # Native entry ownership includes the static token bucket.
                owner = targets[count] if args.native_c6 else target
                result = replay_native_decoder(owner, **roots)
                if result is None:
                    for destination, source in zip(fixed, (residual, pre, positions), strict=True):
                        destination.copy_(source)
                    fixed_roots = dict(roots, hidden_states=fixed[0], pre_mix=fixed[1], positions=fixed[2])
                    with collect_prepared_group_replays(owner=owner, adapter=topology,
                            snapshot=lambda states=states: _Snapshot(states), **fixed_roots):
                        result = compiled(*fixed)
                        record_native_decoder_outputs(*result)
            print(f"RANK {rank} STEP {step} target " + json.dumps(prepared_group_stats()), flush=True)
            insert(result[2], positions)
            first = torch.tensor([step + 1], dtype=torch.int64, device="hpu")
            draft_positions = torch.arange(count, count + 5, dtype=torch.int32, device="hpu")
            hidden, logits = forward(first, draft_positions)
            tokens, confidence = sample(first, hidden, logits)
            print(f"RANK {rank} STEP {step} sampling submitted", flush=True)
            tokens, confidence = tokens.cpu(), confidence.cpu()
            assert torch.isfinite(confidence).all()
            outputs.append({"context": count, "tokens": tokens.tolist()})
            print(f"RANK {rank} STEP {step} host tokens complete", flush=True)
        pp.drain()
        torch.hpu.synchronize()
        pp.group.barrier()
    if profiler is not None:
        profiler.stop()
        profiler.export_chrome_trace(str(evidence / f"rank{rank}.trace.json.gz"))
        recapture_native_decoder_programs()
    from vllm_gaudi.ops.tp2_runtime_profile import verify_loaded_profile_libraries
    (evidence / f"rank{rank}.json").write_text(json.dumps({
        "diagnostic": "four target layers on PP1, synthetic PP0; no full-model qualification",
        "profile_steps": args.profile_steps, "profile_libraries": verify_loaded_profile_libraries(),
        "native_c6": args.native_c6,
        "wait_receive": args.wait_receive, "outputs": outputs, "native": prepared_group_stats(),
        "pp_sends": pp.sends, "pp_receives": pp.receives}, indent=2) + "\n")
    print(f"RANK {rank} shutdown native plans", flush=True)
    shutdown_prepared_group_plans()
    destroy_model_parallel()
    destroy_distributed_environment()
    print(f"RANK {rank} shutdown complete", flush=True)


if __name__ == "__main__":
    with set_current_vllm_config(VllmConfig(
            parallel_config=ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2))):
        main()
