# SPDX-License-Identifier: Apache-2.0
"""Check mHC/TP scheduling against ordinary same-math execution on four real layers."""

import argparse
import faulthandler
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace

rank = int(os.environ["LOCAL_RANK"])
evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").replace(
    "{rank}", str(rank))

from check_deepseek_v41_draft_tp2 import TargetGroup  # noqa: E402
import torch  # noqa: E402
from vllm.distributed import (  # noqa: E402
    init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment)
from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime  # noqa: E402
from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_attention import CSA2SharedState  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_replay import _Snapshot, stage_collectives, stage_state_tensors  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_state import StageStateBlocks  # noqa: E402
from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology  # noqa: E402
from vllm_gaudi.ops.tp2_prepared_plan import (  # noqa: E402
    collect_prepared_group_replays, record_native_decoder_outputs, replay_native_decoder,
    prepared_group_stats, shutdown_prepared_group_plans)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--sync-native", action="store_true")
    args = parser.parse_args()
    torch.hpu.set_device(rank)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    bind_worker_cpu(rank)
    init_distributed_environment(world_size=2, rank=rank, distributed_init_method="env://",
                                 local_rank=rank, backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    initialize_tp2_fused_ar_norm_runtime()
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    stack_file = (evidence / f"rank{rank}-stacks.log").open("w")
    faulthandler.enable(file=stack_file, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=stack_file, all_threads=True, chain=False)
    config = json.loads((args.prepared / "config.json").read_text())
    shard = PreparedV41Shard(args.prepared, 1, rank)
    layers = tuple(f"layers.{index}." for index in range(20, 24))
    specs = {name: spec for name, spec in shard.specs.items() if name.startswith(layers) or name == "head.weight"}
    tree = _weight_tree(specs)
    load_weight_tree(shard, tree, "hpu", specs)
    reduce, gather = stage_collectives(rank, True)
    stage = SimpleNamespace(weights=tree, config=config, tp_rank=rank, shard=shard,
                            shared=CSA2SharedState(config["text_config"], 20, 24, "hpu"),
                            reduce=reduce, all_gather=gather)
    owner = TargetGroup(stage)
    for layer in owner.layers:
        layer.attention.prepare_output_weight()
    owner.pp_rank, owner.generation, owner.replay_owner = 1, 0, None
    state = StageStateBlocks(owner)
    ordinary = torch.compile(owner, backend="hpu_backend", fullgraph=True, dynamic=False)
    from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
    native = torch.compile(owner.native_forward, backend=make_backend(), fullgraph=True, dynamic=False)
    topology = DecoderTopology("deepseek_v41_pp1", (4,), 2, False)
    outputs = []
    sequence = (6, 1, 1, 1, 1, 6)
    for step, count in enumerate(sequence):
        if step == 1:
            state.allocate(2, "hpu")
            state.bind(1)
        if args.reset_state and step >= 1:
            state.clear()
        torch.manual_seed(41 + step)
        residual = torch.randn(count, 4, 5120, device="cpu", dtype=torch.bfloat16).to("hpu")
        pre = torch.tensor([[1., 0., 0., 0.]] * count, dtype=torch.float32, device="hpu")
        positions = torch.arange(count, device="hpu", dtype=torch.int32)
        print(f"RANK {rank} STEP {step} C{count} target start", flush=True)
        reference = None
        expected_states = None
        if count == 1:
            initial = _Snapshot(stage_state_tensors(owner))
            reference = ordinary(residual, pre, positions).clone()
            expected_states = tuple(value.clone() for value in initial.tensors)
            initial.restore()
        if count == 1:
            states = stage_state_tensors(owner)
            roots = dict(hidden_states=residual, pre_mix=pre, positions=positions, residual=None,
                         state_tensors=states, state_generation=owner.generation,
                         metadata=SimpleNamespace(is_prompt=False, native_completion=None))
            result = replay_native_decoder(owner, **roots)
            if result is None:
                with collect_prepared_group_replays(owner=owner, adapter=topology,
                                                    snapshot=lambda states=states: _Snapshot(states), **roots):
                    value = native(residual, pre, positions)
                    record_native_decoder_outputs(value)
                result = value, None
            value = result[0]
        else:
            value = ordinary(residual, pre, positions)
        if reference is not None:
            exact = torch.equal(reference.cpu().view(torch.int16), value.cpu().view(torch.int16))
            states_exact = all(torch.equal(actual.cpu(), expected.cpu()) for actual, expected in
                               zip(stage_state_tensors(owner), expected_states, strict=True))
            print(f"RANK {rank} STEP {step} EXACT output={exact} states={states_exact}", flush=True)
            if not exact or not states_exact:
                raise RuntimeError("mHC/TP candidate differs from the complete ordinary chain")
        print(f"RANK {rank} STEP {step} target submitted " + json.dumps(prepared_group_stats()), flush=True)
        if args.sync_native:
            torch.hpu.synchronize()
            print(f"RANK {rank} STEP {step} target device complete", flush=True)
        logits = gather(torch.nn.functional.linear(value.float(), tree.head.weight), dim=-1)
        print(f"RANK {rank} STEP {step} logits submitted", flush=True)
        sampled = logits.argmax(-1).cpu().tolist()
        outputs.append({"step": step, "count": count, "sampled": sampled})
        print(f"RANK {rank} STEP {step} host complete", flush=True)
        if step == 2:
            bind_worker_helpers(rank)
    (evidence / f"rank{rank}-result.json").write_text(json.dumps({
        "purpose": "Real four-layer C1 state-reset/native/TP-head diagnostic; no PP or full-model speed claim",
        "profiled_c1_calls": 0, "reset_state": args.reset_state,
        "sync_native": args.sync_native, "outputs": outputs,
        "native": prepared_group_stats(), "peak_device_bytes": torch.hpu.max_memory_allocated()}, indent=2) + "\n")
    shutdown_prepared_group_plans()
    destroy_model_parallel()
    destroy_distributed_environment()
    faulthandler.unregister(signal.SIGUSR1)
    faulthandler.disable()
    stack_file.close()


if __name__ == "__main__":
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2))):
        main()
