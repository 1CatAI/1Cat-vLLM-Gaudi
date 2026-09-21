# SPDX-License-Identifier: Apache-2.0
"""Check mHC/TP scheduling against ordinary same-math execution on four real layers."""

import argparse
import faulthandler
import json
import os
from pathlib import Path
import signal
import statistics
import time
from types import SimpleNamespace

rank = int(os.environ["LOCAL_RANK"])
evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").replace("{rank}", str(rank))

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
    collect_prepared_group_replays, record_native_decoder_outputs, replay_native_decoder, prepared_group_stats,
    shutdown_prepared_group_plans)


class ExpertTargetGroup(TargetGroup):
    """Exercise the same mHC/TP contract with N256 experts and BF16 boundaries."""

    def __init__(self, stage, n256):
        super().__init__(stage)
        self.n256 = n256
        for layer in self.layers:
            layer.moe.n256 = n256

    def forward(self, residual, pre, positions):
        mask = torch.zeros(positions.shape, dtype=torch.bool, device=positions.device)
        for layer in self.layers:
            residual, pre, _ = layer(residual, pre, positions, mask, fp8_decode=self.n256 and positions.numel() == 1)
        return (residual.float() * pre.unsqueeze(-1)).sum(1).to(torch.bfloat16)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--sync-native", action="store_true")
    parser.add_argument("--expert-n256", action="store_true")
    parser.add_argument("--fused-quant", action="store_true")
    parser.add_argument("--benchmark-iters", type=int, default=0)
    args = parser.parse_args()
    if args.fused_quant:
        if not args.expert_n256:
            raise ValueError("Fused quantization requires the N256 expert contract")
        os.environ["VLLM_HPU_DSV41_EXPERT_FUSED_QUANT"] = "1"
    torch.hpu.set_device(rank)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    bind_worker_cpu(rank)
    init_distributed_environment(world_size=2,
                                 rank=rank,
                                 distributed_init_method="env://",
                                 local_rank=rank,
                                 backend="hccl")
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
    load_weight_tree(shard, tree, "hpu", specs, expert_n256_layers=range(20, 24) if args.expert_n256 else ())
    reduce, gather = stage_collectives(rank, True)
    stage = SimpleNamespace(weights=tree,
                            config=config,
                            tp_rank=rank,
                            shard=shard,
                            shared=CSA2SharedState(config["text_config"], 20, 24, "hpu"),
                            reduce=reduce,
                            all_gather=gather)
    owner = ExpertTargetGroup(stage, args.expert_n256)
    for layer in owner.layers:
        layer.prepare_mhc_control_weights()
        layer.attention.prepare_output_weight()
        layer.moe.prepare_shared_gate_up_weight()
    owner.pp_rank, owner.generation, owner.replay_owner = 1, 0, None
    state = StageStateBlocks(owner)
    ordinary = torch.compile(owner, backend="hpu_backend", fullgraph=True, dynamic=False)
    from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
    native = torch.compile(owner.native_forward, backend=make_backend(), fullgraph=True, dynamic=False)
    topology = DecoderTopology("deepseek_v41_pp1", (4, ), 2, False)
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
            roots = dict(hidden_states=residual,
                         pre_mix=pre,
                         positions=positions,
                         residual=None,
                         state_tensors=states,
                         state_generation=owner.generation,
                         metadata=SimpleNamespace(is_prompt=False, native_completion=None))
            result = replay_native_decoder(owner, **roots)
            if result is None:
                with collect_prepared_group_replays(owner=owner,
                                                    adapter=topology,
                                                    snapshot=lambda states=states: _Snapshot(states),
                                                    **roots):
                    value = native(residual, pre, positions)
                    record_native_decoder_outputs(value)
                result = value, None
            value = result[0]
        else:
            value = ordinary(residual, pre, positions)
        if reference is not None:
            exact = torch.equal(reference.cpu().view(torch.int16), value.cpu().view(torch.int16))
            states_exact = all(
                torch.equal(actual.cpu(), expected.cpu())
                for actual, expected in zip(stage_state_tensors(owner), expected_states, strict=True))
            print(f"RANK {rank} STEP {step} EXACT output={exact} states={states_exact}", flush=True)
            if not exact or not states_exact:
                raise RuntimeError("mHC/TP candidate differs from the complete ordinary chain")
        print(f"RANK {rank} STEP {step} target submitted " + json.dumps(prepared_group_stats()), flush=True)
        if args.sync_native:
            torch.hpu.synchronize()
            print(f"RANK {rank} STEP {step} target device complete", flush=True)
        logits = gather(
            torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
                value.contiguous(), tree.head.weight),
            dim=-1)
        print(f"RANK {rank} STEP {step} logits submitted", flush=True)
        sampled = logits.argmax(-1).cpu().tolist()
        outputs.append({"step": step, "count": count, "sampled": sampled})
        print(f"RANK {rank} STEP {step} host complete", flush=True)
        if step == 2:
            bind_worker_helpers(rank)
    timings = None
    if args.benchmark_iters:
        residual = torch.empty(1, 4, 5120, device="hpu", dtype=torch.bfloat16)
        pre = torch.empty(1, 4, device="hpu", dtype=torch.float32)
        positions = torch.empty(1, device="hpu", dtype=torch.int32)
        fixtures = []
        for seed in range(4):
            generator = torch.Generator().manual_seed(4100 + seed)
            fixtures.append((
                torch.randn(1, 4, 5120, generator=generator, dtype=torch.bfloat16).to("hpu"),
                torch.tensor([[1.0 - seed * 0.01, seed * 0.01, 0.0, 0.0]],
                             dtype=torch.float32, device="hpu"),
                torch.tensor([32 + seed], dtype=torch.int32, device="hpu"),
            ))
        states = stage_state_tensors(owner)
        roots = dict(hidden_states=residual,
                     pre_mix=pre,
                     positions=positions,
                     residual=None,
                     state_tensors=states,
                     state_generation=owner.generation,
                     metadata=SimpleNamespace(is_prompt=False, native_completion=None))
        device_ms, host_ms, sampled_ids = [], [], []
        total = args.benchmark_iters + 4
        for iteration in range(total):
            fixture = fixtures[iteration % len(fixtures)]
            residual.copy_(fixture[0])
            pre.copy_(fixture[1])
            positions.copy_(fixture[2])
            torch.hpu.synchronize()
            begin = torch.hpu.Event(enable_timing=True)
            end = torch.hpu.Event(enable_timing=True)
            started = time.perf_counter_ns()
            begin.record()
            result = replay_native_decoder(owner, **roots)
            if result is None:
                raise RuntimeError("benchmark requires an instantiated native decoder plan")
            value = result[0]
            logits = gather(
                torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
                    value.contiguous(), tree.head.weight),
                dim=-1)
            sampled = logits.argmax(-1)
            end.record()
            host_sample = sampled.cpu().item()
            torch.hpu.synchronize()
            wall = (time.perf_counter_ns() - started) / 1e6
            if iteration >= 4:
                device_ms.append(begin.elapsed_time(end))
                host_ms.append(wall)
                sampled_ids.append(host_sample)
        timings = {
            "iterations": args.benchmark_iters,
            "boundary": "four decoder layers through TP output head, argmax and CPU token delivery",
            "changing_fixtures": len(fixtures),
            "device_event_ms": device_ms,
            "device_median_ms": statistics.median(device_ms),
            "host_drained_ms": host_ms,
            "host_median_ms": statistics.median(host_ms),
            "sampled_ids": sampled_ids,
        }
        print(f"RANK {rank} BENCHMARK " + json.dumps({
            "device_median_ms": timings["device_median_ms"],
            "host_median_ms": timings["host_median_ms"],
            "iterations": args.benchmark_iters,
        }), flush=True)
    (evidence / f"rank{rank}-result.json").write_text(
        json.dumps(
            {
                "purpose": "Real four-layer C1 state-reset/native/TP-head diagnostic; no PP or full-model speed claim",
                "profiled_c1_calls": 0,
                "reset_state": args.reset_state,
                "sync_native": args.sync_native,
                "outputs": outputs,
                "timings": timings,
                "native": prepared_group_stats(),
                "peak_device_bytes": torch.hpu.max_memory_allocated()
            },
            indent=2) + "\n")
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
