# SPDX-License-Identifier: Apache-2.0
"""Diagnose the real TP2 DSpark path without loading the target or Engram."""

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
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402
from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime  # noqa: E402
from vllm_gaudi.models.deepseek_v41_program import (  # noqa: E402
    PreparedDecoderLayer, PreparedDraft, _weight_tree, load_weight_tree)
from vllm_gaudi.ops.deepseek_v41_attention import CSA2SharedState  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_replay import _Snapshot, stage_collectives, stage_state_tensors  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard  # noqa: E402
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut  # noqa: E402


class TargetGroup(torch.nn.Module):
    """Real layers 20-23, retaining their CSA2 state and TP reductions."""

    def __init__(self, stage, aux_outputs=False):
        super().__init__()
        self.aux_outputs = aux_outputs
        self.shared = stage.shared
        config = stage.config["text_config"]
        lookup = mxfp4_bf16_lut(torch.device("hpu"))
        self.layers = torch.nn.ModuleList([
            PreparedDecoderLayer(stage.weights.layers.get_submodule(str(index)), config, index, stage.shared,
                stage.shard.manifest["normal_scales"][f"layers.{index}.ffn.experts"][stage.tp_rank],
                lookup, stage.reduce, stage.all_gather, "hpu") for index in range(20, 24)])

    def forward(self, residual, pre, positions):
        mask = torch.zeros(positions.shape, dtype=torch.bool, device=positions.device)
        aux = []
        for layer in self.layers:
            if self.aux_outputs and len(aux) < 3:
                aux.append(residual.mean(1))
            residual, pre, _ = layer(residual, pre, positions, mask)
        hidden = (residual.float() * pre.unsqueeze(-1)).sum(1).to(torch.bfloat16)
        return (hidden, pre, torch.cat(aux, -1)) if self.aux_outputs else hidden

    def native_forward(self, residual, pre, positions):
        return self(residual, pre, positions)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--prepared-prefix", action="store_true")
    parser.add_argument("--real-target", action="store_true")
    args = parser.parse_args()
    if args.real_target:
        args.prepared_prefix = True
    torch.hpu.set_device(rank)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    bind_worker_cpu(rank)
    init_distributed_environment(world_size=2, rank=rank, distributed_init_method="env://",
                                 local_rank=rank, backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    initialize_tp2_fused_ar_norm_runtime()
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    directory = args.prepared
    config = json.loads((directory / "config.json").read_text())
    shard = PreparedV41Shard(directory, 1, rank)
    target_names = tuple(f"layers.{index}." for index in range(20, 24))
    specs = {name: spec for name, spec in shard.specs.items() if name.startswith("mtp.") or name == "head.weight"
             or (args.real_target and name.startswith(target_names))}
    tree = _weight_tree(specs)
    load_weight_tree(shard, tree, "hpu", specs)
    reduce, gather = stage_collectives(rank, True)
    stage = SimpleNamespace(weights=tree, config=config, tp_rank=rank, shard=shard,
                            shared=CSA2SharedState(config["text_config"], 20, 24, "hpu"),
                            reduce=reduce, all_gather=gather)
    draft = PreparedDraft(stage, mxfp4_bf16_lut(torch.device("hpu")), "hpu")
    insert = torch.compile(draft.insert_context, backend="hpu_backend", fullgraph=True, dynamic=False)
    forward = torch.compile(draft, backend="hpu_backend", fullgraph=True, dynamic=False)
    sample = torch.compile(draft.sample_greedy, backend="hpu_backend", fullgraph=True, dynamic=False)
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
    from vllm_gaudi.ops.tp2_prepared_plan import (
        collect_prepared_group_replays, record_native_decoder_outputs, replay_native_decoder, prepared_group_stats)
    def prefix(value):
        value = value + torch.ops.vllm_gaudi.tp2_exchange_peer(value)
        value = value * 0.25
        return (value + torch.ops.vllm_gaudi.tp2_exchange_peer(value)) * 0.5
    compiled_prefix = torch.compile(prefix, backend="hpu_backend", fullgraph=True, dynamic=False)
    owner = torch.nn.Identity()
    topology = DecoderTopology("deepseek_v41_pp1", (1,), 2, False)
    if args.real_target:
        owner = TargetGroup(stage)
        ordinary_prefix = torch.compile(owner, backend="hpu_backend", fullgraph=True, dynamic=False)
        compiled_prefix = torch.compile(owner.native_forward, backend="hpu_backend", fullgraph=True, dynamic=False)
        topology = DecoderTopology("deepseek_v41_pp1", (4,), 2, False)
    outputs = []
    for step, count in enumerate((6, 1, 1, 1, 1, 6)):
        if step == 1:
            for layer in draft.layers:
                value = layer.attention.swa
                pool = torch.zeros((2, *value.shape), device=value.device, dtype=value.dtype)
                layer.attention.swa = pool[1]
        torch.manual_seed(41 + step)
        values = torch.randn(count, 15360, device="cpu", dtype=torch.bfloat16).to("hpu")
        positions = torch.arange(count, device="hpu", dtype=torch.int32)
        if args.real_target:
            residual = values[:, :5120].unsqueeze(1).expand(-1, 4, -1).contiguous()
            pre = torch.tensor([[1., 0., 0., 0.]] * count, dtype=torch.float32, device="hpu")
            if count != 1:
                values = torch.cat([ordinary_prefix(residual, pre, positions)] * 3, -1)
        if args.prepared_prefix and count == 1:
            source = residual if args.real_target else values[:, :5120].contiguous()
            states = stage_state_tensors(owner) if args.real_target else ()
            roots = dict(hidden_states=source, positions=positions, residual=None,
                         pre_mix=pre if args.real_target else None, state_tensors=states, state_generation=0,
                         metadata=SimpleNamespace(is_prompt=False, native_completion=None))
            result = replay_native_decoder(owner, **roots)
            if result is None:
                with collect_prepared_group_replays(owner=owner, adapter=topology,
                        snapshot=lambda: _Snapshot(states), **roots):
                    output = compiled_prefix(source, pre, positions) if args.real_target else compiled_prefix(source)
                    record_native_decoder_outputs(output)
                result = output, None
            values = torch.cat([result[0]] * 3, -1)
            print(f"RANK {rank} STEP {step} prefix " + json.dumps(prepared_group_stats()), flush=True)
        print(f"RANK {rank} STEP {step} C{count} context", flush=True)
        insert(values, positions)
        first = torch.tensor([step + 1], device="hpu", dtype=torch.int64)
        draft_positions = torch.arange(count, count + 5, device="hpu", dtype=torch.int32)
        hidden, logits = forward(first, draft_positions)
        print(f"RANK {rank} STEP {step} sampling", flush=True)
        tokens, confidence = sample(first, hidden, logits)
        tokens, confidence = tokens.cpu(), confidence.cpu()
        assert tokens.shape == (5,) and (tokens >= 0).all() and (tokens < 129280).all()
        assert torch.isfinite(confidence).all()
        outputs.append({"context_count": count, "tokens": tokens.tolist()})
        print(f"RANK {rank} STEP {step} complete", flush=True)
    evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
    (evidence / f"draft-rank{rank}.json").write_text(json.dumps({
        "tp": 2, "pp": 1, "purpose": "DSpark synchronization diagnostic; real target and PP absent",
        "prepared_prefix": args.prepared_prefix, "real_target_layers": [20, 21, 22, 23] if args.real_target else [],
        "native": prepared_group_stats(),
        "outputs": outputs, "peak_device_bytes": torch.hpu.max_memory_allocated()}, indent=2) + "\n")
    from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans
    from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
    shutdown_prepared_group_plans()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2))):
        main()
