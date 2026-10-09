# SPDX-License-Identifier: Apache-2.0
"""Diagnose the shared tensor-parallel draft without loading target or Engram."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

rank = int(os.environ["LOCAL_RANK"])
tp_size = int(os.environ["WORLD_SIZE"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
if "PT_HPU_RECIPE_CACHE_CONFIG" in os.environ:
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ["PT_HPU_RECIPE_CACHE_CONFIG"].replace("{rank}", str(rank))
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
    PreparedDecoderLayer, PreparedDraft, _weight_tree, draft_context_state, load_weight_tree)
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
            PreparedDecoderLayer(stage.weights.layers.get_submodule(str(index)),
                                 config,
                                 index,
                                 stage.shared,
                                 stage.shard.manifest["normal_scales"][f"layers.{index}.ffn.experts"][stage.tp_rank],
                                 lookup,
                                 stage.reduce,
                                 stage.all_gather,
                                 "hpu",
                                 tensor_parallel_size=stage.tensor_parallel_size) for index in range(20, 24)
        ])

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
    parser.add_argument("--local-vocab", action="store_true")
    parser.add_argument("--official-sampling", action="store_true",
                        help="Exercise actual C5/Markov probabilities and sampled prefix commit")
    parser.add_argument("--verify-control",
                        action="store_true",
                        help="Exercise production C6 prefix and draft graphs using an 8192-position backing buffer")
    parser.add_argument("--prefill-context",
                        action="store_true",
                        help="Check bounded auxiliary states with real MTP KV insertion")
    parser.add_argument("--paged-context", action="store_true", help="Use the paged attention geometry of serving")
    parser.add_argument("--context-start",
                        type=int,
                        default=0,
                        help="Exercise absolute draft positions beyond the initial RoPE bucket")
    args = parser.parse_args()
    if args.context_start < 0 or args.context_start + 8192 > 65536:
        parser.error("--context-start must leave room for the diagnostic's 8192-position bank")
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    # Load the same DSpark defaults as serving; a library-only component
    # profile must not leave native RoPE or graph replay silently disabled.
    prepare_environment(args.prepared, tensor_parallel_size=tp_size,
                        pipeline_parallel_size=manifest["pipeline_parallel_size"])
    if args.real_target:
        args.prepared_prefix = True
    torch.hpu.set_device(rank)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    bind_worker_cpu(rank)
    init_distributed_environment(world_size=tp_size,
                                 rank=rank,
                                 distributed_init_method="env://",
                                 local_rank=rank,
                                 backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=1)
    initialize_tp2_fused_ar_norm_runtime()
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    directory = args.prepared
    config = json.loads((directory / "config.json").read_text())
    if manifest["tensor_parallel_size"] != tp_size:
        raise ValueError("Draft checkpoint TP geometry differs from the leased worker group")
    sampling_stage = manifest["pipeline_parallel_size"] - 1
    shard = PreparedV41Shard(directory, sampling_stage, rank)
    target_names = tuple(f"layers.{index}." for index in range(20, 24))
    specs = {
        name: spec
        for name, spec in shard.specs.items()
        if name.startswith("mtp.") or name == "head.weight" or (args.real_target and name.startswith(target_names))
    }
    tree = _weight_tree(specs)
    load_weight_tree(shard, tree, "hpu", specs)
    reduce, gather = stage_collectives(rank, True, tp_size)
    if args.paged_context:
        from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
        shared = PagedCSA2SharedState(config["text_config"], 20, 24, "hpu", 65536, tensor_parallel_size=tp_size)
    else:
        shared = CSA2SharedState(config["text_config"], 20, 24, "hpu")
    stage = SimpleNamespace(weights=tree,
                            config=config,
                            tp_rank=rank,
                            tensor_parallel_size=tp_size,
                            shard=shard,
                            shared=shared,
                            bf16_head=False,
                            reduce=reduce,
                            all_gather=gather)
    draft = PreparedDraft(stage, mxfp4_bf16_lut(torch.device("hpu")), "hpu")
    if args.paged_context:
        assert all(layer.attention.rotary.shape[0] == layer.attention.length for layer in draft.layers)
    from vllm_gaudi.v1.worker.deepseek_v41_runner import DraftContextPlans
    insert = DraftContextPlans(draft.insert_context)
    forward = torch.compile(draft.forward_local if args.local_vocab else draft,
                            backend="hpu_backend",
                            fullgraph=True,
                            dynamic=False)
    sample = torch.compile(draft.sample_greedy_local if args.local_vocab else draft.sample_greedy,
                           backend="hpu_backend",
                           fullgraph=True,
                           dynamic=False)
    verify_prefix = torch.compile(draft.verify_prefix, backend="hpu_backend", fullgraph=True, dynamic=False)
    draft_from_prefix = torch.compile(draft.draft_from_prefix, backend="hpu_backend", fullgraph=True, dynamic=False)
    if args.official_sampling:
        from vllm_gaudi.ops.deepseek_v41_speculative_sampling import SpeculativeRequestSampling

        if not args.verify_control or not args.local_vocab:
            parser.error("Official sampling requires verify-control and local-vocab")
        sampled_verify = torch.compile(draft.verify_sampled_prefix_full, backend="hpu_backend", fullgraph=True,
                                       dynamic=False)
        sampled_draft = torch.compile(draft.draft_sampled_from_prefix, backend="hpu_backend", fullgraph=True,
                                      dynamic=False)
        sampling_state = SpeculativeRequestSampling((1., .95, -1.), 42, tree.head.weight.shape[0], "hpu")
    position_bank = torch.arange(args.context_start, args.context_start + 8192, device="hpu", dtype=torch.int32)
    verify_positions = position_bank[:6]
    prefill_context = None
    if args.prefill_context:
        capacity = draft.layers[0].attention.swa.shape[0]
        residuals = [
            torch.randn(8192, 4, 5120, generator=torch.Generator().manual_seed(137 + index),
                        dtype=torch.bfloat16).to("hpu") for index in range(3)
        ]
        original = torch.cat([value.mean(1) for value in residuals], -1)
        insert(original, position_bank)
        torch.hpu.synchronize()
        expected = [layer.attention.swa.cpu() for layer in draft.layers]
        for layer in draft.layers:
            layer.attention.swa.zero_()
        bounded = torch.cat([draft_context_state(value, capacity, grouped_prefill=True) for value in residuals], -1)
        insert(bounded, position_bank)
        torch.hpu.synchronize()
        for layer, reference in zip(draft.layers, expected, strict=True):
            torch.testing.assert_close(layer.attention.swa.cpu(), reference, rtol=0, atol=0)
        prefill_context = dict(status="passed_exact_real_mtp_swa",
                               prompt_rows=8192,
                               auxiliary_rows=bounded.shape[0],
                               position_rows=position_bank.numel(),
                               ring_capacity=capacity,
                               sliding_window=draft.layers[0].attention.window,
                               real_layers=3)
        del original, bounded, residuals, expected
        for layer in draft.layers:
            layer.attention.swa.zero_()
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
    from vllm_gaudi.ops.tp2_prepared_plan import (collect_prepared_group_replays, record_native_decoder_outputs,
                                                  replay_native_decoder, prepared_group_stats)

    def prefix(value):
        value = reduce(value)
        value = value * 0.25
        return reduce(value) * 0.5

    compiled_prefix = torch.compile(prefix, backend="hpu_backend", fullgraph=True, dynamic=False)
    owner = torch.nn.Identity()
    topology = DecoderTopology("deepseek_v41_pp1", (1, ), 2, False)
    if args.real_target:
        owner = TargetGroup(stage)
        ordinary_prefix = torch.compile(owner, backend="hpu_backend", fullgraph=True, dynamic=False)
        compiled_prefix = torch.compile(owner.native_forward, backend="hpu_backend", fullgraph=True, dynamic=False)
        topology = DecoderTopology("deepseek_v41_pp1", (4, ), 2, False)
    outputs = []
    for step, count in enumerate((6, 1, 2, 3, 4, 5, 6)):
        if step == 1:
            for layer in draft.layers:
                value = layer.attention.swa
                pool = torch.zeros((2, *value.shape), device=value.device, dtype=value.dtype)
                layer.attention.swa = pool[1]
        torch.manual_seed(41 + step)
        values = torch.randn(count, 15360, device="cpu", dtype=torch.bfloat16).to("hpu")
        positions = torch.arange(args.context_start, args.context_start + count, device="hpu", dtype=torch.int32)
        if args.real_target:
            residual = values[:, :5120].unsqueeze(1).expand(-1, 4, -1).contiguous()
            pre = torch.tensor([[1., 0., 0., 0.]] * count, dtype=torch.float32, device="hpu")
            if count != 1:
                values = torch.cat([ordinary_prefix(residual, pre, positions)] * 3, -1)
        if args.prepared_prefix and count == 1:
            source = residual if args.real_target else values[:, :5120].contiguous()
            states = stage_state_tensors(owner) if args.real_target else ()
            roots = dict(hidden_states=source,
                         positions=positions,
                         residual=None,
                         pre_mix=pre if args.real_target else None,
                         state_tensors=states,
                         state_generation=0,
                         metadata=SimpleNamespace(is_prompt=False, native_completion=None))
            result = replay_native_decoder(owner, **roots)
            if result is None:
                with collect_prepared_group_replays(owner=owner,
                                                    adapter=topology,
                                                    snapshot=lambda states=states: _Snapshot(states),
                                                    **roots):
                    output = compiled_prefix(source, pre, positions) if args.real_target else compiled_prefix(source)
                    record_native_decoder_outputs(output)
                result = output, None
            values = torch.cat([result[0]] * 3, -1)
            print(f"RANK {rank} STEP {step} prefix " + json.dumps(prepared_group_stats()), flush=True)
        print(f"RANK {rank} STEP {step} C{count} context", flush=True)
        if args.verify_control:
            from vllm_gaudi.ops.deepseek_v41_verify import DRAFT_START, STATUS
            states = torch.zeros(6, values.shape[1], device="hpu", dtype=values.dtype)
            states[:count].copy_(values)
            target_hidden = states[:, :5120].contiguous()
            proposed = torch.full((5, ), -1, device="hpu", dtype=torch.int64)
            metadata = torch.tensor([step + 1, count, 0, 64, args.context_start, 65536, 1],
                                    device="hpu",
                                    dtype=torch.int64)
            if args.official_sampling:
                controls, acceptance, correction, target_controls = sampling_state.next_draws()
                prefix = sampled_verify(target_hidden, proposed, sampling_state.proposal, metadata,
                                        states, verify_positions, target_controls, acceptance, correction)
                record, _, confidence, probability, covered = sampled_draft(
                    metadata, verify_positions, *prefix[:6], controls)
                sampling_state.proposal.copy_(probability)
                assert bool(covered.all().cpu())
                mass = probability.sum(-1)
                torch.distributed.all_reduce(mass)
                torch.testing.assert_close(mass.cpu(), torch.ones(5), rtol=2e-5, atol=2e-6)
            else:
                prefix = verify_prefix(target_hidden, proposed, metadata, states, verify_positions)
                record, _, confidence = draft_from_prefix(metadata, verify_positions, prefix[1], prefix[2], prefix[3],
                                                          prefix[4], prefix[5], prefix[6])
            payload = record.cpu()
            assert payload[1].item() == count and payload[2].item() == 1
            assert payload[3].item() == 5 and payload[STATUS].item() == 0
            tokens = payload[DRAFT_START:STATUS]
            if args.official_sampling and count == 6:
                verify_metadata = metadata.clone()
                verify_metadata[2] = 5
                _, acceptance, correction, target_controls = sampling_state.next_draws()
                verification = sampled_verify(target_hidden, record[DRAFT_START:STATUS], sampling_state.proposal,
                                               verify_metadata, states, verify_positions, target_controls,
                                               acceptance, correction)
                commit = verification[6].cpu()
                assert commit[STATUS].item() == 0 and 1 <= commit[1].item() <= 6
                assert commit[2].item() == commit[1].item()
                outputs.append({"sampled_c6_commit": commit.tolist(), "q_mass": mass.cpu().tolist(),
                                "acceptance_is_synthetic": True})
        else:
            insert(values, positions)
            first = torch.tensor([step + 1], device="hpu", dtype=torch.int64)
            draft_positions = torch.arange(args.context_start + count,
                                           args.context_start + count + 5,
                                           device="hpu",
                                           dtype=torch.int32)
            hidden, logits = forward(first, draft_positions)
            print(f"RANK {rank} STEP {step} sampling", flush=True)
            tokens, confidence = sample(first, hidden, logits)
        tokens, confidence = tokens.cpu(), confidence.cpu()
        assert tokens.shape == (5, ) and (tokens >= 0).all() and (tokens < 129280).all()
        assert torch.isfinite(confidence).all()
        outputs.append({"context_count": count, "tokens": tokens.tolist()})
        print(f"RANK {rank} STEP {step} complete", flush=True)
    evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
    (evidence / f"draft-rank{rank}.json").write_text(
        json.dumps(
            {
                "tp": tp_size,
                "pp": 1,
                "purpose": "DSpark synchronization diagnostic; real target and PP absent",
                "prepared_prefix": args.prepared_prefix,
                "local_vocab": args.local_vocab,
                "verify_control": args.verify_control,
                "official_sampling": args.official_sampling,
                "prefill_context": prefill_context,
                "paged_context": args.paged_context,
                "context_start": args.context_start,
                "draft_rotary_rows": [layer.attention.rotary.shape[0] for layer in draft.layers],
                "position_backing_rows": position_bank.numel() if args.verify_control else None,
                "verify_position_rows": verify_positions.numel() if args.verify_control else None,
                "real_target_layers": [20, 21, 22, 23] if args.real_target else [],
                "native": prepared_group_stats(),
                "outputs": outputs,
                "peak_device_bytes": torch.hpu.max_memory_allocated()
            },
            indent=2) + "\n")
    from vllm_gaudi.ops.tp2_prepared_plan import shutdown_prepared_group_plans
    from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
    shutdown_prepared_group_plans()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp_size))):
        main()
