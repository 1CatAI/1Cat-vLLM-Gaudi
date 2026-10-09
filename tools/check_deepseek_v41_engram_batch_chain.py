# SPDX-License-Identifier: Apache-2.0
"""Actual C6 device histories -> mapped checkpoint -> native TP gather/Wkv/update.

Both arms use the production producer and shared backing. No host upload,
synthetic checkpoint, alternate communication, or attribution profiler.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    cache = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG")
    if cache:
        os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = cache.replace("{rank}", str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    from transformers import AutoTokenizer
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.distributed import init_distributed_environment, initialize_model_parallel, destroy_model_parallel
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree, linear
    from vllm_gaudi.ops.deepseek_v41_device_engram import DeviceEngramRounds
    from vllm_gaudi.ops.deepseek_v41_engram import EngramHashLayout, EngramTokenHistory, build_compressed_token_map
    from vllm_gaudi.ops.deepseek_v41_residency import shared_host_region
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.tp2_prepared_plan import (
        shutdown_prepared_group_plans, collect_prepared_group_replays,
        record_native_decoder_outputs, replay_native_decoder, _native_entries)
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
    from vllm_gaudi.ops.deepseek_v41_replay import _Snapshot
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers

    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(rank=rank, status="loading", checks=[], rounds=[], formal_gain_ms=0,
                  scope="two actual Engram layers, producer -> TP4 gather -> Wkv -> fused update, native consumer")
    path = root / f"engram-chain-rank{rank}.json"
    save = lambda: path.write_text(json.dumps(report, indent=2) + "\n")
    save()
    cursors, descriptors = [], []
    try:
        torch.hpu.set_device(rank)
        bind_worker_cpu(rank)
        torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
        init_distributed_environment(world_size=4, rank=rank, distributed_init_method="env://",
                                     local_rank=rank, backend="hccl")
        config = EngineArgs(model=str(args.prepared), dtype="bfloat16", tensor_parallel_size=4,
                            pipeline_parallel_size=1, load_format="dsv41_prepared", max_model_len=1048576,
                            max_num_seqs=32, max_num_batched_tokens=8192, block_size=128,
                            enable_prefix_caching=False, async_scheduling=False,
                            speculative_config={"method": "dspark", "num_speculative_tokens": 5}).create_engine_config()
        with set_current_vllm_config(config), torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            _, gather = stage_collectives(rank, True, 4)
            # This also initializes the unchanged production communication runtime.
            gather(torch.ones((1, 6), dtype=torch.bfloat16, device="hpu"), 1).cpu()
            bind_worker_helpers(rank)
            text = json.loads((args.prepared / "config.json").read_text())["text_config"]
            layout = EngramHashLayout.from_config(text)
            manifest = json.loads((args.prepared / "manifest.json").read_text())
            table_manifest = json.loads((args.prepared / manifest["engram_host_shards"][str(rank)]["file"]).read_text())
            token_map, compressed = build_compressed_token_map(
                AutoTokenizer.from_pretrained(args.prepared, local_files_only=True))
            assert compressed == layout.vocab_size
            host = SimpleNamespace(tensor_parallel_size=4, tp_rank=rank, layout=layout,
                                   history=EngramTokenHistory(layout, token_map), shards={}, table_sources={})
            raw = [torch.load(args.fixtures / f"rank{rank}/c6-{index}.pt", map_location="cpu", weights_only=False)
                   for index in range(3)]
            assert all(case["request_context_qualified"] and case["history_context_qualified"] for case in raw)
            ids = torch.empty((6,), dtype=torch.int32, device="hpu")
            history = torch.empty((3,), dtype=torch.int32, device="hpu")
            shard = PreparedV41Shard(args.prepared, 0, rank)
            specs = {name: spec for name, spec in shard.specs.items()
                     if any(name.startswith(f"layers.{layer}.engram.") for layer in layout.layer_ids)}
            weights = _weight_tree(specs)
            load_weight_tree(shard, weights, "hpu", specs)

            class Consumer(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weights = weights
                    self.generation = 1
                    self.precision_fingerprint = "production-bf16-engram"

                def forward(self, first, second, residual, image):
                    outputs = []
                    for layer, rows in zip(layout.layer_ids, (first, second), strict=True):
                        w = self.weights.layers.get_submodule(str(layer)).engram
                        kv = linear(gather(rows, 1).flatten(1), w.wkv)
                        outputs.append(torch.ops.custom_op.custom_deepseek_v41_engram_update_bf16_gaudi2(
                            residual, kv, w.q_weight, w.k_weight, ~image, text["rms_norm_eps"]))
                    return tuple(outputs)

            residual = raw[0]["groups"][0]["residual"].to("hpu")
            image = torch.empty((6,), dtype=torch.bool, device="hpu")
            owners, compiled, entries = [], [], []
            metadata = SimpleNamespace(native_completion=None)
            topology = DecoderTopology("deepseek_v41_tp4_engram", (1,), 0, False, 2)

            def roots(owner, rows):
                return dict(hidden_states=residual, pre_mix=None, positions=None, input_ids=None,
                            attention_inputs=(*rows, image), pp_wire=None, metadata=metadata,
                            state_generation=(owner.generation, owner.precision_fingerprint), state_tensors=())

            def replay(arm, rows):
                value = replay_native_decoder(owners[arm], **roots(owners[arm], rows))
                if value is None:
                    raise RuntimeError("Captured Engram native input contract changed")
                return value
            for arm in (0, 1):
                owner = Consumer()
                owners.append(owner)
                function = torch.compile(owner, backend="hpu_backend", fullgraph=True, dynamic=False)
                compiled.append(function)
                ids.copy_(raw[0]["ids"].to(torch.int32))
                history.copy_(raw[0]["cursor_history"].to(torch.int32))
                image.copy_((raw[0]["ids"] == 129264) | (raw[0]["ids"] == 129265))
                # Qualify capture before copying/pinning the large tables.
                # These fixed destinations are the ordinary stage input
                # staging buffers; both measured arms use the same binding.
                rows = tuple(torch.zeros((6, 6, layout.head_dim), dtype=torch.bfloat16, device="hpu")
                             for _ in layout.layer_ids)
                arguments = (*rows, residual, image)
                for _ in range(3):
                    result = replay_native_decoder(owner, **roots(owner, rows))
                    if result is None:
                        with collect_prepared_group_replays(owner=owner, adapter=topology,
                                                           snapshot=lambda: _Snapshot(()), **roots(owner, rows)):
                            result = function(*arguments)
                            record_native_decoder_outputs(*result)
                    torch.hpu.synchronize()
                if owner not in _native_entries:
                    raise RuntimeError("Engram consumer did not capture production joint native replay")
                entries.append(_native_entries[owner])
            report["native_commands"] = [entry[0].captured_command_count() for entry in entries]
            report["producer_launches_per_round"] = [12, 2]
            report["status"] = "native_consumers_ready"
            save()
            print(f"rank{rank}: real-weight native consumers captured", flush=True)
            for layer in layout.layer_ids:
                host.shards[layer] = layout.head_shard(layer, rank, 4)
                sources = []
                for kind in ("weight", "scale"):
                    item = table_manifest["tables"][f"layers.{layer}.engram.embed.{kind}"]
                    region = dict(file=item["file"], offset=item["shard_offset"], length=item["shard_bytes"])
                    descriptor, binding = shared_host_region(region)
                    descriptors.append(descriptor)
                    sources.append(dict(item, file=binding["file"], shard_offset=0, shared_memfd=True))
                host.table_sources[layer] = tuple(sources)
            print(f"rank{rank}: production mapped backing ready", flush=True)
            for arm in (0, 1):
                os.environ["VLLM_HPU_DSV41_DSPARK_BATCH_ENGRAM"] = str(arm)
                cursors.append(DeviceEngramRounds(host, ids, history))
            for index, case in enumerate(raw):
                ids.copy_(case["ids"].to(torch.int32))
                history.copy_(case["cursor_history"].to(torch.int32))
                residual.copy_(case["groups"][0]["residual"])
                image.copy_((case["ids"] == 129264) | (case["ids"] == 129265))
                outputs, histories, rows_saved = [], [], []
                for arm in (0, 1):
                    report["active_case_arm"] = [index, arm]
                    save()
                    rows = cursors[arm].prepare("request", ids, history)
                    value = replay(arm, rows)
                    outputs.append(tuple(v.cpu() for v in value))
                    histories.append(cursors[arm].histories.cpu())
                    rows_saved.append(tuple(v.cpu() for v in rows))
                assert torch.equal(histories[0], histories[1])
                for first, second in zip((*rows_saved[0], *outputs[0]), (*rows_saved[1], *outputs[1]), strict=True):
                    assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))
                if case.get("engram_histories") is not None:
                    assert torch.equal(histories[0], case["engram_histories"].to(torch.int32))
                report["checks"].append(dict(case=index, rows_histories_consumers_exact=True))
                save()
            for arm in (0, 1):
                for _ in range(4):
                    rows = cursors[arm].prepare("request", ids, history)
                    replay(arm, rows)
            torch.hpu.synchronize()
            for iteration in range(3):
                for arm in (0, 1):
                    samples = []
                    for sample in range(6):
                        case = raw[sample % 3]
                        ids.copy_(case["ids"].to(torch.int32))
                        history.copy_(case["cursor_history"].to(torch.int32))
                        residual.copy_(case["groups"][0]["residual"])
                        image.copy_((case["ids"] == 129264) | (case["ids"] == 129265))
                        torch.hpu.synchronize()
                        begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                        begin.record()
                        rows = cursors[arm].prepare("request", ids, history)
                        replay(arm, rows)
                        end.record()
                        end.synchronize()
                        samples.append(begin.elapsed_time(end))
                    report["rounds"].append(dict(iteration=iteration, arm=arm, device_ms=samples,
                                                median_device_ms=statistics.median(samples)))
                    save()
            report["status"] = "component_passed"
            save()
    except BaseException as error:
        import traceback
        traceback.print_exc()
        report.update(status="failed", error=repr(error))
        report["producer_launch_counts"] = [[p.launch_count() for p in cursor.producers] for cursor in cursors]
        save()
        raise
    finally:
        shutdown_prepared_group_plans()
        for cursor in cursors:
            cursor.close()
        for descriptor in descriptors:
            os.close(descriptor)
        if torch.distributed.is_initialized():
            destroy_model_parallel()
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
