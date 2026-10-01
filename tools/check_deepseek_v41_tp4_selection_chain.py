# SPDX-License-Identifier: Apache-2.0
"""Gate distributed index selection through real attention and hc_post consumers.

Layers 2/8/14/20/24 retain their real TP4 attention weights and shared cache.
Layer 24 consumes the candidate blocks just published by layer 20. This is an
attention component with residual feedback, not a full decoder measurement.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from types import FunctionType, MethodType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--physical-tokens", type=int, default=6178944)
    parser.add_argument("--warm-steps", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--local-index-queries", action="store_true")
    parser.add_argument("--logical-mla", action="store_true")
    parser.add_argument("--index-mirror", action="store_true")
    parser.add_argument("--batched-selection", action="store_true",
                        help="Gate the normal FullR1 batched selector against the mirror/streamed parent")
    parser.add_argument("--contracts-only", action="store_true",
                        help="Exercise prepared bindings and request/capacity transitions without timing")
    parser.add_argument("--no-tile-partition", action="store_true")
    parser.add_argument("--reference-timing", type=Path,
                        help="Reuse the archived comparable parent timing; reference execution is correctness only")
    parser.add_argument("--reference-arm", choices=("parent", "candidate"), default="parent")
    parser.add_argument("--steady-drain-boundary", action="store_true")
    parser.add_argument("--query-check-reference", type=Path,
                        help="Reuse unchanged query/tie checks after verifying the archived source contracts")
    parser.add_argument("--diagnostic-pair-state", type=Path,
                        help="Reuse that run's query/tie checks and inspect only the first two stateful steps")
    args = parser.parse_args()
    if args.no_tile_partition and not (args.local_index_queries or args.logical_mla or args.index_mirror):
        parser.error("Disabling tile partition requires a local-query or logical-MLA candidate")
    if args.logical_mla and not args.no_tile_partition:
        parser.error("The logical-MLA gate retains the original local selection")
    if args.index_mirror and (not args.logical_mla or not args.no_tile_partition or args.local_index_queries):
        parser.error("Index mirrors compose with logical MLA and the original query/selection contract")
    if args.contracts_only and not args.index_mirror:
        parser.error("The bounded ownership gate requires the index-mirror candidate")
    if args.batched_selection and (not args.index_mirror or args.contracts_only):
        parser.error("Batched selection requires the mirror attention timing gate")
    query_reference = args.diagnostic_pair_state or args.query_check_reference
    if args.diagnostic_pair_state and (args.warm_steps != 0 or args.steps != 2):
        parser.error("Pair-state diagnostic requires zero warm steps and exactly two steps")
    if args.physical_tokens % 128 or args.physical_tokens < 32768 + 128:
        parser.error("Physical pool must contain whole pages and the populated prefix")
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    if "{rank}" in os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", ""):
        os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ["PT_HPU_RECIPE_CACHE_CONFIG"].replace("{rank}", str(rank))
    if os.getenv("GRAPH_VISUALIZATION") == "1":
        graph_dir = Path(os.environ["GRAPH_VISUALIZATION_DIR"]) / f"rank{rank}"
        graph_dir.mkdir(parents=True, exist_ok=True)
        os.environ["GRAPH_VISUALIZATION_DIR"] = str(graph_dir)
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import torch.distributed as dist
    from habana_frameworks.torch.hpu.metrics import metric_global
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel,
                                  init_distributed_environment, initialize_model_parallel)
    from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree, linear
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_math import hc_post
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention, PagedCSA2SharedState
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives, stage_state_tensors, _PagedSnapshot
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar

    class AttentionConsumer(torch.nn.Module):
        def __init__(self, attention):
            super().__init__()
            self.attention = attention

        def forward(self, residual, positions, post, comb):
            # Residual feedback changes each query and each written KV row.
            # This common fixture collapse is included in both timers.
            value = residual.float().mean(1).to(torch.bfloat16)
            output = self.attention(value, positions, ready_outputs=(post, comb), decode=True)
            return hc_post(output, residual, post, comb)

        def queries(self, value, qr, positions):
            return self.attention._prepare_index_queries(value, qr, positions)

    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method="env://", local_rank=rank,
                                 backend="hccl")
    config = EngineArgs(model=str(args.prepared), dtype="bfloat16", tensor_parallel_size=4,
                        pipeline_parallel_size=1, load_format="dsv41_prepared", max_model_len=1048576,
                        max_num_seqs=32, max_num_batched_tokens=8192, block_size=128,
                        enable_prefix_caching=False, async_scheduling=True).create_engine_config()
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    result_path = root / f"selection-chain-rank{rank}.json"
    report = dict(status="running", rank=rank, scope=__doc__, checks=[], timings={},
                  warm_steps=args.warm_steps, steps=args.steps, physical_tokens=args.physical_tokens,
                  local_index_queries=args.local_index_queries, steady_drain_boundary=args.steady_drain_boundary,
                  tile_partition=not args.no_tile_partition,
                  logical_mla=args.logical_mla,
                  index_mirror=args.index_mirror,
                  batched_selection=args.batched_selection,
                  reference_reason=("Reuse the archived comparable component timing; parent runs are correctness only."
                                    if args.reference_timing else
                                    "Measure the missing component parent; reuse the full serving baseline."))

    def save():
        result_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        with set_current_vllm_config(config), torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            cpu_group = dist.new_group(backend="gloo")
            reduce, gather = stage_collectives(rank, False)
            reduce(torch.ones(1, device="hpu", dtype=torch.bfloat16)).cpu()
            bind_worker_helpers(rank)
            compilation_metric = metric_global("graph_compilation")
            fallback_metric = metric_global("cpu_fallback")
            if query_reference:
                saved = json.loads((query_reference / f"selection-chain-rank{rank}.json").read_text())
                assert len(saved["checks"]) >= 21 and saved["local_index_queries"]
                source = Path(__file__).resolve().parents[1]
                for name in ("deepseek_v41_tp4_selection.py", "deepseek_v41_paged_attention.py"):
                    relative = Path("vllm_gaudi/ops") / name
                    archived = query_reference / "source" / relative
                    if name == "deepseek_v41_tp4_selection.py":
                        def query_helpers(path):
                            nodes = ast.parse(path.read_text()).body
                            return [ast.dump(node, include_attributes=False) for node in nodes
                                    if isinstance(node, ast.FunctionDef)
                                    and node.name != "invalidate_local_index_queries"]
                        assert query_helpers(source / relative) == query_helpers(archived)
                    else:
                        def query_contract(path):
                            methods = {"_prepare_index_queries", "_scores", "_stream_topk", "_select", "_merge_topk"}
                            if args.index_mirror:
                                # Query projections and tie/merge arithmetic
                                # are unchanged. The new scoring producer is
                                # checked against real packed keys below.
                                methods.difference_update(("_scores", "_stream_topk"))
                            tree = ast.parse(path.read_text())
                            return [ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                                    if isinstance(node, ast.FunctionDef) and node.name in methods]
                        assert query_contract(source / relative) == query_contract(archived)
                report["checks"] = saved["checks"][:21]
                report["query_tie_reference"] = str(query_reference)

            class TieContract(torch.nn.Module):
                _merge_topk = staticmethod(PagedCSA2Attention._merge_topk)

                def __init__(self, ratio, visible, collect_blocks, partition):
                    super().__init__()
                    self.ratio, self.visible = ratio, visible
                    self.collect_blocks, self.partition = collect_blocks, partition
                    self.tensor_parallel_size, self.prefill_tp_rank, self.gather = 4, rank, gather

                def _scores(self, positions, rows, q, bank):
                    del q
                    indices = rows.clamp_min(0).remainder(bank.shape[-1]).long().reshape(1, -1)
                    scores = bank.gather(1, indices)
                    count = ((positions + 1) // self.ratio).unsqueeze(-1)
                    return scores.masked_fill((rows < 0) | (rows >= count), -torch.inf)

                def forward(self, positions, rows, q, bank):
                    return PagedCSA2Attention._stream_topk(
                        self, positions, rows, q, bank, collect_blocks=self.collect_blocks,
                        visible_rows=self.visible, partition_tiles=self.partition)

            tie_q = torch.zeros(1, 32, 128, dtype=torch.bfloat16, device="hpu")
            tie_cases = () if query_reference else ((2, 10240, False, False), (1, 20480, True, False),
                                                    (1, None, False, True))
            for ratio, visible, publish, reindex in tie_cases:
                rows = torch.arange(16384 if reindex else 32768 // ratio, dtype=torch.int32)
                position = 16384
                if reindex:
                    rows[::7] = -1
                    rows[::11] = 1048575
                    rows = rows.reshape(1, -1)
                    position = 1048575
                rows = rows.to("hpu")
                pos = torch.tensor([position], dtype=torch.int32, device="hpu")
                functions = [torch.compile(TieContract(ratio, visible, publish, flag), backend=make_backend(),
                                           fullgraph=True, dynamic=False) for flag in (False, True)]
                for all_tied in (False, True):
                    bank = (torch.zeros(32768) if all_tied else torch.arange(32768).remainder(23).float())
                    bank = bank.reshape(1, -1).to("hpu")
                    expected = functions[0](pos, rows, tie_q, bank)
                    actual = functions[1](pos, rows, tie_q, bank)
                    for got, want in zip(actual, expected, strict=True):
                        if want is not None:
                            torch.testing.assert_close(got.cpu(), want.cpu(), rtol=0, atol=0)
                        else:
                            assert got is None
                    report["checks"].append(dict(scope="fourrank_tie_transport", ratio=ratio, reindex=reindex,
                                                  all_tied=all_tied, exact_unsorted_ids_and_publication=True))
                    save()
            if args.batched_selection:
                from vllm_gaudi.ops.deepseek_v41_decode_selection import batched_decode_selection

                class BatchedTieContract(torch.nn.Module):
                    _merge_topk = staticmethod(PagedCSA2Attention._merge_topk)

                    def __init__(self, visible, batched):
                        super().__init__()
                        self.visible, self.batched, self.ratio = visible, batched, 1

                    def _scores(self, positions, rows, q, bank):
                        del q
                        return bank.gather(1, rows.long().reshape(1, -1))

                    def forward(self, bank, rows, positions):
                        scores = bank.masked_fill(rows[None, :] >= (positions + 1)[:, None], -torch.inf)
                        if self.batched:
                            return batched_decode_selection(list(scores.split(2048, -1)), rows, positions, 1,
                                                            collect_blocks=True, selected_tiles=self.visible // 2048)
                        return PagedCSA2Attention._stream_topk(
                            self, positions, rows, bank.new_empty(1, 32, 128), scores,
                            collect_blocks=True, visible_rows=self.visible)

                tie_rows = torch.arange(32768, device="hpu", dtype=torch.int32)
                for visible, position in ((20480, 16384), (24576, 20480), (32768, 32767)):
                    functions = [torch.compile(BatchedTieContract(visible, enabled), backend=make_backend(),
                                               fullgraph=True, dynamic=False) for enabled in (False, True)]
                    tie_position = torch.tensor([position], device="hpu", dtype=torch.int32)
                    for distribution in ("cutoff_ties", "all_tied", "all_invalid"):
                        scores = torch.arange(32768).remainder(23).float()
                        if distribution == "all_tied":
                            scores.zero_()
                        elif distribution == "all_invalid":
                            scores.fill_(-torch.inf)
                        scores = scores.reshape(1, -1).to("hpu")
                        expected = functions[0](scores, tie_rows, tie_position)
                        actual = functions[1](scores, tie_rows, tie_position)
                        for got, want in zip(actual, expected, strict=True):
                            torch.testing.assert_close(got.cpu(), want.cpu(), rtol=0, atol=0)
                        report["checks"].append(dict(scope="batched_full_r1_ties", visible=visible,
                                                      distribution=distribution, exact_ids_and_publication=True))
                        save()
            shard = PreparedV41Shard(args.prepared, 0, rank)
            text = json.loads((args.prepared / "config.json").read_text())["text_config"]
            layers = (2, 8, 14, 20, 24)
            specs = {name: spec for name, spec in shard.specs.items()
                     if name.startswith(tuple(f"layers.{layer}.attn." for layer in layers))}
            tree = _weight_tree(specs)
            load_weight_tree(shard, tree, "hpu", specs,
                             woa_sidecar=WoaFP8Sidecar(args.prepared / "sidecars/wo_a_fp8", shard),
                             woa_layers=layers,
                             dense_sidecar=DenseFP8Sidecar(args.prepared / "sidecars/attention_dense_fp8", shard),
                             dense_config={"wq_b": layers, "wo_b": layers})
            program = torch.nn.Module()
            program.length, program.search_length = 1048576, 32768
            program.generation, program.pp_rank, program.tensor_parallel_size = 1, 0, 4
            program.decode_token_bound, program.replay_owner = 20480, None
            program.shared = PagedCSA2SharedState(text, 0, 28, "hpu", program.length, tensor_parallel_size=4)
            page_end = args.physical_tokens // 128
            program.shared.block_table[:256].copy_(
                torch.arange(page_end - 256, page_end, dtype=torch.int32, device="hpu"))
            for cache in program.shared.sources.values():
                cache.main = torch.zeros(args.physical_tokens // cache.ratio, 288, dtype=torch.uint8, device="hpu")
                cache.index = torch.zeros(args.physical_tokens // cache.ratio, 68, dtype=torch.uint8, device="hpu")
            program.layers = torch.nn.ModuleList()
            for layer in layers:
                attention = PagedCSA2Attention(tree.layers.get_submodule(str(layer)).attn, text, layer,
                                               program.shared, linear, reduce, gather, "hpu", 4)
                attention.prefill_tp_rank = rank
                attention.prepare_qkv_input_weight()
                attention.prepare_compressor_input_weight()
                attention.woa_fp8 = attention.woa_output_roundtrip = True
                program.layers.append(AttentionConsumer(attention))
            for start in (0, 8192):
                generator = torch.Generator().manual_seed(7130 + start)
                value = torch.randn(8192, 5120, generator=generator).bfloat16().to("hpu")
                positions = torch.arange(start, start + 8192, dtype=torch.int32, device="hpu")
                workspace = program.shared.prefill_main_workspace
                if workspace is not None:
                    program.shared.prefill_kv_generation += 1
                    workspace.begin(program.shared.prefill_kv_generation)
                for block in program.layers:
                    attention = block.attention
                    attention.set_search_length(start + 8192)
                    attention.prefill_token_end = start + 8192
                    value = attention(value, positions)
                assert torch.isfinite(value).all().item()
                print(f"TP{rank} real attention prefix {start}+8192 prepared", flush=True)
            # The distributed producer is valid only for replicated source state.
            replica_hashes = []
            for cache in program.shared.sources.values():
                logical = torch.arange(16384 // cache.ratio, device="hpu", dtype=torch.int32)
                physical = program.shared.physical_rows(logical, cache.ratio).long()
                for name in ("main", "index"):
                    data = getattr(cache, name).index_select(0, physical).cpu().contiguous()
                    replica_hashes.append(hashlib.sha256(data.numpy().tobytes()).hexdigest())
            replicas = [None] * 4
            dist.all_gather_object(replicas, replica_hashes, group=cpu_group)
            assert all(value == replicas[0] for value in replicas), "Source caches differ between TP ranks"
            report["replicated_source_hashes"] = replica_hashes
            if args.index_mirror:
                assert program.shared.index_mirror_tokens == 32768
                program.shared.prepare_index_mirror(16384)
                report["index_mirror_capacity"] = program.shared.index_mirror_tokens
                report["index_mirror_rebuilds"] = program.shared.index_mirror_rebuilds
            references = []
            for block in program.layers:
                block.attention.set_search_length(32768)
                block.attention.prefill_token_end = None
                block.attention.set_decode_visible_tokens(20480)
                reference = copy.copy(block)
                reference._modules = dict(block._modules)
                reference.attention = copy.copy(block.attention)
                reference.attention._buffers = dict(block.attention._buffers)
                if args.logical_mla:
                    assert block.attention.paged_mla_logical, "Normal native capability dispatch was not selected"
                    if not args.index_mirror:
                        reference.attention.paged_mla_logical = False
                if args.batched_selection:
                    assert block.attention.index_mirror_scores
                    assert block.attention.decode_batched_selection == (block.attention.layer == 20)
                    reference.attention.decode_batched_selection = False
                elif args.index_mirror:
                    assert block.attention.index_mirror_scores
                    # Both arms maintain the derived state; the untimed
                    # reference still reads/decodes canonical packed keys.
                    reference.attention.index_mirror_scores = False
                references.append(reference)
                block.attention.tp4_tile_selection = not args.no_tile_partition
                if args.local_index_queries:
                    from vllm_gaudi.ops.deepseek_v41_tp4_selection import prepare_local_index_queries
                    prepare_local_index_queries(block.attention)

            def compile_consumers(blocks, arm, method="forward"):
                result = []
                for block in blocks:
                    if args.contracts_only and method == "forward":
                        from vllm_gaudi.compilation.deepseek_v41_prepared import PreparedTP4Group
                        result.append(PreparedTP4Group(block, program, make_backend()))
                        continue
                    # Like normal prepared groups, each maintained consumer
                    # gets its own finite code cache. Do not raise global
                    # compiler limits to accommodate unrelated layer guards.
                    forward = getattr(AttentionConsumer, method)
                    name = f"selection_{arm}_{method}_layer{block.attention.layer}"
                    entry = FunctionType(forward.__code__.replace(co_name=name), forward.__globals__,
                                         argdefs=forward.__defaults__, closure=forward.__closure__)
                    bound = MethodType(entry, block)
                    result.append(torch.compile(bound, backend=make_backend(), fullgraph=True, dynamic=False))
                return result

            if args.local_index_queries and not query_reference:
                ref_queries = compile_consumers(references, "parent", "queries")
                new_queries = compile_consumers(program.layers, "candidate", "queries")
                generator = torch.Generator().manual_seed(7442)
                for position in (16384, 19372, 32767):
                    value = torch.randn(1, 5120, generator=generator).bfloat16().to("hpu")
                    qr = torch.randn(1, 1280, generator=generator).bfloat16().to("hpu")
                    pos = torch.tensor([position], dtype=torch.int32, device="hpu")
                    for layer, old, new in zip(layers, ref_queries, new_queries, strict=True):
                        expected, actual = old(value, qr, pos), new(value, qr, pos)
                        for got, want in zip(actual, expected, strict=True):
                            torch.testing.assert_close(got.cpu(), want.cpu(), rtol=0, atol=0)
                        report["checks"].append(dict(scope="full_query_and_head_weights", layer=layer,
                                                      position=position, exact=True))
                        save()
            parent = compile_consumers(references, "parent")
            candidate = compile_consumers(program.layers, "candidate")
            generator = torch.Generator().manual_seed(8811)
            seed = torch.randn(1, 4, 5120, generator=generator).bfloat16().to("hpu")
            post = torch.full((1, 4), 0.25, device="hpu")
            comb = torch.softmax(torch.randn(1, 4, 4, generator=generator), -1).to("hpu")
            total = args.warm_steps + args.steps
            # Production presents zero-offset contiguous positions to compiled
            # groups. Offset slices specialize HPU recipes and are not that
            # contract. Prepare immutable independent inputs before execution;
            # there is no staging work inside this attention-only measurement.
            positions = tuple(torch.tensor([16384 + step], dtype=torch.int32, device="hpu")
                              for step in range(total))
            assert all(value.storage_offset() == 0 and value.is_contiguous() for value in positions)
            report["position_contract"] = "cold independent contiguous int32[1], storage_offset=0"
            states = stage_state_tensors(program)
            names = {id(value): name for name, value in program.named_buffers()}
            touched = torch.cat((torch.arange(16384, 16384 + total + 6, dtype=torch.int32, device="hpu"),
                                 torch.arange(20479, 20487, dtype=torch.int32, device="hpu")))
            initial = _PagedSnapshot(program, touched, states)

            def fingerprints(snapshot):
                rows = [(names[id(value)], saved) for value, saved in
                        zip(snapshot.small.tensors, snapshot.small.saved, strict=True)]
                rows.extend((names[id(value)], saved) for value, _, saved in snapshot.rows)
                return [dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                             sha256=hashlib.sha256(value.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
                        for name, value in rows]

            if args.contracts_only:
                from vllm_gaudi.ops.deepseek_v41_state import PagedStageState
                from vllm_gaudi.v1.worker.deepseek_v41_runner import decode_source_prefix_bound
                pages = PagedStageState(program)
                pages.blocks, pages.active = page_end, "a"
                original_pages = list(range(page_end - 256, page_end))
                pages.published_block_ids = tuple(original_pages)
                transitions = [
                    ("initial_mirror", "a", original_pages, 16384, 32768, 1, True),
                    ("packed_same_bucket", "a", original_pages, 16384, 32768, 1, False),
                    ("mirror_same_bucket", "a", original_pages, 16384, 32768, 1, True),
                    ("C2", "a", original_pages, 16384, 32768, 2, True),
                    ("C6", "a", original_pages, 16384, 32768, 6, True),
                    ("different_request", "b", list(range(1, 257)), 16384, 32768, 1, True),
                    ("request_reuse", "a", original_pages, 16384, 32768, 1, True),
                    ("page_remap", "a", original_pages[::-1], 16384, 32768, 1, True),
                    ("last_bounded_token", "a", original_pages, 32767, 32768, 1, True),
                    ("first_unbounded_token", "a", original_pages + [1], 32768, 65536, 1, False),
                    ("return_to_bounded", "a", original_pages, 16384, 32768, 1, True),
                ]
                report["mirror_lifecycle_cases"] = []
                for label, request_id, block_ids, position, search, count, use_mirror in transitions:
                    pages.activate(request_id, block_ids)
                    program.search_length = search
                    program.decode_token_bound = decode_source_prefix_bound(position + count, search, 4)
                    for block in (*program.layers, *references):
                        block.attention.set_search_length(search)
                        block.attention.set_decode_visible_tokens(program.decode_token_bound)
                    if use_mirror:
                        program.shared.prepare_index_mirror(position)
                    else:
                        program.shared.invalidate_index_mirror()
                    local_pos = torch.arange(position, position + count, dtype=torch.int32, device="hpu")
                    local_seed = seed.expand(count, -1, -1).contiguous()
                    local_post = post.expand(count, -1).contiguous()
                    local_comb = comb.expand(count, -1, -1).contiguous()
                    active_states = stage_state_tensors(program)
                    touched_pos = torch.arange(position, position + count + (count > 1),
                                               dtype=torch.int32, device="hpu")
                    saved = _PagedSnapshot(program, touched_pos, active_states)
                    mirror_before = [cache.index_mirror.clone() for cache in program.shared.sources.values()]
                    expected = local_seed
                    for function in parent:
                        expected = function(expected, local_pos, local_post, local_comb)
                    if count > 1:
                        next_pos = torch.tensor([position + count], dtype=torch.int32, device="hpu")
                        expected = expected[-1:].clone()
                        for function in parent:
                            expected = function(expected, next_pos, post, comb)
                    expected = expected.cpu()
                    wanted = fingerprints(_PagedSnapshot(program, touched_pos, active_states))
                    saved.restore()
                    if use_mirror:
                        program.shared.invalidate_index_mirror()
                        for cache in program.shared.sources.values():
                            cache.index_mirror.fill_(17)
                        program.shared.prepare_index_mirror(position)
                    # Include the first actual reader immediately after restore.
                    observed = local_seed
                    for function in candidate:
                        observed = function(observed, local_pos, local_post, local_comb)
                    if count > 1:
                        observed = observed[-1:].clone()
                        for function in candidate:
                            observed = function(observed, next_pos, post, comb)
                    torch.testing.assert_close(observed.cpu(), expected, rtol=0, atol=0)
                    assert fingerprints(_PagedSnapshot(program, touched_pos, active_states)) == wanted
                    if not use_mirror:
                        for cache, before in zip(program.shared.sources.values(), mirror_before, strict=True):
                            torch.testing.assert_close(cache.index_mirror.cpu(), before.cpu(), rtol=0, atol=0)
                    saved.restore()
                    report["mirror_lifecycle_cases"].append(dict(case=label, position=position, search=search,
                                                                tokens=count, followup_c1=count > 1,
                                                                exact_output=True, exact_state=True))
                    save()
                report.update(status="index_mirror_contracts_exact_no_timing", timing_collected=False,
                              prepared_variants=[len(function.variants) for function in candidate],
                              preparations=[function.preparations for function in candidate])
                save()
                return

            if args.diagnostic_pair_state:
                # CPU snapshots are diagnostic instrumentation, never a timing
                # boundary or evidence overriding the original failed chain.
                page_table = program.shared.block_table.cpu()
                reference_rows = []
                mismatches = []
                for arm, functions in (("parent", parent), ("candidate", candidate)):
                    initial.restore()
                    torch.hpu.synchronize()
                    output = seed.clone()
                    for step in range(2):
                        position = 16384 + step
                        for ordinal, function in enumerate(functions):
                            output = function(output, positions[step], post, comb)
                            attention = program.layers[ordinal].attention
                            row = dict(output=output.cpu(), selection=attention.selection.indices[:1].cpu(),
                                       candidates=program.shared.candidate_pool[:1].cpu(),
                                       swa=attention.swa[position % 256].cpu())
                            if attention.owns_kv:
                                width = 128 // attention.ratio
                                logical = position // attention.ratio
                                physical = int(page_table[logical // width]) * width + logical % width
                                ids = torch.tensor([logical % width, physical], dtype=torch.int64, device="hpu")
                                row["main_rows"] = attention.cache.main.index_select(0, ids).cpu()
                                row["index_rows"] = attention.cache.index.index_select(0, ids).cpu()
                                if attention.ratio == 2:
                                    row["kv_history"] = attention.kv_history.cpu()
                                    row["score_history"] = attention.score_history.cpu()
                            if arm == "parent":
                                reference_rows.append(row)
                            else:
                                reference = reference_rows[step * len(functions) + ordinal]
                                different = {}
                                for name, value in row.items():
                                    wanted = reference[name]
                                    if not torch.equal(value, wanted):
                                        delta = (value.float() - wanted.float()).abs()
                                        different[name] = dict(elements=int((value != wanted).sum()),
                                                               max_abs=float(delta.max()))
                                if different:
                                    mismatches.append(dict(step=step, layer=layers[ordinal], tensors=different))
                                    torch.save(dict(reference=reference, actual=row),
                                               root / f"pair-state-rank{rank}-step{step}-layer{layers[ordinal]}.pt")
                report.update(status="pair_state_diagnostic_complete_not_performance", mismatches=mismatches,
                              steps=2, timing_collected=False)
                save()
                return

            def run(functions, *, measure):
                initial.restore()
                torch.hpu.synchronize()
                output = seed.clone()
                start = torch.hpu.Event(enable_timing=True) if measure else None
                end = torch.hpu.Event(enable_timing=True) if measure else None
                compile_before = dict(compilation_metric.stats())["TotalNumber"] if measure else None
                fallback_before = dict(fallback_metric.stats())["TotalNumber"] if measure else None
                stamps = []
                for step in range(total):
                    if measure and step == args.warm_steps:
                        if args.steady_drain_boundary:
                            # Preserve the continuous state, but exclude queued
                            # unscored warm work from this host timing boundary.
                            torch.hpu.synchronize()
                            dist.barrier(group=cpu_group)
                            torch.hpu.synchronize()
                        started = time.perf_counter_ns()
                        start.record()
                        stamps.append(time.perf_counter_ns())
                    position = positions[step]
                    for function in functions:
                        output = function(output, position, post, comb)
                    if measure and step >= args.warm_steps:
                        stamps.append(time.perf_counter_ns())
                if measure:
                    end.record()
                    end.synchronize()
                    if args.steady_drain_boundary:
                        torch.hpu.synchronize()
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    metric = dict(wall_ms_per_step=elapsed / args.steps,
                                  event_ms_per_step=start.elapsed_time(end) / args.steps,
                                  submission_intervals_ms=[(b - a) / 1e6 for a, b in
                                                           zip(stamps[:-1], stamps[1:], strict=True)],
                                  final_drain_ms=(time.perf_counter_ns() - stamps[-1]) / 1e6)
                    metric["new_graph_compilations"] = dict(compilation_metric.stats())["TotalNumber"] - compile_before
                    metric["new_cpu_fallbacks"] = dict(fallback_metric.stats())["TotalNumber"] - fallback_before
                    assert metric["new_graph_compilations"] == metric["new_cpu_fallbacks"] == 0, metric
                else:
                    torch.hpu.synchronize()
                    metric = None
                actual = output.cpu()
                assert torch.isfinite(actual).all(), "Nonfinite residual feedback"
                return actual, fingerprints(_PagedSnapshot(program, touched, states)), metric

            # Compile and compare the complete producer/consumer dependencies
            # before the matched continuous timing boundary.
            expected, wanted, _ = run(parent, measure=False)
            actual, observed, _ = run(candidate, measure=False)
            if not torch.equal(actual, expected) or observed != wanted:
                # Isolated diagnostic only: preserve the failure above, then
                # locate the earliest real consumer with synchronous readback.
                # If this probe passes, it does not override the unpaced failure.
                trace = []
                first_difference = None
                for arm, functions in (("parent", parent), ("candidate", candidate)):
                    initial.restore()
                    torch.hpu.synchronize()
                    output = seed.clone()
                    stop = False
                    for step in range(total):
                        for ordinal, function in enumerate(functions):
                            output = function(output, positions[step], post, comb)
                            attention = program.layers[ordinal].attention
                            value = output.cpu()
                            selected = attention.selection.indices[:1].cpu()
                            pool = program.shared.candidate_pool[:1].cpu()
                            swa = attention.swa[(16384 + step) % 256].cpu()
                            if arm == "parent":
                                trace.append((value, selected, pool, swa))
                                continue
                            reference = trace[step * len(functions) + ordinal]
                            equal = [torch.equal(a, b) for a, b in
                                     zip((value, selected, pool, swa), reference, strict=True)]
                            mismatch = torch.tensor([not all(equal)], dtype=torch.int32)
                            dist.all_reduce(mismatch, op=dist.ReduceOp.MAX, group=cpu_group)
                            if mismatch.item():
                                delta = (value.float() - reference[0].float()).abs()
                                first_difference = dict(step=step, layer=layers[ordinal],
                                                        output_equal=equal[0], selected_equal=equal[1],
                                                        candidate_pool_equal=equal[2], swa_equal=equal[3],
                                                        max_abs=float(delta.max()))
                                torch.save(dict(reference=reference, actual=(value, selected, pool, swa)),
                                           root / f"first-difference-rank{rank}.pt")
                                stop = True
                                break
                        if stop:
                            break
                report["synchronous_first_difference"] = first_difference
                report["synchronous_probe_did_not_override_unpaced_failure"] = True
                save()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert observed == wanted, "Canonical attention state differs"
            copies = [None] * 4
            dist.all_gather_object(copies, hashlib.sha256(expected.view(torch.uint8).numpy().tobytes()).hexdigest(),
                                   group=cpu_group)
            assert all(value == copies[0] for value in copies), "AllReduce downstream consumers differ"
            torch.save(expected, root / f"continuous-reference-output-rank{rank}.pt")
            report["checks"].append(dict(scope="continuous64+fiveattention+hc_post", exact_output=True,
                                          exact_state=True, equal_ranks=True))
            if args.reference_timing:
                archive = json.loads((args.reference_timing / f"selection-chain-rank{rank}.json").read_text())
                assert archive["status"] == "exact_complete_attention_consumer_chain_measured"
                for field in ("warm_steps", "steps", "physical_tokens", "position_contract", "steady_drain_boundary"):
                    assert archive[field] == report[field], (field, archive[field], report[field])
                if not args.logical_mla:
                    for name in ("deepseek_v41_tp4_selection.py", "deepseek_v41_paged_attention.py"):
                        relative = Path("vllm_gaudi/ops") / name
                        archived = args.reference_timing / "source" / relative
                        assert (Path(__file__).resolve().parents[1] / relative).read_bytes() == archived.read_bytes()
                else:
                    # Address construction and compact immutable offsets are
                    # the intended changes. Retain the original score/select,
                    # projection, state writer and output arithmetic contracts.
                    methods = {"_compress", "_prepare_index_queries", "_scores", "_stream_topk", "_select",
                               "_merge_topk", "_finish_output", "project_query", "project_kv",
                               "project_output", "project_output_consumer"}
                    if args.index_mirror:
                        methods.difference_update(("_compress", "_scores", "_stream_topk"))
                    def math_contract(path):
                        nodes = ast.walk(ast.parse(path.read_text()))
                        return [ast.dump(node, include_attributes=False) for node in nodes
                                if isinstance(node, ast.FunctionDef) and node.name in methods]
                    relative = Path("vllm_gaudi/ops/deepseek_v41_paged_attention.py")
                    assert math_contract(Path(__file__).resolve().parents[1] / relative) == \
                        math_contract(args.reference_timing / "source" / relative)
                report["timings"]["parent"] = archive["timings"][args.reference_arm]
                report["parent_timing_reused_from"] = str(args.reference_timing)
                report["parent_timing_reused_arm"] = args.reference_arm
                timed_arms = (("candidate", candidate),)
            else:
                timed_arms = (("parent", parent), ("candidate", candidate))
            for name, functions in timed_arms:
                actual, observed, metric = run(functions, measure=True)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert observed == wanted, (name, "Canonical timed state differs")
                report["timings"][name] = metric
                save()
            report["continuous_state_fingerprints"] = wanted
            report["no_hot_compilation"] = True
            # Wider normal buckets and a source-prefix boundary retain the
            # same shared cache and the candidate publication dependency.
            for count, position, bound in ((2, 16448, 20480), (6, 16448, 20480),
                                            (1, 20480, 24576), (1, 16448, 20480)):
                for block in (*program.layers, *references):
                    block.attention.set_decode_visible_tokens(bound)
                local_pos = torch.arange(position, position + count, dtype=torch.int32, device="hpu")
                local_seed = seed.expand(count, -1, -1).contiguous()
                local_post, local_comb = post.expand(count, -1).contiguous(), comb.expand(count, -1, -1).contiguous()
                outputs, state_hashes = [], []
                for functions in (parent, candidate):
                    initial.restore()
                    output = local_seed
                    for function in functions:
                        output = function(output, local_pos, local_post, local_comb)
                    outputs.append(output.cpu())
                    state_hashes.append(fingerprints(_PagedSnapshot(program, touched, states)))
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                assert state_hashes[0] == state_hashes[1]
                report["checks"].append(dict(tokens=count, position=position, bound=bound,
                                              exact_output=True, exact_state=True))
                save()
            report["status"] = "exact_complete_attention_consumer_chain_measured"
            save()
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        save()
        raise
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
