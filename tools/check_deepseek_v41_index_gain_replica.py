# SPDX-License-Identifier: Apache-2.0
"""Checkpoint index projections -> native peer exchange -> 2048-row score tile.

The default compares cold head-gain replication; --query-replica also tests
replicating the query weights, with the qualified gain replica held fixed.
Five embedding-derived fixtures qualify query/gain/score bytes on all ranks.
"""
import argparse
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-replica', action='store_true', help='Hold gain replica fixed; compare sharded query+peer against cold full query weights')
    parser.add_argument('--native-query-path', action='store_true', help='Use the production native RoPE table/codec contract in both arms')
    parser.add_argument('--fused-query-codec', action='store_true', help='Hold gain replica and query peer fixed; compare native RoPE+FP4 with the fused codec')
    parser.add_argument('--shared-query-codec', action='store_true', help='Use the qualified fused codec in both query-replication arms')
    parser.add_argument('--selection-chain', action='store_true', help='Native query producer -> full 16K score -> threshold/emit -> MLA consumer')
    parser.add_argument('--wide-reindex', action='store_true', help='Hold selection/reference variant fixed; compare eight 2K score calls with one complete 16K call')
    parser.add_argument('--wide-score-parent', action='store_true', help='Hold qualified complete-pool scoring fixed in both selection arms')
    parser.add_argument('--threshold-variant', type=int, choices=(1,), default=1)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--chain-repeats', type=int, default=64)
    args = parser.parse_args()
    if args.wide_reindex and not args.selection_chain:
        parser.error('--wide-reindex requires --selection-chain')
    if args.wide_score_parent and (not args.selection_chain or args.wide_reindex):
        parser.error('--wide-score-parent requires --selection-chain and excludes --wide-reindex')
    if args.selection_chain and (not args.native_query_path or args.query_replica or args.fused_query_codec or args.shared_query_codec):
        parser.error('--selection-chain requires native query path and a fixed shared query/gain parent')
    if args.fused_query_codec and (not args.native_query_path or args.query_replica):
        parser.error('--fused-query-codec requires --native-query-path and excludes --query-replica')
    if args.shared_query_codec and (not args.native_query_path or not args.query_replica or args.fused_query_codec):
        parser.error('--shared-query-codec requires the native query-replication comparison')
    if args.native_query_path:
        os.environ['VLLM_HPU_DSV41_NATIVE_KV_PACK'] = '1'
        os.environ['VLLM_HPU_DSV41_QUANT_ROUNDTRIP'] = '1'
    rank = int(os.environ['LOCAL_RANK'])
    modules = os.environ['HABANA_VISIBLE_MODULES'].split(',')
    tp = len(modules)
    os.environ['HLS_MODULE_ID'] = modules[rank]
    os.environ['VLLM_HPU_DSV41_TP_MHC_OVERLAP'] = '0'
    if os.environ.get('DSV41_MICRO_RANK_CPUS'):
        os.sched_setaffinity(0, json.loads(os.environ['DSV41_MICRO_RANK_CPUS'])[rank])
    from tools.deepseek_v41_physical_audit import prepare_physical_audit
    prepare_physical_audit(args, rank)
    import torch
    import torch.nn.functional as F
    import habana_frameworks.torch.core  # noqa: F401
    from safetensors import safe_open
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime, _resolve_runtime
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_math import rms_norm, fp4_roundtrip, quantize_activation, rotary_table, apply_rope
    from vllm_gaudi.ops.deepseek_v41_index_mirror import mirror_index_tile
    from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
    from tools.deepseek_v41_micro_replay import RecipeRecorder
    from tools.deepseek_v41_resident_ab import graph_compilation_count, wait_for_loading
    torch.set_num_threads(1)
    torch.hpu.set_device(rank)
    root = args.output.resolve()
    directory = root / f'rank{rank}'
    directory.mkdir(parents=True, exist_ok=True)
    config = set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp)))
    config.__enter__()
    init_distributed_environment(world_size=tp, rank=rank, distributed_init_method='env://', local_rank=rank, backend='hccl')
    initialize_model_parallel(tensor_model_parallel_size=tp)
    initialize_tp2_fused_ar_norm_runtime()
    bridge, backend, _ = _resolve_runtime()
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    recorder = RecipeRecorder(directory, backend=backend, joint_only=True)
    graph_compilation_count()
    shards = [PreparedV41Shard(args.prepared, 0, r) for r in range(tp)]
    shard = shards[rank]
    wq = shard.dense('layers.24.attn.indexer.wq_b.weight', 'hpu')
    global_wq = torch.cat([s.dense('layers.24.attn.indexer.wq_b.weight', 'cpu') for s in shards]).to('hpu') if args.query_replica else None
    local_wp = shard.tensor('layers.24.attn.indexer.weights_proj.weight', 'hpu')
    global_wp = torch.cat([s.tensor('layers.24.attn.indexer.weights_proj.weight', 'cpu') for s in shards]).to('hpu')
    local_heads = local_wp.shape[0]
    total_heads = global_wp.shape[0]
    norm = shard.tensor('layers.24.attn.q_norm.weight', 'hpu')
    wk = shard.tensor('layers.20.attn.indexer.wk.weight', 'cpu')
    knorm = shard.tensor('layers.20.attn.indexer.k_norm.weight', 'cpu')
    with safe_open(args.prepared/'pp0-tp0.safetensors', framework='pt', device='cpu') as checkpoint:
        fixtures = [checkpoint.get_slice('embed.weight')[token:token+1].clone() for token in (17, 41, 128, 512, 1024)]
        key_embeddings = checkpoint.get_slice('embed.weight')[16:16400, :512].clone() if args.selection_chain else None
    keys = fp4_roundtrip(rms_norm(F.linear(key_embeddings if args.selection_chain else torch.cat(fixtures)[:, :512].contiguous(), wk), knorm, 1e-20), 32)
    if args.selection_chain:
        keys = keys.repeat((32, 1))
    keys = (keys if args.selection_chain else keys.repeat((4096, 1))[:16384]).contiguous().to('hpu')
    cfg = json.loads((args.prepared/'config.json').read_text())['text_config']
    scaling = cfg['rope_scaling']
    phase = rotary_table(64, 32768, cfg['compress_rope_theta'], scaling['original_max_position_embeddings'],
                         scaling['factor'], scaling['beta_fast'], scaling['beta_slow']).to('hpu')
    if args.native_query_path:
        # Native input is [cos32,sin32], not the generic interleaved pair table.
        phase = torch.cat((phase[..., 0], phase[..., 1]), -1).contiguous()
    x = fixtures[0].to('hpu')
    positions = torch.tensor([16384], dtype=torch.int32, device='hpu')
    rows = torch.arange(2048, dtype=torch.int32, device='hpu').reshape(1, -1)

    def produce(value, pos, wq, wp, norm, phase, *, replicated):
        qr = rms_norm(value[:, :1280].contiguous(), norm, 1e-20)
        query = F.linear(quantize_activation(qr), wq).reshape(1, wq.shape[0] // 128, 128)
        if args.selection_chain or args.shared_query_codec or (args.fused_query_codec and replicated):
            query = torch.ops.custom_op.custom_deepseek_v41_index_query_rope_fp4_bf16_gaudi2(query, pos, phase)
        else:
            roped = (torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(query, pos, phase)
                     if args.native_query_path else apply_rope(query, pos, phase))
            query = fp4_roundtrip(roped, 32)
        query = query.reshape(1, -1).contiguous()
        gains = F.linear(value, wp) * (128**-.5 * total_heads**-.5)
        return query, gains.contiguous() if args.selection_chain or replicated or args.query_replica or args.fused_query_codec else F.pad(gains, (0, 128-local_heads)).contiguous()

    if args.selection_chain:
        from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa
        # Real checkpoint-derived KV values; retain the full physical cache size.
        wk_main = shard.dense('layers.20.attn.wkv.weight', 'cpu')
        norm_main = shard.tensor('layers.20.attn.kv_norm.weight', 'cpu')
        kv_rows = rms_norm(F.linear(torch.cat(fixtures), wk_main), norm_main, 1e-20)
        packed = pack_fp4(kv_rows, 16)
        main_cache = packed.repeat((104858, 1))[:524288].contiguous().to('hpu')
        swa = pack_swa(kv_rows).repeat((52, 1))[:256].contiguous().to('hpu')
        pages = torch.arange(8192, dtype=torch.int32, device='hpu')
        candidates = torch.arange(2048, dtype=torch.int32, device='hpu').reshape(1, -1)
        sink = shard.tensor('layers.24.attn.attn_sink', 'hpu')
        attn_scale = torch.tensor([512**-.5], device='hpu')
        lengths = torch.tensor([640], dtype=torch.int32, device='hpu')

    def consume(query, gains, keys, positions, rows, *, replicated):
        query = query.reshape(1, total_heads, 128)
        gains = gains.reshape(1, total_heads) if args.selection_chain or replicated or args.query_replica or args.fused_query_codec else gains.reshape(tp, 128)[:, :local_heads].reshape(1, total_heads).contiguous()
        if args.selection_chain:
            logical = torch.where(candidates.unsqueeze(-1) >= 0,
                                  candidates.unsqueeze(-1) * 8 + torch.arange(8, dtype=torch.int32, device=query.device),
                                  -1).flatten(1)
            if args.wide_score_parent or (args.wide_reindex and replicated):
                scores = mirror_index_tile(query, gains, keys, positions, logical, 1, local_heads)
            else:
                scores = torch.cat([mirror_index_tile(query, gains, keys, positions, logical[:, tile*2048:(tile+1)*2048],
                                                      1, local_heads) for tile in range(8)], -1).contiguous()
            stats = torch.ops.custom_op.custom_deepseek_v41_index_threshold_gaudi2(
                scores, positions, 1, 1, 0, args.threshold_variant if replicated and not args.wide_reindex else 0)
            selected = torch.ops.custom_op.custom_deepseek_v41_index_emit_gaudi2(
                scores, positions, candidates, stats, 1, 1, 0)
            # First real selected-KV/QK/softmax/PV consumer, including its actual waits.
            aq = query[:, :sink.numel()].repeat(1, 1, 4).contiguous()
            output, decoded, mask = torch.ops.custom_op.custom_deepseek_v41_main_publish_native_codec_mla_gaudi2(
                aq, swa, main_cache, selected, positions, pages, sink, attn_scale, lengths, 1)
            return scores, selected, output, decoded, mask
        scores = mirror_index_tile(query, gains, keys, positions, rows, 2, local_heads)
        return query, gains, scores

    compile_fn = lambda fn: torch.compile(fn, backend=make_backend(static_int32=True, static_factories=True), fullgraph=True, dynamic=False)
    producers = [compile_fn(lambda *values, replica=replica: produce(*values, replicated=replica)) for replica in (False, True)]
    consumers = [compile_fn(lambda *values, replica=replica: consume(*values, replicated=replica)) for replica in (False, True)]
    plans, graphs, outputs, externals = [], [], [], []
    for replica, producer, consumer in zip((False, True), producers, consumers, strict=True):
        plan = bridge.PreparedGroupPlan()
        slots, tensors, external = {}, {}, []

        def slot(value, incoming):
            key = ((value.data_ptr(), tuple(value.shape), value.stride(), value.dtype)
                   if isinstance(value, torch.Tensor) else (type(value), value))
            if key not in slots:
                if isinstance(value, torch.Tensor) and value.is_contiguous():
                    for known, tensor in tensors.items():
                        if tensor.data_ptr() == value.data_ptr() and tensor.dtype == value.dtype and tensor.numel() == value.numel() and tensor.is_contiguous():
                            slots[key] = plan.add_reshape_view(slots[known], list(value.shape))
                            return slots[key]
                slots[key] = plan.add_slot(value, incoming)
                if isinstance(value, torch.Tensor):
                    tensors[key] = value
                if incoming:
                    external.append(value)
            return slots[key]

        def compute(fn, operands):
            fn(*operands)
            torch.hpu.synchronize()
            recorder.calls = []
            result = fn(*operands)
            torch.hpu.synchronize()
            calls, recorder.calls = recorder.calls, None
            if not calls:
                raise RuntimeError('No native recipe recorded')
            for recipe, inputs, out in calls:
                plan.add_compute(recipe, [slot(v, True) for v in inputs], [slot(v, False) for v in out])
            return result

        query, gains = compute(producer, (x, positions, global_wq if args.query_replica and replica else wq,
                                            global_wp if args.selection_chain or replica or args.query_replica or args.fused_query_codec else local_wp, norm, phase))
        if args.query_replica and replica:
            global_query = query
        else:
            global_query = torch.ops.vllm_gaudi.tp_peer_allgather(query, tp)
            torch.hpu.synchronize()
            plan.add_all_gather(slot(query, False), slot(global_query, False))
        if not args.selection_chain and not replica and not args.query_replica and not args.fused_query_codec:
            global_gains = torch.ops.vllm_gaudi.tp_peer_allgather(gains, tp)
            torch.hpu.synchronize()
            plan.add_all_gather(slot(gains, False), slot(global_gains, False))
        else:
            global_gains = gains
        out = compute(consumer, (global_query, global_gains, keys, positions, rows))
        plan.prepare(backend, [slot(v, False) for v in out])
        graph = bridge.NativeDecodeGraph()
        points = (0 if replica else 1) if args.query_replica else (1 if args.selection_chain or args.fused_query_codec or replica else 2)
        graph.configure_topology(args.chain_repeats, args.chain_repeats*points, False)
        graph.configure_dependency_policy(False)
        graph.capture([plan]*args.chain_repeats, [external]*args.chain_repeats)
        graph.instantiate()
        graph.bind_dynamic_inputs([x, positions])
        plans.append(plan); graphs.append(graph); outputs.append(out); externals.append(external)
    checks = []
    for index, fixture in enumerate(fixtures):
        x.copy_(fixture.to('hpu')); positions.fill_(16383+index if args.selection_chain else 16384+index)
        if args.selection_chain:
            pool = (torch.arange(2048, dtype=torch.int32)*(index+1)) % (keys.shape[0]//8)
            pool[:4] = torch.tensor([-1,-1000,keys.shape[0]//8,keys.shape[0]//4],dtype=torch.int32)
            candidates.copy_(pool.reshape(1,-1).to('hpu'))
        torch.hpu.synchronize()
        values = []
        for graph, out in zip(graphs, outputs, strict=True):
            graph.replay_fixed_with_completion().synchronize()
            values.append([v.cpu() for v in out])
        exact = [torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a,b in zip(*values, strict=True)]
        checks.append(dict(fixture=index, query_gain_score_exact=exact))
        agreed = torch.tensor([int(all(exact))], dtype=torch.int32, device='hpu')
        torch.distributed.all_reduce(agreed, op=torch.distributed.ReduceOp.MIN)
        (directory/'checks.json').write_text(json.dumps(checks, indent=2))
        if not bool(agreed.cpu().item()):
            raise RuntimeError('Replicated gain projection changed query/gain/score bytes')
    if args.selection_chain:
        pool = torch.arange(2048, dtype=torch.int32)
        candidates.copy_(pool.reshape(1,-1).to('hpu')); positions.fill_(16383); x.copy_(fixtures[0].to('hpu'))
        torch.hpu.synchronize()
    for graph in graphs:
        for _ in range(8): graph.replay_fixed_with_completion().synchronize()
    before = graph_compilation_count()
    periods = []
    tickets, events = [], []
    for label in 'ABABAB':
        wait_for_loading(directory, rank, torch.distributed)
        graph = graphs[label == 'B']
        events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)) for _ in range(args.steps)]
        torch.distributed.barrier();tickets=[]
        for begin, end in events:
            begin.record();tickets.append(graph.replay_fixed_with_completion());end.record()
        torch.hpu.synchronize()
        local = [begin.elapsed_time(end)/args.chain_repeats for begin,end in events]
        ranks = [None]*tp
        torch.distributed.all_gather_object(ranks, local)
        periods.append(dict(arm=label, rank_device_ms=ranks, median_ms=max(statistics.median(v) for v in ranks)))
        if rank == 0:print(json.dumps(dict(arm=label, median_ms=periods[-1]['median_ms'])), flush=True)
    if graph_compilation_count() != before:raise RuntimeError('Hot compilation')
    savings = [a['median_ms']-b['median_ms'] for a,b in zip(periods[::2], periods[1::2], strict=True)]
    result = dict(status='completed', checks=checks, periods=periods, round_savings_ms=savings,
                  saving_ms_per_index_layer=statistics.median(savings), three_consistent_rounds=all(v>0 for v in savings),
                  peer_points_per_iteration=[1,0] if args.query_replica else [1,1] if args.selection_chain or args.fused_query_codec else [2,1],
                  wide_reindex=args.wide_reindex, wide_score_parent=args.wide_score_parent,
                  selection_chain=args.selection_chain, threshold_variant=args.threshold_variant if args.selection_chain else None,
                  native_query_path=args.native_query_path, fused_query_codec=args.fused_query_codec,
                  shared_query_codec=args.shared_query_codec,
                  fixture_scope=__doc__, formal_gain=False)
    (directory/'result.json').write_text(json.dumps(result, indent=2))
    if rank == 0:(root/'result.json').write_text(json.dumps(result, indent=2))
    for graph, plan in zip(graphs, plans, strict=True):graph.close();plan.invalidate()
    graphs.clear();plans.clear();outputs.clear();externals.clear();tickets.clear();events.clear()
    graph=plan=slot=compute=None
    recorder.calls=None
    producers.clear();consumers.clear();producer=consumer=None
    torch.hpu.synchronize()
    for attribute in ('_vllm_gaudi_tp2_fused_ar_norm_runtime','_vllm_gaudi_tp4_allreduce_runtime'):
        if hasattr(torch,attribute):delattr(torch,attribute)
    backend=None
    import gc
    gc.collect()
    destroy_model_parallel();destroy_distributed_environment();config.__exit__(None,None,None)


if __name__ == '__main__':
    import torch
    with torch.inference_mode():main()
