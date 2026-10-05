# SPDX-License-Identifier: Apache-2.0
"""Checkpoint MoE SAT -> native peer exchange -> mHC -> FFN norm/quant A/B.

Inputs are derived from five checkpoint embedding rows, not captured attention
activations. This component screen does not replace the real16 state gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--sidecar',type=Path,required=True)
    parser.add_argument('--disable-tiny-bgemm',action='store_true',help='Cold compiler screen; the legacy setting has not fixed Gaudi2 batch slicing')
    parser.add_argument('--candidate-operator',choices=('paired','silu','groups3'),default='paired')
    parser.add_argument('--chain-repeats',type=int,default=16)
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 4, 14, 19])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--retirement-only', action='store_true',
                        help='Reuse measured A/B; check teardown without timing')
    args = parser.parse_args()
    candidate_guid=('custom_deepseek_v41_expert_token_wide3_sat_fp8_gaudi2' if args.candidate_operator=='groups3' else
                    'custom_deepseek_v41_expert_'+('paired_decode' if args.candidate_operator=='paired' else 'silu_decode')+'_fp8_gaudi2')
    rank = int(os.environ['LOCAL_RANK'])
    modules = os.environ['HABANA_VISIBLE_MODULES'].split(',')
    os.environ['HLS_MODULE_ID'] = modules[rank]
    if os.environ.get('DSV41_MICRO_RANK_CPUS'):
        os.sched_setaffinity(0, json.loads(os.environ['DSV41_MICRO_RANK_CPUS'])[rank])
    from tools.deepseek_v41_physical_audit import prepare_physical_audit
    prepare_physical_audit(args, rank)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from safetensors import safe_open
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime, _resolve_runtime
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_math import hc_pre
    from tools.deepseek_v41_micro_replay import RecipeRecorder
    from tools.deepseek_v41_resident_ab import graph_compilation_count, wait_for_loading

    torch.set_num_threads(1)
    torch.hpu.set_device(rank)
    root = args.output.resolve()
    directory = root / f'rank{rank}'
    directory.mkdir(parents=True, exist_ok=True)
    # Reserve result inodes before acquiring device state; update in place later.
    (directory/'result.json').write_text('{"status":"started"}\n')
    if rank == 0:
        (root/'result.json').write_text('{"status":"started"}\n')
    config = set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4)))
    config.__enter__()
    init_distributed_environment(world_size=4, rank=rank, distributed_init_method='env://',
                                 local_rank=rank, backend='hccl')
    initialize_model_parallel(tensor_model_parallel_size=4)
    initialize_tp2_fused_ar_norm_runtime()
    bridge, backend, _ = _resolve_runtime()
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    recorder = RecipeRecorder(directory, backend=backend, joint_only=True)
    graph_compilation_count()
    shard = PreparedV41Shard(args.prepared, 0, rank)
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    sidecar=DenseFP8Sidecar(args.sidecar,shard)
    import numpy as np
    from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert, saturated_decode_eligible
    from vllm_gaudi.ops.deepseek_v41_fp8 import read_expert
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    lookup = mxfp4_bf16_lut('hpu')
    weights = []
    for layer in args.layers:
        row = []
        for projection in ('w13', 'w2'):
            prefix = f'layers.{layer}.ffn.experts.{projection}'
            qs, ss = [shard.catalog[prefix+suffix] for suffix in ('_q16', '_s16')]
            experts, blocks, stream = qs.shape
            q = torch.empty((experts, blocks//2, stream*2), dtype=torch.int16, device='hpu')
            scales = torch.empty((experts, blocks//2, stream//8+128), dtype=torch.int16, device='hpu')
            channels = torch.empty((experts, blocks//2, 256), dtype=torch.bfloat16, device='hpu')
            for expert in range(12):
                a,b,c,_ = prepare_expert(read_expert(qs, expert), read_expert(ss, expert), compact_scales=True)
                config_values = json.loads((args.prepared/"config.json").read_text())["text_config"]
                active_k = config_values["moe_intermediate_size"] // 4 if projection == "w2" else stream//32
                if np.any(a[:, active_k * 64:]) or not saturated_decode_eligible(b, active_k=active_k):
                    raise ValueError(f'Unqualified SAT scales layer={layer} expert={expert} projection={projection}')
                q[expert].copy_(torch.from_numpy(a))
                scales[expert].copy_(torch.from_numpy(b))
                channels[expert].copy_(torch.from_numpy(c.view('<i2')).view(torch.bfloat16))
            row.append((q, scales, channels))
        projections={}
        for projection in ('w1','w3','w2'):
            key=f'layers.{layer}.ffn.shared_experts.{projection}'
            projections[projection]=(sidecar.tensor(key+'.weight','cpu'),sidecar.tensor(key+'.channel_scale','cpu'))
        width=projections['w1'][0].shape[0];padded=(width+127)//128*128
        pad=lambda w:torch.cat((w,torch.zeros((padded-width,w.shape[1]),dtype=w.dtype)),0)
        gate_up=torch.cat((pad(projections['w1'][0]),pad(projections['w3'][0])),0).to('hpu')
        channel=torch.cat((torch.nn.functional.pad(projections['w1'][1],(0,padded-width),value=1),
            torch.nn.functional.pad(projections['w3'][1],(0,padded-width),value=1)),1).bfloat16().reshape(1,padded*2//256,256).to('hpu')
        down=projections['w2'][0];down=torch.cat((down,torch.zeros((down.shape[0],padded-width),dtype=down.dtype)),1).to('hpu')
        down_scale=projections['w2'][1].to('hpu')
        weights.append((row[0][0],row[1][0],row[0][1],row[1][1],lookup,row[0][2],row[1][2],
                        shard.tensor(f'layers.{layer}.ffn_norm.weight','hpu'),gate_up,channel,down,down_scale))
    norm = weights[0][7]
    control, scale, base = [shard.tensor(f'layers.0.hc_ffn_{name}', 'cpu') for name in ('fn', 'scale', 'base')]
    with safe_open(args.prepared/'pp0-tp0.safetensors', framework='pt', device='cpu') as checkpoint:
        embeddings = [checkpoint.get_slice('embed.weight')[token:token+1].clone()
                      for token in (17, 41, 128, 512, 1024)]
    fixtures = []
    for i, embedding in enumerate(embeddings):
        residual = embedding.unsqueeze(1).expand(-1, 4, -1).contiguous()
        initial_pre = torch.tensor([[1., 0., 0., 0.]])
        collapsed, pre, post, comb = hc_pre(residual, initial_pre, control, scale, base)
        ids = (torch.arange(6, dtype=torch.int32).reshape(1,6)+i)%12
        route = torch.tensor([[.25]*6], dtype=torch.float32)
        fixtures.append((collapsed, residual, post, comb, pre, ids, route))
    x, residual, post, comb, pre, ids, route = [tensor.to('hpu') for tensor in fixtures[0]]
    from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
    compiled = lambda fn, candidate=False: torch.compile(fn, backend=make_backend(static_int32=True,static_factories=True,split_mhc=True,compiler_config={'SYN_SRAM_BGEMM_SLICER_MULTIPLE_TINY_GEMMS_PER_SLICE':'false'} if candidate and args.disable_tiny_bgemm else None), fullgraph=True, dynamic=False)
    def produce(row, ids, route, q13,q2,s13,s2,lut,c13,c2,norm_weight,shared_weight,shared_channel,shared_down,shared_scale, *, sat):
        normalized, quantized, activation_scale = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
            row, norm_weight, 1e-20)
        q,sx=torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(normalized)
        product=torch.ops.hpu.fp8_gemm_v2(q,False,shared_weight,True,None,torch.float32,None,None,None,False)
        shared_ids=torch.zeros((1,1),dtype=torch.int32,device=row.device)
        shared_route=torch.ones((1,1),dtype=torch.float32,device=row.device)
        middle,shared_sx=torch.ops.custom_op.custom_deepseek_v41_shared_silu_quant_gaudi2(
            product.reshape(1,1,-1),shared_ids,sx,shared_channel,shared_route)
        shared=torch.ops.hpu.fp8_gemm_v2(middle.reshape(1,-1),False,shared_down,True,None,torch.bfloat16,
            shared_sx.reshape(1,1),shared_scale,None,False)
        op=(getattr(torch.ops.custom_op,'custom_deepseek_v41_expert_n256_moe_'+('paired_decode_shared' if args.candidate_operator=='paired' else 'silu_decode_shared' if args.candidate_operator=='silu' else 'two_group_w2_sat_shared')+'_fp8_gaudi2') if sat else
            torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_prequant_direct_finalize_shared_prefetch_w2_fp8_gaudi2)
        return op(normalized,ids,route,q13,q2,s13,s2,lut,c13,c2,quantized,activation_scale,shared,True)
    producers = [compiled(lambda row, ids, route, *weights, sat=sat:
                          produce(row,ids,route,*weights,sat=sat),candidate=sat) for sat in (False, True)]
    def consume(peers, r,p,c,n,norm_weight):
        value = peers[0].float()
        for index in range(1,4):
            value = value + peers[index].float()
        updated,collapsed = torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
            value.bfloat16(),r,p,c,n)
        normalized,quantized,activation_scale = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
            collapsed,norm_weight,1e-20)
        return updated,collapsed,normalized,quantized,activation_scale
    consumer = compiled(consume)
    # Compile both production consumers before native-plan allocation. Reject
    # a physically larger graph immediately rather than instantiating it.
    from tools.audit_deepseek_v41_physical_nodes import audit
    for producer in producers:
        producer(x,ids,route,*weights[0])
        torch.hpu.synchronize()
    physical=[audit(p) for p in (root/'graphs'/f'rank{rank}').rglob('*PostGraph-symbol.pbtxt')]
    reference=[p for p in physical if p['operations'].get('custom_deepseek_v41_expert_n256_fp8_gaudi2')]
    candidate=[p for p in physical if p['operations'].get(candidate_guid)]
    (directory/'physical_nodes.json').write_text(json.dumps(physical,indent=2)+'\n')
    local_gate = bool(reference and candidate and
                      max(p['physical_nodes'] for p in candidate) < max(p['physical_nodes'] for p in reference))
    # All ranks leave together if one has missing evidence or a physical regression.
    # Otherwise successful ranks would enter native peer exchange indefinitely.
    gate = torch.tensor([int(local_gate)], dtype=torch.int32, device='hpu')
    torch.distributed.all_reduce(gate, op=torch.distributed.ReduceOp.MIN)
    if not bool(gate.cpu().item()):
        raise RuntimeError(f'Physical gate failed: reference={len(reference)}, candidate={len(candidate)}, '
                           f'local_reduction={local_gate}; no native capture or timed run')
    plans, graphs, visible, externals = [], [], [], []
    for producer in producers:
        plan = bridge.PreparedGroupPlan()
        slots, tensors, external = {}, {}, []

        def slot(value, is_input, slots=slots, tensors=tensors, external=external, plan=plan):
            key = ((value.data_ptr(), tuple(value.shape), value.stride(), value.dtype)
                   if isinstance(value, torch.Tensor) else (type(value), value))
            if key not in slots:
                if isinstance(value, torch.Tensor) and value.is_contiguous():
                    for known, tensor in tensors.items():
                        if (tensor.data_ptr() == value.data_ptr() and tensor.dtype == value.dtype
                                and tensor.numel() == value.numel() and tensor.is_contiguous()):
                            slots[key] = plan.add_reshape_view(slots[known], list(value.shape))
                            return slots[key]
                slots[key] = plan.add_slot(value, is_input)
                if isinstance(value, torch.Tensor):
                    tensors[key] = value
                if is_input:
                    external.append(value)
            return slots[key]

        def compute(fn, operands, slot=slot, plan=plan):
            fn(*operands)
            torch.hpu.synchronize()
            recorder.calls = []
            result = fn(*operands)
            torch.hpu.synchronize()
            calls, recorder.calls = recorder.calls, None
            if not calls:
                raise RuntimeError('Missing physical recipe recording')
            for recipe, inputs, outputs in calls:
                plan.add_compute(recipe, [slot(v, True) for v in inputs], [slot(v, False) for v in outputs])
            return result

        outputs = []
        for layer_weights in weights:
            partial = compute(producer, (x, ids, route, *layer_weights))
            peers = torch.ops.vllm_gaudi.tp_peer_allgather(partial, 4)
            torch.hpu.synchronize()
            plan.add_all_gather(slot(partial, False), slot(peers, False))
            outputs.extend(compute(consumer, (peers.reshape(4, 1, 5120), residual, post, comb, pre, norm)))
        plan.prepare(backend, [slot(value, False) for value in outputs])
        graph = bridge.NativeDecodeGraph()
        graph.configure_topology(args.chain_repeats, args.chain_repeats*len(args.layers), False)
        graph.configure_dependency_policy(False)
        graph.capture([plan]*args.chain_repeats, [external]*args.chain_repeats)
        graph.instantiate()
        graph.bind_dynamic_inputs([x, residual, post, comb, pre, ids, route])
        plans.append(plan)
        graphs.append(graph)
        visible.append(outputs)
        externals.append(external)
    checks = []
    for index, fixture in enumerate(fixtures):
        for destination, value in zip((x, residual, post, comb, pre, ids, route), fixture, strict=True):
            destination.copy_(value.to('hpu'))
        torch.hpu.synchronize()
        observed = []
        for graph, outputs in zip(graphs, visible, strict=True):
            graph.replay_fixed_with_completion().synchronize()
            observed.append([value.cpu() for value in outputs])
        exact = [torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                 for a, b in zip(*observed, strict=True)]
        if not all(exact):
            torch.save(dict(fixture=fixture, observed=observed), directory/'failure.pt')
            raise RuntimeError(f'Paired W13/W2 chain differs for input {index}: {exact}')
        checks.append(dict(input=index, all_outputs_exact=True))
    (directory/'result.json').write_text(json.dumps(dict(status='correctness_passed', checks=checks))+'\n')
    # A larger final compiler graph cannot qualify as kernel-count fusion.
    from tools.audit_deepseek_v41_physical_nodes import audit
    physical=[audit(p) for p in (root/'graphs'/f'rank{rank}').rglob('*PostGraph-symbol.pbtxt')]
    reference=[p for p in physical if p['operations'].get('custom_deepseek_v41_expert_n256_fp8_gaudi2')]
    candidate=[p for p in physical if p['operations'].get(candidate_guid)]
    (directory/'physical_nodes.json').write_text(json.dumps(physical,indent=2)+'\n')
    if not reference or not candidate or max(p['physical_nodes'] for p in candidate)>=max(p['physical_nodes'] for p in reference):
        raise RuntimeError('Final physical node count did not decrease; no performance credit or timed run')
    for graph in graphs:
        for _ in range(20):
            graph.replay_fixed_with_completion().synchronize()
    compiled_before = graph_compilation_count()
    periods = []
    tickets, events = [], []
    for label in '' if args.retirement_only else 'ABABAB':
        wait_for_loading(directory,rank,torch.distributed)
        graph = graphs[label == 'B']
        events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True))
                  for _ in range(args.steps)]
        torch.distributed.barrier()
        tickets = []
        for begin, end in events:
            begin.record()
            tickets.append(graph.replay_fixed_with_completion())
            end.record()
        torch.hpu.synchronize()
        local = [begin.elapsed_time(end)/args.chain_repeats for begin, end in events]
        ranks = [None]*4
        torch.distributed.all_gather_object(ranks, local)
        periods.append(dict(arm=label, rank_device_ms=ranks,
                            median_ms=statistics.median(max(values) for values in zip(*ranks, strict=True))))
        (directory/'result.json').write_text(json.dumps(dict(status='measuring', checks=checks, periods=periods))+'\n')
        if rank == 0:
            print(json.dumps(dict(arm=label, median_ms=periods[-1]['median_ms'])), flush=True)
    if graph_compilation_count() != compiled_before:
        raise RuntimeError('Hot compilation invalidates timing')
    savings = [a['median_ms']-b['median_ms'] for a, b in zip(periods[::2], periods[1::2], strict=True)]
    result = dict(status='retirement_check' if args.retirement_only else 'completed', checks=checks, periods=periods,
                  saving_ms_per_layer=statistics.median(savings)/len(args.layers) if savings else None,
                  round_savings_ms=savings,
                  three_consistent_rounds=bool(savings) and all(value > 0 for value in savings),
                  full_model_gain_credit=False, physical_node_gate_pending=False, layers=args.layers, selected_experts=12,
                  input_source=__doc__, native_replay=True, chain_repeats=args.chain_repeats, production_shared_expert=True,
                  native_library_sha256=hashlib.sha256(Path(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY']).read_bytes()).hexdigest())
    (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    if rank == 0:
        (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    for graph, plan in zip(graphs, plans, strict=True):
        graph.close()
        plan.invalidate()
    # Retire native owners and compiled observers before their HCCL backend.
    # Keeping a loop-local plan or the cached runtime tuple alive beyond group
    # destruction delays native device teardown until Python finalization.
    graphs.clear()
    plans.clear()
    tickets.clear()
    events.clear()
    visible.clear()
    externals.clear()
    graph = plan = slot = compute = None
    producer = consumer = None
    producers.clear()
    recorder.backend = None
    for attribute in ('_vllm_gaudi_tp2_fused_ar_norm_runtime', '_vllm_gaudi_tp4_allreduce_runtime'):
        if hasattr(torch, attribute):
            delattr(torch, attribute)
    del backend
    torch._dynamo.reset()
    import gc
    gc.collect()
    torch.hpu.synchronize()
    torch.distributed.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    config.__exit__(None, None, None)
    (directory/'retirement.json').write_text(json.dumps(dict(native_owners_closed=True, groups_destroyed=True))+'\n')


if __name__ == '__main__':
    main()
