# SPDX-License-Identifier: Apache-2.0
"""Checkpoint compressor norm/Wk -> index norm/RoPE/FP4 publication -> index MME and paged MLA A/B.

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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--chain-repeats', type=int, default=16)
    parser.add_argument('--retirement-only', action='store_true',
                        help='Reuse measured A/B; check teardown without timing')
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    modules = os.environ['HABANA_VISIBLE_MODULES'].split(',')
    os.environ['HLS_MODULE_ID'] = modules[rank]
    if os.environ.get('DSV41_MICRO_RANK_CPUS'):
        os.sched_setaffinity(0, json.loads(os.environ['DSV41_MICRO_RANK_CPUS'])[rank])
    if os.environ.get('GRAPH_VISUALIZATION') == '1':
        graph_dir = args.output.resolve() / 'graphs' / f'rank{rank}'
        graph_dir.mkdir(parents=True, exist_ok=True)
        os.environ['PT_HPU_GRAPH_DUMP_PREFIX'] = str(graph_dir)
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
    from tools.deepseek_v41_resident_ab import graph_compilation_count

    torch.set_num_threads(1)
    torch.hpu.set_device(rank)
    root = args.output.resolve()
    directory = root / f'rank{rank}'
    directory.mkdir(parents=True, exist_ok=True)
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
    from vllm_gaudi import envs as gaudi_envs
    if not gaudi_envs.VLLM_HPU_DSV41_NATIVE_KV_PACK:
        raise RuntimeError("Production native KV codec must be enabled on the reference")
    from vllm_gaudi.ops.deepseek_v41_math import rotary_table, rms_norm, pack_fp4, fp4_roundtrip
    import torch.nn.functional as F
    layer=2;ratio=2
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_indexer import _bounded_mme_scores
    a_sidecar=WoaFP8Sidecar(args.prepared/'sidecars/wo_a_fp8',shard)
    b_sidecar=DenseFP8Sidecar(args.prepared/'sidecars/attention_dense_fp8',shard)
    wa=a_sidecar.tensor(f'layers.{layer}.attn.wo_a.weight','hpu')
    sa=a_sidecar.tensor(f'layers.{layer}.attn.wo_a.channel_scale','hpu')
    wb=b_sidecar.tensor(f'layers.{layer}.attn.wo_b.weight','hpu')
    sb=b_sidecar.tensor(f'layers.{layer}.attn.wo_b.channel_scale','hpu')
    index_weights=torch.ones((1,32),dtype=torch.bfloat16,device='hpu')
    candidate_rows=torch.arange(2048,dtype=torch.int32,device='hpu').reshape(1,-1)
    config_values=json.loads((args.prepared/'config.json').read_text())['text_config']
    eps=config_values['rms_norm_eps'];heads=config_values['num_attention_heads']//4
    comp_norm=shard.tensor(f'layers.{layer}.attn.compressor.norm.weight','hpu')
    index_weight=shard.tensor(f'layers.{layer}.attn.indexer.wk.weight','hpu')
    index_norm=shard.tensor(f'layers.{layer}.attn.indexer.k_norm.weight','hpu')
    scaling=config_values['rope_scaling']
    table=rotary_table(64,32768,config_values['compress_rope_theta'],
        scaling['original_max_position_embeddings'],scaling['factor'],scaling['beta_fast'],scaling['beta_slow'])
    phase=torch.cat((table[...,0],table[...,1]),-1).contiguous().to('hpu')
    pages=(torch.arange(256,dtype=torch.int32)+1).to('hpu')
    swa=torch.zeros((256,528),dtype=torch.uint8,device='hpu')
    lengths=torch.tensor([640],dtype=torch.int32,device='hpu')
    sink=shard.tensor(f'layers.{layer}.attn.attn_sink','hpu').float()
    scale=torch.tensor([512**-.5],device='hpu')
    with safe_open(args.prepared/'pp0-tp0.safetensors',framework='pt',device='cpu') as cp:
        embeddings=[cp.get_slice('embed.weight')[token:token+1].clone() for token in (17,41,128,512,1024)]
    fixtures=[]
    for embedding,position in zip(embeddings,(0,1,255,8191,16383),strict=True):
        latent=embedding[:,:512].contiguous()
        query=latent.reshape(1,1,512).expand(1,heads,512).contiguous()
        index_query=embedding[:,:128].reshape(1,128).expand(32,128).contiguous()
        # A repeated selected id exercises the freshly published logical row,
        # including incomplete groups whose authoritative packed row is null.
        selections=torch.full((1,512),position//ratio,dtype=torch.int32)
        fixtures.append((latent,torch.tensor([position],dtype=torch.int32),query,index_query,selections))
    x,position,query,index_query,selections=[v.to('hpu') for v in fixtures[0]]
    from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
    production_backend=make_backend(static_int32=True,static_factories=True,split_mhc=True)
    compiled=lambda fn:torch.compile(fn,backend=production_backend,fullgraph=True,dynamic=False)
    def produce(value,norm,weight):
        latent=rms_norm(value,norm,eps)
        return latent,F.linear(latent,weight)
    producer=compiled(produce)
    state=[]
    def consume(latent,raw,norm,pos,phase,pages,main,index,decoded,hot,mirror,q,index_query,chosen,*,fused):
        if fused:
            done=torch.ops.custom_op.custom_deepseek_v41_fp4_norm_rope_publish_gaudi2(
                latent,raw,norm,pos,phase,pages,main,index,decoded,hot,mirror,eps,ratio,False,False,True)
            pos=done[:1]
        else:
            first=(pos & -ratio).int()
            normalized=rms_norm(raw,norm,eps)
            k=torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(normalized.reshape(1,1,128),first,phase).reshape(1,128)
            v=torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(latent.reshape(1,1,512),first,phase).reshape(1,512)
            logical=pos >> 1
            physical=pages.index_select(0,(logical >> 6).long())*64+(logical&63)
            physical=torch.where((pos&1)==1,physical,logical&63).int()
            mirror.index_copy_(0,logical.long(),fp4_roundtrip(k,32))
            index.index_copy_(0,physical.long(),pack_fp4(k,32))
            main.index_copy_(0,physical.long(),pack_fp4(v,16))
            done=pos.expand(36).contiguous()
        # The production paged-key/MME scorer consumes the writer's position
        # edge before reading packed index state. No synthetic add-zero edge.
        scores=_bounded_mme_scores(index_query.reshape(1,32,128),index_weights,index,pages,pos,
            candidate_rows,ratio,8,8192,False,mirror)
        out=torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(
            q,swa,main,chosen,pos,pages,sink,scale,lengths,ratio)
        out=torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(out,pos,phase)
        out=torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2(out.reshape(1,2,4096),wa,sa)
        out=torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(out,wb,sb)
        return done,scores,out
    consumers=[compiled(lambda *args,fused=fused:consume(*args,fused=fused)) for fused in (False,True)]
    finisher=compiled(lambda peers:sum((peers[i].float() for i in range(4))).bfloat16())
    plans, graphs, visible, externals = [], [], [], []
    for consumer in consumers:
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

        main=torch.zeros((8256,288),dtype=torch.uint8,device='hpu')
        index=torch.zeros((8256,68),dtype=torch.uint8,device='hpu')
        decoded=torch.zeros((16384,512),dtype=torch.bfloat16,device='hpu')
        hot=torch.zeros((8192,128),dtype=torch.bfloat16,device='hpu')
        mirror=torch.zeros_like(hot)
        state.append((main,index,decoded,hot,mirror))
        latent,raw=compute(producer,(x,comp_norm,index_weight))
        outputs=compute(consumer,(latent,raw,index_norm,position,phase,pages,main,index,decoded,hot,mirror,
                                  query,index_query,selections))
        peers=torch.ops.vllm_gaudi.tp_peer_allgather(outputs[2],4)
        torch.hpu.synchronize();plan.add_all_gather(slot(outputs[2],False),slot(peers,False))
        outputs=tuple(outputs)+(compute(finisher,(peers.reshape(4,1,5120),)),)
        plan.prepare(backend, [slot(value, False) for value in outputs])
        graph = bridge.NativeDecodeGraph()
        graph.configure_topology(args.chain_repeats,args.chain_repeats,False)
        graph.configure_dependency_policy(False)
        graph.capture([plan]*args.chain_repeats,[external]*args.chain_repeats)
        graph.instantiate()
        graph.bind_dynamic_inputs([x,position,query,index_query,selections])
        plans.append(plan)
        graphs.append(graph)
        visible.append(outputs)
        externals.append(external)
    checks = []
    for index, fixture in enumerate(fixtures):
        for destination, value in zip((x,position,query,index_query,selections), fixture, strict=True):
            destination.copy_(value.to('hpu'))
        torch.hpu.synchronize()
        observed = []
        for buffers in state:
            for buffer in buffers: buffer.zero_()
        torch.hpu.synchronize()
        for graph,outputs,buffers in zip(graphs,visible,state,strict=True):
            graph.replay_fixed_with_completion().synchronize()
            observed.append([value.cpu() for value in outputs] + [buffer.cpu() for buffer in buffers])
        exact = [torch.equal(a.view(torch.uint8),b.view(torch.uint8))
                 for i,(a, b) in enumerate(zip(*observed, strict=True))]
        if not all(exact):
            torch.save(dict(fixture=fixture, observed=observed), directory/'failure.pt')
            raise RuntimeError(f'KV publication differs for input {index}: {exact}')
        checks.append(dict(input=index, all_outputs_exact=True))
    for graph in graphs:
        for _ in range(20):
            graph.replay_fixed_with_completion().synchronize()
    compiled_before = graph_compilation_count()
    periods = []
    tickets, events = [], []
    for label in '' if args.retirement_only else 'ABABAB':
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
        local = [begin.elapsed_time(end)/args.chain_repeats for begin,end in events]
        ranks = [None]*4
        torch.distributed.all_gather_object(ranks, local)
        periods.append(dict(arm=label, rank_device_ms=ranks,
                            median_ms=statistics.median(max(values) for values in zip(*ranks, strict=True))))
    if graph_compilation_count() != compiled_before:
        raise RuntimeError('Hot compilation invalidates timing')
    savings = [a['median_ms']-b['median_ms'] for a, b in zip(periods[::2], periods[1::2], strict=True)]
    result = dict(status='retirement_check' if args.retirement_only else 'completed', checks=checks, periods=periods,
                  saving_ms_per_layer=statistics.median(savings) if savings else None,
                  round_savings_ms=savings,
                  three_consistent_rounds=bool(savings) and all(value > 0 for value in savings),
                  full_model_gain_credit=False, physical_node_gate_pending=True,chain_repeats=args.chain_repeats,
                  production_compiler_static_coordinates=True,
                  hot_mirror_enabled=False,index_mirror_enabled=True,
                  index_consumer="production mirrored index MME",
                  input_source=__doc__, native_replay=True,
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
    producer = consumer = finisher = None
    consumers.clear()
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
