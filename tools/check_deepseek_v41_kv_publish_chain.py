# SPDX-License-Identifier: Apache-2.0
"""Checkpoint KV projection -> norm/RoPE/cache publication -> paged MLA A/B.

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
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--mirror', action='store_true')
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
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
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
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_qkv import direct_dense_fp8
    from vllm_gaudi.ops.deepseek_v41_math import rotary_table, pack_swa
    config_values = json.loads((args.prepared/'config.json').read_text())['text_config']
    sidecar = DenseFP8Sidecar(args.sidecar, shard)
    layer = 2
    weight = sidecar.tensor(f'layers.{layer}.attn.wkv.weight', 'hpu')
    channel = sidecar.tensor(f'layers.{layer}.attn.wkv.channel_scale', 'hpu')
    norm = shard.tensor(f'layers.{layer}.attn.kv_norm.weight', 'hpu')
    output_sidecar = DenseFP8Sidecar(args.prepared/'sidecars/attention_dense_fp8', shard)
    a_sidecar = WoaFP8Sidecar(args.prepared/'sidecars/wo_a_fp8', shard)
    weight_a = a_sidecar.tensor(f'layers.{layer}.attn.wo_a.weight', 'hpu')
    scale_a = a_sidecar.tensor(f'layers.{layer}.attn.wo_a.channel_scale', 'hpu')
    weight_b = output_sidecar.tensor(f'layers.{layer}.attn.wo_b.weight', 'hpu')
    scale_b = output_sidecar.tensor(f'layers.{layer}.attn.wo_b.channel_scale', 'hpu')
    ffn_norm = shard.tensor(f'layers.{layer}.ffn_norm.weight', 'hpu')
    control, control_scale, control_base = [shard.tensor(f'layers.{layer}.hc_attn_{name}', 'cpu')
                                          for name in ('fn','scale','base')]
    scaling = config_values['rope_scaling']
    table = rotary_table(64, 32768, config_values['compress_rope_theta'],
                         scaling['original_max_position_embeddings'],scaling['factor'],
                         scaling['beta_fast'],scaling['beta_slow'])
    phase = torch.cat((table[...,0],table[...,1]),-1).contiguous().to('hpu')
    heads = config_values['num_attention_heads']//4
    sink = shard.tensor(f'layers.{layer}.attn.attn_sink','hpu').float()
    softmax_scale = torch.tensor([512**-.5],device='hpu')
    main = torch.zeros((8192,288),dtype=torch.uint8,device='hpu')
    main_rows = torch.cat((torch.arange(256),torch.arange(512)+256)).reshape(1,768).int().to('hpu')
    pages=torch.arange(4096,dtype=torch.int32,device='hpu')
    ids = torch.cat((torch.arange(128),torch.arange(512)+256)).reshape(1,-1).int().to('hpu')
    lengths = torch.tensor([640],dtype=torch.int32,device='hpu')
    with safe_open(args.prepared/'pp0-tp0.safetensors', framework='pt', device='cpu') as checkpoint:
        embeddings = [checkpoint.get_slice('embed.weight')[token:token+1].clone()
                      for token in (17, 41, 128, 512, 1024)]
    fixtures=[]
    for position, embedding in zip((0,127,255,8191,16384),embeddings,strict=True):
        query=embedding[:,:512].reshape(1,1,512).expand(1,heads,512).contiguous()
        # Include the updated physical row in the real selected gather.
        selections=torch.arange(512).reshape(1,512).int()
        residual=embedding.unsqueeze(1).expand(-1,4,-1).contiguous()
        _,pre,post,comb=hc_pre(residual,torch.tensor([[1.,0.,0.,0.]]),control,control_scale,control_base)
        fixtures.append((embedding,torch.tensor([position],dtype=torch.int32),query,selections,residual,post,comb,pre))
    x,position,query,selections,residual,post,comb,pre=[v.to('hpu') for v in fixtures[0]]
    compiled=lambda fn:torch.compile(fn,backend='hpu_backend',fullgraph=True,dynamic=False)
    producer=compiled(direct_dense_fp8)
    offset=0 if args.mirror else -1
    state=[]
    def consume(raw, norm_weight, pos, phase_table, cache, decoded, q, chosen,
                main, rows, sink, scale, lens, weight_a, scale_a, weight_b, scale_b, *, fused):
        if fused:
            kv,done=torch.ops.custom_op.custom_deepseek_v41_kv_norm_rope_publish_gaudi2(
                raw,norm_weight,pos,phase_table,cache,decoded,1e-20,offset)
            pos=done[:1]
        else:
            kv=torch.ops.custom_op.custom_deepseek_v41_kv_norm_rope_bf16_gaudi2(raw,norm_weight,pos,phase_table,1e-20)
            if args.mirror:
                done=torch.ops.custom_op.custom_deepseek_v41_swa_paged_decoded_write_bf16_gaudi2(
                    cache,kv,pos,pos,decoded,0)
            else:
                packed=torch.ops.custom_op.custom_deepseek_v41_swa_pack_bf16_gaudi2(kv)
                cache.index_copy_(0,(pos&255).long(),packed)
                done=pos.expand(16).contiguous()
        output=torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(
            q,cache,main,chosen,pos,pages,sink,scale,lens,2)
        output=torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(output,pos,phase_table)
        output=torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2(
            output.reshape(1,2,4096).contiguous(),weight_a,scale_a)
        output=torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(output.contiguous(),weight_b,scale_b)
        return kv,output,done
    consumers=[compiled(lambda *args,fused=fused:consume(*args,fused=fused)) for fused in (False,True)]
    def finish(peers,residual,post,comb,pre,norm):
        value=peers[0].float()
        for rank_id in range(1,4):value=value+peers[rank_id].float()
        residual,collapsed=torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
            value.bfloat16(),residual,post,comb,pre)
        normalized,quantized,scale=torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(collapsed,norm,1e-20)
        return residual,collapsed,normalized,quantized,scale
    finisher=compiled(finish)
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

        cache=torch.zeros((256,528),dtype=torch.uint8,device='hpu')
        decoded=torch.zeros((512,512),dtype=torch.bfloat16,device='hpu')
        state.append((cache,decoded))
        partial=compute(producer,(x,weight,channel))
        outputs=compute(consumer,(partial,norm,position,phase,cache,decoded,query,selections,
                                  main,main_rows,sink,softmax_scale,lengths,weight_a,scale_a,weight_b,scale_b))
        peers=torch.ops.vllm_gaudi.tp_peer_allgather(outputs[1],4)
        torch.hpu.synchronize()
        plan.add_all_gather(slot(outputs[1],False),slot(peers,False))
        outputs=tuple(outputs)+tuple(compute(finisher,(peers.reshape(4,1,5120),residual,post,comb,pre,ffn_norm)))
        plan.prepare(backend, [slot(value, False) for value in outputs])
        graph = bridge.NativeDecodeGraph()
        graph.configure_topology(1, 1, False)
        graph.configure_dependency_policy(False)
        graph.capture([plan], [external])
        graph.instantiate()
        graph.bind_dynamic_inputs([x, position, query, selections,residual,post,comb,pre])
        plans.append(plan)
        graphs.append(graph)
        visible.append(outputs)
        externals.append(external)
    checks = []
    for index, fixture in enumerate(fixtures):
        for destination, value in zip((x, position, query, selections,residual,post,comb,pre), fixture, strict=True):
            destination.copy_(value.to('hpu'))
        torch.hpu.synchronize()
        observed = []
        for cache,decoded in state:
            cache.zero_();decoded.zero_()
        torch.hpu.synchronize()
        for graph, outputs, (cache,decoded) in zip(graphs, visible, state, strict=True):
            graph.replay_fixed_with_completion().synchronize()
            observed.append([value.cpu() for value in outputs] + [cache.cpu(),decoded.cpu()])
        exact = [torch.equal(a.ge(0),b.ge(0)) if args.mirror and i==2 else torch.equal(a.view(torch.uint8), b.view(torch.uint8))
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
        local = [begin.elapsed_time(end) for begin, end in events]
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
                  full_model_gain_credit=False, physical_node_gate_pending=True,mirror=args.mirror,
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
