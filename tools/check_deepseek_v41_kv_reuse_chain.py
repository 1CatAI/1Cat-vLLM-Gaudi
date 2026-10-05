# SPDX-License-Identifier: Apache-2.0
"""Checkpoint fused Q/KV plus Q-norm projection -> KV norm/RoPE/SWA + shared-main reuse MLA -> WO/TP/mHC A/B.

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
    parser.add_argument('--wait-measurement-file',type=Path,help='Prepare native plans, then wait for the shared service timing window to finish')
    parser.add_argument('--qkv-transpose', action='store_true', help='Cold transpose fused QKV FP8 weights; vector MLA and all consumers held fixed')
    parser.add_argument('--mla-tensor-mask', action='store_true', help='Publish per-row partial tensor masks with unchanged row mapping')
    parser.add_argument('--mla-hardware-codec', action='store_true', help='Hold vector masks/codecs fixed; compare exact hardware E4M3 conversion')
    parser.add_argument('--mla-vector-mask', action='store_true', help='Hold vector codec fixed; vectorize reuse/publish masks and eliminate power-of-two page division')
    parser.add_argument('--mla-vector', action='store_true', help='Vectorize the MLA reuse gather with QKV and WO held fixed')
    parser.add_argument('--mla-publish',action='store_true',help='Exercise and retain shared-row/mask publication in the projection chain')
    parser.add_argument('--mla-projection',action='store_true',help='Fuse shared-main PV cast/inverse-RoPE/WO quant with both MME projections')
    parser.add_argument('--peer-post-norm',action='store_true',help='Fuse qualified peer/post collapse with FFN norm/quant; QKV stays fixed')
    parser.add_argument('--rope-handoff', action='store_true', help='Combine inverse RoPE/WO quant and WO scale/dense quant as one module; requires --woa-handoff')
    parser.add_argument('--phase-rows', type=int, default=32768)
    parser.add_argument('--woa-handoff',action='store_true',help='Compare fused WO scale/roundtrip/dense quant with joint QKV held fixed')
    parser.add_argument('--qkv-fusion',action='store_true',help='Fuse Q preparation with KV publication before Q MME')
    parser.add_argument('--reference-state-dir',type=Path,help='Archived production packed main KV/page mapping for realistic reuse fixtures')
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--chain-repeats', type=int, default=128,
                        help='Keep each native device interval near a full token to amortize rank enqueue skew')
    parser.add_argument('--retirement-only', action='store_true',
                        help='Reuse measured A/B; check teardown without timing')
    args = parser.parse_args()
    if args.mla_tensor_mask:
        if not args.mla_publish:
            parser.error("--mla-tensor-mask requires --mla-publish")
        args.mla_vector = True
    if args.mla_hardware_codec:
        args.mla_vector_mask = True
    if args.qkv_transpose or args.mla_vector_mask:
        args.mla_vector = True
    if args.mla_publish and not (args.mla_projection or args.mla_vector):
        parser.error("--mla-publish requires --mla-projection")
    if args.rope_handoff and not args.woa_handoff:
        parser.error("--rope-handoff requires --woa-handoff")
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
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_math import hc_pre
    from tools.deepseek_v41_micro_replay import RecipeRecorder
    from tools.deepseek_v41_resident_ab import graph_compilation_count, wait_for_loading

    os.environ['VLLM_HPU_DSV41_TP_MHC_OVERLAP']='0'
    torch.set_num_threads(1)
    torch.hpu.set_device(rank)
    root = args.output.resolve()
    directory = root / f'rank{rank}'
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"result.json").write_text('{"status":"started"}\n')
    if rank == 0:
        (root/"result.json").write_text('{"status":"started"}\n')
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
    layer = 3
    qa=sidecar.tensor(f'layers.{layer}.attn.wq_a.weight','cpu')
    kv_weight=sidecar.tensor(f'layers.{layer}.attn.wkv.weight','cpu')
    weight=torch.cat((qa,kv_weight),0).contiguous().to('hpu')
    channel=torch.cat((sidecar.tensor(f'layers.{layer}.attn.wq_a.channel_scale','cpu'),
                       sidecar.tensor(f'layers.{layer}.attn.wkv.channel_scale','cpu')),1).to('hpu')
    norm=shard.tensor(f'layers.{layer}.attn.kv_norm.weight','hpu')
    qnorm=shard.tensor(f'layers.{layer}.attn.q_norm.weight','hpu')
    output_sidecar = DenseFP8Sidecar(args.prepared/'sidecars/attention_dense_fp8', shard)
    qb=output_sidecar.tensor(f'layers.{layer}.attn.wq_b.weight','hpu')
    sqb=output_sidecar.tensor(f'layers.{layer}.attn.wq_b.channel_scale','hpu')
    a_sidecar = WoaFP8Sidecar(args.prepared/'sidecars/wo_a_fp8', shard)
    weight_a = a_sidecar.tensor(f'layers.{layer}.attn.wo_a.weight', 'hpu')
    scale_a = a_sidecar.tensor(f'layers.{layer}.attn.wo_a.channel_scale', 'hpu')
    weight_b = output_sidecar.tensor(f'layers.{layer}.attn.wo_b.weight', 'hpu')
    scale_b = output_sidecar.tensor(f'layers.{layer}.attn.wo_b.channel_scale', 'hpu')
    ffn_norm = shard.tensor(f'layers.{layer}.ffn_norm.weight', 'hpu')
    if args.peer_post_norm:
        downstream_control=shard.tensor(f'layers.{layer}.hc_ffn_fn','hpu')
        downstream_scale=shard.tensor(f'layers.{layer}.hc_ffn_scale','hpu')
        downstream_base=shard.tensor(f'layers.{layer}.hc_ffn_base','hpu')
        downstream_router=shard.tensor(f'layers.{layer}.ffn.gate.weight','hpu')
    control, control_scale, control_base = [shard.tensor(f'layers.{layer}.hc_attn_{name}', 'cpu')
                                          for name in ('fn','scale','base')]
    scaling = config_values['rope_scaling']
    table = rotary_table(64, args.phase_rows, config_values['compress_rope_theta'],
                         scaling['original_max_position_embeddings'],scaling['factor'],
                         scaling['beta_fast'],scaling['beta_slow'])
    phase = torch.cat((table[...,0],table[...,1]),-1).contiguous().to('hpu')
    heads = config_values['num_attention_heads']//4
    sink = shard.tensor(f'layers.{layer}.attn.attn_sink','hpu').float()
    softmax_scale = torch.tensor([512**-.5],device='hpu')
    lengths=torch.tensor([640],dtype=torch.int32,device='hpu')
    shared_mask=torch.ones((1,640),dtype=torch.float32,device='hpu')
    # Derived checkpoint rows are nonzero: test both ring history and main reuse.
    with safe_open(args.prepared/'pp0-tp0.safetensors', framework='pt', device='cpu') as checkpoint:
        embeddings = [checkpoint.get_slice('embed.weight')[token:token+1].clone()
                      for token in (17, 41, 128, 512, 1024)]
    base_hidden=embeddings[0][:,:512].clone()
    main_values=base_hidden.reshape(1,1,512).expand(1,640,512).contiguous().to('hpu')
    if args.reference_state_dir:
        from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4
        reference=torch.load(args.reference_state_dir/f'state-original-rank{rank}.pt',map_location='cpu',weights_only=True)
        if not isinstance(reference,(list,tuple)) or len(reference)<2:
            raise ValueError('Reference KV snapshot must contain pages and packed source2 main rows')
        pages,packed=reference[:2]
        if pages.dtype!=torch.int32 or pages.ndim!=1 or packed.dtype!=torch.uint8 or packed.ndim!=2 or packed.shape[1]!=288:
            raise ValueError('Unsupported reference page/packed-KV format')
        logical=torch.arange(512)*13
        physical=pages[logical//64].long()*64+logical%64
        if bool((physical<0).any() or (physical>=packed.shape[0]).any()):
            raise ValueError('Reference selected rows exceed packed source storage')
        selected_main=unpack_fp4(packed).to(torch.bfloat16)[physical]
        selected_main=torch.where(selected_main==0,torch.zeros_like(selected_main),selected_main)
        main_values=torch.cat((torch.zeros((128,512),dtype=torch.bfloat16),selected_main),0).unsqueeze(0).to('hpu')
    if args.mla_publish:
        from vllm_gaudi.ops.deepseek_v41_math import pack_fp4
        packed_main=pack_fp4(main_values[0],16).contiguous()
        selected_rows=torch.arange(512,dtype=torch.int32,device='hpu').reshape(1,512)
        page_table=torch.arange(4096,dtype=torch.int32,device='hpu')
    history=torch.cat([embeddings[i%5][:,:512] for i in range(256)],0).contiguous().to('hpu')
    base_cache=torch.ops.custom_op.custom_deepseek_v41_swa_pack_bf16_gaudi2(history)
    fixtures=[]
    for position,embedding in zip((0,127,255,8191,16384),embeddings,strict=True):
        residual=embedding.unsqueeze(1).expand(-1,4,-1).contiguous()
        _,pre,post,comb=hc_pre(residual,torch.tensor([[1.,0.,0.,0.]]),control,control_scale,control_base)
        fixtures.append((embedding,torch.tensor([position],dtype=torch.int32),residual,post,comb,pre))
    x,position,residual,post,comb,pre=[v.to('hpu') for v in fixtures[0]]
    from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
    production_backend=make_backend(static_int32=True,static_factories=True,split_mhc=True)
    compiled=lambda fn:torch.compile(fn,backend=production_backend,fullgraph=True,dynamic=False)
    def project(value,weight,channel,pos,qnorm,qb,sqb,phase):
        projected=direct_dense_fp8(value,weight,channel)
        qr=projected[:,:1280].contiguous()
        raw=projected[:,1280:].contiguous()
        query=torch.ops.custom_op.custom_deepseek_v41_q_norm_projection_rope_gaudi2(
            qr,qnorm,qb,sqb,pos,phase,1e-20).reshape(1,heads,512)
        return query,raw
    decoded_unused=torch.zeros((512,512),dtype=torch.bfloat16,device='hpu')
    state=[]
    def consume(q,raw,norm,pos,phase,cache,main,mask,sink,scale,lens,wa,sa,wb,sb,*,fused):
        if fused:
            out=torch.ops.custom_op.custom_deepseek_v41_kv_norm_reuse_mla_gaudi2(
                q,raw,norm,cache,main,mask,pos,phase,sink,scale,lens,1e-20)
        else:
            kv=torch.ops.custom_op.custom_deepseek_v41_kv_norm_rope_bf16_gaudi2(raw,norm,pos,phase,1e-20)
            packed=torch.ops.custom_op.custom_deepseek_v41_swa_pack_bf16_gaudi2(kv)
            cache.index_copy_(0,(pos&255).long(),packed)
            out=torch.ops.custom_op.custom_deepseek_v41_main_reuse_mla_gaudi2(
                q,cache,main,mask,pos,sink,scale,lens)
        out=torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(out,pos,phase)
        out=torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2(out.reshape(1,2,4096),wa,sa)
        out=torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(out,wb,sb)
        return out
    transposed_qkv = weight.cpu().T.contiguous().to("hpu") if args.qkv_transpose else None
    def full_chain(x,weight,channel,pos,qnorm,qb,sqb,phase,norm,cache,main,mask,sink,scale,lens,wa,sa,wb,sb,*,fused):
        if (fused and args.qkv_fusion) or args.woa_handoff or args.peer_post_norm or args.mla_projection or args.mla_vector:
            if args.qkv_transpose:
                quantized, activation_scale = torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(x)
                projection = torch.ops.hpu.fp8_gemm_v2(
                    quantized, False, weight, weight.shape[-1] == x.shape[-1], None, torch.bfloat16,
                    activation_scale, channel, None, False)
            else:
                projection=direct_dense_fp8(x,weight,channel)
            q,kv,completion,qr=torch.ops.custom_op.custom_deepseek_v41_qkv_projection_publish_gaudi2(
                projection[:,:1280].contiguous(),qnorm,projection[:,1280:].contiguous(),norm,
                pos,phase,cache,decoded_unused,qb,sqb,1e-20,-1)
            pos=completion[:1]
            if args.mla_publish:
                if fused and args.mla_projection:
                    return torch.ops.custom_op.custom_deepseek_v41_main_publish_projection_gaudi2(
                        q.reshape(1,heads,512),cache,packed_main,selected_rows,pos,page_table,
                        sink,scale,lens,1,wa,sa,phase,wb,sb)
                mla_op = (torch.ops.custom_op.custom_deepseek_v41_main_publish_tensor_mask_mla_gaudi2
                          if fused and args.mla_tensor_mask else
                          torch.ops.custom_op.custom_deepseek_v41_main_publish_native_codec_mla_gaudi2
                          if fused and args.mla_hardware_codec else
                          torch.ops.custom_op.custom_deepseek_v41_main_publish_vector_mask_mla_gaudi2
                          if args.mla_hardware_codec or (fused and args.mla_vector_mask) else
                          torch.ops.custom_op.custom_deepseek_v41_main_publish_vector_mla_gaudi2
                          if args.mla_tensor_mask or args.qkv_transpose or args.mla_vector_mask or (fused and args.mla_vector) else
                          torch.ops.custom_op.custom_deepseek_v41_main_publish_mla_gaudi2)
                out,published_rows,published_mask=mla_op(
                    q.reshape(1,heads,512),cache,packed_main,selected_rows,pos,page_table,sink,scale,lens,1)
            else:
                if args.mla_projection and fused:
                    return torch.ops.custom_op.custom_deepseek_v41_main_reuse_projection_gaudi2(
                        q.reshape(1,heads,512),cache,main,mask,pos,sink,scale,lens,wa,sa,phase,wb,sb)
                mla_op = (torch.ops.custom_op.custom_deepseek_v41_main_reuse_native_codec_mla_gaudi2
                          if fused and args.mla_hardware_codec else
                          torch.ops.custom_op.custom_deepseek_v41_main_reuse_vector_mask_mla_gaudi2
                          if args.mla_hardware_codec or (fused and args.mla_vector_mask) else
                          torch.ops.custom_op.custom_deepseek_v41_main_reuse_vector_mla_gaudi2
                          if args.mla_tensor_mask or args.qkv_transpose or args.mla_vector_mask or (fused and args.mla_vector) else
                          torch.ops.custom_op.custom_deepseek_v41_main_reuse_mla_gaudi2)
                out=mla_op(q.reshape(1,heads,512),cache,main,mask,pos,sink,scale,lens)
            if args.woa_handoff and args.rope_handoff and fused:
                return torch.ops.custom_op.custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2(
                    out,wa,sa,wb,sb,pos,phase)
            out=torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(out,pos,phase)
            if args.woa_handoff and fused:
                return torch.ops.custom_op.custom_deepseek_v41_woa_wob_roundtrip_fp8_gaudi2(
                    out.reshape(1,heads//8,4096),wa,sa,wb,sb)
            out=torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2(out.reshape(1,heads//8,4096),wa,sa)
            partial=torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(out,wb,sb)
            return (partial,published_rows,published_mask) if args.mla_publish else partial
        q,raw=project(x,weight,channel,pos,qnorm,qb,sqb,phase)
        return consume(q,raw,norm,pos,phase,cache,main,mask,sink,scale,lens,wa,sa,wb,sb,fused=fused)
    consumers=[compiled(lambda *args,fused=fused:full_chain(*args,fused=fused)) for fused in (False,True)]
    def post_consumers(residual,collapsed,normalized,quantized,scale):
        # Retain both downstream dependency branches. The control projection
        # consumes the updated residual; the router consumes the normalized
        # row. A fused producer must not claim time saved by delaying either.
        control=torch.ops.custom_op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(
            residual.flatten(1).contiguous(),downstream_control,1e-20)
        gates=torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(
            control[:,:24].contiguous(),control[:,24:].contiguous(),downstream_scale,downstream_base)
        logits=torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(normalized,downstream_router)
        return residual,collapsed,normalized,quantized,scale,control,gates,logits

    def finish(peers,residual,post,comb,pre,norm,*,fused):
        if args.peer_post_norm:
            if fused:
                outputs=torch.ops.custom_op.custom_deepseek_v41_peer_post_norm_quant_gaudi2(
                    peers,residual,post,comb,pre,norm,1e-20)
                return post_consumers(*outputs)
            residual,collapsed=torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
                peers,residual,post,comb,pre)
            normalized,quantized,scale=torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
                collapsed,norm,1e-20)
            return post_consumers(residual,collapsed,normalized,quantized,scale)
        value=peers[0].float()
        for rank_id in range(1,4):value=value+peers[rank_id].float()
        residual,collapsed=torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
            value.bfloat16(),residual,post,comb,pre)
        normalized,quantized,scale=torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(collapsed,norm,1e-20)
        return residual,collapsed,normalized,quantized,scale
    finishers=[compiled(lambda *args,fused=fused:finish(*args,fused=fused)) for fused in (False,True)]
    plans, graphs, visible, externals = [], [], [], []
    for arm,(consumer,finisher) in enumerate(zip(consumers,finishers,strict=True)):
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

        cache=base_cache.clone();state.append(cache)
        output=compute(consumer,(x,transposed_qkv if args.qkv_transpose and arm else weight,channel,position,qnorm,qb,sqb,phase,norm,cache,main_values,shared_mask,
                                sink,softmax_scale,lengths,weight_a,scale_a,weight_b,scale_b))
        publication=()
        if args.mla_publish:
            output,*publication=output
        peers=torch.ops.vllm_gaudi.tp_peer_allgather(output,4)
        torch.hpu.synchronize();plan.add_all_gather(slot(output,False),slot(peers,False))
        outputs=(output,)+tuple(compute(finisher,(peers.reshape(4,1,5120),residual,post,comb,pre,ffn_norm)))
        outputs=outputs+tuple(publication)
        plan.prepare(backend, [slot(value, False) for value in outputs])
        graph = bridge.NativeDecodeGraph()
        graph.configure_topology(args.chain_repeats,args.chain_repeats,False)
        graph.configure_dependency_policy(False)
        graph.capture([plan]*args.chain_repeats,[external]*args.chain_repeats)
        graph.instantiate()
        graph.bind_dynamic_inputs([x,position,residual,post,comb,pre])
        plans.append(plan)
        graphs.append(graph)
        visible.append(outputs)
        externals.append(external)
    checks = []
    for index, fixture in enumerate(fixtures):
        for destination, value in zip((x,position,residual,post,comb,pre), fixture, strict=True):
            destination.copy_(value.to('hpu'))
        torch.hpu.synchronize()
        observed = []
        for cache in state: cache.copy_(base_cache)
        torch.hpu.synchronize()
        for graph,outputs,cache in zip(graphs,visible,state,strict=True):
            graph.replay_fixed_with_completion().synchronize()
            observed.append([value.cpu() for value in outputs] + [cache.cpu()])
        exact = [torch.equal(a.view(torch.uint8),b.view(torch.uint8))
                 for i,(a, b) in enumerate(zip(*observed, strict=True))]
        if not all(exact):
            torch.save(dict(fixture=fixture, observed=observed), directory/'failure.pt')
            raise RuntimeError(f'KV publication differs for input {index}: {exact}')
        checks.append(dict(input=index, all_outputs_exact=True))
    (directory/"result.json").write_text(json.dumps(dict(status="correctness_passed",checks=checks))+"\n")
    if args.wait_measurement_file is not None:
        import time
        torch.distributed.barrier()
        if rank == 0:
            args.wait_measurement_file.with_suffix('.ready.json').write_text(
                json.dumps(dict(status='correctness_and_native_capture_ready',checks=checks),indent=2)+'\n')
        while not args.wait_measurement_file.exists():
            time.sleep(2)
        torch.distributed.barrier()
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
        local = [begin.elapsed_time(end)/args.chain_repeats for begin,end in events]
        ranks = [None]*4
        torch.distributed.all_gather_object(ranks, local)
        periods.append(dict(arm=label, rank_device_ms=ranks,
                            median_ms=statistics.median(max(values) for values in zip(*ranks, strict=True))))
        (directory/'result.json').write_text(json.dumps(dict(status='measuring',checks=checks,periods=periods))+'\n')
        if rank == 0:
            print(json.dumps(dict(arm=label,median_ms=periods[-1]['median_ms'])),flush=True)
    if graph_compilation_count() != compiled_before:
        raise RuntimeError('Hot compilation invalidates timing')
    savings = [a['median_ms']-b['median_ms'] for a, b in zip(periods[::2], periods[1::2], strict=True)]
    result = dict(status='retirement_check' if args.retirement_only else 'completed', checks=checks, periods=periods,
                  saving_ms_per_layer=statistics.median(savings) if savings else None,
                  round_savings_ms=savings,
                  three_consistent_rounds=bool(savings) and all(value > 0 for value in savings),
                  full_model_gain_credit=False, physical_node_gate_pending=True,chain_repeats=args.chain_repeats,
                  production_compiler_static_coordinates=True,full_qkv_query_producer=True,
                  candidate_kind="mla_publish_tensor_mask" if args.mla_tensor_mask else ("mla_publish_hardware_codec" if args.mla_publish else "mla_reuse_hardware_codec") if args.mla_hardware_codec else ("mla_publish_mask" if args.mla_publish else "mla_vector_mask") if args.mla_vector_mask else "qkv_cold_transpose" if args.qkv_transpose else ("mla_publish_vector" if args.mla_publish else "mla_reuse_vector") if args.mla_vector else "main_publish_projection" if args.mla_publish else "main_reuse_projection" if args.mla_projection else "rope_woa_handoff" if args.rope_handoff else "peer_post_norm" if args.peer_post_norm else "woa_handoff" if args.woa_handoff else "qkv_joint" if args.qkv_fusion else "kv_reuse",
                  input_source=__doc__, native_replay=True,
                  native_library_sha256=hashlib.sha256(Path(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY']).read_bytes()).hexdigest(),
                  native_kernel_library_sha256=hashlib.sha256(Path(os.environ['GC_KERNEL_PATH']).read_bytes()).hexdigest())
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
    consumer = finisher = None
    consumers.clear()
    finishers.clear()
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
