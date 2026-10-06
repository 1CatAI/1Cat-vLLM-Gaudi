# SPDX-License-Identifier: Apache-2.0
"""Checkpoint WO projection/control -> TP -> hand-fused gates/post -> FFN router A/B.

Inputs are derived from five checkpoint embedding rows, not captured attention
activations. Exact norm/quant/router outputs and control error are reported separately.
The MME approximation is not declared globally bit-exact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weighted-post-statistics',action='store_true',help='Use producer RSS/weighted amax with feature-parallel norm dual quant; include shared W13/Silu and next actual mHC consumer')
    parser.add_argument('--peer-gates-vector',action='store_true',help='Compare whole gate-packet peer/post against the existing sliced-gate consumer')
    parser.add_argument('--positive-gates',action='store_true',help='Same parallel controller, Sinkhorn uses positive-denominator Newton reciprocal')
    parser.add_argument('--k-controller-peer-window', action='store_true',
                        help='Move the qualified K-controller recipe behind peer submit, before gates/post')
    parser.add_argument('--require-controller-sram', action='store_true',
                        help='Before timing, require real K-controller producer/consumer SRAM alias')
    parser.add_argument('--k-tiled-controller', action='store_true',
                        help='K23 BF16-weight controller; qualified gates-in-exchange parent in both arms')
    parser.add_argument('--gates-during-exchange', action='store_true',
                        help='Schedule exact gates after native exchange submission and before its first consumer')
    parser.add_argument('--bf16-control-weights', action='store_true',
                        help='Cold BF16 controller weights and native BF16 MAC with FP32 accumulation')
    parser.add_argument('--dense-transpose', action='store_true', help='Cold transpose WO weights and flip the native MME transpose flag; no hot weight copies')
    parser.add_argument('--early-gates', action='store_true', help='Use qualified parallel control in both arms; compute gates alongside WO before peer')
    parser.add_argument('--direct-rrms-post', action='store_true', help='Specialize the post kernel for the shared control/RRMS layout')
    parser.add_argument('--swizzled-control', action='store_true',
                        help='Compare packed K128/head weights with the qualified parallel controller')
    parser.add_argument('--late-control', action='store_true',
                        help='Schedule the qualified independent controller during the native peer exchange')
    parser.add_argument('--peer-prune', action='store_true',
                        help='Keep both compute arms identical; vary experimental HCL scale-out stream pruning at capture')
    parser.add_argument('--official-tolerance', action='store_true',
                        help='Qualify MME against official mHC equations with DeepGEMM normalized-error limits')
    parser.add_argument('--mme-shared-rrms', action='store_true',
                        help='Compute one control RMS statistic before the peer exchange')
    parser.add_argument('--parallel-controller', action='store_true',
                        help='Use independent FP32 K accumulators in the TPC controller')
    parser.add_argument('--unpack-controller', action='store_true', help='Use exact linear unpack loads in the candidate control producer')
    parser.add_argument('--exact-controller',action='store_true',help='Keep the production FP32 control/rrms producer and fuse only its exact gates/post consumer')
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--sidecar',type=Path,required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--chain-repeats', type=int, default=128,
                        help='Repeat complete native layers inside one device replay to amortize host enqueue gaps')
    parser.add_argument('--retirement-only', action='store_true',
                        help='Reuse measured A/B; check teardown without timing')
    args = parser.parse_args()
    if args.weighted_post_statistics:
        args.peer_gates_vector=True
    if args.peer_gates_vector:
        args.positive_gates=True
    if args.positive_gates:
        args.gates_during_exchange=True
        args.official_tolerance=True
    if args.k_controller_peer_window:
        args.k_tiled_controller = True
    if args.k_tiled_controller:
        args.gates_during_exchange = True
        args.official_tolerance = True
    if args.gates_during_exchange:
        if any((args.bf16_control_weights, args.dense_transpose, args.early_gates,
                args.direct_rrms_post, args.swizzled_control, args.late_control,
                args.peer_prune, args.mme_shared_rrms, args.unpack_controller)):
            parser.error('--gates-during-exchange is a separate scheduling experiment')
        if args.chain_repeats % 8:
            parser.error('--gates-during-exchange requires repetitions divisible by eight')
        args.parallel_controller = True
        args.swizzled_control = True
    if args.bf16_control_weights:
        if any((args.dense_transpose, args.early_gates, args.direct_rrms_post, args.swizzled_control,
                args.late_control, args.peer_prune, args.mme_shared_rrms, args.unpack_controller)):
            parser.error('--bf16-control-weights is a separate controller experiment')
        args.parallel_controller = True
        args.official_tolerance = True
    if args.dense_transpose:
        args.parallel_controller = True
    if args.early_gates:
        args.parallel_controller = True
    if args.direct_rrms_post:
        args.parallel_controller = True
    if args.swizzled_control and (args.late_control or args.peer_prune):
        parser.error('--swizzled-control excludes other experiments')
    if args.swizzled_control:
        args.parallel_controller = True
    if args.late_control:
        if not args.parallel_controller or args.peer_prune:
            parser.error('--late-control requires --parallel-controller and excludes other experiments')
        if args.chain_repeats % 8:
            parser.error('--late-control requires repetitions divisible by eight')
        os.environ['VLLM_HPU_DSV41_TP_MHC_OVERLAP'] = '1'
    elif args.gates_during_exchange:
        os.environ['VLLM_HPU_DSV41_TP_MHC_OVERLAP'] = '1'
    else:
        # One peer per component group is not the production eight-peer topology.
        os.environ['VLLM_HPU_DSV41_TP_MHC_OVERLAP'] = '0'
    if args.unpack_controller and not args.exact_controller:
        raise ValueError('Unpack controller requires --exact-controller')
    if args.chain_repeats < 1:
        raise ValueError('Native chain repetition must be positive')
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
    layer = 2
    config_values = json.loads((args.prepared/'config.json').read_text())['text_config']
    eps = config_values['rms_norm_eps']
    from vllm_gaudi.ops.deepseek_v41_qkv import direct_dense_fp8
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    sidecar=DenseFP8Sidecar(args.prepared/'sidecars/attention_dense_fp8',shard)
    wo_weight=sidecar.tensor(f'layers.{layer}.attn.wo_b.weight','hpu')
    wo_scale=sidecar.tensor(f'layers.{layer}.attn.wo_b.channel_scale','hpu')
    norm = shard.tensor(f'layers.{layer}.ffn_norm.weight','hpu')
    control = shard.tensor(f'layers.{layer}.hc_attn_fn' if args.weighted_post_statistics else f'layers.{layer}.hc_ffn_fn','hpu').contiguous()
    if args.weighted_post_statistics:
        shared_sidecar = DenseFP8Sidecar(args.sidecar,shard)
        projections={name:(shared_sidecar.tensor(f'layers.{layer}.ffn.shared_experts.{name}.weight','cpu'),
                          shared_sidecar.tensor(f'layers.{layer}.ffn.shared_experts.{name}.channel_scale','cpu')) for name in ('w1','w3')}
        width=projections['w1'][0].shape[0];padded=(width+127)//128*128
        pad=lambda v:torch.cat((v,torch.zeros((padded-width,v.shape[1]),dtype=v.dtype)),0)
        shared_weight=torch.cat((pad(projections['w1'][0]),pad(projections['w3'][0])),0).to('hpu')
        shared_channel=torch.cat((torch.nn.functional.pad(projections['w1'][1],(0,padded-width),value=1),
            torch.nn.functional.pad(projections['w3'][1],(0,padded-width),value=1)),1).bfloat16().reshape(1,padded*2//256,256).to('hpu')
        shared_id=torch.zeros((1,1),dtype=torch.int32,device='hpu');shared_route=torch.ones((1,1),dtype=torch.float32,device='hpu')
        next_control=shard.tensor(f'layers.{layer}.hc_ffn_fn','hpu').contiguous()
        next_weight=next_control.reshape(24,160,128).permute(1,0,2).contiguous()
        next_scale=shard.tensor(f'layers.{layer}.hc_ffn_scale','hpu');next_base=shard.tensor(f'layers.{layer}.hc_ffn_base','hpu')
    high = control.bfloat16()
    low = (control - high.float()).bfloat16()
    mme_weight = torch.cat((high,low),0).contiguous()
    control_scale = shard.tensor(f'layers.{layer}.hc_attn_scale' if args.weighted_post_statistics else f'layers.{layer}.hc_ffn_scale','hpu')
    control_base = shard.tensor(f'layers.{layer}.hc_attn_base' if args.weighted_post_statistics else f'layers.{layer}.hc_ffn_base','hpu')
    router_weight = shard.tensor(f'layers.{layer}.ffn.gate.weight','hpu')
    bias = shard.tensor(f'layers.{layer}.ffn.gate.bias','hpu')
    bias_vl = shard.tensor(f'layers.{layer}.ffn.gate.bias_vl','hpu')
    mask = torch.zeros((1,),dtype=torch.bool,device='hpu')
    attn_fn,attn_scale,attn_base = [shard.tensor(f'layers.{layer}.hc_attn_{n}','cpu')
                                  for n in ('fn','scale','base')]
    with safe_open(args.prepared/'pp0-tp0.safetensors',framework='pt',device='cpu') as cp:
        embeddings=[cp.get_slice('embed.weight')[token:token+1].clone()
                    for token in (17,41,128,512,1024)]
    fixtures=[]
    for i,embedding in enumerate(embeddings):
        residual=torch.stack([embeddings[(i+j)%5].squeeze(0) for j in range(4)],0).unsqueeze(0).contiguous()
        _,pre,post,comb=hc_pre(residual,torch.tensor([[1.,0.,0.,0.]]),attn_fn,attn_scale,attn_base,eps)
        fixtures.append((embedding[:,:wo_weight.shape[-1]].contiguous(),residual,post,comb,pre))
    x,residual,post,comb,pre=[v.to('hpu') for v in fixtures[0]]
    from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
    compiled=lambda fn:torch.compile(fn,backend=make_backend(static_int32=True,static_factories=True,split_mhc=True),fullgraph=True,dynamic=False)
    transposed_wo = wo_weight.cpu().T.contiguous().to("hpu") if args.dense_transpose else None
    packed_control = control.reshape(24, 160, 128).permute(1, 0, 2).contiguous() if args.swizzled_control else None
    bf16_control = (packed_control.bfloat16() if args.k_tiled_controller else
                    control.bfloat16() if args.bf16_control_weights else None)
    def produce(x,residual,control,mme_weight,wo_weight,wo_scale,*,fused):
        if args.k_tiled_controller:
            op = (torch.ops.custom_op.custom_deepseek_v41_control_k_tiled_bf16_gaudi2
                  if control.dtype == torch.bfloat16 else
                  torch.ops.custom_op.custom_deepseek_v41_control_rrms_swizzled_bf16_gaudi2)
            return direct_dense_fp8(x,wo_weight,wo_scale), op(residual.flatten(1),control,eps)
        if args.bf16_control_weights:
            op = (torch.ops.custom_op.custom_deepseek_v41_control_rrms_bf16_weight_gaudi2
                  if control.dtype == torch.bfloat16 else
                  torch.ops.custom_op.custom_deepseek_v41_control_rrms_parallel_bf16_gaudi2)
            return direct_dense_fp8(x,wo_weight,wo_scale), op(residual.flatten(1),control,eps)
        if args.dense_transpose:
            q, scale = torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(x)
            value = torch.ops.hpu.fp8_gemm_v2(q, False, wo_weight, wo_weight.shape[-1] == x.shape[-1],
                                             None, torch.bfloat16, scale, wo_scale, None, False)
            control_out = torch.ops.custom_op.custom_deepseek_v41_control_rrms_parallel_bf16_gaudi2(
                residual.flatten(1), control, eps)
            return value, control_out
        if args.swizzled_control:
            op = (torch.ops.custom_op.custom_deepseek_v41_control_rrms_swizzled_bf16_gaudi2 if control.ndim == 3 else
                  torch.ops.custom_op.custom_deepseek_v41_control_rrms_parallel_bf16_gaudi2)
            return direct_dense_fp8(x,wo_weight,wo_scale), op(residual.flatten(1),control,eps)
        flat=residual.flatten(1)
        projected=(torch.ops.custom_op.custom_deepseek_v41_control_rrms_unpack_bf16_gaudi2(flat,control,eps)
            if fused and args.unpack_controller else
            torch.ops.custom_op.custom_deepseek_v41_control_mme_f32_gaudi2(flat,mme_weight) if fused and not args.exact_controller else
            torch.ops.custom_op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(flat,control,eps))
        if args.early_gates or (fused and args.parallel_controller):
            projected = torch.ops.custom_op.custom_deepseek_v41_control_rrms_parallel_bf16_gaudi2(flat, control, eps)
        if fused and args.mme_shared_rrms:
            projected = torch.ops.custom_op.custom_deepseek_v41_control_mme_finish_gaudi2(flat, projected, eps)
        value=direct_dense_fp8(x,wo_weight,wo_scale)
        if fused:return value,projected
        gates=torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(
            projected[:,:24].contiguous(),projected[:,24:].contiguous(),control_scale,control_base)
        return value,gates
    def consume(peers,residual,projection,scale,base,norm,router,bias,bias_vl,mask,*,fused,positive=False):
        ready_quant=None
        if fused:
            # The same TPC owns fixed-rank summation and its BF16 boundary;
            # do not materialize a separate collective reduction node.
            post_op = (torch.ops.custom_op.custom_deepseek_v41_mhc_rrms_post_gaudi2
                       if args.gates_during_exchange or (positive and args.direct_rrms_post) else
                       torch.ops.custom_op.custom_deepseek_v41_mhc_mme_post_collapse_gaudi2)
            updated,collapsed,gates=post_op(
                peers,residual,projection,scale,base,eps)
            pre=gates[:,:4].contiguous()
        else:
            gates=projection
            pre=projection[:,:4].contiguous()
            # Match the already-qualified peer/post baseline. Counting a
            # separate rank sum here would double-credit the preceding batch.
            if args.weighted_post_statistics and positive:
                updated,collapsed,normalized,quantized,act_scale,dense_q,dense_scale = (
                    torch.ops.custom_op.custom_deepseek_v41_mhc_post_norm_statistics_gaudi2(
                        peers,residual,projection.contiguous(),norm,eps))
                ready_quant=normalized,quantized,act_scale,dense_q,dense_scale
            elif (args.peer_gates_vector and positive) or args.weighted_post_statistics:
                updated,collapsed=torch.ops.custom_op.custom_deepseek_v41_mhc_gates_post_gaudi2(
                    peers,residual,projection.contiguous())
            else:
                updated,collapsed=torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
                    peers,residual,projection[:,4:8].contiguous(),projection[:,8:].reshape(1,4,4).contiguous(),pre)
        if ready_quant is not None:
            normalized,quantized,act_scale,dense_q,dense_scale=ready_quant
        elif args.peer_gates_vector:
            normalized,quantized,act_scale,dense_q,dense_scale=torch.ops.custom_op.custom_deepseek_v41_ffn_norm_dual_bf16_quant_gaudi2(collapsed,norm,eps)
        else:
            normalized,quantized,act_scale=torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(collapsed,norm,eps)
        logits=torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(normalized,router)
        ids,routing=torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2(logits,bias,bias_vl,mask)
        outputs=(updated,collapsed,pre,normalized,quantized,act_scale,ids,routing,gates)
        if args.weighted_post_statistics:
            product=torch.ops.hpu.fp8_gemm_v2(dense_q,False,shared_weight,True,None,torch.float32,None,None,None,False)
            middle,middle_scale=torch.ops.custom_op.custom_deepseek_v41_shared_silu_quant_gaudi2(
                product.reshape(1,1,-1),shared_id,dense_scale,shared_channel,shared_route)
            control_next=torch.ops.custom_op.custom_deepseek_v41_control_rrms_swizzled_bf16_gaudi2(updated.flatten(1),next_weight,eps)
            gates_next=torch.ops.custom_op.custom_deepseek_v41_mhc_gates_positive_gaudi2(
                control_next[:,:24].contiguous(),control_next[:,24:].contiguous(),next_scale,next_base)
            outputs+=(middle,middle_scale,control_next,gates_next)
        return outputs
    arms = (True, False) if args.early_gates else (True, True) if args.late_control or args.swizzled_control or args.direct_rrms_post or args.dense_transpose or args.bf16_control_weights else (False, False) if args.peer_prune else (False, True)
    if args.gates_during_exchange:
        arms = (False, False) if args.k_tiled_controller or args.positive_gates else (True, False)
    independent_gates = compiled(lambda projection, scale, base:
        torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(
            projection[:, :24].contiguous(), projection[:, 24:25].contiguous(), scale, base)
    ) if args.gates_during_exchange else None
    if args.peer_gates_vector:
        independent_gates = compiled(lambda projection,scale,base:torch.ops.custom_op.custom_deepseek_v41_mhc_gates_positive_gaudi2(projection[:,:24].contiguous(),projection[:,24:25].contiguous(),scale,base))
    positive_gates=compiled(lambda projection,scale,base:torch.ops.custom_op.custom_deepseek_v41_mhc_gates_positive_gaudi2(projection[:,:24].contiguous(),projection[:,24:25].contiguous(),scale,base)) if args.positive_gates else None
    late_projection = compiled(lambda x,w,s:direct_dense_fp8(x,w,s)) if args.late_control or args.k_controller_peer_window else None
    late_controller = (compiled(lambda r,w:
        torch.ops.custom_op.custom_deepseek_v41_control_k_tiled_bf16_gaudi2(r.flatten(1),w,eps))
        if args.k_controller_peer_window else compiled(lambda r,w:
        torch.ops.custom_op.custom_deepseek_v41_control_rrms_parallel_bf16_gaudi2(r.flatten(1),w,eps))
        if args.late_control else None)
    producers=[compiled(lambda *args,fused=fused:produce(*args,fused=fused)) for fused in arms]
    consumers=[compiled(lambda *values,fused=fused,positive=bool((args.direct_rrms_post or args.peer_gates_vector) and arm):consume(*values,fused=fused,positive=positive)) for arm,fused in enumerate(arms)]
    prune_setter = None
    if args.peer_prune:
        import ctypes
        hcl = ctypes.CDLL('libhcl.so')
        prune_setter = hcl.hcclSetSingleBoxGatherPruneExperimental
        prune_setter.argtypes, prune_setter.restype = [ctypes.c_int], None
    plans, graphs, visible, externals = [], [], [], []
    for arm, (producer,consumer) in enumerate(zip(producers,consumers,strict=True)):
        if prune_setter is not None:
            torch.hpu.synchronize()
            torch.distributed.barrier()
            prune_setter(arm)
        plan = bridge.PreparedGroupPlan()
        slots, tensors, external = {}, {}, []
        recorded_nodes = []

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
            from vllm_gaudi.compilation.deepseek_v41_compiler_config import compiler_configuration
            policy = {}
            with compiler_configuration(policy):
                fn(*operands)
                torch.hpu.synchronize()
                recorder.calls = []
                result = fn(*operands)
                torch.hpu.synchronize()
            calls, recorder.calls = recorder.calls, None
            if not calls:
                raise RuntimeError('Missing physical recipe recording')
            for recipe, inputs, outputs in calls:
                input_slots, output_slots = [slot(v, True) for v in inputs], [slot(v, False) for v in outputs]
                plan.add_compute(recipe, input_slots, output_slots)
                recorded_nodes.append(('compute', recipe, input_slots, output_slots))
            return result

        if (args.late_control or args.k_controller_peer_window) and arm == 1:
            value=compute(late_projection,(x,wo_weight,wo_scale))
            projection=None
        else:
            arm_control = (bf16_control if (args.bf16_control_weights or args.k_tiled_controller) and arm == 1 else
                           packed_control if args.swizzled_control and (arm == 1 or args.gates_during_exchange) else control)
            value,projection=compute(producer,(x,residual,arm_control,mme_weight,transposed_wo if args.dense_transpose and arm == 1 else wo_weight,wo_scale))
        peers=torch.ops.vllm_gaudi.tp_peer_allgather(value,4)
        torch.hpu.synchronize()
        gather_slots=(slot(value,False),slot(peers,False))
        plan.add_all_gather(*gather_slots)
        recorded_nodes.append(('gather', *gather_slots))
        if (args.late_control or args.k_controller_peer_window) and arm == 1:
            projection=compute(late_controller,(residual,bf16_control if args.k_controller_peer_window else control))
        if args.gates_during_exchange and (arm == 1 or args.k_tiled_controller or args.positive_gates):
            # This recipe reads only control/scale/base. It must not bind peers:
            # the native tensor dependency plan can defer the collective wait
            # until the subsequent post consumer actually needs peer output.
            projection=compute(positive_gates if args.positive_gates and arm==1 else independent_gates,(projection,control_scale,control_base))
        outputs=tuple(compute(consumer,(peers.reshape(4,1,5120),residual,projection,
            control_scale,control_base,norm,router_weight,bias,bias_vl,mask)))
        repeats_per_group = 8 if args.late_control or args.gates_during_exchange else 1
        for _ in range(repeats_per_group - 1):
            for operation in recorded_nodes:
                if operation[0] == 'compute':
                    plan.add_compute(*operation[1:])
                else:
                    plan.add_all_gather(*operation[1:])
        plan.prepare(backend, [slot(value, False) for value in outputs])
        graph = bridge.NativeDecodeGraph()
        graph.configure_topology(args.chain_repeats // repeats_per_group, args.chain_repeats, False)
        graph.configure_dependency_policy((args.late_control or args.gates_during_exchange) and (arm == 1 or args.k_tiled_controller or args.positive_gates))
        graph.capture([plan] * (args.chain_repeats // repeats_per_group),
                      [external] * (args.chain_repeats // repeats_per_group))
        graph.instantiate()
        graph.bind_dynamic_inputs([x,residual])
        plans.append(plan)
        graphs.append(graph)
        visible.append(outputs)
        externals.append(external)
    if prune_setter is not None:
        prune_setter(0)
    tickets, events = [], []
    failure = None
    try:
        checks = []
        for index, fixture in enumerate(fixtures):
            for destination, value in zip((x,residual,post,comb,pre), fixture, strict=True):
                destination.copy_(value.to('hpu'))
            torch.hpu.synchronize()
            observed=[]
            for graph,outputs in zip(graphs,visible,strict=True):
                graph.replay_fixed_with_completion().synchronize()
                observed.append([value.cpu() for value in outputs])
            exact=[torch.equal(a.view(torch.uint8),b.view(torch.uint8))
                   for a,b in zip(*observed,strict=True)]
            # Check the new TPC byte-for-byte at an identical projection; only
            # the explicit hi/lo controller is permitted to introduce error.
            control_reference=(torch.ops.custom_op.custom_deepseek_v41_control_rrms_parallel_bf16_gaudi2
                               if args.late_control or args.swizzled_control or args.direct_rrms_post or args.early_gates or args.dense_transpose or args.bf16_control_weights else
                               torch.ops.custom_op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2)
            reference_control=control_reference(residual.flatten(1),control,eps)
            mix=torch.cat((reference_control[:,:24],torch.zeros_like(reference_control[:,:24])),1).contiguous()
            value=direct_dense_fp8(x,wo_weight,wo_scale)
            peers=torch.ops.vllm_gaudi.tp_peer_allgather(value,4).reshape(4,1,5120)
            if args.peer_gates_vector:
                reference_projection=torch.ops.custom_op.custom_deepseek_v41_mhc_gates_positive_gaudi2(
                    reference_control[:,:24].contiguous(),reference_control[:,24:].contiguous(),control_scale,control_base)
            else:
                reference_projection=reference_control if args.gates_during_exchange else mix
            pure=consume(peers,residual,reference_projection,
                         control_scale,control_base,norm,router_weight,bias,bias_vl,mask,fused=not args.peer_gates_vector)
            torch.hpu.synchronize()
            exact_tpc=[torch.equal(a.view(torch.uint8),b.cpu().view(torch.uint8)) for a,b in zip(observed[0],pure,strict=True)]
            if args.gates_during_exchange:
                mme_gates = observed[1][8]
            else:
                mme_gates,*_=torch.ops.custom_op.custom_deepseek_v41_mhc_mme_gates_norm_gaudi2(
                    torch.ops.custom_op.custom_deepseek_v41_control_mme_f32_gaudi2(residual.flatten(1),mme_weight),
                    residual.flatten(1),visible[0][1],norm,control_scale,control_base,eps)
            reference_gates=torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(
                reference_control[:,:24].contiguous(),reference_control[:,24:].contiguous(),control_scale,control_base)
            gate_error=(reference_gates.cpu()-mme_gates.cpu()).abs()
            entry=dict(input=index,outputs_exact=exact,same_projection_tpc_exact=exact_tpc,
                mme_gates_max_abs=float(gate_error.max()),router_ids_exact=exact[6],
                residual_max_abs=float((observed[0][0].float()-observed[1][0].float()).abs().max()))
            if args.official_tolerance:
                from tools.deepseek_v41_mhc_reference import post_reference, check_outputs, normalized_error
                reference = post_reference(peers.cpu(), residual.cpu(), control.cpu(),
                                           control_scale.cpu(), control_base.cpu(), norm.cpu(), eps)
                entry['official_equation_errors'] = [check_outputs(arm, reference) for arm in observed]
                entry['remaining_output_errors'] = {
                    str(i): dict(error=normalized_error(observed[1][i].float(), observed[0][i].float()),
                                 limit=2e-4 if i in (4,9) else 5e-5)
                    for i in ((4,5,7,9,10,11,12) if args.weighted_post_statistics else (4,5,7))}
                errors = [v for arm in entry['official_equation_errors'] for v in arm.values()]
                errors += list(entry['remaining_output_errors'].values())
                decisions = [None] * 4
                torch.distributed.all_gather_object(decisions, all(row['error'] < row['limit'] for row in errors))
                if not all(decisions):
                    (directory/'failure.json').write_text(json.dumps(entry, indent=2)+'\n')
                    raise RuntimeError('mHC official-equation numerical contract failed')
            checks.append(entry)
            if args.weighted_post_statistics:
                # The compiled shared MME and the independent eager reference
                # can use different legal accumulation orders. Apply the same
                # official downstream error contract; retain exact post state.
                pure_cpu=[tensor.cpu() for tensor in pure]
                entry['same_projection_downstream_errors']={
                    str(i):dict(error=normalized_error(observed[0][i].float(),pure_cpu[i].float()),
                                limit=2e-4 if i==9 else 5e-5)
                    for i in (9,10,11,12)}
                for row in entry['same_projection_downstream_errors'].values():
                    if row['error']>=row['limit']:
                        raise RuntimeError('Independent downstream reference tolerance failed')
                exact_prefix=all(exact_tpc[:9])
            else:
                exact_prefix=all(exact_tpc)
            if (not exact_prefix or not exact[6] or
                    (not args.official_tolerance and not torch.allclose(reference_gates.cpu(),mme_gates.cpu(),rtol=1e-5,atol=1e-5))):
                torch.save(dict(fixture=fixture,observed=observed),directory/'failure.pt')
                (directory/'failure.json').write_text(json.dumps(entry,indent=2)+'\n')
                raise RuntimeError(f'mHC fused post correctness failed: {entry}')
        (directory/"result.json").write_text(json.dumps(dict(status="correctness_checked",checks=checks))+"\n")
        # Retain the exact gate for exact-controller/communication experiments.
        # MME may opt into the upstream equation/metric gate; report both errors.
        if not args.official_tolerance and not all(all(row["outputs_exact"]) for row in checks):
            raise RuntimeError("MME controller changed observable outputs; not bit-exact qualified")
        if args.require_controller_sram:
            from tools.audit_deepseek_v41_physical_nodes import audit
            inspected=[]
            for path in (root/'graphs'/f'rank{rank}').rglob('*PostGraph-symbol.pbtxt'):
                graph_data=audit(path)
                partial=[n for n in graph_data['nodes'] if n['op']=='custom_deepseek_v41_control_k_partial_gaudi2']
                finish=[n for n in graph_data['nodes'] if n['op']=='custom_deepseek_v41_control_k_finish_gaudi2']
                if not partial: continue
                passed=(len(partial)==1 and len(finish)==1 and
                    'location = in SRAM' in partial[0]['tensors']['outputTensor:0'] and
                    'location = in SRAM' in finish[0]['tensors']['inputTensor:0'] and
                    partial[0]['tensors']['outputTensor:0'].split('|')[0].strip()==
                    finish[0]['tensors']['inputTensor:0'].split('|')[0].strip())
                inspected.append(dict(graph=graph_data['graph'],passed=passed,physical_nodes=graph_data['physical_nodes']))
            (directory/'sram-gate.json').write_text(json.dumps(inspected,indent=2)+'\n')
            decisions=[None]*4
            torch.distributed.all_gather_object(decisions,bool(inspected) and all(v['passed'] for v in inspected))
            if not all(decisions):
                raise RuntimeError('K-controller SRAM producer/consumer contract failed before timing')
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
            (directory/'result.json').write_text(json.dumps(dict(status='measuring',checks=checks,periods=periods))+'\n')
            if rank == 0:print(json.dumps(dict(arm=label,median_ms=periods[-1]['median_ms'])),flush=True)
        if graph_compilation_count() != compiled_before:
            raise RuntimeError('Hot compilation invalidates timing')
        savings = [a['median_ms']-b['median_ms'] for a, b in zip(periods[::2], periods[1::2], strict=True)]
        result = dict(status='retirement_check' if args.retirement_only else 'completed', checks=checks, periods=periods,
                      saving_ms_per_layer=statistics.median(savings) if savings else None,
                      round_savings_ms=savings,
                      three_consistent_rounds=bool(savings) and all(value > 0 for value in savings),
                      full_model_gain_credit=False, physical_node_gate_pending=True, chain_repeats=args.chain_repeats,
                      input_source=__doc__, native_replay=True, k_tiled_controller=args.k_tiled_controller, k_controller_peer_window=args.k_controller_peer_window, positive_gates=args.positive_gates,
                      native_library_sha256=hashlib.sha256(Path(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY']).read_bytes()).hexdigest())
        (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        if rank == 0:
            (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    except BaseException as error:
        import traceback
        (directory/'validation-error.txt').write_text(traceback.format_exc())
        failure = RuntimeError(str(error))
        (directory/'result.json').write_text(json.dumps(dict(status='failed_before_retirement',
            error=str(error), performance_valid=False))+'\n')
        if rank == 0:
            (root/'result.json').write_text((directory/'result.json').read_text())
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
    consumers.clear()
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
    if failure is not None:
        raise failure


if __name__ == '__main__':
    main()
