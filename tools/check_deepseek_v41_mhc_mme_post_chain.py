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
    control = shard.tensor(f'layers.{layer}.hc_ffn_fn','hpu').contiguous()
    high = control.bfloat16()
    low = (control - high.float()).bfloat16()
    mme_weight = torch.cat((high,low),0).contiguous()
    control_scale = shard.tensor(f'layers.{layer}.hc_ffn_scale','hpu')
    control_base = shard.tensor(f'layers.{layer}.hc_ffn_base','hpu')
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
    def produce(x,residual,control,mme_weight,wo_weight,wo_scale,*,fused):
        flat=residual.flatten(1)
        projected=(torch.ops.custom_op.custom_deepseek_v41_control_rrms_unpack_bf16_gaudi2(flat,control,eps)
            if fused and args.unpack_controller else
            torch.ops.custom_op.custom_deepseek_v41_control_mme_f32_gaudi2(flat,mme_weight) if fused and not args.exact_controller else
            torch.ops.custom_op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(flat,control,eps))
        if fused and args.parallel_controller:
            projected = torch.ops.custom_op.custom_deepseek_v41_control_rrms_parallel_bf16_gaudi2(flat, control, eps)
        if fused and args.mme_shared_rrms:
            projected = torch.ops.custom_op.custom_deepseek_v41_control_mme_finish_gaudi2(flat, projected, eps)
        value=direct_dense_fp8(x,wo_weight,wo_scale)
        if fused:return value,projected
        gates=torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(
            projected[:,:24].contiguous(),projected[:,24:].contiguous(),control_scale,control_base)
        return value,gates
    def consume(peers,residual,projection,scale,base,norm,router,bias,bias_vl,mask,*,fused):
        if fused:
            # The same TPC owns fixed-rank summation and its BF16 boundary;
            # do not materialize a separate collective reduction node.
            updated,collapsed,gates=torch.ops.custom_op.custom_deepseek_v41_mhc_mme_post_collapse_gaudi2(
                peers,residual,projection,scale,base,eps)
            pre=gates[:,:4].contiguous()
        else:
            gates=projection
            pre=projection[:,:4].contiguous()
            # Match the already-qualified peer/post baseline. Counting a
            # separate rank sum here would double-credit the preceding batch.
            updated,collapsed=torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
                peers,residual,projection[:,4:8].contiguous(),projection[:,8:].reshape(1,4,4).contiguous(),pre)
        normalized,quantized,act_scale=torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(collapsed,norm,eps)
        logits=torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(normalized,router)
        ids,routing=torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2(logits,bias,bias_vl,mask)
        return updated,collapsed,pre,normalized,quantized,act_scale,ids,routing,gates
    arms = (False, False) if args.peer_prune else (False, True)
    producers=[compiled(lambda *args,fused=fused:produce(*args,fused=fused)) for fused in arms]
    consumers=[compiled(lambda *args,fused=fused:consume(*args,fused=fused)) for fused in arms]
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

        value,projection=compute(producer,(x,residual,control,mme_weight,wo_weight,wo_scale))
        peers=torch.ops.vllm_gaudi.tp_peer_allgather(value,4)
        torch.hpu.synchronize()
        plan.add_all_gather(slot(value,False),slot(peers,False))
        outputs=tuple(compute(consumer,(peers.reshape(4,1,5120),residual,projection,
            control_scale,control_base,norm,router_weight,bias,bias_vl,mask)))
        plan.prepare(backend, [slot(value, False) for value in outputs])
        graph = bridge.NativeDecodeGraph()
        graph.configure_topology(args.chain_repeats, args.chain_repeats, False)
        graph.configure_dependency_policy(False)
        graph.capture([plan] * args.chain_repeats, [external] * args.chain_repeats)
        graph.instantiate()
        graph.bind_dynamic_inputs([x,residual])
        plans.append(plan)
        graphs.append(graph)
        visible.append(outputs)
        externals.append(external)
    if prune_setter is not None:
        prune_setter(0)
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
        reference_control=torch.ops.custom_op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(residual.flatten(1),control,eps)
        mix=torch.cat((reference_control[:,:24],torch.zeros_like(reference_control[:,:24])),1).contiguous()
        value=direct_dense_fp8(x,wo_weight,wo_scale)
        peers=torch.ops.vllm_gaudi.tp_peer_allgather(value,4).reshape(4,1,5120)
        pure=consume(peers,residual,mix,control_scale,control_base,norm,router_weight,bias,bias_vl,mask,fused=True)
        torch.hpu.synchronize()
        exact_tpc=[torch.equal(a.view(torch.uint8),b.cpu().view(torch.uint8)) for a,b in zip(observed[0],pure,strict=True)]
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
                             limit=2e-4 if i == 4 else 5e-5)
                for i in (4, 5, 7)}
            errors = [v for arm in entry['official_equation_errors'] for v in arm.values()]
            errors += list(entry['remaining_output_errors'].values())
            if any(not row['error'] < row['limit'] for row in errors):
                (directory/'failure.json').write_text(json.dumps(entry, indent=2)+'\n')
                raise RuntimeError('mHC official-equation numerical contract failed')
        checks.append(entry)
        if not all(exact_tpc) or not torch.allclose(reference_gates.cpu(),mme_gates.cpu(),rtol=1e-5,atol=1e-5) or not exact[6]:
            torch.save(dict(fixture=fixture,observed=observed),directory/'failure.pt')
            (directory/'failure.json').write_text(json.dumps(entry,indent=2)+'\n')
            raise RuntimeError(f'mHC fused post correctness failed: {entry}')
    (directory/"result.json").write_text(json.dumps(dict(status="correctness_checked",checks=checks))+"\n")
    # Retain the exact gate for exact-controller/communication experiments.
    # MME may opt into the upstream equation/metric gate; report both errors.
    if not args.official_tolerance and not all(all(row["outputs_exact"]) for row in checks):
        raise RuntimeError("MME controller changed observable outputs; not bit-exact qualified")
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


if __name__ == '__main__':
    main()
