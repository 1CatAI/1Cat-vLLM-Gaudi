# SPDX-License-Identifier: Apache-2.0
"""Screen native TP2 peer transport for a possible four-rank doubling reduction.

Both arms compute the same TWO-rank partial sum, with real checkpoint WO
producer and post/norm consumer. Arm A receives all four rows, B just its peer.
This is transport headroom evidence, NOT a complete TP4 AllReduce candidate.
It cannot be credited as a model gain or deployed through serving.
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
    parser.add_argument('--chain-repeats', type=int, default=64)
    parser.add_argument('--retirement-only', action='store_true',
                        help='Reuse measured A/B; check teardown without timing')
    args = parser.parse_args()
    if args.chain_repeats < 1 or args.chain_repeats > 128:
        parser.error('native repetitions must be 1..128')
    os.environ['VLLM_HPU_DSV41_TP_MHC_OVERLAP'] = '0'
    rank = int(os.environ['LOCAL_RANK'])
    modules = os.environ['HABANA_VISIBLE_MODULES'].split(',')
    os.environ['HLS_MODULE_ID'] = modules[rank]
    if os.environ.get('DSV41_MICRO_RANK_CPUS'):
        os.sched_setaffinity(0, json.loads(os.environ['DSV41_MICRO_RANK_CPUS'])[rank])
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
    # All ranks create groups in one deterministic order. Initialize only the
    # member group, retaining exactly the normal direct-exchange contract.
    pair_groups = [torch.distributed.new_group(ranks=pair, backend='hccl')
                   for pair in ([0, 1], [2, 3])]
    pair_group = pair_groups[rank // 2]
    probe = torch.ones(128, device='hpu', dtype=torch.bfloat16)
    torch.distributed.all_reduce(probe, group=pair_group)
    torch.hpu.synchronize()
    if not torch.equal(probe.cpu(), torch.full((128,), 2., dtype=torch.bfloat16)):
        raise RuntimeError('Pair group initialization failed')
    pair_backend = pair_group._get_backend(torch.device('hpu'))
    pair_id = bridge.communicator_id(pair_backend)
    (args.output / f'pair-rank{rank}.json').write_text(json.dumps(dict(
        ranks=[2*(rank//2), 2*(rank//2)+1], communicator=pair_id)))
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    recorder = RecipeRecorder(directory, backend=backend, joint_only=True)
    graph_compilation_count()
    shard = PreparedV41Shard(args.prepared, 0, rank)
    sidecar = DenseFP8Sidecar(args.sidecar, shard)
    weight = sidecar.tensor('layers.0.attn.wo_b.weight', 'hpu')
    channel = sidecar.tensor('layers.0.attn.wo_b.channel_scale', 'hpu')
    norm = shard.tensor('layers.0.ffn_norm.weight', 'hpu')
    control, scale, base = [shard.tensor(f'layers.0.hc_attn_{name}', 'cpu') for name in ('fn', 'scale', 'base')]
    with safe_open(args.prepared/'pp0-tp0.safetensors', framework='pt', device='cpu') as checkpoint:
        embeddings = [checkpoint.get_slice('embed.weight')[token:token+1].clone()
                      for token in (17, 41, 128, 512, 1024)]
    fixtures = []
    for embedding in embeddings:
        residual = embedding.unsqueeze(1).expand(-1, 4, -1).contiguous()
        initial_pre = torch.tensor([[1., 0., 0., 0.]])
        _, pre, post, comb = hc_pre(residual, initial_pre, control, scale, base)
        row = embedding.roll(rank*127, dims=-1)[:, :weight.shape[1]].contiguous()
        fixtures.append((row, residual, post, comb, pre))
    x, residual, post, comb, pre = [tensor.to('hpu') for tensor in fixtures[0]]
    compiled = lambda fn: torch.compile(fn, backend='hpu_backend', fullgraph=True, dynamic=False)
    producer = compiled(lambda row, w, s: torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(row, w, s))

    def post_consume(value, r, p, c, n, norm_weight):
        updated, collapsed = torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(value, r, p, c, n)
        normalized, quantized, activation_scale = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
            collapsed, norm_weight, 1e-20)
        return updated, collapsed, normalized, quantized, activation_scale

    pair_start = 2*(rank//2)
    def consume_gather(peers, r, p, c, n, norm_weight):
        rows = peers.reshape(4, 1, 5120)
        value = (rows[pair_start].float() + rows[pair_start+1].float()).bfloat16()
        return post_consume(value, r, p, c, n, norm_weight)

    def consume_peer(local, remote, r, p, c, n, norm_weight):
        # A pair has only two BF16 partials. F32 addition is commutative here;
        # the proposed next round must separately retain F32 intermediate sums.
        value = (local.float() + remote.float()).bfloat16()
        return post_consume(value, r, p, c, n, norm_weight)

    consumers = [compiled(consume_gather), compiled(consume_peer)]
    plans, graphs, visible, externals = [], [], [], []
    for arm, consumer in enumerate(consumers):
        recorded_nodes = []
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
                input_slots, output_slots = [slot(v, True) for v in inputs], [slot(v, False) for v in outputs]
                plan.add_compute(recipe, input_slots, output_slots)
                recorded_nodes.append(('compute', recipe, input_slots, output_slots))
            return result

        partial = compute(producer, (x, weight, channel))
        if arm == 0:
            peers = torch.ops.vllm_gaudi.tp_peer_allgather(partial, 4)
            torch.hpu.synchronize()
            exchange_slots = (slot(partial, False), slot(peers, False))
            plan.add_all_gather(*exchange_slots)
            recorded_nodes.append(('gather', *exchange_slots))
            outputs = compute(consumer, (peers, residual, post, comb, pre, norm))
            plan_backend = backend
        else:
            peers = torch.empty_like(partial)
            bridge.tp2_exchange_peer_current_stream(pair_backend, partial, peers)
            torch.hpu.synchronize()
            exchange_slots = (slot(partial, False), slot(peers, False))
            plan.add_peer_exchange(*exchange_slots)
            recorded_nodes.append(('peer', *exchange_slots))
            outputs = compute(consumer, (partial, peers, residual, post, comb, pre, norm))
            plan_backend = pair_backend
        for _ in range(args.chain_repeats - 1):
            for operation in recorded_nodes:
                if operation[0] == 'compute':
                    plan.add_compute(*operation[1:])
                elif operation[0] == 'gather':
                    plan.add_all_gather(*operation[1:])
                else:
                    plan.add_peer_exchange(*operation[1:])
        plan.prepare(plan_backend, [slot(value, False) for value in outputs])
        graph = bridge.NativeDecodeGraph()
        graph.configure_topology(1, args.chain_repeats, False)
        graph.configure_dependency_policy(False)
        graph.capture([plan], [external])
        graph.instantiate()
        graph.bind_dynamic_inputs([x, residual, post, comb, pre])
        plans.append(plan)
        graphs.append(graph)
        visible.append(outputs)
        externals.append(external)
    checks = []
    for index, fixture in enumerate(fixtures):
        for destination, value in zip((x, residual, post, comb, pre), fixture, strict=True):
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
            raise RuntimeError(f'Peer post/collapse differs for input {index}: {exact}')
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
                            median_ms=statistics.median(max(values) for values in zip(*ranks, strict=True))/args.chain_repeats))
    if graph_compilation_count() != compiled_before:
        raise RuntimeError('Hot compilation invalidates timing')
    savings = [a['median_ms']-b['median_ms'] for a, b in zip(periods[::2], periods[1::2], strict=True)]
    result = dict(status='retirement_check' if args.retirement_only else 'completed', checks=checks, periods=periods,
                  saving_ms_per_boundary=statistics.median(savings) if savings else None,
                  round_savings_ms=savings,
                  three_consistent_rounds=bool(savings) and all(value > 0 for value in savings),
                  full_model_gain_credit=False, complete_tp4_reduction=False, chain_repeats=args.chain_repeats,
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
    recorder.backend = None
    for attribute in ('_vllm_gaudi_tp2_fused_ar_norm_runtime', '_vllm_gaudi_tp4_allreduce_runtime'):
        if hasattr(torch, attribute):
            delattr(torch, attribute)
    recorder.calls = None
    plan_backend = pair_backend = pair_group = None
    pair_groups.clear()
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
