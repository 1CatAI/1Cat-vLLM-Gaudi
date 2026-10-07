# SPDX-License-Identifier: Apache-2.0
"""Real BF16 LM head -> local nucleus packet -> peer -> sample -> embedding.

Both arms keep the full F32 normalizer, draw, peer wire and embedding consumer.
Only local Top128 construction changes. No host input updates in timed replay.
"""
import argparse
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--sidecar', type=Path, required=True)  # Shared launcher contract.
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fused-consumer', action='store_true', help='Hold qualified threshold producer in BOTH arms')
    parser.add_argument('--shared-max',
                        action='store_true',
                        help='Hold qualified fused producer/consumer parent, reuse one max/index pass')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--chain-repeats', type=int, default=16)
    args = parser.parse_args()
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
    from vllm.distributed import (init_distributed_environment, initialize_model_parallel, destroy_model_parallel,
                                  destroy_distributed_environment)
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime, _resolve_runtime
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_sampling import (local_nucleus_packet, sample_nucleus_packet,
                                                      sample_nucleus_packet_fused)
    from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
    from tools.deepseek_v41_micro_replay import RecipeRecorder
    from tools.deepseek_v41_resident_ab import graph_compilation_count, wait_for_loading

    torch.set_num_threads(1)
    torch.hpu.set_device(rank)
    root = args.output.resolve()
    directory = root / f'rank{rank}'
    directory.mkdir(parents=True, exist_ok=True)
    context = set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp)))
    context.__enter__()
    init_distributed_environment(world_size=tp,
                                 rank=rank,
                                 distributed_init_method='env://',
                                 local_rank=rank,
                                 backend='hccl')
    initialize_model_parallel(tensor_model_parallel_size=tp)
    initialize_tp2_fused_ar_norm_runtime()
    bridge, backend, _ = _resolve_runtime()
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    recorder = RecipeRecorder(directory, backend=backend, joint_only=True)
    graph_compilation_count()
    shard = PreparedV41Shard(args.prepared, 0, rank)
    head = shard.tensor('head.weight', 'hpu')
    final_norm = shard.tensor('norm.weight', 'hpu')
    embedding = shard.tensor('embed.weight', 'hpu')
    with safe_open(args.prepared / 'pp0-tp0.safetensors', framework='pt', device='cpu') as checkpoint:
        fixtures = [checkpoint.get_slice('embed.weight')[token:token + 1].clone() for token in (17, 41, 128, 512, 1024)]
    epsilon = json.loads((args.prepared / 'config.json').read_text())['text_config']['rms_norm_eps']
    columns = head.shape[0]
    position = torch.full((1, ), columns - 1, dtype=torch.int32, device='hpu')
    scratch = torch.zeros((1, 2048), dtype=torch.int32, device='hpu')
    controls = torch.tensor([[1., .95, .5, -1.]], dtype=torch.float32, device='hpu')
    x = fixtures[0].to('hpu')

    def produce(value, norm, weight, controls, position, scratch, *, threshold, shared_max):
        value = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2(value, norm, epsilon)
        logits = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(value, weight)
        packet = local_nucleus_packet(logits,
                                      controls,
                                      rank,
                                      128,
                                      threshold_state=(position, scratch) if threshold else None,
                                      shared_max=shared_max)
        # Same exact FP32 -> padded BF16 wire used by stage_collectives.
        wire = packet.view(torch.bfloat16)
        wire = F.pad(wire, (0, -wire.shape[-1] % 128))
        return wire, logits

    def consume(peers, controls, embedding, *, fused):
        wire_columns = ((2 * (3 + 2 * 128) + 127) // 128) * 128
        shards = peers.reshape(tp, wire_columns)
        packet = torch.cat(tuple(shards[r, :2 * (3 + 2 * 128)].view(torch.float32).reshape(1, -1) for r in range(tp)),
                           dim=-1)
        sampler = sample_nucleus_packet_fused if fused else sample_nucleus_packet
        selected, covered = sampler(packet, controls, tp_size=tp, width=128)
        covered = covered.to(torch.int32)
        # This is the real embedding consumer of the selected device token,
        # including the provisional token when ordinary coverage is false.
        next_hidden = F.embedding(selected.flatten().long(), embedding)
        return selected, covered, next_hidden

    compiler = make_backend(static_int32=True, static_factories=True)
    producers = [
        torch.compile(lambda *v, flag=flag: produce(
            *v, threshold=True if args.fused_consumer else flag, shared_max=flag and args.shared_max),
                      backend=compiler,
                      fullgraph=True,
                      dynamic=False) for flag in (False, True)
    ]
    consumers = [
        torch.compile(lambda *v, fused=fused: consume(*v, fused=args.shared_max or (fused and args.fused_consumer)),
                      backend=compiler,
                      fullgraph=True,
                      dynamic=False) for fused in (False, True)
    ]
    plans, graphs, outputs, externals = [], [], [], []
    for producer, consumer in zip(producers, consumers, strict=True):
        plan = bridge.PreparedGroupPlan()
        slots, tensors, external = {}, {}, []

        def slot(value, incoming, slots=slots, tensors=tensors, external=external, plan=plan):
            key = ((value.data_ptr(), tuple(value.shape), value.stride(),
                    value.dtype) if isinstance(value, torch.Tensor) else (type(value), value))
            if key not in slots:
                if isinstance(value, torch.Tensor) and value.is_contiguous():
                    for known, tensor in tensors.items():
                        if (tensor.data_ptr() == value.data_ptr() and tensor.dtype == value.dtype
                                and tensor.numel() == value.numel() and tensor.is_contiguous()):
                            slots[key] = plan.add_reshape_view(slots[known], list(value.shape))
                            return slots[key]
                slots[key] = plan.add_slot(value, incoming)
                if isinstance(value, torch.Tensor):
                    tensors[key] = value
                if incoming:
                    external.append(value)
            return slots[key]

        def compute(fn, operands, slot=slot, plan=plan):
            fn(*operands)
            torch.hpu.synchronize()
            recorder.calls = []
            out = fn(*operands)
            torch.hpu.synchronize()
            calls, recorder.calls = recorder.calls, None
            if not calls:
                raise RuntimeError('No native recipe recorded')
            for recipe, inputs, result in calls:
                plan.add_compute(recipe, [slot(v, True) for v in inputs], [slot(v, False) for v in result])
            return out

        wire, logits = compute(producer, (x, final_norm, head, controls, position, scratch))
        peers = torch.ops.vllm_gaudi.tp_peer_allgather(wire, tp)
        torch.hpu.synchronize()
        plan.add_all_gather(slot(wire, False), slot(peers, False))
        out = compute(consumer, (peers.reshape(1, -1), controls, embedding))
        plan.prepare(backend, [slot(v, False) for v in (*out, logits)])
        graph = bridge.NativeDecodeGraph()
        graph.configure_topology(args.chain_repeats, args.chain_repeats, False)
        graph.configure_dependency_policy(False)
        graph.capture([plan] * args.chain_repeats, [external] * args.chain_repeats)
        graph.instantiate()
        graph.bind_dynamic_inputs([x, controls])
        plans.append(plan)
        graphs.append(graph)
        outputs.append((*out, logits))
        externals.append(external)
    checks = []
    for index, fixture in enumerate(fixtures):
        x.copy_(fixture.to('hpu'))
        controls[:, 2:3].fill_((index + .5) / 5)
        torch.hpu.synchronize()
        values = []
        for graph, out in zip(graphs, outputs, strict=True):
            graph.replay_fixed_with_completion().synchronize()
            values.append([v.cpu() for v in out])
        exact = [
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(values[0][:3], values[1][:3], strict=True)
        ]
        packet_exact = torch.equal(values[0][3].view(torch.uint8), values[1][3].view(torch.uint8))
        checks.append(
            dict(fixture=index,
                 selected_coverage_embedding_exact=exact,
                 raw_logits_exact=packet_exact,
                 covered=bool(values[0][1].item())))
        agreement = torch.tensor([int(all(exact))], dtype=torch.int32, device='hpu')
        torch.distributed.all_reduce(agreement, op=torch.distributed.ReduceOp.MIN)
        (directory / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
        if not bool(agreement.cpu().item()):
            torch.save(dict(reference=values[0], candidate=values[1]), directory / 'failure.pt')
            raise RuntimeError('Sampler/embedding byte gate failed before timing')
    for graph in graphs:
        for _ in range(8):
            graph.replay_fixed_with_completion().synchronize()
    before = graph_compilation_count()
    periods = []
    tickets, events = [], []
    for label in 'ABABAB':
        wait_for_loading(directory, rank, torch.distributed)
        graph = graphs[label == 'B']
        events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)) for _ in range(args.steps)]
        torch.distributed.barrier()
        tickets = []
        for begin, end in events:
            begin.record()
            tickets.append(graph.replay_fixed_with_completion())
            end.record()
        torch.hpu.synchronize()
        local = [begin.elapsed_time(end) / args.chain_repeats for begin, end in events]
        ranks = [None] * tp
        torch.distributed.all_gather_object(ranks, local)
        periods.append(dict(arm=label, rank_device_ms=ranks, median_ms=max(statistics.median(v) for v in ranks)))
        if rank == 0:
            print(json.dumps(dict(arm=label, median_ms=periods[-1]['median_ms'])), flush=True)
    if graph_compilation_count() != before:
        raise RuntimeError('Hot compilation')
    savings = [a['median_ms'] - b['median_ms'] for a, b in zip(periods[::2], periods[1::2], strict=True)]
    result = dict(status='completed',
                  checks=checks,
                  periods=periods,
                  round_savings_ms=savings,
                  saving_ms_per_token=statistics.median(savings),
                  three_consistent_rounds=all(v > 0 for v in savings),
                  formal_gain=False,
                  scope=__doc__,
                  fused_consumer=args.fused_consumer)
    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if rank == 0:
        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    for graph, plan in zip(graphs, plans, strict=True):
        graph.close()
        plan.invalidate()
    graphs.clear()
    plans.clear()
    outputs.clear()
    externals.clear()
    tickets.clear()
    events.clear()
    graph = plan = slot = compute = None
    recorder.calls = None
    producers.clear()
    consumers.clear()
    consumer = producer = None
    torch.hpu.synchronize()
    for attribute in ('_vllm_gaudi_tp2_fused_ar_norm_runtime', '_vllm_gaudi_tp4_allreduce_runtime'):
        if hasattr(torch, attribute):
            delattr(torch, attribute)
    backend = None
    import gc
    gc.collect()
    destroy_model_parallel()
    destroy_distributed_environment()
    context.__exit__(None, None, None)


if __name__ == '__main__':
    import torch
    with torch.inference_mode():
        main()
