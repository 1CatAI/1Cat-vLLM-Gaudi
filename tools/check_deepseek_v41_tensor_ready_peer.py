# SPDX-License-Identifier: Apache-2.0
"""Four-rank tensor-signal/replay epoch capability with a real consumer.

This isolates readiness semantics, not the complete production independent
mHC branch. It must not receive performance or pending-ledger credit.
"""
import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--peer-fixtures', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--payload-marker', action='store_true')
    parser.add_argument('--ordinary-completion', action='store_true',
                        help='Keep the marked recipe but let the NIC wait for final producer completion')
    parser.add_argument('--drain-input-copies', action='store_true',
                        help='Untimed diagnostic: complete framework input copies before native submission')
    parser.add_argument('--clone-readback', action='store_true',
                        help='Untimed diagnostic: read a fresh clone without normalizing captured tensor storage')
    parser.add_argument('--production-roundtrip', action='store_true',
                        help='Include the shared BF16 wo_b group32 activation producer in both arms and reference')
    args = parser.parse_args()
    rank, tp = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    if tp != 4:
        raise ValueError('This communication capability uses the production four-rank group')
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = os.environ.get('PT_HPU_RECIPE_CACHE_CONFIG', '').format(rank=rank)
    os.environ['VLLM_HPU_DSV41_TP_MHC_OVERLAP'] = '0'
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

    prepare_environment(args.prepared, tensor_parallel_size=tp, pipeline_parallel_size=1)
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    import torch.distributed as dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment, initialize_model_parallel, destroy_model_parallel, get_tp_group,
    )
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_math import hc_pre, hc_post, rms_norm
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.tp2_graph_inputs import FixedDecodeInputs
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
    from vllm_gaudi.ops import tp2_prepared_plan as plans

    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    init_distributed_environment(world_size=tp, rank=rank, distributed_init_method='env://', local_rank=rank,
                                 backend='hccl')
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    report = dict(rank=rank, status='running', cases=[], performance_measured=False)
    graphs = []
    path = root / f'tensor-ready-rank{rank}.json'

    def readback(value):
        return value.clone().cpu() if args.clone_readback else value.cpu()

    def layout(values):
        getter = getattr(torch.ops.custom_op, 'deepseek_v41_peer_storage_layout', None)
        return [getter(value) for value in values] if getter is not None else []
    try:
        with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp))), \
                torch.inference_mode():
            initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            plans.register_tp2_prepared_group_pass()
            bind_worker_helpers(rank)
            from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators

            load_native_operators()
            torch.ops.load_library(args.library)
            bridge, _ = plans._runtime()
            shard = PreparedV41Shard(args.prepared, pp_rank=0, tp_rank=rank)
            weights = shard.dense('layers.20.attn.wo_b.weight', 'cpu').bfloat16().contiguous().to('hpu')
            cw = shard.tensor('layers.20.hc_ffn_fn', 'cpu').bfloat16().contiguous().to('hpu')
            norm = shard.tensor('layers.20.ffn_norm.weight', 'hpu')
            router = shard.tensor('layers.20.ffn.gate.weight', 'hpu').float()
            fn, scale, base = [shard.tensor('layers.20.hc_attn_'+name, 'hpu')
                               for name in ('fn', 'scale', 'base')]
            fixtures = []
            for case in range(3):
                record = torch.load(args.fixtures / f'rank{rank}/c6-{case}.pt', weights_only=True,
                                    map_location='cpu')
                if not record['request_context_qualified'] or not record['groups_qualified']:
                    raise ValueError('Actual qualified request state required')
                packets = torch.load(args.peer_fixtures / f'peer-inputs-rank{rank}-case{case}.pt',
                                     weights_only=True, map_location='cpu')
                x = packets[0]['projection_input'].bfloat16().contiguous().to('hpu')
                residual = record['groups'][4]['residual'].to('hpu')
                previous = record['groups'][4]['pre'].to('hpu')
                _, pre, post, comb = hc_pre(residual, previous, fn, scale, base, 1e-20, 1e-6, 20,
                                            packed_fn=fn, decode=True, control_mme_weight=fn.bfloat16())
                flat = residual.float().flatten(1)
                high = flat.bfloat16()
                control = torch.cat((high, (flat-high.float()).bfloat16())).contiguous()
                fixtures.append((x, control, residual, pre.contiguous(), post.contiguous(), comb.contiguous()))
            torch.hpu.synchronize()
            calls, retained = [], []
            for early in (False, True):
                owner = torch.nn.Module()
                metadata = SimpleNamespace(native_completion=None)
                fixed = tuple(value.clone() for value in fixtures[0])

                def body(x, control, residual, pre, post, comb, early=early):
                    if args.production_roundtrip:
                        x = torch.ops.custom_op.custom_deepseek_v41_quant_roundtrip_bf16_gaudi2(x)
                    partial, projected_control = torch.ops.custom_op.custom_deepseek_v41_peer_signal_probe_gaudi2(
                        x, weights, control, cw, early and not args.payload_marker)
                    if args.payload_marker:
                        partial = torch.ops.custom_op.custom_deepseek_v41_peer_ready_identity_gaudi2(
                            partial, early)
                    peers = torch.ops.vllm_gaudi.tp_peer_allgather(partial.reshape(1, -1).contiguous(), tp)
                    shards = peers.reshape(tp, 6, 5120)
                    summed = shards[0].float()
                    for index in range(1, tp):
                        summed = summed+shards[index].float()
                    reduced = summed.bfloat16()
                    output = hc_post(reduced, residual, post, comb)
                    collapsed = (output.float()*pre.unsqueeze(-1)).sum(1).bfloat16()
                    normalized = rms_norm(collapsed, norm, 1e-20)
                    scores = torch.nn.functional.linear(normalized.float(), router)
                    # The shared prepared-owner contract requires two peer
                    # boundaries. Retain a real downstream projection consumer
                    # and its communication in both arms; only the first peer
                    # point uses the external tensor-ready signal.
                    score_peers = torch.ops.vllm_gaudi.tp_peer_allgather(
                        scores.bfloat16().reshape(1, -1).contiguous(), tp).reshape(tp, 6, -1)
                    combined_scores = score_peers[0].float()
                    for index in range(1, tp):
                        combined_scores = combined_scores+score_peers[index].float()
                    return combined_scores, projected_control, reduced, partial

                compiled = torch.compile(body, backend='hpu_backend', fullgraph=True, dynamic=False)
                adapter = DecoderTopology('deepseek_v41_tensor_ready_probe', (1,), 2, False)
                roots = dict(hidden_states=fixed[0], pre_mix=None, positions=None, input_ids=None,
                             attention_inputs=fixed[1:], metadata=metadata, state_generation=1, state_tensors=())
                pending = None
                for _ in range(2):
                    with plans.collect_prepared_group_replays(owner=owner, adapter=adapter,
                                                              diagnostic_host_replay=True, **roots) as context:
                        context['group_index'] = 0
                        outputs = compiled(*fixed)
                        plans.record_native_decoder_outputs(*outputs)
                        pending = list(plans._local.pending)
                    torch.hpu.synchronize()
                if len(pending) != 1:
                    raise AssertionError('Capability must capture one complete prepared owner')
                native = bridge.NativeDecodeGraph()
                native.configure_topology(1, 2, False)
                native.configure_tensor_ready_signals([1 if early and not args.ordinary_completion else 0, 0])
                inputs = [row[1] for row in pending]
                native.capture([row[0] for row in pending], inputs)
                native.instantiate()
                bindings = FixedDecodeInputs(owner, dict(roots, adapter=adapter), inputs, native_bridge=bridge)
                native.bind_dynamic_inputs(bindings.tensors())
                graphs.append(native)
                retained.append((owner, fixed, bindings, roots, outputs))

                def call(values, native=native, bindings=bindings, roots=roots, outputs=outputs):
                    dist.barrier(group=get_tp_group().cpu_group)
                    dynamic = dict(roots, hidden_states=values[0], attention_inputs=values[1:])
                    updates = bindings.updates(dynamic)
                    if updates is None:
                        raise AssertionError('Signal capability changed its captured input allocations')
                    if args.drain_input_copies:
                        for destination, source in updates:
                            destination.copy_(source)
                        torch.hpu.synchronize()
                    else:
                        bindings.apply(updates, native)
                    completion = native.replay_fixed_with_completion()
                    # Native submission is independent of the framework D2H
                    # stream. This is an untimed correctness check: retain and
                    # wait for the final owner before reading any intermediate
                    # output, as the production readback protocol does.
                    completion.synchronize()
                    before = layout((*bindings.tensors(), *outputs))
                    checks = [dict(source=binding.source, name=binding.name,
                                   exact=torch.equal(readback(binding.destination), readback(binding.read(dynamic))))
                              for binding in bindings.bindings]
                    report.setdefault('fixed_input_checks', []).append(checks)
                    result = tuple(readback(value) for value in outputs)
                    report.setdefault('storage_layouts', []).append(
                        dict(before_readback=before, after_readback=layout((*bindings.tensors(), *outputs))))
                    return result

                calls.append(call)
            report['signal_metadata'] = graphs[1].tensor_ready_info()
            report['all_signal_metadata'] = [graph.tensor_ready_info() for graph in graphs]
            report['ordinary_completion'] = args.ordinary_completion
            report['drain_input_copies'] = args.drain_input_copies
            report['clone_readback'] = args.clone_readback
            report['production_roundtrip'] = args.production_roundtrip
            expected = []
            for fixture in fixtures:
                x = fixture[0]
                if args.production_roundtrip:
                    x = torch.ops.custom_op.custom_deepseek_v41_quant_roundtrip_bf16_gaudi2(x)
                partial, control = torch.ops.custom_op.custom_deepseek_v41_peer_signal_probe_gaudi2(
                    x, weights, fixture[1], cw, False)
                torch.hpu.synchronize()
                peers = [None]*tp
                dist.all_gather_object(peers, partial.cpu(), group=get_tp_group().cpu_group)
                summed = peers[0].float()
                for peer in peers[1:]:
                    summed = summed+peer.float()
                expected.append((summed.bfloat16(), control.cpu(), partial.cpu()))
            # Changing actual prefixes and replay epochs expose stale early
            # targets; a signal-ready result must match the complete producer.
            for case in (0, 1, 2, 2, 1, 0):
                values = [call(fixtures[case]) for call in calls]
                errors = [dict(exact=torch.equal(a, b), maximum_abs=float((a.float()-b.float()).abs().max()))
                          for a, b in zip(*values, strict=True)]
                reference_errors = [dict(reduced_exact=torch.equal(value[2], expected[case][0]),
                                         reduced_within_mme_tolerance=torch.allclose(
                                             value[2].float(), expected[case][0].float(),
                                             atol=0.015625, rtol=0.015625),
                                         reduced_max_abs=float((value[2].float()-expected[case][0].float()).abs().max()),
                                         local_partial_max_abs=float(
                                             (value[3].float()-expected[case][2].float()).abs().max()),
                                         control_exact=torch.equal(value[1], expected[case][1])) for value in values]
                report['cases'].append(dict(case=case, errors=errors, ordinary_reference=reference_errors))
                path.write_text(json.dumps(report, indent=2)+'\n')
                if (not all(error['exact'] for error in errors)
                        or not all(error['reduced_within_mme_tolerance'] and error['control_exact']
                                   for error in reference_errors)):
                    torch.save(dict(case=case, arms=values, expected=expected[case]),
                               root/f'signal-mismatch-rank{rank}.pt')
                    raise AssertionError('Early peer signal changes the native producer/consumer result')
            report.update(status='passed', native_replays=[g.replay_count() for g in graphs],
                          signal_metadata=graphs[1].tensor_ready_info(),
                          payload_marker=args.payload_marker,
                          scope=('Native signal/epoch/peer semantics only; '
                                 'full production independent branch not measured'))
            for graph in graphs:
                graph.close()
            graphs.clear()
            plans.shutdown_prepared_group_plans()
            # Retire manual capture and compiler-owned roots while the device
            # and communicator are both still alive, instead of at interpreter
            # shutdown in an unspecified module destruction order.
            calls.clear()
            retained.clear()
            pending.clear()
            del call, compiled, native, bindings, inputs, outputs, fixed, owner
            torch._dynamo.reset()
            gc.collect()
            torch.hpu.synchronize()
            dist.barrier()
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2)+'\n')
        for graph in graphs:
            graph.close()
        if dist.is_initialized():
            destroy_model_parallel()
            dist.destroy_process_group()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
