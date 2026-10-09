# SPDX-License-Identifier: Apache-2.0
"""Native producer/peer/post/norm/Router chain; no serving/default selection."""
import argparse
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--chain-fixtures', type=Path)
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--iterations', type=int, default=32)
    parser.add_argument('--plan-library', type=Path, required=True)
    parser.add_argument('--operator-library', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--event-library', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment, prepare_native_libraries
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    prepare_native_libraries()
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = str(root / f'recipes/rank{rank}')+',false,4096'
    import ctypes
    import torch
    import torch.distributed as dist
    import habana_frameworks.torch.core  # noqa: F401
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (init_distributed_environment, initialize_model_parallel,
                                 destroy_model_parallel, get_tp_group)
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_math import hc_pre, quantize_activation

    bind_worker_cpu(rank)
    torch.set_grad_enabled(False)
    scope = None
    graphs = []
    report = dict(status='running', rank=rank, checks=[], timed=False, gain_credit_ms=0)
    output = root / f'future-plan-rank{rank}.json'

    def save():
        output.write_text(json.dumps(report, indent=2)+'\n')

    def raw(name, dtype, shape):
        data = bytearray((args.fixtures / f'rank{rank}' / name).read_bytes())
        return torch.frombuffer(data, dtype=dtype).reshape(shape).clone().to('hpu')

    try:
        load_native_operators()
        torch.ops.load_library(str(args.operator_library))
        os.environ['VLLM_HPU_DSV41_BOUNDED_RECEIVE_PARENT_KERNEL'] = os.environ['GC_KERNEL_PATH']
        os.environ['GC_KERNEL_PATH'] = str(args.database.resolve())
        report['GC_before_device_initialization'] = os.environ['GC_KERNEL_PATH']
        torch.hpu.set_device(rank)
        init_distributed_environment(world_size=4, rank=rank, distributed_init_method='env://', local_rank=rank,
                                     backend='hccl')
        scope = set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4)))
        scope.__enter__()
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
        initialize_tp2_fused_ar_norm_runtime()
        bind_worker_helpers(rank)
        hccl_paths = {line.split(None, 5)[5].strip() for line in Path('/proc/self/maps').read_text().splitlines()
                      if line.rstrip().endswith('/_hccl_eager_C.so')}
        if len(hccl_paths) != 1:
            raise RuntimeError(f'Initialized HCCL Bridge ownership ambiguous: {sorted(hccl_paths)}')
        hccl_path = next(iter(hccl_paths))
        hccl_owner = ctypes.CDLL(hccl_path, mode=ctypes.RTLD_GLOBAL)
        report['HCCL_owner'] = hccl_path
        helper = ctypes.CDLL(str(args.event_library), mode=ctypes.RTLD_GLOBAL)
        spec = importlib.util.spec_from_file_location('dsv41_future_plan', args.plan_library)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        shard = PreparedV41Shard(args.prepared, 0, rank)
        layers = [20] if args.chain_fixtures is None else [20, 21, 22, 23]
        if args.timing and args.chain_fixtures is None:
            raise ValueError('Timings require multiple actual weight working sets')
        peers = torch.zeros((2, 128, 4, 6, 5120), dtype=torch.bfloat16, device='hpu')
        flags = torch.zeros((1 << 24,), dtype=torch.int32, device='hpu')
        image = torch.zeros(6, dtype=torch.bool, device='hpu')
        chain = []
        for point, layer in enumerate(layers):
            weight = shard.dense(f'layers.{layer}.attn.wo_b.weight', 'cpu').bfloat16().contiguous().to('hpu')
            gate = shard.tensor(f'layers.{layer}.ffn.gate.weight', 'hpu').float().contiguous()
            norm = shard.tensor(f'layers.{layer}.ffn_norm.weight', 'hpu')
            bias = shard.tensor(f'layers.{layer}.ffn.gate.bias', 'hpu')
            bias_vl = shard.tensor(f'layers.{layer}.ffn.gate.bias_vl', 'hpu')
            controls = [shard.tensor(f'layers.{layer}.hc_ffn_{key}', 'hpu').float().contiguous()
                        for key in ('fn', 'scale', 'base')]
            control_mme = controls[0].bfloat16().contiguous()
            fixtures = []
            for case in range(3):
                if args.chain_fixtures is None:
                    values = tuple(raw(f'{key}-{case}.bin', dtype, shape) for key, dtype, shape in (
                        ('input', torch.bfloat16, (6, 2048)), ('residual', torch.bfloat16, (6, 4, 5120)),
                        ('post', torch.float32, (6, 4)), ('comb', torch.float32, (6, 4, 4)),
                        ('pre', torch.float32, (6, 4))))
                else:
                    record = torch.load(args.chain_fixtures / f'rank{rank}/layer{layer}-case{case}.pt',
                                        weights_only=True, map_location='cpu', mmap=True)
                    if not record['actual_request_context']:
                        raise ValueError('Production request inputs required')
                    values = tuple(record[key].to('hpu').contiguous()
                                   for key in ('projection_input', 'residual', 'post', 'comb', 'pre'))
                fixtures.append(values)
            fixed = tuple(v.clone() for v in fixtures[0])

            def make_functions(point, weight, gate, norm, bias, bias_vl, controls, control_mme):
                def producer(x):
                    return torch.nn.functional.linear(quantize_activation(x), weight)

                def consumer(value, residual, post, comb, pre):
                    residual_out, collapsed = torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
                        value.contiguous(), residual, post, comb, pre)
                    collapsed, pre_out, post_out, comb_out = hc_pre(
                        residual_out, pre, *controls, packed_fn=controls[0], decode=True,
                        collapsed_input=collapsed, control_mme_weight=control_mme)
                    normalized, quantized, scale = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
                        collapsed, norm, 1e-20)
                    logits = torch.nn.functional.linear(normalized.float(), gate)
                    return residual_out, normalized, quantized, scale, logits, pre_out, post_out, comb_out

                def base_post(local, residual, post, comb, pre, peer_values):
                    value = peer_values[0, point, 0].float()
                    for peer_rank in range(1, 4):
                        value = value + peer_values[0, point, peer_rank].float()
                    return consumer(value.bfloat16(), residual, post, comb, pre)

                def future_body(x, residual, post, comb, pre, peer_values, flag_values):
                    if point == 0:
                        x = torch.ops.custom_op.custom_deepseek_v41_future_epoch_gaudi2(x, flag_values)
                    local = producer(x)
                    value, packet, status = torch.ops.custom_op.custom_deepseek_v41_future_peer_sum_gaudi2(
                        local.contiguous(), peer_values, flag_values, flag_values, point, rank, 65536, True)
                    return (*consumer(value, residual, post, comb, pre), packet, status)

                def route(logits):
                    return torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2(
                        logits.contiguous(), bias, bias_vl, image)

                return [torch.compile(fn, backend='hpu_backend', fullgraph=True, dynamic=False)
                        for fn in (producer, base_post, future_body, route)]
            functions = make_functions(point, weight, gate, norm, bias, bias_vl, controls, control_mme)
            chain.append(dict(layer=layer, point=point, functions=functions, fixed=fixed, fixtures=fixtures,
                              weights=[weight, gate, norm, bias, bias_vl, *controls, control_mme]))
        working_set = sum(v.numel()*v.element_size() for c in chain for v in c['weights'])
        report.update(layers=layers, working_set_weight_bytes=working_set,
                      boundary=('real WO_B quantization/projection + peer + C1 post/norm '
                                '+ production MME mHC + FP32 Router + route consumer'),
                      scope='Saved per-layer producer inputs; sequential native chains, not full model forward')
        backend = dist.distributed_c10d._get_default_group()._get_backend(torch.device('hpu'))
        # Compile with prefilled packets. Every scored replay uses native
        # communication and changing device generations; warm/copies are outside timers.
        flags[:256].fill_(2)
        flags[-1] = 0
        for c in chain:
            functions, fixed, point = c['functions'], c['fixed'], c['point']
            partial = functions[0](fixed[0])
            torch.hpu.synchronize()
            dist.all_gather_into_tensor(peers[0, point].reshape(24, 5120), partial,
                                        group=get_tp_group().device_group)
            warm_base = functions[1](partial, *fixed[1:], peers)
            warm_future = functions[2](*fixed, peers, flags)
            functions[3](warm_base[4])
            functions[3](warm_future[4])
        torch.hpu.synchronize()
        for future in (False, True):
            flags[-1] = 0
            flags[:256].fill_(2)
            torch.hpu.synchronize()
            plan = native.Plan()
            graphs.append(plan)
            plan.begin()
            output_groups, packets, retained = [], [], [image]
            producers, consumers, fragments = [], [], []
            for c in chain:
                functions, fixed = c['functions'], c['fixed']
                start_segment = plan.info()[0]
                if future:
                    fields = functions[2](*fixed, peers, flags)
                    packet, status = fields[-2:]
                    outputs = fields[:-2]
                    after_producer = plan.info()[0]
                    producers.append(after_producer-1)
                    consumers.append(after_producer)
                    routes = functions[3](outputs[4])
                else:
                    packet = functions[0](fixed[0])
                    after_producer = plan.info()[0]
                    producers.append(after_producer-1)
                    consumers.append(after_producer)
                    outputs = functions[1](packet, *fixed[1:], peers)
                    routes = functions[3](outputs[4])
                    status = None
                fragments.append(dict(layer=c['layer'], start=start_segment,
                                      after_producer=after_producer, end=plan.info()[0]))
                packets.append(packet)
                output_groups.append((outputs, routes, status))
                retained.extend([*fixed, *c['weights'], *outputs, *routes])
                if status is not None:
                    retained.append(status)
            captured = plan.info()[0]
            report.setdefault('captured_fragments', []).append(dict(future=future, segments=captured,
                                                                  fragments=fragments))
            save()
            plan.end(captured)
            plan.prepare(backend, packets, peers, flags, producers, consumers,
                         [1 if future else 0]*len(chain), retained, future)
            graphs[-1] = (plan, output_groups)
        for case in range(3):
            for c in chain:
                for destination, value in zip(c['fixed'], c['fixtures'][case], strict=True):
                    destination.copy_(value)
            torch.hpu.synchronize()
            answers = []
            for plan, groups in graphs:
                plan.replay()
                torch.hpu.synchronize()
                fields = []
                for outputs, routes, status in groups:
                    fields.extend(v.clone().cpu() for v in (*outputs, *routes))
                    if status is not None and not bool((status.clone().cpu() > 0).all()):
                        raise AssertionError('Native future receive timed out')
                answers.append(fields)
            exact = [torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                     for a, b in zip(*answers, strict=True)]
            report['checks'].append(dict(case=case, exact_fields=exact))
            save()
            if not all(exact):
                raise AssertionError('Actual native producer/consumer outputs differ')
        if args.timing:
            report.update(timed=True, timing=[], iterations=args.iterations)
            for case in range(3):
                for c in chain:
                    for destination, value in zip(c['fixed'], c['fixtures'][case], strict=True):
                        destination.copy_(value)
                torch.hpu.synchronize()
                values, timed_answers = [], []
                for plan, groups in graphs:
                    start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    dist.barrier(group=get_tp_group().cpu_group)
                    begin_wall = time.perf_counter()
                    start.record()
                    for _ in range(args.iterations):
                        plan.replay()
                    end.record()
                    torch.hpu.synchronize()
                    values.append(dict(device_ms=start.elapsed_time(end)/args.iterations,
                                       wall_ms=(time.perf_counter()-begin_wall)*1000/args.iterations))
                    timed_fields = []
                    for outputs, routes, status in groups:
                        if status is not None and not bool((status.clone().cpu() > 0).all()):
                            raise AssertionError('Queued multi-replay future receive timed out')
                        timed_fields.extend(v.clone().cpu() for v in (*outputs, *routes))
                    timed_answers.append(timed_fields)
                timed_exact = [torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                               for a, b in zip(*timed_answers, strict=True)]
                if not all(timed_exact):
                    raise AssertionError('Queued native A/B changed producer/consumer outputs')
                peers_timing = [None]*4
                dist.all_gather_object(peers_timing, values, group=get_tp_group().cpu_group)
                baseline = max(v[0]['device_ms'] for v in peers_timing)
                candidate = max(v[1]['device_ms'] for v in peers_timing)
                report['timing'].append(dict(case=case, ranks=peers_timing, baseline_ms=baseline,
                                              candidate_ms=candidate, critical_gain_ms=baseline-candidate,
                                              queued_outputs_exact=timed_exact))
                save()
            gains = [v['critical_gain_ms'] for v in report['timing']]
            report.update(all_three_positive=all(v > 0 for v in gains), median_chain_gain_ms=statistics.median(gains),
                          performance_qualified=all(v > 0 for v in gains), gain_credit_ms=0)
        report.update(status='passed', native_info=[v[0].info() for v in graphs],
                      fixed_capture_reused=True, helper_loaded=bool(helper) and bool(hccl_owner))
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        save()
        torch.hpu.synchronize()
        for graph in reversed(graphs):
            (graph[0] if isinstance(graph, tuple) else graph).close()
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
        if scope is not None:
            scope.__exit__(None, None, None)


if __name__ == '__main__':
    main()
