# SPDX-License-Identifier: Apache-2.0
"""Screen shared attention norm + exact dense quantization through an FP8 MME consumer."""
import argparse
import json
import os
from pathlib import Path

import torch
import habana_frameworks.torch.core  # noqa: F401
import vllm_gaudi.distributed.tp2_fused_ar_norm  # noqa: F401

from deepseek_v41_resident_ab import compiler_settings, compare_periods, device_load, summarize
from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
from vllm_gaudi.ops.deepseek_v41_math import rms_norm


def separate(x, norm, weight, channel):
    normalized = rms_norm(x, norm, 1e-6)
    quantized, scale = torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(normalized)
    product = torch.ops.hpu.fp8_gemm_v2(quantized, False, weight, True, None, torch.bfloat16, scale, channel, None,
                                        False)
    return normalized, quantized, scale, product


def fused(x, norm, weight, channel):
    normalized, quantized, scale = torch.ops.custom_op.custom_deepseek_v41_attention_norm_quant_gaudi2(x, norm, 1e-6)
    product = torch.ops.hpu.fp8_gemm_v2(quantized, False, weight, True, None, torch.bfloat16, scale, channel, None,
                                        False)
    return normalized, quantized, scale, product


def errors(a, b):
    bit_exact = torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    return dict(exact=bit_exact,
                max_absolute=float(diff.max()),
                relative_l2=float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(a).clamp_min(1e-20)),
                differing=int(torch.count_nonzero(diff)),
                elements=a.numel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-replay', action='store_true')
    parser.add_argument('--native-replay', action='store_true')
    parser.add_argument('--synchronize-each-interval', action='store_true')
    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help='Queue this many actual prepared compute frames per event interval')
    options = parser.parse_args()
    if options.repeat < 1 or (options.repeat != 1 and not (options.prepared_replay or options.native_replay)):
        parser.error('Repeats require prepared replay and a positive frame count')
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    if options.prepared_replay:
        # One-rank compute-only diagnostic; use the production prepared recipe
        # executor without acquiring model weights or changing serving passes.
        import torch.distributed as dist
        from torch import fx
        from vllm_gaudi.ops import tp2_prepared_plan as prepared
        from vllm_gaudi.ops.deepseek_v41_completion import _control_bridge
        from habana_frameworks.torch.dynamo.compile_backend.passes import (OptimizationPassPlacement,
                                                                           register_pass_at_optimization_pass)

        dist.init_process_group('hccl', rank=0, world_size=1, init_method=f'file://{root}/compute-rendezvous')
        bridge, backend = _control_bridge(), dist.distributed_c10d._get_default_group()._get_backend(
            torch.device('hpu'))
        prepared._runtime = lambda: (bridge, backend)
        original_forward = prepared.PreparedGroupModule.forward

        def remember_inputs(module, inputs):
            module.screen_inputs = list(inputs)
            return original_forward(module, inputs)

        prepared.PreparedGroupModule.forward = remember_inputs

        def prepare_compute_only(context):
            graph = context.graph_module
            original = fx.GraphModule(graph, graph.graph)
            graph.add_module('_screen_prepared', prepared.PreparedGroupModule(original))
            replacement = fx.Graph()
            inputs = replacement.placeholder('input_list')
            result = replacement.call_module('_screen_prepared', (inputs, ))
            replacement.output(result)
            graph.graph = replacement
            graph.delete_all_unused_submodules()
            graph.recompile()
            return True

        register_pass_at_optimization_pass(prepare_compute_only, OptimizationPassPlacement.POST_PARTITIONER)
    recorder = None
    if options.native_replay:
        from deepseek_v41_micro_replay import RecipeRecorder
        recorder = RecipeRecorder(root)
    torch.manual_seed(30)
    # Same fused wq_a/wkv geometry; no weight dequantization in the measured path.
    weight = (torch.randn(1792, 5120) * .03125).to(torch.float8_e4m3fn).to('hpu')
    channel = torch.ones(1, 1792, dtype=torch.float32).to('hpu')
    norm = (torch.randn(5120) * .125 + 1).to(torch.bfloat16).to('hpu')
    cases, timed = [], None
    for count in (1, 2, 6):
        cpu_x = torch.randn(count, 5120).to(torch.bfloat16)
        x = cpu_x.to('hpu')
        args = (x, norm, weight, channel)
        arms = {}
        for name, fn in (('separate', separate), ('fused', fused)):
            directory = root / 'compiler' / f'{name}-c{count}'
            directory.mkdir(parents=True, exist_ok=True)
            compiled = torch.compile(fn, backend=make_backend(), fullgraph=True, dynamic=False)
            previous = set(prepared._modules) if options.prepared_replay else set()
            with compiler_settings(dict(DUMP_POST_GRAPHS=str(directory))):
                tuple(v.cpu() for v in compiled(*args))
            if options.prepared_replay:
                modules = [m for m in prepared._modules if m not in previous]
                if len(modules) != 1 or len(modules[0].plans) != 1:
                    raise RuntimeError('The short chain must have exactly one complete prepared plan')
                module = modules[0]
                plan, bindings = module.plans[0], module.screen_inputs
                plans, input_frames = [plan] * options.repeat, [bindings] * options.repeat

                def execute(*ignored, plan=plan, plans=plans, input_frames=input_frames, module=module):
                    bridge.replay_prepared_groups(plans, input_frames)
                    return tuple(plan.outputs())

                arms[name] = execute
            else:
                arms[name] = compiled
        for pattern in ('random', 'zero', 'tiny', 'reordered'):
            value = (cpu_x if pattern == 'random' else torch.zeros_like(cpu_x) if pattern == 'zero' else cpu_x *
                     .0001 if pattern == 'tiny' else cpu_x.flip(0) * 1.0625)
            x.copy_(value.to('hpu'))
            a, b = [tuple(v.cpu() for v in arms[name](*args)) for name in ('separate', 'fused')]
            report = dict(count=count,
                          pattern=pattern,
                          outputs={
                              name: errors(v, w)
                              for name, v, w in zip(('normalized', 'quantized', 'scale', 'MME'), a, b)
                          })
            # The shared native norm uses its qualified accumulation order.
            # Keep the BF16 boundary, require the scale and final GEMM contract.
            report['passed'] = (report['outputs']['normalized']['relative_l2'] <= .002
                                and report['outputs']['MME']['relative_l2'] <= .005
                                and report['outputs']['scale']['exact'])
            cases.append(report)
        x.copy_(cpu_x.to('hpu'))
        if count == 1:
            if recorder is not None:
                plans = {
                    name: recorder.prepare(fn, [x] * options.repeat, [args[1:]] * options.repeat)
                    for name, fn in arms.items()
                }
                for plan in plans.values():
                    plan()
                    torch.hpu.synchronize()
                timed = plans, args
            else:
                timed = arms, args
    report = dict(status='passed' if all(r['passed'] for r in cases) else 'failed',
                  cases=cases,
                  scope='BF16 norm -> power-of-two FP8 operand -> actual FP8 MME, C1/C2/C6',
                  formal_quality_pending=True,
                  gain_ledger_eligible=False)
    (root / 'NUMERICAL_CONTRACT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if report['status'] != 'passed':
        raise RuntimeError('Attention norm/quant consumer contract failed')
    arms, args = timed
    periods = []
    for label in 'ABABAB':
        plan = arms['separate' if label == 'A' else 'fused']
        fn = (lambda *ignored: plan()) if options.native_replay else plan
        for _ in range(32):
            fn(*args)
        torch.hpu.synchronize()
        events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)) for _ in range(200)]
        load = device_load()
        for start, end in events:
            start.record()
            fn(*args)
            end.record()
            if options.synchronize_each_interval:
                end.synchronize()
        torch.hpu.synchronize()
        values = [start.elapsed_time(end) / options.repeat for start, end in events]
        periods.append(dict(arm=label, token_intervals_ms=values, competing_load=load, summary=summarize(values)))
        print(label, periods[-1]['summary'], flush=True)
    timing = dict(periods=periods,
                  comparison=compare_periods(periods),
                  no_real16_gain_credit=True,
                  prepared_replay=options.prepared_replay,
                  native_replay=options.native_replay,
                  prepared_frames_per_interval=options.repeat,
                  synchronize_each_interval=options.synchronize_each_interval,
                  scope='Single norm/quant/MME consumer chain; event scope includes submission')
    (root / 'SMALL_CHAIN_AB.json').write_text(json.dumps(timing, indent=2) + '\n')
    print(json.dumps(timing['comparison']), flush=True)
    if options.native_replay:
        for plan in arms.values():
            plan.close()
        torch.distributed.destroy_process_group()
    if options.prepared_replay:
        prepared.shutdown_prepared_group_plans()
        dist.destroy_process_group()


if __name__ == '__main__':
    with torch.inference_mode():
        main()
