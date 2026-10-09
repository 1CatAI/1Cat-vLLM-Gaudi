# SPDX-License-Identifier: Apache-2.0
"""Common FP8 shared FFN through the next Q/KV input projection, C2/C6."""
import argparse
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

from check_deepseek_v41_c6_dense_restore import Projection


class SharedWeights:
    def __init__(self, w1, w3, w2):
        self.w1, self.w3, self.w2 = w1, w3, w2


class SharedView:
    def __init__(self, weights, gate_up, channel, down):
        self.weights = weights
        self.shared_gate_up = channel is not None
        self.shared_gate_up_weight, self.shared_gate_up_channel = gate_up, channel
        self.shared_down_weight = down


class WeightView:
    def __init__(self, shared):
        self.shared_experts = shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('sidecar', type=Path)
    parser.add_argument('--samples', type=int, default=32)
    args = parser.parse_args()
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from deepseek_v41_micro_replay import RecipeRecorder
    from vllm_gaudi.models.deepseek_v41_program import PreparedMoE
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_math import rms_norm, quantize_activation
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_native_trace import configure_post_graph_directory
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    recorder = RecipeRecorder(root)
    configure_post_graph_directory(root / 'graphs/rank0')
    shard = PreparedV41Shard(args.prepared, 0, 0)
    sidecar = DenseFP8Sidecar(args.sidecar, shard)
    config = json.loads((args.prepared / 'config.json').read_text())['text_config']
    eps = config['rms_norm_eps']

    def body(candidate):
        def execute(value, w1, w3, w2, gate_up, channel, down, down_channel, norm, qkv):
            shared = SharedWeights(Projection(w1, None, False, False), Projection(w3, None, False, False),
                                   Projection(w2, down_channel, candidate, candidate))
            view = SharedView(WeightView(shared), gate_up, channel, down)
            output = PreparedMoE.shared_expert(view, value)
            consumed = rms_norm((value + output).bfloat16(), norm, eps)
            return torch.nn.functional.linear(quantize_activation(consumed), qkv), output
        return execute

    report = dict(status='running', cases=[], credited_e2e_ms=0, real16_qualified=False)
    with torch.inference_mode():
        torch.manual_seed(4613)
        operands = [[], []]
        for layer in range(20, 28):
            prefix = f'layers.{layer}.ffn.shared_experts.'
            original = [shard.dense(prefix + p + '.weight', 'hpu') for p in ('w1', 'w3', 'w2')]
            modules = torch.nn.Module()
            for name in ('w1', 'w3', 'w2'):
                m = torch.nn.Module()
                m.register_buffer('weight', sidecar.tensor(prefix + name + '.weight', 'hpu'))
                m.register_buffer('channel_scale', sidecar.tensor(prefix + name + '.channel_scale', 'hpu'))
                m.dense_fp8_direct_input = True
                modules.add_module(name, m)
            owner = SimpleNamespace(weights=SimpleNamespace(shared_experts=modules), shared_gate_up=True,
                                    refresh_sat_eligibility=lambda: None)
            PreparedMoE.prepare_shared_gate_up_weight(owner)
            norm = shard.tensor(f'layers.{layer}.attn_norm.weight', 'hpu')
            qkv = torch.cat(tuple(shard.dense(f'layers.{layer}.attn.{p}.weight', 'cpu')
                                  for p in ('wq_a', 'wkv'))).to('hpu')
            operands[0].append((*original, None, None, None, None, norm, qkv))
            operands[1].append((*original, owner.shared_gate_up_weight, owner.shared_gate_up_channel,
                                owner.shared_down_weight, modules.w2.channel_scale, norm, qkv))
        functions = [torch.compile(body(candidate), backend='hpu_backend', fullgraph=True, dynamic=False)
                     for candidate in (False, True)]
        for count in (2, 6):
            inputs = [torch.randn(count, 5120).bfloat16().to('hpu') for _ in operands[0]]
            errors = []
            for x, wa, wb in zip(inputs, *operands, strict=True):
                a, b = [tuple(t.cpu() for t in f(x, *w)) for f, w in zip(functions, (wa, wb), strict=True)]
                if not all(bool(torch.isfinite(t).all()) for t in b):
                    raise AssertionError('Non-finite common shared FP8 consumer')
                errors.append(dict(max_abs=[float((aa.float() - bb.float()).abs().max())
                                            for aa, bb in zip(a, b, strict=True)],
                                   rms=[float((aa.float() - bb.float()).square().mean().sqrt())
                                        for aa, bb in zip(a, b, strict=True)]))
            plans = [recorder.prepare(f, inputs, operands[a]) for a, f in enumerate(functions)]
            try:
                for arm, plan in enumerate(plans):
                    for x in inputs:
                        x.add_(.03125)
                    expected = [tuple(t.cpu() for t in functions[arm](x, *w))
                                for x, w in zip(inputs, operands[arm], strict=True)]
                    plan()
                    torch.hpu.synchronize()
                    for index, values in enumerate(expected):
                        for j, value in enumerate(values):
                            if not torch.equal(value, plan.outputs[index * 2 + j].cpu()):
                                raise AssertionError('Shared native replay differs from its compiled arm')
                    for _ in range(8):
                        plan()
                    torch.hpu.synchronize()
                timings = []
                for arm in (0, 1, 0, 1, 0, 1):
                    start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.samples):
                        plans[arm]()
                    end.record()
                    end.synchronize()
                    timings.append(dict(arm=arm, ms_per_layer=start.elapsed_time(end) / args.samples / 8))
                report['cases'].append(dict(count=count, errors=errors, timings=timings,
                                           means_ms=[statistics.fmean(t['ms_per_layer'] for t in timings
                                                                     if t['arm'] == arm) for arm in (0, 1)]))
            finally:
                torch.hpu.synchronize()
                for plan in plans:
                    plan.close()
    report['status'] = 'passed_component_contract'
    (root / 'c6-shared-restore.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
