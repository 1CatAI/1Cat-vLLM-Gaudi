# SPDX-License-Identifier: Apache-2.0
"""Compare existing inverse-RoPE/wo_a producers through the deployed wo_b."""
import argparse
import json
import os
from pathlib import Path

import torch
import habana_frameworks.torch.core  # noqa: F401

from deepseek_v41_micro_replay import RecipeRecorder
from deepseek_v41_resident_ab import compiler_settings, compare_periods, device_load, summarize
from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
import vllm_gaudi.distributed.tp2_fused_ar_norm  # noqa: F401
from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
from vllm_gaudi.ops.deepseek_v41_math import rotary_table
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar


def reference(value, wa, sa, wb, sb, positions, phase):
    rotated = torch.ops.custom_op.custom_deepseek_v41_rope_inverse_bf16_gaudi2(value, positions, phase)
    output = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2(
        rotated.reshape(value.shape[0], -1, 4096), wa, sa)
    return (torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(output, wb, sb),)


def candidate(value, wa, sa, wb, sb, positions, phase):
    output = torch.ops.custom_op.custom_deepseek_v41_rope_woa_fp8_roundtrip_gaudi2(
        value, wa, sa, positions, phase)
    return (torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(output, wb, sb),)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--rows', nargs='+', type=int, choices=(1, 2, 6), default=(1, 2, 6))
    parser.add_argument('--timing-rows', type=int, choices=(1, 2, 6), default=1)
    parser.add_argument('--layers', type=int, choices=(1, 4), default=1)
    parser.add_argument('--checks-only', action='store_true',
                        help='Validate missing row buckets without repeating C1 performance measurements')
    args = parser.parse_args()
    if not args.checks_only and args.timing_rows not in args.rows:
        parser.error('The timed row bucket must be included in --rows')
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    recorder = None if args.checks_only else RecipeRecorder(root)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    a = WoaFP8Sidecar(args.prepared / 'sidecars/wo_a_fp8', shard)
    b = DenseFP8Sidecar(args.prepared / 'sidecars/attention_dense_fp8', shard)
    prefix = 'layers.2.attn.'
    wa, sa = [a.tensor(prefix + 'wo_a.' + name, 'hpu') for name in ('weight', 'channel_scale')]
    wb, sb = [b.tensor(prefix + 'wo_b.' + name, 'hpu') for name in ('weight', 'channel_scale')]
    config = json.loads((args.prepared / 'config.json').read_text())['text_config']
    scaling = config['rope_scaling']
    table = rotary_table(64, 32768, config['compress_rope_theta'], scaling['original_max_position_embeddings'],
                         scaling['factor'], scaling['beta_fast'], scaling['beta_slow'])
    phase = torch.cat((table[..., 0], table[..., 1]), -1).contiguous().to('hpu')
    cases, plans = [], {}
    torch.manual_seed(1001)
    for rows in args.rows:
        value = torch.randn(rows, 16, 512).bfloat16().to('hpu')
        positions = torch.full((rows,), 16514, dtype=torch.int32).to('hpu')
        arguments = value, wa, sa, wb, sb, positions, phase
        arms = {}
        for name, function in (('parent', reference), ('candidate', candidate)):
            directory = root / 'compiler' / f'{name}-c{rows}'
            directory.mkdir(parents=True, exist_ok=True)
            compiled = torch.compile(function, backend=make_backend(), fullgraph=True, dynamic=False)
            with compiler_settings({'DUMP_POST_GRAPHS': str(directory)}):
                compiled(*arguments)[0].cpu()
            arms[name] = compiled
        for pattern, factor in (('random', 1), ('zero', 0), ('tiny', 1e-8), ('reordered', 1)):
            value.copy_((torch.randn(rows, 16, 512) * factor).flip(0).bfloat16().to('hpu'))
            positions.copy_((torch.arange(rows, dtype=torch.int32) + 16383).flip(0).to('hpu'))
            x, y = [fn(*arguments)[0].cpu() for fn in arms.values()]
            exact = torch.equal(x, y)
            cases.append(dict(rows=rows, pattern=pattern, exact=exact,
                              max_absolute=float((x.float() - y.float()).abs().max())))
            assert exact, cases[-1]
        if rows == args.timing_rows and not args.checks_only:
            timed_inputs, timed_weights = [], []
            for layer in range(2, 2 + args.layers):
                prefix = f'layers.{layer}.attn.'
                matrix_a, scale_a = [a.tensor(prefix + 'wo_a.' + field, 'hpu')
                                     for field in ('weight', 'channel_scale')]
                matrix_b, scale_b = [b.tensor(prefix + 'wo_b.' + field, 'hpu')
                                     for field in ('weight', 'channel_scale')]
                timed_inputs.append(value.clone())
                timed_weights.append((matrix_a, scale_a, matrix_b, scale_b, positions, phase))
            for name, function in arms.items():
                plans[name] = recorder.prepare(function, timed_inputs * (32 // args.layers),
                                               timed_weights * (32 // args.layers))
    (root / 'NUMERICAL_CONTRACT.json').write_text(json.dumps(dict(status='passed', cases=cases,
        reference='Existing shared native inverse RoPE -> wo_a roundtrip -> wo_b', checkpoint_layer=2), indent=2)+'\n')
    if args.checks_only:
        return
    periods = []
    for label in 'ABABAB':
        plan = plans['parent' if label == 'A' else 'candidate']
        for _ in range(32):
            plan()
        torch.hpu.synchronize()
        events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)) for _ in range(200)]
        load = device_load()
        for start, end in events:
            start.record()
            plan()
            end.record()
            end.synchronize()
        samples = [start.elapsed_time(end) / 32 for start, end in events]
        periods.append(dict(arm=label, token_intervals_ms=samples, summary=summarize(samples), competing_load=load))
        print(label, periods[-1]['summary'], flush=True)
    report = dict(periods=periods, comparison=compare_periods(periods), repeat=32,
                  rows=args.timing_rows, real_weight_layers=args.layers,
                  no_real16_gain_credit=True, no_formal_gain_credit=True)
    (root / 'SMALL_CHAIN_AB.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['comparison']), flush=True)
    for plan in plans.values():
        plan.close()


if __name__ == '__main__':
    with torch.inference_mode():
        main()
