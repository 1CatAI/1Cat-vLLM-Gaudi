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
    args = parser.parse_args()
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    recorder = RecipeRecorder(root)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    a = WoaFP8Sidecar(args.prepared / 'sidecars/wo_a_fp8', shard)
    b = DenseFP8Sidecar(args.prepared / 'sidecars/attention_dense_fp8', shard)
    prefix = 'layers.2.attn.'
    wa, sa = [a.tensor(prefix + 'wo_a.' + name, 'hpu') for name in ('weight', 'channel_scale')]
    wb, sb = [b.tensor(prefix + 'wo_b.' + name, 'hpu') for name in ('weight', 'channel_scale')]
    config = json.loads((args.prepared / 'config.json').read_text())['text_config']
    scaling = config['rope_scaling']
    table = rotary_table(64, 32768, config['rope_theta'], scaling['original_max_position_embeddings'],
                         scaling['factor'], scaling['beta_fast'], scaling['beta_slow'])
    phase = torch.cat((table[..., 0], table[..., 1]), -1).contiguous().to('hpu')
    cases, plans = [], {}
    torch.manual_seed(1001)
    for rows in (1, 2, 6):
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
        if rows == 1:
            for name, function in arms.items():
                plans[name] = recorder.prepare(function, [value] * 32, [arguments[1:]] * 32)
    (root / 'NUMERICAL_CONTRACT.json').write_text(json.dumps(dict(status='passed', cases=cases,
        reference='Existing shared native inverse RoPE -> wo_a roundtrip -> wo_b', checkpoint_layer=2), indent=2)+'\n')
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
    report = dict(periods=periods, comparison=compare_periods(periods, device_events=True), repeat=32,
                  no_real16_gain_credit=True, no_formal_gain_credit=True)
    (root / 'SMALL_CHAIN_AB.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['comparison']), flush=True)
    for plan in plans.values():
        plan.close()


if __name__ == '__main__':
    with torch.inference_mode():
        main()
