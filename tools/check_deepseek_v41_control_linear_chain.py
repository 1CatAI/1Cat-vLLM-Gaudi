# SPDX-License-Identifier: Apache-2.0
"""Screen the linear control load through real mHC gate/Sinkhorn consumption."""
import argparse
import json
import os
from pathlib import Path

import torch
import habana_frameworks.torch.core  # noqa: F401

from check_deepseek_v41_engram_update import load_weight
from deepseek_v41_micro_replay import RecipeRecorder
from deepseek_v41_resident_ab import compiler_settings, compare_periods, device_load, summarize
from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend


def gate(control, scale, base):
    return torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(
        control[:, :24].contiguous(), control[:, 24:].contiguous(), scale, base)


def reference(value, weight, scale, base):
    control = torch.ops.custom_op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(value, weight, 1e-20)
    return control, gate(control, scale, base)


def candidate(value, weight, scale, base):
    control = torch.ops.custom_op.custom_deepseek_v41_control_rrms_linear_bf16_gaudi2(value, weight, 1e-20)
    return control, gate(control, scale, base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    options = parser.parse_args()
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    recorder = RecipeRecorder(root)
    torch.manual_seed(1001)
    fn, scale, base = [load_weight(options.prepared, 'layers.2.' + name)
                       for name in ('hc_attn_fn', 'hc_attn_scale', 'hc_attn_base')]
    weight = fn.contiguous() if fn.shape == (24, 20480) else fn.t().contiguous()
    cases, plans = [], {}
    for rows in (1, 2, 6):
        value = torch.randn(rows, 20480).bfloat16().to('hpu')
        arguments = value, weight, scale, base
        arms = {}
        for name, function in (('parent', reference), ('candidate', candidate)):
            directory = root / 'compiler' / f'{name}-c{rows}'
            directory.mkdir(parents=True, exist_ok=True)
            compiled = torch.compile(function, backend=make_backend(), fullgraph=True, dynamic=False)
            with compiler_settings({'DUMP_POST_GRAPHS': str(directory)}):
                tuple(x.cpu() for x in compiled(*arguments))
            arms[name] = compiled
        for pattern, factor in (('random', 1), ('zero', 0), ('tiny', 1e-8), ('reordered', 1)):
            value.copy_((torch.randn(rows, 20480) * factor).flip(0).bfloat16().to('hpu'))
            a, b = [tuple(x.cpu() for x in function(*arguments)) for function in arms.values()]
            exact = all(torch.equal(x, y) for x, y in zip(a, b))
            cases.append(dict(rows=rows, pattern=pattern, exact=exact,
                              max_absolute=[float((x - y).abs().max()) for x, y in zip(a, b)]))
            assert exact, cases[-1]
        if rows == 1:
            for name, function in arms.items():
                plans[name] = recorder.prepare(function, [value] * 32, [arguments[1:]] * 32)
    (root / 'NUMERICAL_CONTRACT.json').write_text(json.dumps(dict(status='passed', cases=cases,
        scope='Real layer2 control/RRMS -> gate/Sinkhorn consumer; unchanged packed weights'), indent=2)+'\n')
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
        values = [start.elapsed_time(end) / 32 for start, end in events]
        periods.append(dict(arm=label, token_intervals_ms=values, summary=summarize(values), competing_load=load))
        print(label, periods[-1]['summary'], flush=True)
    result = dict(periods=periods, comparison=compare_periods(periods, device_events=True), repeat=32,
                  no_real16_gain_credit=True, no_formal_gain_credit=True)
    (root / 'SMALL_CHAIN_AB.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['comparison']), flush=True)
    for plan in plans.values():
        plan.close()


if __name__ == '__main__':
    with torch.inference_mode():
        main()
