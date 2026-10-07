# SPDX-License-Identifier: Apache-2.0
"""Screen BF16 FFN post/collapse through control and the next FP8 input MME."""
import json
import os
from pathlib import Path

import torch
import habana_frameworks.torch.core  # noqa: F401
import vllm_gaudi.distributed.tp2_fused_ar_norm  # noqa: F401

from deepseek_v41_micro_replay import RecipeRecorder
from deepseek_v41_resident_ab import compiler_settings, compare_periods, device_load, summarize
from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
from vllm_gaudi.ops.deepseek_v41_math import hc_post


def consume(updated, collapsed, control_weight, norm, weight, channel):
    control = torch.ops.custom_op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(
        updated.flatten(1).contiguous(), control_weight, 1e-20)
    normalized, q, sx = torch.ops.custom_op.custom_deepseek_v41_attention_norm_quant_gaudi2(collapsed, norm, 1e-20)
    output = torch.ops.hpu.fp8_gemm_v2(q, False, weight, True, None, torch.bfloat16, sx, channel, None, False)
    return updated, control, normalized, output


def reference(value, residual, post, comb, next_pre, control_weight, norm, weight, channel):
    updated = hc_post(value, residual, post, comb)
    collapsed = (updated.float() * next_pre.unsqueeze(-1)).sum(1).bfloat16()
    return consume(updated, collapsed, control_weight, norm, weight, channel)


def candidate(value, residual, post, comb, next_pre, control_weight, norm, weight, channel):
    updated, collapsed = torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
        value, residual, post, comb, next_pre)
    return consume(updated, collapsed, control_weight, norm, weight, channel)


def main():
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    recorder = RecipeRecorder(root)
    torch.manual_seed(30)
    control = (torch.randn(24, 20480) * .01).to('hpu')
    norm = (torch.randn(5120) * .125 + 1).bfloat16().to('hpu')
    weight = (torch.randn(1792, 5120) * .03125).to(torch.float8_e4m3fn).to('hpu')
    channel = torch.ones(1, 1792).to('hpu')
    cases, plans = [], {}
    for rows in (1, 2):
        value = torch.randn(rows, 5120).bfloat16().to('hpu')
        residual = torch.randn(rows, 4, 5120).bfloat16().to('hpu')
        post = torch.sigmoid(torch.randn(rows, 4)).to('hpu')
        comb = torch.rand(rows, 4, 4)
        for _ in range(20):
            comb /= comb.sum(-1, keepdim=True)
            comb /= comb.sum(-2, keepdim=True)
        comb = comb.to('hpu')
        next_pre = torch.sigmoid(torch.randn(rows, 4)).to('hpu')
        args = (value, residual, post, comb, next_pre, control, norm, weight, channel)
        arms = {}
        for name, function in (('parent', reference), ('candidate', candidate)):
            directory = root / 'compiler' / f'{name}-c{rows}'
            directory.mkdir(parents=True, exist_ok=True)
            compiled = torch.compile(function, backend=make_backend(), fullgraph=True, dynamic=False)
            with compiler_settings({'DUMP_POST_GRAPHS': str(directory)}):
                tuple(output.cpu() for output in compiled(*args))
            arms[name] = compiled
        for pattern in ('random', 'zero', 'tiny', 'reordered'):
            factor = 0 if pattern == 'zero' else 1e-8 if pattern == 'tiny' else 1
            value.copy_((torch.randn(rows, 5120) * factor).bfloat16().flip(0).to('hpu'))
            residual.copy_((torch.randn(rows, 4, 5120) * factor).bfloat16().flip(0).to('hpu'))
            a, b = [tuple(output.cpu() for output in function(*args)) for function in arms.values()]
            errors = {
                name:
                dict(exact=torch.equal(x.view(torch.uint8), y.view(torch.uint8)),
                     relative_l2=float(
                         torch.linalg.vector_norm(x.float() - y.float()) /
                         torch.linalg.vector_norm(x.float()).clamp_min(1e-20)))
                for name, x, y in zip(('residual', 'control', 'normalized', 'MME'), a, b)
            }
            cases.append(
                dict(rows=rows,
                     pattern=pattern,
                     outputs=errors,
                     passed=all(error['exact'] for error in errors.values())))
        if rows == 1:
            value.copy_(torch.randn_like(value))
            residual.copy_(torch.randn_like(residual))
            for name, function in arms.items():
                plans[name] = recorder.prepare(function, [value] * 32, [args[1:]] * 32)
    contract = dict(status='passed' if all(case['passed'] for case in cases) else 'failed',
                    cases=cases,
                    scope='FFN post/BF16 collapse -> next control and native norm/FP8 MME',
                    distinct_from_f32_handoff=True,
                    formal_quality_pending=True)
    (root / 'NUMERICAL_CONTRACT.json').write_text(json.dumps(contract, indent=2) + '\n')
    if contract['status'] != 'passed':
        raise RuntimeError('BF16 handoff changes the maintained producer/consumer arithmetic')
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
    report = dict(periods=periods,
                  comparison=compare_periods(periods, device_events=True),
                  repeat=32,
                  synchronize_each_interval=True,
                  no_real16_gain_credit=True,
                  no_formal_gain_credit=True)
    (root / 'SMALL_CHAIN_AB.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['comparison']), flush=True)
    for plan in plans.values():
        plan.close()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    with torch.inference_mode():
        main()
