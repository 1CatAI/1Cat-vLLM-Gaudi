# SPDX-License-Identifier: Apache-2.0
"""Screen Engram gate/update through its real mHC control and collapse consumers."""
import argparse
import json
import os
from pathlib import Path

import torch
import habana_frameworks.torch.core  # noqa: F401
import vllm_gaudi.distributed.tp2_fused_ar_norm  # noqa: F401

from deepseek_v41_micro_replay import RecipeRecorder
from deepseek_v41_resident_ab import compiler_settings, compare_periods, device_load, summarize
from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
from vllm_gaudi.ops.deepseek_v41_math import engram_update
from vllm_gaudi.ops.deepseek_v41_weights import read_header


def consume(updated, previous, packed):
    # These are the two direct consumers in hc_pre. Return the full control
    # allocation, avoiding offset gate views unsupported by the micro recorder.
    control = torch.ops.custom_op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(
        updated.flatten(1).contiguous(), packed, 1e-20)
    collapsed = (updated.float() * previous.unsqueeze(-1)).sum(1).bfloat16()
    return updated, control, collapsed


def reference(residual, kv, q, k, active, previous, fn, packed, scale, base):
    updated = engram_update(residual, kv, q, k, active, 1e-20)
    return consume(updated, previous, packed)


def candidate(residual, kv, q, k, active, previous, fn, packed, scale, base):
    updated = torch.ops.custom_op.custom_deepseek_v41_engram_update_bf16_gaudi2(residual, kv, q, k, active, 1e-20)
    return consume(updated, previous, packed)


def load_weight(prepared, name):
    manifest = json.loads((prepared / 'manifest.json').read_text())
    path = prepared / manifest['rank_files']['pp0-tp0']['file']
    spec = read_header(path)[name]
    with path.open('rb') as stream:
        stream.seek(spec.offset)
        data = bytearray(stream.read(spec.nbytes))
    dtype = {'BF16': torch.bfloat16, 'F32': torch.float32}[spec.dtype]
    return torch.frombuffer(data, dtype=dtype).reshape(spec.shape).clone().to('hpu')


def error(a, b):
    a32, b32 = a.float(), b.float()
    return dict(exact=torch.equal(a.view(torch.uint8), b.view(torch.uint8)),
                relative_l2=float(torch.linalg.vector_norm(a32 - b32) / torch.linalg.vector_norm(a32).clamp_min(1e-20)),
                max_absolute=float((a32 - b32).abs().max()),
                differing=int((a32 != b32).count_nonzero()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    options = parser.parse_args()
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(options.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    recorder = RecipeRecorder(root)
    torch.manual_seed(30)
    q, k, fn, scale, base = [
        load_weight(options.prepared, 'layers.1.' + name)
        for name in ('engram.q_weight', 'engram.k_weight', 'hc_attn_fn', 'hc_attn_scale', 'hc_attn_base')
    ]
    packed = fn.contiguous() if fn.shape == (24, 20480) else fn.t().contiguous()
    cases, plans = [], {}
    for rows in (1, 2, 6):
        x = torch.randn(rows, 4, 5120).bfloat16().to('hpu')
        kv = torch.randn(rows, 25600).bfloat16().to('hpu')
        active = torch.ones(rows, dtype=torch.bool).to('hpu')
        previous = torch.full((rows, 4), .25, dtype=torch.float32).to('hpu')
        args = (x, kv, q, k, active, previous, fn, packed, scale, base)
        arms = {}
        for name, function in (('parent', reference), ('candidate', candidate)):
            directory = root / 'compiler' / f'{name}-c{rows}'
            directory.mkdir(parents=True, exist_ok=True)
            compiled = torch.compile(function, backend=make_backend(), fullgraph=True, dynamic=False)
            with compiler_settings({'DUMP_POST_GRAPHS': str(directory)}):
                tuple(value.cpu() for value in compiled(*args))
            arms[name] = compiled
        for pattern in ('random', 'zero', 'tiny', 'masked_reordered'):
            factor = 0 if pattern == 'zero' else 1e-8 if pattern == 'tiny' else 1
            x.copy_((torch.randn(rows, 4, 5120) * factor).bfloat16().flip(0).to('hpu'))
            kv.copy_((torch.randn(rows, 25600) * factor).bfloat16().flip(0).to('hpu'))
            active.copy_((torch.arange(rows) %
                          2 == 1 if pattern == 'masked_reordered' else torch.ones(rows, dtype=torch.bool)).to('hpu'))
            a, b = [tuple(value.cpu() for value in function(*args)) for function in arms.values()]
            outputs = {
                name: error(left, right)
                for name, left, right in zip(('updated_residual', 'control_rrms', 'collapsed'), a, b)
            }
            passed = all(value['relative_l2'] <= .002 for value in outputs.values())
            if pattern == 'masked_reordered':
                mask = ~active.cpu()
                passed &= torch.equal(b[0][mask], x.cpu()[mask])
            cases.append(dict(rows=rows, pattern=pattern, outputs=outputs, passed=bool(passed)))
        if rows == 1:
            x.copy_(torch.randn_like(x))
            kv.copy_(torch.randn_like(kv))
            active.fill_(True)
            for name, function in arms.items():
                plans[name] = recorder.prepare(function, [x] * 32, [args[1:]] * 32)
    contract = dict(
        status='passed' if all(case['passed'] for case in cases) else 'failed',
        cases=cases,
        epsilon=1e-20,
        real_checkpoint_layer=1,
        scope='Engram update -> deployed mHC control/RRMS and parallel collapse; full direct-consumer allocations',
        reference='shared generic TP2/TP4 arithmetic; reduction-order tolerance .002',
        formal_quality_pending=True)
    (root / 'NUMERICAL_CONTRACT.json').write_text(json.dumps(contract, indent=2) + '\n')
    if contract['status'] != 'passed':
        raise RuntimeError('Engram update downstream numerical contract failed')
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
                  comparison=compare_periods(periods),
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
