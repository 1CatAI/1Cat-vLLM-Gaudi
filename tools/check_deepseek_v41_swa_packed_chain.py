# SPDX-License-Identifier: Apache-2.0
"""Screen SWA-only packed MLA with the existing TP-parameterized consumers."""
import json
import math
import os
from pathlib import Path

import torch
import habana_frameworks.torch.core  # noqa: F401
import vllm_gaudi.distributed.tp2_fused_ar_norm  # noqa: F401

from deepseek_v41_resident_ab import compiler_settings, compare_periods, device_load, summarize
from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
from vllm_gaudi.ops.deepseek_v41_math import _pack_swa_torch, pack_swa, unpack_swa


def dense(query, swa, main, rows, indices, sink, scale, lengths, kv, positions):
    swa.index_copy_(0, positions.remainder(256).long(), pack_swa(kv))
    return torch.ops.custom_op.custom_deepseek_v41_selected_mla_mme_gaudi2(
        query, unpack_swa(swa), indices, sink, scale, lengths)


def packed(query, swa, main, rows, indices, sink, scale, lengths, kv, positions):
    swa.index_copy_(0, positions.remainder(256).long(), pack_swa(kv))
    operation = (torch.ops.custom_op.custom_deepseek_v41_paged_mla_direct_mme_gaudi2
                 if query.shape[0] == 1 else torch.ops.custom_op.custom_deepseek_v41_paged_mla_mme_gaudi2)
    return operation(query, swa, main, rows, indices, sink, scale, lengths)


def main():
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    torch.manual_seed(30)
    cache = torch.randn(256, 512, dtype=torch.bfloat16)
    cache[0, :8] = 0
    cache[1, :8] = -0.0
    swa = _pack_swa_torch(cache).to('hpu')
    main = torch.zeros((1, 288), dtype=torch.uint8, device='hpu')
    rows = torch.arange(256, dtype=torch.int32).reshape(1, 256).to('hpu')
    reports, timed = [], None
    for heads in (16, 32):
        for count in (1, 2, 6):
            query = torch.randn(count, heads, 512, dtype=torch.bfloat16).to('hpu')
            indices = torch.empty((count, 128), dtype=torch.int32, device='hpu')
            sink = torch.randn(heads, dtype=torch.float32).to('hpu')
            scale = torch.tensor([1 / math.sqrt(512)], dtype=torch.float32, device='hpu')
            lengths = torch.full((count,), 128, dtype=torch.int32, device='hpu')
            kv = torch.randn(count, 512, dtype=torch.bfloat16).to('hpu')
            positions = torch.arange(count, dtype=torch.int32).to('hpu')
            arguments = (query, swa, main, rows, indices, sink, scale, lengths, kv, positions)
            arms = {}
            for name, fn in (('baseline', dense), ('packed', packed)):
                folder = root / 'compiler' / f'{name}-h{heads}-c{count}'
                folder.mkdir(parents=True, exist_ok=True)
                compiled = torch.compile(fn, backend=make_backend(), fullgraph=True, dynamic=False)
                ids = torch.arange(128, dtype=torch.int32).repeat(count, 1).to('hpu')
                indices.copy_(ids)
                with compiler_settings(dict(DUMP_POST_GRAPHS=str(folder))):
                    compiled(*arguments).cpu()
                arms[name] = compiled
            for position in (0, 255, 16384):
                windows = position + torch.arange(count).reshape(-1, 1) - 127 + torch.arange(128)
                ids = torch.where(windows >= 0, windows.remainder(256), -1).to(torch.int32)
                indices.copy_(ids.to('hpu'))
                positions.copy_((position + torch.arange(count, dtype=torch.int32)).to('hpu'))
                a, b = [arms[name](*arguments).cpu() for name in ('baseline', 'packed')]
                error = (a.float()-b.float()).abs()
                reports.append(dict(heads=heads, count=count, position=position,
                                    exact=torch.equal(a, b), max_absolute_error=float(error.max())))
            if heads == 16 and count == 1:
                timed = (arms, arguments)
    report = dict(status='passed' if all(r['exact'] for r in reports) else 'failed', cases=reports,
                  scope='Packed SWA -> exact existing MLA consumer; TP4/TP2 geometry and C1/C2/C6',
                  gain_ledger_eligible=False)
    (root / 'SWA_CONSUMER_CONTRACT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if report['status'] != 'passed':
        raise RuntimeError('SWA-only consumer contract did not pass')
    arms, arguments = timed
    periods = []
    for label in ('A', 'B', 'A', 'B', 'A', 'B'):
        fn = arms['baseline' if label == 'A' else 'packed']
        for _ in range(32):
            fn(*arguments)
        torch.hpu.synchronize()
        events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)) for _ in range(200)]
        load = device_load()
        for start, end in events:
            start.record()
            fn(*arguments)
            end.record()
        torch.hpu.synchronize()
        values = [start.elapsed_time(end) for start, end in events]
        periods.append(dict(arm=label, token_intervals_ms=values, competing_load=load, summary=summarize(values)))
        print(label, periods[-1]['summary'], flush=True)
    timing = dict(periods=periods, comparison=compare_periods(periods, device_events=True),
                  scope='Single SWA-only Attention chain, device events include exposed submission gaps',
                  no_real16_gain_credit=True)
    (root / 'SMALL_CHAIN_AB.json').write_text(json.dumps(timing, indent=2)+'\n')
    print(json.dumps(timing['comparison']), flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
