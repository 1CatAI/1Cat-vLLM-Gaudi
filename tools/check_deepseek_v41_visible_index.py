# SPDX-License-Identifier: Apache-2.0
"""Compare capacity and visible-bucket Full -> Reindex -> packed-row consumers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--module', type=int, default=0)
    parser.add_argument('--capacity', type=int, default=524288)
    parser.add_argument('--bucket', type=int, default=65536)
    parser.add_argument('--local-heads', type=int, default=8)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--mme-candidate', action='store_true')
    args = parser.parse_args()
    if not 32768 < args.bucket <= args.capacity or args.bucket % 128:
        raise ValueError('The visible bucket must be aligned and above the decoded mirror')
    handles = []
    roots = (Path.home() / '.local/state/1cat-vllm/locks', Path.cwd().parent / 'locks',
             Path('/tmp/1cat-gaudi-module-locks'))
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        paths = set(root.glob(f'*module{args.module}.lock')) | {root / f'gaudi-module{args.module}.lock'}
        for path in sorted(paths):
            handle = path.open('a+')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(handle)
    cards = subprocess.check_output(['hl-smi', '-Q', 'module_id,memory.used,utilization.aip', '-f', 'csv,noheader'],
                                    text=True)
    row = next(row for row in cards.splitlines() if int(row.split(',')[0]) == args.module)
    if int(row.split(',')[1].split()[0]) > 1024 or int(row.split(',')[2].split()[0]):
        raise RuntimeError(f'The leased module is not idle: {row}')
    os.environ.update(HABANA_VISIBLE_MODULES=str(args.module), HLS_MODULE_ID=str(args.module))
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v41_indexer import runtime_index_select
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, unpack_fp4

    torch.set_num_threads(1)
    torch.ops.load_library(str(args.library))
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(scope=__doc__, cards=cards, module=args.module, cpus=sorted(os.sched_getaffinity(0)),
                  capacity=args.capacity, bucket=args.bucket, local_heads=args.local_heads, cases=[])

    for ratio in (1, 2):
        capacity, bucket = args.capacity // ratio, args.bucket // ratio
        page_rows = 128 // ratio
        # Padding is deliberately finite but has distinct values. Positions
        # must exclude it in both arms; page IDs have a reserved null page.
        keys = torch.randn(capacity, 128).bfloat16()
        packed = torch.cat((torch.zeros(page_rows, 68, dtype=torch.uint8), pack_fp4(keys, 32)), 0).to('hpu')
        del keys
        pages = torch.arange(1, capacity // page_rows + 1, dtype=torch.int32).to('hpu')
        sources = [(unpack_fp4(pack_fp4(torch.randn(1, 32, 128).bfloat16(), 32), 128, 32).to('hpu'),
                    (torch.randn(1, 32) * .02).bfloat16().to('hpu')) for _ in range(3)]
        positions = [torch.tensor([end - 1], dtype=torch.int32).to('hpu')
                     for end in (32769, 41983, 62463, 65536) if end <= args.bucket]

        def chain(q, weights, cache, table, position, columns, mme=False):
            candidates = torch.full((1, 2048), -1, dtype=torch.int32, device=q.device)
            selected, blocks = runtime_index_select(q, weights, cache, table, position, candidates,
                                                    ratio=ratio, capacity=columns, local_heads=args.local_heads,
                                                    publish_candidates=True, ordered_candidates=True,
                                                    search_rows=columns if mme else None)
            downstream, _ = runtime_index_select(q, weights, cache, table, position, blocks,
                                                  ratio=ratio, capacity=columns, local_heads=args.local_heads,
                                                  reindex=True, ordered_candidates=True,
                                                  search_rows=columns if mme else None)
            logical = downstream.clamp_min(0).flatten()
            physical = table.index_select(0, (logical // page_rows).long()) * page_rows + logical % page_rows
            consumed = cache.index_select(0, physical.long())
            return selected, blocks, downstream, consumed

        def reference(q, weights, cache, table, position):
            return chain(q, weights, cache, table, position, capacity)

        def candidate(q, weights, cache, table, position):
            return chain(q, weights, cache, table, position, bucket, args.mme_candidate)

        arms = [torch.compile(function, backend='hpu_backend', fullgraph=True, dynamic=False)
                for function in (reference, candidate)]
        checks = []
        for i, position in enumerate(positions):
            q, weights = sources[i % len(sources)]
            old = arms[0](q, weights, packed, pages, position)
            new = arms[1](q, weights, packed, pages, position)
            torch.hpu.synchronize()
            equal = [torch.equal(a.cpu(), b.cpu()) for a, b in zip(old, new, strict=True)]
            assert all(equal), (ratio, i, equal)
            checks.append(equal)
        periods = []
        for arm in (0, 1, 0, 1, 0, 1):
            values, device_values = [], []
            begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
            for step in range(args.steps):
                q, weights = sources[step % len(sources)]
                # Keep visibility constant during timing; otherwise context
                # growth dominates the IQR. Boundary cases are checked above.
                position = positions[1 if len(positions) > 1 else 0]
                torch.hpu.synchronize()
                stamp = time.perf_counter_ns()
                begin.record()
                result = arms[arm](q, weights, packed, pages, position)
                end.record()
                end.synchronize()
                values.append((time.perf_counter_ns() - stamp) / 1e6)
                device_values.append(begin.elapsed_time(end))
                del result
            q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
            summary = dict(arm='AB'[arm], median_ms=statistics.median(values), iqr_ms=q3 - q1,
                           device_median_ms=statistics.median(device_values))
            periods.append(dict(summary=summary, wall_ms=values, device_ms=device_values))
            print(json.dumps(dict(ratio=ratio, **summary)), flush=True)
        baseline = [x for p in periods if p['summary']['arm'] == 'A' for x in p['wall_ms']]
        changed = [x for p in periods if p['summary']['arm'] == 'B' for x in p['wall_ms']]
        q1, _, q3 = statistics.quantiles(baseline, n=4, method='inclusive')
        delta = statistics.median(baseline) - statistics.median(changed)
        report['cases'].append(dict(ratio=ratio, checks=checks, periods=periods, delta_ms=delta,
                                    threshold_ms=2 * (q3 - q1), effective=delta > 2 * (q3 - q1)))
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        del packed, pages, sources, positions, arms
    print(json.dumps({'component_effective': all(case['effective'] for case in report['cases']),
                      'end_to_end_qualified': False}), flush=True)


if __name__ == '__main__':
    main()
