# SPDX-License-Identifier: Apache-2.0
"""Build the round ledger from a retired serving batch's companion capture."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference-kernels', type=Path, required=True)
    parser.add_argument('--raw-workers', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--raw-chunk-ms', type=float, default=2000)
    args = parser.parse_args()
    run = args.run.resolve()
    released = json.loads((run / 'device-release.json').read_text())
    if not released.get('healthy_cards_idle') or not released.get('checked_before_unlock'):
        raise ValueError('Retire the owned serving process and release healthy cards before offline analysis')
    summary = json.loads((run / 'sampled-batch-summary.json').read_text())
    if not summary.get('requests'):
        raise ValueError('The complete request/round ledger is required')
    capture = run / 'official-16k-eos-trace/capture'
    if json.loads((capture / 'manifest.json').read_text())['phase'] != 'decode':
        raise ValueError('Use the companion decode capture from this same batch')
    analysis, anatomy = run / 'trace-analysis-round-ledger', run / 'round-anatomy'
    tools = Path(__file__).resolve().parent
    record = dict(run=str(run), formal_round_ledger=str(run / 'sampled-batch-summary.json'),
                  capture=str(capture), anatomy=str(anatomy), commands=[], analysis_sources={},
                  status='offline analysis', raw_workers=args.raw_workers, raw_chunk_ms=args.raw_chunk_ms)
    record_path = run / 'round-anatomy-pipeline.json'

    def step(name, *values):
        command = [sys.executable, str(tools / name), *map(str, values)]
        record['commands'].append(command)
        record['analysis_sources'][name] = hashlib.sha256((tools / name).read_bytes()).hexdigest()
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        with (run / (name.removesuffix('.py') + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        if not (analysis / 'collection.json').exists():
            step('collect_deepseek_v41_trace.py', run, '--capture-dir', capture,
                 '--output', analysis, '--raw-workers', args.raw_workers, '--raw-chunk-ms', args.raw_chunk_ms,
                 '--decode-roi-padding-ms', 30)
        step('restore_deepseek_v41_cached_trace_metadata.py', run, analysis,
             '--recipe-cache', (run / 'recipe_cache').resolve(), '--allow-partial')
        step('report_deepseek_v41_raw_entry_periods.py', analysis)
        step('report_deepseek_v41_round_anatomy.py', analysis, '--output', anatomy,
             '--reference-kernels', args.reference_kernels.resolve())
        step('report_deepseek_v41_native_target_periods.py', analysis, '--rank', 0)
        record.update(status='complete', actual_target_completion_latency_ms=None,
                      windows='Raw-clock matched host-entry periods; activity union is not completion latency')
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
