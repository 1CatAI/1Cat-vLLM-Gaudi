# SPDX-License-Identifier: Apache-2.0
"""Build the per-round ledger from an already archived official companion trace."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parser-source', type=Path, required=True,
                        help='Preserved collector/extractor source from a comparable capture')
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    capture = run / 'official-16k-eos-trace/capture'
    manifest = json.loads((capture / 'manifest.json').read_text())
    if not manifest.get('natural_eos_qualified') or manifest.get('diagnostic_only'):
        raise ValueError('A complete official EOS companion capture is required')
    if {row['rank'] for row in manifest['hardware']} != {0, 1, 2, 3}:
        raise ValueError('Four owned rank captures are required')
    analysis = run / 'analysis'
    if analysis.exists():
        raise FileExistsError('Reuse existing analysis; do not overwrite a partial attempt')
    steps = [
        [str(args.parser_source / 'collect_deepseek_v41_trace.py'), str(run), '--output', str(analysis),
         '--capture-dir', str(capture), '--raw-workers', '4', '--decode-roi-padding-ms', '50'],
        *[['tools/restore_deepseek_v41_raw_debug_symbols.py', str(run), '--rank', str(rank),
           '--allow-unresolved'] for rank in range(4)],
        ['tools/report_deepseek_v41_raw_entry_periods.py', str(analysis)],
        ['tools/report_deepseek_v41_c6_named_periods.py', str(analysis)],
        ['tools/report_deepseek_v41_round_anatomy.py', str(analysis), '--output', str(run / 'round-anatomy'),
         '--serving-run', str(run)],
    ]
    record = dict(source_capture=str(capture), collector=str(args.parser_source),
                  collector_sha256=hashlib.sha256(
                      (args.parser_source / 'collect_deepseek_v41_trace.py').read_bytes()).hexdigest(),
                  new_device_acquisition=False, status='offline analysis running', steps=[])
    evidence = run / 'round-analysis-driver.json'
    try:
        for index, command in enumerate(steps):
            with (run / f'round-analysis-{index}.log').open('xb') as log:
                completed = subprocess.run([sys.executable, *command], cwd=workspace, stdout=log,
                                           stderr=subprocess.STDOUT)
            if completed.returncode:
                # Legacy collectors require exported private recipes even
                # after successfully normalizing all raw timelines. Recover
                # those timelines and exact cache/debug identities offline;
                # never acquire another profiler run to supply graph exports.
                parsed = all((analysis / f'rank{rank}/hardware.jsonl.gz').is_file()
                             and (analysis / f'rank{rank}/inventory.json').is_file() for rank in range(4))
                if index == 0 and parsed:
                    recovery = ['tools/resume_deepseek_v41_round_analysis.py', '--run', str(run)]
                    with (run / 'round-analysis-recovery.log').open('xb') as log:
                        subprocess.run([sys.executable, *recovery], cwd=workspace, stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                    record.update(status='complete; recovered normalized raw trace without graph exports',
                                  legacy_collector_returncode=completed.returncode)
                    record['steps'].append(recovery)
                    break
                completed.check_returncode()
            record['steps'].append(command)
            evidence.write_text(json.dumps(record, indent=2) + '\n')
        else:
            record['status'] = 'complete; submission-window activity is not Target completion latency'
    except Exception as error:
        record.update(status='failed; preserve partial artifacts', error=str(error))
        raise
    finally:
        evidence.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
