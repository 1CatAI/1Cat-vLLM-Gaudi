# SPDX-License-Identifier: Apache-2.0
"""Finish already parsed raw timelines without requiring exported graphs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    workspace = Path(__file__).resolve().parents[1]
    analysis = run / 'analysis'
    capture = run / 'official-16k-eos-trace/capture'
    manifest = json.loads((capture / 'manifest.json').read_text())
    if not manifest.get('natural_eos_qualified') or manifest.get('diagnostic_only'):
        raise ValueError('An official EOS companion capture is required')
    rows = {row['rank']: row for row in manifest['hardware']}
    if set(rows) != set(range(4)):
        raise ValueError('Four captured rank identities are required')
    provenance = []
    ranks = []
    for rank in range(4):
        row, root = rows[rank], analysis / f'rank{rank}'
        inventory = json.loads((root / 'inventory.json').read_text())
        raw = json.loads((root / 'raw-provenance.json').read_text())
        if not inventory.get('hardware_events') or not (root / 'hardware.jsonl.gz').is_file():
            raise ValueError(f'Rank{rank} has no previously parsed hardware timeline')
        for name, key in (('bundle', 'raw_sha256'), ('cpu_trace', 'cpu_sha256')):
            with (capture / row[name]).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != raw[key]:
                    raise ValueError(f'Rank{rank} parsed timeline differs from its capture')
        if not list((root / 'raw-input/debug-info').glob('*_dbg.bin')):
            raise ValueError(f'Rank{rank} has no same-acquisition raw debug symbols')
        stats = json.loads((capture / f'rank{rank}-native-profile-stop.json').read_text())
        pid = row['metadata']['pid']
        ranks.append(dict(rank=rank, pp=0, tp=rank, pid=pid, stats=stats,
                          trace=str(capture / row['cpu_trace']), serialized_recipes=0,
                          exported_graph_metadata_available=False))
        provenance.append(dict(rank=rank, source=raw, hardware_events=inventory['hardware_events']))
        metadata = analysis / f'rank{rank}-metadata.json'
        if not metadata.exists():
            metadata.write_text(json.dumps(dict(rank=rank, pid=pid, source=str(capture / 'manifest.json')),
                                           indent=2) + '\n')
    collection = analysis / 'collection.json'
    if not collection.exists():
        collection.write_text(json.dumps(dict(format='raw_hltv_cpu',
            status='Parsed device/host timelines; graph shapes and placement unobserved',
            source_capture=str(capture), ranks=ranks, provenance=provenance), indent=2) + '\n')
    record_path = run / 'round-analysis-recovery.json'
    if record_path.exists():
        raise FileExistsError('Reuse the recorded recovery; never repeat completed parsing')
    steps = [
        *[['tools/restore_deepseek_v41_cached_recipe_symbols.py', str(run), '--rank', str(rank)]
          for rank in range(4) if (run / f'recipe_cache/rank{rank}').is_dir()
          and not (analysis / f'rank{rank}/cached-binary-symbol-proof.json').exists()],
        *[['tools/restore_deepseek_v41_raw_debug_symbols.py', str(run), '--rank', str(rank),
           '--allow-unresolved'] for rank in range(4)
          if not (analysis / f'rank{rank}/raw-debug-symbol-proof.json').exists()],
        ['tools/report_deepseek_v41_raw_entry_periods.py', str(analysis)],
        ['tools/report_deepseek_v41_c6_named_periods.py', str(analysis)],
        ['tools/report_deepseek_v41_round_anatomy.py', str(analysis), '--output', str(run / 'round-anatomy'),
         '--serving-run', str(run)],
    ]
    record = dict(status='recovering parsed capture', new_acquisition=False, repeated_raw_parser=False,
                  prior_failure=str(run / 'round-analysis-driver.json'), steps=[])
    try:
        for index, command in enumerate(steps):
            with (run / f'round-recovery-{index}.log').open('xb') as log:
                subprocess.run([sys.executable, *command], cwd=workspace, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            record['steps'].append(command)
            record_path.write_text(json.dumps(record, indent=2) + '\n')
        record['status'] = 'complete; shapes/placement remain unobserved without cached graph metadata'
    except Exception as error:
        record.update(status='failed; preserve completed recovery stages', error=str(error))
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
