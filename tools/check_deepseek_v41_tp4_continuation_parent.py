# SPDX-License-Identifier: Apache-2.0
"""Own production Engram backing for the four-rank continuation component."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--speed-probe', action='store_true',
                        help='Candidate timing screen with exact saved tokens; broader state/lifecycle gate deferred')
    parser.add_argument('--qualify-positive', action='store_true')
    parser.add_argument('--reuse-reference', type=Path)
    parser.add_argument('--diagnose-state', action='store_true')
    parser.add_argument('--state-reference-original-copy', action='store_true')
    parser.add_argument('--state-reference-paged-projection', action='store_true')
    parser.add_argument('--state-reference-paged-gather', action='store_true')
    parser.add_argument('--state-reference-visible-prefix', action='store_true')
    parser.add_argument('--state-reference-logical-mla', action='store_true')
    parser.add_argument('--state-reference-index-mirror', action='store_true')
    parser.add_argument('--state-reference-shared-main', action='store_true')
    parser.add_argument('--state-reference-native-groups', action='store_true')
    parser.add_argument('--shared-stage-replay', action='store_true')
    parser.add_argument('--tp2-ordered-selection', action='store_true')
    parser.add_argument('--selection-trace-only', action='store_true')
    parser.add_argument('--state-reference-decode-metadata', action='store_true')
    parser.add_argument('--state-reference-post-collapse', action='store_true')
    parser.add_argument('--state-reference-interlayer-collapse', action='store_true')
    parser.add_argument('--state-reference-feature-silu', action='store_true')
    parser.add_argument('--state-reference-mirror-partition', action='store_true')
    parser.add_argument('--production-visible-prefix', action='store_true')
    parser.add_argument('--candidate-local-index-queries', action='store_true')
    parser.add_argument('--diagnose-local-query-state', action='store_true')
    parser.add_argument('--continuous-warm-steps', type=int, default=0)
    parser.add_argument('--measure-continuous-reference', action='store_true')
    args = parser.parse_args()
    from vllm_gaudi.ops.deepseek_v41_residency import EngramResidency, table_regions
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    owner = EngramResidency(table_regions(args.prepared), 224 * 1024**3, device_layers=(1,))
    process = None
    try:
        owner.start()
        path = root / 'engram-bindings.json'
        path.write_text(json.dumps(owner.worker_bindings(), indent=2)+'\n')
        (root / 'engram-residency.json').write_text(json.dumps(dict(admission=owner.admission,
                                                                   owners=owner.reports), indent=2)+'\n')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   str(Path(__file__).with_name('check_deepseek_v41_tp4_continuation.py')),
                   str(args.prepared), '--bindings', str(path)]
        if args.speed_probe:
            command.append('--speed-probe')
        if args.qualify_positive:
            command.append('--qualify-positive')
        if args.reuse_reference:
            command.extend(['--reuse-reference', str(args.reuse_reference.resolve())])
        if args.diagnose_state:
            command.append('--diagnose-state')
        if args.state_reference_original_copy:
            command.append('--state-reference-original-copy')
        if args.state_reference_paged_projection:
            command.append('--state-reference-paged-projection')
        if args.state_reference_paged_gather:
            command.append('--state-reference-paged-gather')
        if args.state_reference_visible_prefix:
            command.append('--state-reference-visible-prefix')
        if args.state_reference_logical_mla:
            command.append('--state-reference-logical-mla')
        if args.state_reference_index_mirror:
            command.append('--state-reference-index-mirror')
        if args.state_reference_decode_metadata:
            command.append('--state-reference-decode-metadata')
        if args.state_reference_post_collapse:
            command.append('--state-reference-post-collapse')
        if args.state_reference_interlayer_collapse:
            command.append('--state-reference-interlayer-collapse')
        if args.state_reference_feature_silu:
            command.append('--state-reference-feature-silu')
        if args.state_reference_mirror_partition:
            command.append('--state-reference-mirror-partition')
        if args.state_reference_native_groups:
            command.append('--state-reference-native-groups')
        if args.shared_stage_replay:
            command.append('--shared-stage-replay')
        if args.tp2_ordered_selection:
            command.append('--tp2-ordered-selection')
        if args.selection_trace_only:
            command.append('--selection-trace-only')
        if args.state_reference_shared_main:
            command.append('--state-reference-shared-main')
        if args.production_visible_prefix:
            command.append('--production-visible-prefix')
        if args.candidate_local_index_queries:
            command.append('--candidate-local-index-queries')
        if args.diagnose_local_query_state:
            command.append('--diagnose-local-query-state')
        if args.continuous_warm_steps:
            command.extend(['--continuous-warm-steps', str(args.continuous_warm_steps)])
        if args.measure_continuous_reference:
            command.append('--measure-continuous-reference')
        process = subprocess.Popen(command)
        owner.watch(lambda: process.terminate())
        result = process.wait()
        owner.check()
        raise SystemExit(result)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        owner.close()


if __name__ == '__main__':
    main()
