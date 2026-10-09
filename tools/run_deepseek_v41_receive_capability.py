# SPDX-License-Identifier: Apache-2.0
"""Lease-only standalone SDK capability; no serving/default dispatch changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-event-wait', action='store_true')
    parser.add_argument('--separate-affinity', action='store_true')
    parser.add_argument('--network', action='store_true')
    parser.add_argument('--native-replay', action='store_true')
    parser.add_argument('--joint-plan', action='store_true')
    parser.add_argument('--production-post', action='store_true')
    parser.add_argument('--late-point-copy', action='store_true')
    parser.add_argument('--cold-event-template', action='store_true')
    parser.add_argument('--device-epoch', action='store_true')
    args = parser.parse_args()
    if args.native_replay and not args.network:
        parser.error('--native-replay requires --network')
    if args.joint_plan and not args.native_replay:
        parser.error('--joint-plan requires --native-replay')
    if args.production_post and not args.joint_plan:
        parser.error('--production-post requires --joint-plan')
    if args.late_point_copy and not args.joint_plan:
        parser.error('--late-point-copy requires --joint-plan')
    if args.device_epoch and not (args.joint_plan and args.cold_event_template):
        parser.error('--device-epoch requires --joint-plan --cold-event-template')
    if args.cold_event_template and not args.joint_plan:
        parser.error('--cold-event-template requires --joint-plan')
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    fixture = args.fixtures / f'rank{rank}' if args.network else args.fixtures
    required = ['producer-weight.bin', 'consumer-weight.bin',
                *[f'{kind}-{c}.bin' for c in range(3) for kind in ('input', 'peers')]]
    if args.production_post:
        required += ['norm-weight.bin', *[f'{kind}-{c}.bin' for c in range(3)
                                          for kind in ('residual', 'post', 'comb', 'pre')]]
    if any(not (fixture / name).is_file() for name in required):
        raise FileNotFoundError('Regenerate bounded CPU fixture files before running the capability')
    module = os.environ['HABANA_VISIBLE_MODULES'].split(',')
    if len(module) not in ((2, 4) if args.network else (1,)):
        raise RuntimeError('Capability module count differs from its network mode')
    os.environ['HLS_MODULE_ID'] = module[rank]
    if args.network:
        os.environ['DSV41_RECEIVE_NETWORK'] = '1'
        record = json.loads((Path(os.environ['DSV41_RUN_EVIDENCE']) / 'process.json').read_text())
        cpus = record['modules'][rank]
        os.sched_setaffinity(0, {cpus['main_cpu'], *cpus['helper_cpus']})
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_native_libraries
    prepare_native_libraries()
    parent = Path(os.environ['GC_KERNEL_PATH'])
    candidate = args.build.resolve() / 'libbounded_receive.so'
    os.environ['VLLM_HPU_DSV41_BOUNDED_RECEIVE_PARENT_KERNEL'] = str(parent)
    os.environ['GC_KERNEL_PATH'] = str(candidate)
    if args.separate_affinity:
        os.environ['DSV41_RECEIVE_SEPARATE_AFFINITY'] = '1'
    if args.no_event_wait:
        os.environ['DSV41_RECEIVE_DIAGNOSTIC_NO_WAIT'] = '1'
    if args.native_replay:
        os.environ['DSV41_RECEIVE_NATIVE_REPLAY'] = '1'
        helper = args.build / 'libreceive_event_contract.so'
        os.environ['LD_PRELOAD'] = ':'.join(filter(None, (str(helper), os.environ.get('LD_PRELOAD'))))
    if args.joint_plan:
        os.environ['DSV41_RECEIVE_JOINT_PLAN'] = '1'
    if args.production_post:
        os.environ['DSV41_RECEIVE_PRODUCTION_POST'] = '1'
    if args.late_point_copy:
        os.environ['DSV41_RECEIVE_LATE_POINT_COPY'] = '1'
    if args.device_epoch:
        os.environ['DSV41_RECEIVE_DEVICE_EPOCH'] = '1'
    if args.cold_event_template:
        os.environ['DSV41_RECEIVE_COLD_TEMPLATE'] = '1'
    run = Path(os.environ['DSV41_RUN_EVIDENCE'])
    artifacts = (parent, candidate, args.build / 'check_receive')
    if args.native_replay:
        artifacts += (helper,)
    artifact_name = f'capability-artifacts-rank{rank}.json' if args.network else 'capability-artifacts.json'
    (run / artifact_name).write_text(json.dumps({
        'artifacts': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts],
        'timed': False, 'serving_dispatch_changed': False,
        'producer_wait_removed_diagnostic': args.no_event_wait,
        'separate_physical_copy_queue': args.separate_affinity,
        'mode': ('N-card future-NIC capability' if args.network else 'Single-module future-DMA capability'),
        'performance_tested': False,
        'device_transport_generation': args.device_epoch,
        'native_compute_and_native_NIC_replay': args.native_replay,
        'NIC_and_flag_published_before_compute': args.joint_plan,
    }, indent=2) + '\n')
    if args.debug:
        os.execv('/usr/bin/gdb', ['gdb', '-batch', '-ex', 'run', '-ex', 'bt 12', '--args',
                                str(args.build / 'check_receive'), str(fixture.resolve()), str(run)])
    os.execv(str(args.build / 'check_receive'), ['check_receive', str(fixture.resolve()), str(run)])


if __name__ == '__main__':
    main()
