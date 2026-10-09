# SPDX-License-Identifier: Apache-2.0
"""Build isolated native future plan against the exact production Bridge."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-profile', type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root, output = workspace.parent, args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    for name in ('dsv41_future_plan.cpp', 'dsv41_future_receive_publisher.h'):
        shutil.copy2(workspace / 'tools/communication' / name, source / name)
    setup = (workspace / 'tools/communication/setup_dsv41_micro_compute.py').read_text()
    setup = setup.replace('dsv41_micro_compute', 'dsv41_future_plan')
    # Reuse locked generated dependency paths supplied by the maintained PT2 build.
    setup = setup.replace('str(build / "_deps")', 'os.environ["GAUDI_PYTORCH_BRIDGE_DEPS_ROOT"]')
    setup = setup.replace('str(build / "_deps" / path)',
                          'str(Path(os.environ["GAUDI_PYTORCH_BRIDGE_DEPS_ROOT"]) / path)')
    (source / 'setup.py').write_text(setup)
    environment = dict(os.environ, **json.loads(args.runtime_profile.read_text())['environment'])
    environment.update(MAX_JOBS=str(len(os.sched_getaffinity(0))),
        GAUDI_PYTORCH_BRIDGE_ROOT=str(root / 'gaudi-pytorch-bridge-tp2-fused-native'),
        GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=str(root / 'builds/pytorch_bridge_tp2_fused_native_upstream'),
        GAUDI_PYTORCH_BRIDGE_DEPS_ROOT=str(root / 'builds/pytorch_modules_multi_build/torch'
                                         / 'py3.12/pt2.11.0/Release/_deps'))
    with (output / 'build.log').open('w') as log:
        subprocess.run([sys.executable, 'setup.py', 'build_ext', '--build-lib', str(output / 'addon'),
                        '--build-temp', str(output / 'temp')], cwd=source, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    files = [*source.iterdir(), *(output / 'addon').glob('*.so')]
    (output / 'build.json').write_text(json.dumps(dict(
        source_and_binary={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},
        serving_selected=False, performance_qualified=False), indent=2)+'\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
