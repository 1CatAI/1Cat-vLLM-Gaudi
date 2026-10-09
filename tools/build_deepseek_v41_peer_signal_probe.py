# SPDX-License-Identifier: Apache-2.0
"""Build only the isolated tensor-ready capability on immutable inputs."""
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
    for name in ('peer_signal_probe.cpp', 'setup.py'):
        shutil.copy2(workspace / 'csrc/deepseek_v41_unique/pytorch' / name, source / name)
    environment = dict(os.environ)
    environment.update(json.loads(args.runtime_profile.read_text())['environment'])
    for key in tuple(environment):
        if key.startswith('DSV41_UNIQUE_ONLY_'):
            del environment[key]
    environment.update(MAX_JOBS=str(len(os.sched_getaffinity(0))), DSV41_UNIQUE_ONLY_PEER_SIGNAL_PROBE='1',
        GAUDI_PYTORCH_BRIDGE_ROOT=str(root / 'gaudi-pytorch-bridge-tp2-fused-native'),
        GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=str(root / 'builds/pytorch_bridge_tp2_fused_native_upstream'),
        GAUDI_PYTORCH_BRIDGE_DEPS_ROOT=str(root / 'builds/pytorch_modules_multi_build/torch'
                                         / 'py3.12/pt2.11.0/Release/_deps'))
    with (output / 'build.log').open('w') as log:
        subprocess.run([sys.executable, 'setup.py', 'build_ext', '--build-lib', str(output / 'addon'),
                        '--build-temp', str(output / 'temp')], cwd=source, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    files = [*source.iterdir(), *(output / 'addon').glob('*.so')]
    proof = dict(sources_and_binary={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in files if path.is_file()},
                 serving_selected=False, performance_qualified=False,
                 runtime_profile=str(args.runtime_profile.resolve()))
    (output / 'build.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
