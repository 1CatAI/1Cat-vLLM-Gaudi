# SPDX-License-Identifier: Apache-2.0
"""Build a numerical boundary probe, retaining the parent TPC database exactly."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--reuse-addon', type=Path)
    parser.add_argument('--extra-registration', type=Path)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root = workspace.parent
    build = args.build_root.resolve()
    build.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.profile.read_text())
    parent = Path(profile['environment']['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR'])
    snapshot = build / 'source'
    shutil.copytree(workspace / 'csrc/deepseek_v41_unique/pytorch', snapshot)
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith('DSV41_UNIQUE_ONLY_'):
            del environment[key]
    environment.update(DSV41_UNIQUE_ONLY_W13_BOUNDARY='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0',
                       MAX_JOBS=str(len(os.sched_getaffinity(0))),
                       GAUDI_PYTORCH_BRIDGE_ROOT=str(root / 'gaudi-pytorch-bridge-tp2-fused-native'),
                       GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=str(root / 'builds/pytorch_bridge_tp2_fused_native_upstream'),
                       GAUDI_PYTORCH_BRIDGE_DEPS_ROOT=str(root / 'builds/pytorch_modules_multi_build/torch'
                                                        '/py3.12/pt2.11.0/Release/_deps'))
    if args.reuse_addon:
        locked = json.loads((args.reuse_addon / 'installed/deepseek_v41_unique_build.json').read_text())
        for name in ('setup.py', 'w13_boundary_probe.cpp'):
            key = str(workspace / 'csrc/deepseek_v41_unique/pytorch' / name)
            if locked['w13_boundary_probe']['sources'][key] != digest(snapshot / name):
                raise ValueError('Changed probe cannot reuse compiled addon')
        shutil.copytree(args.reuse_addon / 'addon', build / 'addon')
        for path in (build / 'addon').glob('*.so'):
            if locked['binaries'][path.name] != digest(path):
                raise ValueError('Probe addon differs from its manifest')
    else:
        with (build / 'compile.log').open('w') as log:
            subprocess.run([sys.executable, 'setup.py', 'build_ext', '--build-lib', str(build / 'addon'),
                            '--build-temp', str(build / 'addon-temp')], cwd=snapshot, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    installed = build / 'installed'
    installed.mkdir()
    for path in parent.iterdir():
        if path.is_file():
            if path.suffix == '.json' or path.name == 'libdeepseek_v41_unique_kernels.so':
                shutil.copy2(path, installed / path.name)
            else:
                (installed / path.name).symlink_to(path.resolve())
    manifest_path = installed / 'deepseek_v41_unique_build.json'
    manifest = json.loads(manifest_path.read_text())
    for path in (build / 'addon').glob('*.so'):
        shutil.copy2(path, installed / path.name)
        manifest['binaries'][path.name] = digest(path)
        manifest.setdefault('extra_registrations', []).append(path.name)
    if args.extra_registration:
        source = args.extra_registration.resolve()
        old = json.loads((source.parent / 'deepseek_v41_unique_build.json').read_text())
        if old['binaries'][source.name] != digest(source):
            raise ValueError('Legacy diagnostic addon differs from its manifest')
        if source.name in manifest.get('extra_registrations', []):
            raise ValueError('Duplicate operator registration')
        shutil.copy2(source, installed / source.name)
        manifest['binaries'][source.name] = digest(source)
        manifest['extra_registrations'].append(source.name)
        manifest['legacy_diagnostic_registration'] = dict(source=str(source), sha256=digest(source),
                                                         performance_vote=False)
    manifest['w13_boundary_probe'] = dict(diagnostic_only=True, performance_vote=False,
                                        parent_gc_sha256=digest(parent / 'libdeepseek_v41_unique_kernels.so'),
                                        sources={str(workspace / 'csrc/deepseek_v41_unique/pytorch' / name):
                                                 digest(snapshot / name)
                                                 for name in ('setup.py', 'w13_boundary_probe.cpp')})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    profile['environment'].update(VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR=str(installed),
                                  GC_KERNEL_PATH=str(installed / 'libdeepseek_v41_unique_kernels.so'))
    profile['diagnostic_only'] = 'Materialized W13 boundaries; cannot establish performance gain'
    (build / 'runtime-profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    print(build / 'runtime-profile.json', flush=True)


if __name__ == '__main__':
    main()
