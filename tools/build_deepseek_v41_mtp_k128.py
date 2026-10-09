# SPDX-License-Identifier: Apache-2.0
"""Add an immutable draft K128 entrypoint to the qualified decoder installation."""
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


def install_registration(parent, installed, addon):
    """Retain parent artifacts without ever writing through their symlinks."""
    installed.mkdir()
    for path in parent.iterdir():
        if not path.is_file() or path.name == addon.name:
            continue
        if path.suffix == '.json':
            shutil.copy2(path, installed / path.name)
        else:
            (installed / path.name).symlink_to(path.resolve())
    shutil.copy2(addon, installed / addon.name)
    manifest_path = installed / 'deepseek_v41_unique_build.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['binaries'][addon.name] = digest(addon)
    manifest['extra_registrations'] = list(dict.fromkeys([*manifest.get('extra_registrations', []), addon.name]))
    return manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-profile', type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root, output = workspace.parent, args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.runtime_profile.read_text())
    source = output / 'source'
    translation = source / 'csrc/deepseek_v41_unique/pytorch'
    translation.mkdir(parents=True)
    inputs = ('csrc/deepseek_v41_unique/pytorch/setup.py', 'csrc/deepseek_v41_unique/pytorch/mtp_k128.cpp',
              'csrc/deepseek_v4/pytorch/hpu_dsv41_expert_n256_pt2.cpp',
              'csrc/deepseek_v4/pytorch/dsv41_c6_expert_backend.h')
    for relative in inputs:
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace / relative, destination)
    environment = dict(os.environ, **profile['environment'])
    for key in tuple(environment):
        if key.startswith('DSV41_UNIQUE_ONLY_'):
            del environment[key]
    environment.update(MAX_JOBS=str(len(os.sched_getaffinity(0))),
                       DSV41_UNIQUE_ONLY_MTP_K128='1',
                       GAUDI_PYTORCH_BRIDGE_ROOT=str(root / 'gaudi-pytorch-bridge-tp2-fused-native'),
                       GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=str(root / 'builds/pytorch_bridge_tp2_fused_native_upstream'),
                       GAUDI_PYTORCH_BRIDGE_DEPS_ROOT=str(root / 'builds/pytorch_modules_multi_build/torch' /
                                                          'py3.12/pt2.11.0/Release/_deps'))
    with (output / 'build.log').open('w') as log:
        subprocess.run([
            sys.executable, 'setup.py', 'build_ext', '--build-lib',
            str(output / 'addon'), '--build-temp',
            str(output / 'temp')
        ],
                       cwd=translation,
                       env=environment,
                       stdout=log,
                       stderr=subprocess.STDOUT,
                       check=True)
    addons = list((output / 'addon').glob('hpu_dsv41_mtp_k128_pt2*.so'))
    if len(addons) != 1:
        raise RuntimeError('Expected exactly one draft K128 registration')
    installed = output / 'installed'
    parent = Path(profile['environment']['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR']).resolve()
    addon = addons[0]
    manifest_path, manifest = install_registration(parent, installed, addon)
    manifest['mtp_k128'] = dict(parent=str(parent),
                                default_enabled=False,
                                source={relative: digest(source / relative)
                                        for relative in inputs},
                                contract='C1-C6, top3/E128, H5120/I640, normal scales; common BF16 decoder unchanged')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    # Keep the immutable GC bytes, while naming the same installation in all
    # loader/database fields. The launcher validates this directory contract.
    for key in ('GC_KERNEL_PATH', 'VLLM_HPU_DSV41_UNIQUE_KERNEL'):
        profile['environment'][key] = str(installed / 'libdeepseek_v41_unique_kernels.so')
    profile['environment']['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR'] = str(installed)
    profile['environment']['VLLM_HPU_DSV41_DSPARK_MTP_K128'] = '0'
    profile['candidate'] = 'Default-off draft K128 BF16 entrypoint; teacher/native gates pending'
    profile['diagnostic_only'] = True
    profile['additional_libraries'] = [
        row for row in profile.get('additional_libraries', []) if Path(row['path']).name != addon.name
    ]
    profile['additional_libraries'].append(dict(path=str(installed / addon.name), sha256=digest(addon)))
    profile.setdefault('configuration_files', []).append(dict(path=str(manifest_path), sha256=digest(manifest_path)))
    (output / 'runtime-profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    (output / 'build.json').write_text(json.dumps(manifest['mtp_k128'], indent=2) + '\n')
    print(output / 'runtime-profile.json', flush=True)


if __name__ == '__main__':
    main()
