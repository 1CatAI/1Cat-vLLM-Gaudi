# SPDX-License-Identifier: Apache-2.0
"""Immutable Router-only compound reusing the parent's exact TPC ELF."""
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-profile', type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root, output = workspace.parent, args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.runtime_profile.read_text())
    parent = Path(profile['environment']['VLLM_HPU_DSV41_UNIQUE_KERNEL']).resolve()
    source = output / 'source'
    translation = source / 'csrc/deepseek_v41_unique/pytorch'
    translation.mkdir(parents=True)
    files = ('csrc/deepseek_v41_unique/router_ready_host.cpp',
             'csrc/deepseek_v41_unique/pytorch/router_ready.cpp',
             'csrc/deepseek_v41_unique/pytorch/setup.py')
    for name in files:
        shutil.copy2(workspace / name, source / name)
    gc = output / 'libdeepseek_v41_unique_kernels.so'
    subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                    '-I/usr/include/habanalabs', str(source / files[0]), '-ldl', '-Wl,-z,noexecstack',
                    '-Wl,--no-undefined', '-o', str(gc)], check=True)
    environment = dict(os.environ, **profile['environment'])
    for key in tuple(environment):
        if key.startswith('DSV41_UNIQUE_ONLY_'):
            del environment[key]
    environment.update(MAX_JOBS=str(len(os.sched_getaffinity(0))), DSV41_UNIQUE_ONLY_ROUTER_READY='1',
        GAUDI_PYTORCH_BRIDGE_ROOT=str(root / 'gaudi-pytorch-bridge-tp2-fused-native'),
        GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=str(root / 'builds/pytorch_bridge_tp2_fused_native_upstream'),
        GAUDI_PYTORCH_BRIDGE_DEPS_ROOT=str(root / 'builds/pytorch_modules_multi_build/torch'
                                         / 'py3.12/pt2.11.0/Release/_deps'))
    with (output / 'build.log').open('w') as log:
        subprocess.run([sys.executable, 'setup.py', 'build_ext', '--build-lib', str(output / 'addon'),
                        '--build-temp', str(output / 'temp')], cwd=translation, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    installed = output / 'installed'
    installed.mkdir()
    registration_root = Path(profile['environment']['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR'])
    for path in registration_root.iterdir():
        if not path.is_file() or path.name == gc.name:
            continue
        if path.suffix == '.json':
            shutil.copy2(path, installed / path.name)
        else:
            (installed / path.name).symlink_to(path.resolve())
    shutil.copy2(gc, installed / gc.name)
    manifest_path = installed / 'deepseek_v41_unique_build.json'
    manifest = json.loads(manifest_path.read_text())
    addons = list((output / 'addon').glob('hpu_dsv41_router_ready_pt2*.so'))
    if len(addons) != 1:
        raise RuntimeError('Expected one additive prepared Router registration')
    for addon in addons:
        shutil.copy2(addon, installed / addon.name)
        manifest['binaries'][addon.name] = digest(addon)
        manifest.setdefault('extra_registrations', []).append(addon.name)
        profile.setdefault('additional_libraries', []).append(
            dict(path=str(installed / addon.name), sha256=digest(addon)))
    manifest['binaries'][gc.name] = digest(gc)
    manifest['router_ready'] = dict(parent=str(parent), parent_sha256=digest(parent), default_enabled=False,
        source={name: digest(source / name) for name in files},
        arithmetic='Unchanged inherited scaled Router ELF; actual product512, columns0..383; MME consumes FFN FP8 rows')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    profile['environment'].update(GC_KERNEL_PATH=str(installed / gc.name),
        VLLM_HPU_DSV41_UNIQUE_KERNEL=str(installed / gc.name),
        VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR=str(installed),
        VLLM_HPU_DSV41_ROUTER_READY_PARENT_KERNEL=str(parent),
        VLLM_HPU_DSV41_DSPARK_ROUTER_READY_FP8='0', VLLM_HPU_DSV41_DSPARK_DRAFT_MHC='0')
    profile.setdefault('configuration_files', []).append(dict(path=str(manifest_path), sha256=digest(manifest_path)))
    profile['candidate'] = 'Default-off prepared Router; inherited-ELF/shape and native teacher/timing gates pending'
    profile['diagnostic_only'] = True
    (output / 'runtime-profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    (output / 'build.json').write_text(json.dumps(manifest['router_ready'], indent=2) + '\n')
    print(output / 'runtime-profile.json', flush=True)


if __name__ == '__main__':
    main()
