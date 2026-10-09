# SPDX-License-Identifier: Apache-2.0
"""Immutable Two-plane Router compound; original Router arithmetic outside the new precision epilogue."""
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
    parser.add_argument('--reuse-addon-build', type=Path)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root, output = workspace.parent, args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.runtime_profile.read_text())
    parent = Path(profile['environment']['VLLM_HPU_DSV41_UNIQUE_KERNEL']).resolve()
    source = output / 'source'
    translation = source / 'csrc/deepseek_v41_unique/pytorch'
    translation.mkdir(parents=True)
    files = ('csrc/deepseek_v41_unique/router_pair_host.cpp',
             'csrc/deepseek_v41_unique/pytorch/router_pair.cpp',
             'csrc/deepseek_v41_unique/pytorch/setup.py',
             'csrc/deepseek_v41_unique/kernels/deepseek_v41_router_pair_input_gaudi2.c',
             'csrc/deepseek_v41_unique/kernels/deepseek_v41_router_pair_finish_gaudi2.c',
             'csrc/deepseek_v4/kernels/deepseek_v41_router_logits_top6_gaudi2.c')
    for name in files:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace / name, source / name)
    objects = []
    for name in files[3:5]:
        stem = Path(name).stem
        subprocess.run(['tpc-clang', '-Wall', '-Werror', '-O2', '-march=gaudi2',
                        str(source / name), '-c', '-o', str(output / f'{stem}.o')], check=True)
        subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                        f'./{stem}.o', f'{stem}_embedded.o'], cwd=output, check=True)
        objects.append(str(output / f'{stem}_embedded.o'))
        with (output / f'{stem}.asm').open('w') as asm:
            subprocess.run(['tpc-llvm-objdump', '--arch-name=tpc', '--mcpu=gaudi2', '-d',
                            str(output / f'{stem}.o')], stdout=asm, check=True)
    gc = output / 'libdeepseek_v41_unique_kernels.so'
    subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                    '-I/usr/include/habanalabs', str(source / files[0]), *objects, '-ldl', '-Wl,-z,noexecstack',
                    '-Wl,--no-undefined', '-o', str(gc)], check=True)
    environment = dict(os.environ, **profile['environment'])
    for key in tuple(environment):
        if key.startswith('DSV41_UNIQUE_ONLY_'):
            del environment[key]
    environment.update(MAX_JOBS=str(len(os.sched_getaffinity(0))), DSV41_UNIQUE_ONLY_ROUTER_PAIR='1',
        GAUDI_PYTORCH_BRIDGE_ROOT=str(root / 'gaudi-pytorch-bridge-tp2-fused-native'),
        GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=str(root / 'builds/pytorch_bridge_tp2_fused_native_upstream'),
        GAUDI_PYTORCH_BRIDGE_DEPS_ROOT=str(root / 'builds/pytorch_modules_multi_build/torch'
                                         / 'py3.12/pt2.11.0/Release/_deps'))
    if args.reuse_addon_build:
        reuse = args.reuse_addon_build.resolve()
        for name in files[1:3]:
            if digest(source / name) != digest(reuse / 'source' / name):
                raise ValueError('Reused Router PT2 source differs')
        binaries = list((reuse / 'addon').glob('hpu_dsv41_router_pair_pt2*.so'))
        manifest = json.loads((reuse / 'installed/deepseek_v41_unique_build.json').read_text())
        if len(binaries) != 1 or digest(binaries[0]) != manifest['binaries'][binaries[0].name]:
            raise ValueError('Reused Router PT2 binary differs')
        (output / 'addon').mkdir()
        shutil.copy2(binaries[0], output / 'addon' / binaries[0].name)
        (output / 'build.log').write_text(f'Unchanged PT2 source/binary reused: {reuse}\n')
    else:
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
    addons = list((output / 'addon').glob('hpu_dsv41_router_pair_pt2*.so'))
    if len(addons) != 1:
        raise RuntimeError('Expected one additive prepared Router registration')
    for addon in addons:
        shutil.copy2(addon, installed / addon.name)
        manifest['binaries'][addon.name] = digest(addon)
        manifest.setdefault('extra_registrations', []).append(addon.name)
        profile.setdefault('additional_libraries', []).append(
            dict(path=str(installed / addon.name), sha256=digest(addon)))
    manifest['binaries'][gc.name] = digest(gc)
    manifest['router_pair'] = dict(parent=str(parent), parent_sha256=digest(parent), default_enabled=False,
        source={name: digest(source / name) for name in files},
        arithmetic=('FFN row scale reused without amax; residual activation/weight FP8 planes '
                    'consumed by one GEMM; stock Router selection'))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    profile['environment'].update(GC_KERNEL_PATH=str(installed / gc.name),
        VLLM_HPU_DSV41_UNIQUE_KERNEL=str(installed / gc.name),
        VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR=str(installed),
        VLLM_HPU_DSV41_ROUTER_PAIR_PARENT_KERNEL=str(parent),
        VLLM_HPU_DSV41_DSPARK_ROUTER_READY_FP8='0', VLLM_HPU_DSV41_DSPARK_ROUTER_READY_PAIR='0',
        VLLM_HPU_DSV41_DSPARK_DRAFT_MHC='0')
    profile.setdefault('configuration_files', []).append(dict(path=str(manifest_path), sha256=digest(manifest_path)))
    profile['candidate'] = 'Default-off prepared Router; inherited-ELF/shape and native teacher/timing gates pending'
    profile['diagnostic_only'] = True
    (output / 'runtime-profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    (output / 'build.json').write_text(json.dumps(manifest['router_pair'], indent=2) + '\n')
    print(output / 'runtime-profile.json', flush=True)


if __name__ == '__main__':
    main()
