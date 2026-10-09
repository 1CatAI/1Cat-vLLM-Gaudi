# SPDX-License-Identifier: Apache-2.0
"""Immutable additive two-node vocabulary softmax; serving default stays off."""
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
    parser.add_argument('--reuse-addon-build', type=Path,
                        help='Reuse a fingerprinted, unchanged PT2 translation unit')
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root, output = workspace.parent, args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.runtime_profile.read_text())
    parent = Path(profile['environment']['VLLM_HPU_DSV41_UNIQUE_KERNEL']).resolve()
    source = output / 'source'
    kernels = source / 'kernels'
    kernels.mkdir(parents=True)
    files = ['deepseek_v41_softmax_shuffle.h', 'deepseek_v41_softmax_parts_gaudi2.c',
             'deepseek_v41_softmax_normalize_gaudi2.c']
    for name in files:
        shutil.copy2(workspace / 'csrc/deepseek_v41_unique/kernels' / name, kernels / name)
    for name in ('softmax.cpp', 'setup.py'):
        shutil.copy2(workspace / 'csrc/deepseek_v41_unique/pytorch' / name, source / name)
    shutil.copy2(workspace / 'csrc/deepseek_v41_unique/softmax_host.cpp', source / 'softmax_host.cpp')
    objects = []
    for name in files[1:]:
        stem = Path(name).stem
        subprocess.run(['tpc-clang', '-Wall', '-Werror', '-O2', '-march=gaudi2',
                        str(kernels / name), '-c', '-o', str(output / f'{stem}.o')], check=True)
        subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                        f'./{stem}.o', f'{stem}_embedded.o'], cwd=output, check=True)
        objects.append(str(output / f'{stem}_embedded.o'))
        with (output / f'{stem}.asm').open('w') as asm:
            subprocess.run(['tpc-llvm-objdump', '--arch-name=tpc', '--mcpu=gaudi2', '-d',
                            str(output / f'{stem}.o')], stdout=asm, check=True)
    gc = output / 'libdeepseek_v41_unique_kernels.so'
    subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                    '-I/usr/include/habanalabs', str(source / 'softmax_host.cpp'), *objects,
                    '-ldl', '-Wl,-z,noexecstack', '-Wl,--no-undefined', '-o', str(gc)], check=True)
    environment = dict(os.environ, **profile['environment'])
    for key in tuple(environment):
        if key.startswith('DSV41_UNIQUE_ONLY_'):
            del environment[key]
    environment.update(MAX_JOBS=str(len(os.sched_getaffinity(0))), DSV41_UNIQUE_ONLY_VOCAB_SOFTMAX='1',
        GAUDI_PYTORCH_BRIDGE_ROOT=str(root / 'gaudi-pytorch-bridge-tp2-fused-native'),
        GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=str(root / 'builds/pytorch_bridge_tp2_fused_native_upstream'),
        GAUDI_PYTORCH_BRIDGE_DEPS_ROOT=str(root / 'builds/pytorch_modules_multi_build/torch'
                                         / 'py3.12/pt2.11.0/Release/_deps'))
    if args.reuse_addon_build:
        reuse = args.reuse_addon_build.resolve()
        for name in ('softmax.cpp', 'setup.py'):
            if digest(reuse / 'source' / name) != digest(source / name):
                raise ValueError('Reused PT2 source differs from current translation unit')
        old_manifest = json.loads((reuse / 'installed/deepseek_v41_unique_build.json').read_text())
        binaries = list((reuse / 'addon').glob('hpu_dsv41_vocab_softmax_pt2*.so'))
        if len(binaries) != 1 or digest(binaries[0]) != old_manifest['binaries'][binaries[0].name]:
            raise ValueError('Reused PT2 binary differs from its build manifest')
        (output / 'addon').mkdir()
        shutil.copy2(binaries[0], output / 'addon' / binaries[0].name)
        (output / 'build.log').write_text(f'Unchanged PT2 registration reused from {reuse}\n')
    else:
        with (output / 'build.log').open('w') as log:
            subprocess.run([sys.executable, 'setup.py', 'build_ext', '--build-lib', str(output / 'addon'),
                            '--build-temp', str(output / 'temp')], cwd=source, env=environment,
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
    for addon in (output / 'addon').glob('*.so'):
        shutil.copy2(addon, installed / addon.name)
        manifest['binaries'][addon.name] = digest(addon)
        manifest.setdefault('extra_registrations', []).append(addon.name)
    manifest['binaries'][gc.name] = digest(gc)
    manifest['vocab_softmax'] = dict(parent=str(parent), parent_sha256=digest(parent),
        default_enabled=False, source={str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()})
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    old = profile['environment']['VLLM_HPU_DSV41_UNIQUE_KERNEL']
    for key, value in list(profile['environment'].items()):
        if value == old:
            profile['environment'][key] = str(installed / gc.name)
    profile['environment']['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR'] = str(installed)
    profile['environment']['VLLM_HPU_DSV41_SOFTMAX_PARENT_KERNEL'] = str(parent)
    profile['environment']['VLLM_HPU_DSV41_DSPARK_VOCAB_SOFTMAX'] = '0'
    profile['candidate'] = 'Default-off full-q two-node softmax; probabilities/teacher gate pending'
    profile['diagnostic_only'] = True
    # The selected GC and its same-basename delegation chain are validated by
    # prepare_native_libraries against the locked manifest. The companion
    # process-map checker intentionally requires unique names, so only the
    # additive PT2 registration belongs in its list.
    for library in installed.glob('hpu_dsv41_vocab_softmax_pt2*.so'):
        profile.setdefault('additional_libraries', []).append(dict(path=str(library), sha256=digest(library)))
    profile.setdefault('configuration_files', []).append(dict(path=str(manifest_path), sha256=digest(manifest_path)))
    (output / 'runtime-profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    (output / 'build.json').write_text(json.dumps(manifest['vocab_softmax'], indent=2)+'\n')
    print(output / 'runtime-profile.json', flush=True)


if __name__ == '__main__':
    main()
