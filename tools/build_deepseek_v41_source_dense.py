# SPDX-License-Identifier: Apache-2.0
"""Build an additive exact source weight/BF16 consumer in an immutable private directory."""
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
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--codec', choices=('fp8-source', 'bf16-bits12'), default='fp8-source')
    parser.add_argument('--reuse-addon-build', type=Path)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    build, parent = args.build_root.resolve(), args.parent.resolve()
    build.mkdir(parents=True, exist_ok=False)
    kernel_sources, objects = [], []
    bits12 = args.codec == 'bf16-bits12'
    flags = ([] if bits12 else ['-DDSV41_DENSE_SOURCE_INTEGER_TINY=1', '-DDSV41_DENSE_SOURCE_SIGNED=1'])
    kernel_name = 'deepseek_v41_dense_bits12_gaudi2' if bits12 else 'deepseek_v41_dense_source_fp8_decode_gaudi2'
    for name in (kernel_name,):
        source = workspace / f'csrc/deepseek_v41_unique/kernels/{name}.c'
        kernel_sources.append(source)
        subprocess.run(['tpc-clang', '-Wall', '-Werror', '-O2', '-march=gaudi2',
                    '-I/usr/lib/habanatools/include', *flags, '-I'+str(workspace/'csrc/deepseek_v4/include'),
                    str(source), '-c', '-o', str(build / f'{name}.o')], check=True)
        subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                    f'./{name}.o', f'{name}_embedded.o'], cwd=build, check=True)
        objects.append(str(build / f'{name}_embedded.o'))
    gc = build / 'libdeepseek_v41_unique_kernels.so'
    host = workspace / 'csrc/deepseek_v41_unique/source_dense_host.cpp'
    subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                    *(['-DDSV41_DENSE_BITS12=1'] if bits12 else []),
                    '-I/usr/include/habanalabs', str(host), *objects,
                    '-ldl', '-Wl,-z,noexecstack', '-Wl,--no-undefined', '-o', str(gc)], check=True)
    snapshot = build / 'source'
    for directory in ('deepseek_v4', 'deepseek_v41_unique/pytorch', 'deepseek_v41_unique/kernels',
                      'deepseek_v41_unique/include', 'flashinfer_gaudi'):
        shutil.copytree(workspace / 'csrc' / directory, snapshot / 'csrc' / directory)
    environment = dict(os.environ)
    root = workspace.parent
    for key in tuple(environment):
        if key.startswith('DSV41_UNIQUE_ONLY_'):
            del environment[key]
    environment.update(MAX_JOBS=str(len(os.sched_getaffinity(0))), DSV41_UNIQUE_ONLY_SOURCE_DENSE='1',
        DSV41_UNIQUE_SOURCE_DENSE_BITS12=str(int(bits12)),
        GAUDI_PYTORCH_BRIDGE_ROOT=str(root / 'gaudi-pytorch-bridge-tp2-fused-native'),
        GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=str(root / 'builds/pytorch_bridge_tp2_fused_native_upstream'),
        GAUDI_PYTORCH_BRIDGE_DEPS_ROOT=str(root / 'builds/pytorch_modules_multi_build/torch'
                                         / 'py3.12/pt2.11.0/Release/_deps'))
    if args.reuse_addon_build:
        locked = json.loads((args.reuse_addon_build/'build.json').read_text())
        if locked.get('codec', 'fp8-source') != args.codec:
            raise ValueError('Cannot reuse registration with another source encoding')
        for relative in ('csrc/deepseek_v41_unique/pytorch/source_dense.cpp',
                         'csrc/deepseek_v41_unique/pytorch/setup.py'):
            path = workspace/relative
            if locked['sources'].get(str(path)) != digest(path):
                raise ValueError('Changed frontend cannot reuse its compiled addon')
        previous = json.loads((args.reuse_addon_build/'installed/deepseek_v41_unique_build.json').read_text())
        for addon in (args.reuse_addon_build/'addon').glob('*.so'):
            if previous['binaries'].get(addon.name) != digest(addon):
                raise ValueError('Registration differs from the source-locked binary')
        shutil.copytree(args.reuse_addon_build/'addon', build/'addon')
    else:
        with (build / 'addon-build.log').open('w') as log:
            subprocess.run([sys.executable, 'setup.py', 'build_ext', '--build-lib', str(build / 'addon'),
                            '--build-temp', str(build / 'addon-temp')],
                           cwd=snapshot / 'csrc/deepseek_v41_unique/pytorch', env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    installed = build / 'installed'
    installed.mkdir()
    for path in parent.iterdir():
        if path.is_file() and path.name != gc.name:
            if path.suffix == '.json':
                shutil.copy2(path, installed / path.name)
            else:
                (installed / path.name).symlink_to(path.resolve())
    shutil.copy2(gc, installed / gc.name)
    manifest_path = installed / 'deepseek_v41_unique_build.json'
    manifest = json.loads(manifest_path.read_text())
    addons = list((build / 'addon').glob('*.so'))
    if len(addons) != 1:
        raise RuntimeError('Expected exactly one private registration')
    for addon in addons:
        shutil.copy2(addon, installed / addon.name)
        manifest['binaries'][addon.name] = digest(addon)
        manifest.setdefault('extra_registrations', []).append(addon.name)
    manifest['binaries'][gc.name] = digest(gc)
    relatives = ('csrc/deepseek_v41_unique/pytorch/source_dense.cpp',
                 'csrc/deepseek_v41_unique/pytorch/setup.py',
                 f'csrc/deepseek_v41_unique/kernels/{kernel_name}.c',
                 'csrc/deepseek_v4/kernels/deepseek_v41_qnorm_quant_gaudi2.c',
                 'csrc/deepseek_v4/kernels/deepseek_v41_kv_norm_rope_bf16_gaudi2.c')
    record = dict(parent_gc_path=str(parent / gc.name), parent_gc_sha256=digest(parent / gc.name),
                  default_enabled=False, codec=args.codec, sources={str(p): digest(p) for p in
                                                  (*kernel_sources, host, *(workspace / p for p in relatives))},
                  boundary=('lossless high8/low4 planes -> private BF16 weight -> BF16 GEMM' if bits12 else
                            'source I8 E4M3FN + positive I16 UE8M0 -> private exact BF16 weight -> BF16 GEMM'))
    manifest['source_dense_build'] = record
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (build / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
    profile = json.loads(args.profile.read_text())
    old = profile['environment']['VLLM_HPU_DSV41_UNIQUE_KERNEL']
    for key, value in list(profile['environment'].items()):
        if value == old:
            profile['environment'][key] = str(installed / gc.name)
    profile['environment']['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR'] = str(installed)
    profile['environment']['VLLM_HPU_DSV41_SOURCE_DENSE_PARENT_KERNEL'] = str(parent / gc.name)
    for addon in addons:
        path = installed / addon.name
        profile.setdefault('additional_libraries', []).append(dict(path=str(path), sha256=digest(path)))
    profile.setdefault('configuration_files', []).append(dict(path=str(manifest_path), sha256=digest(manifest_path)))
    profile['candidate'] = dict(name='source_dense', codec=args.codec, default_enabled=False, formal_qualified=False)
    (build / 'runtime-profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    print(installed, flush=True)


if __name__ == '__main__':
    main()
