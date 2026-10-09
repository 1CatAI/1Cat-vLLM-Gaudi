# SPDX-License-Identifier: Apache-2.0
"""Build the private, default-off W2 scale producer and its original SAT chain."""
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
    parser.add_argument('--reuse-addon-build', type=Path, help='Reuse an unchanged fingerprinted PT2 translation unit')
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    build, parent = args.build_root.resolve(), args.parent.resolve()
    build.mkdir(parents=True, exist_ok=False)
    sources, objects = [], []
    for stem in ('scale', 'reduce'):
        name = f'deepseek_v41_w2_ready_{stem}_gaudi2'
        source = workspace / f'csrc/deepseek_v41_unique/kernels/{name}.c'
        sources.append(source)
        subprocess.run(['tpc-clang', '-Wall', '-Werror', '-O2', '-march=gaudi2',
                        '-I/usr/lib/habanatools/include', str(source), '-c', '-o', str(build / f'{name}.o')],
                       check=True)
        subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                        f'./{name}.o', f'{name}_embedded.o'], cwd=build, check=True)
        objects.append(str(build / f'{name}_embedded.o'))
    gc = build / 'libdeepseek_v41_unique_kernels.so'
    host = workspace / 'csrc/deepseek_v41_unique/w2_ready_scale_host.cpp'
    sources.append(host)
    subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                    '-I/usr/include/habanalabs', str(host), *objects, '-ldl', '-Wl,-z,noexecstack',
                    '-Wl,--no-undefined', '-o', str(gc)], check=True)
    snapshot = build / 'source'
    environment = dict(os.environ)
    root = workspace.parent
    environment.update(MAX_JOBS=str(len(os.sched_getaffinity(0))),
        GAUDI_PYTORCH_BRIDGE_ROOT=str(root / 'gaudi-pytorch-bridge-tp2-fused-native'),
        GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=str(root / 'builds/pytorch_bridge_tp2_fused_native_upstream'),
        GAUDI_PYTORCH_BRIDGE_DEPS_ROOT=str(root / 'builds/pytorch_modules_multi_build/torch'
                                         / 'py3.12/pt2.11.0/Release/_deps'),
        DSV41_UNIQUE_ONLY_W2_READY_SCALE='1')
    for key in tuple(environment):
        if key.startswith('DSV41_UNIQUE_ONLY_') and key != 'DSV41_UNIQUE_ONLY_W2_READY_SCALE':
            del environment[key]
    if args.reuse_addon_build:
        locked = json.loads((args.reuse_addon_build / 'build.json').read_text())
        for relative in ('csrc/deepseek_v4/pytorch/hpu_dsv41_expert_n256_pt2.cpp',
                         'csrc/deepseek_v41_unique/pytorch/sat_w2_ready_scale.cpp'):
            source = workspace / relative
            if locked['sources'].get(str(source)) != digest(source):
                raise ValueError('Cannot reuse an addon after changing its compiled source')
        old_manifest = json.loads((args.reuse_addon_build / 'installed/deepseek_v41_unique_build.json').read_text())
        for addon in (args.reuse_addon_build / 'addon').glob('*.so'):
            if old_manifest['binaries'].get(addon.name) != digest(addon):
                raise ValueError('Reused addon differs from its build manifest')
        shutil.copytree(args.reuse_addon_build / 'addon', build / 'addon')
    else:
        for directory in ('deepseek_v4', 'deepseek_v41_unique/pytorch'):
            shutil.copytree(workspace / 'csrc' / directory, snapshot / 'csrc' / directory)
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
    for addon in (build / 'addon').glob('*.so'):
        shutil.copy2(addon, installed / addon.name)
        manifest['binaries'][addon.name] = digest(addon)
        manifest.setdefault('extra_registrations', []).append(addon.name)
    manifest['binaries'][gc.name] = digest(gc)
    sources += [workspace / 'csrc/deepseek_v4/pytorch/hpu_dsv41_expert_n256_pt2.cpp',
                workspace / 'csrc/deepseek_v41_unique/pytorch/sat_w2_ready_scale.cpp']
    manifest['w2_factor_build'] = dict(parent_gc_path=str(parent / gc.name),
        parent_gc_sha256=digest(parent / gc.name), default_enabled=False,
        sources={str(path): digest(path) for path in sources},
        numerical_change='W2 accumulator*(channel*sx), retaining BF16 route boundaries and routing sum order')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (build / 'build.json').write_text(json.dumps(manifest['w2_factor_build'], indent=2) + '\n')
    print(installed, flush=True)


if __name__ == '__main__':
    main()
