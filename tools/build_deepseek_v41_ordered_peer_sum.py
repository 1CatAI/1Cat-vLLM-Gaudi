# SPDX-License-Identifier: Apache-2.0
"""Add the unchanged C1 peer consumer to an immutable DSpark native bundle."""
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
    parser.add_argument('--peer-post-collapse', action='store_true')
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    build, parent = args.build_root.resolve(), args.parent.resolve()
    build.mkdir(parents=True, exist_ok=False)
    # Compile and hash the same immutable inputs even while Python work
    # continues in the checkout during the slower extension build.
    snapshot = build / 'source'
    for directory in ('deepseek_v4', 'deepseek_v41_unique'):
        shutil.copytree(workspace / 'csrc' / directory, snapshot / 'csrc' / directory)
    name = ('deepseek_v41_mhc_post_collapse_gaudi2' if args.peer_post_collapse
            else 'deepseek_v41_ordered_peer_sum_gaudi2')
    host = snapshot / 'csrc/deepseek_v4/host'
    source = snapshot / f'csrc/deepseek_v4/kernels/{name}.c'
    subprocess.run(['tpc-clang', '-Wall', '-Werror', '-O2', '-march=gaudi2',
                    '-I/usr/lib/habanatools/include', str(source), '-c', '-o', str(build / f'{name}.o')], check=True)
    subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                    f'./{name}.o', f'{name}_embedded.o'], cwd=build, check=True)
    with (build / 'kernel.isa').open('w') as output:
        subprocess.run(['tpc-llvm-objdump', '--triple=tpc', '--mcpu=gaudi2', '-d', '--no-show-raw-insn',
                        str(build / f'{name}.o')], stdout=output, check=True)
    kind = 'peer_post_collapse' if args.peer_post_collapse else 'ordered_peer_sum'
    wrapper = snapshot / f'csrc/deepseek_v41_unique/{kind}_host.cpp'
    gc = build / 'libdeepseek_v41_unique_kernels.so'
    subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                    '-I/usr/include/habanalabs', '-I/usr/lib/habanatools/include', '-I' + str(host),
                    str(wrapper), str(host / f'{name}.cpp'),
                    str(build / f'{name}_embedded.o'), '-ldl', '-Wl,-z,noexecstack',
                    '-Wl,--no-undefined', '-o', str(gc)], check=True)
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith('DSV41_UNIQUE_ONLY_'):
            del environment[key]
    root = workspace.parent
    environment.update(MAX_JOBS=str(len(os.sched_getaffinity(0))),
        GAUDI_PYTORCH_BRIDGE_ROOT=str(root / 'gaudi-pytorch-bridge-tp2-fused-native'),
        GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=str(root / 'builds/pytorch_bridge_tp2_fused_native_upstream'),
        GAUDI_PYTORCH_BRIDGE_DEPS_ROOT=str(root / 'builds/pytorch_modules_multi_build/torch'
                                         / 'py3.12/pt2.11.0/Release/_deps'),
        DSV41_UNIQUE_ONLY_ORDERED_PEER_SUM='1')
    if args.peer_post_collapse:
        environment.pop('DSV41_UNIQUE_ONLY_ORDERED_PEER_SUM')
        environment['DSV41_UNIQUE_ONLY_PEER_POST_COLLAPSE'] = '1'
        original = snapshot / 'csrc/deepseek_v4/pytorch/hpu_dsv41_mhc_post_collapse_pt2.cpp'
        alias = original.read_text().replace('custom_deepseek_v41_mhc_post_collapse_gaudi2',
                                             'custom_deepseek_v41_peer_mhc_post_collapse_gaudi2')
        (snapshot / 'csrc/deepseek_v41_unique/pytorch/peer_post_collapse.cpp').write_text(alias)
    with (build / 'addon-build.log').open('w') as output:
        subprocess.run([sys.executable, 'setup.py', 'build_ext', '--build-lib', str(build / 'addon'),
                        '--build-temp', str(build / 'addon-temp')],
                       cwd=snapshot / 'csrc/deepseek_v41_unique/pytorch', env=environment,
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    installed = build / 'installed'
    installed.mkdir()
    manifest = json.loads((parent / 'deepseek_v41_unique_build.json').read_text())
    for path in parent.iterdir():
        if not path.is_file() or path.name == gc.name:
            continue
        if path.suffix == '.json':
            shutil.copy2(path, installed / path.name)
        else:
            (installed / path.name).symlink_to(path.resolve())
    for basename, expected in manifest['binaries'].items():
        if digest(parent / basename) != expected:
            raise ValueError(f'Inherited binary changed: {basename}')
    shutil.copy2(gc, installed / gc.name)
    manifest['binaries'][gc.name] = digest(gc)
    for addon in (build / 'addon').glob('*.so'):
        shutil.copy2(addon, installed / addon.name)
        manifest['binaries'][addon.name] = digest(addon)
        manifest.setdefault('extra_registrations', []).append(addon.name)
    sources = [source, wrapper, host / f'{name}.cpp', host / f'{name}.hpp',
               snapshot / ('csrc/deepseek_v4/pytorch/hpu_dsv41_mhc_post_collapse_pt2.cpp'
                            if args.peer_post_collapse else
                            'csrc/deepseek_v4/pytorch/hpu_dsv41_ordered_peer_sum_pt2.cpp')]
    proof = dict(parent_gc_path=str(parent / gc.name), parent_gc_sha256=digest(parent / gc.name),
                 sources={str(path): digest(path) for path in sources}, default_enabled=False,
                 unchanged_c1_arithmetic=True, communicator_sources_changed=False,
                 required_parent_env=('VLLM_HPU_DSV41_PEER_COLLAPSE_PARENT_KERNEL' if args.peer_post_collapse
                                      else 'VLLM_HPU_DSV41_ORDERED_PEER_PARENT_KERNEL'))
    if args.peer_post_collapse:
        alias_path = snapshot / 'csrc/deepseek_v41_unique/pytorch/peer_post_collapse.cpp'
        proof['generated_registration'] = dict(path=str(alias_path), sha256=digest(alias_path),
                                               replacement='C1 schema/GUID alias only')
    manifest[kind + '_build'] = proof
    (installed / 'deepseek_v41_unique_build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (build / 'build.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(installed, flush=True)


if __name__ == '__main__':
    main()
