# SPDX-License-Identifier: Apache-2.0
"""Build additive C2-C6 FP16 PV; inherited kernels and default dispatch stay unchanged."""
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
    parser.add_argument('--direct', action='store_true', help='Decode/softmax directly into FP16; no conversion nodes')
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root, output = workspace.parent, args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.runtime_profile.read_text())
    source = output / 'source'
    files = ('csrc/deepseek_v41_unique/pytorch/' + ('mla_fp16_direct.cpp' if args.direct else 'mla_fp16_pv.cpp'),
             'csrc/deepseek_v41_unique/pytorch/setup.py',
             'csrc/deepseek_v4/pytorch/hpu_dsv41_shared_main_mla_pt2.cpp',
             'csrc/deepseek_v4/pytorch/dsv41_c6_flat_qk_member.h')
    if args.direct:
        files += ('csrc/deepseek_v41_unique/mla_fp16_host.cpp',
                  'csrc/deepseek_v4/include/deepseek_v41_selected_kv_vector.h',
                  'csrc/deepseek_v4/kernels/deepseek_v41_main_reuse_gather_gaudi2.c',
                  'csrc/deepseek_v4/kernels/deepseek_v41_selected_mla_softmax_gaudi2.c')
        files += tuple(f'csrc/deepseek_v41_unique/kernels/deepseek_v41_mla_fp16_{name}_gaudi2.c'
                       for name in ('publish_gather', 'reuse_gather', 'softmax'))
    for name in files:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace / name, source / name)
    database = None
    if args.direct:
        objects = []
        for name in files[-3:]:
            stem = Path(name).stem
            subprocess.run(['tpc-clang', '-Wall', '-Werror', '-O2', '-march=gaudi2',
                            '-I' + str(source / 'csrc/deepseek_v4/include'),
                            str(source / name), '-c', '-o', str(output / f'{stem}.o')], check=True)
            subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                            f'./{stem}.o', f'{stem}_embedded.o'], cwd=output, check=True)
            objects.append(str(output / f'{stem}_embedded.o'))
            with (output / f'{stem}.asm').open('w') as asm:
                subprocess.run(['tpc-llvm-objdump', '--arch-name=tpc', '--mcpu=gaudi2', '-d',
                                str(output / f'{stem}.o')], stdout=asm, check=True)
        database = output / 'libdeepseek_v41_unique_kernels.so'
        subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                        '-I/usr/include/habanalabs', str(source / 'csrc/deepseek_v41_unique/mla_fp16_host.cpp'),
                        *objects, '-ldl', '-Wl,-z,noexecstack', '-Wl,--no-undefined', '-o', str(database)], check=True)
    environment = dict(os.environ, **profile['environment'])
    for key in tuple(environment):
        if key.startswith('DSV41_UNIQUE_ONLY_'):
            del environment[key]
    environment.update(MAX_JOBS=str(len(os.sched_getaffinity(0))), DSV41_UNIQUE_ONLY_MLA_FP16_PV='1',
        GAUDI_PYTORCH_BRIDGE_ROOT=str(root / 'gaudi-pytorch-bridge-tp2-fused-native'),
        GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=str(root / 'builds/pytorch_bridge_tp2_fused_native_upstream'),
        GAUDI_PYTORCH_BRIDGE_DEPS_ROOT=str(root / 'builds/pytorch_modules_multi_build/torch'
                                         / 'py3.12/pt2.11.0/Release/_deps'))
    if args.direct:
        environment['DSV41_UNIQUE_ONLY_MLA_FP16_DIRECT'] = '1'
        environment['DSV41_UNIQUE_ONLY_MLA_FP16_PV'] = '0'
    with (output / 'build.log').open('w') as log:
        subprocess.run([sys.executable, 'setup.py', 'build_ext', '--build-lib', str(output / 'addon'),
                        '--build-temp', str(output / 'temp')], cwd=source / 'csrc/deepseek_v41_unique/pytorch',
                       env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    installed = output / 'installed'
    installed.mkdir()
    old_root = Path(profile['environment']['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR'])
    for path in old_root.iterdir():
        if not path.is_file() or (database is not None and path.name == database.name):
            continue
        if path.suffix == '.json':
            shutil.copy2(path, installed / path.name)
        else:
            (installed / path.name).symlink_to(path.resolve())
    if database is not None:
        shutil.copy2(database, installed / database.name)
    manifest_path = installed / 'deepseek_v41_unique_build.json'
    manifest = json.loads(manifest_path.read_text())
    pattern = 'hpu_dsv41_mla_fp16_direct_pt2*.so' if args.direct else 'hpu_dsv41_mla_fp16_pv_pt2*.so'
    addons = list((output / 'addon').glob(pattern))
    if len(addons) != 1:
        raise RuntimeError('Expected one FP16 PV registration')
    addon = addons[0]
    shutil.copy2(addon, installed / addon.name)
    manifest['binaries'][addon.name] = digest(addon)
    manifest.setdefault('extra_registrations', []).append(addon.name)
    if database is not None:
        manifest['binaries'][database.name] = digest(database)
    proof = dict(default_enabled=False, source={name: digest(source / name) for name in files},
                 parent_runtime_profile=str(args.runtime_profile.resolve()),
                 parent_profile_sha256=digest(args.runtime_profile),
                 scope='Same QK/softmax/public cache; FP16 PV inputs and FP32 accumulation; casts timed')
    manifest['mla_fp16_pv'] = proof
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    for key, value in tuple(profile['environment'].items()):
        if isinstance(value, str) and value.startswith(str(old_root) + '/'):
            name = Path(value).name
            if (installed / name).is_file():
                profile['environment'][key] = str(installed / name)
    profile['environment']['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR'] = str(installed)
    if database is not None:
        profile['environment']['VLLM_HPU_DSV41_MLA_FP16_PARENT_KERNEL'] = (
            json.loads(args.runtime_profile.read_text())['environment']['VLLM_HPU_DSV41_UNIQUE_KERNEL'])
        for key in ('GC_KERNEL_PATH', 'VLLM_HPU_DSV41_UNIQUE_KERNEL'):
            profile['environment'][key] = str(installed / database.name)
    profile.setdefault('additional_libraries', []).append(dict(path=str(installed / addon.name), sha256=digest(addon)))
    profile.setdefault('configuration_files', []).append(dict(path=str(manifest_path), sha256=digest(manifest_path)))
    profile['candidate'] = 'Disabled additive C2-C6 FP16 PV capability, numeric and native-chain gates pending'
    profile['diagnostic_only'] = True
    (output / 'runtime-profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    (output / 'build.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(output / 'runtime-profile.json', flush=True)


if __name__ == '__main__':
    main()
