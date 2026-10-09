# SPDX-License-Identifier: Apache-2.0
"""Immutable, isolated SDK receiver and transport-generation capability build."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--command-reference', type=Path, required=True)
    args = parser.parse_args()
    root, output = Path(__file__).resolve().parents[1], args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    files = ('csrc/deepseek_v41_unique/bounded_receive_host.cpp',
             'csrc/deepseek_v41_unique/receive_event_contract.cpp',
             'csrc/deepseek_v41_unique/kernels/deepseek_v41_bounded_peer_receive_gaudi2.c',
             'csrc/deepseek_v41_unique/kernels/deepseek_v41_future_epoch_gaudi2.c',
             'tools/check_deepseek_v41_monolithic_receive.cpp',
             'tools/check_deepseek_v41_bounded_receive_glue.cpp',
             'tools/run_deepseek_v41_receive_capability.py',
             'tools/communication/dsv41_future_receive_publisher.h')
    for name in files:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, source / name)
    objects = []
    with (output / 'build.log').open('w') as log:
        def run(command, **kwargs):
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, **kwargs)
        for name in files[2:4]:
            stem = Path(name).stem
            run(['tpc-clang', '-Wall', '-Werror', '-O2', '-march=gaudi2',
                 str(source / name), '-c', '-o', str(output / f'{stem}.o')])
            run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                 f'./{stem}.o', f'{stem}_embedded.o'], cwd=output)
            objects.append(str(output / f'{stem}_embedded.o'))
            with (output / f'{stem}.asm').open('w') as asm:
                subprocess.run(['tpc-llvm-objdump', '--arch-name=tpc', '--mcpu=gaudi2', '-d',
                                str(output / f'{stem}.o')], stdout=asm, check=True)
        run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
             '-I/usr/include/habanalabs', str(source / files[0]), *objects, '-ldl',
             '-Wl,-z,noexecstack', '-Wl,--no-undefined', '-o', str(output / 'libbounded_receive.so')])
        for record, target, filename in (('command.json', 'libreceive_event_contract.so', files[1]),
                                         ('capability-command.json', 'check_receive', files[4])):
            command = json.loads((args.command_reference / record).read_text())
            command[command.index('-o')+1] = str(output / target)
            command = [str(source / filename) if value.endswith('/'+filename) else value for value in command]
            (output / record).write_text(json.dumps(command, indent=2)+'\n')
            run(command)
        run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-I/usr/include/habanalabs',
             str(source / files[5]), '-ldl', '-o', str(output / 'check_glue')])
        run([str(output / 'check_glue'), str(output / 'libbounded_receive.so')])
    (output / 'build.json').write_text(json.dumps(dict(
        sources={name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files},
        serving_selected=False, performance_qualified=False), indent=2)+'\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
