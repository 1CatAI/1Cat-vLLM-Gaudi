# SPDX-License-Identifier: Apache-2.0
"""Build an isolated receiver; reuse unchanged owned epoch/event artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    output, parent = args.output.resolve(), args.parent.resolve()
    inherited = ('deepseek_v41_future_epoch_gaudi2_embedded.o', 'libreceive_event_contract.so')
    for name in inherited:
        if not (parent / name).is_file():
            raise FileNotFoundError(parent / name)
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    kernel = 'deepseek_v41_bounded_peer_receive_gaudi2'
    kernel_path = workspace / f'csrc/deepseek_v41_unique/kernels/{kernel}.c'
    host_path = workspace / 'csrc/deepseek_v41_unique/bounded_receive_host.cpp'
    for path in (kernel_path, host_path):
        shutil.copy2(path, source / path.name)
    for name in inherited:
        (output / name).symlink_to(parent / name)
    commands = [
        ['tpc-clang', '-Wall', '-Werror', '-O2', '-march=gaudi2', '-DDSV41_RECEIVE_POLL_ONCE=1',
         '-I/usr/lib/habanatools/include', str(source / kernel_path.name), '-c', '-o', str(output / (kernel + '.o'))],
        ['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
         './' + kernel + '.o', kernel + '_embedded.o'],
        ['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
         '-I/usr/include/habanalabs', str(source / host_path.name), str(output / (kernel + '_embedded.o')),
         str(output / inherited[0]), '-ldl', '-Wl,-z,noexecstack', '-Wl,--no-undefined',
         '-o', str(output / 'libbounded_receive.so')],
    ]
    with (output / 'build.log').open('w') as log:
        for command in commands:
            subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (output / 'receiver.asm').open('w') as assembly:
        subprocess.run(['tpc-llvm-objdump', '--triple=tpc', '--mcpu=gaudi2', '-d', '--no-show-raw-insn',
                        str(output / (kernel + '.o'))], stdout=assembly, check=True)
    files = [source / path.name for path in (kernel_path, host_path)]
    files += [parent / name for name in inherited]
    files += [output / 'libbounded_receive.so']
    proof = dict(parent=str(parent), commands=commands,
                 artifacts={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
                 default_enabled=False, serving_selected=False, performance_qualified=False)
    (output / 'build.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
