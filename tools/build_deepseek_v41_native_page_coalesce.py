# SPDX-License-Identifier: Apache-2.0
"""Rebuild only two native runtime units in a private DSpark installation."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_profile(build, source, output):
    if source is None or output is None or output.exists():
        raise ValueError("Recording needs a parent profile and a new output path")
    proof = json.loads((build / 'RESULT.json').read_text())
    if digest(build / 'libSynapse.so') != proof['sha256']:
        raise RuntimeError("Private runtime differs from its build proof")
    profile = json.loads(source.read_text())
    previous = Path(profile['environment']['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR'])
    installed = build / 'installed'
    installed.mkdir(exist_ok=True)
    for path in previous.iterdir():
        if path.name == 'libSynapse.so':
            continue
        destination = installed / path.name
        if path.is_file() and not destination.exists():
            destination.symlink_to(path.resolve())
    binary = installed / 'libSynapse.so'
    if binary.is_symlink():
        if binary.resolve() != (previous / 'libSynapse.so').resolve():
            raise RuntimeError('Private installation contains an unrecognized runtime link')
        binary.unlink()
    if not binary.exists():
        os.link(build / 'libSynapse.so', binary)
    if digest(binary) != proof['sha256']:
        raise RuntimeError("Installed runtime changed")
    environment = profile['environment']
    database = Path(environment['GC_KERNEL_PATH']).name
    environment['GC_KERNEL_PATH'] = str(installed / database)
    environment['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR'] = str(installed)
    environment['LD_LIBRARY_PATH'] = str(installed) + ':' + environment['LD_LIBRARY_PATH']
    environment['VLLM_HPU_DSV41_DSPARK_NATIVE_PAGE_COALESCE'] = '0'
    for item in profile['additional_libraries']:
        if Path(item['path']).name == 'libSynapse.so':
            item.update(path=str(binary), sha256=proof['sha256'])
    profile['candidate'] = dict(name=proof.get('candidate_name', 'native compute-page publication coalescing'),
                                default_enabled=False, runtime_build=str(build), no_new_math=True)
    output.write_text(json.dumps(profile, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--runtime-source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--profile-input', type=Path)
    parser.add_argument('--profile-output', type=Path)
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()
    parent, runtime, build = args.parent.resolve(), args.runtime_source.resolve(), args.build.resolve()
    if args.profile_only:
        record_profile(build, args.profile_input, args.profile_output)
        return
    build.mkdir(parents=True, exist_ok=False)
    source = build / 'source'
    common = source / 'synapse/src/runtime/scal/common'
    (common / 'entities').mkdir(parents=True)
    specifications = [
        ('native_compute_program.hpp', parent / 'source/native_compute_program.hpp'),
        ('stream_compute_scal.cpp', parent / 'source/stream_compute_scal.cpp'),
        ('entities/scal_stream_base.cpp', runtime / 'synapse/src/runtime/scal/common/entities/scal_stream_base.cpp'),
        ('entities/scal_stream_base.hpp', runtime / 'synapse/src/runtime/scal/common/entities/scal_stream_base.hpp'),
    ]
    originals, patches = {}, []
    for relative, path in specifications:
        originals[str(path)] = digest(path)
        original = path.read_text()
        candidate = original
        if relative == 'native_compute_program.hpp':
            candidate = candidate.replace('#include <cstdint>\n', '#include <cstdint>\n#include <cstdlib>\n')
            anchor = '    bool externalInputCompletion = false;\n'
            assert candidate.count(anchor) == 1
            candidate = candidate.replace(anchor, anchor + '    bool coalesceSubmissions = false;\n')
            anchor = '        return plan;\n'
            assert candidate.count(anchor) == 1
            candidate = candidate.replace(anchor, '''        // Capture policy once; later arm changes cannot alter an
        // already-instantiated graph. Whole programs only: segmented plans
        // retain the established prefix/suffix publication policy.
        const char* mode = std::getenv("VLLM_HPU_DSV41_DSPARK_NATIVE_PAGE_COALESCE");
        const uint64_t padding = uint64_t(alignment) * (plan.pages.size() + 2);
        plan.coalesceSubmissions = mode && mode[0] == '1' && mode[1] == '\\0' &&
            padding <= availableBytes && plan.byteCount <= availableBytes - padding;
        return plan;
''')
        elif relative == 'entities/scal_stream_base.cpp':
            anchor = 'getStreamCyclicBuffer()->addCommand(page.bytes.size(), true, copy, scratch)'
            assert candidate.count(anchor) == 1
            candidate = candidate.replace(anchor, '''getStreamCyclicBuffer()->addCommand(
            page.bytes.size(), !program.coalesceSubmissions || &page == &program.pages.back(), copy, scratch)''')
        (common / relative).write_text(candidate)
        patches.extend(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
                                            fromfile=f'a/synapse/src/runtime/scal/common/{relative}',
                                            tofile=f'b/synapse/src/runtime/scal/common/{relative}'))
    (build / 'source.patch').write_text(''.join(patches))
    commands = json.loads((parent / 'commands.json').read_text())
    template = commands['compile']
    jobs = []
    for name, relative in [('stream_compute_scal.cpp', 'stream_compute_scal.cpp'),
                           ('scal_stream_base.cpp', 'entities/scal_stream_base.cpp')]:
        command = list(template)
        command[1:1] = ['-I' + str(source / 'synapse/src'),
                            '-I' + str(runtime / 'synapse/src/runtime/scal/common/entities')]
        command[command.index('-o') + 1] = str(build / (name + '.o'))
        command[command.index('-c') + 1] = str(common / relative)
        jobs.append(command)
    link = list(commands['link'])
    link[link.index('-o') + 1] = str(build / 'libSynapse.so')
    for index, item in enumerate(link):
        if item.endswith('/scal_stream_base.cpp.o'):
            link[index] = str(build / 'scal_stream_base.cpp.o')
        elif item.endswith('/stream_compute_scal.cpp.o'):
            link[index] = str(build / 'stream_compute_scal.cpp.o')
    assert sum(str(build) in item and item.endswith('.cpp.o') for item in link) == 2
    (build / 'commands.json').write_text(json.dumps({'compile': jobs, 'link': link}, indent=2) + '\n')
    with (build / 'build.log').open('w') as log:
        for command in (*jobs, link):
            log.write(shlex.join(command) + '\n')
            log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    for path, fingerprint in originals.items():
        if digest(Path(path)) != fingerprint:
            raise RuntimeError('Shared prerequisite changed during private build')
    proof = dict(parent_library=str(parent / 'libSynapse.so'), parent_sha256=digest(parent / 'libSynapse.so'),
                 library=str(build / 'libSynapse.so'), sha256=digest(build / 'libSynapse.so'),
                 source_prerequisites=originals, coalescing_default=False, hcl_and_bridge_unchanged=True,
                 rule='automatic CCB alignment/chunk submissions and per-page retirement remain unchanged',
                 status='built_unqualified')
    (build / 'RESULT.json').write_text(json.dumps(proof, indent=2) + '\n')
    if args.profile_input is not None:
        record_profile(build, args.profile_input, args.profile_output)
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
