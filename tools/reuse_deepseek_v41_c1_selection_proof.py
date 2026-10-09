# SPDX-License-Identifier: Apache-2.0
"""Reuse a passed C1 selector oracle only when its implementation is unchanged."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct


def sections(data):
    if data[:4] != b'\x7fELF' or data[5] != 1 or data[4] not in (1, 2):
        raise ValueError('Expected little-endian ELF32/ELF64')
    wide = data[4] == 2
    offset = struct.unpack_from('<Q' if wide else '<I', data, 40 if wide else 32)[0]
    size, count, names = struct.unpack_from('<HHH', data, 58 if wide else 46)
    form = '<IIQQQQIIQQ' if wide else '<IIIIIIIIII'
    rows = [struct.unpack_from(form, data, offset + i * size) for i in range(count)]
    strings = data[rows[names][4]:rows[names][4] + rows[names][5]]
    return rows, [strings[row[0]:].split(b'\0', 1)[0].decode() for row in rows]


def kernel_text(path, name):
    data = path.read_bytes()
    rows, _ = sections(data)
    symbols = {}
    for row in rows:
        if row[1] not in (2, 11):
            continue
        strings = rows[row[6]]
        strings = data[strings[4]:strings[4] + strings[5]]
        for offset in range(row[4], row[4] + row[5], row[9]):
            item = struct.unpack_from('<IBBHQQ', data, offset)
            symbols[strings[item[0]:].split(b'\0', 1)[0].decode()] = item
    prefix = '_binary___' + name + '_o_'
    begin, end = symbols[prefix + 'start'], symbols[prefix + 'end']
    section = rows[begin[3]]
    blob = data[section[4] + begin[4] - section[3]:section[4] + end[4] - section[3]]
    inner, names = sections(blob)
    text = inner[names.index('.text')]
    return hashlib.sha256(blob[text[4]:text[4] + text[5]]).hexdigest()


def helper_hash(path):
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'threshold_decode_selection')
    return hashlib.sha256(ast.dump(function).encode()).hexdigest()


def validate_current(proof, native):
    reference = Path(proof['reference'])
    if hashlib.sha256(reference.read_bytes()).hexdigest() != proof['reference_sha256']:
        raise ValueError('Archived C1 oracle record changed')
    if proof['kind'] != 'c1_selection_reuse' or len(proof['oracle']) != 4 or not all(
            len(cases) == 3 and all(case['passed'] and case['scores_have_bf16_boundary']
                                  and case['selected_ids_exact'] and case['candidate_blocks_exact']
                                  for case in cases) for cases in proof['oracle']):
        raise ValueError('The archived actual-input C1 oracle did not pass all ranks')
    helper = Path(__file__).resolve().parents[1] / 'vllm_gaudi/ops/deepseek_v41_decode_selection.py'
    if helper_hash(helper) != proof['helper_ast_sha256']:
        raise ValueError('C1 row mapping changed after the reused oracle')
    for name, expected in proof['kernel_text_sha256'].items():
        if kernel_text(native / 'libdeepseek_v4_gaudi2_kernels.so', name) != expected:
            raise ValueError(f'C1 selection kernel changed: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads(args.reference.read_text())
    run = Path(previous['source_run'])
    environment = json.loads((run / 'process.json').read_text())['environment']
    old_native = Path(environment['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR'])
    old_library = old_native / 'libdeepseek_v4_gaudi2_kernels.so'
    if hashlib.sha256(old_library.read_bytes()).hexdigest() != previous['native_binaries'][old_library.name]:
        raise ValueError('Archived oracle native library changed')
    old_helper = run / 'source/vllm_gaudi/ops/deepseek_v41_decode_selection.py'
    fixtures = [[hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted((args.fixtures / f'rank{rank}').glob('c6-*.pt'))] for rank in range(4)]
    if not all(3 <= len(cases) <= 5 for cases in fixtures):
        raise ValueError('Need three to five current actual request inputs on every rank')
    proof = dict(kind='c1_selection_reuse', candidate='runtime_selection', eligible_groups=[0, 5, 6],
                 reference=str(args.reference.resolve()), reference_sha256=hashlib.sha256(
                     args.reference.read_bytes()).hexdigest(), oracle=previous['oracle'],
                 reference_fixtures_by_rank=previous['fixtures_by_rank'], fixtures_by_rank=fixtures,
                 helper_ast_sha256=helper_hash(old_helper), kernel_text_sha256={
                     name: kernel_text(old_library, name) for name in
                     ('deepseek_v41_index_threshold_gaudi2', 'deepseek_v41_index_emit_gaudi2')},
                 scope='Unchanged C1 selector on causal BF16 score planes and ratio-one per-query candidate blocks; '
                       'current native chain checks remain required; whole-Target acceptance remains pending')
    validate_current(proof, args.native)
    args.output.write_text(json.dumps(proof, indent=2) + '\n')


if __name__ == '__main__':
    main()
