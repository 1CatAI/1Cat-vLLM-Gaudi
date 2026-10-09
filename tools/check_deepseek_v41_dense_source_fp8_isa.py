# SPDX-License-Identifier: Apache-2.0
"""Screen weight-only streaming before building/registering a native operator.

Use actual checkpoint matrices and the maintained loader's source decoder.
ISA issue floors exclude memory stalls, synchronization, tile fill/drain and
MME overlap. They reject an impossible budget; they never qualify a gain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from vllm_gaudi.ops.deepseek_v41_weights import read_header
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import decode_e4m3fn

    workspace = Path(__file__).resolve().parents[1]
    source = workspace/'csrc/deepseek_v41_unique/kernels/deepseek_v41_dense_source_fp8_decode_gaudi2.c'
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.prepared/'pp0-tp0.safetensors'
    header = read_header(checkpoint)
    rows = []
    with checkpoint.open('rb') as stream:
        for projection in ('wq_a', 'wkv', 'wq_b', 'wo_b'):
            name = f'layers.20.attn.{projection}.weight'
            spec = header[name]
            scale_spec = header[name.removesuffix('weight')+'scale']
            stream.seek(spec.offset)
            raw = stream.read(spec.nbytes)
            if len(raw) != spec.nbytes or spec.dtype != 'F8_E4M3':
                raise ValueError('A complete actual checkpoint FP8 matrix is required')
            source_codes = np.frombuffer(raw, np.uint8).reshape(spec.shape)
            stream.seek(scale_spec.offset)
            scale_raw = stream.read(scale_spec.nbytes)
            scales = np.frombuffer(scale_raw, np.uint8).reshape(scale_spec.shape)
            if scales.min() < 16 or scales.max() > 239:
                raise ValueError('Prototype only covers normal BF16 scaled source weights')
            expanded = scales.astype(np.int32).repeat(32, 0).repeat(32, 1)
            magnitude = (source_codes & 127).astype(np.int32)
            mantissa = magnitude & 7
            tiny = (mantissa << 5) + 0x3b80
            tiny = np.where(mantissa < 4, (mantissa << 6) + 0x3b00, tiny)
            tiny = np.where(mantissa < 2, 0x3b00, tiny)
            tiny += (expanded-127) << 7
            bits = np.where(magnitude < 8, tiny, (magnitude << 4)+((expanded-7) << 7))
            bits = np.where(magnitude == 0, 0, bits) | ((source_codes.astype(np.int32)&128) << 8)
            bits = bits.astype(np.uint16)
            reference = decode_e4m3fn(source_codes) * np.exp2(expanded.astype(np.float32)-127)
            reference_bits = reference.view(np.uint32)
            if np.any(reference_bits & 65535):
                raise ValueError('Prototype assumption of exact BF16 source weights is violated')
            exact = np.array_equal(bits, (reference_bits >> 16).astype(np.uint16))
            if not exact:
                raise AssertionError('Integer codec differs from production loader BF16 bytes')
            integers = magnitude.astype(np.float32).view(np.uint32) >> 16
            native_tiny = integers.astype(np.int32) + ((expanded-136) << 7)
            native_bits = np.where(magnitude < 8, native_tiny,
                                   (magnitude << 4)+((expanded-7) << 7))
            signed_codes = source_codes.view(np.int8).astype(np.int16).view(np.uint16)
            native_bits = np.where(magnitude == 0, 0, native_bits) | (signed_codes & 0x8000)
            native_exact = np.array_equal(native_bits.astype(np.uint16), bits)
            if not native_exact:
                raise AssertionError('Integer-to-BF16 or signed-load arithmetic changed source bytes')
            unusual = ((magnitude > 0)&(magnitude < 8)) | (magnitude >= 120) | (source_codes == 128)
            rows.append(dict(matrix=name, shape=list(spec.shape), source_sha256=hashlib.sha256(raw).hexdigest(),
                scale_sha256=hashlib.sha256(scale_raw).hexdigest(), bf16_bytes_exact=exact,
                integer_tiny_signed_bytes_exact=native_exact,
                special_elements_fraction=float(unusual.mean()),
                special128_rows_fraction=float(unusual.reshape(spec.shape[0], -1, 128).any(-1).mean())))
    variants = []
    choices = (('linear', False, False, False), ('unpack', True, False, False),
               ('integer-tiny', True, True, False), ('integer-tiny-signed', True, True, True))
    for stem, unpack, native_tiny, signed in choices:
        object_path = args.output/f'{stem}.o'
        subprocess.run(['tpc-clang', '-Wall', '-Werror', '-O2', '-march=gaudi2',
                        '-I/usr/lib/habanatools/include', f'-DDSV41_DENSE_SOURCE_UNPACK={int(unpack)}',
                        f'-DDSV41_DENSE_SOURCE_INTEGER_TINY={int(native_tiny)}',
                        f'-DDSV41_DENSE_SOURCE_SIGNED={int(signed)}',
                        str(source), '-c', '-o', str(object_path)], check=True)
        disassembly = subprocess.check_output(['/usr/bin/tpc-llvm-objdump', '--triple=tpc', '-d',
                                               '--no-show-raw-insn', str(object_path)], text=True)
        (args.output/f'{stem}.isa').write_text(disassembly)
        packets = [line for line in disassembly.splitlines() if re.match(r'^\s*[0-9a-f]+:', line)]
        inner = re.search(r'loop 0, 32, 1, <, (\.LBB\w+)\n', disassembly)
        if inner is None:
            raise ValueError('An independently identifiable fixed32-row loop is required')
        tail = disassembly[inner.end():]
        end = re.search(r'^0*[0-9a-f]+ '+re.escape(inner[1])+':', tail, re.MULTILINE)
        if end is None:
            raise ValueError('ISA loop endpoint missing')
        body = tail[:end.start()]
        vector_ops = []
        for line in body.splitlines():
            fields = line.split(';')
            if re.match(r'^\s*[0-9a-f]+:', line) and len(fields) == 4 and fields[2].strip() != 'nop':
                vector_ops.append(fields[2].strip())
        variants.append(dict(name=stem, object_sha256=hashlib.sha256(object_path.read_bytes()).hexdigest(),
            printed_instruction_lines=len(packets), hot_vector_ops=vector_ops,
            vector_instructions_per128_values=len(vector_ops),
            local_ops=[line for line in packets if re.search(r'\b(?:ld_l|st_l)\b', line)],
            device_dtype_binding_qualified=False))
    report = dict(status='CPU_exact_ISA_screen_only', matrices=rows,
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        variants=variants,
        native_operator_registered=False, device_acquired=False, gain_credit_ms=0,
        constraint='No performance qualification: inspect hot issue count and full producer/MME contract before cards')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], bf16_bytes_exact=all(r['bf16_bytes_exact'] for r in rows),
                         variants=[dict(name=v['name'], vector_ops=v['vector_instructions_per128_values'])
                                   for v in variants], device_acquired=False, gain_credit_ms=0)))


if __name__ == '__main__':
    main()
