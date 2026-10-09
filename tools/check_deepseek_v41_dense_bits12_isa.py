# SPDX-License-Identifier: Apache-2.0
"""Exact actual BF16-weight packing and static full-consumer budget; no cards."""
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
    source = workspace/'csrc/deepseek_v41_unique/kernels/deepseek_v41_dense_bits12_gaudi2.c'
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.prepared/'pp0-tp0.safetensors'
    catalog = read_header(checkpoint)
    rows = []
    with checkpoint.open('rb') as stream:
        for projection in ('wq_a', 'wkv', 'wq_b', 'wo_b'):
            name = f'layers.20.attn.{projection}.weight'
            weight, scale = catalog[name], catalog[name.removesuffix('weight')+'scale']
            stream.seek(weight.offset)
            raw = stream.read(weight.nbytes)
            stream.seek(scale.offset)
            raw_scale = stream.read(scale.nbytes)
            codes = np.frombuffer(raw, np.uint8).reshape(weight.shape)
            scales = np.frombuffer(raw_scale, np.uint8).reshape(scale.shape)
            if weight.dtype != 'F8_E4M3' or scale.dtype != 'U8' or len(raw) != weight.nbytes:
                raise ValueError('Complete actual checkpoint weights/scales are required')
            expanded = scales.astype(np.float32).repeat(32, 0).repeat(32, 1)
            values = decode_e4m3fn(codes) * np.exp2(expanded-127)
            f32_bits = values.view(np.uint32)
            if np.any(f32_bits & 65535) or np.any((codes & 127) == 127):
                raise ValueError('Source values must be finite and exactly representable as BF16')
            bits = (f32_bits >> 16).astype(np.uint16)
            if np.any(bits & 15) or bits.shape[1] % 256:
                raise ValueError('Weight-only packing requires zero low4 bits and complete K256 tiles')
            high = (bits >> 8).astype(np.uint8)
            low = ((bits >> 4) & 15).astype(np.uint8)
            packed = low[:, ::2] | (low[:, 1::2] << 4)
            recovered = np.empty_like(low)
            recovered[:, ::2], recovered[:, 1::2] = packed & 15, packed >> 4
            restored = (high.astype(np.uint16) << 8) | (recovered.astype(np.uint16) << 4)
            exact = np.array_equal(bits, restored)
            if not exact:
                raise AssertionError('Lossless BF16 bit packing changed source bytes')
            rows.append(dict(name=name, shape=list(bits.shape), bf16_bytes_exact=exact,
                             source_sha256=hashlib.sha256(raw).hexdigest(),
                             scale_sha256=hashlib.sha256(raw_scale).hexdigest(),
                             original_bytes=int(bits.nbytes), packed_bytes=int(high.nbytes+packed.nbytes)))
    object_path = args.output/'dense_bits12.o'
    subprocess.run(['tpc-clang', '-Wall', '-Werror', '-O2', '-march=gaudi2',
                    '-I/usr/lib/habanatools/include', str(source), '-c', '-o', str(object_path)], check=True)
    disassembly = subprocess.check_output(['/usr/bin/tpc-llvm-objdump', '--triple=tpc', '-d',
                                           '--no-show-raw-insn', str(object_path)], text=True)
    (args.output/'dense_bits12.isa').write_text(disassembly)
    inner = re.search(r'loop 0, 32, 1, <, (\.LBB\w+)\n', disassembly)
    if inner is None:
        raise ValueError('Fixed32-row loop required for independent hot-loop issue accounting')
    tail = disassembly[inner.end():]
    end = re.search(r'^0*[0-9a-f]+ '+re.escape(inner[1])+':', tail, re.MULTILINE)
    if end is None:
        raise ValueError('ISA loop endpoint missing')
    operations = []
    for line in tail[:end.start()].splitlines():
        fields = line.split(';')
        if re.match(r'^\s*[0-9a-f]+:', line) and len(fields) == 4 and fields[2].strip() != 'nop':
            operations.append(fields[2].strip())
    total_values = 0
    for layer in range(40):
        for projection in ('wq_a', 'wkv', 'wq_b', 'wo_b'):
            shape = catalog[f'layers.{layer}.attn.{projection}.weight'].shape
            total_values += int(np.prod(shape))
    # Optimistic issue floor only. Real memory, scheduling and MME overlap
    # belong to the native producer/consumer gate and receive no credit here.
    floor_ms = total_values/256*len(operations)/(24*1.8e9)*1000
    report = dict(status='CPU_ISA_ONLY', matrices=rows, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  vector_instructions_per256_values=len(operations), hot_vector_ops=operations,
                  full40_source_values=total_values, optimistic_tpc_issue_floor_ms=floor_ms,
                  source_bytes_per_weight=1.5, bf16_activation_unchanged=True,
                  device_lane_permutation_qualified=False, native_operator_registered=False,
                  device_acquired=False, native_producer_consumer_qualified=False, gain_credit_ms=0)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'kernel-source.c').write_bytes(source.read_bytes())
    (args.output/'checker-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status=report['status'], exact_matrices=len(rows),
                          vector_ops_per256=len(operations), optimistic_full40_issue_floor_ms=floor_ms,
                          device_lane_permutation_qualified=False, gain_credit_ms=0)))


if __name__ == '__main__':
    main()
