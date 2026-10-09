# SPDX-License-Identifier: Apache-2.0
"""Untimed native lane qualification and four actual BF16 weight-byte checks."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators
    from vllm_gaudi.ops.deepseek_v41_weights import read_header
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import decode_e4m3fn

    load_native_operators()
    torch.hpu.set_device(0)

    def decode(high, low):
        return torch.ops.custom_op.custom_deepseek_v41_dense_bits12_bf16_gaudi2(high, low)

    # Compilation is used exclusively for functional checking. Performance
    # admission must include the original native producer and MME consumer.
    compiled = torch.compile(decode, backend='hpu_backend', fullgraph=True, dynamic=False)

    def actual(high, low):
        h = torch.from_numpy(high.view(np.int8)).to('hpu')
        l = torch.from_numpy(low.view(np.int8)).to('hpu')
        return compiled(h, l).cpu().view(torch.int16).numpy().view(np.uint16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='UNTIMED_BINDING', timed=False, gain_credit_ms=0,
                  native_producer_consumer_qualified=False, device_lane_permutation_qualified=False,
                  actual_matrices=[])

    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')

    with torch.inference_mode():
        # Two independent base16 digits identify every one of 256 lanes.
        # This discovers the hardware mapping, rather than assuming a CUDA
        # or SW_LINEAR lane layout. No calibration of model values is used.
        index = np.arange(256, dtype=np.uint16)
        digits = np.zeros((32, 256), np.uint8)
        digits[0], digits[1] = index & 15, index >> 4
        probe = actual(np.zeros_like(digits), digits[:, ::2] | (digits[:, 1::2] << 4))
        lane = ((probe[0] >> 4) & 15) | (((probe[1] >> 4) & 15) << 4)
        bijective = np.array_equal(np.sort(lane), index)
        clean = bool(np.all((probe & np.uint16(0xff0f)) == 0))
        report.update(mapping_is_bijective=bool(bijective), probe_has_only_nibble_bits=clean,
                      lane_map=lane.tolist(), probe_first32=probe[:2, :32].tolist())
        save()
        if not bijective or not clean:
            raise AssertionError('LSU4-to8/default conversion lacks the required lossless lane contract')
        report['device_lane_permutation_qualified'] = True

        checkpoint = args.prepared/'pp0-tp0.safetensors'
        catalog = read_header(checkpoint)
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
                values = decode_e4m3fn(codes) * np.exp2(
                    scales.astype(np.float32).repeat(32, 0).repeat(32, 1)-127)
                f32_bits = values.view(np.uint32)
                if np.any(f32_bits & 65535):
                    raise ValueError('Actual source is not exactly BF16')
                bits = (f32_bits >> 16).astype(np.uint16)
                if np.any(bits & 15):
                    raise ValueError('Actual source has nonzero low4 bits')
                high = (bits >> 8).astype(np.uint8)
                desired = ((bits >> 4) & 15).astype(np.uint8).reshape(-1, 256)
                shuffled = np.empty_like(desired)
                shuffled[:, lane] = desired
                low = (shuffled[:, ::2] | (shuffled[:, 1::2] << 4)).reshape(bits.shape[0], -1)
                restored = actual(high, low)
                exact = np.array_equal(restored, bits)
                report['actual_matrices'].append(dict(name=name, shape=list(bits.shape),
                    bf16_bytes_exact=bool(exact), mismatched_values=int(np.count_nonzero(restored != bits)),
                    mantissa_nibbles_exact=bool(np.all((restored & 255) == (bits & 255))),
                    source_sha256=hashlib.sha256(raw).hexdigest(),
                    negative_zero_count=int(np.count_nonzero(bits == 0x8000))))
                save()
                if not exact:
                    report['mismatch_first16'] = dict(actual=restored.reshape(-1)[:16].tolist(),
                                                    expected=bits.reshape(-1)[:16].tolist())
                    save()
                    raise AssertionError('Native reconstructed BF16 weights differ from actual source bytes')
        torch.hpu.synchronize()
        report['status'] = 'WEIGHT_BYTES_FUNCTIONAL_ONLY'
        save()
    print(json.dumps(dict(status=report['status'], matrices=len(report['actual_matrices']),
                          bf16_bytes_exact=all(row['bf16_bytes_exact'] for row in report['actual_matrices']),
                          device_lane_permutation_qualified=True, gain_credit_ms=0)), flush=True)


if __name__ == '__main__':
    main()
