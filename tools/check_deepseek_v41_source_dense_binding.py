# SPDX-License-Identifier: Apache-2.0
"""Untimed LSU/conversion capability on every finite E4M3FN code and source scales."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import decode_e4m3fn

    load_native_operators()
    torch.hpu.set_device(0)
    with torch.inference_mode():
        raw = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
        raw[:, [127, 255]] = 0  # Source loader rejects these two NaN codes.
        # HPU supports I16, whereas UInt16 cannot be uploaded. Positive UE8M0
        # codes retain identical 16-bit storage and the unsigned LSU semantics.
        scales = np.array([[16, 77, 126, 127, 128, 140, 210, 239]], dtype=np.int16)
        expanded = scales.repeat(32, 0).repeat(32, 1).astype(np.float32)
        reference = decode_e4m3fn(raw) * np.exp2(expanded-127)
        if np.any(reference.view(np.uint32) & 65535):
            raise AssertionError('Normal BF16 source range assumption failed')
        raw_device = torch.from_numpy(raw.view(np.int8)).to('hpu')
        scale_device = torch.from_numpy(scales).to('hpu')

        def decode(w, s):
            return torch.ops.custom_op.custom_deepseek_v41_source_weight_bf16_gaudi2(w, s)

        # Functional capability only. Production performance must use the
        # complete native producer/MME/consumer chain, never this wrapper.
        actual = torch.compile(decode, backend='hpu_backend', fullgraph=True, dynamic=False)(
            raw_device, scale_device).cpu().view(torch.int16).numpy().view(np.uint16)
        expected = (reference.view(np.uint32) >> 16).astype(np.uint16)
        exact = np.array_equal(actual, expected)
        report = dict(status='FUNCTIONAL_BINDING_ONLY', finite_codes=254, scales=scales[0].tolist(),
                      negative_zero_bits_preserved=bool(np.all(actual[:, 128] == 0x8000)),
                      bf16_bytes_exact=exact, mismatched_values=int(np.count_nonzero(actual != expected)),
                      timed=False, gain_credit_ms=0, native_producer_consumer_qualified=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        if not exact:
            raise AssertionError('Signed LSU/native integer conversion differs from exact source BF16 bytes')
    torch.hpu.synchronize()


if __name__ == '__main__':
    main()
