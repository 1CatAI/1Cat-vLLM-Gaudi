# SPDX-License-Identifier: Apache-2.0
"""Functional FP8 scaled-GEMM capability only; no performance qualification."""
import json
import os
from pathlib import Path

def main():
    os.environ["VLLM_HPU_DSV41_UNIQUE_BASE_KERNEL"] = str(
        Path(os.environ["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"]) / "libdeepseek_v4_gaudi2_kernels.so")
    import torch
    import habana_frameworks.torch.core  # noqa: F401

    results = []
    for batched in (False, True):
        for out in (torch.float32, torch.bfloat16):
            shape_a, shape_b = ((6, 128), (128, 256)) if not batched else ((18, 1, 128), (18, 128, 256))
            scale_a, scale_b = ((6, 1), (1, 256)) if not batched else ((18, 1, 1), (18, 1, 256))
            a = torch.ones(shape_a, dtype=torch.bfloat16).to(torch.float8_e4m3fn).to('hpu')
            b = torch.ones(shape_b, dtype=torch.bfloat16).to(torch.float8_e4m3fn).to('hpu')
            sa = torch.full(scale_a, 0.5, dtype=torch.float32, device='hpu')
            sb = torch.full(scale_b, 0.25, dtype=torch.float32, device='hpu')
            row = dict(batched=batched, output_dtype=str(out))
            try:
                def operation(a, b, sa, sb, dtype=out):
                    return torch.ops.hpu.fp8_gemm_v2(a, False, b, False, None, dtype, sa, sb, None, False)

                c = torch.compile(operation, backend='hpu_backend', fullgraph=True, dynamic=False)(a, b, sa, sb)
                actual = c.cpu().float()
                row.update(success=bool((actual == 16).all()), shape=list(actual.shape))
            except Exception as error:
                row.update(success=False, error=f'{type(error).__name__}: {error}')
            results.append(row)
            Path('scaled-gemm-capability.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
