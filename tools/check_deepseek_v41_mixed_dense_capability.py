# SPDX-License-Identifier: Apache-2.0
"""Check a mixed MME API contract only, without measuring performance."""
import json
import os
from pathlib import Path


def main():
    os.environ['VLLM_HPU_DSV41_UNIQUE_BASE_KERNEL'] = str(
        Path(os.environ['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR']) / 'libdeepseek_v4_gaudi2_kernels.so')
    import torch
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(os.environ['DSV41_MIXED_DENSE_PROBE_LIBRARY'])
    result = []
    for rows, width, columns in ((2, 128, 256), (6, 128, 256), (6, 5120, 1792)):
        x = torch.full((rows, width), 0.5, dtype=torch.bfloat16).to('hpu')
        w = torch.full((columns, width), 1.75, dtype=torch.bfloat16).to(torch.float8_e4m3fn).to('hpu')
        row = dict(shape_a=list(x.shape), shape_b=list(w.shape), output_dtype='float32')
        try:
            def operation(x, w):
                return torch.ops.custom_op.custom_deepseek_v41_mixed_dense_probe_gaudi2(x, w)

            y = torch.compile(operation, backend='hpu_backend', fullgraph=True, dynamic=False)(x, w).cpu()
            row.update(api_success=bool((y == 0.875 * width).all()), shape=list(y.shape))
        except Exception as error:
            row.update(api_success=False, error=f'{type(error).__name__}: {error}')
        result.append(row)
        Path('mixed-dense-capability.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
