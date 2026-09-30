# SPDX-License-Identifier: Apache-2.0
"""Small compile/consumer gate for the shared integer-constant pass."""
import json
import os
from pathlib import Path

import torch
import habana_frameworks.torch.core  # noqa: F401
import vllm_gaudi.distributed.tp2_fused_ar_norm  # noqa: F401

from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend


def consume(positions, pages, cache):
    blocks = pages.index_select(0, torch.bitwise_right_shift(positions, 7).long())
    physical = blocks * 128 + torch.bitwise_and(positions, 127)
    ring = torch.bitwise_and(positions, 255)
    # A real downstream gather consumes the newly calculated row addresses.
    values = cache.index_select(0, physical.long())
    return values, ring, positions == 255, torch.bitwise_and(positions, -2)


def main():
    pages = torch.arange(8192, dtype=torch.int32)
    cache = torch.arange(1048576, dtype=torch.float32).reshape(-1, 1)
    hp, hc = pages.to('hpu'), cache.to('hpu')
    base = torch.compile(consume, backend=make_backend(), fullgraph=True, dynamic=False)
    candidate = torch.compile(consume, backend=make_backend(static_int32=True), fullgraph=True, dynamic=False)
    reports = []
    for count in (1, 2, 6):
        for offset in (0, 1, 3):
            positions = torch.tensor([0, 255, 256, 16384, 16385, 1048575], dtype=torch.int32).roll(offset)[:count]
            expected = consume(positions, pages, cache)
            inputs = positions.to('hpu')
            for name, fn in (('baseline', base), ('candidate', candidate)):
                output = fn(inputs, hp, hc)
                for actual, wanted in zip(output, expected):
                    torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
                reports.append(dict(arm=name, count=count, offset=offset, exact=True))
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    (root / 'INTEGER_CHAIN_CONTRACT.json').write_text(json.dumps(dict(status='passed', cases=reports,
                                                                   performance_gain_credit=False), indent=2)+'\n')
    print(f'Integer producer/consumer: {len(reports)} exact cases', flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
