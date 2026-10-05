# SPDX-License-Identifier: Apache-2.0
"""Functional native coordinate contract; no performance credit from this screen."""
import json
import os
from pathlib import Path

import torch
import habana_frameworks.torch.core  # noqa: F401

from vllm_gaudi.ops.deepseek_v41_decode_coordinates import decode_coordinates_reference


def main():
    torch.set_num_threads(1)
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    op = torch.compile(torch.ops.custom_op.custom_deepseek_v41_decode_metadata_gaudi2,
                       backend='hpu_backend', fullgraph=True, dynamic=False)
    checks = []
    pages = torch.arange(4096, dtype=torch.int32).flip(0)
    pages[3] = -1
    ids = torch.tensor([17, 129264, 129265, 512, 1024, 31], dtype=torch.int32)
    fixtures = ([0, 1, 7, 127, 128, 255], [256, 257, 383, 384, 511, 512],
                [16383, 16384, 32767, 32768, 131071, 131072],
                [524286, 524287, -1, 524288, 384, 385], [255, 127, 16384, 0, 32768, 131072])
    for tokens in (1, 2, 6):
        for index, values in enumerate(fixtures):
            pos = torch.tensor(values[:tokens], dtype=torch.int32)
            token = ids[:tokens]
            expected = (decode_coordinates_reference(pos, token, pages), pos.bitwise_and(255),
                        torch.full_like(pos, 640), ((token == 129264) | (token == 129265)).int())
            actual = tuple(v.cpu() for v in op(pos.to('hpu'), token.to('hpu'), pages.to('hpu')))
            exact = [torch.equal(a, b) for a, b in zip(actual, expected, strict=True)]
            if not all(exact):
                raise RuntimeError(f'Coordinate mismatch: C{tokens}, fixture {index}, outputs {exact}')
            checks.append(dict(tokens=tokens, fixture=index, exact=exact))
    Path(os.environ['DSV41_CHECK_RESULT']).write_text(json.dumps(dict(checks=checks, performance_credit=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
