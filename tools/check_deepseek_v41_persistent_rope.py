# SPDX-License-Identifier: Apache-2.0
"""Check sparse runtime RoPE addressing against independently bound rows."""
import argparse
import json
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import habana_frameworks.torch.core  # noqa: F401
    from habana_frameworks.torch.hpu.metrics import metric_global

    torch.set_num_threads(1)
    torch.manual_seed(6)
    torch.ops.load_library(str(args.library))
    phase = torch.rand(1 << 20, 64) * 2 - 1
    table = phase.to("hpu")
    position = torch.zeros(1, dtype=torch.int32, device="hpu")
    zero = position.clone()
    row = torch.empty(1, 64, device="hpu")
    x = torch.randn(1, 32, 512).bfloat16().to("hpu")
    product = torch.randn(1, 16384, device="hpu")
    scale = torch.ones(1, 16384, device="hpu")
    activation = torch.full((1, 1), .25, device="hpu")

    def chain(x, product, scale, activation, position, table):
        ops = torch.ops.custom_op
        return (ops.custom_deepseek_v41_rope_bf16_gaudi2(x, position, table),
                ops.custom_deepseek_v41_rope_inverse_bf16_gaudi2(x, position, table),
                ops.custom_deepseek_v41_q_scale_rope_gaudi2(product, scale, activation, position, table))

    compiled = torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)
    reports = []
    for index in (0, 511, 512, 1023, 1024, 524287, 1048575, 1):
        position.copy_(torch.tensor([index], dtype=torch.int32))
        row.copy_(phase[index:index + 1])
        torch.hpu.synchronize()
        before = dict(metric_global("graph_compilation").stats())
        start = time.perf_counter()
        actual = compiled(x, product, scale, activation, position, table)
        torch.hpu.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        after = dict(metric_global("graph_compilation").stats())
        expected = compiled(x, product, scale, activation, zero, row)
        exact = [torch.equal(a.cpu(), b.cpu()) for a, b in zip(actual, expected, strict=True)]
        record = dict(position=index, exact=exact, wall_ms=elapsed, compilation_before=before,
                      compilation_after=after)
        reports.append(record)
        args.output.write_text(json.dumps(reports, indent=2))
        print(json.dumps(record), flush=True)
        if not all(exact) or (len(reports) > 1 and before != after):
            raise RuntimeError("Persistent RoPE changed row semantics or recompiled on a new position")


if __name__ == "__main__":
    main()
