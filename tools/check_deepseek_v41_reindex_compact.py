# SPDX-License-Identifier: Apache-2.0
"""Native compact-descriptor contract gate; not full Reindex qualification."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    torch.set_num_threads(1)
    path = Path(__file__).resolve().parents[1] / "vllm_gaudi/ops/deepseek_v41_reindex_compact.py"
    spec = importlib.util.spec_from_file_location("compact_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    report = {"scope": "descriptor correctness only; no MME skip or performance qualification", "cases": []}
    started = time.time()
    for batch in (1, 4, 8, 32, 64):
        for ratio in (1, 2):
            # Separate code objects isolate the finite batch/ratio contracts.
            from types import FunctionType

            def entry(pool, positions, ratio=ratio):
                return torch.ops.custom_op.custom_deepseek_v41_reindex_compact_gaudi2(pool, positions, ratio)

            fn = FunctionType(entry.__code__.replace(co_name=f"compact_B{batch}_R{ratio}"),
                              entry.__globals__,
                              argdefs=entry.__defaults__,
                              closure=entry.__closure__)
            compiled = torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
            pool = torch.empty((batch, 2048), dtype=torch.int32, device="hpu")
            positions = torch.empty(batch, dtype=torch.int32, device="hpu")
            generator = torch.Generator().manual_seed(412 + batch + ratio)
            for visible in (8193, 531, 511, 0, 513, 2048, 2049, 1048576 // ratio, 512):
                # Long -> short -> empty -> long tests stale descriptor reuse.
                host = torch.randint(-1, 4096, (batch, 2048), dtype=torch.int32, generator=generator)
                host[:, :6] = torch.tensor([0, -1, 0, 65, 2**31 - 1, -(2**31)])
                lengths = torch.tensor([max(0, visible - row * 7) for row in range(batch)], dtype=torch.int32)
                host_pos = lengths * ratio - 1
                pool.copy_(host)
                positions.copy_(host_pos)
                expected = module.compact_reindex_reference(host, host_pos, ratio)
                eager = torch.ops.custom_op.custom_deepseek_v41_reindex_compact_gaudi2(pool, positions, ratio)
                output = compiled(pool, positions)
                equality = [
                    torch.equal(e.cpu(), c) and torch.equal(a.cpu(), c)
                    for e, a, c in zip(eager, output, expected, strict=True)
                ]
                case = {
                    "batch": batch,
                    "ratio": ratio,
                    "visible": visible,
                    "exact": equality,
                    "valid_blocks": expected[2].tolist()
                }
                report["cases"].append(case)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                if not all(equality):
                    torch.save(
                        {
                            "input": host,
                            "positions": host_pos,
                            "expected": expected,
                            "eager": [x.cpu() for x in eager],
                            "compiled": [x.cpu() for x in output]
                        }, args.output.with_suffix(".failure.pt"))
                    raise RuntimeError(f"Descriptor ownership mismatch: {case}")
            print(f"PASS B{batch} ratio{ratio}", flush=True)
    report.update(exact=True, seconds=time.time() - started)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
