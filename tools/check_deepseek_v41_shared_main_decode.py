# SPDX-License-Identifier: Apache-2.0
"""Qualify shared selected-main KV through four complete MLA consumers."""
import argparse
import json
import os
from pathlib import Path
import time
from types import FunctionType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers

    torch.hpu.set_device(0)
    bind_worker_cpu(0)
    bind_worker_helpers(0)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(status="running", endpoint="four MLA layers -> BF16 outputs and argmax consumer",
                  scope="synthetic packed pages; no model/TP communication", formal_qualified=False, cases=[])

    def save():
        (root / "shared-main.json").write_text(json.dumps(report, indent=2) + "\n")

    def body(reuse, ratio, count, heads):
        def forward(queries, rings, main, selected, positions, pages, sink, scale, lengths):
            outputs, shared, mask = [], None, None
            for layer in range(4):
                if reuse and layer == 0:
                    output, shared, mask = torch.ops.custom_op.custom_deepseek_v41_main_publish_mla_gaudi2(
                        queries[layer], rings[layer], main, selected, positions, pages, sink, scale, lengths, ratio)
                elif reuse:
                    output = torch.ops.custom_op.custom_deepseek_v41_main_reuse_mla_gaudi2(
                        queries[layer], rings[layer], shared, mask, positions, sink, scale, lengths)
                else:
                    output = torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(
                        queries[layer], rings[layer], main, selected, positions, pages, sink, scale, lengths, ratio)
                outputs.append(output)
            output = torch.cat(outputs, 0)
            return output, output.float().argmax(-1)
        # Each static fixture has its own compiler entry; the fixture sweep must
        # not consume the recompilation budget of a production-shaped entry.
        name = f"shared_main_{int(reuse)}_r{ratio}_c{count}_h{heads}"
        entry = FunctionType(forward.__code__.replace(co_name=name), forward.__globals__, name,
                             forward.__defaults__, forward.__closure__)
        return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)

    save()
    with torch.inference_mode():
        generator = torch.Generator().manual_seed(6416)
        main = torch.randint(0, 256, (257 * 128, 288), dtype=torch.uint8, generator=generator)
        main[:, 256:] = 56
        main = main.to("hpu")
        pages = torch.arange(257, dtype=torch.int32, device="hpu")
        for count in range(1, 7):
            for heads in (16, 32):
                for ratio in (1, 2):
                    queries = torch.randn(4, count, heads, 512, generator=generator).bfloat16().to("hpu")
                    rings = torch.randint(0, 126, (4, 256, 528), dtype=torch.uint8, generator=generator)
                    rings[:, :, 512:] = 127
                    rings = rings.to("hpu")
                    selected = torch.randint(0, 16384 // ratio, (count, 512), generator=generator, dtype=torch.int32)
                    selected[:, :4] = torch.tensor([-1, 0, 0, 16384 // ratio], dtype=torch.int32)
                    positions = torch.arange(16384, 16384 + count, dtype=torch.int32, device="hpu")
                    operands = (queries, rings, main, selected.to("hpu"), positions, pages,
                                torch.zeros(heads, device="hpu"), torch.tensor([512 ** -.5], device="hpu"),
                                torch.full((count,), 640, dtype=torch.int32, device="hpu"))
                    old, new = body(False, ratio, count, heads), body(True, ratio, count, heads)
                    a, b = tuple(x.cpu() for x in old(*operands)), tuple(x.cpu() for x in new(*operands))
                    for left, right in zip(a, b, strict=True):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    row = dict(count=count, heads=heads, ratio=ratio, exact=True)
                    if count == 6 and heads == 16 and ratio == 1:
                        row["blocks"] = []
                        for candidate in (False, True, False, True, False, True):
                            fn = new if candidate else old
                            for _ in range(32):
                                fn(*operands)
                            torch.hpu.synchronize()
                            samples = []
                            for _ in range(args.steps):
                                begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                                begin.record()
                                start = time.perf_counter_ns()
                                fn(*operands)
                                end.record()
                                end.synchronize()
                                samples.append(dict(device_ms=begin.elapsed_time(end),
                                                    drained_host_ms=(time.perf_counter_ns() - start) / 1e6))
                            row["blocks"].append(dict(candidate=candidate, samples=samples))
                    report["cases"].append(row)
                    save()
        report["status"] = "completed_component_ab"
        save()


if __name__ == "__main__":
    main()
