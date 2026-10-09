# SPDX-License-Identifier: Apache-2.0
"""Compare FP4 encoding/cache publication through its ordered row consumer."""
import argparse
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-consumer", action="store_true")
    args = parser.parse_args()
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

    prepare_environment(tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, unpack_fp4
    from vllm_gaudi.ops.deepseek_v41_batch_attention import read_state_rows

    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.ops.load_library(os.environ["DSV41_UNIQUE_OPERATOR_LIBRARY"])
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.empty(1, device="hpu")
    torch.hpu.synchronize()
    from vllm_gaudi.ops.deepseek_v41_native_trace import configure_post_graph_directory

    configure_post_graph_directory(root / "graphs" / "rank0")

    def body(candidate):
        def forward(main, index, latent, key, slots, query, swa, selected, pages, positions, sink, scale, lengths):
            if candidate:
                done = torch.ops.custom_op.custom_deepseek_v41_fp4_cache_write_rows_gaudi2(
                    main, index, latent, key, slots)
                if args.attention_consumer:
                    output = torch.ops.custom_op.custom_deepseek_v41_logical_mla_write_ordered_gaudi2(
                        query, swa, main, selected, positions, pages, sink, scale, lengths, 1, done, True)
                    keys = torch.ops.custom_op.custom_deepseek_v41_index_keys_write_ordered_gaudi2(
                        index, pages, slots.reshape(1, -1), done, 1)
                    return output, keys
                completion = done[:, 0].contiguous()
                packed = read_state_rows(main, slots, completion)
                index_rows = read_state_rows(index, slots, completion)
            else:
                main.index_copy_(0, slots.long(), pack_fp4(latent, 16))
                index.index_copy_(0, slots.long(), pack_fp4(key, 32))
                if args.attention_consumer:
                    output = torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(
                        query, swa, main, selected, positions, pages, sink, scale, lengths, 1, True)
                    keys = torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2(
                        index, pages, slots.reshape(1, -1), 1)
                    return output, keys
                packed = main.index_select(0, slots.long())
                index_rows = index.index_select(0, slots.long())
            # Keep quantization, mutation, then actual packed-value consumption
            # in one graph. Reading completion alone is not the endpoint.
            return unpack_fp4(packed), index_rows, packed

        return torch.compile(forward, backend="hpu_backend", fullgraph=True, dynamic=False)

    report = dict(status="running", cases=[], serving_qualified=False)
    try:
        with torch.inference_mode():
            torch.manual_seed(42)
            functions = [body(False), body(True)]
            for count in (2, 6):
                main = torch.zeros(32768, 288, dtype=torch.uint8, device="hpu")
                index = torch.zeros(32768, 68, dtype=torch.uint8, device="hpu")
                latent = torch.randn(count, 512).bfloat16().to("hpu")
                key = torch.randn(count, 128).bfloat16().to("hpu")
                # Include reserved-null-page rows used by incomplete ratio2
                # groups, plus non-contiguous physical page destinations.
                slots = torch.tensor([4, 23174, 5, 23175, 6, 23176][:count], dtype=torch.int32, device="hpu")
                if args.attention_consumer:
                    slots = torch.arange(16384, 16384 + count, dtype=torch.int32, device="hpu")
                query = torch.randn(count, 16, 512).bfloat16().to("hpu")
                swa = torch.zeros(256, 528, dtype=torch.uint8, device="hpu")
                selected = torch.arange(512, dtype=torch.int32, device="hpu").repeat(count, 1)
                selected[:, :count] = slots
                pages = torch.arange(256, dtype=torch.int32, device="hpu")
                positions = torch.arange(16384, 16384 + count, dtype=torch.int32, device="hpu")
                sink = torch.zeros(16, device="hpu")
                scale = torch.tensor([512**-.5], device="hpu")
                lengths = torch.full((count,), 640, dtype=torch.int32, device="hpu")
                operands = main, index, latent, key, slots, query, swa, selected, pages, positions, sink, scale, lengths
                for _ in range(3):
                    latent.add_(.125)
                    key.mul_(.75)
                    outputs = [tuple(t.cpu() for t in fn(*operands)) for fn in functions]
                    if any(not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                           for a, b in zip(*outputs, strict=True)):
                        raise AssertionError("Native cache write differs at its packed/decoded consumer")
                for fn in functions:
                    for _ in range(8):
                        fn(*operands)
                torch.hpu.synchronize()
                timings = []
                for arm in (0, 1, 0, 1, 0, 1):
                    start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    start.record()
                    for _ in range(64):
                        functions[arm](*operands)
                    stop.record()
                    stop.synchronize()
                    timings.append(dict(arm=arm, device_ms=start.elapsed_time(stop) / 64))
                means = [statistics.fmean(x["device_ms"] for x in timings if x["arm"] == arm) for arm in (0, 1)]
                report["cases"].append(dict(count=count, exact=True, timings=timings,
                                            parent_ms=means[0], candidate_ms=means[1]))
            report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (root / "fp4-cache-rows.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
