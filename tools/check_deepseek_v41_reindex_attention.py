# SPDX-License-Identifier: Apache-2.0
"""Real layer24 projections, live SWA writes, Reindex and output consumption.

Single-rank component: TP head gather is duplicated locally. Compare the two
normal request-batch Attention paths at identical precision, not serial C1.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--reference-timing", action="store_true")
    args = parser.parse_args()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi import envs
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree, linear
    from vllm_gaudi.ops.deepseek_v41_batch_state import BatchLayerState
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar, precision_config
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention, PagedCSA2SharedState
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar, layer_selection
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa

    torch.set_num_threads(1)
    torch.manual_seed(2951)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    shard = PreparedV41Shard(args.prepared, 1, 0)
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    specs = {k: v for k, v in shard.specs.items() if k.startswith("layers.24.attn.")}
    weights = _weight_tree(specs)
    dc = precision_config(envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG)
    wc = layer_selection(envs.VLLM_HPU_DSV41_WO_A_FP8_CONFIG)
    load_weight_tree(shard,
                     weights,
                     "hpu",
                     specs,
                     woa_sidecar=WoaFP8Sidecar(envs.VLLM_HPU_DSV41_WO_A_FP8_SIDECAR, shard),
                     woa_layers=wc["layers"],
                     dense_sidecar=DenseFP8Sidecar(envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR, shard),
                     dense_config=dc)
    batch = args.batch
    shared = PagedCSA2SharedState(config, 20, 25, "hpu", 1048576)
    cache = shared.sources["20"]
    page_count = 72
    physical_rows = (batch * page_count + 1) * 128
    cache.main = torch.empty(physical_rows, 288, dtype=torch.uint8, device="hpu")
    cache.index = torch.empty(physical_rows, 68, dtype=torch.uint8, device="hpu")
    for first in range(0, physical_rows, 4096):
        count = min(4096, physical_rows - first)
        cache.main[first:first + count].copy_(pack_fp4(torch.randn(count, 512).bfloat16(), 16))
        cache.index[first:first + count].copy_(pack_fp4(torch.randn(count, 128).bfloat16(), 32))
    state = BatchLayerState(batch, "hpu", compressor=False)
    state.swa.copy_(pack_swa(torch.randn(batch * 256, 512).bfloat16()))
    original = state.swa.clone()
    programs = {}
    for name, enabled in (("reference", False), ("candidate", True)):
        module = PagedCSA2Attention(
            weights.layers.get_submodule("24").attn, config, 24, shared, linear, lambda x, **kwargs: x,
            lambda x, dim: torch.cat((x, x), dim), "hpu")
        module.woa_fp8 = 24 in wc["layers"]
        module.batch_reindex_mme = enabled
        module.prepare_output_weight()
        module.prepare_qkv_input_weight()
        module.prepare_compressor_input_weight()
        module.set_search_length(1048576)
        module.batch_state = state
        programs[name] = torch.compile(module.forward_batch, backend="hpu_backend", fullgraph=True, dynamic=False)
    slots = torch.randperm(batch).int()
    pages = torch.zeros(batch, 8192, dtype=torch.int32)
    for row in range(batch):
        pages[row, :page_count] = torch.arange(page_count) + 1 + int(slots[row]) * page_count
    slots, pages = slots.to("hpu"), pages.to("hpu")
    value = torch.randn(batch, 5120).bfloat16().to("hpu")
    positions = torch.zeros(batch, dtype=torch.int32, device="hpu")
    selected = torch.full((batch, 512), -1, dtype=torch.int32, device="hpu")
    candidates = torch.full((batch, 2048), -1, dtype=torch.int32, device="hpu")
    done = torch.zeros(batch, dtype=torch.int32, device="hpu")
    inputs = value, positions, slots, pages, selected, candidates, done, done
    report = dict(scope=__doc__, batch=batch, layer=24, cases=[], all_exact=False)
    for visible in (511, 531, 1024, 2048, 8193):
        positions.copy_(torch.full((batch, ), visible - 1, dtype=torch.int32))
        pool = torch.full((batch, 2048), -1, dtype=torch.int32)
        for row in range(batch):
            blocks = torch.randperm((visible + 7) // 8).int()
            pool[row, :blocks.numel()] = blocks
        candidates.copy_(pool[:, torch.randperm(2048)].contiguous())
        case = dict(visible=visible, checks=[], timings={})
        for change in range(2):
            value.copy_(torch.randn(batch, 5120).bfloat16())
            outputs, states = {}, {}
            for name, program in programs.items():
                state.swa.copy_(original)
                outputs[name] = tuple(t.cpu() for t in program(*inputs))
                states[name] = state.swa.cpu()
            equal = [torch.equal(x, y) for x, y in zip(outputs["reference"], outputs["candidate"], strict=True)]
            same_state = torch.equal(states["reference"], states["candidate"])
            case["checks"].append(dict(outputs=equal, swa_exact=same_state))
            if not all(equal) or not same_state:
                torch.save(dict(outputs=outputs, states=states, input=value.cpu()),
                           args.output.with_name(f"failure-{visible}-{change}.pt"))
                report["cases"].append(case)
                args.output.write_text(json.dumps(report, indent=2))
                raise RuntimeError("Real complete Reindex Attention contract failed")
        for name, program in programs.items():
            if name == "reference" and not args.reference_timing:
                continue
            for _ in range(3):
                program(*inputs)
            samples = []
            for _ in range(15):
                torch.hpu.synchronize()
                begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                resident = torch.hpu.memory_allocated()
                torch.hpu.reset_peak_memory_stats()
                wall = time.perf_counter()
                begin.record()
                program(*inputs)
                end.record()
                end.synchronize()
                samples.append(
                    dict(device_ms=begin.elapsed_time(end),
                         wall_ms=(time.perf_counter() - wall) * 1000,
                         extra_peak_bytes=torch.hpu.max_memory_allocated() - resident))
            case["timings"][name] = dict(samples=samples,
                                         median_device_ms=statistics.median(x["device_ms"] for x in samples),
                                         median_wall_ms=statistics.median(x["wall_ms"] for x in samples))
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2))
        print(visible, {k: v["median_device_ms"] for k, v in case["timings"].items()}, flush=True)
    report["all_exact"] = True
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
