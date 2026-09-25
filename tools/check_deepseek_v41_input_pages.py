# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Owned metadata/page preparation through the actual packed attention consumer.

One rank, production TP shard head count and context/page capacity. Native
compute replay includes packed SWA mutation, paged KV decoding, QK/softmax/PV
and output completion. This does not measure TP/PP or complete serving.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
cpus = [int(os.environ["VLLM_HPU_DSV4_WORKER_CPUS"].split(",")[0])]
cpus += [int(c) for c in os.environ["VLLM_HPU_DSV4_WORKER_HELPER_CPUS"].split(";")[0].split(",") if c]
os.sched_setaffinity(0, cpus)

import torch
import habana_frameworks.torch.core  # noqa: F401
from deepseek_v41_micro_replay import RecipeRecorder
from vllm_gaudi.ops.deepseek_v41_batch_input import RequestInputFrame, fill_request_metadata
from vllm_gaudi.ops.deepseek_v41_batch_attention import batch_packed_mla, write_state_rows
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa


def consume(query, value, positions, slots, pages, selected, swa, main, sink, scale, done, ratio):
    packed = pack_swa(value)
    rows = torch.where(slots >= 0, slots * 256 + positions.remainder(256), -1).int()
    written = write_state_rows(swa, packed, rows)
    # Production B2/B16 uses this unfused packed gather -> QK/PV path.
    return batch_packed_mla(query,
                            swa,
                            main,
                            selected,
                            pages,
                            positions,
                            slots,
                            sink,
                            scale,
                            written,
                            done,
                            ratio=ratio)


def legacy(frame, requests, owners, ids):
    fill_request_metadata(frame.host, requests, owners, input_ids=ids)
    frame.metadata.copy_(frame.host, non_blocking=True)
    inputs = frame.metadata.unbind(0)
    torch.index_select(frame.bank.pages, 0, inputs[2].clamp_min(0).long(), out=frame.pages)


@torch.inference_mode()
def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--pairs", type=int, default=63)
    cli.add_argument("--state-check",
                     action="store_true",
                     help="Run candidate first so reference preparation cannot mask a stale cached page frame")
    args = cli.parse_args()
    evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.set_num_threads(1)
    torch.manual_seed(260925)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    recorder = RecipeRecorder(evidence)
    report = dict(scope=__doc__,
                  context_capacity=1048576,
                  page_width=8193,
                  request_slots=32,
                  query_heads=32,
                  head_dim=512,
                  pairs=args.pairs,
                  correctness_only=args.state_check,
                  cases=[])
    saved = {}
    compiled = torch.compile(consume, backend="hpu_backend", fullgraph=True, dynamic=False)
    output = evidence / "result.json"
    try:
        for ratio, width, lanes in ((2, 1, 1), (2, 2, 1), (2, 16, 2), (1, 16, 2)):
            case_name = f"ratio{ratio}-b{width}x{lanes}"
            bank_host = torch.zeros(32, 8193, dtype=torch.int32)
            for slot in range(32):
                bank_host[slot, :18] = torch.arange(18) + 1 + slot * 18
            bank = SimpleNamespace(
                pages=bank_host.to("hpu"),
                page_versions={slot: (1, tuple(bank_host[slot, :18].tolist()))
                               for slot in range(32)})
            swa = pack_swa(torch.randn(32 * 256, 512).bfloat16()).to("hpu")
            main_cache = pack_fp4(torch.randn((32 * 18 + 1) * (128 // ratio), 512).bfloat16(), 16).to("hpu")
            sink = torch.randn(32, device="cpu").float().to("hpu")
            scale = torch.tensor([512**-0.5], device="hpu")
            frames, queries, values, fixed, specs = [], [], [], [], []
            for lane in range(lanes):
                host = torch.zeros(3, width, dtype=torch.int32).pin_memory("hpu")
                metadata = torch.zeros_like(host, device="hpu")
                page = torch.empty(width, 8193, dtype=torch.int32, device="hpu")
                frame = RequestInputFrame(bank, host, metadata, page)
                query = torch.randn(width, 32, 512).bfloat16().to("hpu")
                value = torch.randn(width, 512).bfloat16().to("hpu")
                selected = (torch.arange(512, dtype=torch.int32) * 2).expand(width, -1).clone().to("hpu")
                done = torch.zeros(width, dtype=torch.int32, device="hpu")
                owners = [SimpleNamespace(index=lane * width + i, generation=1) for i in range(width)]
                requests = [SimpleNamespace(num_computed_tokens=2048 + i) for i in range(width)]
                ids = [19 + i for i in range(width)]
                frame.prepare(requests, owners, input_ids=ids)
                frames.append(frame)
                queries.append(query)
                values.append(value)
                fixed.append((value, frame.inputs[1], frame.inputs[2], page, selected, swa, main_cache, sink, scale,
                              done, ratio))
                specs.append((requests, owners, ids))
            for _ in range(3):
                for query, parameters in zip(queries, fixed, strict=True):
                    compiled(query, *parameters)
            torch.hpu.synchronize()
            native = recorder.prepare(compiled, queries, fixed)
            samples = []
            begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
            try:
                for pair in range(args.pairs if width > 1 else 5):
                    # Values and positions always change; ownership usually remains
                    # stable as it does during a steady request wave. Exercise
                    # padding, row reorder, page replacement and slot reuse too.
                    reorder = pair // 16
                    if pair in (24, 48):
                        slot = 0 if pair == 24 else min(5, width * lanes - 1)
                        bank_host[slot, :18] = bank_host[slot, :18].roll(1)
                        generation = bank.page_versions[slot][0] + int(pair == 48)
                        bank.pages[slot].copy_(bank_host[slot])
                        bank.page_versions[slot] = (generation, tuple(bank_host[slot, :18].tolist()))
                    specs = []
                    for lane, (frame, query, value) in enumerate(zip(frames, queries, values, strict=True)):
                        count = width - 1 if width > 2 and pair in (8, 9) else width
                        slots = list(range(lane * width, (lane + 1) * width))
                        shift = reorder % width
                        slots = (slots[shift:] + slots[:shift])[:count]
                        owners = [SimpleNamespace(index=i, generation=bank.page_versions[i][0]) for i in slots]
                        requests = [SimpleNamespace(num_computed_tokens=2048 + pair + i) for i in range(count)]
                        ids = [17 + pair + i for i in slots]
                        query.copy_(torch.randn(query.shape).bfloat16())
                        value.copy_(torch.randn(value.shape).bfloat16())
                        specs.append((requests, owners, ids))
                    torch.hpu.synchronize()
                    arms = (True, False) if args.state_check or pair % 2 else (False, True)
                    tensors, row = {}, {"pair": pair, "reference_first": arms[0] is False}
                    for candidate in arms:
                        name = "candidate" if candidate else "reference"
                        started = time.perf_counter_ns()
                        begin.record()
                        mark = time.perf_counter_ns()
                        old_keys = [frame.page_key for frame in frames]
                        for frame, (requests, owners, ids) in zip(frames, specs, strict=True):
                            if candidate:
                                frame.prepare(requests, owners, input_ids=ids)
                            else:
                                legacy(frame, requests, owners, ids)
                        row[name + "_prepare_ms"] = (time.perf_counter_ns() - mark) / 1e6
                        native()
                        end.record()
                        end.synchronize()
                        row[name + "_wall_ms"] = (time.perf_counter_ns() - started) / 1e6
                        row[name + "_device_ms"] = begin.elapsed_time(end)
                        if candidate:
                            row["candidate_page_refreshes"] = sum(frame.page_key != old
                                                                  for frame, old in zip(frames, old_keys, strict=True))
                        tensors[name] = [value.cpu().clone() for value in native.outputs]
                        for frame in frames:
                            tensors[name] += [frame.metadata.cpu().clone(), frame.pages.cpu().clone()]
                    exact = all(
                        torch.equal(a, b) for a, b in zip(tensors["reference"], tensors["candidate"], strict=True))
                    if not exact:
                        torch.save(tensors, evidence / f"mismatch-{case_name}-{pair}.pt")
                        raise RuntimeError(f"Consumer or metadata output differs: {case_name} pair {pair}")
                    row["exact"] = True
                    samples.append(row)
                    if pair in (0, 8, 16, 24, 48, args.pairs - 1):
                        saved[f"{case_name}-pair{pair}"] = tensors["candidate"]
                info = native.native.info()
                case = dict(
                    name=case_name,
                    ratio=ratio,
                    width=width,
                    lanes=lanes,
                    samples=samples,
                    native_info=info,
                    recipes=native.recipes,
                    captured_inputs=[{
                        "shape": list(x.shape),
                        "dtype": str(x.dtype)
                    } for x in native.inputs if isinstance(x, torch.Tensor)],
                    median={
                        key: statistics.median(row[key] for row in samples)
                        for key in samples[0] if key.endswith("_ms")
                    },
                    reference_scope="Missing page-consuming component reference only; not a serving baseline",
                    consumer="production B2/B16 packed SWA/main gather and QK/softmax/PV, completed native replay")
                report["cases"].append(case)
                output.write_text(json.dumps(report, indent=2) + "\n")
                torch.save(saved, evidence / "outputs.pt")
                print(json.dumps({"case": case_name, "median": case["median"], "native_info": info}), flush=True)
            finally:
                native.close()
        report["completed"] = True
        output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        torch.hpu.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
