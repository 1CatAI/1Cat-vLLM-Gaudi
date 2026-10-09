# SPDX-License-Identifier: Apache-2.0
"""Prototype a derived MLA mirror through canonical writes and QK/softmax/PV.

The bounded mirror is initialized once from packed pages. Both arms publish
changing SWA/main rows; the candidate refreshes main rows after publication.
This synthetic component does not establish serving ownership or round cost.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import time


def make_body(use_mirror, ratio, count, native_mirror=False):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa

    def body(q, swa, main, mirror, selection, positions, pages, sink, scale, lengths, new_main, new_swa):
        logical = torch.div(positions, ratio, rounding_mode="floor")
        width = 128 // ratio
        physical = pages.index_select(0, (logical // width).long()) * width + logical % width
        main.index_copy_(0, physical.long(), pack_fp4(new_main, 16))
        swa.index_copy_(0, (positions & 255).long(), pack_swa(new_swa))
        if not use_mirror:
            return torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(q, swa, main, selection, positions, pages,
                                                                              sink, scale, lengths, ratio, True)
        # Read canonical rows after all writes: duplicate ratio-2 rows are
        # identical in the derived cache, including incomplete pair handling.
        fresh = torch.ops.custom_op.custom_deepseek_v41_prefill_main_decode_gaudi2(main, pages, logical.contiguous(),
                                                                                   ratio)
        if native_mirror:
            from vllm_gaudi.ops.deepseek_v41_batch_attention import write_state_rows

            completion = write_state_rows(mirror, fresh, physical.int())
            ring = torch.arange(256, dtype=torch.int32, device=q.device).reshape(1, -1)
            window, _ = torch.ops.custom_op.custom_deepseek_v41_selected_kv_bf16_gaudi2(
                swa, main[:1].contiguous(), ring, True)
            return torch.ops.custom_op.custom_deepseek_v41_logical_mla_decoded_gaudi2(
                q, window, mirror, selection, positions, pages, sink, scale, lengths, ratio, completion, True)
        mirror.index_copy_(0, logical.long(), fresh)
        valid = (selection >= 0) & (selection < mirror.shape[0])
        gathered = mirror.index_select(0, selection.clamp(0, mirror.shape[0] - 1).reshape(-1).long())
        gathered = torch.where(valid.reshape(-1, 1), gathered, 0)
        ring = torch.arange(256, dtype=torch.int32, device=q.device).reshape(1, -1)
        window, _ = torch.ops.custom_op.custom_deepseek_v41_selected_kv_bf16_gaudi2(swa, main[:1].contiguous(), ring,
                                                                                    True)
        cache = torch.cat((window, gathered), 0).contiguous()
        absolute = positions[:, None] - 127 + torch.arange(128, dtype=torch.int32, device=q.device)
        window_ids = torch.where(absolute >= 0, absolute & 255, -1)
        main_ids = torch.arange(count * 512, dtype=torch.int32, device=q.device).reshape(count, 512) + 256
        main_ids = torch.where(selection >= 0, main_ids, -1)
        indices = torch.cat((window_ids, main_ids), -1).contiguous()
        return torch.ops.custom_op.custom_deepseek_v41_selected_mla_mme_gaudi2(q, cache, indices, sink, scale, lengths)

    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warm", type=int, default=8)
    parser.add_argument("--native-mirror", action="store_true")
    parser.add_argument("--native-replay", action="store_true")
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    if args.native_mirror:
        torch.ops.load_library(os.environ["DSV41_UNIQUE_OPERATOR_LIBRARY"])
    torch._dynamo.config.recompile_limit = 16
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    if os.environ.get("GRAPH_VISUALIZATION") == "1":
        from vllm_gaudi.ops.deepseek_v41_native_trace import configure_post_graph_directory

        # SDK configuration is valid only after device/runtime initialization.
        torch.empty(1, device="hpu")
        torch.hpu.synchronize()
        configure_post_graph_directory(root / "graphs" / "rank0")
    recorder = None
    if args.native_replay:
        from deepseek_v41_micro_replay import RecipeRecorder

        recorder = RecipeRecorder(root)
    report = dict(status="running",
                  scope="synthetic complete MLA mirror writer/consumer",
                  cases=[],
                  end_to_end_qualified=False)
    try:
        with torch.inference_mode():
            torch.manual_seed(10307)
            for ratio in (1, 2):
                for count in (2, 6):
                    capacity = 32768 // ratio
                    main = pack_fp4(torch.randn(capacity, 512).bfloat16(), 16).to("hpu")
                    swa = pack_swa(torch.randn(256, 512).bfloat16()).to("hpu")
                    pages = torch.arange(capacity // (128 // ratio), dtype=torch.int32).roll(17).to("hpu")
                    logical = torch.arange(capacity, dtype=torch.int32).to("hpu")
                    mirror = torch.ops.custom_op.custom_deepseek_v41_prefill_main_decode_gaudi2(
                        main, pages, logical, 0 if args.native_mirror else ratio)
                    q = torch.randn(count, 16, 512).bfloat16().to("hpu")
                    positions = torch.arange(16384, 16384 + count, dtype=torch.int32).to("hpu")
                    rows = torch.arange(512).repeat(count, 1).int()
                    for token in range(count):
                        rows[token, 384:] += token * 128
                    rows[:, -3:] = -1
                    rows[0, -4] = capacity * 2
                    # Include freshly published rows, rather than only a
                    # constant historical prefix in the correctness check.
                    rows[:, :count] = torch.arange(16384 // ratio, 16384 // ratio + count)
                    rows = rows.to("hpu")
                    sink = torch.randn(16).float().to("hpu")
                    scale = torch.tensor([512**-.5], device="hpu")
                    lengths = torch.full((count, ), 640, dtype=torch.int32, device="hpu")
                    fresh = torch.randn(count, 512).bfloat16()
                    if ratio == 2:
                        fresh[1::2] = fresh[::2]
                    fresh = fresh.to("hpu")
                    fresh_swa = torch.randn(count, 512).bfloat16().to("hpu")
                    operands = q, swa, main, mirror, rows, positions, pages, sink, scale, lengths, fresh, fresh_swa
                    functions = [
                        torch.compile(make_body(arm, ratio, count, args.native_mirror),
                                      backend="hpu_backend",
                                      fullgraph=True,
                                      dynamic=False) for arm in (False, True)
                    ]
                    replays = None
                    for version in range(3):
                        fresh.mul_(.5)
                        q.add_(.0625)
                        output = [fn(*operands).cpu() for fn in functions]
                        if not torch.equal(output[0].view(torch.int16), output[1].view(torch.int16)):
                            raise AssertionError(
                                dict(ratio=ratio,
                                     count=count,
                                     version=version,
                                     max_abs=float((output[0].float() - output[1].float()).abs().max())))
                    for fn in functions:
                        for _ in range(args.warm):
                            fn(*operands)
                    torch.hpu.synchronize()
                    if recorder is not None:
                        replays = [recorder.prepare(fn, [q], [tuple(operands[1:])]) for fn in functions]
                        for _ in range(2):
                            q.add_(.0625)
                            for function, replay in zip(functions, replays, strict=True):
                                expected = function(*operands).cpu()
                                replay()
                                torch.hpu.synchronize()
                                actual = replay.outputs[0].cpu()
                                if not torch.equal(expected.view(torch.int16), actual.view(torch.int16)):
                                    raise AssertionError("Native mirror consumer differs from its compiled arm")
                    timings = []
                    for arm in (() if args.checks_only else (0, 1, 0, 1, 0, 1)):
                        start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                        start.record()
                        wall = time.perf_counter_ns()
                        for _ in range(args.steps):
                            if replays is None:
                                functions[arm](*operands)
                            else:
                                replays[arm]()
                        stop.record()
                        stop.synchronize()
                        timings.append(
                            dict(arm=arm,
                                 device_ms=start.elapsed_time(stop) / args.steps,
                                 host_ms=(time.perf_counter_ns() - wall) / 1e6 / args.steps))
                    a = statistics.fmean(v["device_ms"] for v in timings if v["arm"] == 0) if timings else None
                    b = statistics.fmean(v["device_ms"] for v in timings if v["arm"] == 1) if timings else None
                    report["cases"].append(
                        dict(ratio=ratio,
                             count=count,
                             byte_exact_three_versions=True,
                             mirror_bytes=mirror.numel() * mirror.element_size(),
                             timings=timings,
                             parent_ms=a,
                             candidate_ms=b,
                             saved_ms=a - b if timings else None))
                    # Drain before releasing captured recipe buffers. Preserve
                    # the native plan's owners through its last device consumer.
                    torch.hpu.synchronize()
                    if replays is not None:
                        for replay in replays:
                            replay.close()
                        replays = None
            report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (root / "mla-mirror.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
