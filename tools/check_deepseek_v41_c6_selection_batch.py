# SPDX-License-Identifier: Apache-2.0
"""Preserve C6 TopK/tie/publication order through the real MLA consumer."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--native-threshold", action="store_true")
    args = parser.parse_args()
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v41_decode_selection import batched_decode_selection
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch._dynamo.config.recompile_limit = 16
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(status="running",
                  cases=[],
                  scope="pre-scored C6 selection through packed MLA",
                  formal_target_met=False)

    class Chain(torch.nn.Module):
        _merge_topk = staticmethod(PagedCSA2Attention._merge_topk)

        def __init__(self, ratio, publish, batched, columns, visible):
            super().__init__()
            self.ratio, self.publish, self.batched = ratio, publish, batched
            self.columns, self.visible = columns, visible
            self.decode_threshold_selection = args.native_threshold and batched

        def _scores(self, positions, rows, q, bank):
            del positions, q
            return bank.index_select(1, rows.long())

        def forward(self, bank, rows, positions, q, swa, main, pages, sink, scale, lengths):
            scores = bank.masked_fill(rows[None, :] >= ((positions + 1) // self.ratio)[:, None], -torch.inf)
            if self.batched and not args.native_threshold:
                indices, values, blocks = batched_decode_selection(list(scores.split(2048, -1)),
                                                                   rows,
                                                                   positions,
                                                                   self.ratio,
                                                                   collect_blocks=self.publish,
                                                                   selected_tiles=self.visible // 2048)
            else:
                indices, values, blocks = PagedCSA2Attention._stream_topk(self,
                                                                          positions,
                                                                          rows,
                                                                          bank.new_empty(6, 32, 128),
                                                                          scores,
                                                                          collect_blocks=self.publish,
                                                                          visible_rows=self.visible)
            ids = torch.where((indices >= 0) & (indices < ((positions + 1) // self.ratio)[:, None]), indices, 1048576)
            ids = ids.sort(-1).values
            ids = torch.where(ids < 1048576, ids, -1).int().contiguous()
            output = torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(q, swa, main, ids, positions, pages,
                                                                                sink, scale, lengths, self.ratio, True)
            if self.publish:
                blocks = torch.where(values > -torch.inf, blocks, -1).int()
                return output, ids, blocks
            return output, ids

    try:
        with torch.inference_mode():
            torch.manual_seed(10308)
            q = torch.randn(6, 16, 512).bfloat16().to("hpu")
            swa = pack_swa(torch.randn(256, 512).bfloat16()).to("hpu")
            main = pack_fp4(torch.randn(32768, 512).bfloat16(), 16).to("hpu")
            pages = torch.arange(512, dtype=torch.int32).to("hpu")
            positions = torch.arange(16384, 16390, dtype=torch.int32).to("hpu")
            sink = torch.randn(16).float().to("hpu")
            scale = torch.tensor([512**-.5], device="hpu")
            lengths = torch.full((6, ), 640, dtype=torch.int32, device="hpu")
            for ratio, columns, visible, publish in ((1, 32768, 20480, True), (2, 16384, 10240, False),
                                                     (1, 32768, 32768, True), (2, 16384, 16384, False)):
                rows = torch.arange(columns, dtype=torch.int32).to("hpu")
                functions = [
                    torch.compile(Chain(ratio, publish, arm, columns, visible),
                                  backend="hpu_backend",
                                  fullgraph=True,
                                  dynamic=False) for arm in (False, True)
                ]
                exact = []
                bank = torch.randn(6, columns).bfloat16().float().to("hpu")
                for distribution in ("bf16", "ties", "all_tied", "invalid"):
                    source = bank if distribution == "bf16" else bank.round() if distribution == "ties" else (
                        torch.zeros_like(bank) if distribution == "all_tied" else torch.full_like(bank, -torch.inf))
                    operands = source, rows, positions, q, swa, main, pages, sink, scale, lengths
                    outputs = [tuple(v.cpu() for v in fn(*operands)) for fn in functions]
                    if args.native_threshold:
                        visible_scores = source.cpu().masked_fill(
                            rows.cpu()[None, :] >= ((positions.cpu() + 1) // ratio)[:, None], -torch.inf)
                        cutoff = visible_scores.topk(512, dim=-1).values[:, -1:]
                        candidate_ids = outputs[1][1]
                        chosen = visible_scores.gather(1, candidate_ids.clamp_min(0).long())
                        valid = candidate_ids >= 0
                        if not bool(((chosen >= cutoff) | ~valid).all()):
                            raise AssertionError("Threshold selection retained a row below the exact cutoff")
                        expected_count = torch.minimum((visible_scores > -torch.inf).sum(-1),
                                                       torch.full((6,), 512))
                        if not torch.equal(valid.sum(-1), expected_count):
                            raise AssertionError("Threshold selection lost a valid row")
                        if not bool(torch.isfinite(outputs[1][0]).all()):
                            raise AssertionError("MLA consumer is non-finite after threshold selection")
                        difference = (outputs[0][0].float() - outputs[1][0].float()).abs().max()
                        exact.append(dict(distribution=distribution, selected_cutoff_valid=True,
                                          indices_exact=torch.equal(outputs[0][1], candidate_ids),
                                          mla_max_abs=float(difference)))
                        continue
                    for a, b in zip(*outputs, strict=True):
                        if not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
                            raise AssertionError(
                                dict(ratio=ratio, distribution=distribution, different=int((a != b).sum())))
                    exact.append(distribution)
                operands = bank, rows, positions, q, swa, main, pages, sink, scale, lengths
                for fn in functions:
                    for _ in range(8):
                        fn(*operands)
                torch.hpu.synchronize()
                timings = []
                for arm in (0, 1, 0, 1, 0, 1):
                    start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    start.record()
                    wall = time.perf_counter_ns()
                    for _ in range(args.steps):
                        functions[arm](*operands)
                    stop.record()
                    stop.synchronize()
                    timings.append(
                        dict(arm=arm,
                             device_ms=start.elapsed_time(stop) / args.steps,
                             host_ms=(time.perf_counter_ns() - wall) / 1e6 / args.steps))
                a = statistics.fmean(v["device_ms"] for v in timings if v["arm"] == 0)
                b = statistics.fmean(v["device_ms"] for v in timings if v["arm"] == 1)
                report["cases"].append(
                    dict(ratio=ratio,
                         publish=publish,
                         exact_distributions=exact,
                         parent_ms=a,
                         candidate_ms=b,
                         saved_ms=a - b,
                         timings=timings))
            report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (root / "selection-batch.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
