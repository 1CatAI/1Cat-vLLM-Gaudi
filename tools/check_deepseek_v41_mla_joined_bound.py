# SPDX-License-Identifier: Apache-2.0
"""Direction gate for one shared-KV GEMM across six queries.

Union membership is prepared on CPU outside timing. The measured device
chain includes packed KV decode, QK, sink softmax and PV. This is an upper
bound for the compute path, not a complete union implementation or a gain
eligible for the real16 ledger.
"""
import argparse
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

    prepare_environment(tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa
    from deepseek_v41_micro_replay import RecipeRecorder

    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch.ops.load_library(os.environ["DSV41_UNIQUE_OPERATOR_LIBRARY"])
    recorder = RecipeRecorder(root)
    from vllm_gaudi.ops.deepseek_v41_native_trace import configure_post_graph_directory

    configure_post_graph_directory(root / "graphs" / "rank0")

    def reference(q, swa, main, selection, positions, pages, sink, scale, lengths, physical, membership):
        return torch.ops.custom_op.custom_deepseek_v41_logical_mla_gaudi2(
            q, swa, main, selection, positions, pages, sink, scale, lengths, 1, True)

    def joined(q, swa, main, selection, positions, pages, sink, scale, lengths, physical, membership):
        window, _ = torch.ops.custom_op.custom_deepseek_v41_selected_kv_bf16_gaudi2(
            swa, swa[:1].contiguous(), torch.arange(256, dtype=torch.int32, device=q.device)[None, :], True)
        window = window.index_select(0, physical[0, :133].long())
        main_ids = (physical[0, 133:] - 256).clamp_min(0).contiguous()
        packed = main.index_select(0, main_ids.long()).contiguous()
        decoded = torch.ops.custom_op.custom_deepseek_v41_prefill_main_decode_gaudi2(
            packed, pages, main_ids, 0)
        kv = torch.cat((window, decoded), 0).contiguous()
        scores = torch.ops.custom_op.custom_deepseek_v41_mla_joined_gemm_f32_gaudi2(q.flatten(0, 1), kv)
        scores = scores.reshape(q.shape[0], q.shape[1], kv.shape[0]) * scale
        scores = scores + membership[:, None, :]
        extended = torch.cat((scores, sink[None, :, None].expand(q.shape[0], -1, -1)), -1)
        probability = extended.softmax(-1)[..., :-1].contiguous().flatten(0, 1)
        result = probability @ kv.float()
        return result.reshape(q.shape).bfloat16()

    functions = [torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
                 for fn in (reference, joined)]
    report = dict(status="running", cases=[], device_union_metadata_included=False,
                  ledger_credit_ms=0, serving_qualified=False, mla_heads=16)
    try:
        with torch.inference_mode():
            torch.manual_seed(4206)
            q = torch.randn(6, 16, 512).bfloat16().to("hpu")
            swa = pack_swa(torch.randn(256, 512).bfloat16()).to("hpu")
            main = pack_fp4(torch.randn(32768, 512).bfloat16(), 16).to("hpu")
            pos = torch.arange(16384, 16390, dtype=torch.int32)
            positions = pos.to("hpu")
            pages = torch.arange(256, dtype=torch.int32).to("hpu")
            sink = torch.randn(16).to("hpu")
            scale = torch.tensor([512**-.5], device="hpu")
            lengths = torch.full((6,), 640, dtype=torch.int32, device="hpu")
            for common in (512, 384, 0):
                selected = torch.arange(512, dtype=torch.int32).repeat(6, 1)
                for token in range(6):
                    selected[token, common:] += token * (512 - common)
                window = ((pos[:, None] - 127 + torch.arange(128)) & 255).int()
                original = torch.cat((window, selected + 256), -1)
                union = original.unique(sorted=True)
                width = (union.numel() + 63) // 64 * 64
                physical = torch.full((1, width), -1, dtype=torch.int32)
                physical[0, :union.numel()] = union
                counts = (original[:, :, None] == union[None, None, :]).sum(1).float()
                membership = torch.full((6, width), -torch.inf)
                membership[:, :union.numel()] = torch.where(counts > 0, counts.clamp_min(1).log(), -torch.inf)
                operands = (q, swa, main, selected.to("hpu"), positions, pages, sink, scale, lengths,
                            physical.to("hpu"), membership.to("hpu"))
                errors = []
                for _ in range(2):
                    q.add_(.0625)
                    a, b = [fn(*operands).cpu() for fn in functions]
                    if not bool(torch.isfinite(b).all()):
                        raise AssertionError("Joined MLA is non-finite")
                    errors.append(dict(max_abs=float((a.float() - b.float()).abs().max()),
                                       changed=int((a != b).sum())))
                replays = [recorder.prepare(fn, [q], [tuple(operands[1:])]) for fn in functions]
                for fn, replay in zip(functions, replays, strict=True):
                    expected = fn(*operands).cpu()
                    replay()
                    torch.hpu.synchronize()
                    if not torch.equal(expected.view(torch.uint8), replay.outputs[0].cpu().view(torch.uint8)):
                        raise AssertionError("Joined native replay does not match its compiled arm")
                    for _ in range(8):
                        replay()
                    torch.hpu.synchronize()
                timings = []
                for arm in (0, 1, 0, 1, 0, 1):
                    start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.steps):
                        replays[arm]()
                    stop.record()
                    stop.synchronize()
                    timings.append(dict(arm=arm, device_ms=start.elapsed_time(stop) / args.steps))
                report["cases"].append(dict(common=common, union_rows=int(union.numel()), padded_rows=width,
                                            errors=errors, timings=timings,
                                            means_ms=[statistics.fmean(x["device_ms"] for x in timings if x["arm"] == a)
                                                      for a in (0, 1)]))
                torch.hpu.synchronize()
                for replay in replays:
                    replay.close()
            report["status"] = "passed_direction_gate"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (root / "joined-bound.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
