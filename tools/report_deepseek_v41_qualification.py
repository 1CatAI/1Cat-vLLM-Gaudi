# SPDX-License-Identifier: Apache-2.0
"""Summarize saved exact-prompt requests without running another measurement."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    phases = ("warm-16k", "speed-01", "speed-02", "speed-03", "trace-prefill", "trace-decode")
    result = dict(run=str(run), requests=[], memory=[], quality_review="pending human-readable output review")
    review = run / "quality-review.json"
    if review.exists():
        result["quality_review"] = json.loads(review.read_text())
    capture_failure = run / "trace-hardware-failure.json"
    reference = None
    for phase in phases:
        root = run / phase
        if not (root / "result.json").exists():
            result["requests"].append(dict(phase=phase, status="not_run"))
            continue
        saved = json.loads((root / "result.json").read_text())
        row = {key: value for key, value in saved.items() if key not in ("sse_events", "client_inter_token_latency")}
        row["phase"] = phase
        if saved.get("profile"):
            manifest = root / "capture/manifest.json"
            row["hardware_capture_status"] = (json.loads(manifest.read_text()).get("status", "unvalidated")
                                              if manifest.exists() else "not_archived")
            if capture_failure.exists():
                captures = json.loads(capture_failure.read_text()).get(phase, [])
                if captures and any(not capture["hardware_events"] for capture in captures):
                    row["hardware_capture_status"] = "failed_no_device_events"
        intervals = saved.get("client_inter_token_latency", {})
        row["client_inter_token_latency"] = {key: value for key, value in intervals.items() if key != "samples_ms"}
        engine = saved.get("engine", {})
        for label, metric in (("engine_ttft_s", "time_to_first_token_seconds"),
                              ("engine_prefill_s", "request_prefill_time_seconds"), ("engine_decode_s",
                                                                                     "request_decode_time_seconds")):
            if engine.get(metric, {}).get("samples") == 1:
                row[label] = engine[metric]["seconds"]
        if "stream_chunk_size" not in saved and (run / "client-streaming-diagnosis.json").exists():
            row["client_timing_limitations"] = (
                "Bytewise parsing of the 85KB prompt-ID SSE delayed first-token arrival and created an early "
                "buffered burst. Client TTFT/ITL do not qualify delivery latency. Engine metrics remain valid; "
                "see client-streaming-diagnosis.json. Final completion arrived after the backlog drained.")
            row["client_inter_token_latency"]["valid_for_delivery_latency"] = False
        ids = json.loads((root / "token_ids.json").read_text()) if (root / "token_ids.json").exists() else []
        row["token_ids_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        if saved["status"] == "passed":
            if reference is None:
                reference = ids
            row["matches_warm_tokens"] = ids == reference
            row["first_different_token"] = next((i for i, pair in enumerate(zip(ids, reference)) if pair[0] != pair[1]),
                                                min(len(ids), len(reference)) if ids != reference else None)
        result["requests"].append(row)
    speed = [row for row in result["requests"] if row["phase"].startswith("speed-") and row["status"] == "passed"]
    result["speed_rounds"] = len(speed)
    if speed:
        result["unprofiled_summary"] = {
            key:
            dict(median=statistics.median(row[key] for row in speed),
                 minimum=min(row[key] for row in speed),
                 maximum=max(row[key] for row in speed))
            for key in ("client_ttft_s", "client_first_content_s", "client_total_s", "engine_ttft_s",
                        "engine_prefill_s", "engine_decode_s", "prefill_tokens_per_s", "decode_tokens_per_s",
                        "decode_ms_per_token")
        }
    process = json.loads((run / "process.json").read_text())
    maxima = {}
    if (run / "resource-samples.jsonl").exists():
        with (run / "resource-samples.jsonl").open() as stream:
            for line in stream:
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A live monitor may still be appending its last record.
                for device in sample.get("hpu", []):
                    module = device["module"]
                    maxima[module] = max(maxima.get(module, 0), device["used_MiB"])
    for rank, device in enumerate(process["modules"]):
        ready_path = run / f"traces/rank{rank}-native-ready.json"
        ready = json.loads(ready_path.read_text()) if ready_path.exists() else {}
        result["memory"].append(
            dict(rank=rank,
                 module=device["module"],
                 driver_sampled_peak_MiB=maxima.get(device["module"]),
                 allocator_at_ready={
                     key: ready.get(key)
                     for key in ("allocated_bytes", "peak_allocated_bytes", "state_bytes")
                 },
                 profile_memory_admission=ready.get("profile_memory_admission")))
    result["memory_boundary"] = ("Driver maximum is sampled every 2 seconds over startup and all requests, "
                                 "not allocator-exact.")
    (run / "qualification-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# TP4 16K → EOS 实测",
        "",
        "以下指标按原始请求保存；warm 与 profiler 请求单列，不纳入无 profiler 三轮汇总。",
        "",  # noqa: E501
        "| 请求 | 状态 | 生成 tokens | 服务端 TTFT s | Prefill tok/s | Decode tok/s | Decode ms/token | 总耗时 s |",
        "|---|---|---:|---:|---:|---:|---:|---:|"
    ]
    for row in result["requests"]:
        values = [row["phase"], row["status"]]
        for key in ("tokens", "engine_ttft_s", "prefill_tokens_per_s", "decode_tokens_per_s", "decode_ms_per_token",
                    "client_total_s"):
            value = row.get(key)
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value) if value is not None else "—")
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "",
        "Prefill 速度 = 16384 / 服务端 scheduled→first token；Decode = (生成数−1) / first→last token。",
        "表中 TTFT 使用服务端统计，总耗时为客户端边界；完整统计、EOS 证明和重复 token 一致性见 qualification-summary.json。",  # noqa: E501
        "若存在 client-streaming-diagnosis.json，旧客户端的首包逐字节解析会污染客户端 TTFT/ITL；"
        "原始值保留但不能作为交付延迟结论，服务端测速不受此测量问题影响。",
        "语义审核单独保存，正常 EOS 本身不证明回答正确。",
        ""  # noqa: E501
    ])
    for row in result["requests"]:
        if "hardware_capture_status" in row:
            lines.append(f"- {row['phase']} 硬件采集：{row['hardware_capture_status']}；表中 passed 仅代表请求生成验收。")  # noqa: E501
    (run / "MEASUREMENTS.md").write_text("\n".join(lines))
    print(json.dumps({"speed_rounds": len(speed), "requests": [(r["phase"], r["status"]) for r in result["requests"]]}))


if __name__ == "__main__":
    main()
