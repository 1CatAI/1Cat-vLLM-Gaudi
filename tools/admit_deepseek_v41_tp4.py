# SPDX-License-Identifier: Apache-2.0
"""Check the frozen four-rank serving run before sending its first request."""
import argparse
import json
from pathlib import Path
import time

import requests

from collect_deepseek_v41_trace import recipe_symbols


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == "Z" else int(fields[19])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18444")
    parser.add_argument("--wait-seconds", type=int, default=1800)
    args = parser.parse_args()
    run = args.run.resolve()
    process = json.loads((run / "process.json").read_text())
    # /proc start ticks survive wall-clock/NTP adjustments. Epoch create_time
    # reconstructed from boot time can change for the same live process.
    created = process_identity(process["pid"])
    if created is None:
        raise RuntimeError("Recorded serving process exited before admission")
    deadline = time.monotonic() + args.wait_seconds
    while True:
        if process_identity(process["pid"]) != created:
            raise RuntimeError("Recorded serving process exited before admission")
        try:
            response = requests.get(args.url + "/health", timeout=3)
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Serving admission deadline exceeded")
        time.sleep(3)
    result = dict(status="passed", time_ns=time.time_ns(), run=str(run), ranks=[],
                  modules=process["modules"], pid=process["pid"], checks=[])
    for rank in range(4):
        stats = json.loads((run / f"traces/rank{rank}-native-ready.json").read_text())
        assert stats["profiler_environment"]["HABANA_PROFILE"] == "1"
        assert stats["topology"] == {"tensor_parallel_size": 4, "pipeline_parallel_size": 1}
        assert stats["execution_path"] == "prepared_n256_tp4"
        assert stats["compiled_stage"]["calls"] > 0
        assert stats["compiled_input_calls"] > 0
        assert stats["compiled_literals"]["graphs"] > 0 and stats["compiled_literals"]["scalars"] > 0
        assert stats["compiled_stage"]["groups"] == 10 * stats["compiled_stage"]["calls"]
        assert stats["pp"]["sends"] == stats["pp"]["receives"] == 0
        assert all(stats["moe"][name] for name in ("n256_fp8_decode", "fused_quant", "fused_reduce",
                                                   "grouped_prefill", "occupied_groups_only", "fused_mhc_post"))
        assert stats["prefill_plan"]["executed"]["largest_token_bucket"] == 8192
        assert stats["prefill_plan"]["replays"] > 0
        assert stats["engram"]["native_c1"] > 0 and stats["engram"]["c1_packets"] > 0
        files = sorted((run / f"recipes/rank{rank}").glob("*.recipe"))
        assert files, f"No private compiled recipes for rank {rank}"
        sample = recipe_symbols(files[len(files) // 2])
        graphs = list((run / f"graphs/rank{rank}").rglob("*-symbol.pbtxt"))
        assert graphs, f"No saved tensor contracts for rank {rank}"
        result["ranks"].append(dict(rank=rank, ready=stats, serialized_recipes=len(files),
                                    graph_files=len(graphs), symbol_sample=sample))
    result["checks"] = ["TP4/PP1", "no PP transport", "N256 fused decode and native hybrid prefill",
                        "C8192 prepared-plan execution", "native C1 Engram packets",
                        "private recipe symbols and compiler graphs"]
    models = requests.get(args.url + "/v1/models", timeout=10)
    models.raise_for_status()
    result["models"] = models.json()
    assert all(item.get("max_model_len") == 1048576 for item in result["models"]["data"])
    (run / "admission.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "ranks"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
