# SPDX-License-Identifier: Apache-2.0
"""Run the fixed sampled micro gates after the owned input collector finishes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def derive_request_history(fixtures, service_run, prepared):
    """Rebuild the actual text request's mapped lookback without another service."""
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    from transformers import AutoTokenizer
    from vllm_gaudi.ops.deepseek_v41_engram import build_compressed_token_map

    result = service_run / "unscored-input-request"
    prefix = json.loads((result / "prompt_token_ids.json").read_text())
    output = json.loads((result / "token_ids.json").read_text())
    stream = prefix + output
    tokenizer = AutoTokenizer.from_pretrained(prepared, local_files_only=True, trust_remote_code=True)
    token_map, _ = build_compressed_token_map(tokenizer)
    destination = fixtures.with_name(fixtures.name + "-with-history")
    destination.mkdir(exist_ok=False)
    for rank in range(4):
        target = destination / f"rank{rank}"
        target.mkdir()
        for path in sorted((fixtures / f"rank{rank}").glob("c6-*.pt")):
            data = torch.load(path, map_location="cpu", weights_only=True)
            start = int(data["positions"][0])
            if start < 3 or start >= len(stream) or int(data["ids"][0]) != stream[start]:
                raise ValueError("Actual request anchor does not match its returned committed token stream")
            history = torch.tensor([token_map[stream[start - shift]] for shift in (1, 2, 3)], dtype=torch.int32)
            histories = [history]
            for token in data["ids"].tolist():
                histories.append(torch.cat((torch.tensor([token_map[token]], dtype=torch.int32), histories[-1][:2])))
            data.update(cursor_history=history, engram_histories=torch.stack(histories),
                        decoder_state_root=str(path.parent), history_source=str(result),
                        history_context_qualified=True)
            torch.save(data, target / path.name)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-record", type=Path, required=True)
    parser.add_argument("--collector-pid", type=int, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--modules", default="0,4,5,1")
    parser.add_argument("--cpu-mask", default="10-19,38-47")
    parser.add_argument("--stages", nargs="+", choices=("sharded", "native"), default=["sharded", "native"])
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    report = dict(pid=os.getpid(), status="waiting for actual request inputs", qualified_gain_ms=0, runs=[])
    record = args.evidence_root / f"{args.name}-driver.json"

    def save():
        record.write_text(json.dumps(report, indent=2) + "\n")

    save()
    while True:
        if args.collector_record.exists():
            captured = json.loads(args.collector_record.read_text())
            if captured["status"] == "captured":
                break
            if captured["status"] == "failed":
                raise RuntimeError("Actual request collection failed; no micro fallback to startup tensors")
        if not Path(f"/proc/{args.collector_pid}").exists():
            raise RuntimeError("Owned collector exited without qualifying its actual input sets")
        time.sleep(15)
    for rank in range(4):
        for index in range(3):
            if not (args.fixtures / f"rank{rank}" / f"c6-{index}.pt").is_file():
                raise RuntimeError("Actual C6 fixture set is incomplete")
    service_run = args.collector_record.with_name(args.collector_record.name.removesuffix("-fixture-client.json"))
    fixtures = derive_request_history(args.fixtures, service_run, args.prepared)
    report["derived_actual_history_fixtures"] = str(fixtures)
    stages = (
        ("sharded", "tools/check_deepseek_v41_sharded_bounded_backend.py", ["--width", "64"]),
        ("native", "tools/check_deepseek_v41_sampled_native_control.py", ["--repair-journal"]),
    )
    for index, (name, script, extra) in enumerate(stages):
        if name not in args.stages:
            continue
        run = args.evidence_root / f"{args.name}-{name}"
        profile = json.loads(args.profile.read_text())
        profile["environment"]["PT_HPU_RECIPE_CACHE_CONFIG"] = str(run.with_name(run.name + "-recipes") /
                                                                 "rank{rank}") + ",true,1024"
        profile["environment"]["VLLM_HPU_DSV41_REQUEST_FIXTURE_DIR"] = ""
        if name == "native":
            profile["environment"]["GRAPH_VISUALIZATION"] = "1"
        profile_path = args.evidence_root / f"{args.name}-{name}-profile.json"
        profile_path.write_text(json.dumps(profile, indent=2) + "\n")
        command = ["taskset", "-c", args.cpu_mask, sys.executable, "tools/run_deepseek_v41.py",
                   "--devices", "4", "--modules", args.modules, "--lock-dir", str(workspace.parent / "locks"),
                   "--runtime-profile", str(profile_path), "--engine-source", str(args.engine), str(run), "--",
                   sys.executable, "-m", "torch.distributed.run", "--nproc-per-node=4",
                   f"--master-port={29651 + index}", script, "--prepared", str(args.prepared),
                   "--fixtures", str(fixtures), "--samples", "6", *extra]
        report.update(status=f"running {name}")
        save()
        with run.with_name(run.name + "-launch.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT)
            report["runs"].append(dict(stage=name, pid=child.pid, command=command, completed=False))
            save()
            code = child.wait()
        report["runs"][-1].update(exit_code=code, completed=True)
        save()
    report.update(status="micro gates finished; batch qualification still required")
    save()


if __name__ == "__main__":
    main()
