# SPDX-License-Identifier: Apache-2.0
"""One official EOS request and its companion decode trace on an owned service."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def machine_load(modules, cpu_pool=()):
    from run_deepseek_v41 import cpuset

    telemetry_error = None
    try:
        body = subprocess.check_output(
            ["hl-smi", "-Q", "module_id,memory.used,utilization.aip", "--format=csv,noheader"], text=True, timeout=30
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        body, telemetry_error = "", str(error)
    cards = []
    for line in body.splitlines():
        module, memory, utilization = line.split(",")
        if not module.strip().isdigit():
            cards.append(dict(module=None, memory_mib=None, utilization=None, status="unavailable"))
            continue
        cards.append(
            dict(module=int(module), memory_mib=int(memory.split()[0]), utilization=int(utilization.split()[0]))
        )
    overlapping = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            environment = dict(
                item.split(b"=", 1) for item in (process / "environ").read_bytes().split(b"\0") if b"=" in item
            )
            visible = environment.get(b"HABANA_VISIBLE_MODULES", b"").decode()
            if not visible or set(map(int, visible.split(","))) <= modules:
                continue
            assigned = set()
            for key in (
                b"VLLM_HPU_DSV4_WORKER_CPUS",
                b"VLLM_HPU_DSV4_WORKER_HELPER_CPUS",
                b"VLLM_HPU_DSV41_ENGINE_CPUS",
                b"VLLM_HPU_DSV41_API_CPUS",
            ):
                binding = environment.get(key, b"").decode().replace(";", ",")
                if binding.strip():
                    assigned.update(cpuset(binding))
            common = assigned & set(cpu_pool)
            if common:
                overlapping.append(
                    dict(pid=int(process.name), modules=visible, cpus=sorted(common), assigned_cpus=sorted(assigned))
                )
        except (OSError, ValueError):
            continue
    return dict(
        time_ns=time.time_ns(),
        cards=cards,
        foreign_cpu_tasks=overlapping,
        foreign_cpu_overlap=bool(overlapping),
        telemetry_error=telemetry_error,
        foreign_device_load=any(
            c["module"] is not None and c["module"] not in modules and c["memory_mib"] > 1024 for c in cards
        ),
    )


def relocate_owned_cpus(process, load):
    """Move only owned threads to free physical cores in the same NUMA nodes."""
    from run_deepseek_v41 import cpuset

    original = set(process["initial_process_cpus"])
    if not load["foreign_cpu_overlap"]:
        return dict(status="original NUMA binding available", cpus=sorted(original), changed_threads=0)
    busy = set().union(*(set(item["assigned_cpus"]) for item in load["foreign_cpu_tasks"]))
    for cpu in tuple(busy):
        busy.update(cpuset(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text()))
    mapping = {}
    for node in {item["numa"] for item in process["modules"]}:
        local = cpuset(Path(f"/sys/devices/system/node/node{node}/cpulist").read_text())
        old = sorted(local & original)
        available = sorted(
            cpu
            for cpu in local - busy
            if cpu == min(cpuset(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text()))
        )
        if len(available) < len(old):
            return dict(
                status="insufficient free NUMA cores; measuring with disclosed overlap",
                cpus=sorted(original),
                changed_threads=0,
                foreign_cpu_tasks=load["foreign_cpu_tasks"],
            )
        retained = set(old) & set(available)
        mapping.update({cpu: cpu for cpu in retained})
        mapping.update(zip(sorted(set(old) - retained), sorted(set(available) - retained), strict=False))
    changed = 0
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            if os.getpgid(int(candidate.name)) != process["pgid"]:
                continue
            for thread in (candidate / "task").iterdir():
                tid = int(thread.name)
                before = os.sched_getaffinity(tid)
                after = {mapping.get(cpu, cpu) for cpu in before}
                if after != before:
                    os.sched_setaffinity(tid, after)
                    changed += 1
        except ProcessLookupError:
            continue
    return dict(
        status="owned threads rebound within NUMA",
        cpus=sorted(set(mapping.values())),
        mapping=mapping,
        changed_threads=changed,
        foreign_cpu_tasks=load["foreign_cpu_tasks"],
    )


def summarize_completed_run(run):
    """Reuse post-release ledgers; never substitute trace time for EOS speed."""
    from run_deepseek_v41_dspark_batch import summarize_rounds

    workers = [json.loads((run / f"round-timing/rank{rank}.json").read_text()) for rank in range(4)]
    results = []
    failed_requests = []
    for directory in sorted(run.glob("official-16k-eos-*")):
        if not directory.is_dir() or not (directory / "result.json").is_file():
            continue
        result = json.loads((directory / "result.json").read_text())
        if result.get("status") != "passed":
            failed_requests.append(dict(name=directory.name, error=result.get("error")))
            continue
        with (directory / "response.sse").open() as stream:
            response = next(json.loads(line[6:]) for line in stream if line.startswith("data: {") and '"id"' in line)
        request_id = response["id"]
        summaries = [summarize_rounds(worker["records"], request_id, result["tokens"]) for worker in workers]
        fields = ("rounds", "accepted_count_histogram", "acceptance_rate", "useful_tokens_per_round")
        if any(any(summary[field] != summaries[0][field] for field in fields) for summary in summaries[1:]):
            raise ValueError("Sampling-owner round ledgers disagree")
        results.append(
            dict(
                name=directory.name,
                request_id=request_id,
                profiled=result["profile"] is not None,
                **summaries[0],
                rank_mean_round_ms=[s["mean_round_ms"] for s in summaries],
                # One-ahead submission can overlap transaction
                # start/end spans. Retain those spans, and report
                # actual request wall cost per completed C6 round
                # separately rather than summing queue residence.
                complete_round_wall_ms=(result["decode_ms_per_token"] * summaries[0]["useful_tokens_per_round"]),
                effective_tpot_ms=result["decode_ms_per_token"],
                prefill_tokens_per_s=result["prefill_tokens_per_s"],
                eos_proof=result["eos_proof"],
            )
        )
    audits = [json.loads((run / f"traces/rank{rank}-native-shutdown.json").read_text())["v41"] for rank in range(4)]
    repairs = [audit.get("sampled_exact_repairs", 0) for audit in audits]
    if len(set(repairs)) != 1:
        raise ValueError("Ranks disagree about the official probability repair count")
    rounds = sum(result["rounds"] for result in results)
    if any(audit["requests"] != len(results) + len(failed_requests) for audit in audits):
        raise ValueError("The aggregate certificate audit includes another request")
    environment = json.loads((run / "process.json").read_text())["environment"]
    global_bounded = environment.get("VLLM_HPU_DSV41_DSPARK_GLOBAL_BOUNDED_SAMPLING") == "1"
    exact_full = environment.get("VLLM_HPU_DSV41_DSPARK_NATIVE_FULL_MAIN") == "1" and not global_bounded
    bounded = environment.get("VLLM_HPU_DSV41_DSPARK_NATIVE_SAMPLED_PROTOCOL") == "1" and not exact_full
    summary = dict(
        requests=results,
        failed_requests=failed_requests,
        all_request_exact_repairs=repairs[0],
        # One certificate failure repairs a round of six p and five
        # q rows. This bounds the row failure fraction conservatively;
        # it is not a fabricated count of failing individual rows.
        sampling_mode=("exact_full_native" if exact_full else "bounded_native" if bounded else "segmented_full"),
        native_bounded_sampling=bounded,
        uncovered_row_fraction_upper_bound=repairs[0] / rounds if bounded and rounds else None,
        certificate_scope=(
            "not applicable: exact full distribution"
            if exact_full
            else "published native C6 rounds across both EOS requests; aggregate bound"
        ),
        formal_qualified=False,
        goal_met=False,
    )
    (run / "sampled-batch-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--modules", default="0,4,5,1")
    parser.add_argument("--keep-service", action="store_true")
    parser.add_argument("--trace-only", action="store_true",
                        help="Recover just the companion trace on identical frozen code")
    parser.add_argument(
        "--diagnostic", action="store_true", help="Allow other-card workloads; do not qualify a formal TPOT"
    )
    parser.add_argument("--require-machine-lease", action="store_true",
                        help="Wait for the fixed wrapper to reserve all non-serving modules")
    args = parser.parse_args()
    modules = {int(value) for value in args.modules.split(",")}
    workspace = Path(__file__).resolve().parents[1]
    record = args.run.with_name(args.run.name + "-client.json")
    report = dict(
        client_pid=os.getpid(),
        launcher_pid=args.launcher_pid,
        status="waiting for full warmup",
        run=str(args.run),
        requests=[],
        achieved_gain_ms=None,
        diagnostic=args.diagnostic,
        formal_qualified=False,
    )

    def save():
        record.write_text(json.dumps(report, indent=2) + "\n")

    save()
    process = None
    try:
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            manifest = args.run / "process.json"
            if manifest.is_file():
                process = json.loads(manifest.read_text())
                if process.get("exit_code") is not None:
                    raise RuntimeError("Owned service exited before full warmup")
                owned_stat = Path(f"/proc/{process['pid']}/stat")
                try:
                    owned_state = owned_stat.read_text().rsplit(") ", 1)[1].split()[0]
                except FileNotFoundError as error:
                    raise RuntimeError("Owned service disappeared during warmup; inspect run.log") from error
                if owned_state == "Z":
                    raise RuntimeError("Owned service exited during warmup; inspect run.log")
                log_path = args.run / "run.log"
                if log_path.exists():
                    with log_path.open("rb") as stream:
                        stream.seek(max(0, log_path.stat().st_size - 65536))
                        tail = stream.read()
                    if b"WorkerProc hit an exception" in tail:
                        raise RuntimeError("Owned worker failed during warmup; inspect run.log")
                try:
                    with urllib.request.urlopen(args.url + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
            try:
                launcher_state = Path(f"/proc/{args.launcher_pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
            except FileNotFoundError as error:
                raise RuntimeError("Owned launcher exited before readiness") from error
            if launcher_state in ("Z", "X"):
                raise RuntimeError("Owned launcher exited before readiness; inspect launcher log")
            time.sleep(15)
        else:
            raise TimeoutError("Full warmup exceeded two hours")
        report["server_pid"] = process["pid"]
        report["process_manifest"] = str(args.run / "process.json")
        unprofiled_name = "official-16k-eos-diagnostic" if args.diagnostic else "official-16k-eos-formal"
        requests = [("official-16k-eos-trace", True)]
        if not args.trace_only:
            requests.insert(0, (unprofiled_name, False))
        for name, profiled in requests:
            if args.require_machine_lease and not profiled:
                report.update(status='waiting for formal machine lease')
                save()
                deadline = time.monotonic() + 7200
                lease_path = args.run / "formal-machine-lease.json"
                while True:
                    lease = json.loads(lease_path.read_text()) if lease_path.exists() else None
                    if lease is not None and lease["status"] == "held":
                        if (lease["owner_pid"], lease["owner_pgid"]) != (process["pid"], process["pgid"]):
                            raise RuntimeError("Machine lease belongs to another service")
                        if not Path(f"/proc/{lease['guard_pid']}").exists():
                            raise RuntimeError("Machine lease guard exited before formal timing")
                        report["machine_lease"] = lease
                        break
                    if lease is not None and lease["status"] == "released":
                        raise RuntimeError("Machine lease released before formal timing")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("No exclusive machine lease; formal request was not measured")
                    time.sleep(15)
            current = report.get("cpu_binding", {}).get("cpus", process["initial_process_cpus"])
            load = machine_load(modules, current)
            if not profiled and not args.diagnostic and (load["foreign_device_load"] or load["telemetry_error"]):
                raise RuntimeError("Formal timing requires idle non-serving cards and available telemetry")
            if load["foreign_cpu_overlap"]:
                report["cpu_binding"] = relocate_owned_cpus(dict(process, initial_process_cpus=current), load)
            else:
                report.setdefault("cpu_binding", dict(status="original NUMA binding available", cpus=current))
            report.update(status=name, machine_load=load)
            save()
            (args.run / f"{name}-load.json").write_text(json.dumps(load, indent=2) + "\n")
            command = [
                sys.executable,
                "tools/qualify_deepseek_v41_request.py",
                str(args.request),
                str(args.run / name),
                "--url",
                args.url,
            ]
            if profiled:
                command += ["--profile", "decode", "--decode-trace-skip-tokens", "128", "--decode-trace-tokens", "64"]
            with (args.run / f"{name}-client.log").open("xb") as log:
                subprocess.run(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((args.run / name / "result.json").read_text())
            if profiled:
                # Freeze all four raw acquisitions before another session can
                # replace profiler files. Offline analysis uses this manifest.
                with (args.run / f"{name}-archive.log").open("xb") as log:
                    subprocess.run([sys.executable, "tools/archive_deepseek_v41_capture.py",
                                    str(args.run), str(args.run / name)], cwd=workspace,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            tokens = json.loads((args.run / name / "token_ids.json").read_text())
            baseline = json.loads(args.baseline_output.read_text())
            first_diff = next((i for i, (a, b) in enumerate(zip(tokens, baseline)) if a != b), None)
            if first_diff is None and len(tokens) != len(baseline):
                first_diff = min(len(tokens), len(baseline))
            report["requests"].append(
                dict(
                    name=name,
                    profiled=profiled,
                    tokens=len(tokens),
                    eos_proof=result["eos_proof"],
                    effective_tpot_ms=result["decode_ms_per_token"],
                    prefill_tokens_per_s=result["prefill_tokens_per_s"],
                    baseline_token_exact=tokens == baseline,
                    first_different_token=first_diff,
                )
            )
            save()
        report.update(status="EOS requests and decode trace complete; round accounting and promotion pending")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        if process is not None and not args.keep_service:
            # Only the server whose startup manifest belongs to this run.
            pid = process.get("pid")
            try:
                if pid and os.getpgid(pid) == process["pgid"] == pid:
                    os.killpg(pid, signal.SIGTERM)
                    report["owned_retirement_requested"] = True
                    save()
                    from run_deepseek_v41 import retire_process_group

                    report["owned_retirement"] = retire_process_group(
                        pid, grace_seconds=45, terminate_seconds=10)
                    save()
            except ProcessLookupError:
                pass
            if report["requests"]:
                # Shutdown publishes worker ledgers. Finish accounting in
                # this driver so a completed request cannot be stranded with
                # only its SSE timing available.
                required = [
                    args.run / f"{folder}/rank{rank}{suffix}.json"
                    for rank in range(4)
                    for folder, suffix in (("round-timing", ""), ("traces", "-native-shutdown"))
                ]
                deadline = time.monotonic() + 180
                while not all(path.is_file() for path in required) and time.monotonic() < deadline:
                    time.sleep(2)
                try:
                    report["round_summary"] = summarize_completed_run(args.run)
                    report["status"] = "EOS, companion trace and round accounting complete; promotion pending"
                except (OSError, ValueError, KeyError) as error:
                    report["round_accounting_error"] = f"{type(error).__name__}: {error}"
                save()


if __name__ == "__main__":
    main()
