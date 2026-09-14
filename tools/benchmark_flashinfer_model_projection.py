# SPDX-License-Identifier: Apache-2.0
"""Isolated compiled-model A/B/A screen for the experimental projection fusion.

Warm complete requests are measured separately from compilation and profiling.
This small screen is not the general LLM production acceptance benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def warm_model(llm, prompts, sampling, rounds):
    if rounds < 2:
        raise ValueError("At least two symmetric model warmup rounds are required")
    for _ in range(rounds):
        llm.generate(prompts, sampling, use_tqdm=False)


def warm_timings_stable(workers):
    if len(workers) != 3:
        return False
    for worker_report in workers:
        rates = [row["output_tokens_per_s"] for row in worker_report.get("rounds", ())]
        if len(rates) < 3 or not all(math.isfinite(rate) and rate > 0 for rate in rates):
            return False
        median = statistics.median(rates)
        if min(rates) < median * .9 or max(rates) > median * 1.1:
            return False
    return True


class ProjectionTrialWorkerExtension:

    def configure_projection_trial(self, enabled, output_dir):
        import torch
        from habana_frameworks.torch.dynamo.compile_backend import passes
        from vllm_gaudi.ops.flashinfer_projection_fusion import register_projection_fusion_pass

        directory = Path(output_dir)
        if enabled:
            register_projection_fusion_pass(((8, 5120, 34816), ))
        dumps = []

        def dump(ctx):
            if len(dumps) >= 2:
                return False
            if any(
                    str(node.target).startswith("hpu.fp8_gemm_v2")
                    and node.meta.get("output_shapes") == [torch.Size((8, 34816))]
                    for node in ctx.graph_module.graph.nodes):
                path = directory / f"model-graph-{len(dumps)}.txt"
                path.write_text(ctx.graph_module.code)
                dumps.append(path.name)
            return False

        passes.register_pass_at_optimization_pass(dump, passes.OptimizationPassPlacement.PRE_PLACEMENT)
        torch._dynamo.reset()
        return {"enabled": enabled, "model_class": type(self.get_model()).__name__}

    def projection_trial_stats(self):
        from vllm_gaudi.ops.flashinfer_projection_fusion import projection_fusion_stats
        return projection_fusion_stats()

    def start_projection_trial_trace(self):
        import torch
        from vllm.forward_context import get_forward_context

        self._projection_trial_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU])
        self._projection_trial_trace_started = False
        self._projection_trial_trace_stopped = False
        self._projection_trial_steps = 0
        self._projection_trial_active_step = False

        def before(module, args, kwargs):
            metadata = get_forward_context().attn_metadata
            inputs = kwargs.get("input_ids", args[0] if args else None)
            active = (metadata is not None and not bool(getattr(metadata, "is_prompt", False))
                      and isinstance(inputs, torch.Tensor) and inputs.numel() == 8
                      and not self._projection_trial_trace_stopped)
            self._projection_trial_active_step = active
            if active and not self._projection_trial_trace_started:
                self._projection_trial_profiler.start()
                self._projection_trial_trace_started = True

        def after(module, args, output):
            if self._projection_trial_active_step:
                self._projection_trial_steps += 1
                if self._projection_trial_steps == 5:
                    torch.hpu.synchronize()
                    self._projection_trial_profiler.stop()
                    self._projection_trial_trace_stopped = True

        model = self.get_model()
        self._projection_trial_hooks = (model.register_forward_pre_hook(before, with_kwargs=True),
                                        model.register_forward_hook(after))

    def stop_projection_trial_trace(self, output):
        import torch
        torch.hpu.synchronize()
        for handle in self._projection_trial_hooks:
            handle.remove()
        del self._projection_trial_hooks
        if not self._projection_trial_trace_stopped:
            if self._projection_trial_trace_started:
                self._projection_trial_profiler.stop()
            raise RuntimeError("Trace did not observe five B=8 decode model forwards")
        self._projection_trial_profiler.export_chrome_trace(output)
        del self._projection_trial_profiler
        events = json.loads(Path(output).read_text())["traceEvents"]
        counts = Counter(event["name"] for event in events if event.get("cat") == "kernel")
        return {
            "kernels": dict(counts),
            "scope": "five B=8 decode model forwards; profiler excluded from request timings",
            "native_projection_kernel_events": counts["flashinfer_gaudi_add_rmsnorm_quant_bf16_gaudi2"]
        }


def worker(args):
    import habana_frameworks.torch  # noqa: F401
    from vllm import LLM, SamplingParams

    root = Path(__file__).resolve().parents[1]

    def fingerprints():
        paths = (Path(__file__).resolve(), root / "vllm_gaudi/ops/flashinfer_projection_fusion.py",
                 root / "vllm_gaudi/ops/hpu_layernorm.py", root / "vllm_gaudi/extension/ops.py",
                 root / "vllm_gaudi/models/qwen3_next.py", root / "vllm_gaudi/models/qwen3_5.py",
                 root / "flashinfer_gaudi/lib/libflashinfer_gaudi_kernels.so", root / "flashinfer_gaudi/norm.py")
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    source_sha256 = fingerprints()
    args.output.mkdir(parents=True, exist_ok=False)
    llm = LLM(model=args.model,
              tensor_parallel_size=1,
              dtype="bfloat16",
              max_model_len=4096,
              max_num_seqs=32,
              max_num_batched_tokens=2048,
              gpu_memory_utilization=.70,
              enforce_eager=False,
              enable_prefix_caching=False,
              generation_config="vllm",
              seed=31,
              worker_extension_cls="tools.benchmark_flashinfer_model_projection.ProjectionTrialWorkerExtension")
    enabled = args.mode == "candidate"
    configuration = llm.collective_rpc("configure_projection_trial", args=(enabled, str(args.output.resolve())))
    tokenizer = llm.get_tokenizer()
    questions = ("What is the capital of France?", "What is 17 multiplied by 23?", "Explain a residual connection.",
                 "Write Python that reverses a list.", "Name three planets.", "Explain the water cycle.",
                 "Translate hello into French.", "Why is normalization useful in a neural network?")
    prompts = []
    for question in questions:
        text = tokenizer.apply_chat_template([{
            "role": "user",
            "content": question
        }],
                                             tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
        prompts.append({"prompt_token_ids": tokenizer.encode(text, add_special_tokens=False)})
    sampling = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
    warm_model(llm, prompts, sampling, args.warmup_rounds)
    report = {
        "mode": args.mode,
        "warmup_rounds": args.warmup_rounds,
        "source_sha256": source_sha256,
        "whole_model_eager_fallback_policy": os.environ.get("PT_HPU_USE_EAGER_FALLBACK", "1"),
        "device_module": os.environ.get("HLS_MODULE_ID"),
        "configuration": configuration,
        "model_config_sha256": hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest(),
        "measurement": "warm complete requests including prefill, decode and host scheduling",
        "not_general_acceptance": True,
        "rounds": [],
        "production_promoted": False
    }
    for index in range(args.rounds):
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - started
        token_ids = [list(output.outputs[0].token_ids) for output in outputs]
        report["rounds"].append({
            "index": index,
            "elapsed_s": elapsed,
            "token_ids": token_ids,
            "output_tokens_per_s": sum(map(len, token_ids)) / elapsed
        })
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["median_output_tokens_per_s"] = statistics.median(row["output_tokens_per_s"] for row in report["rounds"])
    report["fusion_stats"] = llm.collective_rpc("projection_trial_stats")
    if args.trace:
        llm.collective_rpc("start_projection_trial_trace")
        llm.generate(prompts, sampling, use_tqdm=False)
        report["trace"] = llm.collective_rpc("stop_projection_trial_trace",
                                             args=(str((args.output / "trace.json").resolve()), ))
    report["route_hit"] = bool(enabled and any(stats["compiled_matches"] > 0 for stats in report["fusion_stats"])
                               and (not args.trace or any(item["native_projection_kernel_events"] > 0
                                                          for item in report["trace"])))
    report["sources_unchanged"] = fingerprints() == source_sha256
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("rounds", "trace")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--mode", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("Require positive rounds")
    if args.warmup_rounds < 2:
        parser.error("Require at least two symmetric model warmup rounds")
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"production_promoted": False, "general_acceptance": False, "workers": [], "worker_errors": []}
    for index, mode in enumerate(("baseline", "candidate", "baseline")):
        directory = args.output / f"{index}-{mode}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()), "--worker", "--mode", mode, "--model", args.model, "--output",
            str(directory), "--rounds",
            str(args.rounds), "--warmup-rounds",
            str(args.warmup_rounds)
        ]
        if args.trace and mode == "candidate":
            command.append("--trace")
        print(f"Starting model process {index}: {mode}", flush=True)
        with (args.output / f"{index}-{mode}.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            report["worker_errors"].append({"mode": mode, "index": index, "returncode": completed.returncode})
        path = directory / "report.json"
        if path.is_file():
            report["workers"].append(json.loads(path.read_text()))
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["worker_errors"] and len(report["workers"]) == 3:
        first, candidate, last = report["workers"]
        report["source_control_valid"] = (all(item.get("sources_unchanged") for item in report["workers"]) and len(
            {json.dumps(item["source_sha256"], sort_keys=True)
             for item in report["workers"]}) == 1)
        baseline = statistics.median(
            [row["output_tokens_per_s"] for worker_report in (first, last) for row in worker_report["rounds"]])
        report["candidate_speedup"] = candidate["median_output_tokens_per_s"] / baseline
        report["baseline_return_ratio"] = last["median_output_tokens_per_s"] / first["median_output_tokens_per_s"]
        report["token_ids_equal"] = all(row["token_ids"] == first["rounds"][0]["token_ids"]
                                        for worker_report in report["workers"] for row in worker_report["rounds"])
        report["route_hit"] = candidate["route_hit"]
        report["warm_timings_stable"] = warm_timings_stable(report["workers"])
        report["screen_valid"] = bool(report["source_control_valid"] and report["token_ids_equal"]
                                      and report["route_hit"] and report["warm_timings_stable"]
                                      and .95 <= report["baseline_return_ratio"] <= 1.05)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "workers"}, indent=2))


if __name__ == "__main__":
    main()
