# SPDX-License-Identifier: Apache-2.0
"""Real C6 head -> official p -> rejection consumer, with same-process AB3.

Run on one leased module after exporting three sampled warmup sets on all TP
ranks. Foreign logit/proposal shards are unchanged captured boundary inputs;
this measures local computation, not TP transport or complete-round latency.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--candidate", choices=("bounded", "radix", "bf16"), required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--width", type=int, choices=(64, 128, 256), default=64)
    args = parser.parse_args()
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[0]
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    import torch.nn.functional as F

    from vllm_gaudi.models.deepseek_v41_program import bf16_weight_projection
    from vllm_gaudi.ops.deepseek_v41_math import final_collapse_rms_norm
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import (
        sample_bounded_or_full_distribution, sample_full_distribution, sample_speculative_prefix,
        speculative_sampling_draws)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers

    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    files = sorted((args.fixtures / "rank0").glob("c6-*.pt"))
    if not 3 <= len(files) <= 5 or args.samples < 3:
        raise ValueError("Need 3-5 actual warmup sets and at least three device samples per arm")
    bind_worker_cpu(0)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    if args.candidate == "radix":
        torch.ops.load_library(os.environ["DSV41_UNIQUE_OPERATOR_LIBRARY"])
    shard = PreparedV41Shard(args.prepared, 0, 0)
    weight = shard.tensor("head.weight", "hpu")
    if weight.shape != (32320, 5120) or weight.dtype != torch.bfloat16:
        raise ValueError("Expected the actual TP4 checkpoint BF16 vocab shard")
    float_weight = weight.float()
    norm_weight = shard.tensor("norm.weight", "hpu")
    eps = json.loads((args.prepared / "config.json").read_text())["text_config"]["rms_norm_eps"]
    fixtures = []
    fingerprint = []
    for path in files:
        ranks = [torch.load(args.fixtures / f"rank{rank}" / path.name, weights_only=True, map_location="cpu")
                 for rank in range(4)]
        if any(r["tensor_parallel_size"] != 4 or r["rank"] != rank for rank, r in enumerate(ranks)):
            raise ValueError("Fixture rank/topology mismatch")
        if any(not torch.equal(r["positions"], ranks[0]["positions"]) for r in ranks[1:]):
            raise ValueError("Fixtures from different rounds cannot be combined")
        if any(r.get("proposal") is None for r in ranks):
            raise ValueError("Missing actual request-owned q: cannot qualify the rejection consumer")
        p = ranks[0]
        if any(not r.get("request_context_qualified") for r in ranks):
            raise ValueError("Need actual request fixtures; startup warmup is not a 16K sampling workload")
        hidden = p["hidden"]
        if hidden.shape != (6, 5120):
            raise ValueError("Need actual normalized Target C6 hidden")
        producer = p.get("head_input")
        if args.candidate == "bf16" and producer is None:
            raise ValueError("Missing actual final-collapse/norm producer inputs; cannot qualify head precision")
        if producer is not None and (producer["residual"].shape != (6, 4, 5120) or producer["pre"].shape != (6, 4)):
            raise ValueError("Final collapse/norm producer shape mismatch")
        source, mixing = (producer["residual"], producer["pre"]) if producer is not None else (hidden, None)
        foreign = torch.cat([r["logits"] for r in ranks[1:]], -1)
        q = torch.cat([r["proposal"] for r in ranks], -1)
        _, acceptance, correction, settings = speculative_sampling_draws(
            p["sampling_parameters"].clone(), p["sampling_seed"], p["sampling_counter"].clone(),
            p["sampling_offsets"])
        fixtures.append(tuple(v.to("hpu") if v is not None else None for v in
                              (source, mixing, foreign, q, p["ids"][1:], settings,
                               acceptance, correction)))
        fingerprint.append(dict(name=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def chain(residual, mixing, foreign, proposal, proposed, controls, acceptance, correction,
              *, candidate, diagnostic=False):
        hidden = final_collapse_rms_norm(residual, mixing, norm_weight, eps) if mixing is not None else residual
        logits = (bf16_weight_projection(hidden, weight) if candidate and args.candidate == "bf16"
                  else F.linear(hidden.float(), float_weight))
        full = torch.cat((logits, foreign), -1)
        if candidate and args.candidate in ("bounded", "radix"):
            ids, p = sample_bounded_or_full_distribution(full, controls, width=args.width,
                                                        radix=args.candidate == "radix")
        else:
            ids, p = sample_full_distribution(full, controls)
        output, count, valid = sample_speculative_prefix(p, proposal, proposed, acceptance, correction)
        consumer = output, count, valid, ids
        return (*consumer, logits, p) if diagnostic else consumer

    def parent(*inputs):
        return chain(*inputs, candidate=False)

    def candidate(*inputs):
        return chain(*inputs, candidate=True)

    def parent_debug(*inputs):
        return chain(*inputs, candidate=False, diagnostic=True)

    def candidate_debug(*inputs):
        return chain(*inputs, candidate=True, diagnostic=True)

    # Target replay currently hands its normalized output to these ordinary
    # hpu_backend protocol regions. Match that boundary rather than capturing
    # one conditional branch as an invariant native recipe sequence.
    compiled = {name: torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
                for name, fn in (("parent", parent), ("candidate", candidate))}
    diagnostic = {name: torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
                  for name, fn in (("parent", parent_debug), ("candidate", candidate_debug))}
    report = dict(candidate=args.candidate, width=args.width, fixtures=fingerprint, checks=[], rounds=[],
                  scope="TP4 collapse/norm/head/probability/rejection computation; unchanged transport excluded",
                  execution="production framework compiled sampled protocol", micro_qualified=False)
    with torch.inference_mode():
        for index, inputs in enumerate(fixtures):
            expected = tuple(v.cpu() for v in diagnostic["parent"](*inputs))
            actual = tuple(v.cpu() for v in diagnostic["candidate"](*inputs))
            outputs_equal = all(torch.equal(a, b) for a, b in zip(expected[:4], actual[:4], strict=True))
            report["checks"].append(dict(fixture=index, consumer_outputs_exact=outputs_equal,
                                        logits_max_abs=float((expected[4] - actual[4]).abs().max()),
                                        probability_max_abs=float((expected[5] - actual[5]).abs().max())))
            if not outputs_equal:
                (root / "sampled-head-chain.json").write_text(json.dumps(report, indent=2) + "\n")
                raise AssertionError("Changed rejection output, committed count, valid bit or sampled token")
            torch.testing.assert_close(actual[5], expected[5], rtol=2e-5, atol=2e-7)
            # Debug tensors are not outputs of the timed production consumer.
            for name, debug in (("parent", expected), ("candidate", actual)):
                values = tuple(v.cpu() for v in compiled[name](*inputs))
                if not all(torch.equal(a, b) for a, b in zip(values, debug[:4], strict=True)):
                    raise AssertionError("Removing diagnostic outputs changes the production consumer")
        bind_worker_helpers(0)
        for fn in compiled.values():
            for inputs in fixtures:
                fn(*inputs)
        torch.hpu.synchronize()
        for round_index in range(3):
            medians = {}
            for name in ("parent", "candidate"):
                device, host = [], []
                for index in range(args.samples):
                    inputs = fixtures[index % len(fixtures)]
                    begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    started = time.perf_counter_ns()
                    begin.record()
                    compiled[name](*inputs)
                    end.record()
                    end.synchronize()
                    device.append(begin.elapsed_time(end))
                    host.append((time.perf_counter_ns() - started) / 1e6)
                medians[name] = statistics.median(device)
                report["rounds"].append(dict(round=round_index, arm=name, device_ms=device,
                                             median_device_ms=medians[name], drained_host_ms=host))
            report.setdefault("paired_saved_ms", []).append(medians["parent"] - medians["candidate"])
        report["saved_ms_per_round"] = statistics.median(report["paired_saved_ms"])
        report["micro_qualified"] = all(v > 0 for v in report["paired_saved_ms"]) and (
            report["saved_ms_per_round"] >= .3)
        (root / "sampled-head-chain.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("candidate", "micro_qualified", "paired_saved_ms")}), flush=True)


if __name__ == "__main__":
    main()
