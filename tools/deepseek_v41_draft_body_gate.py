# SPDX-License-Identifier: Apache-2.0
"""Real C6 prefix -> native C5 -> official p/q -> next input, in one process."""
import statistics
import time
from pathlib import Path
import os
from functools import partial

import torch

from vllm_gaudi.ops.deepseek_v41_draft_body_replay import NativeDraftBody
from vllm_gaudi.ops.deepseek_v41_verify import encode_record_wire, pack_record
from vllm_gaudi.ops.tp2_prepared_plan import prepared_group_stats


def qualify(args, draft, cases, reference, verify, draws, consume, full_probability, report, save):
    framework_head = getattr(args, "framework_head_ab", False)
    stochastic_only = getattr(args, "stochastic_only_ab", False) or framework_head
    body = None if stochastic_only else NativeDraftBody(draft, generation=1)
    verify_known = torch.compile(partial(draft.verify_sampled_prefix_full, known_stochastic=True),
                                 backend="hpu_backend", fullgraph=True, dynamic=False) if stochastic_only else None
    propose_known = torch.compile(partial(draft.draft_sampled_from_prefix, known_stochastic=True),
                                  backend="hpu_backend", fullgraph=True, dynamic=False) if stochastic_only else None
    sampler = torch.compile(
        lambda anchor, hidden, logits, controls: draft.sample_proposal_local(
            anchor, hidden, logits, controls, full=True, force_legacy=True),
        backend="hpu_backend", fullgraph=True, dynamic=False,
    )

    def inputs_for_body(anchor, positions, committed):
        return anchor.reshape(1), positions[0].int() + committed.int() + draft.offsets

    frontend = torch.compile(inputs_for_body, backend="hpu_backend", fullgraph=True, dynamic=False)

    def publish(metadata, output, committed, count, ids, enabled, status):
        record = pack_record(metadata, committed, count, output, ids, enabled, status)
        return record, encode_record_wire(record)

    publish = torch.compile(publish, backend="hpu_backend", fullgraph=True, dynamic=False)

    def reset(case):
        inputs, caches = cases[case]
        inputs[8].copy_(counters[case])
        for layer, cache in zip(draft.layers, caches, strict=True):
            layer.attention.swa.copy_(cache)

    last_draws = {}

    def candidate(inputs):
        hidden, ids, metadata, auxiliary, positions, proposal, params, seed, counter, offsets = inputs
        dc, acceptance, correction, tc = draws(params, seed, counter, offsets)
        last_draws["controls"] = dc
        if stochastic_only:
            prefix = verify_known(hidden, ids, proposal, metadata, auxiliary, positions, tc, acceptance, correction)
            record, wire, confidence, q, covered = propose_known(metadata, positions, *prefix[:6], dc)
            return record, wire, q, confidence, counter.clone(), covered
        prefix = verify(hidden, ids, proposal, metadata, auxiliary, positions, tc, acceptance, correction)
        output, committed, count, anchor, enabled, status = prefix[:6]
        first_token, draft_positions = frontend(anchor, positions, committed)
        draft_hidden, logits = body(first_token, draft_positions)
        ids, q, confidence, covered = sampler(first_token, draft_hidden, logits, dc)
        record, wire = publish(metadata, output, committed, count, ids, enabled, status)
        return record, wire, q, confidence, counter.clone(), covered

    counters = [inputs[8].clone() for inputs, _ in cases]
    try:
        if stochastic_only:
            if any(not bool((inputs[6][:, 0] > 0).all()) for inputs, _ in cases):
                raise ValueError("Stochastic specialization requires real positive request temperatures")
            report["candidate_dispatch"] = (
                "Production full p/q sampler; positive-temperature dispatch omits greedy argmax/one-hot "
                "and reuses the nucleus prefix CDF")
        else:
            reset(0)
            inputs = cases[0][0]
            dc, acceptance, correction, tc = draws(inputs[6], inputs[7], inputs[8], inputs[9])
            prefix = verify(inputs[0], inputs[1], inputs[5], inputs[2], inputs[3], inputs[4],
                            tc, acceptance, correction)
            body.prepare(*frontend(prefix[3], inputs[4], prefix[1]))
            body.require_ready()
            report["candidate_dispatch"] = (
                "Native C5 body: seven existing exchanges; production full HCCL sampler unchanged")
        if framework_head:
            report["candidate_dispatch"] = (
                "Common BF16 LM head versus FP32, production full HCCL sampled protocol; "
                "CDF specialization in both arms")
        report["correctness_scope"] = "Actual C6 prefix, C5 states, official full p/q and next embedding"
        for case, (inputs, _) in enumerate(cases):
            results, states, embeddings, producer = [], [], [], []
            for function in (reference, candidate):
                reset(case)
                values = function(inputs)
                next_embedding = consume(values[0])
                torch.hpu.synchronize()
                results.append(tuple(value.cpu() for value in values))
                states.append(tuple(layer.attention.swa.cpu() for layer in draft.layers))
                embeddings.append(tuple(value.cpu() for value in next_embedding))
                producer.append({name: getattr(draft, "sampling_" + name + "_debug").cpu()
                                 for name in ("normalized", "base_logits", "logits")})
            exact_indices = (0, 1, 4, 5)
            discrete_exact = all(torch.equal(results[0][i], results[1][i]) for i in exact_indices)
            swa_exact = all(torch.equal(a, b) for a, b in zip(*states, strict=True))
            embedding_exact = all(torch.equal(a, b) for a, b in zip(*embeddings, strict=True))
            q_error = float((results[0][2] - results[1][2]).abs().max())
            q_ok = torch.allclose(results[0][2], results[1][2], rtol=2e-5, atol=2e-7)
            confidence_error = float((results[0][3] - results[1][3]).abs().max())
            confidence_ok = torch.allclose(results[0][3], results[1][3], rtol=2e-5, atol=2e-7)
            check = dict(case=case, record_wire_rng_coverage_exact=discrete_exact,
                         swa_exact=swa_exact, next_embedding_exact=embedding_exact,
                         probability_max_abs=q_error, confidence_max_abs=confidence_error,
                         passed=discrete_exact and swa_exact and embedding_exact and q_ok and confidence_ok)
            # All ranks evaluate the same-logit full sampler together, even
            # when only one vocabulary shard contains nonzero nucleus mass.
            oracle = full_probability(draft.sampling_logits_debug, last_draws["controls"]).cpu()
            same_producer_ok = torch.allclose(results[1][2], oracle, rtol=2e-5, atol=2e-7)
            check["same_biased_logits_full_q_max_abs"] = float((results[1][2] - oracle).abs().max())
            if not q_ok:
                # Diagnose the actual head operands before changing a
                # probability tolerance. Preserve checkpoint FP32 weights.
                head = draft.output_head.weight.cpu().double()
                diagnostics = []
                for data in producer:
                    expected = data["normalized"].double() @ head.T
                    error = data["base_logits"].double() - expected
                    diagnostics.append(dict(max_abs=float(error.abs().max()),
                                            rms=float(error.square().mean().sqrt())))
                bits = [data["normalized"].view(torch.int16).int() for data in producer]
                keys = [torch.where(value < 0, -32768 - value, value) for value in bits]
                check["normalized_max_bf16_ulp"] = int((keys[0] - keys[1]).abs().max())
                check["head_error_to_actual_bf16_input_fp64"] = diagnostics
                path = Path(os.environ["DSV41_RUN_EVIDENCE"])
                # Runner already owns this output directory; do not export graphs.
                torch.save(dict(producer=producer, probabilities=[row[2] for row in results]),
                           path / f"draft-body-operands-rank{report['rank']}-case{case}.pt")
                # The established native protocol allows <=2 BF16 ULP in
                # the internal producer, while q must match that producer's
                # actual full distribution. A nucleus boundary can amplify
                # that internal rounding; do not compare two different logits
                # as if they were the same probability calculation.
                precision_ok = (check["normalized_max_bf16_ulp"] <= args.producer_bf16_max_ulp
                                and diagnostics[1]["rms"] <= 2 * diagnostics[0]["rms"] + 1e-6)
                check["internal_producer_precision_qualified"] = precision_ok
                check["passed"] = (discrete_exact and swa_exact and embedding_exact and confidence_ok
                                   and same_producer_ok and precision_ok)
            else:
                check["passed"] = check["passed"] and same_producer_ok
            report["checks"].append(check)
            save()
            import torch.distributed as dist
            from vllm.distributed import get_tp_group

            agreement = [None] * dist.get_world_size()
            dist.all_gather_object(agreement, check["passed"], group=get_tp_group().cpu_group)
            if not all(agreement):
                raise AssertionError("Native C5 differs from the actual official producer/consumer")
        report["capability_passed"] = True
        save()
        # Probability/FP64 diagnostics use a different consumer and CPU
        # working set. Warm the measured complete consumer in both arms
        # after those diagnostics, rather than count its first enqueue.
        for function in (reference, candidate):
            for _ in range(2):
                reset(0)
                consume(function(cases[0][0])[0])
                torch.hpu.synchronize()
        for iteration in range(3):
            medians = {}
            for arm, function in (("parent", reference), ("native", candidate)):
                samples, walls = [], []
                for _ in range(args.samples):
                    reset(0)
                    torch.hpu.synchronize()
                    start, stop = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    wall = time.perf_counter()
                    start.record()
                    values = function(cases[0][0])
                    consume(values[0])
                    stop.record()
                    stop.synchronize()
                    samples.append(start.elapsed_time(stop))
                    walls.append((time.perf_counter() - wall) * 1000)
                medians[arm] = statistics.median(samples)
                report["rounds"].append(dict(iteration=iteration, arm=arm, device_ms=samples,
                                             synchronized_wall_ms=walls, median_device_ms=medians[arm]))
                save()
            report.setdefault("paired_saved_ms", []).append(medians["parent"] - medians["native"])
        saved = statistics.median(report["paired_saved_ms"])
        report.update(status="passed", saved_ms_per_round=saved, native_stats=prepared_group_stats(),
                      micro_qualified=saved >= .3 and all(x > 0 for x in report["paired_saved_ms"]))
        save()
    finally:
        if body is not None:
            body.close()
