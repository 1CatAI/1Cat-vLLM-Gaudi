# SPDX-License-Identifier: Apache-2.0
"""Actual layer24-39 C6 -> official sampling/MTP -> next embedding native A/B."""
import os
import statistics
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm_gaudi.models.deepseek_v41_program import PreparedDraft, PreparedInput
from vllm_gaudi.ops.deepseek_v41_draft_replay import NativeDraftProtocol
from vllm_gaudi.ops.deepseek_v41_joint_sampled_replay import JoinedTargetDraftProtocol
from vllm_gaudi.ops.deepseek_v41_round_inputs import DeviceRoundInputs, advance_round
from vllm_gaudi.ops.deepseek_v41_round_repair import SampledRoundRepairFrame
from vllm_gaudi.ops.deepseek_v41_verify import STATUS
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut


def qualify(args, program, target, raw, source_state, selection_name, sources, report, save):
    from vllm.distributed import get_tp_group
    from vllm_gaudi.ops.tp2_prepared_plan import _native_entries, prepared_group_stats

    report.update(projection_multiplier=1, projected_layer_count=16,
                  scope="real16 final Target layers, official C5/verify protocol and next embedding",
                  hypothesis="one native Target+sampled replay instead of two; all math/peer kernels unchanged",
                  qualified_gain_ms=0, formal_gain_ms=0)
    required = {"history_context_qualified", "request_context_qualified"}
    if any(not all(data.get(name) for name in required) for data in raw):
        raise ValueError("Actual request history and complete decoder fixtures are mandatory")
    for name in ("VLLM_HPU_DSV41_DSPARK_WEIGHTED_SPARSE_BINS", "VLLM_HPU_DSV41_DSPARK_WEIGHTED_STATIC_MASK"):
        if os.getenv(name, "0") != "0":
            raise ValueError("Join comparison excludes unqualified sampling candidates")
    program.draft = PreparedDraft(program, mxfp4_bf16_lut(torch.device("hpu")), "hpu")
    draft = program.draft
    embedding = PreparedInput(program.weights.embed, program.tp_rank, program.reduce)
    cases, state_banks, draft_banks = [], [], []
    for data in raw:
        source = data["groups"][args.group - 1]
        target_inputs = tuple(v.to("hpu") for v in
                              (source["residual"], source["pre"], data["positions"], data["ids"]))
        sampled = tuple(data[name].to("hpu") for name in
                        ("proposal", "sampling_parameters", "sampling_seed", "sampling_counter", "sampling_offsets"))
        control = data["control"][:7].contiguous().to("hpu")
        proposed = data["ids"][1:].contiguous().to("hpu")
        cases.append((target_inputs, proposed, control, *sampled))
        active = {}
        for name in data["decoder_state_files"]:
            if name.startswith("shared.sources.") and name.split(".")[2] not in sources:
                continue
            if not (name.startswith("shared.") or
                    any(name.startswith(f"layers.{i}.") for i in range(24, 40))):
                continue
            if name == "shared.candidate_pool" and program.shared.candidate_pool is None:
                continue
            source_name = name
            destination = selection_name if name.startswith("shared.topk.") else name
            parts = destination.split(".")
            if parts[0] == "layers":
                parts[1] = str(int(parts[1])-24)
            name = ".".join(parts)
            owner, _, leaf = name.rpartition(".")
            if hasattr(program.get_submodule(owner), leaf):
                active[name] = source_state(data, source_name).to("hpu")
        state_banks.append(active)
        draft_banks.append(tuple(value.to("hpu") for value in data["draft_swa"]))
    counters = tuple(row[6].clone() for row in cases)
    initial_positions = tuple(row[0][2].clone() for row in cases)
    initial_ids = tuple(row[0][3].clone() for row in cases)
    resident_cursor = tuple(tuple(data[name].to("hpu") for name in
                                   ("control", "cursor_history", "engram_histories")) for data in raw)
    separate_cursor = args.candidate in ("deep_queue", "round_input_publication")
    cursor = DeviceRoundInputs(
        torch.empty_like(cases[0][0][3]) if separate_cursor else cases[0][0][3],
        torch.empty_like(cases[0][0][2]) if separate_cursor else cases[0][0][2],
        frames=4 if args.candidate == "deep_queue" else 2)
    cursor.controls[0].copy_(raw[0]["control"].to("hpu"))
    cursor.history.copy_(raw[0]["cursor_history"].to("hpu"))
    engram = SimpleNamespace(histories=raw[0]["engram_histories"].to("hpu"))

    def reset(index):
        cases[index][0][2].copy_(initial_positions[index])
        cases[index][0][3].copy_(initial_ids[index])
        cursor.ids.copy_(initial_ids[index])
        cursor.positions.copy_(initial_positions[index])
        for name, value in state_banks[index].items():
            owner, _, leaf = name.rpartition(".")
            getattr(program.get_submodule(owner), leaf).copy_(value)
        for layer, value in zip(draft.layers, draft_banks[index], strict=True):
            layer.attention.swa.copy_(value)
        cases[index][6].copy_(counters[index])
        cursor.controls[0].copy_(resident_cursor[index][0])
        cursor.history.copy_(resident_cursor[index][1])
        engram.histories.copy_(resident_cursor[index][2])

    def target_inputs(index):
        return (*cases[index][0], ())

    if args.candidate == "deep_queue":
        from deepseek_v41_round_queue_gate import qualify as qualify_queue

        return qualify_queue(args, program, target, cases, reset, cursor, engram, embedding, report, save)

    if args.candidate == "round_input_publication":
        from deepseek_v41_round_publication_gate import qualify as qualify_publication

        return qualify_publication(args, program, target, cases, reset, cursor, engram, embedding, report, save)

    protocol = repair = joined = None
    try:
        reset(0)
        for _ in range(3):
            reset(0)
            target(*target_inputs(0))
            torch.hpu.synchronize()
        if target not in _native_entries:
            raise RuntimeError("Real16 Target did not capture the production native stage")
        reset(0)
        hidden, _, auxiliary = target(*target_inputs(0))
        _, proposed, control, *sampling = cases[0]
        prototype = hidden, proposed, control, auxiliary, cases[0][0][2], *sampling
        frame = SampledRoundRepairFrame(program, cursor, engram, prototype, 0)
        protocol = NativeDraftProtocol(draft, generation=1, sampled=True, repair_frame=frame,
                                       full=True, full_main=True, known_stochastic=True)
        protocol.prepare(*prototype)
        repair = NativeDraftProtocol(draft, generation=1, sampled=True, repair_frame=frame,
                                     full=True, known_stochastic=True)
        repair.prepare_repair(*prototype, journal_positions=cases[0][0][2])
        joined = JoinedTargetDraftProtocol(target, protocol)

        def joined_inputs(index):
            target_args, proposed, control, *sampling = cases[index]
            return *target_args, (), proposed, control, *sampling

        reset(0)
        joined.prepare(*joined_inputs(0))
        report["native_capture"] = dict(target_groups=target.adapter.groups,
                                        joined_groups=joined.adapter.groups,
                                        joined_collectives=joined.adapter.collectives)
        report["journal_bytes"] = frame.journal.bytes

        def executor_info():
            result = {}
            for name, owner in (("target", target), ("protocol", protocol), ("joined", joined)):
                graph, bindings = _native_entries[owner][:2]
                result[name] = dict(joint_info=list(graph.joint_info()),
                                    segments=graph.segment_count(), collectives=graph.collective_count(),
                                    commands=graph.captured_command_count(),
                                    input_copies=bindings.input_copies,
                                    dynamic_inputs=len(bindings.bindings), state_tensors=len(bindings.state_tensors))
            return result

        report["executors_before"] = executor_info()
        save()

        def execute(arm, index):
            target_args, proposed, control, *sampling = cases[index]
            if arm:
                return joined(*target_args, (), proposed, control, *sampling)
            values = target(*target_args, ())
            result = protocol(values[0], proposed, control, values[2], target_args[2], *sampling)
            return *values, *result

        def next_input(record, control):
            advance_round(record, control, engram.histories, cursor.ids, cursor.positions,
                          cursor.controls[1], cursor.history)
            return embedding(cursor.ids.long())

        compiled_consumer = torch.compile(next_input, backend="hpu_backend", fullgraph=True, dynamic=False)

        def consume(values, index):
            return compiled_consumer(values[3], cases[index][2])

        fallback = set()
        for index in range(len(cases)):
            results, mutations = [], []
            statuses = []
            for arm in (0, 1):
                reset(index)
                values = execute(arm, index)
                status = int(values[3][STATUS].cpu())
                statuses.append(status)
                if status:
                    sampled = repair(*frame.payload)
                    values = (*values[:3], *sampled)
                next_input = consume(values, index)
                result = tuple(v.cpu().clone() for v in (*values, *next_input))
                # Compare precisely the Target state rows actually written,
                # not multi-GB full allocations or unrelated prefix capacity.
                frame.journal(initial_positions[index])
                mutations.append(tuple(value.cpu().clone() for value in (
                    *(layer.attention.swa for layer in draft.layers),
                    *(getattr(frame.journal, f"saved_{i}") for i in range(len(frame.journal.specs))),
                    cursor.ids, cursor.positions, cursor.controls[1], cursor.history)))
                results.append(result)
            if statuses[0] != statuses[1]:
                raise AssertionError("Join changed the coverage certificate")
            if statuses[0]:
                fallback.add(index)
            for a, b in zip(*results, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for a, b in zip(*mutations, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            report["checks"].append(dict(case=index, outputs_and_swa_exact=True,
                                          tokens_rng_probabilities_next_embedding_and_target_writes_exact=True,
                                          full_repair=bool(statuses[0])))
            save()
        report["fallback_cases"] = sorted(fallback)
        for arm in (0, 1):
            for index in range(len(cases)):
                reset(index)
                result = execute(arm, index)
                if index in fallback:
                    result = (*result[:3], *repair(*frame.payload))
                consume(result, index)
                torch.hpu.synchronize()
        report["timer_boundary"] = (
            "both arms queued behind the same real16 Target producer; resident GPU reset; "
            "no frontend H2D or drain between primer and start marker")
        for iteration in range(3):
            for arm in (0, 1):
                elapsed = []
                dispatch = []
                for sample in range(args.samples):
                    index = sample % len(cases)
                    reset(index)
                    torch.hpu.synchronize()
                    dist.barrier(group=get_tp_group().cpu_group)
                    start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    # Serving has one-round lookahead. Measure behind the
                    # actual native producer, rather than an empty queue that
                    # charges Python binding latency as device computation.
                    target(*target_inputs(index))
                    reset(index)
                    start.record()
                    host_start = time.perf_counter()
                    result = execute(arm, index)
                    if index in fallback:
                        result = (*result[:3], *repair(*frame.payload))
                    consume(result, index)
                    dispatch.append((time.perf_counter()-host_start)*1000)
                    end.record()
                    end.synchronize()
                    elapsed.append(start.elapsed_time(end))
                report["rounds"].append(dict(iteration=iteration, arm=arm, device_ms=elapsed,
                                              median_device_ms=statistics.median(elapsed),
                                              host_dispatch_ms=dispatch,
                                              median_host_dispatch_ms=statistics.median(dispatch)))
                save()
        paired = [report["rounds"][i]["median_device_ms"] - report["rounds"][i+1]["median_device_ms"]
                  for i in (0, 2, 4)]
        report["executors_after"] = executor_info()
        report.update(status="component_passed", paired_saved_ms=paired,
                      micro_qualified=all(x>0 for x in paired) and statistics.median(paired)>=.3,
                      native_stats=prepared_group_stats())
        save()
    finally:
        for plan in (joined, repair, protocol):
            if plan is not None:
                plan.close()
