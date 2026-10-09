# SPDX-License-Identifier: Apache-2.0
"""Native real16/C5/official protocol/readback with one or two rounds retained.

Three actual fixed-prefix transactions are replayed independently. This gate
measures the scheduling opportunity, including additional retained journals;
it does not replace whole-request cursor, repair or scheduler qualification.
There is no synthetic host delay, graph join, or communication substitution.
"""
import statistics
import time

import torch
import torch.distributed as dist

from vllm_gaudi.ops.deepseek_v41_completion import resolve_device_runtime
from vllm_gaudi.ops.deepseek_v41_draft_replay import NativeDraftProtocol
from vllm_gaudi.ops.deepseek_v41_round_inputs import advance_round
from vllm_gaudi.ops.deepseek_v41_round_repair import SampledRoundRepairFrame
from vllm_gaudi.ops.deepseek_v41_verify import STATUS, _copy_commit_words


def qualify(args, program, target, cases, reset, cursor, engram, embedding, report, save):
    from vllm.distributed import get_tp_group
    from vllm_gaudi.ops.tp2_prepared_plan import _native_entries, prepared_group_stats

    report.update(projection_multiplier=1, projected_layer_count=16,
                  scope="real16 native Target, sampled C5/Markov, next embedding and native output readback",
                  hypothesis="retain two following rounds instead of one; include four frames and wider rollback",
                  limitation="independent actual fixed-prefix transactions; serving IPC/cursor progression pending",
                  qualified_gain_ms=0, formal_gain_ms=0)
    bridge, _ = resolve_device_runtime(program.tensor_parallel_size)
    plans, repairs = [[], []], [[], []]
    expected, fallback = [], set()
    scratch_control = torch.empty_like(cursor.controls[0])
    word_frames = [[tuple(torch.empty((1, 4), dtype=torch.int64, device=cursor.ids.device)
                           for _ in range(4)) for _ in range(frames)] for frames in (2, 4)]
    copy_words = torch.compile(_copy_commit_words, backend="hpu_backend", fullgraph=True, dynamic=False)

    def target_args(index):
        return (*cases[index][0], ())

    def consume_embedding(record, control):
        advance_round(record, control, engram.histories, cursor.ids, cursor.positions,
                      scratch_control, cursor.history)
        return embedding(cursor.ids.long())

    consumer = torch.compile(consume_embedding, backend="hpu_backend", fullgraph=True, dynamic=False)
    try:
        for _ in range(3):
            reset(0)
            target(*target_args(0))
            torch.hpu.synchronize()
        if target not in _native_entries:
            raise RuntimeError("Queue gate requires the production native real16 Target")
        reset(0)
        hidden, _, auxiliary = target(*target_args(0))
        _, proposed, control, *sampling = cases[0]
        prototype = hidden, proposed, control, auxiliary, cases[0][0][2], *sampling
        for value in cursor.controls:
            value.copy_(torch.cat((control, proposed)))
        for arm, frames in enumerate((2, 4)):
            for frame_index in range(frames):
                frame = SampledRoundRepairFrame(program, cursor, engram, prototype, frame_index,
                                               lookahead=arm + 1)
                plan = NativeDraftProtocol(program.draft, generation=1, sampled=True, repair_frame=frame,
                                           full=True, full_main=True, known_stochastic=True)
                plan.prepare(*prototype)
                plans[arm].append(plan)
                repair = NativeDraftProtocol(program.draft, generation=1, sampled=True, repair_frame=frame,
                                             full=True, known_stochastic=True)
                repair.prepare_repair(*prototype, journal_positions=cases[0][0][2])
                repairs[arm].append(repair)
        report["native_frames"] = [len(values) for values in plans]
        report["journal_bytes_per_frame"] = [values[0].repair_frame.journal.bytes for values in plans]
        report["recorded_hcl_commands_per_transaction"] = [
            _native_entries[target][0].captured_command_count()
            + _native_entries[values[0]][0].captured_command_count() for values in plans]
        save()

        def execute(arm, index, ordinal, *, repair):
            frame_index = ordinal % len(plans[arm])
            values = target(*target_args(index))
            _, proposed, control, *sampling = cases[index]
            result = plans[arm][frame_index](values[0], proposed, control, values[2], cases[index][0][2], *sampling)
            if repair:
                result = repairs[arm][frame_index](*plans[arm][frame_index].repair_frame.payload)
            following = consumer(result[0], control)
            return result, following

        for index in range(len(cases)):
            outputs = []
            statuses = []
            for arm in (0, 1):
                reset(index)
                result, following = execute(arm, index, 0, repair=False)
                status = int(result[0][STATUS].cpu())
                statuses.append(status)
                if status:
                    fallback.add(index)
                    result = repairs[arm][0](*plans[arm][0].repair_frame.payload)
                    following = consumer(result[0], cases[index][2])
                outputs.append(tuple(value.cpu().clone() for value in (*result, *following)))
            if statuses[0] != statuses[1]:
                raise AssertionError("Retained depth changed the coverage certificate")
            for a, b in zip(*outputs, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            expected.append(outputs[0][0].tolist())
            report["checks"].append(dict(case=index, measured_protocol_outputs_exact=True,
                                          tokens_rng_probability_next_embedding_exact=True,
                                          full_repair=bool(statuses[0]),
                                          state_contract="CPU ownership and two-round rollback tests"))
            save()
        report["fallback_cases"] = sorted(fallback)

        def enqueue(record, arm, ordinal):
            # Same four-word ABI and producer ordering as serving504.
            words = word_frames[arm][ordinal % len(word_frames[arm])]
            copy_words(record, words)
            return [bridge.copy_integer_record_to_host(word) for word in words]

        def read(copies, index):
            values = []
            for host, done in copies:
                done.synchronize()
                values.extend(host.tolist()[0])
            if values != expected[index]:
                raise AssertionError("Queued replay/readback changed an actual transaction")

        transactions = 12
        report["transactions_per_window"] = transactions
        report["timer_boundary"] = (
            "device markers enclose real16, full official native protocol, next embedding, "
            "producer-ordered output copies and host result consumption; no synthetic host work")
        for arm in (0, 1):
            for index in range(len(cases)):
                reset(index)
                for ordinal in range(len(plans[arm])):
                    result, _ = execute(arm, index, ordinal, repair=index in fallback)
                    read(enqueue(result[0], arm, ordinal), index)
                torch.hpu.synchronize()
        for iteration in range(3):
            for arm in (0, 1):
                elapsed, wall = [], []
                for sample in range(args.samples):
                    index = sample % len(cases)
                    reset(index)
                    torch.hpu.synchronize()
                    dist.barrier(group=get_tp_group().cpu_group)
                    start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    target(*target_args(index))  # identical actual device producer ahead of both arms
                    start.record()
                    host_start = time.perf_counter()
                    retained = []
                    for ordinal in range(transactions):
                        result, _ = execute(arm, index, ordinal, repair=index in fallback)
                        retained.append(enqueue(result[0], arm, ordinal))
                        if len(retained) >= arm + 2:
                            read(retained.pop(0), index)
                    for copies in retained:
                        read(copies, index)
                    end.record()
                    end.synchronize()
                    elapsed.append(start.elapsed_time(end) / transactions)
                    wall.append((time.perf_counter() - host_start) * 1000 / transactions)
                report["rounds"].append(dict(iteration=iteration, arm=arm, device_ms=elapsed,
                                              median_device_ms=statistics.median(elapsed),
                                              drained_wall_ms=wall, median_drained_wall_ms=statistics.median(wall)))
                save()
        paired = [report["rounds"][i]["median_device_ms"] - report["rounds"][i+1]["median_device_ms"]
                  for i in (0, 2, 4)]
        report.update(status="component_passed", paired_saved_ms=paired,
                      micro_qualified=all(value > 0 for value in paired) and statistics.median(paired) >= .3,
                      native_stats=prepared_group_stats())
        save()
    finally:
        for values in (*repairs, *plans):
            for plan in values:
                plan.close()
