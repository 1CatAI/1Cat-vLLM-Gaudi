# SPDX-License-Identifier: Apache-2.0
"""Real16 C6/protocol -> cursor/Engram staging -> TP embedding publication A/B."""
import statistics
import time

import torch
import torch.distributed as dist

from vllm_gaudi.ops.deepseek_v41_draft_replay import NativeDraftProtocol
from vllm_gaudi.ops.deepseek_v41_round_inputs import RoundInputPublication
from vllm_gaudi.ops.deepseek_v41_round_repair import SampledRoundRepairFrame
from vllm_gaudi.ops.deepseek_v41_verify import STATUS


def qualify(args, program, target, cases, reset, cursor, engram, embedding, report, save):
    from vllm.distributed import get_tp_group
    from vllm_gaudi.ops.tp2_prepared_plan import _native_entries, invalidate_prepared_group_plans

    report.update(projection_multiplier=1, projected_layer_count=16,
                  scope="real16 native Target and full official protocol through next TP embedding consumer",
                  hypothesis="capture cursor advance and Engram input publication in the native control plan",
                  limitation="Engram staging bytes checked; mapped Engram lookup and serving IPC not timed",
                  qualified_gain_ms=0, formal_gain_ms=0)
    engram.batched = True
    engram.ids = torch.empty_like(cursor.ids, dtype=torch.int32)
    engram.initial_history = torch.empty_like(cursor.history)
    cursor_next = cursor.controls[1]
    plans, repairs, frames = [], [], []
    proposal = torch.empty_like(cases[0][3])
    counter = torch.empty_like(cases[0][6])
    valid = torch.empty((1,), dtype=torch.bool, device=cursor.ids.device)

    def copy_engram(ids, history):
        engram.ids.copy_(ids.to(torch.int32))
        engram.initial_history.copy_(history)

    stage_engram = torch.compile(copy_engram, backend="hpu_backend", fullgraph=True, dynamic=False)
    consumer = torch.compile(embedding, backend="hpu_backend", fullgraph=True, dynamic=False)
    try:
        for _ in range(3):
            reset(0)
            target(*cases[0][0], ())
            torch.hpu.synchronize()
        if target not in _native_entries:
            raise RuntimeError("Publication gate requires the actual native real16 Target")
        reset(0)
        hidden, _, auxiliary = target(*cases[0][0], ())
        prototype = hidden, cases[0][1], cases[0][2], auxiliary, cases[0][0][2], *cases[0][3:]
        for arm in (0, 1):
            frame = SampledRoundRepairFrame(program, cursor, engram, prototype, 0)
            publication = RoundInputPublication(cursor, engram, 0) if arm else None
            plan = NativeDraftProtocol(program.draft, generation=1, sampled=True, repair_frame=frame,
                                       full=True, full_main=True, known_stochastic=True,
                                       input_publication=publication)
            plan.prepare(*prototype)
            repair = NativeDraftProtocol(program.draft, generation=1, sampled=True, repair_frame=frame,
                                         full=True, known_stochastic=True)
            repair.prepare_repair(*prototype, journal_positions=cases[0][0][2])
            plans.append(plan)
            repairs.append(repair)
            frames.append(frame)
        report["native_capture"] = [dict(
            arm=arm, publication=plan.input_publication is not None,
            segments=_native_entries[plan][0].segment_count(),
            collectives=_native_entries[plan][0].collective_count(),
            state_tensors=len(_native_entries[plan][1].state_tensors),
        ) for arm, plan in enumerate(plans)]
        save()

        def execute(arm, index, *, exact=False):
            target_args, proposed, control, *sampling = cases[index]
            hidden, _, auxiliary = target(*target_args, ())
            result = plans[arm](hidden, proposed, control, auxiliary, target_args[2], *sampling)
            if exact:
                result = repairs[arm](*frames[arm].payload)
            # Match request-owned p/q and RNG publication in production. No
            # graph output becomes a request buffer or escapes frame reuse.
            proposal.copy_(result[2])
            counter.copy_(result[4])
            valid.copy_(result[0][STATUS:STATUS + 1] == 0)
            if not arm or exact:
                cursor.advance(result[0], control, engram.histories, cursor.ids, cursor.positions,
                               cursor_next, cursor.history)
                stage_engram(cursor.ids, cursor.history)
            following = consumer(cursor.ids.long())
            return result, following

        fallback = set()
        for index in range(len(cases)):
            outputs, statuses = [], []
            for arm in (0, 1):
                reset(index)
                result, following = execute(arm, index)
                status = int(result[0][STATUS].cpu())
                statuses.append(status)
                if status:
                    reset(index)
                    result, following = execute(arm, index, exact=True)
                frames[arm].journal(cases[index][0][2])
                outputs.append(tuple(value.cpu().clone() for value in (
                    *result, *following, proposal, counter, valid, cursor.ids, cursor.positions,
                    cursor_next, cursor.history, engram.ids, engram.initial_history,
                    *(layer.attention.swa for layer in program.draft.layers),
                    *(getattr(frames[arm].journal, f"saved_{i}") for i in range(len(frames[arm].journal.specs))),
                )))
            if statuses[0] != statuses[1]:
                raise AssertionError("Native publication changed coverage or repair decisions")
            for baseline, candidate in zip(*outputs, strict=True):
                torch.testing.assert_close(baseline, candidate, rtol=0, atol=0)
            if statuses[0]:
                fallback.add(index)
            report["checks"].append(dict(case=index, outputs_and_swa_exact=True,
                                          p_q_rng_cursor_staging_and_next_embedding_exact=True,
                                          full_repair=bool(statuses[0])))
            save()
        report["fallback_cases"] = sorted(fallback)
        for arm in (0, 1):
            for index in range(len(cases)):
                reset(index)
                execute(arm, index, exact=index in fallback)
                torch.hpu.synchronize()
        report["timer_boundary"] = "native real16, full sampled protocol, publication and next TP embedding"
        for iteration in range(3):
            for arm in (0, 1):
                elapsed, wall = [], []
                for sample in range(args.samples):
                    index = sample % len(cases)
                    reset(index)
                    torch.hpu.synchronize()
                    dist.barrier(group=get_tp_group().cpu_group)
                    target(*cases[index][0], ())
                    reset(index)
                    start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    start.record()
                    begin = time.perf_counter()
                    execute(arm, index, exact=index in fallback)
                    end.record()
                    end.synchronize()
                    elapsed.append(start.elapsed_time(end))
                    wall.append((time.perf_counter() - begin) * 1000)
                report["rounds"].append(dict(iteration=iteration, arm=arm, device_ms=elapsed,
                                              median_device_ms=statistics.median(elapsed),
                                              drained_wall_ms=wall, median_drained_wall_ms=statistics.median(wall)))
                save()
        report["paired_saved_ms"] = [report["rounds"][i]["median_device_ms"]
                                     - report["rounds"][i+1]["median_device_ms"] for i in (0, 2, 4)]
        report["status"] = "component_passed"
        save()
    finally:
        for plan in (*plans, *repairs):
            plan.close()
        invalidate_prepared_group_plans(owner=target, reason="round_publication_gate_close")
