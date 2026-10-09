# SPDX-License-Identifier: Apache-2.0
"""Validate sampling-owner completion and token-weighted DSpark accounting."""
import pytest

from tools.report_deepseek_v41_rounds import reconcile, summarize


def ledgers(tp):
    row = dict(request_id="sample",
               generation=1,
               target_count=6,
               proposed_count=5,
               committed=3,
               output_count=3,
               output=[11, 12, 13],
               start_ns=1_000_000,
               end_ns=21_000_000)
    workers = {
        rank:
        dict(clock="perf_counter_ns",
             tensor_parallel_size=tp,
             sampling_owner=rank >= 4 - tp,
             records=[dict(row, ring_released=rank >= 4 - tp)])
        for rank in range(4)
    }
    engine = dict(clock="perf_counter_ns", records=[dict(row, scheduler_consumed_ns=22_000_000)])
    return workers, engine


@pytest.mark.parametrize("tp", [2, 4])
def test_round_ends_after_sampling_ring_release_and_scheduler_consumption(tp):
    workers, engine = ledgers(tp)
    rows = reconcile(workers, engine, tensor_parallel_size=tp)
    assert rows[0]["full_round_ms"] == 21
    assert rows[0]["end_ns"] == 22_000_000
    workers[4 - tp]["records"][0]["ring_released"] = False
    with pytest.raises(ValueError, match="sampling-owner post-release"):
        reconcile(workers, engine, tensor_parallel_size=tp)


def test_tp4_requires_ring_completion_on_rank_zero():
    workers, engine = ledgers(4)
    workers[0]["records"][0]["ring_released"] = False
    with pytest.raises(ValueError, match="sampling-owner post-release"):
        reconcile(workers, engine, tensor_parallel_size=4)


def test_rank_disagreement_cannot_become_a_speed_result():
    workers, engine = ledgers(4)
    workers[1]["records"][0]["output"] = [99, 12, 13]
    with pytest.raises(ValueError, match="Rank disagreement"):
        reconcile(workers, engine, tensor_parallel_size=4)


def test_sampling_topology_must_match_the_report():
    workers, engine = ledgers(4)
    with pytest.raises(ValueError, match="different TP geometry"):
        reconcile(workers, engine, tensor_parallel_size=2)


def round_row(committed, cost=24):
    return dict(request_id="sample", target_count=6, proposed_count=5, committed=committed, full_round_ms=cost)


def test_histogram_and_weighted_token_cost_include_every_acceptance_prefix():
    result = summarize([round_row(count) for count in range(1, 7)], 0, c1_tpot_ms=11)["sample"]
    assert result["accepted_count_distribution"] == {str(count): 1 for count in range(6)}
    assert result["mean_committed"] == 3.5
    assert result["acceptance_rate"] == .5
    assert result["round_derived_ms_per_token"] == pytest.approx(144 / 21)
    assert result["break_even"] and result["round_derived_speed_target_pass"]
    assert result["formal_qualification"] is False


def test_component_cost_without_c1_is_never_a_speed_pass():
    result = summarize([round_row(6)], 0)["sample"]
    assert result["full_round_target_pass"] is True
    assert result["break_even"] is False
    assert result["round_derived_speed_target_pass"] is False


def test_eight_ms_alone_does_not_satisfy_twenty_five_percent_improvement():
    result = summarize([round_row(3)], 0, c1_tpot_ms=10)["sample"]
    assert result["round_derived_ms_per_token"] == 8
    assert result["break_even"] is True
    assert result["round_derived_speed_target_pass"] is False


def test_exact_break_even_is_not_a_gain():
    result = summarize([round_row(3)], 0, c1_tpot_ms=8)["sample"]
    assert result["break_even"] is False


@pytest.mark.parametrize("committed", [0, 7])
def test_anchor_and_five_draft_capacity_are_enforced(committed):
    with pytest.raises(ValueError, match="zero to five drafts"):
        summarize([round_row(committed)], 0)
