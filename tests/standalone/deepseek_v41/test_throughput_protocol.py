# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "tools/benchmark_deepseek_v41_throughput.py"
SPEC = importlib.util.spec_from_file_location("throughput_protocol", SOURCE)
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


def record(times, *, counts=None, error=None):
    counts = counts or [1] * len(times)
    return dict(started_ns=0,
                finished_ns=10_000_000_000,
                input_tokens=512,
                events=[dict(received_ns=t * 1_000_000_000, token_ids=[1] * n) for t, n in zip(times, counts)],
                usage={"completion_tokens": sum(counts)},
                error=error,
                done=True)


def test_common_window_counts_completed_intervals_not_first_tokens():
    result = protocol.score_wave([record([1, 3, 5, 7]), record([2, 4, 6, 8])])
    assert result["resident_valid"]
    assert result["common_seconds"] == 5
    assert result["common_tokens"] == 5
    assert result["http_output_tokens_s"] == .8
    assert result["input_tokens"] == 1024


def test_coalesced_sse_preserves_http_counts_but_never_makes_fake_itl():
    result = protocol.score_wave([record([1, 3, 5], counts=[1, 2, 1])])
    assert result["coalesced_events"] == 1
    assert result["output_tokens"] == 4
    assert not result["resident_valid"]
    assert result["itl_ms"] == []


def test_disjoint_requests_have_no_common_residency():
    result = protocol.score_wave([record([1, 2]), record([3, 4])])
    assert result["complete"]
    assert not result["resident_valid"]
    assert result["common_seconds"] == 0


def test_missing_tokens_and_errors_cannot_qualify():
    row = record([1, 2, 3])
    row["usage"]["completion_tokens"] = 4
    assert not protocol.score_wave([row])["token_accounting_valid"]
    assert not protocol.score_wave([row])["resident_valid"]
    assert not protocol.score_wave([record([1, 2], error="HTTP500")])["complete"]


def test_single_request_residency_matches_itl():
    result = protocol.score_wave([record([1, 2, 3, 4])])
    assert result["common_tokens"] / result["common_seconds"] == 1
    assert result["itl_ms"] == [1000, 1000, 1000]


def test_prefill_last_first_token_excludes_http_drain():
    a, b = record([2]), record([4])
    a["input_tokens"], b["input_tokens"] = 8192, 8191
    wave = protocol.score_wave([a, b])
    result = protocol.score_round([wave], 2, "prefill", 0, 30)
    assert result["valid"]
    assert result["prefill_seconds"] == 4
    assert result["http_seconds"] == 10
    assert result["input_tokens_s"] == (8192 + 8191) / 4


def test_decode_short_residency_is_invalid_even_when_requests_succeed():
    result = protocol.score_round([protocol.score_wave([record([1, 3, 5])])], 1, "decode", 0, 30)
    assert not result["valid"]
    assert result["input_tokens_s"] is None


def test_prefill_more_than_one_output_per_request_is_invalid():
    wave = protocol.score_wave([record([1, 2]), record([2, 3])])
    assert not protocol.score_round([wave], 2, "prefill", 0, 30)["valid"]


def test_http_requires_pooled_and_median_to_exceed_target():
    rounds = [
        dict(concurrency=1,
             kind="decode",
             round=i,
             valid=True,
             resident_output_tokens_s=100.,
             http_output_tokens_s=rate,
             http_seconds=seconds,
             output_tokens=rate * seconds) for i, (rate, seconds) in enumerate(((100., 1), (100., 1), (10., 100)))
    ]
    result = protocol.compare_rounds(rounds, 1)["http_output_tokens_s"]
    assert result["qualified"] and result["median"] == 100
    assert result["pooled"] == 1200 / 102
    assert not result["exceeded"]


@pytest.mark.parametrize("cached", [None, 0, 1919, 2049, True, "2048"])
def test_cached_decode_requires_reported_cache_hits_for_every_request(cached):
    rows = [record([1, 20, 40]), record([2, 21, 41])]
    for row in rows:
        row["input_tokens"] = 2048
        row["usage"]["prompt_tokens_details"] = {"cached_tokens": 1920}
    rows[1]["usage"]["prompt_tokens_details"]["cached_tokens"] = cached
    wave = protocol.score_wave(rows, min_cached_tokens=1920)
    assert not wave["resident_valid"] and not wave["cache_hit_valid"]
    assert not protocol.score_round([wave], 2, "decode", 0, 30)["valid"]


def test_cached_decode_uses_common_window_and_ignores_prefill_duration():
    a, b = record([101, 120, 140]), record([102, 121, 141])
    for row in (a, b):
        row["input_tokens"] = 2048
        row["finished_ns"] = 142_000_000_000
        row["usage"]["prompt_tokens_details"] = {"cached_tokens": 1920}
    wave = protocol.score_wave([a, b], min_cached_tokens=1920)
    result = protocol.score_round([wave], 2, "decode", 0, 30)
    assert result["valid"] and result["common_seconds"] == 38
    assert result["resident_output_tokens_s"] == 3 / 38
    assert result["http_output_tokens_s"] == 6 / 142


def test_new_scaling_targets_do_not_reuse_paper_targets_or_http_gate():
    for concurrency, target in protocol.DECODE_SCALING_TARGETS.items():
        rounds = [
            dict(concurrency=concurrency,
                 kind="decode",
                 valid=True,
                 resident_output_tokens_s=target,
                 http_output_tokens_s=1.) for _ in range(3)
        ]
        result = protocol.compare_rounds(rounds, concurrency, "cached-decode")["resident_output_tokens_s"]
        assert result["exceeded"] and result["target"] == target
        assert result["scaling_efficiency_vs_80"] == target / (80 * concurrency)


def test_cached_corpus_freezes_2k_tokens_without_requiring_prefill_cohort():
    corpus = dict(source="test",
                  revision="frozen",
                  tokenizer_sha256="test",
                  same_paper_corpus=False,
                  decode=[dict(id=str(i), token_ids=[i] * 2048) for i in range(32)])
    protocol.validate_corpus(corpus, "cached-decode")
    corpus["decode"][0]["token_ids"].pop()
    with pytest.raises(ValueError, match="Invalid decode"):
        protocol.validate_corpus(corpus, "cached-decode")


def metrics_fixture(count=0, hits=0, computed=0, small=0):
    return dict(config={
        "enable_prefix_caching": "True",
        "block_size": "128"
    },
                buckets={
                    100.: 0,
                    200.: small,
                    5000.: count
                },
                values={
                    "prefix_cache_queries_total": count * 2048,
                    "prefix_cache_hits_total": hits,
                    "request_prefill_kv_computed_tokens_count": count,
                    "request_prefill_kv_computed_tokens_sum": computed,
                    "num_preemptions_total": 0,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "finished_stop": count,
                    "finished_abort": 0,
                    "finished_error": 0
                })


def test_metrics_prove_each_request_without_fabricating_usage():
    rows = [record([1, 20, 40]), record([2, 21, 41])]
    for row in rows:
        row["input_tokens"] = 2048
    proof = protocol.prefix_metrics_proof(metrics_fixture(), metrics_fixture(2, 3840, 256, 2), rows, 1920)
    assert proof["valid"] and proof["cached_tokens_lower_bound_per_request"] == 1920
    wave = protocol.score_wave(rows, 1920, proof)
    assert wave["cache_hit_valid"] and wave["resident_valid"]
    assert wave["cached_tokens"] == [None, None]
    rows[0]["usage"]["prompt_tokens_details"] = {"cached_tokens": 0}
    assert not protocol.score_wave(rows, 1920, proof)["resident_valid"]


def test_high_total_hits_cannot_hide_a_single_cache_miss():
    rows = [dict(input_tokens=2048)] * 2
    # 2048 hits + 1792 hits has the same total as two 1920 hits.
    proof = protocol.prefix_metrics_proof(metrics_fixture(), metrics_fixture(2, 3840, 256, 1), rows, 1920)
    assert not proof["valid"]
    assert "one request" in proof["error"]


@pytest.mark.parametrize("key,value", [("prefix_cache_queries_total", 6144),
                                       ("finished_stop", 3), ("finished_abort", 1), ("num_preemptions_total", 1),
                                       ("num_requests_running", 1), ("request_prefill_kv_computed_tokens_sum", 512),
                                       ("prefix_cache_hits_total", -1)])
def test_metrics_reject_unrelated_work_or_invalid_boundaries(key, value):
    after = metrics_fixture(2, 3840, 256, 2)
    after["values"][key] = value
    proof = protocol.prefix_metrics_proof(metrics_fixture(), after, [dict(input_tokens=2048)] * 2, 1920)
    assert not proof["valid"]


def test_metrics_require_a_histogram_fine_enough_for_the_cache_claim():
    before, after = metrics_fixture(), metrics_fixture(2, 3840, 256, 2)
    before["buckets"] = {5000.: 0}
    after["buckets"] = {5000.: 2}
    assert not protocol.prefix_metrics_proof(before, after, [dict(input_tokens=2048)] * 2, 1920)["valid"]


def test_metrics_parser_reads_upstream_samples_and_rejects_multiple_engines():
    document = '''# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="model"} 3840
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{engine="0",enable_prefix_caching="True",block_size="128"} 1
# TYPE vllm:request_prefill_kv_computed_tokens histogram
vllm:request_prefill_kv_computed_tokens_bucket{engine="0",model_name="model",le="200"} 2
# TYPE vllm:api_server_requests_total counter
vllm:api_server_requests_total{model_name="model"} 2
'''
    snapshot = protocol.prefix_metrics_snapshot(document, "model")
    assert snapshot["values"]["prefix_cache_hits_total"] == 3840
    assert snapshot["buckets"][200.] == 2
    with pytest.raises(ValueError, match="one isolated engine"):
        protocol.prefix_metrics_snapshot(document + document.replace('engine="0"', 'engine="1"'), "model")
