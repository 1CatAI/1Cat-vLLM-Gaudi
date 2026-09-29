# SPDX-License-Identifier: Apache-2.0
"""Four-rank stage coverage must distinguish a valid halo from missing spans."""
import json
import pytest

from tools.analyze_deepseek_v41_prefill_event_trace import summarize


def fixture(path, layers):
    spans = []
    for index, layer in enumerate(layers):
        for name, begin, end in (("layer", index * 3 + 1, index * 3 + 4), ("attention", index * 3 + 1, index * 3 + 2),
                                 ("moe", index * 3 + 2, index * 3 + 3)):
            spans.append(
                dict(name=name,
                     layer=layer if name != "moe" else "unknown",
                     rows=16384,
                     host_start_ns=begin,
                     host_end_ns=end,
                     device_start_ms=begin,
                     device_end_ms=end))
    stop = len(layers) * 3 + 5
    spans.append(
        dict(name="transaction_chunk",
             layer=None,
             rows=16384,
             host_start_ns=0,
             host_end_ns=stop,
             device_start_ms=0,
             device_end_ms=stop))
    path.write_text(json.dumps(dict(pp_rank=0, tp_rank=0, generation=1, device_ms=stop, spans=spans)))


def test_tp4_full_stage_is_forty_layers(tmp_path):
    path = tmp_path / "full.json"
    fixture(path, list(range(40)))
    result = summarize(path, stage_layers=range(40))
    assert result["nominal_attention_span_count"] == 40 and result["python_attention_complete"]
    assert result["groups_ms"]["Routed expert MoE"] == 40


def test_prefix_only_is_accepted_only_under_explicit_halo_policy(tmp_path):
    path = tmp_path / "prefix.json"
    fixture(path, list(range(21)))
    assert not summarize(path, stage_layers=range(40))["python_attention_complete"]
    result = summarize(path, stage_layers=range(40), allow_decoder_halo=True)
    assert result["python_attention_complete"] and result["nominal_attention_span_count"] == 21
    assert result["layer_coverage"][0]["policy"] == "prefix_only"
    fixture(path, [layer for layer in range(21) if layer != 10])
    assert not summarize(path, stage_layers=range(40), allow_decoder_halo=True)["python_attention_complete"]


def test_mhc_details_partition_layer_other_without_adding_overlaps(tmp_path):
    path = tmp_path / "detail.json"
    fixture(path, [0])
    raw = json.loads(path.read_text())
    for name, begin, end in (("mhc_input", 2.5, 3.6), ("mhc_post", 3.5, 3.9)):
        raw["spans"].append(
            dict(name=name,
                 layer="unknown",
                 rows=16384,
                 device_start_ms=begin,
                 device_end_ms=end,
                 host_start_ns=begin,
                 host_end_ns=end))
    path.write_text(json.dumps(raw))
    result = summarize(path, stage_layers=[0])
    detail = result["layer_other_components_ms"]
    assert detail == pytest.approx(dict(mhc_input=.6, mhc_post=.3, layer_other_unattributed=.1))
    assert sum(detail.values()) == pytest.approx(result["groups_ms"]["Layer other"])
