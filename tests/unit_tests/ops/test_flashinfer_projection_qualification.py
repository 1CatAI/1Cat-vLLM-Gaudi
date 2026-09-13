# SPDX-License-Identifier: Apache-2.0
"""Complete projection evidence cannot promote a mismatched consumer contract."""

import copy
import operator
from unittest import mock

import pytest
import torch

from tools.benchmark_flashinfer_native_projection import BASELINES, audit_fx, audit_trace, qualify_case


def _sessions():
    return [{
        "session_id": str(index),
        "pid": index + 100,
        "artifacts_unchanged": True,
        "use_eager_fallback": False,
        "sha256": {
            "kernel": "same"
        },
        "contract": {
            "scale_mode": "bf16_reciprocal"
        },
        "cases": {
            "shape": {
                "inputs_unchanged": True,
                "native_fx": [["norm", "gemm"]],
                "fixture_sha256": {
                    "input": "same"
                },
                "baseline_correctness": dict.fromkeys(BASELINES, True),
                "reentry": dict.fromkeys(BASELINES, True),
                "samples_ms": {
                    "native": [1.] * 15,
                    **{
                        label: [1.2] * 15
                        for label in BASELINES
                    }
                },
                "traces": {
                    "native": {
                        "launch_events": 5,
                        "replays": 5,
                        "fx_audited": True
                    }
                } if index == 0 else {}
            }
        }
    } for index in range(3)]


@pytest.mark.parametrize("missing", [
    None, "fx", "trace", "launch", "hash", "contract", "fixture", "mutation", "fallback", "worker", "session", "pid",
    "timings"
])
def test_requires_every_process_and_unchanged_inputs_and_artifacts(missing):
    sessions = _sessions()
    errors = []
    case = sessions[1]["cases"]["shape"]
    if missing == "fx":
        case["native_fx"] = []
    elif missing == "trace":
        sessions[0]["cases"]["shape"]["traces"] = {}
    elif missing == "launch":
        sessions[0]["cases"]["shape"]["traces"]["native"]["launch_events"] = 10
    elif missing == "hash":
        sessions[1]["sha256"] = {"kernel": "changed"}
    elif missing == "contract":
        sessions[1]["contract"]["scale_mode"] = "fp32_divide"
    elif missing == "fixture":
        case["fixture_sha256"] = {"input": "different"}
    elif missing == "mutation":
        case["inputs_unchanged"] = False
    elif missing == "fallback":
        sessions[1]["use_eager_fallback"] = True
    elif missing == "worker":
        errors = [{"returncode": 1}]
    elif missing in ("session", "pid"):
        key = "session_id" if missing == "session" else "pid"
        sessions[1][key] = sessions[0][key]
    elif missing == "timings":
        case["samples_ms"]["native"] = []
    assert qualify_case(sessions, "shape", 3, errors)["qualified"] is (missing is None)


@pytest.mark.parametrize("phase", ["baseline_correctness", "reentry"])
def test_incompatible_baseline_stays_failed_without_discarding_other_evidence(phase):
    sessions = _sessions()
    sessions[2]["cases"]["shape"][phase]["vendor"] = False
    result = qualify_case(sessions, "shape", 3, [])
    assert not result["qualified"] and not result["production_promoted"]
    assert not result["baselines"]["vendor"]["qualified"]
    assert result["baselines"]["formula"]["qualified"]
    assert result["baselines"]["cguid"]["qualified"]


@pytest.mark.parametrize("violation", [None, "missing_norm", "missing_gemm", "extra_kernel", "extra_launch", "cpu_op"])
def test_trace_includes_mme_and_vendor_generated_kernel(violation):
    events = [{
        "cat": "kernel",
        "name": name
    } for name in ("flashinfer_gaudi_add_rmsnorm_quant_bf16_gaudi2", "GEMM", "fused_kernel_0x123_f32")]
    events += [{"cat": "privateuse1_runtime", "name": "Launch"} for _ in range(5)]
    if violation == "missing_norm":
        events.pop(0)
    elif violation == "missing_gemm":
        events.pop(1)
    elif violation == "extra_kernel":
        events.append({"cat": "kernel", "name": "cast_to_fp8"})
    elif violation == "extra_launch":
        events.append({"cat": "privateuse1_runtime", "name": "Launch"})
    elif violation == "cpu_op":
        events.append({"cat": "cpu_op", "name": "aten::mul"})
    if violation is None:
        assert audit_trace(events)["vendor_generated_kernels"] == ["fused_kernel_0x123_f32"]
    else:
        with pytest.raises(RuntimeError, match="trace audit failed"):
            audit_trace(events)


def _norm(*args):
    return args


def _gemm(*args):
    return args


@pytest.mark.parametrize("violation", [None, "extra_math", "wrong_scale", "wrong_output"])
def test_fx_requires_direct_native_norm_to_gemm_dataflow(violation):
    graph = torch.fx.Graph()
    x, residual, gamma, weight, weight_scale = [graph.placeholder(name) for name in ("x", "r", "g", "w", "s")]
    norm = graph.call_function(_norm, (x, residual, gamma, 1e-6, False))
    q = graph.call_function(operator.getitem, (norm, 0))
    scale = graph.call_function(operator.getitem, (norm, 1))
    summed = graph.call_function(operator.getitem, (norm, 3))
    if violation == "extra_math":
        q = graph.call_function(operator.add, (q, 1))
    gemm = graph.call_function(_gemm,
                               (q, False, weight, True, None, torch.bfloat16,
                                weight_scale if violation == "wrong_scale" else scale, weight_scale, None, False))
    graph.output((gemm, residual if violation == "wrong_output" else summed))
    module = torch.fx.GraphModule({}, graph)
    names = {_norm: "custom_op.flashinfer_gaudi_add_rmsnorm_quant", _gemm: "hpu.fp8_gemm_v2"}
    with mock.patch("tools.benchmark_flashinfer_native_projection.target_name",
                    side_effect=lambda node: names.get(node.target, str(node.target))):
        if violation is None:
            assert len(audit_fx(module)) == 5
        else:
            with pytest.raises(RuntimeError):
                audit_fx(module)


def test_fewer_processes_and_missing_cases_fail_closed():
    sessions = _sessions()
    assert not qualify_case(sessions[:2], "shape", 3, [])["qualified"]
    assert not qualify_case([], "shape", 3, [])["qualified"]
    other = copy.deepcopy(sessions)
    other[2]["cases"] = {}
    assert not qualify_case(other, "shape", 3, [])["qualified"]
