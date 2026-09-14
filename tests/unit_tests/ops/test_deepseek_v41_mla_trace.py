# SPDX-License-Identifier: Apache-2.0
"""New MLA preparation must not disappear into another trace budget."""
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def reports(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    return (importlib.import_module("report_deepseek_v41_trace"), importlib.import_module("account_deepseek_v41_trace"))


@pytest.mark.parametrize("kernel,engine", [("custom_deepseek_v41_mla_gather_gaudi2", "TPC"),
                                           ("custom_deepseek_v41_mla_softmax_gaudi2", "TPC"), ("DmaMemcpy", "DMA"),
                                           ("cast_f32_to_bf16", "TPC")])
def test_complete_mla_work_charged_to_attention(reports, kernel, engine):
    report, accounting = reports
    category, purpose = report.classify("custom_deepseek_v41_mla_mme_gaudi2/internal", kernel, [], [])
    assert accounting.group_for({"engine": engine, "category": category, "purpose": purpose}) == 1


def test_matrix_type_uses_operand_contract(reports):
    report, _ = reports
    name = "custom_deepseek_v41_mla_mme_gaudi2/gemm"
    for dtype, expected in (("bf16", "QK"), ("float32", "PV")):
        assert expected in report.classify(name, "GEMM", [{"dtype": dtype}], [])[1]
    assert "尚未关联" in report.classify(name, "GEMM", [], [])[1]


def test_norm_ownership_requires_attention_dependency(reports):
    report, accounting = reports

    def t(name, shape, dtype="float32"):
        return f"{name} | Sizes = {json.dumps(shape)} | {dtype} | sizeInBytes = 4 | "

    def node(name, op, inputs, outputs):
        attrs = {f"inputTensor:{i}": x for i, x in enumerate(inputs)}
        attrs.update({f"outputTensor:{i}": x for i, x in enumerate(outputs)})
        return {"name": name, "op": op, "attrs": attrs}

    nodes = [
        node("layer/attention/wqa", "GEMM", [t("x", [1, 5120]), t("w", [1280, 5120])],
             [t("raw", [1, 1280], "bf16")]),
        node("fused_cast_square", "fused_kernel_1", [t("raw", [1, 1280], "bf16")],
             [t("wide", [1, 1280]), t("square", [1, 1280])]),
        node("layer/attention/mean", "reduce_mean", [t("square", [1, 1280])], [t("mean", [1, 1])]),
        node("fused_norm", "fused_kernel_2", [t("mean", [1, 1]), t("wide", [1, 1280]), t("nw", [1280])],
             [t("y", [1, 1280], "bf16")]),
    ]
    assert set(report.attention_norm_owners(nodes)) == {"fused_cast_square", "layer/attention/mean", "fused_norm"}
    assert accounting.group_for({"category": "Attention", "purpose": "Q RMSNorm", "engine": "DMA"}) == 1
    nodes[0]["name"] = "layer/moe/projection"
    assert report.attention_norm_owners(nodes) == {}
