# SPDX-License-Identifier: Apache-2.0
"""New MLA preparation must not disappear into another trace budget."""
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def reports(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    return (importlib.import_module("report_deepseek_v41_trace"),
            importlib.import_module("account_deepseek_v41_trace"))


@pytest.mark.parametrize("kernel,engine", [("custom_deepseek_v41_mla_gather_gaudi2", "TPC"),
                                          ("custom_deepseek_v41_mla_softmax_gaudi2", "TPC"),
                                          ("DmaMemcpy", "DMA"), ("cast_f32_to_bf16", "TPC")])
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
