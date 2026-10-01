# SPDX-License-Identifier: Apache-2.0
"""Request geometry changes must not evict another function's warm executor."""
from collections import OrderedDict

import torch

from vllm_gaudi.ops import deepseek_v41_prefill_regions as regions


def setup_cache(monkeypatch):
    compiled = []
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_REGIONS", "1")
    monkeypatch.setattr(regions, "_function_regions", OrderedDict())
    monkeypatch.setattr(regions, "_qualified_function_regions", None)

    def compile_entry(function, **kwargs):
        compiled.append(function.__name__)
        return function

    monkeypatch.setattr(torch, "compile", compile_entry)
    return compiled


def test_long_function_contracts_keep_short_function_warm_and_inputs_mutable(monkeypatch):
    compiled = setup_cache(monkeypatch)

    @regions.prefill_function_region
    def short(value, weight):
        return value * weight

    @regions.prefill_function_region
    def long(value):
        return value + 1

    value, weight = torch.ones(1), torch.tensor([2.])
    assert short(value, weight).item() == 2
    for rows in range(1, 65):
        long(torch.zeros(rows))
    value.fill_(3)
    weight.fill_(4)
    assert short(value, weight).item() == 12
    assert len(compiled) == 65
    assert len(regions._function_regions) == 2


def test_each_function_remains_bounded_and_preserves_layout_contract(monkeypatch):
    compiled = setup_cache(monkeypatch)

    @regions.prefill_function_region
    def function(value):
        return value + 1

    for rows in range(1, 66):
        function(torch.zeros(rows))
    assert len(next(iter(regions._function_regions.values()))) == 64
    function(torch.zeros(1))
    assert len(compiled) == 66
    function(torch.zeros(1, dtype=torch.float64))
    assert len(compiled) == 67
    assert len(next(iter(regions._function_regions.values()))) == 64
    regions.clear_prefill_function_regions()
    assert not regions._function_regions
    function(torch.zeros(1))
    assert len(compiled) == 68
