# SPDX-License-Identifier: Apache-2.0
"""Request geometry changes must not evict another function's warm executor."""
from collections import OrderedDict

import pytest
import torch

from vllm_gaudi.ops import deepseek_v41_prefill_regions as regions


def setup_cache(monkeypatch):
    compiled = []
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_REGIONS", "1")
    monkeypatch.setattr(regions, "_function_regions", OrderedDict())
    monkeypatch.setattr(regions, "_regions_frozen", False)
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


def test_long_geometry_preserves_all_short_swa_layer_contracts(monkeypatch):
    compiled = setup_cache(monkeypatch)

    @regions.prefill_function_region
    def swa(value, offset):
        return value + offset

    short = torch.ones(256, 1)
    long = torch.ones(8192, 1)
    for layer in range(40):
        swa(short, layer * 512)
    for layer in range(40):
        swa(long, layer * 512)
    short.fill_(2)
    for layer in range(40):
        assert torch.equal(swa(short, layer * 512), short + layer * 512)
    assert len(compiled) == 80


def test_geometry_and_layer_contract_caches_remain_bounded(monkeypatch):
    compiled = setup_cache(monkeypatch)

    @regions.prefill_function_region
    def function(value, offset=0):
        return value + offset

    for rows in range(1, 10):
        function(torch.zeros(rows))
    geometries = next(iter(regions._function_regions.values()))
    assert len(geometries) == 8
    function(torch.zeros(1))
    assert len(compiled) == 10
    for offset in range(65):
        function(torch.zeros(1), offset)
    assert len(geometries[1]) == 64
    count = len(compiled)
    function(torch.zeros(1), 0)
    assert len(compiled) == count + 1
    function(torch.zeros(1, dtype=torch.float64))
    assert len(geometries[1]) == 64
    regions.clear_prefill_function_regions()
    assert not regions._function_regions
    function(torch.zeros(1))
    assert len(compiled) == count + 3


@pytest.mark.parametrize("method", [False, True])
def test_frozen_serving_reuses_warm_contracts_without_compiling_or_evicting_on_miss(monkeypatch, method):
    compiled = setup_cache(monkeypatch)
    monkeypatch.setattr(regions, "_qualified_regions", None)

    if method:
        class Owner:
            @regions.prefill_region
            def invoke(self, value, offset):
                return value + offset

        invoke = Owner().invoke
    else:
        @regions.prefill_function_region
        def invoke(value, offset):
            return value + offset

    value = torch.ones(256, 1)
    assert torch.equal(invoke(value, 3), value + 3)
    regions.freeze_prefill_regions()
    for rows in range(1, 32):
        current = torch.full((rows, 1), 7.)
        assert torch.equal(invoke(current, 4), current + 4)
    # Scalar, dtype and view-offset changes cannot use the warmed contract.
    for current in (value.double(), torch.ones(257, 1)[1:]):
        assert torch.equal(invoke(current, 3), current + 3)
    value.fill_(11)
    assert torch.equal(invoke(value, 3), value + 3)
    assert len(compiled) == 1
    regions.clear_prefill_function_regions()
    invoke(torch.zeros(17, 1), 4)
    assert len(compiled) == 2
