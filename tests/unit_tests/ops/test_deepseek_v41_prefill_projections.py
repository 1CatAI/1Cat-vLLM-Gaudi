# SPDX-License-Identifier: Apache-2.0
"""Keep prompt-only projection compilation out of ordinary decode batches."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_attention import CSA2Attention
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention
from vllm_gaudi.ops.deepseek_v41_prefill_regions import validate_prefill_region_config


@pytest.mark.parametrize("cls", [CSA2Attention, PagedCSA2Attention])
@pytest.mark.parametrize("tokens,decode", [(1, False), (6, False), (1, True), (8, True), (64, True)])
def test_query_region_uses_execution_phase_not_decode_batch_size(monkeypatch, cls, tokens, decode):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_Q_PROJECTION", "1")
    projected = torch.zeros(tokens, 16384, dtype=torch.bfloat16)
    marker = object()
    seen = []

    def linear(value, weight):
        seen.append(weight)
        return projected

    owner = SimpleNamespace(weights=SimpleNamespace(wq_b=marker),
                            q_scale_rope=False,
                            heads=32,
                            linear=linear,
                            _rope=lambda x, p: x)
    value = torch.zeros(tokens, 1280, dtype=torch.bfloat16)
    result = cls.project_query(owner, value, torch.arange(tokens, dtype=torch.int32), decode=decode)
    assert result.shape == (tokens, 32, 512)
    assert seen == [marker]


@pytest.mark.parametrize("tokens,prefill", [(1, True), (6, True), (8, False), (64, False)])
def test_output_region_does_not_replace_decode_or_small_tail(monkeypatch, tokens, prefill):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_OUTPUT_PROJECTION", "1")
    seen = []
    result = object()
    owner = SimpleNamespace(layer=0,
                            groups=4,
                            heads=32,
                            _rope=lambda x, p, inverse: x,
                            project_output=lambda x: x,
                            project_output_consumer=lambda x: x,
                            reduce=lambda x, ready_outputs: (seen.append(ready_outputs), result)[1],
                            _finish_projected_output=lambda x, ready: (seen.append(ready), result)[1])
    value = torch.zeros(tokens, 32, 512, dtype=torch.bfloat16)
    ready = (torch.ones(1), )
    observed = PagedCSA2Attention._finish_output(owner,
                                                 value,
                                                 torch.arange(tokens, dtype=torch.int32),
                                                 ready,
                                                 prefill=prefill)
    assert observed is result
    assert seen == [ready]


@pytest.mark.parametrize("name,dependencies", [
    ("Q", ("REGIONS", "NATIVE_ROPE", "PREFILL_ROPE", "ATTN_DENSE_FP8")),
    ("OUTPUT", ("REGIONS", "WO_A_FP8", "WOA_OUTPUT_ROUNDTRIP", "PREFILL_VECTOR_QUANT", "ATTN_DENSE_FP8")),
])
def test_selected_projection_requires_complete_contract(monkeypatch, name, dependencies):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_Q_PROJECTION", str(int(name == "Q")))
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_OUTPUT_PROJECTION", str(int(name == "OUTPUT")))
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_MHC_INPUT", "0")
    names = ["VLLM_HPU_DSV41_" + ("PREFILL_REGIONS" if dep == "REGIONS" else dep) for dep in dependencies]
    for variable in names:
        monkeypatch.setenv(variable, "1")
    validate_prefill_region_config()
    for variable in names:
        monkeypatch.setenv(variable, "0")
        with pytest.raises(ValueError, match="Compiled prefill"):
            validate_prefill_region_config()
        monkeypatch.setenv(variable, "1")
