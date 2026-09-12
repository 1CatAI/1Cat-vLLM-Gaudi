# SPDX-License-Identifier: Apache-2.0
"""Compare the changed output projection against its original device contract."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_native_moe import HPU, torch
from vllm_gaudi.ops.deepseek_v41_attention import CSA2Attention
from vllm_gaudi.ops.deepseek_v41_math import quantize_activation
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard


def projection(weight, direct):
    # Exercise the maintained preparation and projection methods without
    # allocating unrelated attention caches for this local contract check.
    module = CSA2Attention.__new__(CSA2Attention)
    torch.nn.Module.__init__(module)
    module.heads, module.groups = 32, 4
    module.prepared_output, module.output_gemm_layout = True, direct
    module.weights = torch.nn.Module()
    module.weights.wo_a = torch.nn.Module()
    module.weights.wo_a.register_buffer("weight", weight, False)
    module.prepare_output_weight()
    return module


@HPU
@pytest.mark.parametrize("pp,tp,layer", [(1, 0, 20), (0, 1, 1), (1, 1, 39)])
def test_real_output_projection_changes_and_prefill(pp, tp, layer):
    torch._dynamo.reset()
    directory = Path(os.environ["DSV41_TEST_PREPARED"])
    shard = PreparedV41Shard(directory, pp, tp)
    name = f"layers.{layer}.attn.wo_a.weight"
    spec = shard.specs[name]
    weight = shard.dense(name, "hpu") if spec["dtype"] == "F8_E4M3" else shard.tensor(name, "hpu")
    assert weight.shape == (4096, 4096)
    candidate = projection(weight, True)
    reference = projection(weight, False)
    assert candidate.weights.wo_a.weight.data_ptr() == weight.data_ptr()
    changed = torch.compile(candidate.project_output, backend="hpu_backend", fullgraph=True, dynamic=False)
    original = torch.compile(reference.project_output, backend="hpu_backend", fullgraph=True, dynamic=False)
    records = []
    for step, count in enumerate((1, 1, 1, 6, 1)):
        torch.manual_seed(4120 + step)
        value = torch.randn(count, 4, 4096).bfloat16()
        value[:, 0] *= 0.25
        value[:, 1] *= 2
        value[0, 2, ::16] = 0
        value[0, 3, 1::16] = -0.
        value = quantize_activation(value.to("hpu"))
        expected = original(value).cpu()
        actual = changed(value).cpu()
        different = int((actual.view(torch.int16) != expected.view(torch.int16)).sum())
        records.append(
            dict(step=step,
                 count=count,
                 different_bf16_values=different,
                 max_absolute_difference=float((actual.float() - expected.float()).abs().max())))
        output = Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"output-layout-pp{pp}-tp{tp}-layer{layer}.json"
        output.write_text(json.dumps(records, indent=2) + "\n")
        if different:
            torch.save(dict(input=value.cpu(), expected=expected, actual=actual), output.with_suffix(".pt"))
        assert different == 0, records[-1]


def test_output_gemm_layout_requires_prepared_output(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_OUTPUT_GEMM_LAYOUT", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREPARED_OUTPUT", "0")
    with pytest.raises(ValueError, match="requires prepared output"):
        CSA2Attention(SimpleNamespace(), dict(num_hidden_layers=40), 0, None, None, None, "cpu")
