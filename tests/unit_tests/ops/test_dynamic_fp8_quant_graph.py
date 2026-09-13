# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import patch

import pytest
import torch

from vllm_gaudi.extension.ops import dynamic_quant

pytestmark = pytest.mark.skipif(os.environ.get("FLASHINFER_GAUDI_TEST_HPU") != "1",
                                reason="Set FLASHINFER_GAUDI_TEST_HPU=1 when a test HPU is available")


@pytest.mark.parametrize("rows", [1, 8, 32])
@pytest.mark.parametrize("width", [256, 5120, 6144, 17408])
def test_compiled_gated_cguid_quant_preserves_values(monkeypatch, rows, width):
    from habana_frameworks.torch.dynamo.compile_backend import config as hpu_config

    monkeypatch.setenv("VLLM_HPU_CGUID_DYNAMIC_QUANT", "1")
    monkeypatch.setenv("VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS", "32")
    generator = torch.Generator().manual_seed(37)
    data = torch.randn(rows, width, generator=generator).to(torch.bfloat16).to("hpu")
    # sigmoid(0) is exactly representable: this checks graph composition, not
    # differences in fused versus unfused sigmoid approximation.
    gate = torch.zeros_like(data)

    def candidate(x, g):
        return dynamic_quant(x * torch.sigmoid(g))

    with torch.inference_mode(), patch.object(hpu_config, "use_eager_fallback", False):
        expected_quant, expected_scale = candidate(data, gate)
        expected_quant, expected_scale = expected_quant.cpu(), expected_scale.cpu()
        compiled = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
        for factor in (1.0, 0.125, 4.0):
            x = data * factor
            reference_quant, reference_scale = candidate(x, gate)
            actual_quant, actual_scale = compiled(x, gate)
            torch.testing.assert_close(actual_scale.cpu(), reference_scale.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(actual_quant.cpu().float(), reference_quant.cpu().float(), rtol=0, atol=0)
        assert torch.isfinite(expected_scale).all()
        assert bool((expected_quant.float() < 0).any())
    torch._dynamo.reset()


@pytest.mark.parametrize("rows", [1, 8, 32])
@pytest.mark.parametrize("value", [0.0, 1e-8, -1e-8])
def test_compiled_gated_cguid_quant_zero_and_tiny_rows(monkeypatch, rows, value):
    from habana_frameworks.torch.dynamo.compile_backend import config as hpu_config

    monkeypatch.setenv("VLLM_HPU_CGUID_DYNAMIC_QUANT", "1")
    monkeypatch.setenv("VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS", "32")
    data = torch.full((rows, 6144), value, dtype=torch.bfloat16, device="hpu")
    gate = torch.zeros_like(data)

    def candidate(x, g):
        return dynamic_quant(x * torch.sigmoid(g))

    with torch.inference_mode(), patch.object(hpu_config, "use_eager_fallback", False):
        expected_quant, expected_scale = candidate(data, gate)
        compiled = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
        actual_quant, actual_scale = compiled(data, gate)
        torch.testing.assert_close(actual_scale.cpu(), expected_scale.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(actual_quant.cpu().float(), expected_quant.cpu().float(), rtol=0, atol=0)
        assert torch.isfinite(actual_scale.cpu()).all()
        assert (actual_scale.cpu() > 0).all()
        if value == 0:
            assert torch.count_nonzero(actual_quant.cpu().float()) == 0
    torch._dynamo.reset()
