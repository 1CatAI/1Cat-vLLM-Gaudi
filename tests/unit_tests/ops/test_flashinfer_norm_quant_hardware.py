# SPDX-License-Identifier: Apache-2.0
"""Opt-in whole-operation norm/quant checks, independent of performance gates."""

import os

import pytest
import torch

from flashinfer_gaudi import (
    clear_backend_policy_override,
    fused_add_rmsnorm_quant,
    load_native_extensions,
    set_backend_policy,
)
from flashinfer_gaudi.norm import _reference

pytestmark = pytest.mark.skipif(os.environ.get("FLASHINFER_GAUDI_RUN_HARDWARE_TESTS") != "1",
                                reason="explicit Gaudi2 hardware test opt-in required")


@pytest.fixture(autouse=True)
def native_policy():
    import habana_frameworks.torch  # noqa: F401
    from habana_frameworks.torch.dynamo.compile_backend import config

    assert load_native_extensions()
    set_backend_policy("native")
    saved_fallback = config.use_eager_fallback
    config.use_eager_fallback = False
    try:
        with torch._dynamo.config.patch(cache_size_limit=128):
            yield
    finally:
        config.use_eager_fallback = saved_fallback
        clear_backend_policy_override()


def cpu_inputs(batch, width, amplitude=1.):
    generator = torch.Generator().manual_seed(911 + batch + width)
    x = torch.randn(batch, width, generator=generator, dtype=torch.bfloat16)
    residual = torch.randn(batch, width, generator=generator, dtype=torch.bfloat16)
    weight = (torch.randn(width, generator=generator) * .1 + 1.).bfloat16()
    return (x.float() * amplitude).bfloat16(), (residual.float() * amplitude).bfloat16(), weight


def check_outputs(outputs, source, epsilon=1e-6, mode="bf16_reciprocal"):
    q, scale, normed, residual = (value.cpu() for value in outputs)
    x, r, weight = source
    expected_sum = (x.float() + r.float()).bfloat16()
    weighted = (expected_sum * weight).double()
    expected_norm = (weighted / (expected_sum.double().square().mean(-1, keepdim=True) + epsilon).sqrt()).bfloat16()
    torch.testing.assert_close(residual, expected_sum, rtol=0, atol=0)
    torch.testing.assert_close(normed, expected_norm, rtol=.02, atol=.002)
    # Quantization must implement its stated BF16 rounding boundaries exactly,
    # independent of tolerance for the normalization reduction above.
    inverse_range = 0.004180908203125 if mode == "bf16_reciprocal" else 1. / 240.
    expected_scale = ((normed.abs().amax(-1, keepdim=True) + 1e-8).float() * inverse_range).bfloat16()
    expected_q = (normed * expected_scale.reciprocal()).float().clamp(-240, 240).to(torch.float8_e4m3fn)
    torch.testing.assert_close(scale, expected_scale.float(), rtol=0, atol=0)
    torch.testing.assert_close(q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["bf16_reciprocal", "fp32_divide"])
def test_compiled_native_reentry_mutation_alias_and_rounding(mode):

    def run(x, residual, weight):
        return fused_add_rmsnorm_quant(x, residual, weight, scale_mode=mode)

    compiled = torch.compile(run, backend="hpu_backend", fullgraph=True, dynamic=False)
    for batch, width in ((1, 5120), (8, 384), (8, 8192), (8, 8320), (32, 17408), (1, 5120)):
        for amplitude in (0., 1e-5, .25, 1., 8.):
            source = cpu_inputs(batch, width, amplitude)
            inputs = tuple(value.to("hpu") for value in source)
            outputs = compiled(*inputs)
            torch.hpu.synchronize()
            check_outputs(outputs, source, mode=mode)
            pointers = {value.data_ptr() for value in inputs}
            assert len({value.data_ptr() for value in outputs}) == 4
            assert not pointers.intersection(value.data_ptr() for value in outputs)
            for value, expected in zip(inputs, source):
                torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)
    # Read-only input aliases are valid. Reuse the compiled recipe after writes.
    x = torch.empty(8, 384, dtype=torch.bfloat16, device="hpu")
    weight = torch.ones(384, dtype=torch.bfloat16, device="hpu")
    for scalar in (.125, -.25):
        x.fill_(scalar)
        outputs = compiled(x, x, weight)
        torch.hpu.synchronize()
        source = torch.full((8, 384), scalar, dtype=torch.bfloat16)
        check_outputs(outputs, (source, source, torch.ones(384).bfloat16()), mode=mode)


def test_reference_keeps_bf16_scale_boundary_after_compile():

    def reference(x, residual, weight):
        return _reference(x, residual, weight, 1e-6)

    compiled = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    for amplitude in (1e-5, .25, 1., 8.):
        source = cpu_inputs(1, 5120, amplitude)
        actual = compiled(*(value.to("hpu") for value in source))
        torch.hpu.synchronize()
        scale = actual[1].cpu()
        torch.testing.assert_close(scale, scale.bfloat16().float(), rtol=0, atol=0)
        check_outputs(actual, source)


def test_native_nondefault_streams_and_changed_inputs():
    compiled = torch.compile(fused_add_rmsnorm_quant, backend="hpu_backend", fullgraph=True, dynamic=False)
    results = []
    for stream, amplitude in ((torch.hpu.Stream(), .25), (torch.hpu.Stream(), 8.)):
        source = cpu_inputs(8, 384, amplitude)
        with torch.hpu.stream(stream):
            inputs = tuple(value.to("hpu") for value in source)
            results.append((compiled(*inputs), source))
    torch.hpu.synchronize()
    for outputs, source in results:
        check_outputs(outputs, source)
