# SPDX-License-Identifier: Apache-2.0
"""Strict policy and scale contracts for the native gated-quantization candidate."""

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer_gaudi import (
    clear_backend_policy_override,
    load_native_extensions,
    set_backend_policy,
    silu_and_mul_quant,
)
from flashinfer_gaudi._dispatch import BackendUnavailableError
from flashinfer_gaudi._bridge import load_bridge_adapter
from flashinfer_gaudi.quantization import _silu_and_mul_quant_reference


@pytest.fixture(autouse=True)
def reset_policy():
    clear_backend_policy_override()
    with torch._dynamo.config.patch(cache_size_limit=64):
        yield
    clear_backend_policy_override()


@pytest.mark.parametrize("policy", ["native", "public", "bridge"])
def test_strict_rejects_cpu_without_reference(policy):
    set_backend_policy(policy)
    with (mock.patch("flashinfer_gaudi.quantization._silu_and_mul_quant_reference") as
          reference, pytest.raises(BackendUnavailableError)):
        silu_and_mul_quant(torch.zeros((2, 512), dtype=torch.bfloat16))
    reference.assert_not_called()


@pytest.mark.parametrize("shape", [(0, 512), (2, 0), (2, 513), (512, )])
def test_bad_shape_rejected(shape):
    with pytest.raises(ValueError):
        silu_and_mul_quant(torch.empty(shape, dtype=torch.bfloat16))


def test_reference_zero_scale_and_output_contract():
    set_backend_policy("pytorch")
    quantized, scale = silu_and_mul_quant(torch.zeros((2, 512), dtype=torch.bfloat16))
    expected = (torch.tensor(1e-8, dtype=torch.bfloat16) / 240).float()
    torch.testing.assert_close(scale, expected.expand(2, 1), atol=0, rtol=0)
    assert quantized.dtype == torch.float8_e4m3fn and quantized.shape == (2, 256)
    assert torch.count_nonzero(quantized.float()) == 0
    assert (scale > 0).all()


@pytest.mark.parametrize("policy", ["native", "bridge"])
def test_native_missing_or_failed_op_never_falls_back(policy):
    set_backend_policy(policy)
    x = SimpleNamespace(ndim=2,
                        shape=(2, 512),
                        dtype=torch.bfloat16,
                        requires_grad=False,
                        device=SimpleNamespace(type="hpu"),
                        is_contiguous=lambda: True)
    with mock.patch("flashinfer_gaudi.quantization._silu_and_mul_quant_reference") as reference:
        with (mock.patch("flashinfer_gaudi.quantization.silu_mul_quant_op",
                         return_value=None), pytest.raises(BackendUnavailableError, match="unavailable")):
            silu_and_mul_quant(x)
        with (mock.patch("flashinfer_gaudi.quantization.silu_mul_quant_op",
                         return_value=mock.Mock(side_effect=RuntimeError("native failure"))),
              pytest.raises(RuntimeError, match="native failure")):
            silu_and_mul_quant(x)
    reference.assert_not_called()


def test_public_cannot_use_private_adapter():
    set_backend_policy("public")
    x = SimpleNamespace(ndim=2, shape=(2, 512), dtype=torch.bfloat16, requires_grad=False)
    with (mock.patch("flashinfer_gaudi.quantization.silu_mul_quant_op") as
          resolver, pytest.raises(BackendUnavailableError)):
        silu_and_mul_quant(x)
    resolver.assert_not_called()


def test_unverified_registered_symbol_is_not_advertised(monkeypatch):
    from flashinfer_gaudi import _native
    monkeypatch.setattr(_native, "_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(_native, "_LOADED_PATHS", [])
    monkeypatch.setattr(_native, "_SILU_MUL_QUANT_OP", None)
    monkeypatch.setattr(_native, "_library_candidates", lambda: [])
    with mock.patch.object(torch.ops.custom_op, "flashinfer_gaudi_silu_mul_quant", create=True):
        assert _native.silu_mul_quant_op() is None


def test_verified_adapter_can_republish_after_resolver_reset(monkeypatch):
    from flashinfer_gaudi import _bridge, _native
    identity = {"artifact_id": "verified", "production_promoted": False}
    monkeypatch.setattr(_bridge, "_LOADED", identity)
    monkeypatch.setattr(_native, "_SILU_MUL_QUANT_OP", None)
    with (mock.patch.object(torch.ops.custom_op, "flashinfer_gaudi_silu_mul_quant", create=True) as
          op, mock.patch.object(torch.ops, "load_library") as loader):
        assert _bridge.load_bridge_adapter() == identity
        assert _native._SILU_MUL_QUANT_OP is op
        loader.assert_not_called()


def test_lut_free_reciprocal_bf16_scale_domain():
    bits = torch.arange(1, 0x7f80, dtype=torch.int32)
    scales = (bits << 16).view(torch.float32)
    scales = scales[(scales >= 4e-11) & (scales <= 1.5e36)]
    estimate = (0x7ef311c3 - scales.view(torch.int32)).view(torch.float32)
    for _ in range(3):
        estimate = estimate * (2. - scales * estimate)
    torch.testing.assert_close(estimate.to(torch.bfloat16), scales.reciprocal().to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("vendor", ["silu_fwd_bf16", "fused_kernel_0xA53228_38_bf16"])
def test_native_trace_vendor_silu_names(vendor):
    from tools.benchmark_flashinfer_native_quant import audit_native_kernels
    audit_native_kernels([vendor, "flashinfer_gaudi_silu_mul_quant_bf16_gaudi2"])


def test_native_trace_rejects_missing_or_extra_computation():
    from tools.benchmark_flashinfer_native_quant import audit_native_kernels
    for kernels in (["silu_fwd_bf16"], ["flashinfer_gaudi_silu_mul_quant_bf16_gaudi2"],
                    ["flashinfer_gaudi_silu_mul_quant_bf16_gaudi2", "silu_fwd_bf16", "copy"]):
        with pytest.raises(RuntimeError):
            audit_native_kernels(kernels)


@pytest.mark.parametrize("violation", [None, "aten::mul", "aten::copy_", "hpu::cast_to_fp8_v2", "extra_launch"])
def test_native_trace_recipe_and_cpu_audit(violation):
    from tools.benchmark_flashinfer_native_quant import audit_native_trace
    events = [{
        "cat": "kernel",
        "name": name
    } for name in ("silu_fwd_bf16", "flashinfer_gaudi_silu_mul_quant_bf16_gaudi2")]
    events.extend({"cat": "privateuse1_runtime", "name": "Launch"} for _ in range(5))
    if violation is None:
        assert audit_native_trace(events)["launch_events"] == 5
    else:
        events.append({
            "cat": "privateuse1_runtime",
            "name": "Launch"
        } if violation == "extra_launch" else {
            "cat": "cpu_op",
            "name": violation
        })
        with pytest.raises(RuntimeError, match="audit failed"):
            audit_native_trace(events)


def assert_quant_close(actual, expected):
    quantized, scale = (value.cpu() for value in actual)
    reference, reference_scale = (value.cpu() for value in expected)
    assert quantized.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    torch.testing.assert_close(scale, reference_scale, rtol=.02, atol=1e-8, equal_nan=True)
    torch.testing.assert_close(quantized.float() * scale,
                               reference.float() * reference_scale,
                               rtol=.08,
                               atol=.02,
                               equal_nan=True)


hardware = pytest.mark.skipif(os.environ.get("FLASHINFER_GAUDI_RUN_HARDWARE_TESTS") != "1",
                              reason="explicit Gaudi2 hardware test opt-in required")


@hardware
def test_native_quant_shapes_reentry_and_input_updates():
    import habana_frameworks.torch  # noqa: F401
    assert load_native_extensions()
    load_bridge_adapter()
    set_backend_policy("native")
    candidate = torch.compile(silu_and_mul_quant, backend="hpu_backend", fullgraph=True, dynamic=False)
    baseline = torch.compile(_silu_and_mul_quant_reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    with torch._dynamo.config.patch(cache_size_limit=64):
        for batch, width in ((1, 256), (8, 2048), (8, 2176), (32, 3584), (1, 4096), (8, 5120), (8, 8704), (32, 17408),
                             (1, 256)):
            generator = torch.Generator().manual_seed(batch + width)
            source = torch.randn((batch, 2 * width), generator=generator, dtype=torch.bfloat16)
            x = source.to("hpu")
            for magnitude in (.25, 1., 8.):
                x.copy_((source.float() * magnitude).to(torch.bfloat16))
                expected, actual = baseline(x), candidate(x)
                torch.hpu.synchronize()
                assert_quant_close(actual, expected)
                torch.testing.assert_close(x.cpu(), (source.float() * magnitude).to(torch.bfloat16), atol=0, rtol=0)


@hardware
def test_native_quant_zero_tiny_and_special_values():
    import habana_frameworks.torch  # noqa: F401
    assert load_native_extensions()
    load_bridge_adapter()
    set_backend_policy("native")
    candidate = torch.compile(silu_and_mul_quant, backend="hpu_backend", fullgraph=True, dynamic=False)
    baseline = torch.compile(_silu_and_mul_quant_reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    for value in (0., 1e-8, 1e-4):
        x = torch.full((8, 512), value, dtype=torch.bfloat16, device="hpu")
        expected, actual = baseline(x), candidate(x)
        torch.hpu.synchronize()
        assert_quant_close(actual, expected)
        assert (actual[1].cpu() > 0).all()
        if value == 0:
            torch.testing.assert_close(actual[1].cpu(), expected[1].cpu(), rtol=0, atol=0)
    cpu = torch.ones((8, 512), dtype=torch.bfloat16)
    cpu[:4, 0] = torch.tensor([float("nan"), float("inf"), -float("inf"), -100.], dtype=torch.bfloat16)
    x = cpu.to("hpu")
    expected, actual = baseline(x), candidate(x)
    torch.hpu.synchronize()
    assert_quant_close(actual, expected)


@hardware
def test_native_quant_reduction_covers_every_lane_and_final_tile():
    import habana_frameworks.torch  # noqa: F401
    assert load_native_extensions()
    load_bridge_adapter()
    set_backend_policy("native")
    candidate = torch.compile(silu_and_mul_quant, backend="hpu_backend", fullgraph=True, dynamic=False)
    baseline = torch.compile(_silu_and_mul_quant_reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    for width in (256, 2176, 17408):
        source = torch.ones((128, 2 * width), dtype=torch.bfloat16)
        source[torch.arange(128), 2 * width - 128 + torch.arange(128)] = 32
        x = source.to("hpu")
        actual, expected = candidate(x), baseline(x)
        torch.hpu.synchronize()
        assert_quant_close(actual, expected)
        torch.testing.assert_close(actual[1].cpu(), expected[1].cpu(), rtol=0, atol=0)


@hardware
def test_native_quant_streams_and_unsupported_contracts():
    import habana_frameworks.torch  # noqa: F401
    assert load_native_extensions()
    load_bridge_adapter()
    set_backend_policy("native")
    candidate = torch.compile(silu_and_mul_quant, backend="hpu_backend", fullgraph=True, dynamic=False)
    values = []
    for stream in (torch.hpu.Stream(), torch.hpu.Stream()):
        with torch.hpu.stream(stream):
            x = torch.randn((8, 512), dtype=torch.bfloat16).to("hpu")
            values.append((candidate(x), _silu_and_mul_quant_reference(x)))
    torch.hpu.synchronize()
    for actual, expected in values:
        assert_quant_close(actual, expected)
    for bad in (x.t(), x.float(), x[:, :256], torch.empty((1, 35328), dtype=torch.bfloat16, device="hpu")):
        with pytest.raises(BackendUnavailableError):
            silu_and_mul_quant(bad)
