# SPDX-License-Identifier: Apache-2.0
"""Exact block scale boundaries and native linear recipe contracts."""
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer_gaudi import block_fp8_dequant, block_fp8_linear, clear_backend_policy_override, set_backend_policy
from flashinfer_gaudi._bridge import load_bridge_adapter
from flashinfer_gaudi._dispatch import BackendUnavailableError
from flashinfer_gaudi.block_scaled import _dequant_reference, _linear_reference


@pytest.fixture(autouse=True)
def policy():
    clear_backend_policy_override()
    with torch._dynamo.config.patch(cache_size_limit=64):
        yield
    clear_backend_policy_override()


def inputs(m=8, n=256, k=384):
    generator = torch.Generator().manual_seed(m + n + k)
    x = torch.randn(m, k, generator=generator).to(torch.bfloat16)
    weight = torch.randn(n, k, generator=generator).clamp(-240, 240).to(torch.float8_e4m3fn)
    scale = torch.rand(n // 128, k // 128, generator=generator) * .37 + .031
    return x, weight, scale


@pytest.mark.parametrize("policy", ["native", "bridge", "public"])
def test_strict_cpu_never_resolves_or_decomposes(policy):
    set_backend_policy(policy)
    x, weight, scale = inputs()
    with (mock.patch("flashinfer_gaudi.block_scaled._dequant_reference") as
          dequant, mock.patch("flashinfer_gaudi.block_scaled._linear_reference") as
          linear, mock.patch("flashinfer_gaudi.block_scaled.block_fp8_dequant_op") as dequant_op,
          mock.patch("flashinfer_gaudi.block_scaled.block_fp8_linear_op") as linear_op):
        for fn, args in ((block_fp8_dequant, (weight, scale)), (block_fp8_linear, (x, weight, scale))):
            with pytest.raises(BackendUnavailableError):
                fn(*args)
        for patched in (dequant, linear, dequant_op, linear_op):
            patched.assert_not_called()


@pytest.mark.parametrize("kind", ["dequant", "linear"])
@pytest.mark.parametrize("policy", ["native", "bridge"])
def test_native_missing_or_failing_op_never_falls_back(kind, policy):
    set_backend_policy(policy)

    def fake(shape, dtype):
        return SimpleNamespace(shape=shape,
                               ndim=2,
                               dtype=dtype,
                               device=torch.device("hpu"),
                               requires_grad=False,
                               is_contiguous=lambda: True)

    weight, scale = fake((128, 128), torch.float8_e4m3fn), fake((1, 1), torch.float32)
    args = (weight, scale) if kind == "dequant" else (fake((1, 128), torch.bfloat16), weight, scale)
    fn = block_fp8_dequant if kind == "dequant" else block_fp8_linear
    with mock.patch(f"flashinfer_gaudi.block_scaled._{kind}_reference") as reference:
        with (mock.patch(f"flashinfer_gaudi.block_scaled.block_fp8_{kind}_op",
                         return_value=None), pytest.raises(BackendUnavailableError, match="no fallback")):
            fn(*args)
        with (mock.patch(f"flashinfer_gaudi.block_scaled.block_fp8_{kind}_op",
                         return_value=mock.Mock(side_effect=RuntimeError("native failure"))),
              pytest.raises(RuntimeError, match="native failure")):
            fn(*args)
        reference.assert_not_called()


@pytest.mark.parametrize("kind", ["weight_dtype", "scale_dtype", "weight_stride", "scale_shape", "unaligned", "grad"])
def test_invalid_weight_contract(kind):
    _, weight, scale = inputs()
    if kind == "weight_dtype":
        weight = weight.float()
    elif kind == "scale_dtype":
        scale = scale.bfloat16()
    elif kind == "weight_stride":
        weight = weight.t()
    elif kind == "scale_shape":
        scale = scale[:, :1]
    elif kind == "unaligned":
        weight = weight[:127].contiguous()
    elif kind == "grad":
        scale.requires_grad_(True)
    with pytest.raises(BackendUnavailableError):
        block_fp8_dequant(weight, scale)


def test_reference_rounds_scale_before_product_and_preserves_auto():
    x, weight, scale = inputs()
    expected = (weight.bfloat16().reshape(2, 128, 3, 128) * scale.bfloat16()[:, None, :, None]).reshape(256, 384)
    wrong = (weight.float().reshape(2, 128, 3, 128) * scale[:, None, :, None]).reshape(256, 384).bfloat16()
    assert not torch.equal(expected, wrong)
    with mock.patch("flashinfer_gaudi.block_scaled.block_fp8_linear_op", side_effect=AssertionError("native on auto")):
        torch.testing.assert_close(block_fp8_dequant(weight, scale), expected, rtol=0, atol=0)
        torch.testing.assert_close(block_fp8_linear(x, weight, scale),
                                   torch.nn.functional.linear(x, expected),
                                   rtol=0,
                                   atol=0)


def test_reference_matches_existing_vllm_block_dequant():
    from vllm_gaudi.extension.ops import dequant_block_fp8_weight_naive
    _, weight, scale = inputs()
    torch.testing.assert_close(_dequant_reference(weight, scale),
                               dequant_block_fp8_weight_naive(weight, scale, [128, 128]),
                               rtol=0,
                               atol=0)


@pytest.mark.parametrize("violation", [None, "aten::mul", "hpu::cast_from_fp8", "extra_launch", "missing_mme"])
def test_trace_audit(violation):
    from tools.benchmark_flashinfer_native_block_linear import audit_trace
    events = [{"cat": "kernel", "name": "GEMM"}, {"cat": "kernel", "name": "flashinfer_gaudi_block_fp8_dequant_gaudi2"}]
    events.extend({"cat": "privateuse1_runtime", "name": "Launch"} for _ in range(5))
    if violation is None:
        assert audit_trace(events)["launch_events"] == 5
    else:
        if violation == "missing_mme":
            events = events[1:]
        elif violation == "extra_launch":
            events.append({"cat": "privateuse1_runtime", "name": "Launch"})
        else:
            events.append({"cat": "cpu_op", "name": violation})
        with pytest.raises(RuntimeError, match="audit failed"):
            audit_trace(events)


hardware = pytest.mark.skipif(os.environ.get("FLASHINFER_GAUDI_RUN_HARDWARE_TESTS") != "1",
                              reason="explicit Gaudi2 hardware test opt-in required")


def compiled_pair(candidate, reference):
    load_bridge_adapter()
    set_backend_policy("native")
    return (torch.compile(candidate, backend="hpu_backend", fullgraph=True,
                          dynamic=False), torch.compile(reference, backend="hpu_backend", fullgraph=True,
                                                        dynamic=False))


@hardware
def test_hardware_dequant_all_fp8_codes_and_scale_boundaries():
    import habana_frameworks.torch  # noqa: F401
    candidate, baseline = compiled_pair(block_fp8_dequant, _dequant_reference)
    # Includes signed zero and special exponent encodings. Compare to the
    # established HPU cast, not the CPU interpretation of Gaudi2 FP8 specials.
    raw = torch.arange(256, dtype=torch.uint8).repeat(128, 1)
    weight = raw.view(torch.float8_e4m3fn).to("hpu")
    for values in ((.123456, .789123), (0., 1.), (-.03147, 5.121), (1e-20, 1e20)):
        scale = torch.tensor([values], dtype=torch.float32).to("hpu")
        actual, expected = candidate(weight, scale), baseline(weight, scale)
        torch.hpu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True)
    assert torch.equal(weight.cpu().view(torch.uint8), raw)


@hardware
def test_hardware_linear_shapes_updates_and_reentry():
    import habana_frameworks.torch  # noqa: F401
    candidate, baseline = compiled_pair(block_fp8_linear, _linear_reference)
    dequant, dequant_ref = compiled_pair(block_fp8_dequant, _dequant_reference)
    for m, n, k in ((1, 128, 128), (8, 256, 384), (32, 384, 256), (8, 5120, 512), (1, 128, 128)):
        cpu = inputs(m, n, k)
        x, weight, scale = (t.to("hpu") for t in cpu)
        for magnitude in (1., .25, 8.):
            scale.copy_(cpu[2] * magnitude)
            expected, actual = baseline(x, weight, scale), candidate(x, weight, scale)
            dw, rw = dequant(weight, scale), dequant_ref(weight, scale)
            torch.hpu.synchronize()
            torch.testing.assert_close(dw.cpu(), rw.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=.02, atol=.002)
            torch.testing.assert_close(x.cpu(), cpu[0], rtol=0, atol=0)
            assert torch.equal(weight.cpu().view(torch.uint8), cpu[1].view(torch.uint8))
            torch.testing.assert_close(scale.cpu(), cpu[2] * magnitude, rtol=0, atol=0)
        # Reuse the same Tensor objects with new x/weight contents, not just
        # new scales. A recipe must not treat runtime weights as frozen.
        x.copy_(torch.full_like(cpu[0], .25))
        changed_weight = torch.full((n, k), .5).to(torch.float8_e4m3fn)
        weight.copy_(changed_weight)
        actual, expected = candidate(x, weight, scale), baseline(x, weight, scale)
        torch.hpu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=.02, atol=.002)
        assert torch.equal(weight.cpu().view(torch.uint8), changed_weight.view(torch.uint8))


@hardware
def test_hardware_streams_and_direct_dispatch_rejection():
    import habana_frameworks.torch  # noqa: F401
    candidate, baseline = compiled_pair(block_fp8_linear, _linear_reference)
    values = []
    for stream in (torch.hpu.Stream(), torch.hpu.Stream()):
        with torch.hpu.stream(stream):
            x, weight, scale = (t.to("hpu") for t in inputs())
            values.append((candidate(x, weight, scale), baseline(x, weight, scale)))
    torch.hpu.synchronize()
    for actual, expected in values:
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=.02, atol=.002)
    for bad_x, bad_weight, bad_scale in ((x.float(), weight, scale), (x, weight.t(), scale), (x, weight,
                                                                                              scale.bfloat16())):
        with pytest.raises(RuntimeError):
            torch.ops.custom_op.flashinfer_gaudi_block_fp8_linear(bad_x, bad_weight, bad_scale)
