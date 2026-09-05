# SPDX-License-Identifier: Apache-2.0
"""Opt-in HPU correctness checks for the complete native activation path."""

import os

import pytest
import torch

from flashinfer_gaudi import clear_backend_policy_override, load_native_extensions, set_backend_policy, silu_and_mul
from flashinfer_gaudi._dispatch import BackendUnavailableError

pytestmark = pytest.mark.skipif(os.environ.get("FLASHINFER_GAUDI_RUN_HARDWARE_TESTS") != "1",
                                reason="explicit Gaudi2 hardware test opt-in required")


@pytest.fixture(autouse=True)
def native_policy():
    import habana_frameworks.torch  # noqa: F401

    assert load_native_extensions()
    set_backend_policy("native")
    with torch._dynamo.config.patch(cache_size_limit=32):
        yield
    clear_backend_policy_override()


def test_compiled_native_shape_reentry_and_input_updates():
    fn = torch.compile(silu_and_mul, backend="hpu_backend", fullgraph=True, dynamic=False)
    for batch in (1, 8, 32, 1):
        for width in (128, 5120, 17408):
            x = torch.randn(batch, 2 * width, dtype=torch.bfloat16).to("hpu")
            for _ in range(3):
                source = torch.randn_like(x)
                x.copy_(source)
                expected = torch.nn.functional.silu(x[:, :width]) * x[:, width:]
                actual = fn(x)
                torch.hpu.synchronize()
                torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-3, rtol=2e-2)
                torch.testing.assert_close(x.cpu(), source.cpu(), atol=0, rtol=0)


def test_native_rejects_unsupported_contracts_without_output_mutation():
    x = torch.randn(2, 256, dtype=torch.bfloat16).to("hpu")
    out = torch.full((2, 128), 3.0, dtype=torch.bfloat16, device="hpu")
    for value, kwargs in ((x, {"out": out}), (x.t(), {}), (x.float(), {}), (x[:, :128], {})):
        with pytest.raises(BackendUnavailableError):
            silu_and_mul(value, **kwargs)
    torch.hpu.synchronize()
    assert torch.equal(out.cpu(), torch.full((2, 128), 3.0, dtype=torch.bfloat16))


def test_native_uses_current_stream():
    fn = torch.compile(silu_and_mul, backend="hpu_backend", fullgraph=True, dynamic=False)
    first, second = torch.hpu.Stream(), torch.hpu.Stream()
    values = []
    for stream in (first, second):
        with torch.hpu.stream(stream):
            x = torch.randn(8, 256, dtype=torch.bfloat16).to("hpu")
            expected = torch.nn.functional.silu(x[:, :128]) * x[:, 128:]
            values.append((fn(x), expected))
    torch.hpu.synchronize()
    for actual, expected in values:
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-3, rtol=2e-2)


def test_native_special_values():
    cpu = torch.randn(1, 256, dtype=torch.bfloat16)
    cpu[0, :9] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0., -0., 12., -12., 100., -100.],
                              dtype=torch.bfloat16)
    x = cpu.to("hpu")
    expected = torch.nn.functional.silu(x[:, :128]) * x[:, 128:]
    fn = torch.compile(silu_and_mul, backend="hpu_backend", fullgraph=True, dynamic=False)
    actual = fn(x)
    torch.hpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-3, rtol=2e-2, equal_nan=True)
