# SPDX-License-Identifier: Apache-2.0
"""Opt-in mixed MME/TPC graph and native preallocated output tests."""

import os

import pytest
import torch

from flashinfer_gaudi._dispatch import BackendUnavailableError
from flashinfer_gaudi.gemm import GemmSiluArtifactV1, GemmSiluPlan

pytestmark = pytest.mark.skipif(os.environ.get("FLASHINFER_GAUDI_RUN_HARDWARE_TESTS") != "1",
                                reason="explicit Gaudi2 hardware test opt-in required")


def reference(x, weight):
    gate, up = (x @ weight).chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def inputs(m=8, k=128, d=128):
    generator = torch.Generator().manual_seed(m + k + d)
    x = torch.randn(m, k, generator=generator).to(torch.bfloat16).to("hpu")
    weight = (torch.randn(k, 2 * d, generator=generator) / k**.5).to(torch.bfloat16).to("hpu")
    return x, weight


def test_mixed_graph_functional_and_out_shape_reentry():
    import habana_frameworks.torch  # noqa: F401
    baseline = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    first = None
    with torch._dynamo.config.patch(cache_size_limit=32):
        for m, k, d in ((1, 128, 128), (8, 256, 256), (32, 512, 512), (1, 128, 128)):
            plan = GemmSiluPlan(m, k, d)
            x, weight = inputs(m, k, d)
            saved_x, saved_w = x.cpu(), weight.cpu()
            out = torch.empty((m, d), dtype=x.dtype, device=x.device)
            pointer = out.data_ptr()
            fn = torch.compile(plan.functional_op, backend="hpu_backend", fullgraph=True, dynamic=False)
            for _ in range(3):
                expected = baseline(x, weight)
                actual = fn(x, weight)
                eager = plan.run(x, weight)
                assert plan.run(x, weight, out=out) is out
                torch.hpu.synchronize()
                assert out.data_ptr() == pointer
                for result in (actual, eager, out):
                    torch.testing.assert_close(result.cpu(), expected.cpu(), atol=2e-3, rtol=2e-2)
            torch.testing.assert_close(x.cpu(), saved_x, atol=0, rtol=0)
            torch.testing.assert_close(weight.cpu(), saved_w, atol=0, rtol=0)
            if first is None:
                first = (plan, x, weight, out, expected.cpu())
        plan, x, weight, out, expected = first
        plan.run(x, weight, out=out)
        torch.hpu.synchronize()
        torch.testing.assert_close(out.cpu(), expected, atol=2e-3, rtol=2e-2)


def test_out_updates_and_independent_streams():
    import habana_frameworks.torch  # noqa: F401
    plan = GemmSiluPlan(8, 128, 128)
    values = []
    for stream in (torch.hpu.Stream(), torch.hpu.Stream()):
        with torch.hpu.stream(stream):
            x, weight = inputs()
            out = torch.empty((8, 128), device="hpu", dtype=torch.bfloat16)
            plan.run(x, weight, out=out)
            x.fill_(0.25)
            weight.fill_(0.125)
            expected = reference(x, weight)
            plan.run(x, weight, out=out)
            values.append((out, expected))
    torch.hpu.synchronize()
    for out, expected in values:
        torch.testing.assert_close(out.cpu(), expected.cpu(), atol=2e-3, rtol=2e-2)


def test_out_invalid_and_alias_rejected_without_mutation():
    import habana_frameworks.torch  # noqa: F401
    plan = GemmSiluPlan(8, 128, 128)
    x, weight = inputs()
    out = torch.full((8, 128), 3., device="hpu", dtype=torch.bfloat16)
    for bad_x, bad_w, bad_out in ((x.float(), weight, out), (x, weight.t(), out), (x, weight, out.t()), (x, weight, x)):
        with pytest.raises(BackendUnavailableError):
            plan.run(bad_x, bad_w, out=bad_out)
    # The C++ dispatcher also rejects direct callers before submitting a recipe.
    with pytest.raises(RuntimeError, match="aliases"):
        torch.ops.custom_op.flashinfer_gaudi_gemm_silu_out(x, weight, x)
    torch.hpu.synchronize()
    assert torch.equal(out.cpu(), torch.full((8, 128), 3., dtype=torch.bfloat16))


def test_graph_artifact_binding():
    import habana_frameworks.torch  # noqa: F401
    plan = GemmSiluPlan(8, 128, 128)
    restored = GemmSiluArtifactV1.from_json(plan.artifact.to_json())
    assert GemmSiluPlan(8, 128, 128, artifact=restored).artifact == restored
    with pytest.raises(BackendUnavailableError, match="specification"):
        GemmSiluPlan(8, 128, 128, artifact=GemmSiluArtifactV1(8, 128, 128, "0" * 64))
