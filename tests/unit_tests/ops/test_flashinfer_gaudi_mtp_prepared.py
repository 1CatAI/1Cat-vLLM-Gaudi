# SPDX-License-Identifier: Apache-2.0
"""Explicit-library qualification for the unpromoted graph-native MTP core."""
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer_gaudi._reference import _l2_normalize_rsqrt
from flashinfer_gaudi.gdn_decode import _gated_delta_rule_mtp_packed_reference, gated_delta_rule_mtp_packed


@pytest.fixture(scope="module")
def native():
    library = os.environ.get("FLASHINFER_GAUDI_TEST_LIBRARY")
    if not library:
        pytest.skip("Set FLASHINFER_GAUDI_TEST_LIBRARY to an isolated, freshly built extension")
    torch.ops.load_library(library)
    # This isolated library already supplies the schemas. Do not load a
    # second packaged copy when exercising the public Python dispatcher.
    with mock.patch("flashinfer_gaudi._native._LOAD_ATTEMPTED", True):
        yield torch.ops.custom_op.flashinfer_gaudi_gdn_mtp_prepared


@pytest.mark.parametrize("batch", [1, 2, 16])
def test_prepared_native_meta_contract(native, batch):
    state = torch.empty(batch, 48, 128, 128, device="meta")
    packed = torch.empty(batch, 8, 10240, device="meta")
    gate = torch.empty(batch, 8, 48, device="meta")
    output, checkpoints = native(state, packed, gate, gate)
    assert output.shape == (batch, 8, 48, 128)
    assert checkpoints.shape == (batch, 8, 48, 128, 128)
    assert output.dtype == checkpoints.dtype == torch.float32


@pytest.mark.parametrize("bad_input", ["state_dtype", "packed_width", "gate_dtype", "batch"])
def test_prepared_native_rejects_invalid_meta(native, bad_input):
    batch = 17 if bad_input == "batch" else 1
    state = torch.empty(batch,
                        48,
                        128,
                        128,
                        device="meta",
                        dtype=torch.bfloat16 if bad_input == "state_dtype" else torch.float32)
    packed = torch.empty(batch, 8, 10239 if bad_input == "packed_width" else 10240, device="meta")
    gate = torch.empty(batch,
                       8,
                       48,
                       device="meta",
                       dtype=torch.bfloat16 if bad_input == "gate_dtype" else torch.float32)
    with pytest.raises(RuntimeError):
        native(state, packed, gate, gate)


def test_prepared_native_has_builtin_graph_support(native):
    from habana_frameworks.torch.dynamo.compile_backend.shared_layer import check_for_default_op_support
    node = SimpleNamespace(target=native.default)
    supported, reason = check_for_default_op_support("flashinfer_gaudi_gdn_mtp_prepared", node, False)
    assert supported, reason
    assert native.default.namespace == "custom_op"
    assert all(argument.alias_info is None for argument in native.default._schema.arguments)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("route", ["core", "public"])
@pytest.mark.parametrize("qkv_scale", [0.1, 1e-4])
def test_prepared_native_hardware_rollback(native, batch, route, qkv_scale):
    if os.environ.get("FLASHINFER_GAUDI_TEST_HPU") != "1":
        pytest.skip("Set FLASHINFER_GAUDI_TEST_HPU=1 only when a test card is available")
    from habana_frameworks.torch.dynamo.compile_backend import config as hpu_config
    rng = torch.Generator().manual_seed(817 + batch)
    packed = (torch.randn(batch, 8, 10240, dtype=torch.bfloat16, generator=rng) * qkv_scale).to("hpu")
    g = (-torch.rand(batch, 8, 48, generator=rng) * 0.1).to("hpu")
    beta = torch.sigmoid(torch.randn(batch, 8, 48, dtype=torch.bfloat16, generator=rng)).to("hpu")
    pool = (torch.randn(batch * 8 + 2, 48, 128, 128, generator=rng) * 0.01).to("hpu")
    reference_pool = pool.clone()
    indices = (torch.randperm(batch * 8, generator=rng) + 1).to("hpu", dtype=torch.int32).reshape(batch, 8)
    accepted = torch.ones(batch, dtype=torch.int32, device="hpu")
    lengths = torch.full((batch, ), 8, dtype=torch.int32, device="hpu")

    def candidate(packed, g, beta, pool, indices, accepted):
        if route == "public":
            return gated_delta_rule_mtp_packed(packed,
                                               g,
                                               beta,
                                               pool,
                                               indices,
                                               accepted,
                                               lengths,
                                               assume_full_query=True,
                                               assume_distinct_checkpoints=True)[0]
        loads = indices.gather(1, (accepted.long().clamp(1, 8) - 1).unsqueeze(1)).squeeze(1).long()
        initial = pool.index_select(0, loads)
        qk = _l2_normalize_rsqrt(packed[..., :4096].float().reshape(batch, 8, 32, 128))
        prepared = torch.cat(
            ((qk[..., :16, :] * (128**-0.5)).flatten(2), qk[..., 16:, :].flatten(2), packed[..., 4096:].float()),
            dim=-1)
        output, checkpoints = native(initial, prepared, torch.exp(g), beta.float())
        pool.index_copy_(0, indices.reshape(-1).long(), checkpoints.reshape(-1, 48, 128, 128))
        return output.to(packed.dtype)

    def reference(packed, g, beta, pool, indices, accepted):
        return _gated_delta_rule_mtp_packed_reference(packed,
                                                      g,
                                                      beta,
                                                      pool,
                                                      indices,
                                                      accepted,
                                                      lengths,
                                                      assume_full_query=True)[0]

    # fullgraph alone only forbids Dynamo breaks, not backend eager fallback.
    with (
            mock.patch.object(hpu_config, "use_eager_fallback", False),
            mock.patch.dict(os.environ, {
                "FLASHINFER_GAUDI_BACKEND": "auto",
                "FLASHINFER_GAUDI_ENABLE_MTP_PREPARED": "1"
            }),
    ):
        compiled = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
        oracle = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
        for round_index in range(32):
            accepted.fill_(round_index % 8 + 1)
            actual = compiled(packed, g, beta, pool, indices, accepted)
            expected = oracle(packed, g, beta, reference_pool, indices, accepted)
            torch.hpu.synchronize()
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-2, atol=1e-5)
            torch.testing.assert_close(pool.cpu(), reference_pool.cpu(), rtol=2e-4, atol=2e-6)


def test_packed_native_uses_additive_epsilon(native):
    if os.environ.get("FLASHINFER_GAUDI_TEST_HPU") != "1":
        pytest.skip("Set FLASHINFER_GAUDI_TEST_HPU=1 only when a test card is available")
    rng = torch.Generator().manual_seed(237)
    packed_cpu = torch.randn(1, 8, 10240, dtype=torch.bfloat16, generator=rng) * 0.1
    packed_cpu[..., :4096] *= 0.001
    g_cpu = -torch.rand(1, 8, 48, generator=rng) * 0.1
    beta_cpu = torch.sigmoid(torch.randn(1, 8, 48, dtype=torch.bfloat16, generator=rng))
    pool_cpu = torch.randn(10, 48, 128, 128, generator=rng) * 0.01
    indices_cpu = torch.tensor([[8, 2, 4, 6, 1, 7, 3, 5]], dtype=torch.int32)
    lengths_cpu = torch.tensor([8], dtype=torch.int32)
    packed, g, beta, pool, indices, lengths = (value.to("hpu") for value in (packed_cpu, g_cpu, beta_cpu, pool_cpu,
                                                                             indices_cpu, lengths_cpu))
    for offset in range(1, 9):
        accepted = torch.tensor([offset], dtype=torch.int32)
        expected, _ = _gated_delta_rule_mtp_packed_reference(packed_cpu,
                                                             g_cpu,
                                                             beta_cpu,
                                                             pool_cpu,
                                                             indices_cpu,
                                                             accepted,
                                                             lengths_cpu,
                                                             assume_full_query=True)
        actual = torch.ops.flashinfer_gaudi.gdn_mtp_packed(pool, packed, torch.exp(g), beta, indices,
                                                           accepted.to("hpu"), lengths)
        torch.hpu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-5)
        torch.testing.assert_close(pool.cpu(), pool_cpu, rtol=2e-4, atol=2e-6)
