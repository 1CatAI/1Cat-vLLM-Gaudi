# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path
from unittest import mock

import pytest
import torch

import vllm_gaudi.ops.qwen38_native_qk as native_qk


def test_validate_native_qk_shape():
    repeat = native_qk.validate_qwen38_native_qk_shape(
        tp_size=1,
        qkv_width=10240,
        key_width=2048,
        value_width=6144,
        key_head_dim=128,
        value_head_dim=128,
    )
    assert repeat == 3


def test_validate_native_qk_shape_rejects_tp2():
    with pytest.raises(ValueError, match="TP1 layout"):
        native_qk.validate_qwen38_native_qk_shape(
            tp_size=2,
            qkv_width=5120,
            key_width=1024,
            value_width=3072,
            key_head_dim=128,
            value_head_dim=128,
        )


def test_validate_native_qk_shape_accepts_compact_tp2():
    assert native_qk.validate_qwen38_native_qk_shape(
        tp_size=2,
        qkv_width=5120,
        key_width=1024,
        value_width=3072,
        key_head_dim=128,
        value_head_dim=128,
        compact_qk=True,
    ) == 3


@pytest.mark.parametrize("field,value", [
    ("tp_size", 4),
    ("qkv_width", 10240),
    ("key_width", 2048),
    ("value_width", 6144),
    ("key_head_dim", 64),
    ("value_head_dim", 64),
])
def test_validate_compact_tp2_rejects_mismatched_layout(field, value):
    layout = dict(tp_size=2,
                  qkv_width=5120,
                  key_width=1024,
                  value_width=3072,
                  key_head_dim=128,
                  value_head_dim=128,
                  compact_qk=True)
    layout[field] = value
    with pytest.raises(ValueError, match="compact TP2 layout"):
        native_qk.validate_qwen38_native_qk_shape(**layout)


def test_native_qk_loader_is_opt_in(monkeypatch):
    monkeypatch.setenv("VLLM_GDN_QWEN38_NATIVE_QK_PREP", "0")
    with mock.patch.object(native_qk.torch.ops, "load_library") as load_library:
        assert native_qk.load_qwen38_native_qk_prep() is False
    load_library.assert_not_called()


def test_native_qk_loader_loads_registration_extension(tmp_path, monkeypatch):
    extension = tmp_path / "native_qk.so"
    extension.touch()
    monkeypatch.setenv("VLLM_GDN_QWEN38_NATIVE_QK_PREP", "1")
    monkeypatch.setenv(
        "VLLM_GDN_QWEN38_NATIVE_QK_PREP_EXTENSION",
        str(extension),
    )
    monkeypatch.setattr(native_qk, "_loaded_extension", None)

    with (
            mock.patch.object(native_qk.torch.ops, "load_library") as load_library,
            mock.patch.object(native_qk, "_resolve_post_conv_op", return_value=mock.Mock()),
    ):
        assert native_qk.load_qwen38_native_qk_prep() is True

    load_library.assert_called_once_with(str(Path(extension).resolve()))


def test_compiled_conv_producer_impl(monkeypatch):
    packed = torch.full((2, 10240), 2.0, dtype=torch.bfloat16)
    conv_state = torch.zeros((5, 3, 10240), dtype=torch.bfloat16)
    weight = torch.ones((10240, 4), dtype=torch.bfloat16)
    bias = torch.zeros(10240, dtype=torch.bfloat16)
    has_initial_state = torch.zeros(1, dtype=torch.bool)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    cache_indices = torch.tensor([1], dtype=torch.int64)
    captured = {}

    def fake_conv(*args, **kwargs):
        captured["weight"] = args[2]
        captured["kwargs"] = kwargs
        return args[0], args[1] + 1

    monkeypatch.setattr(
        native_qk,
        "_resolve_hpu_causal_conv_op",
        lambda: fake_conv,
    )
    post_conv, updated_state = native_qk._qwen38_hpu_conv_producer_impl(
        packed,
        conv_state,
        weight,
        bias,
        has_initial_state,
        query_start_loc,
        cache_indices,
    )

    assert post_conv is packed
    assert torch.all(updated_state == 1)
    assert captured["weight"].shape == (4, 10240)
    assert captured["weight"].is_contiguous()
    assert captured["kwargs"] == {"activation": True, "pad_slot_id": -1}


def test_compiled_conv_producer_is_cached(monkeypatch):
    compiled = mock.Mock()
    compile_mock = mock.Mock(return_value=compiled)
    monkeypatch.setattr(native_qk, "_compiled_conv_producer", None)
    monkeypatch.setattr(native_qk.torch, "compile", compile_mock)

    assert native_qk._get_compiled_conv_producer() is compiled
    assert native_qk._get_compiled_conv_producer() is compiled
    compile_mock.assert_called_once_with(
        native_qk._qwen38_hpu_conv_producer_impl,
        backend="hpu_backend",
        fullgraph=True,
        options={"force_static_compile": True},
    )


@pytest.fixture(scope="module")
def built_native_ops():
    """Optional integration checks against the built extension and real HPU."""
    extension = os.environ.get("VLLM_GDN_QWEN38_NATIVE_QK_PREP_EXTENSION")
    if not extension:
        pytest.skip("Set VLLM_GDN_QWEN38_NATIVE_QK_PREP_EXTENSION and GC_KERNEL_PATH for native HPU tests")
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(extension)
    from vllm_gaudi.patches import _patch_hpu_fx_stack_trace_parser
    _patch_hpu_fx_stack_trace_parser()
    return torch.ops.custom_op


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("tokens", [1, 65, 8192])
@pytest.mark.parametrize("scale", [0.0, 1e-8, 1.0])
def test_native_compact_qk_hpu_matches_reference(built_native_ops, heads, tokens, scale):
    generator = torch.Generator().manual_seed(17)
    packed = (torch.randn(tokens, heads * 5 * 128, generator=generator) * scale).bfloat16()
    # Distinct Q and K contents exercise the TP2 K offset and head ordering.
    expected = []
    for offset in (0, heads * 128):
        x = packed[:, offset:offset + heads * 128].reshape(tokens, heads, 128).float()
        expected.append((x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)).bfloat16())
    op = built_native_ops.qwen38_post_conv_qk_compact_bf16_gaudi2
    actual = op(packed.to("hpu"))
    for value, reference in zip(actual, expected):
        assert value.shape == (tokens, heads, 128)
        assert value.dtype == torch.bfloat16
        torch.testing.assert_close(value.cpu(), reference, atol=0.002, rtol=0.01)


@pytest.mark.parametrize("heads", [8, 16])
def test_native_compact_qk_compiled_and_meta(built_native_ops, heads):
    op = built_native_ops.qwen38_post_conv_qk_compact_bf16_gaudi2
    packed = torch.randn(65, heads * 5 * 128, dtype=torch.bfloat16, device="hpu")
    meta = op(torch.empty(packed.shape, dtype=packed.dtype, device="meta"))
    compiled = torch.compile(op, backend="hpu_backend", fullgraph=True, options={"force_static_compile": True})
    expected = op(packed)
    actual = compiled(packed)
    for metadata, value, reference in zip(meta, actual, expected):
        assert metadata.shape == (65, heads, 128)
        assert metadata.dtype == torch.bfloat16
        torch.testing.assert_close(value.cpu(), reference.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("shape,dtype", [((7, 6400), torch.bfloat16), ((7, 5120), torch.float32),
                                         ((1, 7, 5120), torch.bfloat16)])
def test_native_compact_qk_meta_rejects_invalid_layout(built_native_ops, shape, dtype):
    with pytest.raises(RuntimeError):
        built_native_ops.qwen38_post_conv_qk_compact_bf16_gaudi2(torch.empty(shape, dtype=dtype, device="meta"))


@pytest.mark.parametrize("heads", [8, 16])
def test_native_compact_kkt_sharded_shape(built_native_ops, heads):
    torch.manual_seed(18)
    dot = torch.randn(128, heads, 64, 64, dtype=torch.bfloat16)
    beta = torch.rand(128, heads, 3, 64, dtype=torch.bfloat16)
    product = dot.unsqueeze(2) * beta.unsqueeze(-1)
    expected = torch.tril(product, diagonal=-1) + torch.eye(64, dtype=torch.bfloat16)
    op = built_native_ops.qwen38_compact_kkt_bf16_gaudi2
    compiled = torch.compile(op, backend="hpu_backend", fullgraph=True, options={"force_static_compile": True})
    actual = compiled(dot.to("hpu"), beta.to("hpu"))
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
