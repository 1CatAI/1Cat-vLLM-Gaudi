# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
