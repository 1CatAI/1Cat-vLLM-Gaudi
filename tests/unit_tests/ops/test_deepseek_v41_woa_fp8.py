# SPDX-License-Identifier: Apache-2.0
"""Model dispatch must bypass BF16 wo_a expansion only for selected layers."""
from types import SimpleNamespace

import torch

from vllm_gaudi import envs
from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree
from vllm_gaudi.ops.deepseek_v41_attention import CSA2Attention
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention


def test_woa_load_bypasses_dense_for_selected_layers(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_BF16_LM_HEAD", False)
    specs = {f"layers.{i}.attn.wo_a.weight": {"shape": [4, 4], "dtype": "F8_E4M3"} for i in (0, 1)}
    calls = []

    def dense(name, device):
        calls.append(name)
        return torch.ones(4, 4, dtype=torch.bfloat16)

    def side(name, device):
        return (torch.zeros(4, 1, 1024) if name.endswith("channel_scale") else torch.zeros(
            4, 4096, 1024, dtype=torch.uint8).view(torch.float8_e4m3fn))

    tree = _weight_tree(specs)
    shard = SimpleNamespace(specs=specs, dense=dense, check_identity=lambda: None)
    load_weight_tree(shard, tree, "cpu", woa_sidecar=SimpleNamespace(tensor=side), woa_layers=[0])
    assert calls == ["layers.1.attn.wo_a.weight"]
    assert tree.layers.get_submodule("0").attn.wo_a.weight.dtype == torch.float8_e4m3fn
    assert tree.layers.get_submodule("1").attn.wo_a.weight.dtype == torch.bfloat16


def test_woa_c1_and_prefill_share_prepared_weight(monkeypatch):
    weight = object()
    scale = object()
    calls = []

    def operator(x, w, s):
        assert w is weight and s is scale
        calls.append(x.shape)
        return torch.zeros(x.shape[0], 4096, dtype=torch.bfloat16)

    monkeypatch.setattr(torch.ops, "custom_op", SimpleNamespace(custom_deepseek_v41_woa_fp8_gaudi2=operator))
    attention = SimpleNamespace(woa_fp8=True,
                                weights=SimpleNamespace(wo_a=SimpleNamespace(weight=weight, channel_scale=scale)))
    for implementation in (CSA2Attention, PagedCSA2Attention):
        for tokens in (1, 3, 512):
            assert implementation.project_output(attention, torch.zeros(tokens, 4, 4096)).shape == (tokens, 4096)
    assert len(calls) == 6


def test_paged_output_preparation_matches_bounded_layout():
    original = torch.arange(8 * 512 * 1024, dtype=torch.bfloat16).reshape(4096, 1024)
    for implementation in (CSA2Attention, PagedCSA2Attention):
        module = implementation.__new__(implementation)
        torch.nn.Module.__init__(module)
        module.woa_fp8 = False
        module.prepared_output = True
        module.output_gemm_layout = True
        module.heads, module.groups = 8, 4
        module.weights = torch.nn.Module()
        module.weights.wo_a = torch.nn.Module()
        module.weights.wo_a.register_buffer("weight", original.clone(), False)
        module.prepare_output_weight()
        assert module.weights.wo_a.weight.shape == (4, 1024, 1024)
        assert torch.equal(module.weights.wo_a.weight, original.reshape(4, 1024, 1024))


def test_paged_rotary_views_follow_active_bucket_without_copying():
    attention = object.__new__(PagedCSA2Attention)
    torch.nn.Module.__init__(attention)
    attention.register_buffer("rotary", torch.empty(1048576, 32, 2), False)
    attention.register_buffer("rotary_native", torch.empty(1048576, 64), False)
    attention.search_length = 512
    ordinary, native = attention._rotary_table(), attention._rotary_native_table()
    assert ordinary.shape == (512, 32, 2) and native.shape == (512, 64)
    assert ordinary.untyped_storage()._cdata == attention.rotary.untyped_storage()._cdata
    assert native.untyped_storage()._cdata == attention.rotary_native.untyped_storage()._cdata
    attention.search_length = 8192
    assert attention._rotary_table().shape[0] == 8192
    assert attention._rotary_native_table().shape[0] == 8192
