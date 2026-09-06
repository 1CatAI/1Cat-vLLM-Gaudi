# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_gaudi.models.qwen3_next as qwen3_next_module
from vllm_gaudi.models.qwen3_5 import _save_ssm_state
from vllm_gaudi.ops.hpu_layernorm import HPUGemmaRMSNorm, HPURMSNorm
from vllm_gaudi.models.qwen3_next import (
    HpuQwen3DecoderLayerGroup,
    build_hpu_qwen3_layer_groups,
    can_compile_hpu_qwen3_layer_groups,
    can_use_hpu_qwen3_layer_groups,
    compile_hpu_qwen3_layer_groups,
    enable_hpu_qwen3_tp2_fused_ar_norm,
    supports_hpu_qwen3_layer_group_compilation,
)


class _AddLayer(torch.nn.Module):

    def __init__(self, value: int):
        super().__init__()
        self.value = value

    def forward(self, *, positions, hidden_states, residual):
        del positions
        return hidden_states + self.value, residual + self.value


class _FinalNorm(torch.nn.Module):

    def forward(self, hidden_states, residual):
        residual = hidden_states + residual
        return residual * 2, residual


def test_single_prefill_state_save_uses_the_only_real_slot():
    output = torch.tensor([7.0])
    final_state = torch.ones(1, 2, 3, 4)
    state_pool = torch.zeros(5, 2, 3, 4)

    returned = _save_ssm_state(output, final_state, state_pool, torch.tensor([2]))

    assert returned is output
    torch.testing.assert_close(state_pool[2], final_state[0])
    assert torch.count_nonzero(state_pool[:2]) == 0
    assert torch.count_nonzero(state_pool[3:]) == 0


def test_batched_prefill_state_save_ignores_padding_indices():
    output = torch.tensor([7.0])
    final_state = torch.stack((torch.ones(2, 3, 4), torch.full((2, 3, 4), 2.0), torch.full((2, 3, 4), 3.0)))
    state_pool = torch.zeros(5, 2, 3, 4)

    returned = _save_ssm_state(output, final_state, state_pool, torch.tensor([1, -1, 3]))

    assert returned is output
    torch.testing.assert_close(state_pool[1], final_state[0])
    torch.testing.assert_close(state_pool[3], final_state[2])
    assert torch.count_nonzero(state_pool[0]) == 0
    assert torch.count_nonzero(state_pool[2]) == 0
    assert torch.count_nonzero(state_pool[4]) == 0


def test_build_hpu_qwen3_layer_groups_preserves_order_and_tail():
    model = SimpleNamespace(
        layers=torch.nn.ModuleList([_AddLayer(i) for i in range(1, 6)]),
        start_layer=0,
        end_layer=5,
    )

    groups = build_hpu_qwen3_layer_groups(model, group_size=2)

    assert [len(group._layers) for group in groups] == [2, 2, 1]
    hidden_states, residual = torch.tensor(0), torch.tensor(0)
    for group in groups:
        hidden_states, residual = group(
            positions=torch.tensor(0),
            hidden_states=hidden_states,
            residual=residual,
        )
    assert hidden_states.item() == 15
    assert residual.item() == 15


def test_build_hpu_qwen3_layer_groups_puts_final_norm_in_last_group():
    model = SimpleNamespace(
        layers=torch.nn.ModuleList([_AddLayer(i) for i in range(1, 5)]),
        start_layer=0,
        end_layer=4,
    )
    final_norm = _FinalNorm()

    groups = build_hpu_qwen3_layer_groups(model, group_size=2, final_norm=final_norm)

    assert groups[0]._final_norm is None
    assert groups[1]._final_norm is final_norm
    hidden_states, residual = torch.tensor(0), torch.tensor(0)
    for group in groups:
        hidden_states, residual = group(
            positions=torch.tensor(0),
            hidden_states=hidden_states,
            residual=residual,
        )
    assert hidden_states.item() == 40
    assert residual.item() == 20


def test_build_hpu_qwen3_layer_groups_rejects_invalid_size():
    model = SimpleNamespace(layers=torch.nn.ModuleList(), start_layer=0, end_layer=0)

    with pytest.raises(ValueError, match="group_size must be positive"):
        build_hpu_qwen3_layer_groups(model, group_size=0)


def test_compile_hpu_qwen3_layer_groups_attaches_compiled_groups():
    model = SimpleNamespace(
        layers=torch.nn.ModuleList([_AddLayer(i) for i in range(4)]),
        start_layer=0,
        end_layer=4,
    )
    compiled = []

    def compile_fn(group):
        compiled.append(group)
        return group

    groups = compile_hpu_qwen3_layer_groups(model, group_size=2, compile_fn=compile_fn)

    assert groups == tuple(compiled)
    assert model._hpu_compiled_layer_groups == groups
    assert model._hpu_compiled_groups_include_final_norm is False
    assert [len(group._layers) for group in groups] == [2, 2]


def test_compile_hpu_qwen3_layer_groups_marks_fused_final_norm():
    final_norm = _FinalNorm()
    model = SimpleNamespace(
        layers=torch.nn.ModuleList([_AddLayer(1), _AddLayer(2)]),
        start_layer=0,
        end_layer=2,
    )

    groups = compile_hpu_qwen3_layer_groups(
        model,
        group_size=2,
        compile_fn=lambda group: group,
        final_norm=final_norm,
    )

    assert groups[-1]._final_norm is final_norm
    assert model._hpu_compiled_groups_include_final_norm is True


@pytest.mark.parametrize(
    ("group_size", "tp_size", "aux_layers", "expected"),
    [
        (8, 1, [], True),
        (8, 2, [], False),
        (8, 1, [1], False),
        (1, 1, [], False),
    ],
)
def test_can_compile_hpu_qwen3_layer_groups(
    group_size,
    tp_size,
    aux_layers,
    expected,
):
    assert can_compile_hpu_qwen3_layer_groups(group_size, tp_size, aux_layers) is expected


@pytest.mark.parametrize(
    ("layer_groups", "aux_layers", "attn_metadata", "batch_size", "expected"),
    [
        ((HpuQwen3DecoderLayerGroup(tuple()), ), [], SimpleNamespace(is_prompt=False), 16, True),
        ((HpuQwen3DecoderLayerGroup(tuple()), ), [], SimpleNamespace(is_prompt=False), 17, False),
        ((HpuQwen3DecoderLayerGroup(tuple()), ), [], SimpleNamespace(is_prompt=False, direct_gdn_state=True), 32, True),
        ((HpuQwen3DecoderLayerGroup(tuple()), ), [], SimpleNamespace(is_prompt=True), 1, False),
        ((HpuQwen3DecoderLayerGroup(tuple()), ), [1], SimpleNamespace(is_prompt=False), 1, False),
        (None, [], SimpleNamespace(is_prompt=False), 1, False),
        ((HpuQwen3DecoderLayerGroup(tuple()), ), [], None, 1, False),
    ],
)
def test_can_use_hpu_qwen3_layer_groups(layer_groups, aux_layers, attn_metadata, batch_size, expected):
    assert can_use_hpu_qwen3_layer_groups(layer_groups, aux_layers, attn_metadata, batch_size) is expected


@pytest.mark.parametrize(
    ("tensor_parallel_size", "tp2_fused_ar_norm", "expected"),
    [
        (1, False, True),
        (1, True, True),
        (2, False, False),
        (2, True, True),
        (4, True, False),
    ],
)
def test_supports_hpu_qwen3_layer_group_compilation(
    tensor_parallel_size,
    tp2_fused_ar_norm,
    expected,
):
    assert (supports_hpu_qwen3_layer_group_compilation(
        tensor_parallel_size,
        tp2_fused_ar_norm,
    ) is expected)


def _fake_dense_layer(attention_name, norm_factory=SimpleNamespace):
    output_projection = SimpleNamespace(reduce_results=True)
    attention = SimpleNamespace(**{attention_name: output_projection})
    layer = SimpleNamespace(
        mlp=SimpleNamespace(down_proj=SimpleNamespace(reduce_results=True)),
        input_layernorm=norm_factory(),
        post_attention_layernorm=norm_factory(),
        use_attn_reduce_scatter_for_moe=False,
    )
    if attention_name == "o_proj":
        layer.self_attn = attention
    else:
        layer.linear_attn = attention
    return layer


@pytest.mark.parametrize("norm_type", [HPURMSNorm, HPUGemmaRMSNorm])
def test_enable_tp2_fused_ar_norm_moves_all_dense_reductions(monkeypatch, default_vllm_config, norm_type):

    class FakeQwenModel:
        pass

    layers = tuple(_fake_dense_layer(name, lambda: norm_type(8)) for name in ("o_proj", "out_proj"))
    inner_model = FakeQwenModel()
    inner_model.layers = layers
    inner_model.start_layer = 0
    inner_model.end_layer = len(layers)
    inner_model.norm = norm_type(8)
    inner_model.embed_tokens = SimpleNamespace()
    model = SimpleNamespace(model=inner_model)
    initialized = []

    import vllm_gaudi.distributed.tp2_fused_ar_norm as fused_module

    monkeypatch.setattr(qwen3_next_module, "UpstreamQwen3NextModel", FakeQwenModel)
    monkeypatch.setattr(
        fused_module,
        "initialize_tp2_fused_ar_norm_runtime",
        lambda: initialized.append(True),
    )

    assert enable_hpu_qwen3_tp2_fused_ar_norm(model) == 5
    assert initialized == [True]
    assert inner_model._hpu_tp2_defer_embedding_reduce is True
    assert inner_model.embed_tokens._hpu_defer_tp2_reduce is True
    assert layers[0].self_attn.o_proj.reduce_results is False
    assert layers[0].mlp.down_proj.reduce_results is False
    assert layers[1].linear_attn.out_proj.reduce_results is False
    assert layers[1].mlp.down_proj.reduce_results is False
    assert layers[0].input_layernorm._hpu_tp2_fused_ar_norm is True
    assert layers[0].post_attention_layernorm._hpu_tp2_fused_ar_norm is True
    assert layers[1].input_layernorm._hpu_tp2_fused_ar_norm is True
    assert layers[1].post_attention_layernorm._hpu_tp2_fused_ar_norm is True
    assert inner_model.norm._hpu_tp2_fused_ar_norm is True


def test_enable_tp2_fused_ar_norm_rejects_non_dense_topology(monkeypatch):

    class FakeQwenModel:
        pass

    layer = _fake_dense_layer("o_proj")
    del layer.mlp.down_proj
    inner_model = FakeQwenModel()
    inner_model.layers = (layer, )
    inner_model.start_layer = 0
    inner_model.end_layer = 1
    inner_model.norm = SimpleNamespace()
    inner_model.embed_tokens = SimpleNamespace()

    monkeypatch.setattr(qwen3_next_module, "UpstreamQwen3NextModel", FakeQwenModel)

    assert enable_hpu_qwen3_tp2_fused_ar_norm(SimpleNamespace(model=inner_model)) == 0
    assert layer.self_attn.o_proj.reduce_results is True


@pytest.mark.parametrize("probe_passes", [False, True])
def test_gemma_native_probe_precedes_model_mutation(monkeypatch, default_vllm_config, probe_passes):
    import vllm_gaudi.distributed.tp2_fused_ar_norm as fused_module
    import vllm_gaudi.extension.runtime as runtime

    class FakeQwenModel:
        pass

    layer = _fake_dense_layer("o_proj", lambda: HPUGemmaRMSNorm(8).to(torch.bfloat16))
    inner = FakeQwenModel()
    inner.layers, inner.start_layer, inner.end_layer = (layer,), 0, 1
    inner.norm = HPUGemmaRMSNorm(8).to(torch.bfloat16)
    inner.embed_tokens = SimpleNamespace()
    monkeypatch.setattr(qwen3_next_module, "UpstreamQwen3NextModel", FakeQwenModel)
    monkeypatch.setattr(runtime, "get_config", lambda: SimpleNamespace(tp2_gemma_fused_ar_norm=True))
    monkeypatch.setattr(fused_module, "initialize_tp2_fused_ar_norm_runtime", lambda: None)

    def probe(width):
        assert width == 8
        assert layer.self_attn.o_proj.reduce_results
        assert layer.mlp.down_proj.reduce_results
        assert not hasattr(inner, "_hpu_tp2_defer_embedding_reduce")
        if not probe_passes:
            raise RuntimeError("probe failed")

    monkeypatch.setattr(fused_module, "validate_tp2_gemma_fusion_runtime", probe)
    if probe_passes:
        assert enable_hpu_qwen3_tp2_fused_ar_norm(SimpleNamespace(model=inner)) == 3
        assert inner.norm._hpu_tp2_gemma_native_ready
        assert layer.input_layernorm._hpu_tp2_gemma_native_ready
        assert not layer.self_attn.o_proj.reduce_results
    else:
        with pytest.raises(RuntimeError, match="probe failed"):
            enable_hpu_qwen3_tp2_fused_ar_norm(SimpleNamespace(model=inner))
        assert layer.self_attn.o_proj.reduce_results
        assert layer.mlp.down_proj.reduce_results
        assert not hasattr(inner.norm, "_hpu_tp2_gemma_native_ready")


@pytest.mark.parametrize("unsupported_boundary", ["input_layernorm", "post_attention_layernorm", "final_norm"])
def test_enable_tp2_fused_ar_norm_rejects_unsupported_norm_before_mutation(monkeypatch, default_vllm_config,
                                                                           unsupported_boundary):

    class FakeQwenModel:
        pass

    layer = _fake_dense_layer("o_proj", lambda: HPUGemmaRMSNorm(8))
    inner_model = FakeQwenModel()
    inner_model.layers = (layer, )
    inner_model.start_layer = 0
    inner_model.end_layer = 1
    inner_model.norm = HPUGemmaRMSNorm(8)
    inner_model.embed_tokens = SimpleNamespace()
    if unsupported_boundary == "final_norm":
        inner_model.norm = SimpleNamespace()
    else:
        setattr(layer, unsupported_boundary, SimpleNamespace())

    monkeypatch.setattr(qwen3_next_module, "UpstreamQwen3NextModel", FakeQwenModel)
    import vllm_gaudi.distributed.tp2_fused_ar_norm as fused_module

    monkeypatch.setattr(fused_module, "initialize_tp2_fused_ar_norm_runtime",
                        lambda: pytest.fail("Unsupported normalization must not initialize the runtime"))

    assert enable_hpu_qwen3_tp2_fused_ar_norm(SimpleNamespace(model=inner_model)) == 0
    assert layer.self_attn.o_proj.reduce_results is True
    assert layer.mlp.down_proj.reduce_results is True
    assert not hasattr(inner_model.embed_tokens, "_hpu_defer_tp2_reduce")
    assert not hasattr(inner_model, "_hpu_tp2_defer_embedding_reduce")
