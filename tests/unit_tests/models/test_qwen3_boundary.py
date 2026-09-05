# SPDX-License-Identifier: Apache-2.0
"""Dependency, residual and complete-reduction checks for the boundary pipeline."""

from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.models import qwen3_boundary as pipeline


class Projection(torch.nn.Module):

    def __init__(self, n_in, n_out, rank=0):
        super().__init__()
        self.weight = torch.randn(n_out, n_in) * 0.1
        self.bias = torch.randn(n_out) * 0.1
        self.tp_rank = rank
        self.tp_size = 2
        self.skip_bias_add = False
        self.input_is_parallel = True
        self.reduce_results = True
        self.quant_method = SimpleNamespace(apply=self.apply)

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, self.weight, bias)

    def forward(self, x):
        return self.apply(self, x, self.bias), None


class Norm(torch.nn.Module):

    def forward(self, x, residual):
        assert residual.ndim == 3
        residual = residual + x.reshape_as(residual)
        normalized = residual * torch.rsqrt(residual.square().mean(dim=-1, keepdim=True) + 1e-6)
        return normalized.reshape_as(x), residual


class Act(torch.nn.Module):

    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


def make_mlp():
    mlp = pipeline.Qwen2MoeMLP.__new__(pipeline.Qwen2MoeMLP)
    torch.nn.Module.__init__(mlp)
    mlp.gate_up_proj = Projection(4, 8)
    mlp.down_proj = Projection(4, 4)
    mlp.act_fn = Act()
    mlp.expert_gate = None
    return mlp


@pytest.mark.parametrize("chunks", [2, 4, 16])
@pytest.mark.parametrize("residual_3d", [False, True])
def test_pipeline_preserves_outputs_residuals_and_collectives(monkeypatch, chunks, residual_3d):
    torch.manual_seed(19)
    core = torch.randn(7, 3)
    residual = torch.randn(1, 7, 4) if residual_3d else torch.randn(7, 4)
    original_residual = residual.clone()
    projection, norm, mlp = Projection(3, 4), Norm(), make_mlp()
    normalized, reference_residual = norm(pipeline.partial_projection(projection, core) * 2, residual.reshape(1, 7, 4))
    gate_up, _ = mlp.gate_up_proj(normalized)
    reference = pipeline.partial_projection(mlp.down_proj, mlp.act_fn(gate_up)) * 2
    events, sizes = [], []
    monkeypatch.setattr(pipeline, "get_tp_group", lambda: SimpleNamespace(device_group="group"))

    def all_reduce(tensor, *, group, async_op):
        assert group == "group" and async_op
        index = len(sizes)
        sizes.append(tensor.numel())
        events.append(("submit", index))

        def wait():
            tensor.mul_(2)
            events.append(("wait", index))

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    output, updated = pipeline.pipeline_attention_mlp(core, residual, projection, norm, mlp, chunks)
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(updated, reference_residual.reshape(7, 4))
    torch.testing.assert_close(residual, original_residual)
    actual_chunks = len(torch.chunk(core, chunks, dim=0))
    assert len(sizes) == 2 * actual_chunks
    assert sum(sizes) == 2 * 7 * 4
    expected = [("submit", i) for i in range(actual_chunks)]
    for i in range(actual_chunks):
        expected.extend((("wait", i), ("submit", actual_chunks + i)))
    expected.extend(("wait", actual_chunks + i) for i in range(actual_chunks))
    assert events == expected


@pytest.mark.parametrize("shape,prompt,batch,expected", [
    ((7, 4), True, 1, True),
    ((1, 7, 4), True, 1, True),
    ((2, 7, 4), True, 2, False),
    ((7, 4), False, 1, False),
    ((7, 4), True, 2, False),
    ((1, 4), True, 1, False),
])
def test_prefill_guard(monkeypatch, shape, prompt, batch, expected):
    metadata = SimpleNamespace(is_prompt=prompt, seq_lens_tensor=torch.ones(batch))
    monkeypatch.setattr(pipeline, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata))
    assert pipeline.can_pipeline_prefill(torch.zeros(shape), 2) is expected


def make_layer():
    layer = pipeline.Qwen3_5DecoderLayer.__new__(pipeline.Qwen3_5DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.layer_scale = False
    layer.use_attn_reduce_scatter_for_moe = False
    layer.layer_type = "linear_attention"
    layer.mlp = make_mlp()
    attention = pipeline.HPUGatedDeltaNetAttention.__new__(pipeline.HPUGatedDeltaNetAttention)
    torch.nn.Module.__init__(attention)
    attention.out_proj = Projection(3, 4)
    layer.linear_attn = attention
    for name in ("input_layernorm", "post_attention_layernorm"):
        norm = pipeline.HPUGemmaRMSNorm.__new__(pipeline.HPUGemmaRMSNorm)
        torch.nn.Module.__init__(norm)
        setattr(layer, name, norm)
    return layer


def make_model(layers):
    inner = pipeline.Qwen3NextModel.__new__(pipeline.Qwen3NextModel)
    torch.nn.Module.__init__(inner)
    inner.layers = torch.nn.ModuleList(layers)
    inner.start_layer = 0
    inner.end_layer = len(layers)
    return SimpleNamespace(model=inner)


@pytest.mark.parametrize("reason", ["layer_scale", "deferred", "tp1", "expert", "norm"])
def test_enable_rejects_unsupported_topology_before_mutation(reason):
    first, last = make_layer(), make_layer()
    if reason == "layer_scale":
        last.layer_scale = True
    elif reason == "deferred":
        last.linear_attn.out_proj.reduce_results = False
    elif reason == "tp1":
        last.mlp.down_proj.tp_size = 1
    elif reason == "expert":
        last.mlp.expert_gate = torch.nn.Identity()
    else:
        last.post_attention_layernorm = torch.nn.Identity()
    assert pipeline.enable_hpu_qwen3_boundary_pipeline(make_model([first, last]), 4, 2) == 0
    assert type(first) is pipeline.Qwen3_5DecoderLayer
    assert not hasattr(first, "_hpu_boundary_chunks")


def test_enable_preserves_projection_reduction_flags():
    layer = make_layer()
    projection = layer.linear_attn.out_proj
    assert pipeline.enable_hpu_qwen3_boundary_pipeline(make_model([layer]), 4, 2) == 1
    assert isinstance(layer, pipeline.HpuQwen3BoundaryDecoderLayer)
    assert projection is layer.linear_attn.out_proj
    assert projection.reduce_results and layer.mlp.down_proj.reduce_results


@pytest.mark.parametrize("three_dimensional", [False, True])
def test_full_attention_core_preserves_default_projection_path(three_dimensional):
    from vllm_gaudi.models.qwen3_next import _hpu_qwen3next_attention_forward
    torch.manual_seed(23)
    projection = Projection(4, 4)
    attention = SimpleNamespace(
        qkv_proj=Projection(4, 16),
        o_proj=projection,
        q_size=4,
        kv_size=4,
        num_heads=2,
        num_kv_heads=2,
        head_dim=2,
        attn_output_gate=True,
        q_norm=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
        rotary_emb=lambda positions, q, k: (q, k),
        attn=lambda q, k, v: q + k + v.reshape_as(q),
    )

    def project_qkv_gate(qkv, positions):
        q_gate, k, v = qkv.split([8, 4, 4], dim=-1)
        q, gate = q_gate.view(-1, 2, 4).chunk(2, dim=-1)
        return q.reshape(-1, 4), k.reshape(-1, 4), v.reshape(-1, 4), gate.reshape(-1, 4)

    attention._project_qkv_gate = project_qkv_gate
    x = torch.randn(1, 7, 4) if three_dimensional else torch.randn(7, 4)
    positions = torch.arange(7)
    core = _hpu_qwen3next_attention_forward(attention, positions, x, return_core=True)
    output = _hpu_qwen3next_attention_forward(attention, positions, x)
    expected, _ = projection(core)
    torch.testing.assert_close(output, expected.reshape_as(x))


def test_decoder_uses_original_forward_when_disabled(monkeypatch):
    layer = make_layer()
    pipeline.enable_hpu_qwen3_boundary_pipeline(make_model([layer]), 4, 2)
    layer._hpu_boundary_chunks = 1
    expected = object()
    monkeypatch.setattr(pipeline.Qwen3_5DecoderLayer, "forward", lambda self, *args, **kwargs: expected)
    assert layer(torch.zeros(7, 4), None, torch.arange(7)) is expected


@pytest.mark.parametrize("layer_type", ["linear_attention", "full_attention"])
@pytest.mark.parametrize("chunks", [2, 4, 16])
def test_next_layer_prefetch_matches_token_local_reference(monkeypatch, layer_type, chunks):
    torch.manual_seed(31)
    core, residual = torch.randn(7, 3), torch.randn(7, 4)
    projection, norm, mlp = Projection(3, 4), Norm(), make_mlp()
    next_layer = SimpleNamespace(layer_type=layer_type, input_layernorm=Norm())
    if layer_type == "linear_attention":
        next_layer.linear_attn = SimpleNamespace(in_proj_qkvz=Projection(4, 6), in_proj_ba=Projection(4, 2))
        input_projections = (next_layer.linear_attn.in_proj_qkvz, next_layer.linear_attn.in_proj_ba)
    else:
        next_layer.self_attn = SimpleNamespace(qkv_proj=Projection(4, 6))
        input_projections = (next_layer.self_attn.qkv_proj, )
    normed, updated = norm(pipeline.partial_projection(projection, core) * 2, residual.unsqueeze(0))
    gate_up, _ = mlp.gate_up_proj(normed)
    output = pipeline.partial_projection(mlp.down_proj, mlp.act_fn(gate_up)) * 2
    expected_hidden, expected_residual = next_layer.input_layernorm(output, updated)
    expected_projected = tuple(p(expected_hidden)[0] for p in input_projections)
    monkeypatch.setattr(pipeline, "get_tp_group", lambda: SimpleNamespace(device_group="group"))
    monkeypatch.setattr(torch.distributed, "all_reduce",
                        lambda tensor, **kwargs: SimpleNamespace(wait=lambda: tensor.mul_(2)))
    hidden, residual, projected = pipeline.pipeline_attention_mlp(core,
                                                                  residual,
                                                                  projection,
                                                                  norm,
                                                                  mlp,
                                                                  chunks,
                                                                  next_layer=next_layer,
                                                                  return_prefetched=True)
    torch.testing.assert_close(hidden, expected_hidden)
    torch.testing.assert_close(residual, expected_residual.reshape(7, 4))
    for actual, expected in zip(projected, expected_projected):
        torch.testing.assert_close(actual, expected)


def test_prefetch_links_preserve_module_ownership():
    first, second = make_layer(), make_layer()
    model = make_model([first, second])
    before = list(model.model.named_modules())
    assert pipeline.enable_hpu_qwen3_boundary_pipeline(model, 4, 2, prefetch=True) == 2
    assert first._hpu_prefetch_next is second
    assert second._hpu_prefetch_next is None
    assert model.model._hpu_boundary_prefetch
    assert list(model.model.named_modules()) == before


def test_prefetch_rejects_auxiliary_hidden_state_consumers():
    first = make_layer()
    model = make_model([first])
    model.model.aux_hidden_state_layers = (0, )
    assert pipeline.enable_hpu_qwen3_boundary_pipeline(model, 4, 2, prefetch=True) == 0
    assert type(first) is pipeline.Qwen3_5DecoderLayer
