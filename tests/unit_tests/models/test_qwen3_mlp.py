# SPDX-License-Identifier: Apache-2.0
"""CPU checks for ordering, fallback and fail-closed MLP pipeline activation."""

from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.models import qwen3_mlp as pipeline


@pytest.fixture(autouse=True)
def single_prefill_context(monkeypatch):
    metadata = SimpleNamespace(is_prompt=True, seq_lens_tensor=torch.ones(1))
    monkeypatch.setattr(pipeline, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata))
    return metadata


class Projection(torch.nn.Module):

    def __init__(self, weight, events, name, reduce_results=False):
        super().__init__()
        self.weight = weight
        self.events = events
        self.name = name
        self.tp_size = 2
        self.tp_rank = 0
        self.input_is_parallel = True
        self.reduce_results = reduce_results
        self.bias = None
        self.skip_bias_add = False
        self.quant_method = SimpleNamespace(apply=self.apply)

    def apply(self, layer, x, bias=None):
        self.events.append(self.name)
        return torch.nn.functional.linear(x, self.weight, bias)

    def forward(self, x):
        output = self.apply(self, x)
        if self.reduce_results and self.tp_size > 1:
            output = output * 2
        return output, None


class Activation(torch.nn.Module):

    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


def make_mlp(events):
    mlp = pipeline.Qwen2MoeMLP.__new__(pipeline.Qwen2MoeMLP)
    torch.nn.Module.__init__(mlp)
    generator = torch.Generator().manual_seed(17)
    mlp.gate_up_proj = Projection(torch.randn(8, 4, generator=generator), events, "gate")
    mlp.act_fn = Activation()
    mlp.down_proj = Projection(torch.randn(4, 4, generator=generator), events, "down", reduce_results=True)
    mlp.expert_gate = None
    return mlp


def make_model(mlps):
    inner = pipeline.Qwen3NextModel.__new__(pipeline.Qwen3NextModel)
    torch.nn.Module.__init__(inner)
    inner.layers = torch.nn.ModuleList()
    for mlp in mlps:
        layer = torch.nn.Module()
        layer.mlp = mlp
        inner.layers.append(layer)
    inner.start_layer = 0
    inner.end_layer = len(mlps)
    return SimpleNamespace(model=inner)


def mock_collective(monkeypatch, events):
    monkeypatch.setattr(pipeline, "get_tp_group", lambda: SimpleNamespace(device_group="test-group"))

    def all_reduce(tensor, *, group, async_op):
        assert group == "test-group" and async_op
        events.append("reduce")

        def wait():
            events.append("wait")
            tensor.mul_(2)

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)


@pytest.mark.parametrize("chunks", [2, 4, 16])
@pytest.mark.parametrize("shape", [(1, 7, 4), (7, 4)])
def test_chunked_results_and_wait_order(monkeypatch, chunks, shape):
    events = []
    mlp = make_mlp(events)
    x = torch.randn(*shape)
    reference = mlp(x)
    assert pipeline.enable_hpu_qwen3_mlp_chunking(make_model([mlp]), chunks, 2) == 1
    events.clear()
    mock_collective(monkeypatch, events)
    torch.testing.assert_close(mlp(x), reference)
    count = len(torch.chunk(x, chunks, dim=x.ndim - 2))
    assert events == ["gate", "down", "reduce"] * count + ["wait"] * count


@pytest.mark.parametrize("shape", [(1, 1, 4), (2, 8, 4), (1, 4)])
def test_decode_and_unqualified_layout_use_original_path(monkeypatch, shape):
    events = []
    mlp = make_mlp(events)
    x = torch.randn(*shape)
    reference = mlp(x)
    assert pipeline.enable_hpu_qwen3_mlp_chunking(make_model([mlp]), 2, 2) == 1
    events.clear()
    mock_collective(monkeypatch, events)
    torch.testing.assert_close(mlp(x), reference)
    assert events == ["gate", "down"]


@pytest.mark.parametrize("reason", ["decode", "batch", "missing_lengths"])
def test_flattened_input_requires_single_prefill_metadata(monkeypatch, single_prefill_context, reason):
    metadata = single_prefill_context
    if reason == "decode":
        metadata.is_prompt = False
    elif reason == "batch":
        metadata.seq_lens_tensor = torch.ones(2)
    else:
        metadata.seq_lens_tensor = None
    events = []
    mlp = make_mlp(events)
    x = torch.randn(8, 4)
    reference = mlp(x)
    assert pipeline.enable_hpu_qwen3_mlp_chunking(make_model([mlp]), 2, 2) == 1
    events.clear()
    mock_collective(monkeypatch, events)
    torch.testing.assert_close(mlp(x), reference)
    assert events == ["gate", "down"]


@pytest.mark.parametrize("reason", ["deferred", "tp1", "expert", "input_layout"])
def test_unsupported_later_layer_does_not_mutate_earlier_layers(reason):
    first, last = make_mlp([]), make_mlp([])
    if reason == "deferred":
        last.down_proj.reduce_results = False
    elif reason == "tp1":
        last.down_proj.tp_size = 1
    elif reason == "expert":
        last.expert_gate = torch.nn.Identity()
    else:
        last.down_proj.input_is_parallel = False
    assert pipeline.enable_hpu_qwen3_mlp_chunking(make_model([first, last]), 2, 2) == 0
    assert type(first) is pipeline.Qwen2MoeMLP
    assert not hasattr(first, "_hpu_mlp_chunks")


def test_disabled_and_invalid_configuration():
    model = make_model([make_mlp([])])
    assert pipeline.enable_hpu_qwen3_mlp_chunking(model, 1, 2) == 0
    for chunks, threshold in [(0, 2), (2, 1)]:
        with pytest.raises(ValueError):
            pipeline.enable_hpu_qwen3_mlp_chunking(model, chunks, threshold)


def test_repeated_enable_preserves_parameters():
    mlp = make_mlp([])
    model = make_model([mlp])
    original_weight = mlp.down_proj.weight
    for chunks in [2, 4]:
        assert pipeline.enable_hpu_qwen3_mlp_chunking(model, chunks, 2) == 1
        assert mlp.down_proj.weight is original_weight
        assert mlp._hpu_mlp_chunks == chunks
