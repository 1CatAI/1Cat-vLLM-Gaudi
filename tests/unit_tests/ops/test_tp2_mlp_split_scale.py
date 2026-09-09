# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm_gaudi.ops import tp2_mlp_split_scale as split


class Projection(torch.nn.Module):

    def __init__(self, output):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.weight_scale_inv = torch.ones(1)
        self.output = output
        self.calls = []

    def forward(self, x):
        self.calls.append(x)
        return self.output, None


def bare_module(cls):
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    return module


def make_mlp():
    from vllm_gaudi.ops.hpu_fp8 import Fp8LinearMethod

    mlp = bare_module(Qwen2MoeMLP)
    for name, cls, shape in (("gate_up_proj", MergedColumnParallelLinear, (17408, 5120)),
                             ("down_proj", RowParallelLinear, (5120, 8704))):
        projection = bare_module(cls)
        projection.weight = torch.nn.Parameter(torch.empty(shape, device="meta", dtype=torch.float8_e4m3fn),
                                               requires_grad=False)
        projection.weight_scale_inv = torch.nn.Parameter(torch.empty(shape[0], device="meta"), requires_grad=False)
        projection.tp_size = 2
        projection.gather_output = False
        projection.input_is_parallel = True
        projection.return_bias = True
        projection.bias = None
        projection.quant_method = Fp8LinearMethod.__new__(Fp8LinearMethod)
        projection.quant_method.block_quant = True
        setattr(mlp, name, projection)
    mlp.act_fn = bare_module(SiluAndMul)
    mlp.expert_gate = None
    return mlp


@pytest.fixture
def supported_runtime(monkeypatch):
    import vllm.distributed

    monkeypatch.setattr(split.hpu_ops, "is_hpu_gaudi2", True)
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setenv("VLLM_HPU_FORCE_CHANNEL_FP8", "1")
    monkeypatch.setenv("VLLM_HPU_TP2_NATIVE_DYNAMIC_QUANT", "1")


def test_fp32_coefficient_is_preserved_until_result_rounding(monkeypatch):
    generator = torch.Generator().manual_seed(856)
    raw = torch.randn(1, 17408, generator=generator).bfloat16()
    channel = torch.rand(17408, generator=generator)
    scale = torch.tensor([[0.031234567]])
    monkeypatch.setattr(split.hpu_ops, "dynamic_quant", lambda x: (x, scale))
    calls = []

    def gemm(*args):
        calls.append(args)
        return raw

    monkeypatch.setattr(torch.ops.hpu, "fp8_gemm_v2", gemm, raising=False)
    expected_scaled = (raw.float() * (scale * channel)).bfloat16()
    gate, up = expected_scaled.chunk(2, -1)
    expected = torch.nn.functional.silu(gate) * up
    actual = split.split_scaled_gate_up(torch.zeros(1, 5120), torch.empty(0), channel)
    assert torch.equal(actual, expected)
    assert calls[0][-2:] == (None, None)
    assert not torch.equal(expected_scaled, raw * (scale * channel).bfloat16())


@pytest.mark.parametrize("prompt,direct,rows,accepted,full_query", [
    (False, True, 1, None, False),
    (True, True, 1, None, False),
    (False, False, 1, None, False),
    (False, True, 2, None, False),
    (False, True, 1, object(), False),
    (False, True, 1, None, True),
])
def test_decode_scope_and_down_projection_are_preserved(monkeypatch, prompt, direct, rows, accepted, full_query):
    original = bare_module(Qwen2MoeMLP)
    original.gate_up_proj = Projection(torch.ones(rows, 8, dtype=torch.bfloat16))
    original.down_proj = Projection(torch.zeros(rows, 5120, dtype=torch.bfloat16))
    original.act_fn = torch.nn.Identity()
    original.expert_gate = None
    wrapper = split.HpuSplitScaleMLP(original)
    metadata = SimpleNamespace(is_prompt=prompt,
                               direct_gdn_state=direct,
                               num_accepted_tokens=accepted,
                               dflash_full_query=full_query)
    monkeypatch.setattr(split, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata))
    activation = torch.full((rows, 8), 2, dtype=torch.bfloat16)
    monkeypatch.setattr(split, "split_scaled_gate_up", lambda *args: activation)
    result = wrapper(torch.zeros(rows, 5120, dtype=torch.bfloat16))
    eligible = not prompt and direct and rows == 1 and accepted is None and not full_query
    assert len(original.gate_up_proj.calls) == (0 if eligible else 1)
    assert len(original.down_proj.calls) == 1
    assert torch.equal(original.down_proj.calls[0], activation if eligible else original.gate_up_proj.output)
    assert result is original.down_proj.output
    assert set(wrapper.state_dict()) == set(original.state_dict())
    assert wrapper.gate_up_proj.weight is original.gate_up_proj.weight


@pytest.mark.parametrize("failure", ["bias", "geometry", "scale", "moe", "sequence", "alternate"])
def test_prepare_rejects_entire_topology_before_mutation(supported_runtime, failure):
    first, second = make_mlp(), make_mlp()
    layers = [SimpleNamespace(mlp=first), SimpleNamespace(mlp=second)]
    if failure == "bias":
        second.gate_up_proj.bias = torch.nn.Parameter(torch.zeros(1))
    elif failure == "geometry":
        second.gate_up_proj.weight = torch.nn.Parameter(torch.empty(1, device="meta"))
    elif failure == "scale":
        second.gate_up_proj.weight_scale_inv = torch.nn.Parameter(torch.empty(17408, dtype=torch.bfloat16))
    elif failure == "moe":
        second.expert_gate = torch.nn.Identity()
    elif failure == "sequence":
        layers[1].use_attn_reduce_scatter_for_moe = True
    else:
        second.gate_up_proj = torch.nn.Identity()
    with pytest.raises(RuntimeError, match="Split MLP scaling"):
        split.prepare_mlp_split_scale(layers)
    assert layers[0].mlp is first and layers[1].mlp is second


def test_prepare_is_idempotent_and_keeps_loaded_parameters(supported_runtime):
    original = make_mlp()
    layers = [SimpleNamespace(mlp=original)]
    split.prepare_mlp_split_scale(layers)
    installed = layers[0].mlp
    split.prepare_mlp_split_scale(layers)
    assert layers[0].mlp is installed
    assert installed.gate_up_proj is original.gate_up_proj
    assert installed.down_proj is original.down_proj
    assert list(installed.state_dict()) == list(original.state_dict())
