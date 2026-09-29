# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch.fx import Graph, GraphModule

from vllm_gaudi.compilation.deepseek_v41_tp4 import hoist_recipe_scalars


class Recipe(torch.nn.Module):
    _recipe_id = 1
    is_dynamic = _has_randoms = False
    _in_to_out_dups = None

    def forward(self, value, scalar):
        return value * scalar


def graph(*, literal=2., dynamic=False, aliased=False):
    root = torch.nn.Module()
    root.recipe = Recipe()
    if aliased:
        root.recipe._in_to_out_dups = {1: 0}
    g = Graph()
    x = g.placeholder("x")
    arg = g.placeholder("scalar") if dynamic else literal
    scalar = g.call_function(torch.ops.aten.scalar_tensor.default, (arg,),
                             dict(dtype=torch.float32, device=torch.device("cpu")))
    output = g.call_module("recipe", (x, scalar))
    g.output(output)
    return GraphModule(root, g)


def test_fixed_scalars_are_resident_and_changing_inputs_still_reach_recipe():
    gm = graph()
    assert hoist_recipe_scalars(gm, device_type="cpu") == 1
    for value in (1., 7., -2.):
        assert gm(torch.tensor(value)).item() == value * 2
    assert not any(n.target is torch.ops.aten.scalar_tensor.default for n in gm.graph.nodes)
    assert hoist_recipe_scalars(gm, device_type="cpu") == 0


@pytest.mark.parametrize("kwargs", [dict(dynamic=True), dict(aliased=True)])
def test_dynamic_or_mutated_scalar_is_not_frozen(kwargs):
    gm = graph(**kwargs)
    assert hoist_recipe_scalars(gm, device_type="cpu") == 0


def test_literal_nan_codec_sentinel_can_be_retained():
    gm = graph(literal=float("nan"))
    assert hoist_recipe_scalars(gm, device_type="cpu") == 1
    assert torch.isnan(gm(torch.tensor(1.)))


def test_hpu_nan_sentinel_has_identical_bits_and_masking():
    import os
    if os.getenv("DSV41_TEST_HPU") != "1":
        pytest.skip("Requires exclusive HPU lease")
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend
    direct = torch.scalar_tensor(float("nan"), dtype=torch.float32, device="hpu")
    retained = torch.tensor(float("nan"), dtype=torch.float32).to("hpu")
    assert direct.cpu().view(torch.int32).item() == retained.cpu().view(torch.int32).item()

    def codec(value):
        return torch.where(value >= 0, value, float("nan"))
    compiled = torch.compile(codec, backend=make_backend(), fullgraph=True, dynamic=False)
    value = torch.empty(8, device="hpu")
    for shift in (0, 2, -2, 1):
        host = torch.arange(8).float() - 4 + shift
        value.copy_(host)
        actual = compiled(value).cpu()
        wanted = codec(host)
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0, equal_nan=True)
