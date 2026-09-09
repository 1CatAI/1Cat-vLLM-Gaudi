# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm_gaudi.compilation.deepseek_v4_sinkhorn import fuse_sinkhorn4


def program(value, repeat=20, epsilon=1e-6, observe=False):
    value = torch.softmax(value, -1) + epsilon
    value = value / (value.sum(-2, keepdim=True) + epsilon)
    observed = value
    for _ in range(repeat - 1):
        value = value / (value.sum(-1, keepdim=True) + epsilon)
        value = value / (value.sum(-2, keepdim=True) + epsilon)
    return (value, observed) if observe else value


def test_complete_chain_keeps_softmax_and_epsilon():
    graph = make_fx(lambda x: program(x))(torch.randn(1, 4, 4))
    assert fuse_sinkhorn4(graph, torch.ops.aten.clone.default) == 1
    fused = next(node for node in graph.graph.nodes if node.target is torch.ops.aten.clone.default)
    assert fused.args[0].target is torch.ops.aten.add.Tensor
    assert fused.args[0].args[0].target is torch.ops.aten._softmax.default
    assert not any(node.target is torch.ops.aten.div.Tensor for node in graph.graph.nodes)


@pytest.mark.parametrize("batch,dtype,repeat,epsilon,observe", [
    (2, torch.float32, 20, 1e-6, False),
    (1, torch.bfloat16, 20, 1e-6, False),
    (1, torch.float32, 19, 1e-6, False),
    (1, torch.float32, 20, 1e-5, False),
    (1, torch.float32, 20, 1e-6, True),
])
def test_other_shapes_parameters_and_observed_intermediates_retained(batch, dtype, repeat, epsilon, observe):
    graph = make_fx(lambda x: program(x, repeat, epsilon, observe))(torch.randn(batch, 4, 4).to(dtype))
    before = graph.code
    assert fuse_sinkhorn4(graph, torch.ops.aten.clone.default) == 0
    assert graph.code == before
