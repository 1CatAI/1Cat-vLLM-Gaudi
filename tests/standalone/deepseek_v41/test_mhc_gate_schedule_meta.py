# SPDX-License-Identifier: Apache-2.0
"""Real native operator metadata and gate-only compiler isolation."""
import os

import habana_frameworks.torch.core  # noqa: F401
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm_gaudi.compilation.deepseek_v41_overlap import independent_mhc_nodes
from vllm_gaudi.ops.deepseek_v41_mhc_gate_schedule import communication_gates_post

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


@pytest.mark.parametrize('tokens', [1, 2, 6])
@pytest.mark.parametrize('ranks', [2, 4])
def test_real_gate_branch_does_not_read_peer_or_residual(tokens, ranks):
    values = (
        torch.empty(ranks, tokens, 5120, dtype=torch.bfloat16, device='meta'),
        torch.empty(tokens, 4, 5120, dtype=torch.bfloat16, device='meta'),
        torch.empty(tokens, 25, device='meta'),
        torch.empty(3, device='meta'),
        torch.empty(24, device='meta'),
    )
    updated, collapsed, gates = communication_gates_post(*values, 1e-20)
    assert (updated.shape, updated.dtype) == (values[1].shape, torch.bfloat16)
    assert (collapsed.shape, collapsed.dtype) == ((tokens, 5120), torch.bfloat16)
    assert (gates.shape, gates.dtype) == ((tokens, 24), torch.float32)
    graph = make_fx(lambda *args: communication_gates_post(*args, 1e-20))(*values)
    selected = independent_mhc_nodes(graph, [0, 1])
    assert any('mhc_gates_f32' in str(node.target) for node in selected)
    assert not any('post_collapse' in str(node.target) for node in selected)
    placeholders = [node for node in graph.graph.nodes if node.op == 'placeholder']
    assert all(placeholders[0] not in node.all_input_nodes and placeholders[1] not in node.all_input_nodes
               for node in selected)
