# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.models.qwen3_next import (
    HpuQwen3DecoderLayerGroup,
    build_hpu_qwen3_layer_groups,
    can_use_hpu_qwen3_layer_groups,
    compile_hpu_qwen3_layer_groups,
)


class _AddLayer(torch.nn.Module):

    def __init__(self, value: int):
        super().__init__()
        self.value = value

    def forward(self, *, positions, hidden_states, residual):
        del positions
        return hidden_states + self.value, residual + self.value


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
    assert [len(group._layers) for group in groups] == [2, 2]


@pytest.mark.parametrize(
    ("layer_groups", "aux_layers", "attn_metadata", "expected"),
    [
        ((HpuQwen3DecoderLayerGroup(tuple()),), [], SimpleNamespace(is_prompt=False), True),
        ((HpuQwen3DecoderLayerGroup(tuple()),), [], SimpleNamespace(is_prompt=True), False),
        ((HpuQwen3DecoderLayerGroup(tuple()),), [1], SimpleNamespace(is_prompt=False), False),
        (None, [], SimpleNamespace(is_prompt=False), False),
        ((HpuQwen3DecoderLayerGroup(tuple()),), [], None, False),
    ],
)
def test_can_use_hpu_qwen3_layer_groups(layer_groups, aux_layers, attn_metadata, expected):
    assert can_use_hpu_qwen3_layer_groups(layer_groups, aux_layers, attn_metadata) is expected
