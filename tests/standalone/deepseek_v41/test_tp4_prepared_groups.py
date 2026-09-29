# SPDX-License-Identifier: Apache-2.0
"""Keep mutable state and bucket transitions exact when reusing exported groups."""
from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func

from vllm_gaudi.compilation.deepseek_v41_prepared import PreparedTP4Group, input_contract


@pytest.fixture(autouse=True)
def cpu_export_only(monkeypatch):
    # Other test modules may register the HPU backend during collection.
    # CPU reference exports must not acquire a device to inspect its stream.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)


class StatefulGroup(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.arange(64).reshape(8, 8).float() / 100)
        self.register_buffer("state", torch.zeros(32, 8))

    def forward(self, value, positions, extras):
        output = value @ self.weight + extras[0]
        self.state.index_copy_(0, positions, output)
        return output, self.state.index_select(0, positions) + extras[1], None


def backend(graph, inputs):
    return aot_autograd(fw_compiler=lambda gm, _: make_boxed_func(gm.forward),
                        keep_inference_input_mutations=True)(graph, inputs)


def test_input_contract_tracks_layout_and_aliases_without_storage_identity():
    first = torch.zeros(3, 8)
    second = first.clone()
    assert input_contract((first, first)) == input_contract((second, second))
    assert input_contract((first, first)) != input_contract((first, second))
    assert input_contract((first[:2],)) != input_contract((first[1:],))
    assert input_contract((first,)) != input_contract((first.t(),))


@torch.inference_mode()
def test_prepared_group_reuses_and_rebinds_state_across_shapes_and_generation():
    module = StatefulGroup()
    owner = SimpleNamespace(generation=1, precision_fingerprint="test", search_length=32768)
    prepared = PreparedTP4Group(module, owner, backend)
    for index, count in enumerate((1, 2, 6, 1, 2, 6)):
        args = (torch.randn(count, 8), torch.arange(count - 1, -1, -1),
                (torch.randn(count, 8), torch.randn(count, 8)))
        initial = module.state.clone()
        expected = module(*args)
        expected_state = module.state.clone()
        module.state.copy_(initial)
        observed = prepared(*args)
        assert torch.equal(observed[0], expected[0])
        assert torch.equal(observed[1], expected[1])
        assert observed[2] is None
        assert torch.equal(module.state, expected_state)
    assert prepared.preparations == 3
    previous_state = module.state
    previous_value = previous_state.clone()
    module.state = torch.zeros_like(previous_state)
    owner.generation += 1
    args = (args[0] + 1, args[1], args[2])
    expected = module(*args)
    expected_state = module.state.clone()
    module.state.zero_()
    observed = prepared(*args)
    assert torch.equal(observed[0], expected[0]) and torch.equal(observed[1], expected[1])
    assert torch.equal(module.state, expected_state)
    assert prepared.preparations == 4 and len(prepared.variants) == 1
    assert torch.equal(previous_state, previous_value)
    owner.search_length = 65536
    prepared(*args)
    assert prepared.preparations == 5


@torch.inference_mode()
def test_packed_views_keep_independent_dynamic_bindings_on_new_storage():
    class PacketGroup(torch.nn.Module):
        def forward(self, rows):
            return rows[0].float().sum() + rows[1].float().sum() * 3

    module = PacketGroup()
    owner = SimpleNamespace(generation=1, search_length=32768)
    prepared = PreparedTP4Group(module, owner, backend)
    for value in range(3):
        packet = torch.full((2, 1, 6, 264), value, dtype=torch.uint8)
        packet[1].add_(7)
        rows = packet[0], packet[1]
        assert torch.equal(prepared(rows), module(rows))
    assert prepared.preparations == 1


def test_prepared_group_rejects_grad_execution():
    prepared = PreparedTP4Group(StatefulGroup(), SimpleNamespace(), backend)
    with pytest.raises(RuntimeError, match="inference mode"):
        prepared(torch.ones(1, 8))


@torch.inference_mode()
def test_retained_group_never_reuses_a_different_visible_prefix():
    owner = SimpleNamespace(generation=1, search_length=32768, decode_token_bound=2)

    class PrefixGroup(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("rows", torch.arange(8).float())

        def forward(self, value):
            return value + self.rows[:owner.decode_token_bound].sum()

    module = PrefixGroup()
    prepared = PreparedTP4Group(module, owner, backend)
    for bound in (2, 5, 2, 5):
        owner.decode_token_bound = bound
        value = torch.randn(1, 8)
        assert torch.equal(prepared(value), module(value))
    assert prepared.preparations == 2
