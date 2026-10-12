# SPDX-License-Identifier: Apache-2.0
"""Protocol cache restores fresh draft, repair and publication bindings."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_draft_replay import NativeDraftProtocol


class Draft(torch.nn.Module):

    def forward_local(self, proposed, positions):
        return proposed + positions, positions * 2

    def verify_and_propose(self, value, proposed, control):
        return value * 2, proposed + control

    def verify_and_propose_sampled_bound(self, value, proposed, control, *, repair_frame, known_stochastic, full):
        return value * 3 + repair_frame.bias + int(full), proposed + control

    def verify_and_propose_sampled_full(self, value, proposed, control, *, repair_frame, known_stochastic):
        return value * 4 + repair_frame.bias, proposed + control


class Publication(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(6, 8))

    def forward(self, values, control=None):
        self.state.copy_(values[0] if control is None else values + control)


def protocol(flags, bias):
    owner = NativeDraftProtocol.__new__(NativeDraftProtocol)
    torch.nn.Module.__init__(owner)
    owner.draft = Draft()
    owner.sampled, owner.full, owner.full_main = flags
    owner.known_stochastic = False
    owner.repair_frame = torch.nn.Module()
    owner.repair_frame.register_buffer("bias", torch.full((1, ), bias))
    owner.state_publication, owner.input_publication = Publication(), Publication()
    owner.fixed = (torch.ones(6, 8), torch.ones(6, 8), torch.ones(6, 8))
    return owner


@pytest.mark.parametrize("flags",
                         ((False, False, False), (True, False, False), (True, True, True), (True, True, False)))
def test_cached_protocol_uses_fresh_state_and_preserves_protocol_choice(tmp_path, monkeypatch, flags):
    from torch._dynamo.backends import registry

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setenv("VLLM_HPU_DSV41_FRONTEND_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("DSV41_SERVING_RUNTIME", "runtime-proof")
    monkeypatch.setenv("DSV41_SERVING_COMPILE_IDENTITY", "1" * 64)
    monkeypatch.setattr(registry, "lookup_backend", lambda name: lambda graph, inputs: graph.forward)
    old = protocol(flags, 1.)
    old._new_entry()(*old.fixed)
    before = len(list(tmp_path.rglob("*.bin")))
    fresh = protocol(flags, 7.)
    execute = fresh._new_entry()
    for scalar in (1., 2., 3.):
        values = (torch.full((6, 8), scalar), *fresh.fixed[1:])
        reference = fresh._compiled_protocol(*values)
        actual = execute(*values)
        assert all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True))
        assert torch.equal(fresh.state_publication.state, actual[0])
        assert torch.equal(fresh.input_publication.state, actual[0] + values[2])
    assert len(list(tmp_path.rglob("*.bin"))) == before


def test_draft_body_restores_without_recapturing(tmp_path, monkeypatch):
    from torch._dynamo.backends import registry
    from vllm_gaudi.ops.deepseek_v41_draft_body_replay import NativeDraftBody

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setenv("VLLM_HPU_DSV41_FRONTEND_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("DSV41_SERVING_RUNTIME", "runtime-proof")
    monkeypatch.setenv("DSV41_SERVING_COMPILE_IDENTITY", "1" * 64)
    monkeypatch.setattr(registry, "lookup_backend", lambda name: lambda graph, inputs: graph.forward)
    for repetition in range(2):
        owner = NativeDraftBody.__new__(NativeDraftBody)
        torch.nn.Module.__init__(owner)
        owner.draft = Draft()
        token, positions = torch.ones(1), torch.arange(5)
        owner.fixed = (token, token, token, token, positions)
        execute = owner._new_entry()
        for value in (1., 2., 3.):
            token.fill_(value)
            actual = execute(*owner.fixed)
            reference = owner.draft.forward_local(token, positions)
            assert all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True))
    assert len([path for path in tmp_path.rglob("*.bin") if path.parent.name.startswith("rank")]) == 1
