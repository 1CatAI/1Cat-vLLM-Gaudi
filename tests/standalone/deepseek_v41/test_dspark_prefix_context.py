# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.v1.worker.deepseek_v41_prefix import draft_context_views, restore_draft_context


def program():
    layers = []
    for index in range(3):
        attention = SimpleNamespace(swa=torch.full((256, 528), index + 1, dtype=torch.uint8))
        if index == 1:
            attention.swa_decoded = SimpleNamespace(swa=torch.full((256, 512), 1., dtype=torch.bfloat16))
        layers.append(SimpleNamespace(attention=attention))
    return SimpleNamespace(draft=SimpleNamespace(layers=layers))


def test_prefix_restores_draft_history_after_an_intervening_request():
    owner = program()
    saved = tuple(value.clone() for value in draft_context_views(owner))
    for value in draft_context_views(owner):
        value.zero_()
    restore_draft_context(owner, saved)
    assert all(torch.equal(value, expected) for value, expected in zip(draft_context_views(owner), saved, strict=True))
    draft_context_views(owner)[0].fill_(255)
    assert saved[0].eq(1).all()


def test_draft_layout_failure_precedes_any_restore_write():
    owner = program()
    saved = tuple(value.clone() for value in draft_context_views(owner))
    for value in draft_context_views(owner):
        value.zero_()
    saved = (*saved[:-1], saved[-1].float())
    with pytest.raises(ValueError, match="state layout"):
        restore_draft_context(owner, saved)
    assert all(not value.any() for value in draft_context_views(owner))


def test_ordinary_prefix_has_no_draft_state():
    owner = SimpleNamespace(draft=None)
    assert draft_context_views(owner) == ()
    restore_draft_context(owner, ())
