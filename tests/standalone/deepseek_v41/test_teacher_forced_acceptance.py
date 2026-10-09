# SPDX-License-Identifier: Apache-2.0
"""Reject mismatched prefixes and check probability overlap, not greedy IDs."""
import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[3] / 'tools/check_deepseek_v41_teacher_forced_acceptance.py'
SPEC = importlib.util.spec_from_file_location('teacher_acceptance', PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fixture():
    p = torch.full((5, 129280), -100.)
    q = p.clone()
    p[:, :2], q[:, :2] = torch.tensor([2., 0.]), torch.tensor([0., 2.])
    return dict(teacher_forced=True, prefix_sha256=['a' * 64] * 4,
                reference_target_logits=p, reference_draft_logits=q,
                candidate_target_logits=p, candidate_draft_logits=q)


def test_identical_probability_chain_preserves_alpha():
    result = AUDIT.evaluate(fixture())
    assert result['passed'] and result['minimum_delta'] == 0.
    assert 0. < result['alpha_reference'][0] < 1.


def test_overlap_regression_is_rejected():
    case = fixture()
    case['reference_draft_logits'] = case['reference_target_logits']
    result = AUDIT.evaluate(case)
    assert not result['passed'] and result['mean_delta'] < 0.


def test_different_prefix_cannot_qualify_same_distribution():
    case = fixture()
    case['prefix_sha256'][-1] = 'b' * 64
    with pytest.raises(ValueError, match='exact prefix'):
        AUDIT.evaluate(case)
