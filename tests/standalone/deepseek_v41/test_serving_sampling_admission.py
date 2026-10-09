# SPDX-License-Identifier: Apache-2.0
"""A short-prefix micro gate cannot erase failed full-request draft coverage."""
import importlib.util
from pathlib import Path

import pytest


def policy():
    path = Path(__file__).resolve().parents[3] / 'tools/run_deepseek_v41_serving_batch.py'
    spec = importlib.util.spec_from_file_location('serving_sampling_admission', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_draft_coverage_policy


def environment(**values):
    return {'VLLM_HPU_DSV41_DSPARK_' + name: value for name, value in values.items()}


@pytest.mark.parametrize('value', ('1', 'true'))
def test_unchanged_bounded_draft_is_rejected(value):
    with pytest.raises(ValueError, match='complete official requests'):
        policy()(environment(GLOBAL_BOUNDED_SAMPLING=value))


@pytest.mark.parametrize('values', (
    dict(GLOBAL_BOUNDED_SAMPLING='1', WEIGHTED_DRAFT_NUCLEUS='1'),
    dict(GLOBAL_BOUNDED_SAMPLING='0'),
    dict(GLOBAL_BOUNDED_SAMPLING='1', EXACT_DRAFT_SAMPLING='1'),
))
def test_valid_full_mass_draft_paths_remain_allowed(values):
    policy()(environment(**values))


def test_missing_companion_capture_can_be_recovered_without_repeating_formal():
    policy()(environment(GLOBAL_BOUNDED_SAMPLING='1'), trace_only=True)
