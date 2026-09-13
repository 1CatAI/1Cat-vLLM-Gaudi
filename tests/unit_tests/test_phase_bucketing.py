# SPDX-License-Identifier: Apache-2.0
"""Phase overrides must preserve bucket generation and attention guards."""

import pytest

from vllm_gaudi.extension.bucketing.common import HPUBucketingManager
from vllm_gaudi.extension.bucketing.exponential import ExponentialBucketingStrategy
from vllm_gaudi.extension.bucketing.linear import LinearBucketingStrategy
from vllm_gaudi.extension.bucketing.padding_aware import PaddingAwareBucketingStrategy
from vllm_gaudi.extension.runtime import clear_config, get_config
from vllm_gaudi.extension.utils import SlicedFusedSDPABase


@pytest.fixture(autouse=True)
def clean_phase_config(monkeypatch):
    for name in ('VLLM_BUCKETING_STRATEGY', 'VLLM_PROMPT_BUCKETING_STRATEGY', 'VLLM_DECODE_BUCKETING_STRATEGY',
                 'VLLM_EXPONENTIAL_BUCKETING', 'VLLM_BUCKETING_FROM_FILE', 'VLLM_HPU_FSDPA_SLICE_ENABLED'):
        monkeypatch.delenv(name, raising=False)
    previous = HPUBucketingManager._active_instance
    clear_config()
    yield
    clear_config()
    HPUBucketingManager._active_instance = previous


def _manager(speculative_tokens=0):
    manager = HPUBucketingManager()
    manager.initialize(max_num_seqs=16,
                       max_num_prefill_seqs=4,
                       block_size=128,
                       max_num_batched_tokens=1024,
                       max_model_len=2048,
                       num_speculative_tokens=speculative_tokens)
    manager.num_hpu_blocks = 128
    return manager


def _buckets():
    manager = _manager()
    manager.generate_prompt_buckets()
    manager.generate_decode_buckets()
    return manager.prompt_buckets, manager.decode_buckets


@pytest.mark.parametrize('prompt', ['exp', 'lin', 'pad'])
@pytest.mark.parametrize('decode', ['exp', 'lin', 'pad'])
def test_generated_buckets_match_each_independent_global_strategy(monkeypatch, prompt, decode):
    monkeypatch.setenv('VLLM_BUCKETING_STRATEGY', prompt)
    expected_prompt, _ = _buckets()
    monkeypatch.setenv('VLLM_BUCKETING_STRATEGY', decode)
    clear_config()
    _, expected_decode = _buckets()
    monkeypatch.setenv('VLLM_BUCKETING_STRATEGY', 'exp')
    monkeypatch.setenv('VLLM_PROMPT_BUCKETING_STRATEGY', prompt)
    monkeypatch.setenv('VLLM_DECODE_BUCKETING_STRATEGY', decode)
    clear_config()
    assert _buckets() == (expected_prompt, expected_decode)


@pytest.mark.parametrize('strategy,expected', [('exp', ExponentialBucketingStrategy), ('lin', LinearBucketingStrategy),
                                               ('pad', PaddingAwareBucketingStrategy)])
def test_unset_phase_falls_back_to_global(monkeypatch, strategy, expected):
    monkeypatch.setenv('VLLM_BUCKETING_STRATEGY', strategy)
    manager = _manager()
    assert isinstance(manager.get_bucketing_strategy('prompt'), expected)
    assert isinstance(manager.get_bucketing_strategy('decode'), expected)


def test_decode_override_does_not_change_prompt_or_global(monkeypatch):
    monkeypatch.setenv('VLLM_BUCKETING_STRATEGY', 'exp')
    monkeypatch.setenv('VLLM_DECODE_BUCKETING_STRATEGY', 'pad')
    manager = _manager()
    assert isinstance(manager.get_bucketing_strategy(), ExponentialBucketingStrategy)
    assert isinstance(manager.get_bucketing_strategy('prompt'), ExponentialBucketingStrategy)
    assert isinstance(manager.get_bucketing_strategy('decode'), PaddingAwareBucketingStrategy)


@pytest.mark.parametrize('phase', ['prompt', 'decode'])
def test_invalid_phase_strategy_is_rejected(monkeypatch, phase):
    env_name = f'VLLM_{phase.upper()}_BUCKETING_STRATEGY'
    monkeypatch.setenv(env_name, 'invalid')
    with pytest.raises(RuntimeError, match=env_name):
        _manager().get_bucketing_strategy(phase)


def test_invalid_phase_name_is_rejected():
    with pytest.raises(ValueError, match='Invalid bucketing phase'):
        _manager().get_bucketing_strategy('prefill')


@pytest.mark.parametrize('legacy,expected', [('true', ExponentialBucketingStrategy),
                                             ('false', LinearBucketingStrategy)])
def test_legacy_flag_overrides_both_phases(monkeypatch, legacy, expected):
    monkeypatch.setenv('VLLM_EXPONENTIAL_BUCKETING', legacy)
    monkeypatch.setenv('VLLM_PROMPT_BUCKETING_STRATEGY', 'pad')
    monkeypatch.setenv('VLLM_DECODE_BUCKETING_STRATEGY', 'pad')
    manager = _manager()
    assert isinstance(manager.get_bucketing_strategy('prompt'), expected)
    assert isinstance(manager.get_bucketing_strategy('decode'), expected)


def test_file_buckets_take_precedence_over_phase_generation(monkeypatch):
    monkeypatch.setenv('VLLM_BUCKETING_FROM_FILE', 'buckets.json')
    monkeypatch.setenv('VLLM_PROMPT_BUCKETING_STRATEGY', 'lin')
    monkeypatch.setenv('VLLM_DECODE_BUCKETING_STRATEGY', 'pad')
    manager = _manager()
    prompt, decode = [(1, 128, 0)], [(2, 1, 2)]
    monkeypatch.setattr(manager, 'read_from_file', lambda is_prompt: prompt if is_prompt else decode)

    def unexpected_strategy(*args):
        raise AssertionError('File buckets must bypass strategy generation')

    monkeypatch.setattr(manager, 'get_bucketing_strategy', unexpected_strategy)
    manager.generate_prompt_buckets()
    manager.generate_decode_buckets()
    assert manager.prompt_buckets == prompt
    assert manager.decode_buckets == decode


def test_speculative_expansion_uses_phase_selected_decode_seeds(monkeypatch):
    monkeypatch.setenv('VLLM_BUCKETING_STRATEGY', 'pad')
    baseline = _manager(speculative_tokens=7)
    baseline.generate_decode_buckets()
    monkeypatch.setenv('VLLM_BUCKETING_STRATEGY', 'exp')
    monkeypatch.setenv('VLLM_DECODE_BUCKETING_STRATEGY', 'pad')
    clear_config()
    candidate = _manager(speculative_tokens=7)
    candidate.generate_decode_buckets()
    assert candidate.seed_decode_buckets == baseline.seed_decode_buckets
    assert candidate.decode_buckets == baseline.decode_buckets
    assert set(candidate.decode_buckets) > set(candidate.seed_decode_buckets)


@pytest.mark.parametrize('global_strategy,prompt,decode,expected', [
    ('exp', 'pad', 'exp', True),
    ('pad', 'exp', 'pad', False),
    ('exp', None, 'pad', False),
    ('pad', None, 'exp', True),
])
def test_slicing_defaults_follow_prompt_strategy(monkeypatch, global_strategy, prompt, decode, expected):
    monkeypatch.setenv('VLLM_BUCKETING_STRATEGY', global_strategy)
    if prompt is not None:
        monkeypatch.setenv('VLLM_PROMPT_BUCKETING_STRATEGY', prompt)
    monkeypatch.setenv('VLLM_DECODE_BUCKETING_STRATEGY', decode)
    monkeypatch.setattr('vllm_gaudi.extension.features.fsdpa', lambda: object())
    get_config(hw='gaudi2')
    _manager()
    assert get_config().enable_fsdpa_slicing is expected
    base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
    assert base._setup_slicing() is expected


@pytest.mark.parametrize('override,value', [
    ('VLLM_PROMPT_BUCKETING_STRATEGY', 'exp'),
    ('VLLM_EXPONENTIAL_BUCKETING', 'true'),
    ('VLLM_EXPONENTIAL_BUCKETING', 'false'),
    ('VLLM_BUCKETING_FROM_FILE', 'buckets.json'),
])
def test_forced_slicing_rejects_incompatible_prompt_buckets(monkeypatch, override, value):
    monkeypatch.setenv('VLLM_BUCKETING_STRATEGY', 'pad')
    monkeypatch.setenv('VLLM_HPU_FSDPA_SLICE_ENABLED', 'true')
    monkeypatch.setenv(override, value)
    base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
    assert base._setup_slicing() is False
