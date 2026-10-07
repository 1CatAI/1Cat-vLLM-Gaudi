# SPDX-License-Identifier: Apache-2.0
import gc
import weakref
from types import SimpleNamespace
from tools.deepseek_v41_sampling_fixture import OfficialSamplerCache, full_official_sampler


def test_full_sampler_does_not_retain_the_model_owner():
    class Owner:
        pass
    owner = Owner()
    owner.gather = lambda value, dim: ('gathered', value, dim)
    reference = weakref.ref(owner)
    sample = lambda value, draw, filtered: (value, draw, filtered)
    full = full_official_sampler(owner.gather, sample)
    del owner
    gc.collect()
    assert reference() is None
    assert full('logits', 'device-draw') == (('gathered', 'logits', -1), 'device-draw', True)


def test_each_head_contract_is_compiled_once_without_model_keys():
    functions = []
    cache = OfficialSamplerCache(lambda value, draw: (value, draw),
                                 lambda function: functions.append(function) or function)
    stages = [SimpleNamespace(bf16_head=True), SimpleNamespace(bf16_head=True), SimpleNamespace(bf16_head=False)]
    a, b, c = [cache.get(stage.bf16_head) for stage in stages]
    assert a is b and c is not a and len(functions) == 2
    for function in functions:
        values = [cell.cell_contents for cell in function.__closure__]
        assert not any(value is stage for value in values for stage in stages)
    assert set(cache._compiled) == {True, False}
