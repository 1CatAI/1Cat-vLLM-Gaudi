# SPDX-License-Identifier: Apache-2.0
"""Target geometry remains independent of parallel-draft input reservations."""
from types import SimpleNamespace

import pytest

from vllm_gaudi.ops.deepseek_v41_prefill_capacity import (
    prefill_capacity,
    prefill_compute_buckets,
    prefill_target_tokens,
    reserve_dspark_input_slots,
)


@pytest.mark.parametrize('target,tp,requests', [(16384, 4, 1), (8192, 2, 1), (8192, 4, 4), (8190, 4, 1)])
def test_draft_inputs_do_not_reduce_target_geometry_or_grow_worker_storage(target, tp, requests):
    scheduler = SimpleNamespace(max_num_batched_tokens=target, max_num_scheduled_tokens=None, max_num_seqs=requests)
    config = SimpleNamespace(scheduler_config=scheduler,
                             speculative_config=SimpleNamespace(max_num_new_slots_for_drafting=4))
    before = prefill_capacity(target, tp)
    reserve_dspark_input_slots(config)
    reserve_dspark_input_slots(config)
    assert scheduler.max_num_batched_tokens == target + requests * 4
    assert scheduler.max_num_scheduled_tokens == target
    assert prefill_target_tokens(scheduler) == target
    assert prefill_capacity(prefill_target_tokens(scheduler), tp) == before
    assert prefill_compute_buckets(prefill_target_tokens(scheduler)) == prefill_compute_buckets(target)


def test_existing_separate_input_limit_is_preserved():
    scheduler = SimpleNamespace(max_num_batched_tokens=20000, max_num_scheduled_tokens=16384, max_num_seqs=1)
    config = SimpleNamespace(scheduler_config=scheduler,
                             speculative_config=SimpleNamespace(max_num_new_slots_for_drafting=4))
    reserve_dspark_input_slots(config)
    assert scheduler.max_num_batched_tokens == 20000
    assert prefill_target_tokens(scheduler) == 16384
