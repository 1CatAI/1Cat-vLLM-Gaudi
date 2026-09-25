# SPDX-License-Identifier: Apache-2.0
"""Scheduler preemption must retire state and replay the generated prefix."""
from types import SimpleNamespace

import pytest

from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
from vllm_gaudi.ops.deepseek_v41_batch_state import BatchStageState
from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState, V41ModelRunner
from vllm_gaudi.v1.worker.deepseek_v41_v2_runner import V41V2ModelRunner


def fixture():
    runner = object.__new__(V41V2ModelRunner)
    runner.prefix_checkpoints = None
    bank = object.__new__(BatchStageState)
    bank.single_owner = None
    bank.slots = RequestSlots(1)
    bank.acquire = bank.slots.acquire
    bank.acquire("a")
    released = []
    runner.model = SimpleNamespace(batch_state=bank,
                                   engram_host=SimpleNamespace(release_request=released.append),
                                   pp_rank=0,
                                   tp_rank=0)
    runner.state = SimpleNamespace()
    runner.requests = {"a": RequestState("a", [1, 2], [], None, ([1], ), 6, [3, 4, 5, 6, 7])}
    runner.request_batches = object()
    runner.active_request = "a"
    runner.audit = {}
    runner.verify_timing = None
    runner.verify_records = {}
    runner.encoder_cache = {}
    return runner, bank, released


def scheduled(*, preempted=(), resumed=False, computed=0, blocks=([9], ), output_count=5):
    cached = SimpleNamespace(req_ids=["a"] if resumed else [],
                             resumed_req_ids={"a"} if resumed else set(),
                             new_block_ids=[blocks],
                             num_computed_tokens=[computed],
                             num_output_tokens=[output_count],
                             new_token_ids=[],
                             all_token_ids={})
    return SimpleNamespace(scheduled_new_reqs=[],
                           scheduled_cached_reqs=cached,
                           finished_req_ids=set(),
                           free_encoder_mm_hashes=[],
                           preempted_req_ids=set(preempted))


def test_preempt_resume_retires_slot_history_and_retains_generated_prefix():
    runner, bank, released = fixture()
    old = bank.slots.owners["a"]
    runner._update(scheduled(preempted=("a", )))
    assert not bank.slots.owners and released == ["a"] and runner.active_request is None
    other = bank.acquire("other")
    assert other.index == old.index and other.generation > old.generation
    bank.release("other")
    runner._update(scheduled(resumed=True))
    request = runner.requests["a"]
    assert request.tokens == [1, 2, 3, 4, 5, 6, 7] and request.output == [3, 4, 5, 6, 7]
    assert request.num_computed_tokens == 0 and request.block_ids == ([9], )
    assert request.decode_start == 7
    assert bank.acquire("a").generation > other.generation
    with pytest.raises(RuntimeError, match="Stale"):
        bank.slots.submitted(old, None)


@pytest.mark.parametrize("computed,blocks", [(1, ([9], )), (0, None)])
def test_resume_rejects_unrestored_partial_state_before_retirement(computed, blocks):
    runner, bank, released = fixture()
    old = bank.slots.owners["a"]
    with pytest.raises(ValueError, match="recomputation"):
        runner._update(scheduled(resumed=True, computed=computed, blocks=blocks))
    assert bank.slots.owners["a"] == old and not released


def test_generated_prefix_replays_as_prefill_after_prompt_end(monkeypatch):
    runner, _, _ = fixture()
    request = runner.requests["a"]
    request.num_computed_tokens = 3
    request.recompute_until = len(request.tokens)
    calls = []
    runner.round_timing_enabled = False
    runner._bind_request = lambda request: None
    runner.use_dspark = False
    runner.model_config = SimpleNamespace(max_model_len=1048576)
    runner.model.program = SimpleNamespace(length=1048576)
    import torch
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    runner.model.last_aux = None
    runner.model.complete_step = lambda count: None
    runner.positions = list(range(7))
    runner._insert = lambda aux, positions: None
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_last_rank=False),
                                generation=0,
                                drain=lambda: None,
                                complete_packet=lambda: None)
    runner._forward = lambda rid, tokens, start, **kw: calls.append((tokens, start, kw["decode"]))
    V41ModelRunner._execute_request(runner, SimpleNamespace(scheduled_spec_decode_tokens={}), "a", 2)
    assert calls == [([4], 3, False), ([5], 4, False)]
    assert runner.pending[5] is False  # Still rebuilding; no extra sampled token.


def test_preemption_cannot_authorize_speculative_next_input_prefix():
    record = SimpleNamespace(request_id="a")
    step = SimpleNamespace(finished_req_ids=set(), num_scheduled_tokens={"a": 1}, preempted_req_ids={"a"})
    assert not V41V2ModelRunner._continuation_authorized(record, step)


def test_auxiliary_checkpoint_cannot_start_native_prefix_before_state_publish():
    record = SimpleNamespace(request_id="a")
    step = SimpleNamespace(finished_req_ids=set(), num_scheduled_tokens={"a": 1}, auxiliary_prefix_operations=object())
    assert not V41V2ModelRunner._continuation_authorized(record, step)
