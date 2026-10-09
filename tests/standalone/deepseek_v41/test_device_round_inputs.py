# SPDX-License-Identifier: Apache-2.0
"""Accepted-prefix progression and queued-round ownership contracts."""

import pytest
import torch
from types import SimpleNamespace

from vllm_gaudi.ops.deepseek_v41_round_inputs import DeviceRoundInputs
from vllm_gaudi.ops.deepseek_v41_verify import pack_record, verify_control_from_target


@pytest.mark.parametrize("accepted", range(6))
def test_next_round_uses_accepted_history_not_the_speculative_suffix(accepted):
    ids, positions = torch.empty(6, dtype=torch.int64), torch.empty(6, dtype=torch.int32)
    inputs = DeviceRoundInputs(ids, positions)
    inputs.seed("request", [10, 11, 12, 13, 14, 15], 16384, 100, 32768, 9, [9, 8, 7])
    original_control = inputs.control.clone()
    proposed = original_control[7:]
    target = torch.tensor([11, 12, 13, 14, 15, 16])
    if accepted < 5:
        target[accepted] = 50
    _, output, committed, output_count, _, enabled, status = verify_control_from_target(
        target, proposed, inputs.control[:7])
    record = pack_record(inputs.control[:7], committed, output_count, output,
                         torch.tensor([101, 102, 103, 104, 105]), enabled, status)
    histories = torch.tensor([[9, 8, 7], [10, 9, 8], [11, 10, 9], [12, 11, 10],
                              [13, 12, 11], [14, 13, 12], [15, 14, 13]], dtype=torch.int32)
    inputs.next(record, histories)
    count = accepted + 1
    assert torch.equal(inputs.history, histories[count])
    assert ids.tolist() == [int(target[accepted]), 101, 102, 103, 104, 105]
    assert positions.tolist() == list(range(16384 + count, 16390 + count))
    assert inputs.control[:7].tolist() == [10, 6, 5, 100 - count, 16384 + count, 32768, 1]
    assert torch.equal(inputs.controls[0], original_control), "prior queued control must survive the next producer"


def test_cursor_reuse_requires_retiring_its_request():
    inputs = DeviceRoundInputs(torch.empty(6, dtype=torch.int64), torch.empty(6, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="no request owner"):
        inputs.next(torch.empty(16, dtype=torch.int64), torch.empty(7, 3, dtype=torch.int32))
    inputs.seed("a", list(range(6)), 0, 12, 64, 1, [-1, -1, -1])
    with pytest.raises(RuntimeError, match="another request"):
        inputs.retire("b")
    inputs.retire("a")
    inputs.seed("b", list(range(20, 26)), 3, 40, 64, 2, [9, 8, 7])
    assert inputs.owner == "b" and inputs.ids.tolist() == list(range(20, 26))


def test_four_frames_retain_three_outstanding_controls():
    inputs = DeviceRoundInputs(torch.empty(6, dtype=torch.int64), torch.empty(6, dtype=torch.int32), frames=4)
    inputs.seed("a", list(range(10, 16)), 126, 100, 1024, 7, [9, 8, 7])
    histories = torch.arange(21, dtype=torch.int32).reshape(7, 3)
    previous = []
    for committed in (1, 6, 3):
        previous.append(inputs.control.clone())
        record = torch.tensor([int(inputs.control[0]), committed, committed, 5,
                               50, 51, 52, 53, 54, 55, 60, 61, 62, 63, 64, 0])
        inputs.next(record, histories)
    assert inputs.parity == 3
    assert int(inputs.positions[0]) == 136
    assert [int(control[0]) for control in inputs.controls] == [7, 8, 9, 10]
    for control, original in zip(inputs.controls, previous):
        assert torch.equal(control, original)


def test_seed_rejects_out_of_context_before_mutating_device_state():
    inputs = DeviceRoundInputs(torch.empty(6, dtype=torch.int64), torch.empty(6, dtype=torch.int32))
    with pytest.raises(ValueError, match="geometry"):
        inputs.seed("a", list(range(6)), 60, 12, 64, 1, [-1, -1, -1])
    assert inputs.owner is None


def test_startup_warms_both_frames_and_slots_then_retires_the_owner(monkeypatch):
    from vllm_gaudi.ops.deepseek_v41_verify import VerifyRing
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event", lambda: SimpleNamespace())
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    runner = V41ModelRunner.__new__(V41ModelRunner)
    runner.device = torch.device("cpu")
    ids = torch.arange(10, 16)
    positions = torch.arange(512, 518, dtype=torch.int32)
    runner.position_views = {6: positions}
    runner.device_round_inputs = DeviceRoundInputs(ids, positions)
    runner.pp = SimpleNamespace(generation=0)
    runner.model_config = SimpleNamespace(max_model_len=32768)
    runner.verify_ring = VerifyRing(
        "cpu", last_rank=True,
        native_readback=lambda value: (value.clone(), SimpleNamespace(synchronize=lambda: None)))
    order, retired = [], []
    engram = SimpleNamespace(histories=torch.empty(7, 3, dtype=torch.int32))

    def prepare(owner, tokens, lookback):
        engram.histories[0].copy_(lookback)
        for index in range(6):
            engram.histories[index + 1].copy_(torch.cat((tokens[index:index + 1].int(),
                                                       engram.histories[index, :2])))
        return ()

    def target(tokens, token_positions, rows):
        order.append("target")
        return tokens + 1

    def verify(hidden, proposed, metadata, aux, token_positions):
        return (*verify_control_from_target(hidden, proposed, metadata), None, None)

    def draft(metadata, token_positions, output, committed, output_count, anchor, enabled, status):
        record = pack_record(metadata, committed, output_count, output, anchor + torch.arange(1, 6),
                             enabled, status)
        return record, None, None

    consume = runner.verify_ring.consume

    def consume_record(ticket):
        order.append("consume")
        return consume(ticket)

    runner.verify_ring.consume = consume_record
    engram.prepare, engram.retire = prepare, retired.append
    runner.device_round_engram = engram
    runner.model = SimpleNamespace(forward_device_round=target, last_aux=None)
    runner.verify_prefix, runner.draft_from_prefix = verify, draft
    runner._request_speculative_sampling = lambda request: SimpleNamespace(
        parameters=torch.empty(11, 3), seed=torch.zeros(1, dtype=torch.int64),
        counter=torch.zeros(1, dtype=torch.int64), offsets=torch.arange(11),
        proposal=torch.zeros(5, 128), proposal_valid=torch.ones(1, dtype=torch.bool))
    runner.draw_speculative = lambda *args: (None, None, None, None)
    runner.sampled_propose = lambda first, positions, controls: (
        first + torch.arange(1, 6), torch.zeros(5, 128), None, torch.ones(5, dtype=torch.bool))
    runner._prepare_sampled_native_protocols = lambda *args: None

    def sampled_reference(request, hidden, aux, control):
        result = verify(hidden, control[7:], control[:7], aux, positions)
        _, output, committed, count, anchor, enabled, status, _, _ = result
        record, _, confidence = draft(control[:7], positions, output, committed, count, anchor, enabled, status)
        return record, confidence

    runner._sampled_round_reference = sampled_reference
    runner._warm_device_round_inputs()
    assert order == ["target", "target", "consume", "target", "consume", "target", "consume", "consume"] * 2
    assert runner.verify_ring.generations == runner.verify_ring.consumed == [7, 8]
    assert runner.pp.generation == runner.verify_ring.generation == 8
    assert runner.device_round_inputs.parity == 0 and runner.device_round_inputs.owner is None
    assert retired == ["__v41_device_round_warmup__", "__v41_sampled_round_warmup__"]
    assert positions.tolist() == list(range(536, 542))
    assert runner.verify_ring.acquire(6, generation=runner.pp.generation + 1).generation == 9


@pytest.mark.parametrize("replay,tokens,converted", [(True, 6, True), (False, 6, False), (True, 1, False)])
def test_ordinary_model_entry_matches_device_round_engram_dtype(monkeypatch, replay, tokens, converted):
    from vllm_gaudi import envs
    from vllm_gaudi.models.deepseek_v41 import HpuDeepseekV41ForCausalLM
    from vllm_gaudi.ops.deepseek_v41_math import pack_swa, unpack_swa

    for name in ("VLLM_HPU_DSV41_FUSED_STAGE_IO", "VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH",
                 "VLLM_HPU_DSV41_V2_DEVICE_ENGRAM"):
        monkeypatch.setattr(envs, name, False)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_DEVICE_ROUNDS", True)
    model = HpuDeepseekV41ForCausalLM.__new__(HpuDeepseekV41ForCausalLM)
    torch.nn.Module.__init__(model)
    model.pp_rank, model.tensor_parallel_size = 0, 4
    model.native, model.step_use_replay = True, replay
    model.is_first_stage = model.is_last_stage = True
    model._decode_prefix, model.step_ticket = None, object()
    packed = tuple(pack_swa(torch.arange(tokens * 6 * 256, dtype=torch.float32)
                            .reshape(tokens, 6, 256).to(torch.bfloat16) / 256) for _ in range(2))
    model.engram_host = SimpleNamespace(wait=lambda ticket: packed)
    hidden = torch.zeros(tokens, 4, 5120, dtype=torch.bfloat16)
    pre = torch.zeros(tokens, 4)
    model.compiled_input = lambda ids: (hidden, pre)
    model.compiled_input_calls = 0
    consumed = []

    def stage(residual, mixes, positions, ids, rows):
        consumed.append(rows)
        return residual, mixes, None

    model.program = SimpleNamespace(loaded=True, dspark=True, length=32768, replay_owner=stage)
    model.ordinary = stage
    model.forward(torch.arange(tokens), torch.arange(tokens, dtype=torch.int32))
    assert len(consumed) == 1
    for source, actual in zip(packed, consumed[0], strict=True):
        if converted:
            assert actual.dtype == torch.bfloat16
            assert torch.equal(actual, unpack_swa(source, 256))
        else:
            assert actual is source


@pytest.mark.parametrize("fused", (False, True))
def test_device_round_keeps_the_warmed_embedding_and_replay_entry(monkeypatch, fused):
    from vllm_gaudi import envs
    from vllm_gaudi.models.deepseek_v41 import HpuDeepseekV41ForCausalLM

    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_DEVICE_ROUNDS", True)
    monkeypatch.setattr(envs, "VLLM_HPU_DSV41_FUSED_STAGE_IO", fused)
    model = HpuDeepseekV41ForCausalLM.__new__(HpuDeepseekV41ForCausalLM)
    torch.nn.Module.__init__(model)
    model.native = model.is_first_stage = model.is_last_stage = True
    hidden, mixes, calls = torch.zeros(6, 4, 5120), torch.zeros(6, 4), []
    ids, positions, rows = torch.arange(6), torch.arange(512, 518, dtype=torch.int32), ()

    def prepared_input(tokens):
        assert not fused, "a fused native entry must not repeat its embedding or collective"
        calls.append("embedding")
        return hidden, mixes

    def replay(residual, pre, token_positions, tokens, engram, *, fused_text_io):
        assert fused_text_io is fused, "keep the warmed native variant key"
        assert tokens is ids and token_positions is positions and engram is rows
        assert (residual is None and pre is None) if fused else (residual is hidden and pre is mixes)
        calls.append("replay")
        return hidden, mixes, hidden[:, 0]

    model.compiled_input = prepared_input
    model.program = SimpleNamespace(dspark=True, replay_owner=replay)
    assert model.forward_device_round(ids, positions, rows) is hidden
    assert calls == (["replay"] if fused else ["embedding", "replay"])


def test_native_ring_readback_preserves_slots_without_frontend_events(monkeypatch):
    from vllm_gaudi.ops.deepseek_v41_verify import VerifyRing

    waits = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Native readback must not record or wait on a frontend event")

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event", lambda: SimpleNamespace(record=forbidden, synchronize=forbidden))
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)

    def readback(words):
        return words.clone(), SimpleNamespace(synchronize=lambda: waits.append(1))

    ring = VerifyRing("cpu", last_rank=True, native_readback=readback)
    first, second = ring.acquire(6), ring.acquire(6)
    for ticket, token in ((first, 100), (second, 200)):
        ticket.record.copy_(torch.tensor([ticket.generation, 2, 2, 5, token, token + 1, -1, -1, -1, -1,
                                         token + 2, token + 3, token + 4, token + 5, token + 6, 0]))
        ring.stage(ticket)
    assert waits == [], "staging must enqueue only; the scheduler is the consumer"
    assert ring.consume(first) == (2, [100, 101], [102, 103, 104, 105, 106])
    ring.release(first)
    third = ring.acquire(6)
    assert third.slot == first.slot
    assert ring.consume(second) == (2, [200, 201], [202, 203, 204, 205, 206])
    ring.release(second)
    with pytest.raises(RuntimeError, match="Stale"):
        ring.consume(first)


@pytest.mark.parametrize("budget", [11, 17, 100])
@pytest.mark.parametrize("lookahead", [1, 2])
def test_target_and_verify_are_queued_before_current_output_is_consumed(monkeypatch, budget, lookahead):
    import numpy as np
    from vllm.v1.outputs import DraftTokenIds
    from vllm_gaudi.ops.deepseek_v41_verify import VerifyRing
    from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState, V41ModelRunner

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event", lambda: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    runner = V41ModelRunner.__new__(V41ModelRunner)
    runner.position_views = {6: torch.empty(6, dtype=torch.int32)}
    runner.device_round_inputs = DeviceRoundInputs(
        torch.empty(6, dtype=torch.int64), runner.position_views[6], frames=4 if lookahead == 2 else 2)
    runner.device_round_current = runner.device_round_queue = None
    runner.device_round_lookahead, runner.device_round_tail = lookahead, []
    runner.round_timing_enabled, runner.round_context = False, None
    runner.audit = dict(accepted_drafts=0, rejected_drafts=0)
    runner.pp = SimpleNamespace(generation=0)
    runner.model_config = SimpleNamespace(max_model_len=32768)
    runner.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=128))
    runner.verify_ring = VerifyRing(
        "cpu", last_rank=True, size=4 if lookahead == 2 else 2,
        native_readback=lambda value: (value.clone(), SimpleNamespace(synchronize=lambda: None)))
    history = SimpleNamespace(history=np.array([7, 8, 9]), position=16384)

    def commit(tokens, count):
        history.history = np.concatenate((history.history, tokens[:count]))[-3:]
        history.position += count

    history.prepare = lambda owner, tokens: np.array(tokens)
    history.commit = commit
    engram = SimpleNamespace(histories=torch.empty(7, 3, dtype=torch.int32))

    def prepare(owner, ids, lookback):
        engram.histories[0].copy_(lookback)
        for index in range(6):
            engram.histories[index + 1] = torch.cat((ids[index:index + 1].int(), engram.histories[index, :2]))
        return ()

    engram.prepare, engram.retire = prepare, lambda owner: None
    runner.device_round_engram = engram
    targets = []

    def target(ids, positions, rows):
        targets.append(ids.clone())
        return ids + 1

    runner.model = SimpleNamespace(engram_host=SimpleNamespace(history=history), last_aux=None,
                                   program=SimpleNamespace(search_length=32768, decode_token_bound=32768),
                                   forward_device_round=target,
                                   complete_step_device=lambda count: commit(np.arange(10, 16), count))

    def verify(hidden, proposed, metadata, aux, positions):
        result = verify_control_from_target(hidden, proposed, metadata)
        return (*result, None, None)

    def draft(metadata, positions, output, committed, output_count, anchor, enabled, status):
        proposals = anchor + torch.arange(1, 6)
        return pack_record(metadata, committed, output_count, output, proposals, enabled, status), None, None

    runner.verify_prefix, runner.draft_from_prefix = verify, draft
    request = RequestState("r", [0] * 16384, [], SimpleNamespace(max_tokens=budget + 1, temperature=0),
                           ([1] * 256,), 16384)
    request.output = [10]
    timing = SimpleNamespace(active={"generation": 1}, completed=[])

    def finish_timing(committed, count):
        timing.completed.append((committed, count))
        timing.active = None

    timing.finish = finish_timing
    runner.verify_timing = timing
    runner.pending = object()
    first = runner._finish_device_round(request, 16384, 6, list(range(11, 16)), torch.arange(11, 17))
    assert request.output == [10] and history.position == 16384
    expected = min(lookahead, max(0, budget // 6 - 1))
    assert len(targets) == expected, "next targets must precede the first host read"
    output = first.get_output()
    assert timing.active is None and timing.completed == [(6, 6)]
    assert output.sampled_token_ids == [list(range(11, 17))]
    assert request.output == list(range(10, 17)) and history.position == 16390
    if budget < 12:
        assert runner.device_round_inputs.owner is None
        return
    request.num_computed_tokens = 16390
    runner.device_round_current = runner.device_round_queue
    runner.device_round_queue = runner.device_round_tail.pop(0) if runner.device_round_tail else None
    runner.pending = object()
    assert isinstance(runner.draft_token_ids, DraftTokenIds)
    second = runner._finish_device_round(request, 16390, 6, list(range(17, 22)), None)
    expected += min(lookahead, max(0, (budget - 6) // 6 - 1)) - (expected - 1)
    assert len(targets) == expected and request.output == list(range(10, 17))
    assert second.get_output().sampled_token_ids == [list(range(17, 23))]
    assert request.output == list(range(10, 23)) and history.position == 16396
    runner._discard_device_round()
    assert runner.device_round_inputs.owner is None and runner.device_round_queue is None
    assert not runner.device_round_tail
    assert runner.verify_ring.generations == runner.verify_ring.consumed
