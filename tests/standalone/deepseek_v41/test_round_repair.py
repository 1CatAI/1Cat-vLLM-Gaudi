# SPDX-License-Identifier: Apache-2.0
"""C6 lookahead repair restores rings, noncontiguous pages and shared scratch."""
from unittest.mock import patch

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_round_repair import C6LookaheadJournal, C6WriteJournal


def program():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for index in range(2):
        layer = torch.nn.Module()
        layer.layer = index
        layer.attention = torch.nn.Module()
        for name, rows, width in (("swa", 256, 528), ("kv_history", 8, 512), ("score_history", 8, 512)):
            layer.attention.register_buffer(name, torch.arange(rows * width).reshape(rows, width).float())
        model.layers.append(layer)
    model.shared = torch.nn.Module()
    model.shared.register_buffer("block_table", torch.tensor([4, 7, 3, 6, 2, 5, 1, 0], dtype=torch.int32))
    model.shared.sources = torch.nn.ModuleDict()
    for ratio in (1, 2, 32):
        cache = torch.nn.Module()
        cache.ratio = ratio
        for name, width in (("main", 288), ("index", 68)):
            cache.register_buffer(name, torch.arange((1024 // ratio) * width).reshape(-1, width).to(torch.uint8))
        cache.register_buffer("index_mirror", torch.arange((1024 // ratio) * 128).reshape(-1, 128).float())
        model.shared.sources[str(ratio)] = cache
    selection = torch.nn.Module()
    selection.register_buffer("indices", torch.arange(16 * 512).reshape(16, 512).int())
    model.shared.topk = torch.nn.ModuleDict({"0": selection})
    model.shared.register_buffer("candidate_pool", torch.arange(16 * 2048).reshape(16, 2048).int())
    return model


@pytest.mark.parametrize("start", [126, 254, 509])
def test_device_write_set_roundtrip(start):
    model = program()
    original = {name: value.clone() for name, value in model.named_buffers()}
    with patch("vllm_gaudi.ops.deepseek_v41_replay.stage_state_tensors", lambda value: tuple(value.buffers())):
        journal = C6WriteJournal(model)
    positions = torch.arange(start, start + 6, dtype=torch.int32)
    journal(positions)
    for index in range(len(journal.specs)):
        target = getattr(journal, f"target_{index}")
        rows = getattr(journal, f"indices_{index}")
        target.index_fill_(0, rows, 0)
    journal.restore()
    for name, value in model.named_buffers():
        assert torch.equal(value, original[name]), name
    assert journal.bytes < sum(v.numel() * v.element_size() for v in model.buffers())


def test_unaccounted_state_stops_repair():
    model = program()
    model.register_buffer("new_mutable_cache", torch.zeros(6, 8))
    with patch("vllm_gaudi.ops.deepseek_v41_replay.stage_state_tensors", lambda value: tuple(value.buffers())), \
            pytest.raises(ValueError, match="Unaccounted C6 repair state"):
        C6WriteJournal(model)


@pytest.mark.parametrize("start,second_commit", [(126, 1), (254, 6), (509, 3)])
def test_two_round_journal_restores_cross_page_ring_and_prefix(start, second_commit):
    model = program()
    original = {name: value.clone() for name, value in model.named_buffers()}
    with patch("vllm_gaudi.ops.deepseek_v41_replay.stage_state_tensors", lambda value: tuple(value.buffers())):
        journal = C6LookaheadJournal(model)
        first = C6WriteJournal(model)
        second = C6WriteJournal(model)
    journal(torch.arange(start, start + 6, dtype=torch.int32))
    # Mutate exactly the independently mapped writes of two subsequent C6s.
    for actual, positions in ((first, torch.arange(start, start + 6, dtype=torch.int32)),
                              (second, torch.arange(start + second_commit, start + second_commit + 6,
                                                    dtype=torch.int32))):
        actual(positions)
        for index in range(len(actual.specs)):
            getattr(actual, f"target_{index}").index_fill_(0, getattr(actual, f"indices_{index}"), 0)
    journal.restore()
    for name, value in model.named_buffers():
        assert torch.equal(value, original[name]), name


@pytest.mark.parametrize("accepted", [0, 2, 5])
@pytest.mark.parametrize("native_full", [False, True])
@pytest.mark.parametrize("lookahead", [1, 2])
def test_async_repair_keeps_draw_and_does_not_reuse_live_ring_slot(monkeypatch, accepted, native_full, lookahead):
    from types import SimpleNamespace

    from vllm_gaudi.ops.deepseek_v41_round_inputs import DeviceRoundInputs
    from vllm_gaudi.ops.deepseek_v41_round_repair import SampledRoundRepairFrame
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import SpeculativeRequestSampling
    from vllm_gaudi.ops.deepseek_v41_verify import STATUS, VerifyRing, pack_record
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(torch.hpu, "Event", lambda: SimpleNamespace())
    model = program()
    original = {name: value.clone() for name, value in model.named_buffers()}
    model.draft = torch.nn.Module()
    model.draft.layers = torch.nn.ModuleList()
    for _ in range(3):
        layer = torch.nn.Module()
        layer.attention = torch.nn.Module()
        layer.attention.register_buffer("swa", torch.arange(32).reshape(8, 4).float())
        model.draft.layers.append(layer)
    cursor = DeviceRoundInputs(torch.empty(6, dtype=torch.int64), torch.empty(6, dtype=torch.int32),
                               frames=4 if lookahead == 2 else 2)
    cursor.seed("request", list(range(10, 16)), 126, 100, 1024, 1, [9, 8, 7])
    history = torch.arange(21, dtype=torch.int32).reshape(7, 3)
    engram = SimpleNamespace(histories=history.clone())
    # The serving producer creates these tensors in inference mode; the
    # asynchronous output consumer runs outside that producer's context.
    with torch.inference_mode():
        state = SpeculativeRequestSampling((1., .95, -1), 42, 16, "cpu")
        state.proposal.fill_(.125)
    prototype = (torch.ones(6, 1), cursor.ids[1:], cursor.control[:7], torch.ones(6, 3), cursor.positions,
                 state.proposal, state.parameters, state.seed, state.counter, state.offsets)
    active = tuple(value for name, value in model.named_buffers() if not name.startswith("draft."))
    with patch("vllm_gaudi.ops.deepseek_v41_replay.stage_state_tensors", lambda value: active):
        frame = SampledRoundRepairFrame(model, cursor, engram, prototype, 0, lookahead=lookahead)
    frame.capture_inputs(*prototype)
    frame.journal(cursor.positions + 2)
    journals = (frame.journal,) if lookahead == 1 else (frame.journal.first, frame.journal.second)
    for journal in journals:
        for index in range(len(journal.specs)):
            getattr(journal, f"target_{index}").index_fill_(
                0, getattr(journal, f"indices_{index}"), 0)
    for layer in model.draft.layers:
        layer.attention.swa.zero_()
    with torch.inference_mode():
        state.counter.fill_(22)
        state.proposal.zero_()
    cursor.positions.add_(2)
    cursor.ids.add_(100)
    cursor.parity = 1
    engram.histories.zero_()
    ring = VerifyRing("cpu", last_rank=True, size=4 if lookahead == 2 else 2,
                      native_readback=lambda value: (value.clone(), SimpleNamespace(synchronize=lambda: None)))
    current, discarded = ring.acquire(6), ring.acquire(6)
    discarded_tickets = [discarded] + ([ring.acquire(6)] if lookahead == 2 else [])
    for ticket in (current, *discarded_tickets):
        ticket.record.copy_(torch.tensor([ticket.generation, 2, 2, 5, 30, 31, -1, -1, -1, -1,
                                          40, 41, 42, 43, 44, 2]))
        ring.stage(ticket)
    request = SimpleNamespace(req_id="request")
    runner = V41ModelRunner.__new__(V41ModelRunner)
    runner.pp = SimpleNamespace(generation=ring.generation)
    runner.verify_ring = ring
    runner.device_round_inputs, runner.device_round_engram = cursor, engram
    runner.sampled_round_frames = {1: frame, **{value.generation: object() for value in discarded_tickets}}
    runner.device_round_queue = ("request", discarded, None)
    runner.device_round_lookahead = lookahead
    runner.device_round_tail = [("request", value, None) for value in discarded_tickets[1:]]
    runner.audit = {}
    runner._request_speculative_sampling = lambda req: state
    count = accepted + 1

    def exact(req, hidden, aux, control):
        assert torch.is_inference_mode_enabled()
        assert int(state.counter[0]) == 0, "repair must reuse the original official draw"
        assert torch.equal(state.proposal, prototype[5].new_full((5, 16), .125))
        assert torch.equal(engram.histories, history)
        state.counter.add_(11)
        state.proposal.fill_(.25)
        output = torch.cat((torch.arange(30, 30 + count), torch.full((6 - count,), -1)))
        record = pack_record(control[:7], torch.tensor([count]), torch.tensor([count]), output,
                             torch.arange(40, 45), torch.tensor([True]), torch.tensor([0]))
        return record, None

    requeued = []

    def queue(req):
        runner.pp.generation += 1
        expected_generation = (4 if lookahead == 1 else 6) + len(requeued)
        assert runner.pp.generation == expected_generation
        assert int(cursor.control[0]) == expected_generation
        next_ticket = ring.acquire(6, generation=expected_generation)
        assert next_ticket.slot != current.slot
        next_ticket.record.copy_(current.record)
        next_ticket.record[0] = expected_generation
        ring.stage(next_ticket)
        if runner.device_round_queue is None:
            runner.device_round_queue = (req.req_id, next_ticket, None)
        else:
            runner.device_round_tail.append((req.req_id, next_ticket, None))
        requeued.append(next_ticket)

    runner._sampled_round_reference, runner._queue_next_device_round = exact, queue
    if native_full:
        def native(*inputs):
            assert all(a is b for a, b in zip(inputs, frame.payload, strict=True))
            frame.journal.restore()
            frame.restore_protocol()
            state.proposal.copy_(inputs[5])
            state.counter.copy_(inputs[8])
            record, confidence = exact(request, inputs[0], inputs[3], cursor.control)
            q, advanced = state.proposal.clone(), state.counter.clone()
            state.proposal.zero_()
            state.counter.zero_()
            return record, None, q, confidence, advanced
        runner.sampled_native_full_protocols = {frame.parity: native}
    assert not torch.is_inference_mode_enabled()
    committed, output, draft = ring.consume(current, repair=lambda: runner._repair_sampled_round(request, current))
    assert committed == count and output == list(range(30, 30 + count)) and draft == list(range(40, 45))
    assert int(current.record[STATUS]) == 0 and int(state.counter[0]) == 11
    assert cursor.ids.tolist() == [output[-1], *draft]
    for name, value in model.named_buffers():
        if name in original:
            assert torch.equal(value, original[name]), name
    ring.release(current)
    assert len(requeued) == lookahead
    for future in requeued:
        assert ring.consume(future) == (count, output, draft)
        ring.release(future)
    ring.close()
