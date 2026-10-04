# SPDX-License-Identifier: Apache-2.0
"""Ordinary decode lookahead lifetime tests, independent of hardware timing."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_device_loop import DeviceStep, SamplingFrames
from vllm_gaudi.ops.deepseek_v41_replay import capture_engram_inputs
from vllm_gaudi.v1.worker.deepseek_v41_v2_runner import V41V2ModelRunner


def payload(token=7):
    return (torch.tensor([[token * 2 + 1]], dtype=torch.int32), torch.arange(16.).reshape(1, 16),
            torch.tensor([[1., .95, .125]]), torch.tensor([[token]], dtype=torch.int32),
            torch.tensor([16384], dtype=torch.int32))


def test_preserved_draw_survives_next_native_output_overwrite():
    source = payload()
    history = torch.tensor([1, 2, 3], dtype=torch.int32)
    frames = SamplingFrames(source, history, compile_copy=False)
    frame = frames.preserve(source, history)
    expected = tuple(value.clone() for value in (*source, history))
    for value in (*source, history):
        value.zero_()
    frames.preserve(source, history)
    assert all(torch.equal(value, reference) for value, reference in zip(frame, expected, strict=True))
    assert all(value.data_ptr() != reference.data_ptr() for value, reference in zip(frame[:5], source, strict=True))
    assert frames.preserve(source, history) is frame


def test_shared_capture_accepts_two_fixed_bf16_inputs_and_rejects_mixed():
    rows = tuple(torch.zeros(1, 6, 256, dtype=torch.bfloat16) for _ in range(2))
    assert capture_engram_inputs(rows, direct=True, device_layers=True, local_heads=6) == rows
    with pytest.raises(ValueError, match="both fixed BF16"):
        capture_engram_inputs((rows[0], torch.zeros(1, 6, 264, dtype=torch.uint8)),
                              direct=True, device_layers=True, local_heads=6)


def test_discard_drains_before_scheduler_can_reuse_state():
    runner = object.__new__(V41V2ModelRunner)
    calls = []
    runner._device_step = DeviceStep("r", 127, None, SimpleNamespace(synchronize=lambda: calls.append("drain")), None)
    runner.audit = {}
    runner._discard_device_step()
    assert calls == ["drain"]
    assert runner._device_step is None and runner._device_loop_position is None
    runner._discard_device_step()
    assert calls == ["drain"]


def test_repair_drains_provisional_step_and_reuses_preserved_uniform():
    runner = object.__new__(V41V2ModelRunner)
    frame = (*payload(), torch.tensor([1, 2, 3], dtype=torch.int32))
    calls = []
    runner._device_step = DeviceStep("r", 16384, None,
                                    SimpleNamespace(synchronize=lambda: calls.append("drain")), frame)
    request = SimpleNamespace(req_id="r")
    runner.requests, runner.audit = {"r": request}, {}

    def repair(values, destination):
        assert runner._device_step is None
        assert values[2] is frame[2]
        calls.append("repair")
        destination.fill_(11)
        return 11

    def requeue(owner, start, values):
        assert owner is request and start == 16383 and values is frame
        assert values[3].item() == 11
        calls.append("recompute")

    runner._repair_device_sample, runner._queue_device_step = repair, requeue
    assert runner._repair_loop_sample(frame, frame[3]) == 11
    assert calls == ["drain", "repair", "recompute"]


def test_queued_input_commit_updates_host_history_without_lookup_or_h2d():
    runner = object.__new__(V41V2ModelRunner)
    calls = []
    history = SimpleNamespace(prepare=lambda owner, tokens: calls.append((owner, tokens)) or "batch",
                              commit=lambda batch, count: calls.append((batch, count)))
    runner.model = SimpleNamespace(engram_host=SimpleNamespace(history=history),
                                   complete_step=lambda count: pytest.fail("host table lookup transaction"))
    runner.requests = {"r": SimpleNamespace(tokens=[5, 7])}
    runner._device_step_input = "r", 1
    runner._complete_loop_input(SimpleNamespace(request_id="r", start=1))
    assert calls == [("r", [7]), ("batch", 1)]
    assert runner._device_step_input is None
