# SPDX-License-Identifier: Apache-2.0
"""Whole-record completion preserves ring ownership and stale-generation checks."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_verify import VerifyRing


@pytest.mark.parametrize('committed', [1, 3, 6])
def test_full_record_preserves_generation_output_and_draft(monkeypatch, committed):
    monkeypatch.setattr(torch.Tensor, 'pin_memory', lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, 'Event', lambda: SimpleNamespace())
    monkeypatch.setattr(torch, 'compile', lambda function, **kwargs: function)
    calls = []

    def full_record(value):
        calls.append(value.numel())
        return value.clone().reshape(1, -1), SimpleNamespace(synchronize=lambda: None)

    ring = VerifyRing('cpu', last_rank=True, native_readback=lambda _: pytest.fail('four-word fallback'),
                      native_record_readback=full_record)
    for generation in range(1, 4):
        ticket = ring.acquire(6)
        output = list(range(20, 20 + committed))
        draft = list(range(100, 105))
        ticket.record.copy_(torch.tensor([generation, committed, committed, 5,
                                         *(output + [-1] * (6 - committed)), *draft, 0]))
        ring.stage(ticket)
        assert ring.consume(ticket) == (committed, output, draft)
        ring.release(ticket)
        with pytest.raises(RuntimeError, match='Stale'):
            ring.consume(ticket)
    ring.close()
    assert calls == [16, 16, 16]
