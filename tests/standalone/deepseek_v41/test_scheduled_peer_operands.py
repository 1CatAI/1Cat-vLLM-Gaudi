# SPDX-License-Identifier: Apache-2.0
"""Check rank order and ready dependencies at the native peer boundary."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    'peer_operands', Path(__file__).resolve().parents[3] /
    'vllm_gaudi/ops/deepseek_v41_dspark_peer_operands.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('tp_rank,tp_size,rows', ((0, 2, 2), (1, 2, 6), (0, 4, 6), (3, 4, 2)))
def test_rank_order_and_scheduled_dependency(monkeypatch, tp_rank, tp_size, rows):
    value = torch.full((rows, 5120), float(tp_rank), dtype=torch.bfloat16)
    ready = (torch.tensor(7), torch.tensor(8))
    calls = []

    def exchange(flat, dependencies):
        calls.append((flat.shape, dependencies))
        return torch.full_like(flat, float(1 - tp_rank))

    def gather(flat, size, dependencies):
        calls.append((flat.shape, dependencies))
        assert size == tp_size
        return torch.cat([torch.full_like(flat, float(rank)) for rank in range(size)])

    namespace = SimpleNamespace(tp2_exchange_peer_scheduled=exchange, tp_peer_allgather_scheduled=gather)
    monkeypatch.setattr(torch.ops, 'vllm_gaudi', namespace)
    result = module.make_peer_operands(tp_rank, tp_size)(value, ready)
    assert result.shape == (tp_size, rows, 5120)
    for rank in range(tp_size):
        assert torch.equal(result[rank], torch.full_like(value, float(rank)))
    assert len(calls) == 1 and calls[0][1] == list(ready)


def test_large_operand_is_rejected_before_transport():
    with pytest.raises(ValueError, match='small BF16'):
        module.make_peer_operands(0, 4)(torch.empty(7, 5120, dtype=torch.bfloat16))
