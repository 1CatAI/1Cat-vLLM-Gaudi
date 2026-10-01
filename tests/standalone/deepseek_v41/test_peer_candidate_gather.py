# SPDX-License-Identifier: Apache-2.0
"""FP32 greedy candidates retain their exact bits on the common peer wire."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives


@pytest.mark.parametrize('tp_size', [2, 4])
@pytest.mark.parametrize('rows', [1, 2, 6])
def test_candidate_wire_exact_bits_and_rank_order(monkeypatch, tp_size, rows):
    import vllm.distributed as dist

    monkeypatch.setattr(dist, 'tensor_model_parallel_all_gather',
                        lambda *args, **kwargs: pytest.fail('Native candidate unexpectedly used HCCL'))
    # Includes a quiet NaN payload, signed zero, and a subnormal; no BF16
    # arithmetic or numeric conversion may alter any payload word.
    bit_patterns = torch.tensor([0x7fc01234, -2147483648, 1, 0x7f800000, -8388608, 0x3f810001], dtype=torch.int32)
    values = [(bit_patterns[:rows].reshape(rows, 1).expand(-1, 2).clone() + rank).view(torch.float32)
              for rank in range(tp_size)]
    wire = [torch.nn.functional.pad(v.view(torch.bfloat16).flatten(), (0, 128 - rows * 4)) for v in values]
    for rank in range(tp_size):
        monkeypatch.setattr(torch.ops.vllm_gaudi, 'tp2_exchange_peer', lambda x, rank=rank: wire[1-rank].reshape(1,-1),
                            raising=False)
        monkeypatch.setattr(torch.ops.vllm_gaudi, 'tp_peer_allgather', lambda x, n: torch.cat(wire).reshape(1,-1),
                            raising=False)
        _, gather = stage_collectives(rank, True, tp_size, native_fp32_gather=True)
        result = gather(values[rank], dim=-1)
        assert torch.equal(result.view(torch.int32), torch.cat(values, dim=1).view(torch.int32))


def test_baseline_fp32_gather_retains_qualified_hccl(monkeypatch):
    import vllm.distributed as dist

    expected = torch.ones(1, 8)
    calls = []
    def stock(value, dim):
        calls.append((value.dtype, dim))
        return expected
    monkeypatch.setattr(dist, 'tensor_model_parallel_all_gather', stock)
    _, gather = stage_collectives(0, True, 4)
    assert gather(torch.ones(1, 2), dim=-1) is expected
    assert calls == [(torch.float32, -1)]
