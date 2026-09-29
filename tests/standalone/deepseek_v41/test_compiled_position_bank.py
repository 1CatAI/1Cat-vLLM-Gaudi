# SPDX-License-Identifier: Apache-2.0
"""Reuse a compiled copy across offsets without losing late position consumers."""
import os

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_inputs import PositionBank

pytestmark = pytest.mark.skipif(os.getenv('DSV41_TEST_HPU') != '1', reason='An HPU module lease is required')


@torch.inference_mode()
def test_compiled_copy_reuses_graph_and_orders_delayed_shared_position_consumers():
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v41_math import rotary_table
    torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
    # The serving runner allocates and loads outside inference mode. Its
    # prepare method must use the execution mode of subsequent decode calls.
    with torch.inference_mode(False):
        bank = PositionBank(1048576, 6, 'hpu')
        storage = torch.empty(6, dtype=torch.int32, device='hpu')
        positions = {n: storage[:n] for n in range(1, 7)}
        bank.prepare_compiled_copies(positions)
    prepared = bank.copy_preparations
    assert prepared == 6
    table = rotary_table(64, 32768, 10000).reshape(32768, 64).to('hpu')
    torch.manual_seed(9217)
    weight = (torch.randn(512, 512) / 512**.5).bfloat16().to('hpu')

    def consumer(value, pos):
        first = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(value, pos, table)
        delayed = value
        for _ in range(16):
            delayed = torch.matmul(delayed, weight)
        # A short single RoPE can finish before an unordered next DMA by luck.
        # Model layers also read this same position after earlier computation.
        last = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(delayed, pos, table)
        return first, last

    consume = torch.compile(consumer, backend='hpu_backend', fullgraph=True, dynamic=False)
    torch.manual_seed(9217)
    cases = []
    for step, start in enumerate((0, 127, 128, 16384, 16532, 32760, 1, 129)):
        for count in (1, 1, 6, 2):
            value = torch.randn(count, 16, 512).bfloat16().to('hpu')
            source = torch.arange(start, start + count, dtype=torch.int32).to('hpu')
            expected = tuple(part.cpu() for part in consume(value, source))
            cases.append((start, count, value, expected))
    retained = []
    for start, count, value, expected in cases:
        bank.copy_into(positions[count], start)
        retained.append((consume(value, positions[count]), expected))
    # No per-step CPU read is allowed to conceal a premature destination reuse.
    for actual, expected in retained:
        for a, b in zip(actual, expected, strict=True):
            assert torch.equal(a.cpu().view(torch.int16), b.view(torch.int16))
    for count in positions:
        bank.copy_into(positions[count], bank.length - count)
        assert positions[count].cpu().tolist() == list(range(bank.length - count, bank.length))
        for invalid in (-1, bank.length - count + 1):
            with pytest.raises(ValueError, match='range/device'):
                bank.copy_into(positions[count], invalid)
    assert bank.copy_preparations == prepared
    with pytest.raises(RuntimeError, match='already prepared'):
        bank.prepare_compiled_copies(positions)
