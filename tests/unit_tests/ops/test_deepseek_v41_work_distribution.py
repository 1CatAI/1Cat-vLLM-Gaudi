# SPDX-License-Identifier: Apache-2.0
"""Offline ARC decoder boundary checks; no device or recipe mutation."""
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
from inspect_deepseek_v41_work_distribution import scheduled_tpc_contexts, tpc_context  # noqa: E402


def context_bytes():
    return struct.pack('<22I', *([0]*5 + [20, 4, 5, 1, 1] + [5, 2, 2, 1, 1]
                                + [0x20110, 65536, 0, 0, 0, 0, 0]))


def test_edge_boxes_preserve_actual_work():
    result = tpc_context(context_bytes())
    assert result['boxes'] == 24
    assert result['box_work_histogram'] == {10: 8, 20: 16}
    assert result['total_work'] == 400
    assert result['equal_cost_box_utilization'] == pytest.approx(5/6)


def test_context_dma_and_execution_pair():
    packet = struct.pack('<3I', 3 | (2 << 5) | (88 << 8), 0, 2 | (1 << 5))
    data = packet + context_bytes()
    buffers = [{}, {}, dict(offset=len(packet), bytes=88)]
    result = scheduled_tpc_contexts(data, dict(offset=0, bytes=len(packet)), buffers)
    assert len(result) == 1
    assert result[0]['grid'] == [20, 4, 5, 1, 1]
    with pytest.raises(ValueError, match='Scheduled/executed'):
        scheduled_tpc_contexts(data, dict(offset=0, bytes=8), buffers)


def test_corrupt_context_region_is_rejected():
    packet = struct.pack('<3I', 3 | (2 << 5) | (88 << 8), 8, 2 | (1 << 5))
    with pytest.raises(ValueError, match='exceeds dynamic buffer'):
        scheduled_tpc_contexts(packet+context_bytes(), dict(offset=0, bytes=len(packet)),
                               [{}, {}, dict(offset=len(packet), bytes=88)])
    with pytest.raises(ValueError, match='Unsupported Gaudi2 TPC context size'):
        tpc_context(context_bytes()[:-4])
