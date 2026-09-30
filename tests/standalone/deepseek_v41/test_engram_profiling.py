# SPDX-License-Identifier: Apache-2.0
"""Host profiling may stop while the device still owns a gathered transaction."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_gaudi.ops.deepseek_v41_host import EngramHost, host_native


def fixture(tmp_path):
    native = host_native()
    weights = np.arange(16 * 256, dtype=np.uint8).reshape(16, 256)
    scales = np.arange(16 * 8, dtype=np.uint8).reshape(16, 8)
    path = tmp_path / "table.bin"
    path.write_bytes(weights.tobytes() + scales.tobytes())
    table = native.HostRows(str(path), 0, str(path), weights.size, 0, 16, 256, True, False)
    host = EngramHost.__new__(EngramHost)
    gather = native.GatherSlot(6, 256)
    host.slots = {1: [SimpleNamespace(gather=gather)]}
    host.pending = host.ready_ticket = host.profile_records = None
    host.tp_rank = 0
    return host, gather, table, weights


def test_stop_after_host_gather_preserves_device_transaction(tmp_path):
    host, gather, table, weights = fixture(tmp_path)
    host.set_profiling(True)
    ticket = object()
    host.pending = ticket
    ids = np.array([2, 9, 3], dtype=np.int32)
    generation = gather.submit(table, ids)
    gather.wait(generation)
    timing = list(gather.timing)
    np.testing.assert_array_equal(gather.weights[:3], weights[ids])
    host.profile_records.append({"generation": generation, "timing": timing})
    gather.release(generation)
    host.ready_ticket = ticket
    target = tmp_path / "profile.json"
    host.export_profile(target)
    host.set_profiling(False)
    assert host.pending is ticket and host.ready_ticket is ticket
    assert host.profile_records is None
    assert json.loads(target.read_text())["records"][0]["timing"] == timing
    # The next host lookup uses the new mode without mixing old timing data.
    host.pending = host.ready_ticket = None
    generation = gather.submit(table, ids[::-1].copy())
    gather.wait(generation)
    np.testing.assert_array_equal(gather.weights[:3], weights[ids[::-1]])
    with pytest.raises(RuntimeError, match="profiled gather"):
        _ = gather.timing
    gather.release(generation)


@pytest.mark.parametrize("enabled", [False, True])
def test_unconsumed_gather_keeps_original_timer_mode(tmp_path, enabled):
    host, gather, table, _ = fixture(tmp_path)
    host.set_profiling(True)
    records = host.profile_records
    host.pending = object()
    generation = gather.submit(table, np.array([1, 2], dtype=np.int32))
    with pytest.raises(RuntimeError, match="pending host gather"):
        host.set_profiling(enabled)
    assert host.profile_records is records
    gather.wait(generation)
    assert gather.timing[0] == generation
    gather.release(generation)
