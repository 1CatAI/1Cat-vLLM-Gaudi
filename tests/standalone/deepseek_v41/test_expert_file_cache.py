# SPDX-License-Identifier: Apache-2.0
"""Serving expert reads retain bounded buffers without synchronous eviction."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_gaudi.ops.deepseek_v41_fp8 import read_expert


@pytest.mark.parametrize("dtype", ["I16", "BF16"])
def test_serving_expert_read_preserves_bytes_without_fadvise(tmp_path, monkeypatch, dtype):
    values = np.arange(24, dtype=np.int16).reshape(3, 2, 4)
    path = tmp_path / "experts"
    path.write_bytes(b"header" + values.tobytes())
    source = SimpleNamespace(file=path, offset=6, shape=values.shape, nbytes=values.nbytes, dtype=dtype)

    def unexpected_eviction(*args):
        raise AssertionError("Serving must leave page-cache eviction to the VM")

    monkeypatch.setattr(os, "posix_fadvise", unexpected_eviction)
    actual = read_expert(source, 1, keep_file_cache=True)
    assert actual.tobytes() == values[1].tobytes()
    assert actual.flags.writeable


def test_offline_eviction_and_truncated_reads_remain_explicit(tmp_path, monkeypatch):
    path = tmp_path / "experts"
    path.write_bytes(bytes(range(24)))
    source = SimpleNamespace(file=path, offset=0, shape=(3, 2, 2), nbytes=24, dtype="I16")
    calls = []
    monkeypatch.setattr(os, "posix_fadvise", lambda *args: calls.append(args[1:]))
    assert read_expert(source, 2).tobytes() == bytes(range(16, 24))
    assert calls == [(16, 8, os.POSIX_FADV_DONTNEED)]
    path.write_bytes(bytes(range(20)))
    with pytest.raises(ValueError, match="Truncated"):
        read_expert(source, 2, keep_file_cache=True)
    assert len(calls) == 1
