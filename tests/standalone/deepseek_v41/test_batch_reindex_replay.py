# SPDX-License-Identifier: Apache-2.0
"""Replay must use the same bounded-tile ABI as its selected batch scorer."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_batch_replay import BatchStageVariant
from vllm_gaudi.ops import tp2_prepared_plan


def variant(bucket):
    value = object.__new__(BatchStageVariant)
    torch.nn.Module.__init__(value)
    value.bucket = bucket
    value.program = SimpleNamespace(generation=1, precision_fingerprint="test")
    value.metadata = SimpleNamespace(native_completion=None)
    value.states = ()
    # A stage may own Reindex layers even when this bucket uses the regular
    # scorer. Their presence alone must not select the bounded replay ABI.
    value.reindex_ratios = (1, )
    return value


@pytest.mark.parametrize("bucket", [1, 2, 4, 8, 16, 32, 64])
def test_prepared_replay_abi_matches_bucket(monkeypatch, bucket):
    received = []
    sentinel = object()

    def replay(owner, **roots):
        received.append(roots)
        return sentinel

    monkeypatch.setattr(tp2_prepared_plan, "replay_native_decoder", replay)
    positions = torch.full((bucket, ), 2051, dtype=torch.int32)
    value = torch.zeros(bucket, 1)
    result = variant(bucket)(value,
                             value,
                             positions,
                             positions, (),
                             positions,
                             positions,
                             host_positions=[2051] * bucket if bucket == 8 else None)
    assert result is sentinel
    if bucket == 8:
        assert received[0]["reindex_tile_bound"] == 2
    else:
        assert "reindex_tile_bound" not in received[0]


def test_bounded_replay_requires_host_metadata_before_submission(monkeypatch):

    def replay(*args, **kwargs):
        raise AssertionError("An invalid bound must not reach native replay")

    monkeypatch.setattr(tp2_prepared_plan, "replay_native_decoder", replay)
    positions = torch.zeros(8, dtype=torch.int32)
    value = torch.zeros(8, 1)
    with pytest.raises(RuntimeError, match="scheduler-owned positions"):
        variant(8)(value, value, positions, positions, (), positions, positions)
