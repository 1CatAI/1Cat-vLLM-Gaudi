# SPDX-License-Identifier: Apache-2.0
"""Candidate ownership and tie-order contracts, independent of HPU imports."""
import importlib.util
from pathlib import Path

import pytest
import torch

_path = Path(__file__).resolve().parents[3] / "vllm_gaudi/ops/deepseek_v41_reindex_compact.py"
_spec = importlib.util.spec_from_file_location("reindex_compact_contract", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
compact = _module.compact_reindex_reference


@pytest.mark.parametrize("batch", [1, 4, 8, 32, 64])
@pytest.mark.parametrize("ratio", [1, 2])
def test_stable_compaction_and_replay_tail(batch, ratio):
    generator = torch.Generator().manual_seed(731)
    pool = torch.randint(-1, 4096, (batch, 2048), dtype=torch.int32, generator=generator)
    # Deliberate holes, duplicate blocks, out-of-range values and extreme IDs.
    pool[:, :8] = torch.tensor([0, -1, 0, 65, 2**31 - 1, -(2**31), 129, 64])
    for visible in (0, 511, 512, 513, 531, 2047, 2048, 2049, 8193, 1048576 // ratio):
        lengths = torch.tensor([max(0, visible - r * 7) for r in range(batch)], dtype=torch.int32)
        ids, slots, counts = compact(pool, lengths * ratio - 1, ratio)
        for r, length in enumerate(lengths.tolist()):
            expected = [
                s for s, block in enumerate(pool[r].tolist()) if length > 512 and 0 <= block < (length + 7) // 8
            ]
            n = len(expected)
            assert counts[r].item() == n
            assert slots[r, :n].tolist() == expected
            assert torch.equal(ids[r, :n], pool[r, expected])
            assert bool((slots[r, n:] == -1).all())
            assert bool((ids[r, n:] == -1).all())


def test_compaction_preserves_top512_ties_and_logical_mapping():
    pool = torch.full((1, 2048), -1, dtype=torch.int32)
    valid_slots = torch.arange(0, 2048, 3)
    pool[0, valid_slots] = torch.arange(valid_slots.numel()).flip(0).int()
    ids, origin, count = compact(pool, torch.tensor([8192], dtype=torch.int32), 1)
    # Every score tied. First source slots must win, not smallest logical ID.
    original_rows = [(s, k, int(pool[0, s]) * 8 + k) for s in valid_slots.tolist() for k in range(8)]
    compact_rows = [(int(origin[0, s]), k, int(ids[0, s]) * 8 + k) for s in range(int(count[0])) for k in range(8)]
    assert compact_rows[:512] == original_rows[:512]


def test_short_context_needs_no_scoring_even_with_full_pool():
    pool = torch.zeros((64, 2048), dtype=torch.int32)
    ids, origin, counts = compact(pool, torch.full((64, ), 511, dtype=torch.int32), 1)
    assert counts.sum().item() == 0
    assert bool((ids == -1).all()) and bool((origin == -1).all())


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("visible,tiles", [(0, 0), (512, 0), (513, 1), (2048, 1), (2049, 2), (8193, 5), (16385, 8)])
def test_scheduler_bound(ratio, visible, tiles):
    assert _module.unique_pool_tile_bound([visible * ratio - 1, -1], ratio) == tiles


def test_bound_forbids_implicit_device_readback_and_out_of_contract_positions():
    for positions in (torch.tensor([8192]), [], [-2], [1048576], [0] * 65):
        with pytest.raises(ValueError):
            _module.unique_pool_tile_bound(positions, 1)


@pytest.mark.parametrize("bad", ["width", "batch", "dtype", "positions", "ratio", "stride"])
def test_invalid_contract(bad):
    pool = torch.zeros((4, 2048), dtype=torch.int32)
    pos = torch.zeros(4, dtype=torch.int32)
    ratio = 1
    if bad == "width":
        pool = pool[:, :2047]
    elif bad == "batch":
        pool, pos = torch.zeros((65, 2048), dtype=torch.int32), torch.zeros(65, dtype=torch.int32)
    elif bad == "dtype":
        pool = pool.long()
    elif bad == "positions":
        pos = pos[:, None]
    elif bad == "ratio":
        ratio = 4
    elif bad == "stride":
        pool = torch.zeros((4, 4096), dtype=torch.int32)[:, ::2]
    with pytest.raises(ValueError):
        compact(pool, pos, ratio)
