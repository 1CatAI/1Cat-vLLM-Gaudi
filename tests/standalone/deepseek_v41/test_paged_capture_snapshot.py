# SPDX-License-Identifier: Apache-2.0
"""Cold replay capture restores page writes and logical mirrors separately."""

from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_replay import _PagedSnapshot


@pytest.mark.parametrize("ratio", (1, 2))
@pytest.mark.parametrize("positions", ((257,), (125, 126, 127, 128, 129, 130)))
@pytest.mark.parametrize("active_mirror", (False, True))
def test_capture_restores_touched_rows_with_permuted_pages(ratio, positions, active_mirror):
    positions = torch.tensor(positions)
    width = 128 // ratio
    pages = torch.tensor((0, 3, 1, 4, 2, 5, 6, 7))
    def physical_rows(rows, ratio):
        page_width = 128 // ratio
        return pages[rows // page_width] * page_width + rows % page_width
    cache = SimpleNamespace(ratio=ratio)
    generator = torch.Generator().manual_seed(41)
    for name, features in (("main", 16), ("index", 4), ("main_mirror", 16), ("index_mirror", 4)):
        setattr(cache, name, torch.randn(1024 // ratio, features, generator=generator).bfloat16())
    small = torch.randn(128, 32, generator=generator).bfloat16()
    states = (cache.main, cache.index, small)
    if active_mirror:
        states += (cache.main_mirror, cache.index_mirror)
    expected = tuple(value.clone() for value in states)
    program = SimpleNamespace(shared=SimpleNamespace(sources={ratio: cache}, physical_rows=physical_rows))
    snapshot = _PagedSnapshot(program, positions, states)

    logical = positions // ratio
    physical = torch.cat((physical_rows(logical, ratio), logical % width)).unique()
    for value in (cache.main, cache.index):
        value.index_fill_(0, physical, -9)
    if active_mirror:
        for value in (cache.main_mirror, cache.index_mirror):
            value.index_fill_(0, logical.unique(), -7)
    small.fill_(3)
    snapshot.restore()
    for value, reference in zip(states, expected, strict=True):
        assert torch.equal(value.view(torch.int16), reference.view(torch.int16))

    # Rows not written by the captured invocation stay outside its snapshot.
    cache.main[900 // ratio].fill_(17)
    if active_mirror:
        cache.index_mirror[900 // ratio].fill_(19)
    snapshot.restore()
    assert bool((cache.main[900 // ratio] == 17).all())
    if active_mirror:
        assert bool((cache.index_mirror[900 // ratio] == 19).all())
        assert snapshot.bytes < sum(value.numel() * value.element_size() for value in states) // 2
