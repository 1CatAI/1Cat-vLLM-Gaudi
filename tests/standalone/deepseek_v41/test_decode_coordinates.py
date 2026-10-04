# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from vllm_gaudi.ops.deepseek_v41_decode_coordinates import decode_coordinates_reference


def test_coordinates_keep_page_ring_history_and_compressor_boundaries():
    generator = torch.Generator().manual_seed(13)
    pages = torch.randperm(8192, generator=generator).to(torch.int32)
    positions = torch.tensor([0, 1, 7, 127, 128, 255, 256, 16383, 16384, 131071, 524287], dtype=torch.int32)
    ids = torch.full_like(positions, 31)
    ids[0], ids[1] = 129264, 129265
    coords = decode_coordinates_reference(positions, ids, pages)
    assert torch.equal(coords[:, 1], positions.remainder(256))
    assert torch.equal(coords[:, 2], positions.remainder(8))
    for ratio, field in ((1, 5), (2, 6)):
        logical = positions // ratio
        width = 128 // ratio
        physical = pages.index_select(0, logical // width) * width + logical.remainder(width)
        assert torch.equal(coords[:, field], physical)
    assert torch.equal(coords[:, 8].bool(), positions.remainder(2) == 1)
    assert torch.equal(coords[:, 9], positions // 2 * 2)
    assert coords[:, 11].tolist() == [1, 1] + [0] * 9
    absolute = positions[:, None] - 127 + torch.arange(128)
    assert torch.equal(coords[:, 64:], torch.where(absolute >= 0, absolute.remainder(256), -1))
    assert not coords[:, 16:64].any()


@pytest.mark.parametrize('rows', [1, 2, 6])
def test_inactive_or_unowned_rows_do_not_publish_cache_addresses(rows):
    pages = torch.tensor([5, -1], dtype=torch.int32)
    positions = torch.tensor([-1, 128, 256, -127, 257, 129][:rows], dtype=torch.int32)
    ids = torch.full_like(positions, 129264)
    coords = decode_coordinates_reference(positions, ids, pages)
    assert torch.equal(coords[:, 0], positions)
    assert (coords[:, 1:8] == -1).all()
    assert (coords[:, 9:11] == -1).all()
    assert not coords[:, 8].any()
    assert not coords[:, 11:64].any()
    assert (coords[:, 64:] == -1).all()
