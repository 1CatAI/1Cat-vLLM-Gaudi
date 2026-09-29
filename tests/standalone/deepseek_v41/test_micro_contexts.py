# SPDX-License-Identifier: Apache-2.0
import pytest

from tools.deepseek_v41_micro_contexts import micro_contexts


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("first", [2, 20, 24])
def test_decode_profile_executes_index_scoring_with_owned_pages(batch, first):
    positions, pages = micro_contexts(batch, first, "2k-decode")
    assert len(positions) == batch
    assert all((position + 1) // 2 > 512 for position in positions)
    assert all(0 <= position < pages * 128 for position in positions)
    if batch >= 8:
        assert 2559 in positions and 2560 in positions and 4095 in positions


def test_legacy_geometry_is_unchanged_and_distinct_from_two_k():
    assert micro_contexts(8, 2) == ([126, 127, 254, 255, 510, 511, 766, 999], 8)
    assert micro_contexts(8, 24) == ([530, 767, 1023, 1279, 1535, 1791, 1919, 2046], 16)
    assert micro_contexts(4, 24, long_reindex=True) == ([16383] * 4, 128)
    with pytest.raises(ValueError, match="separate"):
        micro_contexts(8, 24, "2k-decode", long_reindex=True)
