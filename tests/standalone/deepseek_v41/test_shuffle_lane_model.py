# SPDX-License-Identifier: Apache-2.0
from tools.check_deepseek_v41_shuffle_lanes import shuffle, dictionary_check


def test_group_mux_uses_selected_direction_not_destination_direction():
    directions = [((i + 1) & 255) | 128 for i in range(256)]
    assert shuffle(list(range(256)), directions, [0] * 256)[30] == 63


def test_disabled_lanes_preserve_income():
    assert shuffle(list(range(256)), [0] * 256, [77] * 256) == [77] * 256


def test_masking_does_not_make_a_flat_64_entry_dictionary_valid():
    assert dictionary_check()['flat_dictionary_mismatches'] > 0
