# SPDX-License-Identifier: Apache-2.0
"""Explicit context profiles for comparable concurrent component measurements."""


def micro_contexts(batch, first_layer, profile="legacy", *, long_reindex=False):
    if batch not in (1, 2, 4, 8, 16, 32, 64) or first_layer not in (2, 20, 24):
        raise ValueError("Unsupported microbenchmark batch or layer group")
    if profile not in ("legacy", "2k-decode"):
        raise ValueError("Unknown microbenchmark context profile")
    if long_reindex:
        if profile != "legacy" or first_layer != 24:
            raise ValueError("Long Reindex uses its separate switching protocol")
        return [16383] * batch, 128
    if profile == "2k-decode":
        # A roughly 2K prompt and up to 2K generated tokens. Include the
        # existing C1 hot-prefix boundary without reducing service capacity.
        values = (2051, 2047, 2303, 2559, 2560, 3071, 3583, 4095)
        pages = 32
    elif first_layer == 24:
        values = (530, 767, 1023, 1279, 1535, 1791, 1919, 2046)
        pages = 16
    else:
        values = (126, 127, 254, 255, 510, 511, 766, 999)
        pages = 8
    return [values[i % len(values)] for i in range(batch)], pages
