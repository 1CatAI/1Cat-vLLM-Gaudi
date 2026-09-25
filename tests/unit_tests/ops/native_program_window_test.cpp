// SPDX-License-Identifier: Apache-2.0
#include "native_program_window.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>

using namespace NativeProgramWindow;

static void verify(uint64_t raw, uint64_t bytes, uint64_t payload, uint64_t selected)
{
    assert(selected >= raw && selected - raw <= bytes - payload);
    assert(selected % alignment == 0);
    assert((selected >> 32) == ((selected + payload) >> 32));
}

int main()
{
    uint64_t bytes = 0, address = 0;
    assert(!allocationSize(0, false, bytes));
    assert(!allocationSize(window, true, bytes));
    assert(!allocationSize(UINT64_MAX, true, bytes));
    assert(!selectAddress(0, 0, 0, address));
    assert(!selectAddress(0, 16, 17, address));
    assert(!selectAddress(UINT64_MAX - 3, 8192, 1, address));

    const uint64_t boundary = 0x1001601800000000ULL;
    assert(allocationSize(0x640, false, bytes));
    assert(selectAddress(boundary - 0x300, bytes, 0x640, address));
    assert(address == boundary);
    // Exact exclusive-end boundaries must also move, like DeviceScal.
    assert(allocationSize(0x10000, false, bytes));
    assert(!selectAddress(boundary - 0x10000, bytes, 0x10000, address));
    assert(allocationSize(0x10000, true, bytes));
    assert(selectAddress(boundary - 0x10000, bytes, 0x10000, address));
    assert(address == boundary);

    std::mt19937_64 random(317);
    uint64_t retries = 0;
    for (uint64_t i = 0; i != 1000000; ++i)
    {
        const uint64_t payload = 1 + random() % (window - 1);
        const uint64_t raw = 0x1001601000000000ULL + (random() % (8 * window));
        assert(allocationSize(payload, false, bytes));
        if (selectAddress(raw, bytes, payload, address))
        {
            verify(raw, bytes, payload, address);
            continue;
        }
        ++retries;
        // A pool can return a different address after releasing the first
        // allocation. The padded contract must not depend on address reuse.
        const uint64_t moved = 0x1001601000000000ULL + (random() % (8 * window));
        assert(allocationSize(payload, true, bytes));
        assert(bytes == 2 * payload + alignment - 1);
        assert(selectAddress(moved, bytes, payload, address));
        verify(moved, bytes, payload, address);
    }
    // Exhaust both sides of the boundary at every alignment offset.
    const uint64_t sizes[] = {1, 64, 8192, 8193, 1048576, window - 1};
    for (uint64_t size : sizes)
    {
        for (uint64_t offset = 0; offset != alignment * 2; ++offset)
        {
            assert(allocationSize(size, true, bytes));
            const uint64_t raw = boundary - alignment + offset;
            assert(selectAddress(raw, bytes, size, address));
            verify(raw, bytes, size, address);
        }
    }
    assert(retries != 0);
    std::cout << "1000000 random regions and 98304 alignment-boundary cases passed; retries=" << retries << '\n';
}
