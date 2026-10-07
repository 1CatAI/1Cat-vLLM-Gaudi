// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <stdexcept>

namespace hcl {
struct NativeReceiveEpochWindow {
    enum class Part { Full, Prefix, Suffix };
    Part part;
    size_t offset;
    bool nextActive;
    static NativeReceiveEpochWindow prepare(size_t total, size_t split, bool active, size_t count) {
        if (!total || !count || count > total || split >= total)
            throw std::invalid_argument("Invalid receive epoch range");
        if (!split) {
            if (active || count != total) throw std::invalid_argument("Invalid full receive epoch");
            return {Part::Full, 0, false};
        }
        if (!active) {
            if (count == total) return {Part::Full, 0, false};
            if (count == split) return {Part::Prefix, 0, true};
        } else if (count == total - split) {
            return {Part::Suffix, split, false};
        }
        throw std::invalid_argument("Receive epoch range is out of sequence");
    }
};
}
