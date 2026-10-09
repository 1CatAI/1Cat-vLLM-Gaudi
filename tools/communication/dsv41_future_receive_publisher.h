// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "synapse_api.h"
#include "hccl.h"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Opt-in native-plan adapter. The caller retains every payload, event and
// stream until compute and NIC retire, and checks error after ReplayPlan,
// before handing any output to the sampler. No default C1 path uses this.
template<class SyncInfo> class Dsv41FutureReceivePublisher {
public:
    struct Point {
        uint64_t readyDelta = 0;
        uint64_t epochAddress = 0;
        uint64_t flagAddress = 0;
        synEventHandle event = nullptr;
    };
    using BatchReplay = hcclResult_t (*)(void*, const SyncInfo*, size_t, SyncInfo*);
    using Bind = synStatus (*)(synEventHandle, synEventHandle, const SyncInfo*, uint64_t, uint64_t);
    struct Api {
        BatchReplay batchReplay = nullptr;
        Bind bind = nullptr;
        decltype(&synStreamWaitEvent) wait = synStreamWaitEvent;
        decltype(&synMemCopyAsync) copy = synMemCopyAsync;
    };
    Dsv41FutureReceivePublisher(Api api, void* batch, synEventHandle seed, synStreamHandle copy,
                                std::vector<Point> points)
        : api_(api), batch_(batch), seed_(seed), copy_(copy), points_(std::move(points)), ready_(points_.size()) {
        if (!api_.batchReplay || !api_.bind || !api_.wait || !api_.copy || !batch_ || !seed_ || !copy_ ||
            points_.empty() || points_.size() > 128) throw std::invalid_argument("Invalid future receive plan");
        for (const auto& point : points_) {
            const bool publishes = point.event != nullptr;
            if (publishes != bool(point.epochAddress) || publishes != bool(point.flagAddress) ||
                (publishes && ((point.epochAddress | point.flagAddress) & 3)))
                throw std::invalid_argument("Future publication requires an aligned owned epoch and flag");
        }
    }
    static int replay(void* context, const SyncInfo* producers, uint64_t count, SyncInfo* completions) noexcept {
        return static_cast<Dsv41FutureReceivePublisher*>(context)->run(producers, count, completions);
    }
    void check() const {
        if (error_) throw std::runtime_error("Future receive publication failed; discard queued outputs");
    }
    int error() const { return error_; }
    size_t failedPoint() const { return failedPoint_; }
private:
    int run(const SyncInfo* producers, uint64_t count, SyncInfo* completions) noexcept {
        error_ = 0;failedPoint_ = std::numeric_limits<size_t>::max();
        if (!producers || !completions || count != points_.size()) {error_ = 1;return 1;}
        for (size_t i = 0; i < count; ++i) {
            if (producers[i].targetValue <= points_[i].readyDelta ||
                producers[i].targetValue >= (uint64_t(1) << 60)) {error_ = 2;failedPoint_ = i;return 1;}
            ready_[i] = producers[i];ready_[i].targetValue -= points_[i].readyDelta;
        }
        const auto status = api_.batchReplay(batch_, ready_.data(), count, completions);
        if (status != hcclSuccess) {error_ = 3;return 1;}
        const auto begin = completions[0].targetValue - 1, end = completions[count-1].targetValue;
        for (size_t i = 0; i < count; ++i) {
            if (!completions[i].targetValue || end >= (uint64_t(1) << 60) ||
                completions[i].longSoIndex != completions[0].longSoIndex ||
                (i && completions[i].targetValue != completions[i-1].targetValue + 1)) {
                error_ = 4;failedPoint_ = i;return 1;
            }
        }
        for (size_t i = 0; i < count; ++i) {
            const auto& point = points_[i];
            if (!point.event) continue; // Existing ordinary NIC consumers share the same batch.
            synStatus result = api_.bind(point.event, seed_, completions+i, begin, end);
            if (result == synSuccess) result = api_.wait(copy_, point.event, 0);
            if (result == synSuccess) result = api_.copy(copy_, point.epochAddress, 4, point.flagAddress, DRAM_TO_DRAM);
            if (result != synSuccess && !error_) {error_ = 5;failedPoint_ = i;}
        }
        // NIC is already waiting for compute signals. Permit compute to be
        // published on a flag error so bounded receivers can drain the work;
        // check() makes this a failure before any generated tokens are used.
        return 0;
    }
    Api api_;
    void* batch_;
    synEventHandle seed_;
    synStreamHandle copy_;
    std::vector<Point> points_;
    std::vector<SyncInfo> ready_;
    int error_ = 0;
    size_t failedPoint_ = std::numeric_limits<size_t>::max();
};
