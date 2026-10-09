// SPDX-License-Identifier: Apache-2.0
#include "native_compute_program.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>
struct Sync { uint32_t longSoIndex; uint64_t targetValue; };
int main() {
    NativeScalCommandBuffer pre, post;
    NativeScalCommand load, complete;
    load.bytes.assign(8, 0x31);
    complete.bytes.assign(4, 0x22);
    complete.completesChunk = true;
    pre.commands = {load, complete};
    post.commands = {load, complete};
    NativeComputeWaitPacket wait;
    wait.bytes.assign(16, 0);
    wait.patches.push_back({4, 0, 0xfffe0000u, 0, 17});
    unsetenv("VLLM_HPU_DSV41_DSPARK_NATIVE_PAGE_COALESCE");
    auto baseline = NativeComputeProgram::prepare({&pre, &post}, {1}, {wait}, 16, 4096);
    assert(!baseline.coalesceSubmissions);
    setenv("VLLM_HPU_DSV41_DSPARK_NATIVE_PAGE_COALESCE", "1", 1);
    auto candidate = NativeComputeProgram::prepare({&pre, &post}, {1}, {wait}, 16, 4096);
    auto bounded = NativeComputeProgram::prepare({&pre, &post}, {1}, {wait}, 16, 64);
    unsetenv("VLLM_HPU_DSV41_DSPARK_NATIVE_PAGE_COALESCE");
    assert(!baseline.coalesceSubmissions && candidate.coalesceSubmissions);
    assert(!bounded.coalesceSubmissions);
    assert(baseline.producerOffsets == candidate.producerOffsets);
    assert(baseline.completionDelta == candidate.completionDelta);
    assert(baseline.byteCount == candidate.byteCount);
    assert(baseline.maximumPendingBytes == candidate.maximumPendingBytes);
    assert(baseline.pages.size() == candidate.pages.size());
    Sync completion {8, 5};
    for (size_t i=0; i<baseline.pages.size(); ++i) {
        const auto& a=baseline.pages[i]; const auto& b=candidate.pages[i];
        assert(a.bytes == b.bytes && a.endsCompletion == b.endsCompletion);
        assert(a.completionOffset == b.completionOffset && a.patches.size() == b.patches.size());
        std::vector<uint8_t> left(a.bytes.size()), right(b.bytes.size());
        NativeComputeProgram::writePage(left.data(), a, &completion);
        NativeComputeProgram::writePage(right.data(), b, &completion);
        assert(left == right);
    }
    setenv("VLLM_HPU_DSV41_DSPARK_NATIVE_PAGE_COALESCE", "true", 1);
    assert(!NativeComputeProgram::prepare({&pre,&post},{1},{wait},16,4096).coalesceSubmissions);
    unsetenv("VLLM_HPU_DSV41_DSPARK_NATIVE_PAGE_COALESCE");
    std::cout << "Capture policy, relocation bytes, retirement targets and capacity checks passed\n";
}
