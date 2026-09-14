// SPDX-License-Identifier: Apache-2.0
#include "native_compute_program.hpp"
#include <array>
#include <cassert>
#include <iostream>

struct Sync { uint32_t longSoIndex; uint64_t targetValue; };

void testContinuationInput()
{
    using Input = NativeComputeProgram::ContinuationInput;
    assert(NativeComputeProgram::continuationProducerOffset(2, 3) == 0);
    assert(NativeComputeProgram::continuationProducerOffset(3, 3) == 0);
    assert(NativeComputeProgram::continuationProducerOffset(4, 3) == 1);
    assert(NativeComputeProgram::continuationProducerOffset(6, 3) == 3);
    assert(NativeComputeProgram::producerAvailableToPrefix(UINT32_MAX, 3));
    assert(NativeComputeProgram::producerAvailableToPrefix(0, 3));
    assert(NativeComputeProgram::producerAvailableToPrefix(2, 3));
    assert(!NativeComputeProgram::producerAvailableToPrefix(3, 3));
    assert(!NativeComputeProgram::producerAvailableToPrefix(4, 3));
    assert(NativeComputeProgram::continuationInput(8, 100, 8, 100, false) == Input::Ready);
    assert(NativeComputeProgram::continuationInput(8, 100, 8, 100, true) == Input::NeedsFence);
    assert(NativeComputeProgram::continuationInput(8, 100, 8, 101, false) == Input::Ready);
    assert(NativeComputeProgram::continuationInput(8, 100, 8, 101, true) == Input::NeedsFence);
    for (bool wait : {false, true}) {
        assert(NativeComputeProgram::continuationInput(8, 100, 16, 100, wait) == Input::Invalid);
        assert(NativeComputeProgram::continuationInput(8, 100, 8, 99, wait) == Input::Invalid);
        assert(NativeComputeProgram::continuationInput(8, 0, 8, 100, wait) == Input::Invalid);
        assert(NativeComputeProgram::continuationInput(8, 100, 8, 1ULL << 60, wait) == Input::Invalid);
    }
}

void testSegmentedPublication()
{
    NativeScalCommandBuffer segment;
    segment.commands = {{{1, 2, 3, 4}, false, true, true}, {{5, 6, 7, 8}, true, false, false}};
    std::vector<const NativeScalCommandBuffer*> segments(8, &segment);
    std::vector<NativeComputeWaitPacket> waits;
    for (uint8_t part : {1, 2, 3, 0})
        waits.push_back({{0, 0, 0, 0}, {{0, 0, 0xfffe0000u, uint8_t(part * 15), 17}}});
    auto whole = NativeComputeProgram::prepare(segments, {1, 3, 5, 5, 7}, waits, 256, 32768,
                                               true, false, {0, 1, 3, 3, 5});
    size_t earlyCarry;
    const auto early = whole.split(whole.segmentPageEnds[2], whole.segmentPageByteEnds[2],
                                   whole.segmentPagePacketEnds[2], whole.segmentCompletionOffsets[2], 2, earlyCarry);
    assert(earlyCarry == 1 && early.first.completionDelta == 3 && early.second.completionDelta == 5);
    assert(early.first.lastWaitDependency() == 0);
    assert(NativeComputeProgram::continuationProducerOffset(whole.producerOffsets[2], 3) == 1);
    assert(NativeComputeProgram::continuationProducerOffset(whole.producerOffsets[3], 3) == 1);
    const size_t pages = whole.segmentPageEnds[4];
    assert(whole.segmentPageByteEnds[4] < whole.pages[pages - 1].bytes.size());
    size_t carry;
    auto parts = whole.split(pages, whole.segmentPageByteEnds[4], whole.segmentPagePacketEnds[4],
                             whole.segmentCompletionOffsets[4], 3, carry);
    const auto& prefix = parts.first;
    const auto& suffix = parts.second;
    assert(carry == 2 && prefix.completionDelta == 5 && suffix.completionDelta == 3);
    assert(prefix.producerOffsets.size() == 3 && prefix.lastWaitDependency() == 1);
    assert(suffix.lastWaitDependency() == 2 && whole.lastWaitDependency() == 4);
    assert(prefix.byteCount + suffix.byteCount == whole.byteCount);
    assert(prefix.sourcePacketCount + suffix.sourcePacketCount == whole.sourcePacketCount);
    assert(!prefix.pages.back().endsCompletion && prefix.pages.back().completionOffset == 6);
    auto emitted = [](const NativeComputeProgram& program, const Sync* completions) {
        std::vector<uint8_t> result;
        for (const auto& page : program.pages) {
            const size_t begin = result.size();
            result.resize(begin + page.bytes.size());
            NativeComputeProgram::writePage(result.data() + begin, page, completions);
        }
        return result;
    };
    for (uint64_t start : {32765ULL, (1ULL << 30) - 3, (1ULL << 45) - 3}) {
        std::array<Sync, 5> completions;
        for (size_t i = 0; i < completions.size(); ++i) completions[i] = {8, start + i};
        auto splitBytes = emitted(prefix, completions.data());
        auto suffixBytes = emitted(suffix, completions.data() + carry);
        splitBytes.insert(splitBytes.end(), suffixBytes.begin(), suffixBytes.end());
        assert(splitBytes == emitted(whole, completions.data()));
        auto earlyBytes = emitted(early.first, completions.data());
        auto lateBytes = emitted(early.second, completions.data() + earlyCarry);
        earlyBytes.insert(earlyBytes.end(), lateBytes.begin(), lateBytes.end());
        assert(earlyBytes == emitted(whole, completions.data()));
    }
    NativeComputeProgram noWait;
    assert(noWait.lastWaitDependency() == std::numeric_limits<size_t>::max());

    auto externalWhole = NativeComputeProgram::prepare(
        segments, {1, 3, 5, 5, 7}, waits, 256, 32768, true, true, {UINT32_MAX, 1, 3, 3, 5});
    size_t externalCarry;
    auto externalParts = externalWhole.split(
        externalWhole.segmentPageEnds[4], externalWhole.segmentPageByteEnds[4],
        externalWhole.segmentPagePacketEnds[4], externalWhole.segmentCompletionOffsets[4], 3, externalCarry);
    assert(externalCarry == 2);
    assert(externalParts.first.externalInputCompletion);
    assert(externalParts.first.producerOffsets.front() == 0);
    assert(!externalParts.second.externalInputCompletion);
}

int main()
{
    testContinuationInput();
    testSegmentedPublication();
    NativeScalCommandBuffer pre, post;
    pre.completionIncrementsBeforeCommands = 1;
    pre.commands = {{{0x81, 0x82, 0x83, 0x84}, false, true, true}};
    post = pre;
    std::vector<NativeComputeWaitPacket> wait;
    for (uint8_t part : {1, 2, 3, 0})
        wait.push_back({{0x91, 0x92, 0x93, 0x94, 0x7f, 0xff, 0x01, 0}, {{4, 0, 0xfffe0000u, uint8_t(part*15), 17}}});
    auto program = NativeComputeProgram::prepare({&pre, &post}, {1}, wait, 16, 128);
    assert(program.completionDelta == 4 && program.producerOffsets == std::vector<uint64_t>{2});
    assert(program.pages.front().completionOffset == 2 && program.pages.back().completionOffset == 4);
    const uint64_t values[] = {1, 32767, 32768, (1ULL<<30)-1, 1ULL<<30, (1ULL<<45)-1,
                               1ULL<<45, (1ULL<<60)-1};
    for (auto value : values)
    {
        Sync sync{8, value};
        unsigned patched = 0;
        for (const auto& page : program.pages)
        {
            std::vector<uint8_t> bytes(page.bytes.size());
            NativeComputeProgram::writePage(bytes.data(), page, &sync);
            for (const auto& patch : page.patches)
            {
                uint32_t word; std::memcpy(&word, bytes.data()+patch.byteOffset, 4);
                assert((word & ~patch.mask) == 0x1ff7fu);
                assert((word >> 17) == ((value >> patch.sourceShift) & 0x7fff));
                assert(bytes[patch.byteOffset-4] == 0x91);
                ++patched;
            }
        }
        assert(patched == 4);
    }
    std::vector<const NativeScalCommandBuffer*> buffers(257, &pre);
    std::vector<uint32_t> consumers;
    for (uint32_t i=1;i<257;i+=2) consumers.push_back(i);
    auto full = NativeComputeProgram::prepare(buffers, consumers, wait, 256, 32768);
    assert(full.producerOffsets.size() == 128 && full.completionDelta == 514);
    for (size_t i=0;i<128;++i) assert(full.producerOffsets[i] == (i*2+1)*2);
    auto external = NativeComputeProgram::prepare({&pre, &post}, {0, 1}, wait, 16, 128, false, true);
    assert(external.externalInputCompletion && external.producerOffsets == std::vector<uint64_t>({0, 2}));
    assert(external.canReplayAt(32767) && external.canReplayAt((1ULL << 60) - external.completionDelta - 3));
    assert(!external.canReplayAt((1ULL << 60) - external.completionDelta - 2));
    assert(full.canReplayAt((1ULL << 60) - full.completionDelta - 2));
    assert(!full.canReplayAt((1ULL << 60) - full.completionDelta - 1));
    auto shared = NativeComputeProgram::prepare({&pre, &post}, {1, 1}, wait, 16, 256);
    assert(shared.producerOffsets == std::vector<uint64_t>({2, 2}));
    assert(shared.completionDelta == program.completionDelta);
    auto overlap = NativeComputeProgram::prepare({&pre, &pre, &post}, {2}, wait, 16, 256,
                                                  false, false, {0});
    assert(overlap.producerOffsets == std::vector<uint64_t>{2});
    assert(overlap.completionDelta == 6);
    uint64_t priorCompletion = 0;
    for (const auto& page : overlap.pages) {
        if (!page.patches.empty()) assert(priorCompletion >= 4);
        if (page.endsCompletion) priorCompletion = page.completionOffset;
    }
    auto distinct = NativeComputeProgram::prepare({&pre, &pre, &post}, {2, 2}, wait, 16, 256,
                                                   false, false, {0, 1});
    assert(distinct.producerOffsets == std::vector<uint64_t>({2, 4}));
    auto delayedExternal = NativeComputeProgram::prepare({&pre, &post}, {1}, wait, 16, 128,
                                                          false, true, {UINT32_MAX});
    assert(delayedExternal.externalInputCompletion && delayedExternal.producerOffsets[0] == 0);
    for (const auto& invalid : std::vector<std::vector<uint32_t>>{{2}, {3}, {UINT32_MAX}, {0, 1}}) {
        bool bad = false;
        try { NativeComputeProgram::prepare({&pre, &pre, &post}, {2}, wait, 16, 256,
                                             false, false, invalid); }
        catch (const std::invalid_argument&) { bad = true; }
        assert(bad);
    }
    std::array<unsigned, 2> wait_counts{};
    const Sync independent[] = {{8, 32767}, {8, 32768}};
    for (const auto& page : shared.pages) {
        std::vector<uint8_t> bytes(page.bytes.size());
        NativeComputeProgram::writePage(bytes.data(), page, independent);
        for (const auto& patch : page.patches) {
            uint32_t word; std::memcpy(&word, bytes.data() + patch.byteOffset, 4);
            assert((word >> 17) == ((independent[patch.dependency].targetValue >> patch.sourceShift) & 0x7fff));
            ++wait_counts.at(patch.dependency);
        }
    }
    assert(wait_counts[0] == 4 && wait_counts[1] == 4);
    for (const auto& invalid : std::vector<std::vector<uint32_t>>{{0},{2},{1,0}})
    {
        bool rejected=false;
        try { NativeComputeProgram::prepare({&pre,&post}, invalid, wait, 16, 128); }
        catch (const std::invalid_argument&) { rejected=true; }
        assert(rejected);
    }
    bool rejected=false;
    try { NativeComputeProgram::prepare({&pre,&post}, {1}, wait, 16, 16); }
    catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
    post.commands.back().completesChunk=false;
    rejected=false;
    try { NativeComputeProgram::prepare({&pre,&post}, {1}, wait, 16, 128); }
    catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
    NativeScalCommandBuffer trailing = pre, terminal = pre;
    trailing.commands.push_back({{0x71,0x72,0x73,0x74}, true, false, false});
    auto deferred = NativeComputeProgram::prepare({&pre,&trailing}, {1}, wait, 16, 128, true);
    assert(deferred.completionDelta == 4 && deferred.pages.back().completionOffset == 5);
    terminal.completionIncrementsBeforeCommands = 0;
    auto retained = NativeComputeProgram::prepare({&pre,&trailing,&terminal}, {1}, wait, 16, 128);
    assert(retained.completionDelta == 5 && retained.pages.back().completionOffset == 5);
    assert(retained.pages.back().bytes.front() == 0x71);
    std::cout << "native compute program: offsets/capacity/128 dependencies and "
                 "15/30/45/60-bit carry boundaries exact\n";
}
