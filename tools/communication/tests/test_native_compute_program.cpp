// SPDX-License-Identifier: Apache-2.0
#include "native_compute_program.hpp"
#include <array>
#include <cassert>
#include <iostream>

struct Sync { uint32_t longSoIndex; uint64_t targetValue; };

int main()
{
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
    std::cout << "native compute program: offsets/capacity/128 dependencies and 15/30/45/60-bit carry boundaries exact\n";
}
