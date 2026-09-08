// SPDX-License-Identifier: Apache-2.0
#include "infra/scal/gen2_arch_common/native_hcl_program.h"
#include <array>
#include <cassert>
#include <iostream>

int main()
{
    using namespace hcl;
    NativeHclStreamTemplate source;
    source.commands = {{{0xaa, 0xa5, 0xcc, 0xdd}, false}, {{0x11, 0x22, 0x33, 0x44}, true}};
    source.relocations.push_back({.commandIndex = 0, .byteOffset = 1, .width = 1,
                                  .mask = 0x3, .destinationShift = 0, .sourceShift = 0,
                                  .kind = NativeHclRelocationKind::WQE_NOTIFY});
    source.relocations.push_back({.commandIndex = 1, .byteOffset = 0, .width = 4,
                                  .mask = 0xffffffff, .destinationShift = 0, .sourceShift = 15,
                                  .kind = NativeHclRelocationKind::DEPENDENCY_TARGET});
    auto published = NativeHclProgram::prepare({{&source, 0}, {&source, 1}}, 16);
    assert(published.pages().size() == 2 && published.pages()[0].publishAfter && published.pages()[1].publishAfter);
    auto packable = source;
    packable.commands.back().submitAfter = false;
    auto program = NativeHclProgram::prepare({{&packable, 0}, {&packable, 1}}, 16);
    assert(program.pages().size() == 1 && program.bytes() == 16 && program.stateCount() == 2);
    std::array<NativeHclReplayState, 2> states;
    states[0].values[static_cast<size_t>(NativeHclRelocationKind::WQE_NOTIFY)] = 2;
    states[1].values[static_cast<size_t>(NativeHclRelocationKind::WQE_NOTIFY)] = 0;
    states[0].values[static_cast<size_t>(NativeHclRelocationKind::DEPENDENCY_TARGET)] = 32767;
    states[1].values[static_cast<size_t>(NativeHclRelocationKind::DEPENDENCY_TARGET)] = 32768;
    std::array<uint8_t, 16> output;
    NativeHclProgram::writePage(output.data(), program.pages()[0], states.data());
    assert(output[1] == 0xa6 && output[9] == 0xa4);
    uint32_t first, second;
    std::memcpy(&first, output.data() + 4, 4);
    std::memcpy(&second, output.data() + 12, 4);
    assert(first == 0 && second == 1);
    assert(program.pages()[0].lastStateIndex == 1);
    auto pages = NativeHclProgram::prepare({{&source, 0}, {&source, 1}}, 8);
    assert(pages.pages().size() == 2);
    // Compare a multi-node prepared stream with the retained per-command
    // relocation implementation across every completion-ring phase.
    NativeHclStreamTemplate wrap = source;
    wrap.commands.push_back({{0x91, 0x92, 0x93, 0x94}, false});
    for (size_t phase = 0; phase < 64; ++phase)
    {
        std::vector<NativeHclProgram::Input> input;
        std::vector<NativeHclReplayState> epochs(128);
        std::vector<uint8_t> expected, actual;
        for (size_t node = 0; node < 128; ++node)
        {
            const auto& templ = (phase + node) % 64 == 0 ? wrap : source;
            input.push_back({&templ, static_cast<uint32_t>(node)});
            epochs[node].values[static_cast<size_t>(NativeHclRelocationKind::WQE_NOTIFY)] = node % 4;
            epochs[node].values[static_cast<size_t>(NativeHclRelocationKind::DEPENDENCY_TARGET)] =
                (uint64_t(1) << 45) - 64 + phase + node;
            for (size_t command = 0; command < templ.commands.size(); ++command)
            {
                auto bytes = templ.commands[command].bytes;
                for (const auto& field : templ.relocations)
                    if (field.commandIndex == command)
                        assert(patchNativeHclRelocation(bytes.data(), bytes.size(), field, epochs[node]));
                expected.insert(expected.end(), bytes.begin(), bytes.end());
            }
        }
        auto batch = NativeHclProgram::prepare(input, 32, false);
        assert(batch.pages().size() < input.size());
        for (const auto& page : batch.pages()) assert(!page.publishAfter);
        for (const auto& page : batch.pages())
        {
            const auto offset = actual.size();
            actual.resize(offset + page.bytes.size());
            NativeHclProgram::writePage(actual.data() + offset, page, epochs.data());
        }
        assert(actual == expected && batch.stateCount() == 128);
    }
    source.relocations[0].byteOffset = 4;
    bool rejected = false;
    try { NativeHclProgram::prepare({{&source, 0}}, 16); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    std::cout << "native HCL pages: mixed epochs, preserved bits, carry, packing, bounds passed\n";
}
