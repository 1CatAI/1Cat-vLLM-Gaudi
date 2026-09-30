// SPDX-License-Identifier: Apache-2.0
#include "infra/scal/gen2_arch_common/native_hcl_program.h"
#include "platform/gaudi2/g2_sched_pkts.h"
#include "gaudi2/asic_reg_structs/sob_objs_regs.h"
#include <cassert>
#include <cstring>
#include <iostream>

template<class Function> void rejects(Function&& function)
{
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

int main()
{
    using namespace hcl;
    using Packet = g2fw::sched_arc_cmd_lbw_write_t;
    static_assert(offsetof(Packet, src_data) + sizeof(uint32_t) == sizeof(Packet));
    const uint32_t base = 0xf8000000u;
    for (uint32_t index = 0; index < 2048; ++index)
    {
        gaudi2::sob_objs::reg_mon_arm arm {};
        arm.sid = index / 8;
        arm.mask = uint8_t(~(1u << (index % 8)));
        arm.sop = index & 1;
        arm.sod = index * 13;
        assert(nativeHclShortMonitorIndex(arm._raw) == index);
        assert(nativeHclShortMonitorBits(0, base, base + index * 4) == (arm._raw & 0xffffu));
    }
    size_t patches = 0;
    for (uint32_t capturedSlot = 0; capturedSlot < 64; ++capturedSlot)
    {
        for (uint32_t ringBase : {0u, 56u, 504u, 960u, 1536u, 1984u})
        {
            const uint32_t capturedIndex = ringBase + capturedSlot;
            gaudi2::sob_objs::reg_mon_arm arm {};
            arm.sid = capturedIndex / 8;
            arm.mask = uint8_t(~(1u << (capturedIndex % 8)));
            arm.sop = capturedSlot & 1;
            arm.sod = 32767 - capturedSlot;
            Packet captured {};
            captured.opcode = 3;
            captured.fence = 1;
            captured.fence_id = 17;
            captured.target = 9;
            captured.dst_addr = 0xfe123400;
            captured.src_data = arm._raw;
            NativeHclStreamTemplate stream;
            stream.commands.resize(1);
            auto& bytes = stream.commands[0].bytes;
            bytes.resize(sizeof(captured));
            std::memcpy(bytes.data(), &captured, sizeof(captured));
            stream.relocations.push_back({.commandIndex=0, .byteOffset=offsetof(Packet, src_data),
                .width=4, .mask=0xffffu, .destinationShift=0, .sourceShift=0,
                .kind=NativeHclRelocationKind::GPSO_0_SHORT_MONITOR});
            const auto program = NativeHclProgram::prepare({{&stream, 0}}, 64);
            for (uint32_t step = 0; step < 260; ++step)
            {
                const uint32_t index = ringBase + (capturedSlot + step) % 64;
                NativeHclReplayState state;
                state.values[size_t(NativeHclRelocationKind::GPSO_0_SHORT_MONITOR)] =
                    nativeHclShortMonitorBits(capturedIndex, base + capturedIndex * 4, base + index * 4);
                std::vector<uint8_t> output(program.pages()[0].bytes.size());
                NativeHclProgram::writePage(output.data(), program.pages()[0], &state);
                arm.sid = index / 8;
                arm.mask = uint8_t(~(1u << (index % 8)));
                Packet expected = captured;
                expected.src_data = arm._raw;
                assert(std::memcmp(output.data(), &expected, sizeof(expected)) == 0);
                assert(std::memcmp(bytes.data(), &captured, sizeof(captured)) == 0);
                ++patches;
            }
        }
    }
    rejects([] { nativeHclShortMonitorIndex(0xffff); });
    rejects([] { nativeHclShortMonitorIndex(0); });
    rejects([&] { nativeHclShortMonitorBits(0, base, base - 4); });
    rejects([&] { nativeHclShortMonitorBits(2047, base, base + 4); });
    rejects([&] { nativeHclShortMonitorBits(16, base, base + 1); });
    rejects([&] { nativeHclShortMonitorBits(2048, base, base); });
    std::cout << "2048 ASIC index encodings, " << patches
              << " actual packet replay/rotation checks and six rejection cases passed\n";
}
