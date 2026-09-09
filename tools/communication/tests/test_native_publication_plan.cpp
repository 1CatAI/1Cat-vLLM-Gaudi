// SPDX-License-Identifier: Apache-2.0
#include "native_publication_plan.hpp"
#include <array>
#include <cassert>
#include <iostream>

// Independent CCB oracle: advance the producer index using the scheduler's
// alignment rules and only allow completions from actually published bytes.
// Reclamation deliberately drains late, at the two-chunk guard, to detect
// waits whose producing command has been written but never published.
struct Ring
{
    uint64_t pi, publishedPi = 0, publishedCompletion = 0, writtenCompletion = 0;
    uint64_t submits = 0, wraps = 0, waits = 0, base = 32760;
    const uint32_t size, chunks, alignment;
    std::vector<uint64_t> guards;
    std::vector<uint8_t> stream;

    Ring(uint32_t size, uint32_t chunks, uint32_t alignment, uint64_t phase)
    : pi(phase), size(size), chunks(chunks), alignment(alignment), guards(chunks) {}

    void publish()
    {
        assert(pi > publishedPi);
        publishedPi = pi;
        publishedCompletion = writtenCompletion;
        ++submits;
    }

    void replay(const NativeComputeProgram& program, const NativePublicationPlan* plan)
    {
        const uint64_t chunk = size / chunks;
        const auto* phase = plan ? &plan->at(pi % size) : nullptr;
        bool pending = false;
        struct Sync { uint64_t targetValue; };
        std::vector<Sync> completions(program.producerOffsets.size());
        for (size_t n = 0; n < completions.size(); ++n) completions[n].targetValue = base + n + 1;
        for (size_t n = 0; n < program.pages.size(); ++n)
        {
            const auto& page = program.pages[n];
            if (phase && phase->publishBefore[n] && pending) { publish(); pending = false; }
            const uint64_t before = pi;
            const auto alignedEnd = ((pi / alignment) + 1) * alignment;
            if (pi + page.bytes.size() > alignedEnd)
            {
                stream.insert(stream.end(), alignedEnd - pi, 0x7f);
                pi = alignedEnd;
                if (!plan) publish();
            }
            if (pi % chunk == 0)
            {
                const auto guard = guards[((pi / chunk) + 2) % chunks];
                if (guard)
                {
                    ++waits;
                    // A future or merely written completion cannot satisfy a wait.
                    assert(guard <= publishedCompletion);
                }
                guards[(pi / chunk) % chunks] = base + page.completionOffset;
            }
            const auto offset = stream.size();
            stream.resize(offset + page.bytes.size());
            NativeComputeProgram::writePage(stream.data() + offset, page, completions.data());
            pi += page.bytes.size();
            wraps += pi / size - before / size;
            if (page.endsCompletion) writtenCompletion = base + page.completionOffset;
            pending = true;
            if (!plan || n + 1 == program.pages.size()) { publish(); pending = false; }
        }
        assert(publishedPi == pi);
        if (phase) assert(pi % chunk == phase->finalOffset);
        base += program.completionDelta;
    }
};

int main()
{
    NativeScalCommandBuffer segment;
    for (unsigned size : {48, 80, 48})
        segment.commands.push_back({std::vector<uint8_t>(size, uint8_t(size)), false, false, false});
    segment.commands.back().completesChunk = true;
    NativeScalCommandBuffer terminal = segment;
    terminal.commands.push_back({std::vector<uint8_t>(8, 0x73), true, false, false});
    std::vector<NativeComputeWaitPacket> waits;
    for (uint8_t part : {1, 2, 3, 0})
        waits.push_back({std::vector<uint8_t>(8, 0), {{4, 0, 0xfffe0000u, uint8_t(part * 15), 17}}});
    std::vector<const NativeScalCommandBuffer*> segments(129, &segment);
    segments.back() = &terminal;
    std::vector<uint32_t> consumers;
    for (uint32_t n = 1; n < segments.size(); ++n) consumers.push_back(n);
    auto program = NativeComputeProgram::prepare(segments, consumers, waits, 256, 7168, true);
    for (uint32_t buffer : {8192, 131072})
    {
        auto plan = NativePublicationPlan::prepare(program, 256, buffer, 16);
        // Every possible word-aligned phase, including alignment padding that
        // lands exactly on a reclaim boundary. Compare complete emitted bytes.
        for (uint32_t phase = 0; phase < plan.chunkBytes; phase += 4)
        {
            Ring immediate(buffer, 16, 256, phase), batched(buffer, 16, 256, phase);
            immediate.replay(program, nullptr);
            batched.replay(program, &plan);
            assert(immediate.stream == batched.stream && immediate.pi == batched.pi);
            assert(batched.submits < immediate.submits);
        }
        Ring delayed(buffer, 16, 256, 252);
        for (unsigned replay = 0; replay < 4096; ++replay)
        {
            delayed.replay(program, &plan);
            delayed.stream.clear();
        }
        assert(delayed.wraps > 1 && delayed.waits > 1 && delayed.base > 32768);
        std::cout << "buffer=" << buffer << " wraps=" << delayed.wraps << " reclaim_checks=" << delayed.waits
                  << " publications=" << delayed.submits << " final_completion=" << delayed.base << '\n';
    }
    for (const auto& geometry : std::vector<std::array<uint32_t, 3>>{{256, 8192, 2}, {256, 8191, 16},
                                                                   {3, 8192, 16}, {1024, 8192, 16}})
    {
        bool rejected = false;
        try { NativePublicationPlan::prepare(program, geometry[0], geometry[1], geometry[2]); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    std::cout << "publication phases: exact bytes, delayed reclaim, terminal guards, wraps and SO carry passed\n";
}
