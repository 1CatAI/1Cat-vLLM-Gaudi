// SPDX-License-Identifier: Apache-2.0
// Exact CPU layout conversion. No FP4 decoding or floating-point arithmetic.
#include <cstdint>
#include <immintrin.h>

namespace {
void scalar(const uint8_t* source, uint8_t* output, int64_t blocks, int64_t k) {
    for (int64_t block = 0; block < blocks / 2; ++block) {
        for (int64_t row = 0; row < k / 2; ++row) {
            for (int half = 0; half != 2; ++half) {
                const auto* input = source + ((2 * block + half) * (k / 2) + row) * 128;
                auto* low = output + (block * k + 2 * row) * 128 + half * 64;
                auto* high = low + 128;
                for (int lane = 0; lane != 64; ++lane) {
                    const auto a = input[2 * lane];
                    const auto b = input[2 * lane + 1];
                    low[lane] = (a & 15) | ((b & 15) << 4);
                    high[lane] = (a >> 4) | (b & 240);
                }
            }
        }
    }
}

__attribute__((target("avx512f,avx512bw")))
void vectorized(const uint8_t* source, uint8_t* output, int64_t blocks, int64_t k) {
    const auto nibble = _mm512_set1_epi16(15);
    const auto upper = _mm512_set1_epi16(240);
    for (int64_t block = 0; block < blocks / 2; ++block) {
        for (int64_t row = 0; row < k / 2; ++row) {
            for (int half = 0; half != 2; ++half) {
                const auto* input = source + ((2 * block + half) * (k / 2) + row) * 128;
                auto* low = output + (block * k + 2 * row) * 128 + half * 64;
                auto* high = low + 128;
                for (int part = 0; part != 2; ++part) {
                    const auto words = _mm512_loadu_si512(input + part * 64);
                    const auto shift4 = _mm512_srli_epi16(words, 4);
                    const auto l = _mm512_or_si512(_mm512_and_si512(words, nibble),
                                                   _mm512_and_si512(shift4, upper));
                    const auto h = _mm512_or_si512(_mm512_and_si512(shift4, nibble),
                                                   _mm512_and_si512(_mm512_srli_epi16(words, 8), upper));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(low + part * 32),
                                        _mm512_cvtepi16_epi8(l));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(high + part * 32),
                                        _mm512_cvtepi16_epi8(h));
                }
            }
        }
    }
}
}  // namespace

extern "C" int dsv41_startup_pack_abi() { return 1; }

extern "C" int dsv41_startup_pack(const uint8_t* source, uint8_t* output, int64_t blocks, int64_t k) {
    if (!source || !output || blocks <= 0 || blocks % 2 || k <= 0 || k % 128 ||
        blocks > (1LL << 20) / k || blocks * k * 64 > (64LL << 20)) {
        return -1;
    }
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) {
        vectorized(source, output, blocks, k);
    } else {
        scalar(source, output, blocks, k);
    }
    return 0;
}
