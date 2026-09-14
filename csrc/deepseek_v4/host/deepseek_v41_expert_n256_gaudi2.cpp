// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_expert_n256_gaudi2.hpp"
#include <cstring>
#include <initializer_list>

#define ELF(name) extern unsigned char _binary___deepseek_v41_expert_n256_##name##_gaudi2_o_start; \
                  extern unsigned char _binary___deepseek_v41_expert_n256_##name##_gaudi2_o_end;
ELF(fp8)
ELF(bf16)
ELF(scale)
ELF(silu_quant)
#undef ELF
namespace {
void map(tpc_lib_api::TensorAccessPattern& p, unsigned dim, unsigned axis,
         int coefficient, int first, int last) {
    p.mapping[dim].indexSpaceDim = axis;
    p.mapping[dim].a = coefficient;
    p.mapping[dim].start_b = first;
    p.mapping[dim].end_b = last;
}
tpc_lib_api::GlueCodeReturn silu_quant(tpc_lib_api::HabanaKernelParams* in,
                                      tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const TensorDataType types[] = {DATA_F32, DATA_I32, DATA_F32, DATA_BF16, DATA_F32};
    for (unsigned i = 0; i < 5; ++i) {
        if (in->inputTensors[i].geometry.dataType != types[i]) {
            in->inputTensors[i].geometry.dataType = types[i]; return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    const auto& p = in->inputTensors[0].geometry;
    const auto& ids = in->inputTensors[1].geometry;
    const auto& sx = in->inputTensors[2].geometry;
    const auto& sw = in->inputTensors[3].geometry;
    const auto& router = in->inputTensors[4].geometry;
    const uint64_t width = p.maxSizes[0] / 2, rows = p.maxSizes[2];
    if (p.dims != 3 || p.maxSizes[1] != 1 || !width || width % 128 || width > 2560 ||
        !rows || rows > 6 || ids.dims != 2 || ids.maxSizes[0] != rows || ids.maxSizes[1] != 1 ||
        sx.dims != 2 || sx.maxSizes[0] != 1 || sx.maxSizes[1] != 1 ||
        sw.dims != 3 || sw.maxSizes[0] != 256 || sw.maxSizes[1] * 256 != width * 2 ||
        !sw.maxSizes[2] || sw.maxSizes[2] > 384 || router.dims != 2 ||
        router.maxSizes[0] != rows || router.maxSizes[1] != 1)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = rows;
    map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, width * 2 - 1);
    map(out->inputTensorAccessPattern[0], 1, 0, 0, 0, 0);
    map(out->inputTensorAccessPattern[0], 2, 0, 1, 0, 0);
    for (unsigned i : {1u, 4u}) {
        map(out->inputTensorAccessPattern[i], 0, 0, 1, 0, 0);
        map(out->inputTensorAccessPattern[i], 1, 0, 0, 0, 0);
    }
    out->inputTensorAccessPattern[2].allRequired = true;
    out->inputTensorAccessPattern[3].allRequired = true;
    for (unsigned i = 0; i < 2; ++i) {
        auto& result = in->outputTensors[i].geometry;
        const uint64_t n = i == 0 ? width : 1;
        const auto type = i == 0 ? DATA_F8_143 : DATA_F32;
        if (result.dataType != type) { result.dataType = type; return GLUE_INCOMPATIBLE_DATA_TYPE; }
        if (result.dims != 3 || result.maxSizes[0] != n || result.maxSizes[1] != 1 || result.maxSizes[2] != rows) {
            result.dims = 3; result.maxSizes[0] = n; result.maxSizes[1] = 1; result.maxSizes[2] = rows;
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        }
        map(out->outputTensorAccessPattern[i], 0, 0, 0, 0, n - 1);
        map(out->outputTensorAccessPattern[i], 1, 0, 0, 0, 0);
        map(out->outputTensorAccessPattern[i], 2, 0, 1, 0, 0);
    }
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___deepseek_v41_expert_n256_silu_quant_gaudi2_o_start;
    const auto* end = &_binary___deepseek_v41_expert_n256_silu_quant_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
}
tpc_lib_api::GlueCodeReturn DeepseekV41ExpertN256Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, mode_ == FP8 ? "custom_deepseek_v41_expert_n256_fp8_gaudi2" :
        mode_ == BF16 ? "custom_deepseek_v41_expert_n256_bf16_gaudi2" :
        mode_ == Scale ? "custom_deepseek_v41_expert_n256_scale_gaudi2" :
                         "custom_deepseek_v41_expert_n256_silu_quant_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn DeepseekV41ExpertN256Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (mode_ == SiluQuant) return silu_quant(in, out);
    if (in->inputTensorNr != 4) { in->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const TensorDataType decodeTypes[] = {DATA_I32, DATA_I16, DATA_I16, DATA_BF16};
    const TensorDataType scaleTypes[] = {DATA_F32, DATA_I32, DATA_F32, DATA_BF16};
    for (unsigned i = 0; i < 4; ++i) {
        const auto expected = mode_ == Scale ? scaleTypes[i] : decodeTypes[i];
        if (in->inputTensors[i].geometry.dataType != expected) {
            in->inputTensors[i].geometry.dataType = expected;
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    auto& result = in->outputTensors[0].geometry;
    const auto outputType = mode_ == FP8 ? DATA_F8_143 : DATA_BF16;
    if (result.dataType != outputType) {
        result.dataType = outputType; return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    uint64_t n, k, slots;
    if (mode_ != Scale) {
        const auto& ids = in->inputTensors[0].geometry;
        const auto& q = in->inputTensors[1].geometry;
        const auto& s = in->inputTensors[2].geometry;
        const auto& lut = in->inputTensors[3].geometry;
        if (ids.dims != 2 || ids.maxSizes[1] != 1 || !ids.maxSizes[0] || ids.maxSizes[0] > 3072 ||
            q.dims != 3 || !q.maxSizes[0] || q.maxSizes[0] % 8192 || q.maxSizes[0] > 327680 ||
            !q.maxSizes[1] || q.maxSizes[1] > 20 || !q.maxSizes[2] || q.maxSizes[2] > 384 ||
            s.dims != 3 || s.maxSizes[0] * 8 != q.maxSizes[0] || s.maxSizes[1] != q.maxSizes[1] ||
            s.maxSizes[2] != q.maxSizes[2] || lut.dims != 1 || lut.maxSizes[0] != 128)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        n = q.maxSizes[1] * 256; k = q.maxSizes[0] / 64; slots = ids.maxSizes[0];
        out->indexSpaceRank = 3;
        out->indexSpaceGeometry[0] = q.maxSizes[1];
        out->indexSpaceGeometry[1] = slots;
        out->indexSpaceGeometry[2] = k / 128;
        map(out->inputTensorAccessPattern[0], 0, 1, 1, 0, 0);
        map(out->inputTensorAccessPattern[0], 1, 0, 0, 0, 0);
        for (unsigned i : {1u, 2u}) {
            const int words = i == 1 ? 8192 : 1024;
            map(out->inputTensorAccessPattern[i], 0, 2, words, 0, words - 1);
            map(out->inputTensorAccessPattern[i], 1, 0, 1, 0, 0);
            map(out->inputTensorAccessPattern[i], 2, 1, 0, 0, q.maxSizes[2] - 1);
        }
        map(out->inputTensorAccessPattern[3], 0, 0, 0, 0, 127);
        map(out->outputTensorAccessPattern[0], 0, 0, 256, 0, 255);
        map(out->outputTensorAccessPattern[0], 1, 2, 128, 0, 127);
    } else {
        const auto& p = in->inputTensors[0].geometry;
        const auto& ids = in->inputTensors[1].geometry;
        const auto& sx = in->inputTensors[2].geometry;
        const auto& sw = in->inputTensors[3].geometry;
        if (p.dims != 3 || !p.maxSizes[0] || p.maxSizes[0] % 256 || p.maxSizes[1] != 1 ||
            ids.dims != 2 || ids.maxSizes[1] != 1 || ids.maxSizes[0] != p.maxSizes[2] ||
            sx.dims != 2 || sx.maxSizes[0] != 1 || (sx.maxSizes[1] != 1 && sx.maxSizes[1] != p.maxSizes[2]) ||
            sw.dims != 3 || sw.maxSizes[0] != 256 || sw.maxSizes[1] * 256 != p.maxSizes[0] ||
            !sw.maxSizes[2] || sw.maxSizes[2] > 384)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        n = p.maxSizes[0]; k = 1; slots = p.maxSizes[2];
        out->indexSpaceRank = 2;
        out->indexSpaceGeometry[0] = n / 64; out->indexSpaceGeometry[1] = slots;
        map(out->inputTensorAccessPattern[0], 0, 0, 64, 0, 63);
        map(out->inputTensorAccessPattern[0], 1, 0, 0, 0, 0);
        map(out->inputTensorAccessPattern[0], 2, 1, 1, 0, 0);
        map(out->inputTensorAccessPattern[1], 0, 1, 1, 0, 0);
        map(out->inputTensorAccessPattern[1], 1, 0, 0, 0, 0);
        out->inputTensorAccessPattern[2].allRequired = true;
        out->inputTensorAccessPattern[3].allRequired = true;
        map(out->outputTensorAccessPattern[0], 0, 0, 64, 0, 63);
        map(out->outputTensorAccessPattern[0], 1, 0, 0, 0, 0);
    }
    map(out->outputTensorAccessPattern[0], 2, 1, 1, 0, 0);
    if (result.dims != 3 || result.maxSizes[0] != n || result.maxSizes[1] != k || result.maxSizes[2] != slots) {
        result.dims = 3; result.maxSizes[0] = n; result.maxSizes[1] = k; result.maxSizes[2] = slots;
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->kernel.paramsNr = 0;
    const unsigned char* start = mode_ == FP8 ? &_binary___deepseek_v41_expert_n256_fp8_gaudi2_o_start :
        mode_ == BF16 ? &_binary___deepseek_v41_expert_n256_bf16_gaudi2_o_start : &_binary___deepseek_v41_expert_n256_scale_gaudi2_o_start;
    const unsigned char* end = mode_ == FP8 ? &_binary___deepseek_v41_expert_n256_fp8_gaudi2_o_end :
        mode_ == BF16 ? &_binary___deepseek_v41_expert_n256_bf16_gaudi2_o_end : &_binary___deepseek_v41_expert_n256_scale_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
