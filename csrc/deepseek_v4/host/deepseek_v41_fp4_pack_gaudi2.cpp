// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_fp4_pack_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_fp4_pack_g16_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_fp4_pack_g16_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_fp4_pack_g32_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_fp4_pack_g32_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_fp4_cache_write_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_fp4_cache_write_bf16_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41Fp4PackGaudi2::GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, mode_ == 16 ? "custom_deepseek_v41_fp4_pack_g16_bf16_gaudi2" :
                      mode_ == 32 ? "custom_deepseek_v41_fp4_pack_g32_bf16_gaudi2" :
                                    "custom_deepseek_v41_fp4_cache_write_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn DeepseekV41Fp4PackGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    const bool write = mode_ == 0;
    const unsigned inputs = write ? 5 : 1;
    if (in->inputTensorNr != inputs) { in->inputTensorNr = inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& result = in->outputTensors[0].geometry;
    if (write) {
        const TensorDataType types[] = {DATA_U8, DATA_U8, DATA_BF16, DATA_BF16, DATA_I32};
        const auto rows = in->inputTensors[0].geometry.maxSizes[1];
        for (unsigned i = 0; i < 5; ++i) {
            const auto& geometry = in->inputTensors[i].geometry;
            if (geometry.dataType != types[i]) return GLUE_INCOMPATIBLE_DATA_TYPE;
            bool valid = geometry.dims == (i == 4 ? 1u : 2u);
            if (i < 2) valid &= geometry.maxSizes[0] == (i == 0 ? 288u : 68u) &&
                                rows > 0 && rows <= 1024 && geometry.maxSizes[1] == rows;
            else if (i < 4) valid &= geometry.maxSizes[0] == (i == 2 ? 512u : 128u) && geometry.maxSizes[1] == 1;
            else valid &= geometry.maxSizes[0] == 1;
            if (!valid) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            // Two different group sizes share one C1 operation. The inputs
            // are small complete rows; cache addresses depend on the token.
            out->inputTensorAccessPattern[i].allRequired = true;
        }
        if (result.dataType != DATA_I32) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (result.dims != 1 || result.maxSizes[0] != 36) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->outputTensorAccessPattern[0].mapping[0] = {0, 1, 0, 0};
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = 36;
    } else {
        const auto& value = in->inputTensors[0].geometry;
        const auto width = value.maxSizes[0], rows = value.maxSizes[1];
        if (value.dataType != DATA_BF16 || result.dataType != DATA_U8) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (value.dims != 2 || width == 0 || width > 16384 || width % mode_ || rows == 0 || rows > 8192)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (result.dims != 2 || result.maxSizes[0] != width / 2 + width / mode_ || result.maxSizes[1] != rows)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->inputTensorAccessPattern[0].mapping[0] = {0, static_cast<float>(mode_), 0, static_cast<float>(mode_ - 1)};
        out->inputTensorAccessPattern[0].mapping[1] = {1, 1, 0, 0};
        out->outputTensorAccessPattern[0].mapping[0] = {0, 1, 0, static_cast<float>(width / 2)};
        out->outputTensorAccessPattern[0].mapping[1] = {1, 1, 0, 0};
        out->indexSpaceRank = 2;
        out->indexSpaceGeometry[0] = width / mode_;
        out->indexSpaceGeometry[1] = rows;
    }
    out->kernel.paramsNr = 0;
    auto* start = write ? &_binary___deepseek_v41_fp4_cache_write_bf16_gaudi2_o_start :
                  mode_ == 16 ? &_binary___deepseek_v41_fp4_pack_g16_bf16_gaudi2_o_start :
                                &_binary___deepseek_v41_fp4_pack_g32_bf16_gaudi2_o_start;
    auto* end = write ? &_binary___deepseek_v41_fp4_cache_write_bf16_gaudi2_o_end :
                mode_ == 16 ? &_binary___deepseek_v41_fp4_pack_g16_bf16_gaudi2_o_end :
                              &_binary___deepseek_v41_fp4_pack_g32_bf16_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
