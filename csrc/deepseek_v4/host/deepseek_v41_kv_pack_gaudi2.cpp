// SPDX-License-Identifier: Apache-2.0
// Pure-codec access mapping shared with the V4.1 C1 native encoders.
#include "deepseek_v41_kv_pack_gaudi2.hpp"
#include <cstring>
#define BINARY(name) extern unsigned char _binary___##name##_o_start; extern unsigned char _binary___##name##_o_end;
BINARY(deepseek_v41_swa_pack_bf16_gaudi2)
BINARY(deepseek_v41_fp4_pack_g16_bf16_gaudi2)
BINARY(deepseek_v41_fp4_pack_g32_bf16_gaudi2)
BINARY(deepseek_v41_fp4_roundtrip_g32_bf16_gaudi2)
#undef BINARY
tpc_lib_api::GlueCodeReturn DeepseekV41KVPackGaudi2::GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, group_ == 0 ? "custom_deepseek_v41_swa_pack_bf16_gaudi2" : group_ == 16
        ? "custom_deepseek_v41_fp4_pack_g16_bf16_gaudi2" : group_ == 32
        ? "custom_deepseek_v41_fp4_pack_g32_bf16_gaudi2"
        : "custom_deepseek_v41_fp4_roundtrip_g32_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn DeepseekV41KVPackGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 1) { in->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& value = in->inputTensors[0].geometry;
    const auto& result = in->outputTensors[0].geometry;
    const bool roundtrip = group_ == FP4_ROUNDTRIP_G32;
    const unsigned group = (group_ == 0 || roundtrip) ? 32 : group_;
    const uint64_t width = value.maxSizes[0], rows = value.maxSizes[1];
    if (value.dataType != DATA_BF16 || result.dataType != (roundtrip ? DATA_BF16 : DATA_U8))
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (value.dims != 2 || width == 0 || width % group || width > 16384 || rows == 0 || rows > 8192)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t codes = group_ ? width / 2 : width;
    const uint64_t result_width = roundtrip ? width : codes + width / group;
    if (result.dims != 2 || result.maxSizes[0] != result_width || result.maxSizes[1] != rows)
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->inputTensorAccessPattern[0].mapping[0] = {0, static_cast<float>(group), 0, static_cast<float>(group - 1)};
    out->inputTensorAccessPattern[0].mapping[1] = {1, 1, 0, 0};
    // The affine union covers distinct value-byte and trailing-scale regions
    // for packed output.  Roundtrip writes the same group-32 interval as its
    // input and never materializes packed bytes.
    if (roundtrip)
        out->outputTensorAccessPattern[0].mapping[0] = {
            0, static_cast<float>(group), 0, static_cast<float>(group - 1)};
    else
        out->outputTensorAccessPattern[0].mapping[0] = {
            0, 1, 0, static_cast<float>(codes)};
    out->outputTensorAccessPattern[0].mapping[1] = {1, 1, 0, 0};
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = width / group;
    out->indexSpaceGeometry[1] = rows;
    out->kernel.paramsNr = 0;
    unsigned char* starts[] = {&_binary___deepseek_v41_swa_pack_bf16_gaudi2_o_start,
        &_binary___deepseek_v41_fp4_pack_g16_bf16_gaudi2_o_start, &_binary___deepseek_v41_fp4_pack_g32_bf16_gaudi2_o_start,
        &_binary___deepseek_v41_fp4_roundtrip_g32_bf16_gaudi2_o_start};
    unsigned char* ends[] = {&_binary___deepseek_v41_swa_pack_bf16_gaudi2_o_end,
        &_binary___deepseek_v41_fp4_pack_g16_bf16_gaudi2_o_end, &_binary___deepseek_v41_fp4_pack_g32_bf16_gaudi2_o_end,
        &_binary___deepseek_v41_fp4_roundtrip_g32_bf16_gaudi2_o_end};
    const unsigned mode = group_ == 0 ? 0 : group_ == 16 ? 1 : group_ == 32 ? 2 : 3;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = ends[mode] - starts[mode];
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, starts[mode], out->kernel.elfSize);
    return GLUE_SUCCESS;
}
