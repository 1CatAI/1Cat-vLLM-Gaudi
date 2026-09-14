// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_swa_pack_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_swa_pack_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_swa_pack_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_swa_pack_write_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_swa_pack_write_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41SwaPackGaudi2::GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, write_ ? "custom_deepseek_v41_swa_pack_write_bf16_gaudi2"
                            : "custom_deepseek_v41_swa_pack_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn DeepseekV41SwaPackGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    const unsigned inputs = write_ ? 3 : 1;
    if (in->inputTensorNr != inputs) { in->inputTensorNr = inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& value = in->inputTensors[write_ ? 1 : 0].geometry;
    const auto width = value.maxSizes[0], rows = value.maxSizes[1];
    if (value.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (value.dims != 2 || width == 0 || width % 32 || width > 16384 || rows == 0 || rows > 8192 ||
        (write_ && (rows != 1 || width != 512))) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    auto& input_access = out->inputTensorAccessPattern[write_ ? 1 : 0];
    input_access.mapping[0] = {0, 32, 0, 31};
    input_access.mapping[1] = {1, 1, 0, 0};
    const auto& result = in->outputTensors[0].geometry;
    if (write_) {
        const auto& cache = in->inputTensors[0].geometry;
        const auto& position = in->inputTensors[2].geometry;
        if (cache.dataType != DATA_U8 || position.dataType != DATA_I32 || result.dataType != DATA_I32)
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (cache.dims != 2 || cache.maxSizes[0] != 528 || cache.maxSizes[1] == 0 || cache.maxSizes[1] > 512 ||
            position.dims != 1 || position.maxSizes[0] != 1) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (result.dims != 1 || result.maxSizes[0] != 16) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->inputTensorAccessPattern[0].allRequired = true;
        out->inputTensorAccessPattern[2].allRequired = true;
        out->outputTensorAccessPattern[0].mapping[0] = {0, 1, 0, 0};
    } else {
        if (result.dataType != DATA_U8) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (result.dims != 2 || result.maxSizes[0] != width + width / 32 || result.maxSizes[1] != rows)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        // Codes and trailing scales have different affine offsets. The union
        // mapping is conservative; every group still writes distinct bytes.
        out->outputTensorAccessPattern[0].mapping[0] = {0, 1, 0, static_cast<float>(width)};
        out->outputTensorAccessPattern[0].mapping[1] = {1, 1, 0, 0};
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = width / 32;
    out->indexSpaceGeometry[1] = rows;
    out->kernel.paramsNr = 0;
    auto* start = write_ ? &_binary___deepseek_v41_swa_pack_write_bf16_gaudi2_o_start
                         : &_binary___deepseek_v41_swa_pack_bf16_gaudi2_o_start;
    auto* end = write_ ? &_binary___deepseek_v41_swa_pack_write_bf16_gaudi2_o_end
                       : &_binary___deepseek_v41_swa_pack_bf16_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
