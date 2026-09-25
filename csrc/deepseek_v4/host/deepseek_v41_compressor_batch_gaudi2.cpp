// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_compressor_batch_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_compressor_batch_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_compressor_batch_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_compressor_batch_gather_f32_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_compressor_batch_gather_f32_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41CompressorBatchGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!in || !out) return GLUE_FAILED;
    if (in->inputTensorNr != 6) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (in->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto batch = in->inputTensors[2].geometry.maxSizes[1];
    const auto history = in->inputTensors[0].geometry.maxSizes[1];
    if (!batch || batch > 64 || history < 8 || history % 8) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const int width = gather_ ? 64 : 128;
    for (unsigned i = 0; i < 4; ++i) {
        const auto& g = in->inputTensors[i].geometry;
        if (g.dataType != DATA_F32 || g.dims != 2 || g.maxSizes[0] != 512 ||
            g.maxSizes[1] != (i < 2 ? history : batch)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto& access = out->inputTensorAccessPattern[i];
        access.mapping[0] = {0, width, 0, width - 1};
        access.mapping[1] = i < 2 ? DimIndexSpaceMapping{1, 0, 0, static_cast<int>(history - 1)}
                                 : DimIndexSpaceMapping{1, 1, 0, 0};
        access.sparseAccess = i < 2;
    }
    for (unsigned i = 4; i < 6; ++i) {
        const auto& g = in->inputTensors[i].geometry;
        if (g.dataType != DATA_I32 || g.dims != 1 || g.maxSizes[0] != batch)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->inputTensorAccessPattern[i].mapping[0] = {1, 1, 0, 0};
    }
    const auto& y = in->outputTensors[0].geometry;
    if (y.dataType != (gather_ ? DATA_F32 : DATA_BF16) || y.dims != (gather_ ? 3 : 2) ||
        y.maxSizes[0] != 512 || y.maxSizes[gather_ ? 2 : 1] != batch || (gather_ && y.maxSizes[1] != 4))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->outputTensorAccessPattern[0].mapping[0] = {0, width, 0, width - 1};
    out->outputTensorAccessPattern[0].mapping[gather_ ? 2 : 1] = {1, 1, 0, 0};
    if (gather_) out->outputTensorAccessPattern[0].mapping[1] = {1, 0, 0, 3};
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = 512 / width;
    out->indexSpaceGeometry[1] = batch;
    out->kernel.paramsNr = 0;
    auto* begin = &_binary___deepseek_v41_compressor_batch_bf16_gaudi2_o_start;
    auto* end = &_binary___deepseek_v41_compressor_batch_bf16_gaudi2_o_end;
    if (gather_) {
        begin = &_binary___deepseek_v41_compressor_batch_gather_f32_gaudi2_o_start;
        end = &_binary___deepseek_v41_compressor_batch_gather_f32_gaudi2_o_end;
    }
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
