// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_mxfp4_column_gaudi2.hpp"
#include <cstring>
#include <initializer_list>

extern unsigned char _binary___deepseek_v41_mxfp4_column_dequant_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_column_dequant_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mxfp4_column_dequant_normal_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_column_dequant_normal_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41Mxfp4ColumnGaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, normal_ ? "custom_deepseek_v41_mxfp4_column_dequant_normal_bf16_gaudi2"
                              : "custom_deepseek_v41_mxfp4_column_dequant_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV41Mxfp4ColumnGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 4) { in->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const TensorDataType types[] = {DATA_I32, DATA_I16, DATA_BF16, DATA_BF16};
    for (unsigned i = 0; i < 4; ++i)
        if (in->inputTensors[i].geometry.dataType != types[i]) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (in->outputTensors[0].geometry.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto& ids = in->inputTensors[0].geometry;
    const auto& q = in->inputTensors[1].geometry;
    const auto& s = in->inputTensors[2].geometry;
    const auto& lut = in->inputTensors[3].geometry;
    const auto& y = in->outputTensors[0].geometry;
    if (ids.dims != 2 || ids.maxSizes[1] != 1 || !ids.maxSizes[0] || ids.maxSizes[0] > 36 ||
        q.dims != 3 || q.maxSizes[0] != 163840 || q.maxSizes[1] != 18 || !q.maxSizes[2] || q.maxSizes[2] > 384 ||
        s.dims != 3 || s.maxSizes[0] * 8 != q.maxSizes[0] || s.maxSizes[1] != 18 || s.maxSizes[2] != q.maxSizes[2] ||
        lut.dims != 1 || lut.maxSizes[0] != 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t columns = ((ids.maxSizes[0] * 2304 + 1023) / 1024) * 1024;
    if (y.dims != 2 || y.maxSizes[0] != columns || y.maxSizes[1] != 5120)
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned index, int a, int begin, int end) {
        p.mapping[dim] = {index, a, begin, end};
    };
    out->indexSpaceRank = 3;
    out->indexSpaceGeometry[0] = columns / 1024;
    out->indexSpaceGeometry[1] = 1;
    out->indexSpaceGeometry[2] = 40;
    out->inputTensorAccessPattern[0].allRequired = true;
    for (unsigned i : {1U, 2U}) {
        auto& p = out->inputTensorAccessPattern[i];
        const int width = i == 1 ? 4096 : 512;
        map(p, 0, 2, width, 0, width - 1);
        // A column tile can cross expert boundaries. Both source dimensions
        // are indirect; describing a fictitious affine row map is unsafe.
        map(p, 1, 0, 0, 0, 17);
        map(p, 2, 0, 0, 0, q.maxSizes[2] - 1);
        p.sparseAccess = true;
    }
    out->inputTensorAccessPattern[3].allRequired = true;
    map(out->outputTensorAccessPattern[0], 0, 0, 1024, 0, 1023);
    map(out->outputTensorAccessPattern[0], 1, 2, 128, 0, 127);
    out->kernel.paramsNr = 0;
    const auto* start = normal_ ? &_binary___deepseek_v41_mxfp4_column_dequant_normal_bf16_gaudi2_o_start
                                : &_binary___deepseek_v41_mxfp4_column_dequant_bf16_gaudi2_o_start;
    const auto* end = normal_ ? &_binary___deepseek_v41_mxfp4_column_dequant_normal_bf16_gaudi2_o_end
                              : &_binary___deepseek_v41_mxfp4_column_dequant_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
