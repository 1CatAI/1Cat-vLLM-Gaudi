// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2.hpp"

#include <cstring>

extern unsigned char
    _binary___deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2_o_end;

namespace {
void map_dimension(
    tpc_lib_api::TensorAccessPattern& pattern,
    unsigned tensor_dim,
    unsigned index_dim,
    int coefficient,
    int start,
    int end) {
    pattern.mapping[tensor_dim].indexSpaceDim = index_dim;
    pattern.mapping[tensor_dim].a = coefficient;
    pattern.mapping[tensor_dim].start_b = start;
    pattern.mapping[tensor_dim].end_b = end;
}
}  // namespace

tpc_lib_api::GlueCodeReturn
DeepseekV41Mxfp4PreparedDequantFP8Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(
        name, "custom_deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV41Mxfp4PreparedDequantFP8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,
    tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 5) {
        in->inputTensorNr = 5;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 2) {
        in->outputTensorNr = 2;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const TensorDataType input_types[] = {
        DATA_I32, DATA_I16, DATA_BF16, DATA_BF16, DATA_BF16};
    for (unsigned index = 0; index < 5; ++index) {
        if (in->inputTensors[index].geometry.dataType != input_types[index]) {
            in->inputTensors[index].geometry.dataType = input_types[index];
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    if (in->outputTensors[0].geometry.dataType != DATA_F8_143) {
        in->outputTensors[0].geometry.dataType = DATA_F8_143;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (in->outputTensors[1].geometry.dataType != DATA_F32) {
        in->outputTensors[1].geometry.dataType = DATA_F32;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    auto& ids = in->inputTensors[0].geometry;
    auto& q16 = in->inputTensors[1].geometry;
    auto& s16 = in->inputTensors[2].geometry;
    auto& channel = in->inputTensors[3].geometry;
    auto& lookup = in->inputTensors[4].geometry;
    if (ids.dims != 2 || ids.maxSizes[0] != 6 || ids.maxSizes[1] != 1 ||
        q16.dims != 3 || s16.dims != 3 || channel.dims != 3 ||
        q16.maxSizes[0] == 0 || q16.maxSizes[0] % 4096 ||
        q16.maxSizes[1] == 0 || q16.maxSizes[1] > 40 ||
        q16.maxSizes[2] == 0 || q16.maxSizes[2] > 384 ||
        q16.maxSizes[0] != s16.maxSizes[0] * 8 ||
        q16.maxSizes[1] != s16.maxSizes[1] ||
        q16.maxSizes[2] != s16.maxSizes[2] ||
        channel.maxSizes[0] != 128 ||
        channel.maxSizes[1] != q16.maxSizes[1] ||
        channel.maxSizes[2] != q16.maxSizes[2] ||
        lookup.dims != 1 || lookup.maxSizes[0] != 128) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const uint64_t n = q16.maxSizes[1] * 128;
    const uint64_t k = q16.maxSizes[0] / 32;
    auto& weights = in->outputTensors[0].geometry;
    auto& selected_scale = in->outputTensors[1].geometry;
    if (weights.dims != 3 || weights.maxSizes[0] != n ||
        weights.maxSizes[1] != k || weights.maxSizes[2] != 6) {
        weights.dims = 3;
        weights.maxSizes[0] = n;
        weights.maxSizes[1] = k;
        weights.maxSizes[2] = 6;
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    if (selected_scale.dims != 3 || selected_scale.maxSizes[0] != n ||
        selected_scale.maxSizes[1] != 1 ||
        selected_scale.maxSizes[2] != 6) {
        selected_scale.dims = 3;
        selected_scale.maxSizes[0] = n;
        selected_scale.maxSizes[1] = 1;
        selected_scale.maxSizes[2] = 6;
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = q16.maxSizes[1];
    out->indexSpaceGeometry[1] = 6;
    map_dimension(out->inputTensorAccessPattern[0], 0, 1, 1, 0, 0);
    map_dimension(out->inputTensorAccessPattern[0], 1, 0, 0, 0, 0);
    map_dimension(
        out->inputTensorAccessPattern[1], 0, 0, 0, 0,
        q16.maxSizes[0] - 1);
    map_dimension(out->inputTensorAccessPattern[1], 1, 0, 1, 0, 0);
    map_dimension(
        out->inputTensorAccessPattern[1], 2, 1, 0, 0,
        q16.maxSizes[2] - 1);
    map_dimension(
        out->inputTensorAccessPattern[2], 0, 0, 0, 0,
        s16.maxSizes[0] - 1);
    map_dimension(out->inputTensorAccessPattern[2], 1, 0, 1, 0, 0);
    map_dimension(
        out->inputTensorAccessPattern[2], 2, 1, 0, 0,
        s16.maxSizes[2] - 1);
    map_dimension(out->inputTensorAccessPattern[3], 0, 0, 0, 0, 127);
    map_dimension(out->inputTensorAccessPattern[3], 1, 0, 1, 0, 0);
    map_dimension(
        out->inputTensorAccessPattern[3], 2, 1, 0, 0,
        channel.maxSizes[2] - 1);
    map_dimension(out->inputTensorAccessPattern[4], 0, 0, 0, 0, 127);

    for (unsigned index = 0; index < 2; ++index) {
        auto& pattern = out->outputTensorAccessPattern[index];
        map_dimension(pattern, 0, 0, 128, 0, 127);
        map_dimension(pattern, 1, 0, 0, 0, index == 0 ? k - 1 : 0);
        map_dimension(pattern, 2, 1, 1, 0, 0);
    }

    out->kernel.paramsNr = 0;
    const unsigned char* start =
        &_binary___deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2_o_start;
    const unsigned char* end =
        &_binary___deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) {
        return GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
