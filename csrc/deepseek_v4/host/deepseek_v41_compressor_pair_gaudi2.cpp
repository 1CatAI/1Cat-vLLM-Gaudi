// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_compressor_pair_gaudi2.hpp"
#include <cstring>

extern unsigned char
    _binary___deepseek_v41_compressor_pair_bf16_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v41_compressor_pair_bf16_gaudi2_o_end;

namespace {
using namespace tpc_lib_api;

bool tensor(const Tensor& value, unsigned type, unsigned rows) {
    return value.geometry.dataType == type && value.geometry.dims == 2 &&
           value.geometry.maxSizes[0] == 512 &&
           value.geometry.maxSizes[1] == rows;
}

void feature_rows(TensorAccessPattern& access, unsigned rows) {
    access.allRequired = false;
    access.mapping[0] = {0, 128, 0, 127};
    access.mapping[1] = {0, 0, 0, static_cast<int>(rows - 1)};
}
}

tpc_lib_api::GlueCodeReturn DeepseekV41CompressorPairGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,
    tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 5) {
        in->inputTensorNr = 5;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1) {
        in->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (!tensor(in->inputTensors[0], DATA_F32, 8) ||
        !tensor(in->inputTensors[1], DATA_F32, 8) ||
        !tensor(in->inputTensors[2], DATA_F32, 1) ||
        !tensor(in->inputTensors[3], DATA_F32, 1) ||
        in->inputTensors[4].geometry.dataType != DATA_I32 ||
        in->inputTensors[4].geometry.dims != 1 ||
        in->inputTensors[4].geometry.maxSizes[0] != 1)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!tensor(in->outputTensors[0], DATA_BF16, 1))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;

    feature_rows(out->inputTensorAccessPattern[0], 8);
    feature_rows(out->inputTensorAccessPattern[1], 8);
    feature_rows(out->inputTensorAccessPattern[2], 1);
    feature_rows(out->inputTensorAccessPattern[3], 1);
    out->inputTensorAccessPattern[4].allRequired = true;
    feature_rows(out->outputTensorAccessPattern[0], 1);
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = 4;
    out->kernel.paramsNr = 0;

    auto* begin =
        &_binary___deepseek_v41_compressor_pair_bf16_gaudi2_o_start;
    auto* end =
        &_binary___deepseek_v41_compressor_pair_bf16_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize)
        return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
