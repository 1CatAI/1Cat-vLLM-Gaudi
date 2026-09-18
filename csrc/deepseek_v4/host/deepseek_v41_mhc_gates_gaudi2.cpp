// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_mhc_gates_gaudi2.hpp"
#include <cstring>
#include <initializer_list>

extern unsigned char _binary___deepseek_v41_mhc_gates_f32_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mhc_gates_f32_gaudi2_o_end;

namespace {
using namespace tpc_lib_api;

bool matches(const Tensor& tensor, unsigned dims, std::initializer_list<unsigned> sizes) {
    if (tensor.geometry.dataType != DATA_F32 || tensor.geometry.dims != dims) return false;
    unsigned dim = 0;
    for (unsigned size : sizes) {
        if (size && tensor.geometry.maxSizes[dim] != size) return false;
        ++dim;
    }
    return true;
}

void map(TensorAccessPattern& access, unsigned dim, bool dynamic, int start, int end) {
    access.mapping[dim].indexSpaceDim = 0;
    access.mapping[dim].a = dynamic ? 1 : 0;
    access.mapping[dim].start_b = start;
    access.mapping[dim].end_b = end;
}
}

tpc_lib_api::GlueCodeReturn DeepseekV41MhcGatesGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 4) {
        in->inputTensorNr = 4;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1) {
        in->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const unsigned tokens = in->inputTensors[0].geometry.maxSizes[1];
    if (!matches(in->inputTensors[0], 2, {24, tokens}) || tokens < 1 || tokens > 8192 ||
        !matches(in->inputTensors[1], 2, {1, tokens}) ||
        !matches(in->inputTensors[2], 1, {3}) || !matches(in->inputTensors[3], 1, {24})) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (!matches(in->outputTensors[0], 2, {24, tokens})) {
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    for (unsigned i = 0; i < 4; ++i) {
        if (in->inputTensors[i].geometry.dataType != DATA_F32) return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (in->outputTensors[0].geometry.dataType != DATA_F32) return GLUE_INCOMPATIBLE_DATA_TYPE;

    map(out->inputTensorAccessPattern[0], 0, false, 0, 23);
    map(out->inputTensorAccessPattern[0], 1, true, 0, 0);
    map(out->inputTensorAccessPattern[1], 0, false, 0, 0);
    map(out->inputTensorAccessPattern[1], 1, true, 0, 0);
    map(out->inputTensorAccessPattern[2], 0, false, 0, 2);
    map(out->inputTensorAccessPattern[3], 0, false, 0, 23);
    map(out->outputTensorAccessPattern[0], 0, false, 0, 23);
    map(out->outputTensorAccessPattern[0], 1, true, 0, 0);

    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = tokens;
    out->kernel.paramsNr = 0;
    auto* begin = &_binary___deepseek_v41_mhc_gates_f32_gaudi2_o_start;
    auto* end = &_binary___deepseek_v41_mhc_gates_f32_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
