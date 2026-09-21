// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_dense_pair_scale_gaudi2.hpp"
#include <cstring>

extern unsigned char
    _binary___deepseek_v41_dense_pair_scale_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v41_dense_pair_scale_gaudi2_o_end;

namespace {
void map(tpc_lib_api::TensorAccessPattern& p, int dim, int index, int a,
         int first, int last) {
    p.mapping[dim].indexSpaceDim = index;
    p.mapping[dim].a = a;
    p.mapping[dim].start_b = first;
    p.mapping[dim].end_b = last;
}
}

tpc_lib_api::GlueCodeReturn
DeepseekV41DensePairScaleGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out) return GLUE_FAILED;
    if (p->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& product = p->inputTensors[0].geometry;
    const auto& weight = p->inputTensors[1].geometry;
    const auto& activation = p->inputTensors[2].geometry;
    const auto& output = p->outputTensors[0].geometry;
    const auto width = product.maxSizes[0];
    const auto rows = product.maxSizes[1];
    const auto batches = product.maxSizes[2];
    if (product.dataType != DATA_F32 || weight.dataType != DATA_F32 ||
        activation.dataType != DATA_F32 || output.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (product.dims != 3 || output.dims != 3 || width != 25600 ||
        rows < 1 || rows > 8192 || batches != 2 ||
        output.maxSizes[0] != width || output.maxSizes[1] != rows ||
        output.maxSizes[2] != batches || weight.dims != 3 ||
        weight.maxSizes[0] != width || weight.maxSizes[1] != 1 ||
        weight.maxSizes[2] != batches || activation.dims != 3 ||
        activation.maxSizes[0] != 1 || activation.maxSizes[1] != rows ||
        activation.maxSizes[2] != batches)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    out->indexSpaceRank = 3;
    out->indexSpaceGeometry[0] = width / 128;
    out->indexSpaceGeometry[1] = rows;
    out->indexSpaceGeometry[2] = batches;
    for (int tensor = 0; tensor < 3; ++tensor) {
        auto& pattern = out->inputTensorAccessPattern[tensor];
        map(pattern, 0, 0, tensor == 2 ? 0 : 128, 0,
            tensor == 2 ? 0 : 127);
        map(pattern, 1, 1, tensor == 1 ? 0 : 1, 0, 0);
        map(pattern, 2, 2, 1, 0, 0);
    }
    auto& pattern = out->outputTensorAccessPattern[0];
    map(pattern, 0, 0, 128, 0, 127);
    map(pattern, 1, 1, 1, 0, 0);
    map(pattern, 2, 2, 1, 0, 0);

    auto* begin =
        &_binary___deepseek_v41_dense_pair_scale_gaudi2_o_start;
    auto* end = &_binary___deepseek_v41_dense_pair_scale_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    out->kernel.paramsNr = 0;
    if (capacity < out->kernel.elfSize)
        return GLUE_INSUFFICIENT_ELF_BUFFER;
    if (!out->kernel.kernelElf) return GLUE_FAILED;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
