// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_main_reuse_gather_gaudi2.hpp"
#include <cstring>
#include <initializer_list>
extern unsigned char _binary___deepseek_v41_main_reuse_gather_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_main_reuse_gather_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41MainReuseGatherGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 3) { in->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto tokens = in->inputTensors[3].geometry.maxSizes[0];
    if (tokens != 1) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    auto matches = [](const Tensor& t, unsigned type, unsigned dims, uint64_t d0) {
        return t.geometry.dataType == type && t.geometry.dims == dims && t.geometry.maxSizes[0] == d0;
    };
    if (!matches(in->inputTensors[0], DATA_U8, 2, 528) || in->inputTensors[0].geometry.maxSizes[1] != 256 ||
        !matches(in->inputTensors[1], DATA_BF16, 3, 512) || in->inputTensors[1].geometry.maxSizes[1] != 640 ||
        in->inputTensors[1].geometry.maxSizes[2] != tokens ||
        !matches(in->inputTensors[2], DATA_F32, 2, 640) || in->inputTensors[2].geometry.maxSizes[1] != tokens ||
        !matches(in->inputTensors[3], DATA_I32, 1, tokens) ||
        !matches(in->inputTensors[4], DATA_I32, 1, tokens)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned axis, int a, int last) {
        p.mapping[dim] = {axis, float(a), 0, float(last)};
    };
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = 640;
    out->indexSpaceGeometry[1] = tokens;
    auto& swa = out->inputTensorAccessPattern[0];
    std::memset(&swa, 0, sizeof(swa));
    swa.sparseAccess = true;
    map(swa, 0, 0, 0, 527);
    map(swa, 1, 0, 0, 255);
    map(out->inputTensorAccessPattern[1], 0, 0, 0, 511);
    map(out->inputTensorAccessPattern[1], 1, 0, 1, 0);
    map(out->inputTensorAccessPattern[1], 2, 1, 1, 0);
    map(out->inputTensorAccessPattern[2], 0, 0, 1, 0);
    map(out->inputTensorAccessPattern[2], 1, 1, 1, 0);
    map(out->inputTensorAccessPattern[3], 0, 1, 1, 0);
    map(out->inputTensorAccessPattern[4], 0, 1, 1, 0);
    for (unsigned i = 0; i < 2; ++i) {
        const auto& g = in->outputTensors[i].geometry;
        if (!matches(in->outputTensors[i], i == 0 ? DATA_BF16 : DATA_F32, 3, 512) ||
            g.maxSizes[1] != 640 || g.maxSizes[2] != tokens) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        map(out->outputTensorAccessPattern[i], 0, 0, 0, 511);
        map(out->outputTensorAccessPattern[i], 1, 0, 1, 0);
        map(out->outputTensorAccessPattern[i], 2, 1, 1, 0);
    }
    if (!matches(in->outputTensors[2], DATA_F32, 2, 640) ||
        in->outputTensors[2].geometry.maxSizes[1] != tokens) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    map(out->outputTensorAccessPattern[2], 0, 0, 1, 0);
    map(out->outputTensorAccessPattern[2], 1, 1, 1, 0);
    out->kernel.paramsNr = 0;
    auto* first = &_binary___deepseek_v41_main_reuse_gather_gaudi2_o_start;
    auto* last = &_binary___deepseek_v41_main_reuse_gather_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = last - first;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, first, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
