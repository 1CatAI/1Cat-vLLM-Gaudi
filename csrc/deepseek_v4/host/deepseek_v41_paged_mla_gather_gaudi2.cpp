// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_paged_mla_gather_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___deepseek_v41_paged_mla_gather_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_paged_mla_gather_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41PagedMlaGatherGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 3) { in->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    auto matches = [](const Tensor& t, unsigned type, unsigned dims, uint64_t width) {
        return t.geometry.dataType == type && t.geometry.dims == dims && t.geometry.maxSizes[0] == width;
    };
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned axis, int a, int last) {
        p.mapping[dim] = {axis, float(a), 0, float(last)};
    };
    const auto& ids = in->inputTensors[3].geometry;
    const uint64_t width = ids.maxSizes[0], tokens = ids.maxSizes[1];
    const auto swa_rows = in->inputTensors[0].geometry.maxSizes[1];
    const auto main_rows = in->inputTensors[1].geometry.maxSizes[1];
    const auto selected_rows = in->inputTensors[2].geometry.maxSizes[0];
    if (!width || width > 640 || width % 64 || tokens != 1 ||
        !matches(in->inputTensors[0], DATA_U8, 2, 528) || !swa_rows || swa_rows > 512 ||
        !matches(in->inputTensors[1], DATA_U8, 2, 288) || !main_rows || main_rows > 0x7ffffdffULL ||
        !matches(in->inputTensors[2], DATA_I32, 2, selected_rows) ||
        !selected_rows || selected_rows > 4096 || in->inputTensors[2].geometry.maxSizes[1] != 1 ||
        !matches(in->inputTensors[3], DATA_I32, 2, width) ||
        !matches(in->inputTensors[4], DATA_I32, 1, tokens)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = width;
    out->indexSpaceGeometry[1] = tokens;
    for (unsigned i = 0; i < 5; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    out->inputTensorAccessPattern[3].allRequired = false;
    map(out->inputTensorAccessPattern[3], 0, 0, 1, 0);
    map(out->inputTensorAccessPattern[3], 1, 1, 1, 0);
    for (unsigned i = 0; i < 2; ++i) {
        if (!matches(in->outputTensors[i], i == 0 ? DATA_BF16 : DATA_F32, 3, 512) ||
            in->outputTensors[i].geometry.maxSizes[1] != width ||
            in->outputTensors[i].geometry.maxSizes[2] != tokens) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        map(out->outputTensorAccessPattern[i], 0, 0, 0, 511);
        map(out->outputTensorAccessPattern[i], 1, 0, 1, 0);
        map(out->outputTensorAccessPattern[i], 2, 1, 1, 0);
    }
    if (!matches(in->outputTensors[2], DATA_F32, 2, width) ||
        in->outputTensors[2].geometry.maxSizes[1] != tokens) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    map(out->outputTensorAccessPattern[2], 0, 0, 1, 0);
    map(out->outputTensorAccessPattern[2], 1, 1, 1, 0);
    out->kernel.paramsNr = 0;
    const auto* first = &_binary___deepseek_v41_paged_mla_gather_gaudi2_o_start;
    const auto* last = &_binary___deepseek_v41_paged_mla_gather_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = last - first;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, first, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
