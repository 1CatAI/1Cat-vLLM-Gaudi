// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_selected_mla_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___deepseek_v41_selected_mla_gather_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_mla_gather_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_selected_mla_softmax_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_mla_softmax_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41SelectedMlaGaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, gather_ ? "custom_deepseek_v41_selected_mla_gather_gaudi2"
                             : "custom_deepseek_v41_selected_mla_softmax_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn DeepseekV41SelectedMlaGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    const unsigned inputs = gather_ ? 3 : 4, outputs = gather_ ? 3 : 1;
    if (in->inputTensorNr != inputs) { in->inputTensorNr = inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != outputs) { in->outputTensorNr = outputs; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    auto matches = [](const Tensor& t, unsigned type, unsigned dims, uint64_t width) {
        return t.geometry.dataType == type && t.geometry.dims == dims && t.geometry.maxSizes[0] == width;
    };
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned axis, int a, int last) {
        p.mapping[dim] = {axis, float(a), 0, float(last)};
    };
    const auto& input = in->inputTensors[gather_ ? 1 : 0].geometry;
    const uint64_t width = input.maxSizes[0], tokens = input.maxSizes[gather_ ? 1 : 2];
    if (!width || width > 640 || width % 64 || !tokens || tokens > 6) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[1] = tokens;
    for (unsigned i = 0; i < inputs; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    if (gather_) {
        if (!matches(in->inputTensors[0], DATA_BF16, 2, 512) ||
            !in->inputTensors[0].geometry.maxSizes[1] || in->inputTensors[0].geometry.maxSizes[1] > 4096 ||
            !matches(in->inputTensors[1], DATA_I32, 2, width) ||
            !matches(in->inputTensors[2], DATA_I32, 1, tokens)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceGeometry[0] = width;
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
        out->inputTensorAccessPattern[1].allRequired = false;
        map(out->inputTensorAccessPattern[1], 0, 0, 1, 0);
        map(out->inputTensorAccessPattern[1], 1, 1, 1, 0);
    } else {
        const auto heads = input.maxSizes[1];
        if (!heads || heads > 64 || !matches(in->inputTensors[0], DATA_F32, 3, width) ||
            !matches(in->inputTensors[1], DATA_F32, 2, width) ||
            in->inputTensors[1].geometry.maxSizes[1] != tokens ||
            !matches(in->inputTensors[2], DATA_F32, 1, heads) ||
            !matches(in->inputTensors[3], DATA_F32, 1, 1)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!matches(in->outputTensors[0], DATA_F32, 3, width) ||
            in->outputTensors[0].geometry.maxSizes[1] != heads ||
            in->outputTensors[0].geometry.maxSizes[2] != tokens) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceGeometry[0] = heads;
        out->inputTensorAccessPattern[0].allRequired = false;
        map(out->inputTensorAccessPattern[0], 0, 0, 0, width - 1);
        map(out->inputTensorAccessPattern[0], 1, 0, 1, 0);
        map(out->inputTensorAccessPattern[0], 2, 1, 1, 0);
        out->outputTensorAccessPattern[0] = out->inputTensorAccessPattern[0];
    }
    out->kernel.paramsNr = 0;
    auto* first = gather_ ? &_binary___deepseek_v41_selected_mla_gather_gaudi2_o_start
                          : &_binary___deepseek_v41_selected_mla_softmax_gaudi2_o_start;
    auto* last = gather_ ? &_binary___deepseek_v41_selected_mla_gather_gaudi2_o_end
                         : &_binary___deepseek_v41_selected_mla_softmax_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = last - first;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, first, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
