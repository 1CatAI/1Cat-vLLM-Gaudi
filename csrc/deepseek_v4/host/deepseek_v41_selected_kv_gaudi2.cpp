// SPDX-License-Identifier: Apache-2.0
// Reuses the selected-row decoder from the V4.1 C1 path. The cache may be
// paged and large; only the explicitly selected rows are emitted.
#include "deepseek_v41_selected_kv_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___deepseek_v41_selected_kv_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_kv_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_selected_kv_vector_scales_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_kv_vector_scales_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41SelectedKVGaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, vector_scales_ ? "custom_deepseek_v41_selected_kv_vector_scales_bf16_gaudi2"
                                    : "custom_deepseek_v41_selected_kv_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV41SelectedKVGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 3) {
        in->inputTensorNr = 3;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 2) {
        in->outputTensorNr = 2;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const auto slots = in->inputTensors[2].geometry.maxSizes[0];
    const TensorDataType types[] = {DATA_U8, DATA_U8, DATA_I32, DATA_BF16, DATA_I32};
    for (unsigned i = 0; i < 5; ++i) {
        auto& g = i < 3 ? in->inputTensors[i].geometry : in->outputTensors[i - 3].geometry;
        if (g.dataType != types[i]) {
            g.dataType = types[i];
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        bool size = g.dims == 2;
        if (i == 0) size &= g.maxSizes[0] == 528 && g.maxSizes[1] > 0 && g.maxSizes[1] <= 512;
        if (i == 1) size &= g.maxSizes[0] == 288 && g.maxSizes[1] > 0 &&
                           g.maxSizes[1] <= 0x7fffffffULL - 512;
        if (i == 2 || i == 4) size &= slots > 0 && slots <= 4096 &&
                                     g.maxSizes[0] == slots && g.maxSizes[1] == 1;
        if (i == 3) size &= g.maxSizes[0] == 512 && g.maxSizes[1] == slots;
        if (!size) return i < 3 ? GLUE_INCOMPATIBLE_INPUT_SIZE : GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        auto& access = i < 3 ? out->inputTensorAccessPattern[i] : out->outputTensorAccessPattern[i - 3];
        if (i < 2) {
            // Runtime IDs address arbitrary cache rows, so an affine row slice
            // would be incorrect.  Describe the complete tensor range while
            // marking it sparse: this keeps the persistent cache at its base
            // address for indirect TPC loads, but lets GC avoid making a
            // per-invocation allRequired materialisation copy.  The old
            // allRequired bit was responsible for repeatedly moving the
            // packed uint8 cache through an internal DMA buffer before the
            // selected decoder could consume it.
            std::memset(&access, 0, sizeof(access));
            access.sparseAccess = true;
            access.mapping[0].indexSpaceDim = 0;
            access.mapping[0].a = 0;
            access.mapping[0].start_b = 0;
            access.mapping[0].end_b = static_cast<float>(g.maxSizes[0] - 1);
            access.mapping[1].indexSpaceDim = 0;
            access.mapping[1].a = 0;
            access.mapping[1].start_b = 0;
            access.mapping[1].end_b = static_cast<float>(g.maxSizes[1] - 1);
        } else {
            for (unsigned d = 0; d < 2; ++d) {
                auto& map = access.mapping[d];
                map.indexSpaceDim = 0;
                map.a = (i == 3 ? d == 1 : d == 0) ? 1 : 0;
                map.start_b = 0;
                map.end_b = i == 3 && d == 0 ? 511 : 0;
            }
        }
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = slots;
    out->kernel.paramsNr = 0;
    auto* start = vector_scales_ ? &_binary___deepseek_v41_selected_kv_vector_scales_bf16_gaudi2_o_start
                                : &_binary___deepseek_v41_selected_kv_bf16_gaudi2_o_start;
    auto* end = vector_scales_ ? &_binary___deepseek_v41_selected_kv_vector_scales_bf16_gaudi2_o_end
                              : &_binary___deepseek_v41_selected_kv_bf16_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
