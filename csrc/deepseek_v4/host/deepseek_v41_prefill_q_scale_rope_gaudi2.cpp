// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_prefill_q_scale_rope_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_prefill_q_scale_rope_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_q_scale_rope_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_prefill_q_scale_rope_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_q_scale_rope_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41PrefillQScaleRopeGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (p->inputTensorNr != 5) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& y = p->outputTensors[0].geometry;
    if (y.dims != 2 || y.dataType != DATA_BF16 || y.maxSizes[0] != 16384 ||
        y.maxSizes[1] < 1 || y.maxSizes[1] > 8192) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    const unsigned tokens = y.maxSizes[1];
    for (unsigned i = 0; i < 5; ++i) {
        const auto& x = p->inputTensors[i].geometry;
        const auto type = i == 0 && bf16_product_ ? DATA_BF16 : (i == 3 ? DATA_I32 : DATA_F32);
        if (x.dataType != type) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (i == 3) {
            if (x.dims != 1 || x.maxSizes[0] != tokens) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            out->inputTensorAccessPattern[i].mapping[0] = {1, 1, 0, 0};
        } else if (i == 4) {
            if (x.dims != 2 || x.maxSizes[0] != 64 || x.maxSizes[1] < 1 || x.maxSizes[1] > 1048576)
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            out->inputTensorAccessPattern[i].allRequired = true;
        } else {
            if (x.dims != 2 || x.maxSizes[0] != (i == 2 ? 1u : 16384u) ||
                x.maxSizes[1] != (i == 1 ? 1u : tokens)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            out->inputTensorAccessPattern[i].mapping[0] = {0, i == 2 ? 0 : 512, 0, i == 2 ? 0 : 511};
            out->inputTensorAccessPattern[i].mapping[1] = {1, i == 1 ? 0 : 1, 0, 0};
        }
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = 32;
    out->indexSpaceGeometry[1] = tokens;
    out->outputTensorAccessPattern[0].mapping[0] = {0, 512, 0, 511};
    out->outputTensorAccessPattern[0].mapping[1] = {1, 1, 0, 0};
    const auto* begin = bf16_product_ ? &_binary___deepseek_v41_prefill_q_scale_rope_bf16_gaudi2_o_start :
                                      &_binary___deepseek_v41_prefill_q_scale_rope_gaudi2_o_start;
    const auto* end = bf16_product_ ? &_binary___deepseek_v41_prefill_q_scale_rope_bf16_gaudi2_o_end :
                                    &_binary___deepseek_v41_prefill_q_scale_rope_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    out->kernel.paramsNr = 0;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
