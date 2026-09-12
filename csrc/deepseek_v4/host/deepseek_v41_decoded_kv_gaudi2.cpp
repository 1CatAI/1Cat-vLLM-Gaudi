// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_decoded_kv_gaudi2.hpp"
#include <cstring>
#include <initializer_list>
extern unsigned char _binary___deepseek_v41_swa_decoded_write_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_swa_decoded_write_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_fp4_decoded_write_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_fp4_decoded_write_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_decoded_attn_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_decoded_attn_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41DecodedKVGaudi2::GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, mode_ == SWA_WRITE ? "custom_deepseek_v41_swa_decoded_write_bf16_gaudi2" :
                      mode_ == FP4_WRITE ? "custom_deepseek_v41_fp4_decoded_write_bf16_gaudi2" :
                                           "custom_deepseek_v41_decoded_attn_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn DeepseekV41DecodedKVGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    const unsigned inputs = mode_ == SWA_WRITE ? 4 : mode_ == FP4_WRITE ? 6 : 9;
    const unsigned outputs = mode_ == ATTENTION ? 3 : 1;
    if (in->inputTensorNr != inputs) { in->inputTensorNr = inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != outputs) { in->outputTensorNr = outputs; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const unsigned params = mode_ == SWA_WRITE ? 1 : mode_ == ATTENTION ? 2 : 0;
    if (params && (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize != params * sizeof(int32_t)))
        return GLUE_NODE_NOT_FOUND;
    const auto* scalar = static_cast<const int32_t*>(in->nodeParams.nodeParams);
    out->kernel.paramsNr = params;
    for (unsigned i = 0; i < params; ++i) out->kernel.scalarParams[i] = scalar[i];
    for (unsigned i = 0; i < inputs; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    const auto shape = [&](unsigned i, unsigned dtype, unsigned dims, uint64_t width) {
        const auto& g = in->inputTensors[i].geometry;
        return g.dataType == dtype && g.dims == dims && g.maxSizes[0] == width &&
               (dims < 2 || g.maxSizes[1] > 0);
    };
    const auto result = [&](unsigned i, unsigned dtype, unsigned dims, uint64_t width) {
        const auto& g = in->outputTensors[i].geometry;
        return g.dataType == dtype && g.dims == dims && g.maxSizes[0] == width;
    };
    if (mode_ == SWA_WRITE) {
        if (!shape(0, DATA_U8, 2, 528) || !shape(1, DATA_BF16, 2, 512) ||
            !shape(2, DATA_I32, 1, 1) || !shape(3, DATA_BF16, 2, 512) ||
            in->inputTensors[0].geometry.maxSizes[1] != 512 ||
            in->inputTensors[1].geometry.maxSizes[1] != 1 || scalar[0] < 0 ||
            scalar[0] % 512 || uint64_t(scalar[0]) + 512 > in->inputTensors[3].geometry.maxSizes[1])
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!result(0, DATA_I32, 1, 16)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 16;
        out->inputTensorAccessPattern[1].allRequired = false;
        out->inputTensorAccessPattern[1].mapping[0] = {0, 32, 0, 31};
        out->inputTensorAccessPattern[1].mapping[1] = {0, 0, 0, 0};
        out->outputTensorAccessPattern[0].mapping[0] = {0, 1, 0, 0};
    } else if (mode_ == FP4_WRITE) {
        if (!shape(0, DATA_U8, 2, 288) || !shape(1, DATA_U8, 2, 68) ||
            !shape(2, DATA_BF16, 2, 512) || !shape(3, DATA_BF16, 2, 128) ||
            !shape(4, DATA_I32, 1, 1) || !shape(5, DATA_BF16, 2, 512) ||
            in->inputTensors[0].geometry.maxSizes[1] != in->inputTensors[1].geometry.maxSizes[1] ||
            in->inputTensors[0].geometry.maxSizes[1] != in->inputTensors[5].geometry.maxSizes[1] ||
            in->inputTensors[2].geometry.maxSizes[1] != 1 || in->inputTensors[3].geometry.maxSizes[1] != 1)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!result(0, DATA_I32, 1, 36)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 36;
        out->outputTensorAccessPattern[0].mapping[0] = {0, 1, 0, 0};
    } else {
        const auto& q = in->inputTensors[0].geometry;
        if (!shape(0, DATA_BF16, 3, 512) || !shape(1, DATA_BF16, 2, 512) ||
            !shape(2, DATA_BF16, 2, 512) || q.maxSizes[1] > 64 || q.maxSizes[2] == 0 ||
            !shape(4, DATA_F32, 1, q.maxSizes[1]) || !shape(5, DATA_F32, 1, 1) ||
            !shape(6, DATA_I32, 1, q.maxSizes[2]) ||
            in->inputTensors[3].geometry.dataType != DATA_I32 || in->inputTensors[3].geometry.dims != 2 ||
            in->inputTensors[3].geometry.maxSizes[0] == 0 ||
            in->inputTensors[3].geometry.maxSizes[1] != q.maxSizes[2] ||
            scalar[0] < 0 || scalar[0] % 512 || scalar[1] < 0 ||
            uint64_t(scalar[0]) + 512 > in->inputTensors[1].geometry.maxSizes[1] ||
            uint64_t(scalar[1]) > in->inputTensors[2].geometry.maxSizes[1])
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for (unsigned i : {7u, 8u})
            if (in->inputTensors[i].geometry.dataType != DATA_I32 ||
                in->inputTensors[i].geometry.dims != 1 || in->inputTensors[i].geometry.maxSizes[0] == 0)
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!result(0, DATA_BF16, 3, 512) || in->outputTensors[0].geometry.maxSizes[1] != q.maxSizes[1] ||
            in->outputTensors[0].geometry.maxSizes[2] != q.maxSizes[2]) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 3;
        out->indexSpaceGeometry[0] = 1; out->indexSpaceGeometry[1] = q.maxSizes[1];
        out->indexSpaceGeometry[2] = q.maxSizes[2];
        out->inputTensorAccessPattern[0].allRequired = false;
        out->inputTensorAccessPattern[0].mapping[0] = {0, 512, 0, 511};
        out->inputTensorAccessPattern[0].mapping[1] = {1, 1, 0, 0};
        out->inputTensorAccessPattern[0].mapping[2] = {2, 1, 0, 0};
        out->outputTensorAccessPattern[0] = out->inputTensorAccessPattern[0];
        for (unsigned i : {1u, 2u}) {
            if (!result(i, DATA_F32, 2, q.maxSizes[1]) ||
                in->outputTensors[i].geometry.maxSizes[1] != q.maxSizes[2]) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            out->outputTensorAccessPattern[i].mapping[0] = {1, 1, 0, 0};
            out->outputTensorAccessPattern[i].mapping[1] = {2, 1, 0, 0};
        }
    }
    auto* start = mode_ == SWA_WRITE ? &_binary___deepseek_v41_swa_decoded_write_bf16_gaudi2_o_start :
                  mode_ == FP4_WRITE ? &_binary___deepseek_v41_fp4_decoded_write_bf16_gaudi2_o_start :
                                       &_binary___deepseek_v41_decoded_attn_bf16_gaudi2_o_start;
    auto* end = mode_ == SWA_WRITE ? &_binary___deepseek_v41_swa_decoded_write_bf16_gaudi2_o_end :
                mode_ == FP4_WRITE ? &_binary___deepseek_v41_fp4_decoded_write_bf16_gaudi2_o_end :
                                     &_binary___deepseek_v41_decoded_attn_bf16_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
