// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_mla_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_mla_gather_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mla_gather_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mla_softmax_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mla_softmax_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mla_shared_kv_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mla_shared_kv_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mla_exp_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mla_exp_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mla_normalize_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mla_normalize_bf16_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41MlaGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    const unsigned inputs = gather_ ? 6 : 4, outputs = bf16_ ? 2 : (gather_ ? 3 : 1);
    if (in->inputTensorNr != inputs) { in->inputTensorNr = inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != outputs) { in->outputTensorNr = outputs; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto matches = [](const Tensor& t, unsigned type, unsigned dims, uint64_t width) {
        return t.geometry.dataType == type && t.geometry.dims == dims && t.geometry.maxSizes[0] == width;
    };
    for (unsigned i = 0; i < inputs; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    out->indexSpaceRank = 1;
    if (gather_) {
        if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize != 2 * sizeof(int32_t))
            return GLUE_NODE_NOT_FOUND;
        const auto* p = static_cast<const int32_t*>(in->nodeParams.nodeParams);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = p[0]; out->kernel.scalarParams[1] = p[1];
        const auto width = in->inputTensors[2].geometry.maxSizes[0];
        if (!matches(in->inputTensors[0], DATA_BF16, 2, 512) ||
            !matches(in->inputTensors[1], DATA_BF16, 2, 512) ||
            !matches(in->inputTensors[2], DATA_I32, 2, width) ||
            in->inputTensors[2].geometry.maxSizes[1] != 1 || !width || width > 640 || width % 64 ||
            !matches(in->inputTensors[3], DATA_I32, 1, 1) || p[0] < 0 || p[0] % 512 || p[1] < 0 ||
            uint64_t(p[0]) + 512 > in->inputTensors[0].geometry.maxSizes[1] ||
            uint64_t(p[1]) > in->inputTensors[1].geometry.maxSizes[1]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for (unsigned i = 4; i < 6; ++i)
            if (in->inputTensors[i].geometry.dataType != DATA_I32 || in->inputTensors[i].geometry.dims != 1 ||
                !in->inputTensors[i].geometry.maxSizes[0]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const unsigned mask_index = bf16_ ? 1 : 2;
        if (!matches(in->outputTensors[0], DATA_BF16, 2, 512) ||
            !matches(in->outputTensors[mask_index], DATA_F32, 1, width) ||
            in->outputTensors[0].geometry.maxSizes[1] != width ||
            (!bf16_ && (!matches(in->outputTensors[1], DATA_F32, 2, 512) ||
                        in->outputTensors[1].geometry.maxSizes[1] != width))) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceGeometry[0] = width;
        for (unsigned i = 0; i < mask_index; ++i) {
            out->outputTensorAccessPattern[i].mapping[0] = {0, 0, 0, 511};
            out->outputTensorAccessPattern[i].mapping[1] = {0, 1, 0, 0};
        }
        out->outputTensorAccessPattern[mask_index].mapping[0] = {0, 1, 0, 0};
        out->inputTensorAccessPattern[2].allRequired = false;
        out->inputTensorAccessPattern[2].mapping[0] = {0, 1, 0, 0};
        out->inputTensorAccessPattern[2].mapping[1] = {0, 0, 0, 0};
    } else {
        const auto& g = in->inputTensors[0].geometry;
        const auto width = g.maxSizes[0], heads = g.maxSizes[1];
        if (!matches(in->inputTensors[0], DATA_F32, 2, width) || !width || width > 640 || width % 64 ||
            !heads || heads > 64 || !matches(in->inputTensors[1], DATA_F32, 1, width) ||
            !matches(in->inputTensors[2], DATA_F32, 1, heads) ||
            !matches(in->inputTensors[3], DATA_F32, 1, 1)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!matches(in->outputTensors[0], bf16_ ? DATA_BF16 : DATA_F32, 2, width) ||
            in->outputTensors[0].geometry.maxSizes[1] != heads) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (bf16_ && (!matches(in->outputTensors[1], DATA_F32, 2, 1) ||
                      in->outputTensors[1].geometry.maxSizes[1] != heads)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceGeometry[0] = heads;
        out->inputTensorAccessPattern[0].allRequired = false;
        out->inputTensorAccessPattern[0].mapping[0] = {0, 0, 0, float(width - 1)};
        out->inputTensorAccessPattern[0].mapping[1] = {0, 1, 0, 0};
        out->outputTensorAccessPattern[0] = out->inputTensorAccessPattern[0];
        if (bf16_) {
            out->outputTensorAccessPattern[1].mapping[0] = {0, 0, 0, 0};
            out->outputTensorAccessPattern[1].mapping[1] = {0, 1, 0, 0};
        }
    }
    auto* first = gather_ ? &_binary___deepseek_v41_mla_gather_gaudi2_o_start : &_binary___deepseek_v41_mla_softmax_gaudi2_o_start;
    auto* last = gather_ ? &_binary___deepseek_v41_mla_gather_gaudi2_o_end : &_binary___deepseek_v41_mla_softmax_gaudi2_o_end;
    if (bf16_) {
        first = gather_ ? &_binary___deepseek_v41_mla_shared_kv_gaudi2_o_start : &_binary___deepseek_v41_mla_exp_bf16_gaudi2_o_start;
        last = gather_ ? &_binary___deepseek_v41_mla_shared_kv_gaudi2_o_end : &_binary___deepseek_v41_mla_exp_bf16_gaudi2_o_end;
    }
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = last - first;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, first, out->kernel.elfSize);
    return GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV41MlaNormalizeGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 2 || in->outputTensorNr != 1) return GLUE_FAILED;
    const auto& x = in->inputTensors[0].geometry;
    const auto& inverse = in->inputTensors[1].geometry;
    const auto& y = in->outputTensors[0].geometry;
    const auto heads = x.maxSizes[1];
    if (x.dims != 2 || x.dataType != DATA_F32 || x.maxSizes[0] != 512 || !heads || heads > 64 ||
        inverse.dims != 2 || inverse.dataType != DATA_F32 || inverse.maxSizes[0] != 1 || inverse.maxSizes[1] != heads ||
        y.dims != 2 || y.dataType != DATA_BF16 || y.maxSizes[0] != 512 || y.maxSizes[1] != heads)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = heads;
    out->inputTensorAccessPattern[0].mapping[0] = {0, 0, 0, 511};
    out->inputTensorAccessPattern[0].mapping[1] = {0, 1, 0, 0};
    out->inputTensorAccessPattern[1].mapping[0] = {0, 0, 0, 0};
    out->inputTensorAccessPattern[1].mapping[1] = {0, 1, 0, 0};
    out->outputTensorAccessPattern[0] = out->inputTensorAccessPattern[0];
    auto* first = &_binary___deepseek_v41_mla_normalize_bf16_gaudi2_o_start;
    auto* last = &_binary___deepseek_v41_mla_normalize_bf16_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize; out->kernel.elfSize = last - first;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, first, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
