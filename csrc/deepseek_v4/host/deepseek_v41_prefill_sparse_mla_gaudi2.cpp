// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_prefill_sparse_mla_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___deepseek_v41_prefill_sparse_kv_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_sparse_kv_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_prefill_sparse_exp_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_sparse_exp_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_prefill_sparse_normalize_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_sparse_normalize_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41PrefillSparseMlaGaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    static constexpr const char* names[] = {
        "custom_deepseek_v41_prefill_sparse_kv_bf16_gaudi2",
        "custom_deepseek_v41_prefill_sparse_exp_bf16_gaudi2",
        "custom_deepseek_v41_prefill_sparse_normalize_bf16_gaudi2",
    };
    std::strcpy(name, names[kind_]);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV41PrefillSparseMlaGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    const unsigned input_count = kind_ == Gather ? 3 : (kind_ == Exp ? 4 : 2);
    const unsigned output_count = kind_ == Normalize ? 1 : 2;
    if (p->inputTensorNr != input_count) { p->inputTensorNr = input_count; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (p->outputTensorNr != output_count) { p->outputTensorNr = output_count; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& x = p->inputTensors[kind_ == Gather ? 1 : 0].geometry;
    const unsigned tokens = x.maxSizes[kind_ == Gather ? 1 : 2];
    if (!tokens || tokens > 32) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[1] = tokens;
    auto map = [](TensorAccessPattern& a, unsigned dim, unsigned axis, int size) {
        a.mapping[dim] = {axis, float(size), 0, float(size - 1)};
    };
    auto constant = [](TensorAccessPattern& a, unsigned dim, int extent) {
        a.mapping[dim] = {0, 0, 0, float(extent - 1)};
    };
    for (unsigned i = 0; i < input_count; ++i) out->inputTensorAccessPattern[i].allRequired = false;
    for (unsigned i = 0; i < output_count; ++i) out->outputTensorAccessPattern[i].allRequired = false;
    if (kind_ == Gather) {
        const auto& cache = p->inputTensors[0].geometry;
        const auto& lengths = p->inputTensors[2].geometry;
        const unsigned width = x.maxSizes[0];
        if (x.dims != 2 || x.dataType != DATA_I32 || !width || width > 640 || width % 64 ||
            cache.dims != 2 || cache.dataType != DATA_BF16 || cache.maxSizes[0] != 512 ||
            !cache.maxSizes[1] || cache.maxSizes[1] > 131072 ||
            lengths.dims != 1 || lengths.dataType != DATA_I32 || lengths.maxSizes[0] != tokens)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto& kv = p->outputTensors[0].geometry;
        const auto& mask = p->outputTensors[1].geometry;
        if (kv.dims != 3 || kv.dataType != DATA_BF16 || kv.maxSizes[0] != 512 ||
            kv.maxSizes[1] != width || kv.maxSizes[2] != tokens ||
            mask.dims != 2 || mask.dataType != DATA_F32 ||
            mask.maxSizes[0] != width || mask.maxSizes[1] != tokens)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceGeometry[0] = width;
        out->inputTensorAccessPattern[0].allRequired = true;
        out->inputTensorAccessPattern[0].sparseAccess = true;
        map(out->inputTensorAccessPattern[1], 0, 0, 1);
        map(out->inputTensorAccessPattern[1], 1, 1, 1);
        map(out->inputTensorAccessPattern[2], 0, 1, 1);
        constant(out->outputTensorAccessPattern[0], 0, 512);
        map(out->outputTensorAccessPattern[0], 1, 0, 1);
        map(out->outputTensorAccessPattern[0], 2, 1, 1);
        map(out->outputTensorAccessPattern[1], 0, 0, 1);
        map(out->outputTensorAccessPattern[1], 1, 1, 1);
    } else if (kind_ == Exp) {
        const unsigned width = x.maxSizes[0], heads = x.maxSizes[1];
        if (x.dims != 3 || x.dataType != DATA_F32 || !width || width > 640 || width % 64 ||
            !heads || heads > 64 ||
            p->inputTensors[1].geometry.dims != 2 || p->inputTensors[1].geometry.dataType != DATA_F32 ||
            p->inputTensors[1].geometry.maxSizes[0] != width ||
            p->inputTensors[1].geometry.maxSizes[1] != tokens ||
            p->inputTensors[2].geometry.dims != 1 || p->inputTensors[2].geometry.dataType != DATA_F32 ||
            p->inputTensors[2].geometry.maxSizes[0] != heads ||
            p->inputTensors[3].geometry.dims != 1 || p->inputTensors[3].geometry.dataType != DATA_F32 ||
            p->inputTensors[3].geometry.maxSizes[0] != 1)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto& prob = p->outputTensors[0].geometry;
        const auto& inverse = p->outputTensors[1].geometry;
        if (prob.dims != 3 || prob.dataType != DATA_BF16 || prob.maxSizes[0] != width ||
            prob.maxSizes[1] != heads || prob.maxSizes[2] != tokens ||
            inverse.dims != 3 || inverse.dataType != DATA_F32 || inverse.maxSizes[0] != 1 ||
            inverse.maxSizes[1] != heads || inverse.maxSizes[2] != tokens)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceGeometry[0] = heads;
        constant(out->inputTensorAccessPattern[0], 0, width);
        map(out->inputTensorAccessPattern[0], 1, 0, 1);
        map(out->inputTensorAccessPattern[0], 2, 1, 1);
        constant(out->inputTensorAccessPattern[1], 0, width);
        map(out->inputTensorAccessPattern[1], 1, 1, 1);
        constant(out->inputTensorAccessPattern[2], 0, 1);
        out->inputTensorAccessPattern[3].allRequired = true;
        out->outputTensorAccessPattern[0] = out->inputTensorAccessPattern[0];
        constant(out->outputTensorAccessPattern[1], 0, 1);
        map(out->outputTensorAccessPattern[1], 1, 0, 1);
        map(out->outputTensorAccessPattern[1], 2, 1, 1);
    } else {
        const unsigned heads = x.maxSizes[1];
        if (x.dims != 3 || x.dataType != DATA_F32 || x.maxSizes[0] != 512 || !heads || heads > 64 ||
            p->inputTensors[1].geometry.dims != 3 || p->inputTensors[1].geometry.dataType != DATA_F32 ||
            p->inputTensors[1].geometry.maxSizes[0] != 1 ||
            p->inputTensors[1].geometry.maxSizes[1] != heads ||
            p->inputTensors[1].geometry.maxSizes[2] != tokens)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto& y = p->outputTensors[0].geometry;
        if (y.dims != 3 || y.dataType != DATA_BF16 || y.maxSizes[0] != 512 ||
            y.maxSizes[1] != heads || y.maxSizes[2] != tokens)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceGeometry[0] = heads;
        constant(out->inputTensorAccessPattern[0], 0, 512);
        map(out->inputTensorAccessPattern[0], 1, 0, 1);
        map(out->inputTensorAccessPattern[0], 2, 1, 1);
        constant(out->inputTensorAccessPattern[1], 0, 1);
        map(out->inputTensorAccessPattern[1], 1, 0, 1);
        map(out->inputTensorAccessPattern[1], 2, 1, 1);
        out->outputTensorAccessPattern[0] = out->inputTensorAccessPattern[0];
    }
    out->kernel.paramsNr = 0;
    const unsigned char* first = kind_ == Gather ? &_binary___deepseek_v41_prefill_sparse_kv_bf16_gaudi2_o_start
                                : (kind_ == Exp ? &_binary___deepseek_v41_prefill_sparse_exp_bf16_gaudi2_o_start
                                                : &_binary___deepseek_v41_prefill_sparse_normalize_bf16_gaudi2_o_start);
    const unsigned char* last = kind_ == Gather ? &_binary___deepseek_v41_prefill_sparse_kv_bf16_gaudi2_o_end
                               : (kind_ == Exp ? &_binary___deepseek_v41_prefill_sparse_exp_bf16_gaudi2_o_end
                                               : &_binary___deepseek_v41_prefill_sparse_normalize_bf16_gaudi2_o_end);
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = last - first;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, first, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
