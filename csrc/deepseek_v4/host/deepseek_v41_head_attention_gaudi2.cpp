// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_head_attention_gaudi2.hpp"
#include <cstring>
#include <initializer_list>

#define BINARY(name) extern unsigned char _binary___##name##_o_start; extern unsigned char _binary___##name##_o_end;
BINARY(deepseek_v41_attn_scores_f32_gaudi2)
BINARY(deepseek_v41_attn_recurrence_f32_gaudi2)
BINARY(deepseek_v41_attn_values_f32_gaudi2)
#undef BINARY
namespace {
using namespace tpc_lib_api;
bool shape(const Tensor& t, std::initializer_list<uint64_t> sizes) {
    if (t.geometry.dims != sizes.size()) return false;
    unsigned dim = 0;
    for (auto size : sizes) if (t.geometry.maxSizes[dim++] != size) return false;
    return true;
}
void map(TensorAccessPattern& p, unsigned d, unsigned index, int a, int last = 0) {
    auto& m = p.mapping[d];
    m.indexSpaceDim = index;
    m.a = a;
    m.start_b = 0;
    m.end_b = last;
}
}
tpc_lib_api::GlueCodeReturn DeepseekV41HeadAttentionGaudi2::GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]) {
    const char* names[] = {"custom_deepseek_v41_attn_scores_f32_gaudi2",
                          "custom_deepseek_v41_attn_recurrence_f32_gaudi2",
                          "custom_deepseek_v41_attn_values_f32_gaudi2"};
    std::strcpy(name, names[phase_]);
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn DeepseekV41HeadAttentionGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    const unsigned inputs = phase_ == 1 ? 4 : 5, outputs = phase_ == 1 ? 2 : 1;
    if (in->inputTensorNr != inputs) { in->inputTensorNr = inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != outputs) { in->outputTensorNr = outputs; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const TensorDataType types[3][5] = {{DATA_BF16, DATA_BF16, DATA_I32, DATA_F32, DATA_I32},
                                     {DATA_F32, DATA_I32, DATA_F32, DATA_I32, DATA_F32},
                                     {DATA_BF16, DATA_I32, DATA_I32, DATA_F32, DATA_F32}};
    for (unsigned i = 0; i < inputs + outputs; ++i) {
        auto& t = i < inputs ? in->inputTensors[i] : in->outputTensors[i - inputs];
        const auto expected = i < inputs ? types[phase_][i] : DATA_F32;
        if (t.geometry.dataType != expected) { t.geometry.dataType = expected; return GLUE_INCOMPATIBLE_DATA_TYPE; }
    }
    const auto& ids = in->inputTensors[phase_ == 0 ? 2 : 1];
    const uint64_t width = ids.geometry.maxSizes[0], tokens = ids.geometry.maxSizes[1];
    if (!shape(ids, {width, tokens}) || width == 0 || width > 4096 || tokens == 0 || tokens > 6)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const unsigned li = phase_ == 0 ? 4 : phase_ == 1 ? 3 : 2;
    if (!shape(in->inputTensors[li], {tokens})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    auto* ia = out->inputTensorAccessPattern;
    auto* oa = out->outputTensorAccessPattern;
    out->indexSpaceRank = 3;
    out->indexSpaceGeometry[0] = phase_ == 0 ? 32 : phase_ == 1 ? 1 : 64;
    out->indexSpaceGeometry[1] = phase_ == 0 ? (width + 7) / 8 : 1;
    out->indexSpaceGeometry[2] = tokens;
    out->kernel.paramsNr = 0;
    if (phase_ == 0) {
        const uint64_t rows = in->inputTensors[1].geometry.maxSizes[1];
        if (!shape(in->inputTensors[0], {512, 32, tokens}) ||
            !shape(in->inputTensors[1], {512, rows}) || rows == 0 || rows > 4096 ||
            !shape(in->inputTensors[3], {1})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!shape(in->outputTensors[0], {32, width, tokens})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        map(ia[0], 0, 0, 0, 511); map(ia[0], 1, 0, 1); map(ia[0], 2, 2, 1);
        ia[1].sparseAccess = true;
        map(ia[1], 0, 0, 0, 511); map(ia[1], 1, 0, 0, rows - 1);
        map(ia[2], 0, 1, 8, 7); map(ia[2], 1, 2, 1);
        map(ia[3], 0, 0, 0); map(ia[4], 0, 2, 1);
        map(oa[0], 0, 0, 1); map(oa[0], 1, 1, 8, 7); map(oa[0], 2, 2, 1);
    } else if (phase_ == 1) {
        if (!shape(in->inputTensors[0], {32, width, tokens}) || !shape(in->inputTensors[2], {32}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!shape(in->outputTensors[0], {32, 2, width, tokens}) ||
            !shape(in->outputTensors[1], {32, tokens})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize != sizeof(int32_t))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const int32_t rows = *static_cast<const int32_t*>(in->nodeParams.nodeParams);
        if (rows <= 0 || rows > 4096) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->kernel.paramsNr = 1;
        out->kernel.scalarParams[0] = rows;
        map(ia[0], 0, 0, 0, 31); map(ia[0], 1, 0, 0, width - 1); map(ia[0], 2, 2, 1);
        map(ia[1], 0, 0, 0, width - 1); map(ia[1], 1, 2, 1);
        map(ia[2], 0, 0, 0, 31); map(ia[3], 0, 2, 1);
        map(oa[0], 0, 0, 0, 31); map(oa[0], 1, 0, 0, 1); map(oa[0], 2, 0, 0, width - 1);
        map(oa[0], 3, 2, 1); map(oa[1], 0, 0, 0, 31); map(oa[1], 1, 2, 1);
    } else {
        const uint64_t rows = in->inputTensors[0].geometry.maxSizes[1];
        if (!shape(in->inputTensors[0], {512, rows}) || rows == 0 || rows > 4096 ||
            !shape(in->inputTensors[3], {32, 2, width, tokens}) ||
            !shape(in->inputTensors[4], {32, tokens})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!shape(in->outputTensors[0], {32, 512, tokens})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        ia[0].sparseAccess = true;
        map(ia[0], 0, 0, 8, 7); map(ia[0], 1, 0, 0, rows - 1);
        map(ia[1], 0, 0, 0, width - 1); map(ia[1], 1, 2, 1); map(ia[2], 0, 2, 1);
        map(ia[3], 0, 0, 0, 31); map(ia[3], 1, 0, 0, 1); map(ia[3], 2, 0, 0, width - 1);
        map(ia[3], 3, 2, 1); map(ia[4], 0, 0, 0, 31); map(ia[4], 1, 2, 1);
        map(oa[0], 0, 0, 0, 31); map(oa[0], 1, 0, 8, 7); map(oa[0], 2, 2, 1);
    }
    unsigned char* starts[] = {&_binary___deepseek_v41_attn_scores_f32_gaudi2_o_start,
        &_binary___deepseek_v41_attn_recurrence_f32_gaudi2_o_start, &_binary___deepseek_v41_attn_values_f32_gaudi2_o_start};
    unsigned char* ends[] = {&_binary___deepseek_v41_attn_scores_f32_gaudi2_o_end,
        &_binary___deepseek_v41_attn_recurrence_f32_gaudi2_o_end, &_binary___deepseek_v41_attn_values_f32_gaudi2_o_end};
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = ends[phase_] - starts[phase_];
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, starts[phase_], out->kernel.elfSize);
    return GLUE_SUCCESS;
}
