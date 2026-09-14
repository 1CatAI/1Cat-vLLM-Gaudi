// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_mxfp4_indexed_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___deepseek_v41_mxfp4_indexed_fc1_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_indexed_fc1_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mxfp4_indexed_fc1_normal_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_indexed_fc1_normal_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mxfp4_indexed_fc2_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_indexed_fc2_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mxfp4_indexed_fc2_normal_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_indexed_fc2_normal_bf16_gaudi2_o_end;

namespace {
constexpr uint64_t kTopK = 6;
constexpr uint64_t kHidden = 5120;
constexpr uint64_t kIntermediate = 1152;
constexpr uint64_t kExperts = 384;
constexpr uint64_t kW13Blocks = 18;
constexpr uint64_t kW2Blocks = 40;
constexpr uint64_t kW13Stream = 163840;
constexpr uint64_t kW2Stream = 36864;

bool type(tpc_lib_api::Tensor& tensor, tpc_lib_api::TensorDataType expected) {
    if (tensor.geometry.dataType == expected) return true;
    tensor.geometry.dataType = expected;
    return false;
}
bool shape2(const tpc_lib_api::Tensor& tensor, uint64_t d0, uint64_t d1) {
    return tensor.geometry.dims == 2 && tensor.geometry.maxSizes[0] == d0 &&
           tensor.geometry.maxSizes[1] == d1;
}
bool shape3(const tpc_lib_api::Tensor& tensor, uint64_t d0, uint64_t d1,
            uint64_t d2) {
    return tensor.geometry.dims == 3 && tensor.geometry.maxSizes[0] == d0 &&
           tensor.geometry.maxSizes[1] == d1 && tensor.geometry.maxSizes[2] == d2;
}
void set3(tpc_lib_api::Tensor& tensor, uint64_t d0, uint64_t d1, uint64_t d2) {
    tensor.geometry.dims = 3;
    tensor.geometry.maxSizes[0] = d0;
    tensor.geometry.maxSizes[1] = d1;
    tensor.geometry.maxSizes[2] = d2;
}
void map(tpc_lib_api::TensorAccessPattern& p, unsigned dim, unsigned index,
         int a, int begin, int end) {
    p.mapping[dim].indexSpaceDim = index;
    p.mapping[dim].a = a;
    p.mapping[dim].start_b = begin;
    p.mapping[dim].end_b = end;
}
template<typename Start, typename End>
tpc_lib_api::GlueCodeReturn elf(Start start, End end,
                                tpc_lib_api::HabanaKernelInstantiation* out) {
    const unsigned size = static_cast<unsigned>(end - start);
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (capacity < size) return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, size);
    return tpc_lib_api::GLUE_SUCCESS;
}
}  // namespace

tpc_lib_api::GlueCodeReturn DeepseekV41Mxfp4IndexedBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    const char* stage = stage_ == FC1 ? "fc1" : "fc2";
    const char* suffix = normal_ ? "_normal_bf16_gaudi2" : "_bf16_gaudi2";
    std::strcpy(name, "custom_deepseek_v41_mxfp4_indexed_");
    std::strcat(name, stage);
    std::strcat(name, suffix);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV41Mxfp4IndexedBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,
    tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    constexpr unsigned kInputs = 5;  // activation, ids, Q16, S16, lookup
    if (in->inputTensorNr != kInputs) {
        in->inputTensorNr = kInputs;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1) {
        in->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const TensorDataType expected[] = {DATA_BF16, DATA_I32, DATA_I16,
                                       DATA_BF16, DATA_BF16};
    for (unsigned i = 0; i < kInputs; ++i)
        if (!type(in->inputTensors[i], expected[i]))
            return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (!type(in->outputTensors[0], DATA_BF16))
        return GLUE_INCOMPATIBLE_DATA_TYPE;

    const auto& activation = in->inputTensors[0];
    const auto& ids = in->inputTensors[1].geometry;
    const auto& q16 = in->inputTensors[2];
    const auto& s16 = in->inputTensors[3];
    const auto& lookup = in->inputTensors[4];
    if (ids.dims != 2 || ids.maxSizes[0] != kTopK || ids.maxSizes[1] == 0 ||
        ids.maxSizes[1] > 6 || lookup.geometry.dims != 1 ||
        lookup.geometry.maxSizes[0] != 128 || q16.geometry.dims != 3 ||
        s16.geometry.dims != 3 || q16.geometry.maxSizes[0] == 0 ||
        q16.geometry.maxSizes[1] == 0 || q16.geometry.maxSizes[2] != kExperts ||
        s16.geometry.maxSizes[0] * 8 != q16.geometry.maxSizes[0] ||
        s16.geometry.maxSizes[1] != q16.geometry.maxSizes[1] ||
        s16.geometry.maxSizes[2] != kExperts)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t tokens = ids.maxSizes[1];
    const uint64_t blocks = stage_ == FC1 ? kW13Blocks : kW2Blocks;
    const uint64_t stream = stage_ == FC1 ? kW13Stream : kW2Stream;
    const uint64_t outputWidth = stage_ == FC1 ? 2 * kIntermediate : kHidden;
    if (!shape3(q16, stream, blocks, kExperts) ||
        !shape3(s16, stream / 8, blocks, kExperts) ||
        (stage_ == FC1 ? !shape2(activation, kHidden, tokens)
                       : !shape3(activation, kIntermediate, kTopK, tokens)))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (stage_ == FC1) {
        if (in->outputTensors[0].geometry.dims != 3 ||
            in->outputTensors[0].geometry.maxSizes[0] != outputWidth ||
            in->outputTensors[0].geometry.maxSizes[1] != kTopK ||
            in->outputTensors[0].geometry.maxSizes[2] != tokens) {
            set3(in->outputTensors[0], outputWidth, kTopK, tokens);
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        }
        out->indexSpaceRank = 3;
        out->indexSpaceGeometry[0] = kIntermediate / 128;
        out->indexSpaceGeometry[1] = kTopK;
        out->indexSpaceGeometry[2] = tokens;
        map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, kHidden - 1);
        map(out->inputTensorAccessPattern[0], 1, 2, 1, 0, 0);
        map(out->inputTensorAccessPattern[1], 0, 1, 1, 0, 0);
        map(out->inputTensorAccessPattern[1], 1, 2, 1, 0, 0);
        map(out->inputTensorAccessPattern[2], 0, 0, 0, 0, stream - 1);
        map(out->inputTensorAccessPattern[2], 1, 0, 0, 0, blocks - 1);
        map(out->inputTensorAccessPattern[2], 2, 1, 0, 0, kExperts - 1);
        map(out->inputTensorAccessPattern[3], 0, 0, 0, 0, s16.geometry.maxSizes[0] - 1);
        map(out->inputTensorAccessPattern[3], 1, 0, 0, 0, blocks - 1);
        map(out->inputTensorAccessPattern[3], 2, 1, 0, 0, kExperts - 1);
        map(out->inputTensorAccessPattern[4], 0, 0, 0, 0, 127);
        // FC1 emits two adjacent 1152-wide halves.  Describe both halves so
        // the slicer does not assume that only the gate tile is touched.
        map(out->outputTensorAccessPattern[0], 0, 0, 128, 0,
            static_cast<int>(kIntermediate) + 127);
        map(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
        map(out->outputTensorAccessPattern[0], 2, 2, 1, 0, 0);
    } else {
        if (in->outputTensors[0].geometry.dims != 3 ||
            in->outputTensors[0].geometry.maxSizes[0] != outputWidth ||
            in->outputTensors[0].geometry.maxSizes[1] != kTopK ||
            in->outputTensors[0].geometry.maxSizes[2] != tokens) {
            set3(in->outputTensors[0], outputWidth, kTopK, tokens);
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        }
        out->indexSpaceRank = 3;
        out->indexSpaceGeometry[0] = kW2Blocks;
        out->indexSpaceGeometry[1] = kTopK;
        out->indexSpaceGeometry[2] = tokens;
        map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, kIntermediate - 1);
        map(out->inputTensorAccessPattern[0], 1, 1, 1, 0, 0);
        map(out->inputTensorAccessPattern[0], 2, 2, 1, 0, 0);
        map(out->inputTensorAccessPattern[1], 0, 1, 1, 0, 0);
        map(out->inputTensorAccessPattern[1], 1, 2, 1, 0, 0);
        map(out->inputTensorAccessPattern[2], 0, 0, 0, 0, stream - 1);
        map(out->inputTensorAccessPattern[2], 1, 0, 0, 0, blocks - 1);
        map(out->inputTensorAccessPattern[2], 2, 1, 0, 0, kExperts - 1);
        map(out->inputTensorAccessPattern[3], 0, 0, 0, 0, s16.geometry.maxSizes[0] - 1);
        map(out->inputTensorAccessPattern[3], 1, 0, 0, 0, blocks - 1);
        map(out->inputTensorAccessPattern[3], 2, 1, 0, 0, kExperts - 1);
        map(out->inputTensorAccessPattern[4], 0, 0, 0, 0, 127);
        map(out->outputTensorAccessPattern[0], 0, 0, 128, 0, 0);
        map(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
        map(out->outputTensorAccessPattern[0], 2, 2, 1, 0, 0);
    }
    out->kernel.paramsNr = 0;
    const unsigned char *start, *end;
    if (stage_ == FC1 && normal_) { start = &_binary___deepseek_v41_mxfp4_indexed_fc1_normal_bf16_gaudi2_o_start; end = &_binary___deepseek_v41_mxfp4_indexed_fc1_normal_bf16_gaudi2_o_end; }
    else if (stage_ == FC1) { start = &_binary___deepseek_v41_mxfp4_indexed_fc1_bf16_gaudi2_o_start; end = &_binary___deepseek_v41_mxfp4_indexed_fc1_bf16_gaudi2_o_end; }
    else if (normal_) { start = &_binary___deepseek_v41_mxfp4_indexed_fc2_normal_bf16_gaudi2_o_start; end = &_binary___deepseek_v41_mxfp4_indexed_fc2_normal_bf16_gaudi2_o_end; }
    else { start = &_binary___deepseek_v41_mxfp4_indexed_fc2_bf16_gaudi2_o_start; end = &_binary___deepseek_v41_mxfp4_indexed_fc2_bf16_gaudi2_o_end; }
    return elf(start, end, out);
}
