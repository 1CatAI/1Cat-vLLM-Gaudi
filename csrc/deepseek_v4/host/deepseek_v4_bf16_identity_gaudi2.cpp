// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v4_bf16_identity_gaudi2.hpp"

#include <cstring>

extern unsigned char _binary___deepseek_v4_bf16_identity_gaudi2_o_start;
extern unsigned char _binary___deepseek_v4_bf16_identity_gaudi2_o_end;

namespace {
constexpr uint64_t kHidden = 4096;
constexpr uint64_t kVectorWidth = 128;

void MapDimension(tpc_lib_api::TensorAccessPattern& pattern,
                  unsigned tensorDim, unsigned indexDim, int a,
                  int startB, int endB) {
    pattern.mapping[tensorDim].indexSpaceDim = indexDim;
    pattern.mapping[tensorDim].a = a;
    pattern.mapping[tensorDim].start_b = startB;
    pattern.mapping[tensorDim].end_b = endB;
}

bool HasDataType(tpc_lib_api::Tensor& tensor,
                 tpc_lib_api::TensorDataType expected) {
    if (tensor.geometry.dataType == expected) return true;
    tensor.geometry.dataType = expected;
    return false;
}

bool HasShape2(const tpc_lib_api::Tensor& tensor,
               uint64_t dim0, uint64_t dim1) {
    return tensor.geometry.dims == 2 &&
        tensor.geometry.maxSizes[0] == dim0 &&
        tensor.geometry.maxSizes[1] == dim1;
}
}  // namespace

tpc_lib_api::GlueCodeReturn DeepseekV4BF16IdentityGaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_deepseek_v4_bf16_identity_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV4BF16IdentityGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,
    tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 1) {
        in->inputTensorNr = 1;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1) {
        in->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (!HasDataType(in->inputTensors[0], DATA_BF16) ||
        !HasDataType(in->outputTensors[0], DATA_BF16)) {
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (!HasShape2(in->inputTensors[0], kHidden, 1)) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (!HasShape2(in->outputTensors[0], kHidden, 1)) {
        in->outputTensors[0].geometry.dims = 2;
        in->outputTensors[0].geometry.maxSizes[0] = kHidden;
        in->outputTensors[0].geometry.maxSizes[1] = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = kHidden / kVectorWidth;
    for (unsigned tensor = 0; tensor < 2; ++tensor) {
        auto& pattern = tensor == 0
            ? out->inputTensorAccessPattern[0]
            : out->outputTensorAccessPattern[0];
        MapDimension(pattern, 0, 0, kVectorWidth, 0, kVectorWidth - 1);
        MapDimension(pattern, 1, 0, 0, 0, 0);
    }

    out->kernel.paramsNr = 0;
    const unsigned char* start =
        &_binary___deepseek_v4_bf16_identity_gaudi2_o_start;
    const unsigned char* end =
        &_binary___deepseek_v4_bf16_identity_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
