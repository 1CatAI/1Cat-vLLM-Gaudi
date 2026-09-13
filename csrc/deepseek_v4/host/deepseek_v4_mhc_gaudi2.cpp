/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "deepseek_v4_mhc_gaudi2.hpp"

extern unsigned char _binary___deepseek_v4_mhc_post_prepare_gaudi2_o_start;
extern unsigned char _binary___deepseek_v4_mhc_post_prepare_gaudi2_o_end;
extern unsigned char _binary___deepseek_v4_mhc_pre_emit_gaudi2_o_start;
extern unsigned char _binary___deepseek_v4_mhc_pre_emit_gaudi2_o_end;
extern unsigned char _binary___deepseek_v4_mhc_pre_emit_norm_gaudi2_o_start;
extern unsigned char _binary___deepseek_v4_mhc_pre_emit_norm_gaudi2_o_end;

namespace {

constexpr uint64_t kHiddenSize = 4096;
constexpr uint64_t kStreams = 4;
constexpr uint64_t kMixes = 24;

void MapDimension(
    tpc_lib_api::TensorAccessPattern& pattern,
    unsigned tensorDim,
    unsigned indexDim,
    int a,
    int startB,
    int endB)
{
    pattern.mapping[tensorDim].indexSpaceDim = indexDim;
    pattern.mapping[tensorDim].a = a;
    pattern.mapping[tensorDim].start_b = startB;
    pattern.mapping[tensorDim].end_b = endB;
}

bool HasDataType(
    tpc_lib_api::Tensor& tensor,
    tpc_lib_api::TensorDataType expected)
{
    if (tensor.geometry.dataType == expected) {
        return true;
    }
    tensor.geometry.dataType = expected;
    return false;
}

bool HasShape1(const tpc_lib_api::Tensor& tensor, uint64_t dim0)
{
    return tensor.geometry.dims == 1 &&
        tensor.geometry.maxSizes[0] == dim0;
}

bool HasShape2(
    const tpc_lib_api::Tensor& tensor,
    uint64_t dim0,
    uint64_t dim1)
{
    return tensor.geometry.dims == 2 &&
        tensor.geometry.maxSizes[0] == dim0 &&
        tensor.geometry.maxSizes[1] == dim1;
}

bool HasShape3(
    const tpc_lib_api::Tensor& tensor,
    uint64_t dim0,
    uint64_t dim1,
    uint64_t dim2)
{
    return tensor.geometry.dims == 3 &&
        tensor.geometry.maxSizes[0] == dim0 &&
        tensor.geometry.maxSizes[1] == dim1 &&
        tensor.geometry.maxSizes[2] == dim2;
}

void SetShape2(
    tpc_lib_api::Tensor& tensor,
    uint64_t dim0,
    uint64_t dim1)
{
    tensor.geometry.dims = 2;
    tensor.geometry.maxSizes[0] = dim0;
    tensor.geometry.maxSizes[1] = dim1;
}

void SetShape3(
    tpc_lib_api::Tensor& tensor,
    uint64_t dim0,
    uint64_t dim1,
    uint64_t dim2)
{
    tensor.geometry.dims = 3;
    tensor.geometry.maxSizes[0] = dim0;
    tensor.geometry.maxSizes[1] = dim1;
    tensor.geometry.maxSizes[2] = dim2;
}

void MapTokenMatrix(
    tpc_lib_api::TensorAccessPattern& pattern,
    uint64_t width)
{
    MapDimension(pattern, 0, 0, 0, 0, static_cast<int>(width - 1));
    MapDimension(pattern, 1, 0, 1, 0, 0);
}

void MapTokenStreams(
    tpc_lib_api::TensorAccessPattern& pattern,
    uint64_t width,
    uint64_t streams)
{
    MapDimension(pattern, 0, 0, 0, 0, static_cast<int>(width - 1));
    MapDimension(pattern, 1, 0, 0, 0, static_cast<int>(streams - 1));
    MapDimension(pattern, 2, 0, 1, 0, 0);
}

tpc_lib_api::GlueCodeReturn CopyKernel(
    unsigned char* start,
    unsigned char* end,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    outDefs->kernel.paramsNr = 0;
    const unsigned isaSize = end - start;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(outDefs->kernel.kernelElf, start, isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}

}  // namespace

tpc_lib_api::GlueCodeReturn DeepseekV4MHCPostPrepareGaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(
        kernelName,
        "custom_deepseek_v4_mhc_post_prepare_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4MHCPostPrepareGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    constexpr unsigned kInputCount = 4;
    constexpr unsigned kOutputCount = 3;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (!HasDataType(inDefs->inputTensors[0], tpc_lib_api::DATA_BF16) ||
        !HasDataType(inDefs->inputTensors[1], tpc_lib_api::DATA_BF16) ||
        !HasDataType(inDefs->inputTensors[2], tpc_lib_api::DATA_F32) ||
        !HasDataType(inDefs->inputTensors[3], tpc_lib_api::DATA_F32) ||
        !HasDataType(inDefs->outputTensors[0], tpc_lib_api::DATA_BF16) ||
        !HasDataType(inDefs->outputTensors[1], tpc_lib_api::DATA_F32) ||
        !HasDataType(inDefs->outputTensors[2], tpc_lib_api::DATA_F32)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const auto tokenCount = inDefs->inputTensors[0].geometry.maxSizes[1];
    if (tokenCount == 0 ||
        !HasShape2(inDefs->inputTensors[0], kHiddenSize, tokenCount) ||
        !HasShape3(
            inDefs->inputTensors[1], kHiddenSize, kStreams, tokenCount) ||
        !HasShape3(inDefs->inputTensors[2], 1, kStreams, tokenCount) ||
        !HasShape3(
            inDefs->inputTensors[3], kStreams, kStreams, tokenCount)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool outputsMatch =
        HasShape3(
            inDefs->outputTensors[0], kHiddenSize, kStreams, tokenCount) &&
        HasShape2(
            inDefs->outputTensors[1], kStreams * kHiddenSize, tokenCount) &&
        HasShape2(inDefs->outputTensors[2], 1, tokenCount);
    if (!outputsMatch) {
        SetShape3(
            inDefs->outputTensors[0], kHiddenSize, kStreams, tokenCount);
        SetShape2(
            inDefs->outputTensors[1], kStreams * kHiddenSize, tokenCount);
        SetShape2(inDefs->outputTensors[2], 1, tokenCount);
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 1;
    outDefs->indexSpaceGeometry[0] = tokenCount;
    MapTokenMatrix(outDefs->inputTensorAccessPattern[0], kHiddenSize);
    MapTokenStreams(
        outDefs->inputTensorAccessPattern[1], kHiddenSize, kStreams);
    MapTokenStreams(outDefs->inputTensorAccessPattern[2], 1, kStreams);
    MapTokenStreams(
        outDefs->inputTensorAccessPattern[3], kStreams, kStreams);
    MapTokenStreams(
        outDefs->outputTensorAccessPattern[0], kHiddenSize, kStreams);
    MapTokenMatrix(
        outDefs->outputTensorAccessPattern[1], kStreams * kHiddenSize);
    MapTokenMatrix(outDefs->outputTensorAccessPattern[2], 1);

    return CopyKernel(
        &_binary___deepseek_v4_mhc_post_prepare_gaudi2_o_start,
        &_binary___deepseek_v4_mhc_post_prepare_gaudi2_o_end,
        outDefs);
}

tpc_lib_api::GlueCodeReturn DeepseekV4MHCPreEmitGaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(kernelName, "custom_deepseek_v4_mhc_pre_emit_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4MHCPreEmitGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    constexpr unsigned kInputCount = 5;
    constexpr unsigned kOutputCount = 3;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (!HasDataType(inDefs->inputTensors[0], tpc_lib_api::DATA_BF16)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    for (unsigned index = 1; index < kInputCount; ++index) {
        if (!HasDataType(inDefs->inputTensors[index], tpc_lib_api::DATA_F32)) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    if (!HasDataType(inDefs->outputTensors[0], tpc_lib_api::DATA_F32) ||
        !HasDataType(inDefs->outputTensors[1], tpc_lib_api::DATA_F32) ||
        !HasDataType(inDefs->outputTensors[2], tpc_lib_api::DATA_BF16)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const auto tokenCount = inDefs->inputTensors[0].geometry.maxSizes[2];
    if (tokenCount == 0 ||
        !HasShape3(
            inDefs->inputTensors[0], kHiddenSize, kStreams, tokenCount) ||
        !HasShape2(inDefs->inputTensors[1], kMixes, tokenCount) ||
        !HasShape2(inDefs->inputTensors[2], 1, tokenCount) ||
        !HasShape1(inDefs->inputTensors[3], 3) ||
        !HasShape1(inDefs->inputTensors[4], kMixes)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool outputsMatch =
        HasShape3(inDefs->outputTensors[0], 1, kStreams, tokenCount) &&
        HasShape3(
            inDefs->outputTensors[1], 1, kStreams * kStreams, tokenCount) &&
        HasShape2(inDefs->outputTensors[2], kHiddenSize, tokenCount);
    if (!outputsMatch) {
        SetShape3(inDefs->outputTensors[0], 1, kStreams, tokenCount);
        SetShape3(
            inDefs->outputTensors[1], 1, kStreams * kStreams, tokenCount);
        SetShape2(inDefs->outputTensors[2], kHiddenSize, tokenCount);
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 1;
    outDefs->indexSpaceGeometry[0] = tokenCount;
    MapTokenStreams(
        outDefs->inputTensorAccessPattern[0], kHiddenSize, kStreams);
    MapTokenMatrix(outDefs->inputTensorAccessPattern[1], kMixes);
    MapTokenMatrix(outDefs->inputTensorAccessPattern[2], 1);
    MapDimension(
        outDefs->inputTensorAccessPattern[3], 0, 0, 0, 0, 2);
    MapDimension(
        outDefs->inputTensorAccessPattern[4], 0, 0, 0, 0, kMixes - 1);
    MapTokenStreams(outDefs->outputTensorAccessPattern[0], 1, kStreams);
    MapTokenStreams(
        outDefs->outputTensorAccessPattern[1], 1, kStreams * kStreams);
    MapTokenMatrix(outDefs->outputTensorAccessPattern[2], kHiddenSize);

    return CopyKernel(
        &_binary___deepseek_v4_mhc_pre_emit_gaudi2_o_start,
        &_binary___deepseek_v4_mhc_pre_emit_gaudi2_o_end,
        outDefs);
}

tpc_lib_api::GlueCodeReturn DeepseekV4MHCPreEmitNormGaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(
        kernelName,
        "custom_deepseek_v4_mhc_pre_emit_norm_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4MHCPreEmitNormGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    constexpr unsigned kInputCount = 6;
    constexpr unsigned kOutputCount = 3;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (!HasDataType(inDefs->inputTensors[0], tpc_lib_api::DATA_BF16) ||
        !HasDataType(inDefs->inputTensors[5], tpc_lib_api::DATA_BF16)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    for (unsigned index = 1; index < 5; ++index) {
        if (!HasDataType(inDefs->inputTensors[index], tpc_lib_api::DATA_F32)) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    if (!HasDataType(inDefs->outputTensors[0], tpc_lib_api::DATA_F32) ||
        !HasDataType(inDefs->outputTensors[1], tpc_lib_api::DATA_F32) ||
        !HasDataType(inDefs->outputTensors[2], tpc_lib_api::DATA_BF16)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const auto tokenCount = inDefs->inputTensors[0].geometry.maxSizes[2];
    if (tokenCount == 0 ||
        !HasShape3(
            inDefs->inputTensors[0], kHiddenSize, kStreams, tokenCount) ||
        !HasShape2(inDefs->inputTensors[1], kMixes, tokenCount) ||
        !HasShape2(inDefs->inputTensors[2], 1, tokenCount) ||
        !HasShape1(inDefs->inputTensors[3], 3) ||
        !HasShape1(inDefs->inputTensors[4], kMixes) ||
        !HasShape1(inDefs->inputTensors[5], kHiddenSize)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool outputsMatch =
        HasShape3(inDefs->outputTensors[0], 1, kStreams, tokenCount) &&
        HasShape3(
            inDefs->outputTensors[1], 1, kStreams * kStreams, tokenCount) &&
        HasShape2(inDefs->outputTensors[2], kHiddenSize, tokenCount);
    if (!outputsMatch) {
        SetShape3(inDefs->outputTensors[0], 1, kStreams, tokenCount);
        SetShape3(
            inDefs->outputTensors[1], 1, kStreams * kStreams, tokenCount);
        SetShape2(inDefs->outputTensors[2], kHiddenSize, tokenCount);
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 1;
    outDefs->indexSpaceGeometry[0] = tokenCount;
    MapTokenStreams(
        outDefs->inputTensorAccessPattern[0], kHiddenSize, kStreams);
    MapTokenMatrix(outDefs->inputTensorAccessPattern[1], kMixes);
    MapTokenMatrix(outDefs->inputTensorAccessPattern[2], 1);
    MapDimension(
        outDefs->inputTensorAccessPattern[3], 0, 0, 0, 0, 2);
    MapDimension(
        outDefs->inputTensorAccessPattern[4], 0, 0, 0, 0, kMixes - 1);
    MapDimension(
        outDefs->inputTensorAccessPattern[5],
        0,
        0,
        0,
        0,
        kHiddenSize - 1);
    MapTokenStreams(outDefs->outputTensorAccessPattern[0], 1, kStreams);
    MapTokenStreams(
        outDefs->outputTensorAccessPattern[1], 1, kStreams * kStreams);
    MapTokenMatrix(outDefs->outputTensorAccessPattern[2], kHiddenSize);

    return CopyKernel(
        &_binary___deepseek_v4_mhc_pre_emit_norm_gaudi2_o_start,
        &_binary___deepseek_v4_mhc_pre_emit_norm_gaudi2_o_end,
        outDefs);
}
