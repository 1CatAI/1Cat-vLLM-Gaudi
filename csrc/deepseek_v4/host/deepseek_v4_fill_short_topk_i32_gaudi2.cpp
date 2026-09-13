/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "deepseek_v4_fill_short_topk_i32_gaudi2.hpp"

extern unsigned char
    _binary___deepseek_v4_fill_short_topk_i32_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_fill_short_topk_i32_gaudi2_o_end;

namespace {

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

}  // namespace

tpc_lib_api::GlueCodeReturn
DeepseekV4FillShortTopkI32Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(
        kernelName, "custom_deepseek_v4_fill_short_topk_i32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4FillShortTopkI32Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    constexpr unsigned kInputCount = 2;
    constexpr unsigned kOutputCount = 1;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    for (unsigned input = 0; input < kInputCount; ++input) {
        if (!HasDataType(
                inDefs->inputTensors[input], tpc_lib_api::DATA_I32)) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    if (!HasDataType(inDefs->outputTensors[0], tpc_lib_api::DATA_I32)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const auto& output = inDefs->inputTensors[0];
    const auto& positions = inDefs->inputTensors[1];
    auto& validCounts = inDefs->outputTensors[0];
    const bool shapesMatch =
        output.geometry.dims == 2 &&
        output.geometry.maxSizes[0] == 512 &&
        output.geometry.maxSizes[1] > 0 &&
        positions.geometry.dims == 1 &&
        positions.geometry.maxSizes[0] == output.geometry.maxSizes[1];
    if (!shapesMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const uint64_t batchSize = output.geometry.maxSizes[1];
    if (validCounts.geometry.dims != 1 ||
        validCounts.geometry.maxSizes[0] != batchSize) {
        validCounts.geometry.dims = 1;
        validCounts.geometry.maxSizes[0] = batchSize;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 1;
    outDefs->indexSpaceGeometry[0] = batchSize;
    MapDimension(outDefs->inputTensorAccessPattern[0], 0, 0, 0, 0, 511);
    MapDimension(outDefs->inputTensorAccessPattern[0], 1, 0, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[1], 0, 0, 1, 0, 0);
    MapDimension(outDefs->outputTensorAccessPattern[0], 0, 0, 1, 0, 0);

    outDefs->kernel.paramsNr = 0;
    const unsigned isaSize =
        &_binary___deepseek_v4_fill_short_topk_i32_gaudi2_o_end -
        &_binary___deepseek_v4_fill_short_topk_i32_gaudi2_o_start;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(
        outDefs->kernel.kernelElf,
        &_binary___deepseek_v4_fill_short_topk_i32_gaudi2_o_start,
        isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
