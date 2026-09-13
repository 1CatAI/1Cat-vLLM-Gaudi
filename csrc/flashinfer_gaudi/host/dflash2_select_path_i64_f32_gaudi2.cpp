// SPDX-License-Identifier: Apache-2.0
/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "dflash2_select_path_i64_f32_gaudi2.hpp"

extern unsigned char _binary_dflash2_select_path_i64_f32_gaudi2_o_start;
extern unsigned char _binary_dflash2_select_path_i64_f32_gaudi2_o_end;

namespace {

constexpr uint64_t kSteps = 7;
constexpr uint64_t kTopK = 16;
constexpr uint64_t kMaxBatch = 16;

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

bool HasTypeAndRank(
    const tpc_lib_api::Tensor& tensor,
    tpc_lib_api::TensorDataType type,
    unsigned dims)
{
    return tensor.geometry.dataType == type && tensor.geometry.dims == dims;
}

}  // namespace

tpc_lib_api::GlueCodeReturn DFlash2SelectPathI64F32Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(kernelName, "flashinfer_gaudi_dflash2_select_path_i32_f32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DFlash2SelectPathI64F32Gaudi2::GetGcDefinitions(
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

    const auto& candidates = inDefs->inputTensors[0];
    const auto& scores = inDefs->inputTensors[1];
    auto& output = inDefs->outputTensors[0];
    if (!HasTypeAndRank(candidates, tpc_lib_api::DATA_I32, 3) ||
        !HasTypeAndRank(scores, tpc_lib_api::DATA_F32, 4)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const uint64_t topK = candidates.geometry.maxSizes[0];
    const uint64_t steps = candidates.geometry.maxSizes[1];
    const uint64_t batchSize = candidates.geometry.maxSizes[2];
    const bool shapesMatch =
        topK == kTopK && steps == kSteps && batchSize > 0 && batchSize <= kMaxBatch &&
        scores.geometry.maxSizes[0] == topK && scores.geometry.maxSizes[1] == topK &&
        scores.geometry.maxSizes[2] == steps && scores.geometry.maxSizes[3] == batchSize;
    if (!shapesMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool outputMatches =
        HasTypeAndRank(output, tpc_lib_api::DATA_I32, 2) &&
        output.geometry.maxSizes[0] == steps && output.geometry.maxSizes[1] == batchSize;
    if (!outputMatches) {
        output.geometry.dataType = tpc_lib_api::DATA_I32;
        output.geometry.dims = 2;
        output.geometry.maxSizes[0] = steps;
        output.geometry.maxSizes[1] = batchSize;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 1;
    outDefs->indexSpaceGeometry[0] = batchSize;

    MapDimension(outDefs->inputTensorAccessPattern[0], 0, 0, 0, 0, static_cast<int>(topK - 1));
    MapDimension(outDefs->inputTensorAccessPattern[0], 1, 0, 0, 0, static_cast<int>(steps - 1));
    MapDimension(outDefs->inputTensorAccessPattern[0], 2, 0, 1, 0, 0);

    MapDimension(outDefs->inputTensorAccessPattern[1], 0, 0, 0, 0, static_cast<int>(topK - 1));
    MapDimension(outDefs->inputTensorAccessPattern[1], 1, 0, 0, 0, static_cast<int>(topK - 1));
    MapDimension(outDefs->inputTensorAccessPattern[1], 2, 0, 0, 0, static_cast<int>(steps - 1));
    MapDimension(outDefs->inputTensorAccessPattern[1], 3, 0, 1, 0, 0);

    MapDimension(outDefs->outputTensorAccessPattern[0], 0, 0, 0, 0, static_cast<int>(steps - 1));
    MapDimension(outDefs->outputTensorAccessPattern[0], 1, 0, 1, 0, 0);

    outDefs->kernel.paramsNr = 0;
    const unsigned isaSize = &_binary_dflash2_select_path_i64_f32_gaudi2_o_end -
        &_binary_dflash2_select_path_i64_f32_gaudi2_o_start;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(
        outDefs->kernel.kernelElf,
        &_binary_dflash2_select_path_i64_f32_gaudi2_o_start,
        isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
