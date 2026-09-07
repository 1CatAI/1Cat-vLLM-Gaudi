/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "deepseek_v4_mxfp4_dequant_fp8_gaudi2.hpp"

extern unsigned char
    _binary___deepseek_v4_mxfp4_dequant_fp8_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_mxfp4_dequant_fp8_gaudi2_o_end;

namespace {

constexpr uint64_t kTopK = 6;
constexpr uint64_t kExperts = 256;
constexpr uint64_t kW13Rows = 2048;
constexpr uint64_t kW13PackedCols = 2048;
constexpr uint64_t kW13Cols = 4096;
constexpr uint64_t kW2Rows = 4096;
constexpr uint64_t kW2PackedCols = 512;
constexpr uint64_t kW2Cols = 1024;
constexpr uint64_t kS13Cols = 128;
constexpr uint64_t kS2Cols = 32;

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

}  // namespace

tpc_lib_api::GlueCodeReturn
DeepseekV4Mxfp4DequantFp8Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(
        kernelName, "custom_deepseek_v4_mxfp4_dequant_fp8_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4Mxfp4DequantFp8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    constexpr unsigned kInputCount = 5;
    constexpr unsigned kOutputCount = 2;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (!HasDataType(inDefs->inputTensors[0], tpc_lib_api::DATA_I32)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    for (unsigned index = 1; index < kInputCount; ++index) {
        if (!HasDataType(inDefs->inputTensors[index], tpc_lib_api::DATA_U8)) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    for (unsigned index = 0; index < kOutputCount; ++index) {
        if (!HasDataType(
                inDefs->outputTensors[index],
                tpc_lib_api::DATA_F8_143)) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }

    const auto& ids = inDefs->inputTensors[0];
    if (ids.geometry.dims != 2 || ids.geometry.maxSizes[0] != kTopK ||
        ids.geometry.maxSizes[1] != 1 ||
        !HasShape3(
            inDefs->inputTensors[1],
            kW13PackedCols,
            kW13Rows,
            kExperts) ||
        !HasShape3(
            inDefs->inputTensors[2],
            kW2PackedCols,
            kW2Rows,
            kExperts) ||
        !HasShape3(
            inDefs->inputTensors[3], kS13Cols, kW13Rows, kExperts) ||
        !HasShape3(
            inDefs->inputTensors[4], kS2Cols, kW2Rows, kExperts)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const bool outputsMatch =
        HasShape3(inDefs->outputTensors[0], kW13Cols, kW13Rows, kTopK) &&
        HasShape3(inDefs->outputTensors[1], kW2Cols, kW2Rows, kTopK);
    if (!outputsMatch) {
        SetShape3(inDefs->outputTensors[0], kW13Cols, kW13Rows, kTopK);
        SetShape3(inDefs->outputTensors[1], kW2Cols, kW2Rows, kTopK);
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 2;
    outDefs->indexSpaceGeometry[0] = kW13Rows;
    outDefs->indexSpaceGeometry[1] = kTopK;

    MapDimension(outDefs->inputTensorAccessPattern[0], 0, 1, 1, 0, 0);
    MapDimension(outDefs->inputTensorAccessPattern[0], 1, 0, 0, 0, 0);

    const uint64_t inputColumns[] = {
        kW13PackedCols, kW2PackedCols, kS13Cols, kS2Cols};
    const uint64_t inputRowOffsets[] = {0, kW13Rows, 0, kW13Rows};
    for (unsigned index = 0; index < 4; ++index) {
        auto& pattern = outDefs->inputTensorAccessPattern[index + 1];
        MapDimension(
            pattern,
            0,
            0,
            0,
            0,
            static_cast<int>(inputColumns[index] - 1));
        MapDimension(
            pattern,
            1,
            0,
            1,
            0,
            static_cast<int>(inputRowOffsets[index]));
        MapDimension(
            pattern,
            2,
            1,
            0,
            0,
            static_cast<int>(kExperts - 1));
    }

    const uint64_t outputColumns[] = {kW13Cols, kW2Cols};
    const uint64_t outputRowOffsets[] = {0, kW13Rows};
    for (unsigned index = 0; index < kOutputCount; ++index) {
        auto& pattern = outDefs->outputTensorAccessPattern[index];
        MapDimension(
            pattern,
            0,
            0,
            0,
            0,
            static_cast<int>(outputColumns[index] - 1));
        MapDimension(
            pattern,
            1,
            0,
            1,
            0,
            static_cast<int>(outputRowOffsets[index]));
        MapDimension(pattern, 2, 1, 1, 0, 0);
    }

    outDefs->kernel.paramsNr = 0;
    const unsigned isaSize =
        &_binary___deepseek_v4_mxfp4_dequant_fp8_gaudi2_o_end -
        &_binary___deepseek_v4_mxfp4_dequant_fp8_gaudi2_o_start;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(
        outDefs->kernel.kernelElf,
        &_binary___deepseek_v4_mxfp4_dequant_fp8_gaudi2_o_start,
        isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
