/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "deepseek_v4_mxfp4_indexed_moe_gaudi2.hpp"

extern unsigned char
    _binary___deepseek_v4_mxfp4_indexed_fc1_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_mxfp4_indexed_fc1_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_mxfp4_indexed_fc2_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_mxfp4_indexed_fc2_gaudi2_o_end;

namespace {

constexpr uint64_t kTopK = 6;
constexpr uint64_t kExperts = 256;
constexpr uint64_t kHidden = 4096;
constexpr uint64_t kIntermediate = 1024;
constexpr uint64_t kW13Rows = 2048;
constexpr uint64_t kW13PackedCols = 2048;
constexpr uint64_t kW13ScaleCols = 128;
constexpr uint64_t kW2Rows = 4096;
constexpr uint64_t kW2PackedCols = 512;
constexpr uint64_t kW2ScaleCols = 32;

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

tpc_lib_api::GlueCodeReturn CopyElf(
    const unsigned char* start,
    const unsigned char* end,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
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

tpc_lib_api::GlueCodeReturn
DeepseekV4Mxfp4IndexedMoeGaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(
        kernelName,
        stage_ == FC1
            ? "custom_deepseek_v4_mxfp4_indexed_fc1_gaudi2"
            : "custom_deepseek_v4_mxfp4_indexed_fc2_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4Mxfp4IndexedMoeGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    const unsigned inputCount = stage_ == FC1 ? 4 : 5;
    constexpr unsigned kOutputCount = 1;
    if (inDefs->inputTensorNr != inputCount) {
        inDefs->inputTensorNr = inputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    const tpc_lib_api::TensorDataType fc1Types[4] = {
        tpc_lib_api::DATA_BF16,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_U8,
        tpc_lib_api::DATA_U8,
    };
    const tpc_lib_api::TensorDataType fc2Types[5] = {
        tpc_lib_api::DATA_BF16,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_BF16,
        tpc_lib_api::DATA_U8,
        tpc_lib_api::DATA_U8,
    };
    const auto* inputTypes = stage_ == FC1 ? fc1Types : fc2Types;
    for (unsigned input = 0; input < inputCount; ++input) {
        if (!HasDataType(inDefs->inputTensors[input], inputTypes[input])) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    if (!HasDataType(
            inDefs->outputTensors[0], tpc_lib_api::DATA_BF16)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const auto& ids = inDefs->inputTensors[1];
    if (!HasShape2(ids, kTopK, 1)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    if (stage_ == FC1) {
        if (!HasShape2(inDefs->inputTensors[0], kHidden, 1) ||
            !HasShape3(
                inDefs->inputTensors[2],
                kW13PackedCols,
                kW13Rows,
                kExperts) ||
            !HasShape3(
                inDefs->inputTensors[3],
                kW13ScaleCols,
                kW13Rows,
                kExperts)) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
        }
        if (!HasShape2(
                inDefs->outputTensors[0], kIntermediate, kTopK)) {
            SetShape2(inDefs->outputTensors[0], kIntermediate, kTopK);
            return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        }

        outDefs->indexSpaceRank = 2;
        outDefs->indexSpaceGeometry[0] = kIntermediate;
        outDefs->indexSpaceGeometry[1] = kTopK;
        MapDimension(
            outDefs->inputTensorAccessPattern[0],
            0,
            0,
            0,
            0,
            kHidden - 1);
        MapDimension(
            outDefs->inputTensorAccessPattern[0], 1, 0, 0, 0, 0);
        MapDimension(
            outDefs->inputTensorAccessPattern[1], 0, 1, 1, 0, 0);
        MapDimension(
            outDefs->inputTensorAccessPattern[1], 1, 0, 0, 0, 0);
        for (unsigned input = 2; input < 4; ++input) {
            const int columns = input == 2
                ? kW13PackedCols
                : kW13ScaleCols;
            MapDimension(
                outDefs->inputTensorAccessPattern[input],
                0,
                0,
                0,
                0,
                columns - 1);
            MapDimension(
                outDefs->inputTensorAccessPattern[input],
                1,
                0,
                1,
                0,
                kIntermediate);
            MapDimension(
                outDefs->inputTensorAccessPattern[input],
                2,
                1,
                0,
                0,
                kExperts - 1);
        }
        MapDimension(
            outDefs->outputTensorAccessPattern[0], 0, 0, 1, 0, 0);
        MapDimension(
            outDefs->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
        outDefs->kernel.paramsNr = 0;
        return CopyElf(
            &_binary___deepseek_v4_mxfp4_indexed_fc1_gaudi2_o_start,
            &_binary___deepseek_v4_mxfp4_indexed_fc1_gaudi2_o_end,
            outDefs);
    }

    if (!HasShape2(
            inDefs->inputTensors[0], kIntermediate, kTopK) ||
        !HasShape2(inDefs->inputTensors[2], kTopK, 1) ||
        !HasShape3(
            inDefs->inputTensors[3],
            kW2PackedCols,
            kW2Rows,
            kExperts) ||
        !HasShape3(
            inDefs->inputTensors[4],
            kW2ScaleCols,
            kW2Rows,
            kExperts)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (!HasShape2(inDefs->outputTensors[0], kHidden, 1)) {
        SetShape2(inDefs->outputTensors[0], kHidden, 1);
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 1;
    outDefs->indexSpaceGeometry[0] = kHidden;
    MapDimension(
        outDefs->inputTensorAccessPattern[0],
        0,
        0,
        0,
        0,
        kIntermediate - 1);
    MapDimension(
        outDefs->inputTensorAccessPattern[0],
        1,
        0,
        0,
        0,
        kTopK - 1);
    for (unsigned input = 1; input <= 2; ++input) {
        MapDimension(
            outDefs->inputTensorAccessPattern[input],
            0,
            0,
            0,
            0,
            kTopK - 1);
        MapDimension(
            outDefs->inputTensorAccessPattern[input], 1, 0, 0, 0, 0);
    }
    for (unsigned input = 3; input < 5; ++input) {
        const int columns = input == 3 ? kW2PackedCols : kW2ScaleCols;
        MapDimension(
            outDefs->inputTensorAccessPattern[input],
            0,
            0,
            0,
            0,
            columns - 1);
        MapDimension(
            outDefs->inputTensorAccessPattern[input], 1, 0, 1, 0, 0);
        MapDimension(
            outDefs->inputTensorAccessPattern[input],
            2,
            0,
            0,
            0,
            kExperts - 1);
    }
    MapDimension(
        outDefs->outputTensorAccessPattern[0], 0, 0, 1, 0, 0);
    MapDimension(
        outDefs->outputTensorAccessPattern[0], 1, 0, 0, 0, 0);
    outDefs->kernel.paramsNr = 0;
    return CopyElf(
        &_binary___deepseek_v4_mxfp4_indexed_fc2_gaudi2_o_start,
        &_binary___deepseek_v4_mxfp4_indexed_fc2_gaudi2_o_end,
        outDefs);
}
