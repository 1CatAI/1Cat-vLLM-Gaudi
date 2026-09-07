/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "deepseek_v4_dequant_gather_bf16_gaudi2.hpp"

extern unsigned char
    _binary___deepseek_v4_dequant_gather_bf16_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_dequant_gather_bf16_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_dual_dequant_gather_bf16_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_dual_dequant_gather_bf16_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_local_dual_dequant_gather_bf16_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_local_dual_dequant_gather_bf16_gaudi2_o_end;

namespace {

constexpr uint64_t kHeadDim = 512;
constexpr uint64_t kBytesPerToken = 584;

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

tpc_lib_api::GlueCodeReturn DeepseekV4DequantGatherBF16Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    const char* name = "custom_deepseek_v4_dequant_gather_bf16_gaudi2";
    if (mode_ == DUAL) {
        name = "custom_deepseek_v4_dual_dequant_gather_bf16_gaudi2";
    } else if (mode_ == LOCAL_DUAL) {
        name =
            "custom_deepseek_v4_local_dual_dequant_gather_bf16_gaudi2";
    }
    std::strcpy(kernelName, name);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4DequantGatherBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    const bool dual = mode_ != SINGLE;
    const bool localDual = mode_ == LOCAL_DUAL;
    const unsigned kInputCount = localDual ? 10 : (dual ? 6 : 3);
    constexpr unsigned kOutputCount = 1;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    bool typesMatch =
        HasDataType(inDefs->inputTensors[0], tpc_lib_api::DATA_U8) &&
        HasDataType(inDefs->inputTensors[1], tpc_lib_api::DATA_I32) &&
        HasDataType(inDefs->inputTensors[2], tpc_lib_api::DATA_I32);
    if (localDual) {
        typesMatch =
            typesMatch &&
            HasDataType(inDefs->inputTensors[3], tpc_lib_api::DATA_I32) &&
            HasDataType(inDefs->inputTensors[4], tpc_lib_api::DATA_I32) &&
            HasDataType(inDefs->inputTensors[5], tpc_lib_api::DATA_I32) &&
            HasDataType(inDefs->inputTensors[6], tpc_lib_api::DATA_I32) &&
            HasDataType(inDefs->inputTensors[7], tpc_lib_api::DATA_U8) &&
            HasDataType(inDefs->inputTensors[8], tpc_lib_api::DATA_I32) &&
            HasDataType(inDefs->inputTensors[9], tpc_lib_api::DATA_I32);
    } else if (dual) {
        typesMatch =
            typesMatch &&
            HasDataType(inDefs->inputTensors[3], tpc_lib_api::DATA_U8) &&
            HasDataType(inDefs->inputTensors[4], tpc_lib_api::DATA_I32) &&
            HasDataType(inDefs->inputTensors[5], tpc_lib_api::DATA_I32);
    }
    if (!typesMatch ||
        !HasDataType(inDefs->outputTensors[0], tpc_lib_api::DATA_BF16)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const auto& cacheStorageU8 = inDefs->inputTensors[0];
    const auto& cacheGeometry = inDefs->inputTensors[1];
    const auto& indices = inDefs->inputTensors[2];
    auto& output = inDefs->outputTensors[0];
    bool dimensionsMatch =
        cacheStorageU8.geometry.dims == 1 &&
        cacheGeometry.geometry.dims == 1 && indices.geometry.dims == 1;
    const unsigned secondStorageIndex = localDual ? 7 : 3;
    const unsigned secondGeometryIndex = localDual ? 8 : 4;
    const unsigned secondIndicesIndex = localDual ? 9 : 5;
    if (dual) {
        dimensionsMatch =
            dimensionsMatch &&
            inDefs->inputTensors[secondStorageIndex].geometry.dims == 1 &&
            inDefs->inputTensors[secondGeometryIndex].geometry.dims == 1 &&
            inDefs->inputTensors[secondIndicesIndex].geometry.dims == 1;
    }
    if (localDual) {
        dimensionsMatch =
            dimensionsMatch && inDefs->inputTensors[3].geometry.dims == 1 &&
            inDefs->inputTensors[4].geometry.dims == 2 &&
            inDefs->inputTensors[5].geometry.dims == 1 &&
            inDefs->inputTensors[6].geometry.dims == 1;
    }
    if (!dimensionsMatch ||
        output.geometry.dims != 2) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const uint64_t storageBytes = cacheStorageU8.geometry.maxSizes[0];
    const uint64_t firstSlots = indices.geometry.maxSizes[0];
    bool inputsMatch =
        storageBytes >= kBytesPerToken &&
        cacheGeometry.geometry.maxSizes[0] == 4 && firstSlots > 0;
    uint64_t secondStorageBytes = 0;
    uint64_t secondSlots = 0;
    if (dual) {
        secondStorageBytes =
            inDefs->inputTensors[secondStorageIndex].geometry.maxSizes[0];
        secondSlots =
            inDefs->inputTensors[secondIndicesIndex].geometry.maxSizes[0];
        inputsMatch =
            inputsMatch && secondStorageBytes >= kBytesPerToken &&
            inDefs->inputTensors[secondGeometryIndex].geometry.maxSizes[0] ==
                4 &&
            secondSlots > 0;
    }
    if (localDual) {
        inputsMatch =
            inputsMatch && inDefs->inputTensors[3].geometry.maxSizes[0] > 0 &&
            inDefs->inputTensors[4].geometry.maxSizes[0] > 0 &&
            inDefs->inputTensors[4].geometry.maxSizes[1] > 0 &&
            inDefs->inputTensors[5].geometry.maxSizes[0] > 0 &&
            inDefs->inputTensors[6].geometry.maxSizes[0] > 0;
    }
    if (!inputsMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    const uint64_t outputSlots = firstSlots + secondSlots;

    const bool outputMatches =
        output.geometry.maxSizes[0] == kHeadDim &&
        output.geometry.maxSizes[1] == outputSlots;
    if (!outputMatches) {
        output.geometry.dims = 2;
        output.geometry.maxSizes[0] = kHeadDim;
        output.geometry.maxSizes[1] = outputSlots;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 2;
    outDefs->indexSpaceGeometry[0] = 1;
    outDefs->indexSpaceGeometry[1] = outputSlots;

    // Indirect cache access may touch any byte in the contiguous backing
    // storage. Layer offset and BLHNC block stride come from cacheGeometry.
    MapDimension(
        outDefs->inputTensorAccessPattern[0],
        0,
        0,
        0,
        0,
        static_cast<int>(storageBytes - 1));
    MapDimension(outDefs->inputTensorAccessPattern[1], 0, 0, 0, 0, 3);
    if (localDual) {
        MapDimension(
            outDefs->inputTensorAccessPattern[2],
            0,
            0,
            0,
            0,
            static_cast<int>(firstSlots - 1));
        MapDimension(
            outDefs->inputTensorAccessPattern[3],
            0,
            0,
            0,
            0,
            static_cast<int>(
                inDefs->inputTensors[3].geometry.maxSizes[0] - 1));
        MapDimension(
            outDefs->inputTensorAccessPattern[4],
            0,
            0,
            0,
            0,
            static_cast<int>(
                inDefs->inputTensors[4].geometry.maxSizes[0] - 1));
        MapDimension(
            outDefs->inputTensorAccessPattern[4],
            1,
            0,
            0,
            0,
            static_cast<int>(
                inDefs->inputTensors[4].geometry.maxSizes[1] - 1));
        MapDimension(
            outDefs->inputTensorAccessPattern[5],
            0,
            0,
            0,
            0,
            static_cast<int>(
                inDefs->inputTensors[5].geometry.maxSizes[0] - 1));
        MapDimension(
            outDefs->inputTensorAccessPattern[6],
            0,
            0,
            0,
            0,
            static_cast<int>(
                inDefs->inputTensors[6].geometry.maxSizes[0] - 1));
        MapDimension(
            outDefs->inputTensorAccessPattern[7],
            0,
            0,
            0,
            0,
            static_cast<int>(secondStorageBytes - 1));
        MapDimension(outDefs->inputTensorAccessPattern[8], 0, 0, 0, 0, 3);
        MapDimension(
            outDefs->inputTensorAccessPattern[9],
            0,
            0,
            0,
            0,
            static_cast<int>(secondSlots - 1));
    } else if (dual) {
        MapDimension(
            outDefs->inputTensorAccessPattern[2],
            0,
            0,
            0,
            0,
            static_cast<int>(firstSlots - 1));
        MapDimension(
            outDefs->inputTensorAccessPattern[3],
            0,
            0,
            0,
            0,
            static_cast<int>(secondStorageBytes - 1));
        MapDimension(outDefs->inputTensorAccessPattern[4], 0, 0, 0, 0, 3);
        MapDimension(
            outDefs->inputTensorAccessPattern[5],
            0,
            0,
            0,
            0,
            static_cast<int>(secondSlots - 1));
    } else {
        MapDimension(outDefs->inputTensorAccessPattern[2], 0, 1, 1, 0, 0);
    }
    MapDimension(outDefs->outputTensorAccessPattern[0], 0, 0, 512, 0, 511);
    MapDimension(outDefs->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);

    outDefs->kernel.paramsNr = 0;
    const unsigned char* isaStart =
        &_binary___deepseek_v4_dequant_gather_bf16_gaudi2_o_start;
    const unsigned char* isaEnd =
        &_binary___deepseek_v4_dequant_gather_bf16_gaudi2_o_end;
    if (mode_ == DUAL) {
        isaStart =
            &_binary___deepseek_v4_dual_dequant_gather_bf16_gaudi2_o_start;
        isaEnd =
            &_binary___deepseek_v4_dual_dequant_gather_bf16_gaudi2_o_end;
    } else if (mode_ == LOCAL_DUAL) {
        isaStart =
            &_binary___deepseek_v4_local_dual_dequant_gather_bf16_gaudi2_o_start;
        isaEnd =
            &_binary___deepseek_v4_local_dual_dequant_gather_bf16_gaudi2_o_end;
    }
    const unsigned isaSize = isaEnd - isaStart;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(
        outDefs->kernel.kernelElf,
        isaStart,
        isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
