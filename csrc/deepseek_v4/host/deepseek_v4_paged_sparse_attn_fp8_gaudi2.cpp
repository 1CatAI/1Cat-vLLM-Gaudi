/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "deepseek_v4_paged_sparse_attn_fp8_gaudi2.hpp"

extern unsigned char
    _binary___deepseek_v4_paged_sparse_attn_fp8_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_paged_sparse_attn_fp8_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_paged_sparse_attn_local_fp8_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_paged_sparse_attn_local_fp8_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_paged_sparse_attn_sequential_fp8_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_paged_sparse_attn_sequential_fp8_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_paged_sparse_attn_pair_sequential_fp8_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_paged_sparse_attn_pair_sequential_fp8_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_qnorm_paged_sparse_attn_sequential_fp8_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_qnorm_paged_sparse_attn_sequential_fp8_gaudi2_o_end;
extern unsigned char
    _binary___deepseek_v4_paged_swa_attn_fp8_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_paged_swa_attn_fp8_gaudi2_o_end;

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

tpc_lib_api::GlueCodeReturn
DeepseekV4PagedSparseAttnFP8Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    const char* name = "custom_deepseek_v4_paged_sparse_attn_fp8_gaudi2";
    if (mode_ == LOCAL_BLOCK_TABLE) {
        name = "custom_deepseek_v4_paged_sparse_attn_local_fp8_gaudi2";
    } else if (mode_ == SEQUENTIAL_BLOCK_TABLE) {
        name = "custom_deepseek_v4_paged_sparse_attn_sequential_fp8_gaudi2";
    } else if (mode_ == PAIR_SEQUENTIAL_BLOCK_TABLE) {
        name =
            "custom_deepseek_v4_paged_sparse_attn_pair_sequential_fp8_gaudi2";
    } else if (mode_ == QNORM_SEQUENTIAL_BLOCK_TABLE) {
        name =
            "custom_deepseek_v4_qnorm_paged_sparse_attn_seq_fp8_gaudi2";
    } else if (mode_ == SWA_ONLY) {
        name = "custom_deepseek_v4_paged_swa_attn_fp8_gaudi2";
    }
    std::strcpy(kernelName, name);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4PagedSparseAttnFP8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    const bool fusedQnorm = mode_ == QNORM_SEQUENTIAL_BLOCK_TABLE;
    const bool pairHeads =
        mode_ == PAIR_SEQUENTIAL_BLOCK_TABLE || mode_ == SWA_ONLY;
    const bool localTopk = mode_ != GLOBAL_SLOTS;
    const bool sequentialTopk =
        mode_ == SEQUENTIAL_BLOCK_TABLE || pairHeads || fusedQnorm;
    const unsigned inputShift = fusedQnorm ? 2 : 0;
    const unsigned tailShift = fusedQnorm ? 1 : 0;
    const unsigned kInputCount = fusedQnorm
        ? 15
        : (sequentialTopk ? 14 : (localTopk ? 13 : 11));
    constexpr unsigned kOutputCount = 1;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    const tpc_lib_api::TensorDataType globalInputTypes[11] = {
        tpc_lib_api::DATA_BF16,
        tpc_lib_api::DATA_U8,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_U8,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_F32,
        tpc_lib_api::DATA_BF16,
    };
    const tpc_lib_api::TensorDataType localInputTypes[13] = {
        tpc_lib_api::DATA_BF16,
        tpc_lib_api::DATA_U8,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_U8,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_F32,
        tpc_lib_api::DATA_BF16,
    };
    const tpc_lib_api::TensorDataType sequentialInputTypes[14] = {
        tpc_lib_api::DATA_BF16,
        tpc_lib_api::DATA_U8,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_U8,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_F32,
        tpc_lib_api::DATA_BF16,
    };
    const tpc_lib_api::TensorDataType fusedSequentialInputTypes[15] = {
        tpc_lib_api::DATA_BF16,
        tpc_lib_api::DATA_I64,
        tpc_lib_api::DATA_F32,
        tpc_lib_api::DATA_U8,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_U8,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_I32,
        tpc_lib_api::DATA_F32,
        tpc_lib_api::DATA_BF16,
    };
    const auto* inputTypes = fusedQnorm
        ? fusedSequentialInputTypes
        : sequentialTopk
        ? sequentialInputTypes
        : (localTopk ? localInputTypes : globalInputTypes);
    for (unsigned input = 0; input < kInputCount; ++input) {
        if (!HasDataType(inDefs->inputTensors[input], inputTypes[input])) {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    if (!HasDataType(inDefs->outputTensors[0], tpc_lib_api::DATA_F32)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const auto& q = inDefs->inputTensors[0];
    const unsigned compressedStorageIndex = 1 + inputShift;
    const unsigned compressedGeometryIndex = 2 + inputShift;
    const unsigned topkIndicesIndex = 3 + inputShift;
    const auto& compressedStorage =
        inDefs->inputTensors[compressedStorageIndex];
    const auto& compressedGeometry =
        inDefs->inputTensors[compressedGeometryIndex];
    const auto& topkIndices = inDefs->inputTensors[topkIndicesIndex];
    const unsigned swaStorageIndex =
        (sequentialTopk ? 8 : (localTopk ? 7 : 5)) + tailShift;
    const unsigned swaGeometryIndex =
        (sequentialTopk ? 9 : (localTopk ? 8 : 6)) + tailShift;
    const unsigned swaIndicesIndex =
        (sequentialTopk ? 10 : (localTopk ? 9 : 7)) + tailShift;
    const unsigned swaLensIndex =
        (sequentialTopk ? 11 : (localTopk ? 10 : 8)) + tailShift;
    const unsigned sinkIndex =
        (sequentialTopk ? 12 : (localTopk ? 11 : 9)) + tailShift;
    const unsigned outputIndex =
        (sequentialTopk ? 13 : (localTopk ? 12 : 10)) + tailShift;
    const unsigned tokenToReqIndex = 4;
    const unsigned blockTableIndex = fusedQnorm ? 6 : 5;
    const unsigned validTokenIndex = fusedQnorm ? 7 : 6;
    const unsigned seqLensIndex = fusedQnorm ? 8 : 7;
    const auto& swaStorage = inDefs->inputTensors[swaStorageIndex];
    const auto& swaGeometry = inDefs->inputTensors[swaGeometryIndex];
    const auto& swaIndices = inDefs->inputTensors[swaIndicesIndex];
    const auto& swaLens = inDefs->inputTensors[swaLensIndex];
    const auto& sink = inDefs->inputTensors[sinkIndex];
    const auto& output = inDefs->inputTensors[outputIndex];
    auto& scoreDebug = inDefs->outputTensors[0];

    bool ranksMatch =
        q.geometry.dims == 3 && compressedStorage.geometry.dims == 1 &&
        compressedGeometry.geometry.dims == 1 &&
        topkIndices.geometry.dims == 2 &&
        swaStorage.geometry.dims == 1 && swaGeometry.geometry.dims == 1 &&
        swaIndices.geometry.dims == 2 && swaLens.geometry.dims == 1 &&
        sink.geometry.dims == 1 && output.geometry.dims == 3 &&
        scoreDebug.geometry.dims == 2;
    if (fusedQnorm) {
        ranksMatch = ranksMatch &&
            inDefs->inputTensors[1].geometry.dims == 1 &&
            inDefs->inputTensors[2].geometry.dims == 2;
    }
    if (localTopk) {
        ranksMatch = ranksMatch &&
            inDefs->inputTensors[blockTableIndex].geometry.dims == 2 &&
            inDefs->inputTensors[validTokenIndex].geometry.dims == 1;
        if (!fusedQnorm) {
            ranksMatch = ranksMatch &&
                inDefs->inputTensors[tokenToReqIndex].geometry.dims == 1;
        }
        if (sequentialTopk) {
            ranksMatch = ranksMatch &&
                inDefs->inputTensors[seqLensIndex].geometry.dims == 1;
        }
    } else {
        ranksMatch = ranksMatch &&
            inDefs->inputTensors[4].geometry.dims == 1;
    }
    if (!ranksMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const uint64_t queryHeadCount = q.geometry.maxSizes[1];
    const uint64_t headCount = queryHeadCount;
    const uint64_t outputHeadCount = output.geometry.maxSizes[1];
    const uint64_t batchSize = q.geometry.maxSizes[2];
    const uint64_t topkWidth = topkIndices.geometry.maxSizes[0];
    const uint64_t swaWidth = swaIndices.geometry.maxSizes[0];
    bool inputsMatch =
        q.geometry.maxSizes[0] == kHeadDim && headCount > 0 &&
        outputHeadCount >= headCount &&
        batchSize > 0 &&
        compressedStorage.geometry.maxSizes[0] >= kBytesPerToken &&
        compressedGeometry.geometry.maxSizes[0] == 4 &&
        topkWidth > 0 && topkIndices.geometry.maxSizes[1] == batchSize &&
        swaStorage.geometry.maxSizes[0] >= kBytesPerToken &&
        swaGeometry.geometry.maxSizes[0] == 4 && swaWidth > 0 &&
        swaIndices.geometry.maxSizes[1] == batchSize &&
        swaLens.geometry.maxSizes[0] == batchSize &&
        sink.geometry.maxSizes[0] >= headCount &&
        output.geometry.maxSizes[0] == kHeadDim &&
        output.geometry.maxSizes[1] == outputHeadCount &&
        output.geometry.maxSizes[2] == batchSize;
    if (localTopk) {
        inputsMatch = inputsMatch &&
            inDefs->inputTensors[blockTableIndex].geometry.maxSizes[0] > 0 &&
            inDefs->inputTensors[blockTableIndex].geometry.maxSizes[1] > 0 &&
            inDefs->inputTensors[validTokenIndex].geometry.maxSizes[0] ==
                batchSize;
        if (!fusedQnorm) {
            inputsMatch = inputsMatch &&
                inDefs->inputTensors[tokenToReqIndex]
                    .geometry.maxSizes[0] == batchSize;
        }
        if (sequentialTopk) {
            inputsMatch = inputsMatch &&
                inDefs->inputTensors[seqLensIndex].geometry.maxSizes[0] > 0;
        }
    } else {
        inputsMatch = inputsMatch &&
            inDefs->inputTensors[4].geometry.maxSizes[0] == batchSize;
    }
    if (fusedQnorm) {
        inputsMatch = inputsMatch &&
            inDefs->inputTensors[1].geometry.maxSizes[0] >= batchSize &&
            inDefs->inputTensors[2].geometry.maxSizes[0] == 64 &&
            inDefs->inputTensors[2].geometry.maxSizes[1] > 0;
    }
    if (!inputsMatch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    if (scoreDebug.geometry.dims != 2 ||
        scoreDebug.geometry.maxSizes[0] != headCount ||
        scoreDebug.geometry.maxSizes[1] != batchSize) {
        scoreDebug.geometry.dims = 2;
        scoreDebug.geometry.maxSizes[0] = headCount;
        scoreDebug.geometry.maxSizes[1] = batchSize;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 3;
    outDefs->indexSpaceGeometry[0] = 1;
    const unsigned headsPerProgram = pairHeads ? 2 : 1;
    outDefs->indexSpaceGeometry[1] =
        (headCount + headsPerProgram - 1) / headsPerProgram;
    outDefs->indexSpaceGeometry[2] = batchSize;

    MapDimension(outDefs->inputTensorAccessPattern[0], 0, 0, 512, 0, 511);
    MapDimension(
        outDefs->inputTensorAccessPattern[0],
        1,
        1,
        headsPerProgram,
        0,
        headsPerProgram - 1);
    MapDimension(outDefs->inputTensorAccessPattern[0], 2, 2, 1, 0, 0);
    if (fusedQnorm) {
        MapDimension(outDefs->inputTensorAccessPattern[1], 0, 2, 1, 0, 0);
        MapDimension(outDefs->inputTensorAccessPattern[2], 0, 0, 0, 0, 63);
        const auto cacheRows =
            inDefs->inputTensors[2].geometry.maxSizes[1];
        MapDimension(
            outDefs->inputTensorAccessPattern[2],
            1,
            0,
            0,
            0,
            static_cast<int>(cacheRows - 1));
    }

    const unsigned storageInputs[2] = {
        compressedStorageIndex, swaStorageIndex};
    for (unsigned storageInput : storageInputs) {
        const auto storageBytes =
            inDefs->inputTensors[storageInput].geometry.maxSizes[0];
        MapDimension(
            outDefs->inputTensorAccessPattern[storageInput],
            0,
            0,
            0,
            0,
            static_cast<int>(storageBytes - 1));
    }
    MapDimension(
        outDefs->inputTensorAccessPattern[compressedGeometryIndex],
        0,
        0,
        0,
        0,
        3);
    MapDimension(
        outDefs->inputTensorAccessPattern[swaGeometryIndex],
        0,
        0,
        0,
        0,
        3);

    const unsigned indexInputs[2] = {
        topkIndicesIndex, swaIndicesIndex};
    for (unsigned indexInput : indexInputs) {
        const auto width =
            inDefs->inputTensors[indexInput].geometry.maxSizes[0];
        MapDimension(
            outDefs->inputTensorAccessPattern[indexInput],
            0,
            0,
            0,
            0,
            static_cast<int>(width - 1));
        MapDimension(
            outDefs->inputTensorAccessPattern[indexInput],
            1,
            2,
            1,
            0,
            0);
    }
    if (localTopk) {
        if (!fusedQnorm) {
            MapDimension(
                outDefs->inputTensorAccessPattern[tokenToReqIndex],
                0,
                2,
                1,
                0,
                0);
        }
        const auto blockWidth =
            inDefs->inputTensors[blockTableIndex].geometry.maxSizes[0];
        const auto blockRows =
            inDefs->inputTensors[blockTableIndex].geometry.maxSizes[1];
        MapDimension(
            outDefs->inputTensorAccessPattern[blockTableIndex],
            0,
            0,
            0,
            0,
            static_cast<int>(blockWidth - 1));
        MapDimension(
            outDefs->inputTensorAccessPattern[blockTableIndex],
            1,
            0,
            0,
            0,
            static_cast<int>(blockRows - 1));
        MapDimension(
            outDefs->inputTensorAccessPattern[validTokenIndex],
            0,
            2,
            1,
            0,
            0);
        if (sequentialTopk) {
            const auto seqRows =
                inDefs->inputTensors[seqLensIndex].geometry.maxSizes[0];
            MapDimension(
                outDefs->inputTensorAccessPattern[seqLensIndex],
                0,
                0,
                0,
                0,
                static_cast<int>(seqRows - 1));
        }
    } else {
        MapDimension(outDefs->inputTensorAccessPattern[4], 0, 2, 1, 0, 0);
    }
    MapDimension(
        outDefs->inputTensorAccessPattern[swaLensIndex], 0, 2, 1, 0, 0);
    MapDimension(
        outDefs->inputTensorAccessPattern[sinkIndex],
        0,
        1,
        headsPerProgram,
        0,
        headsPerProgram - 1);
    MapDimension(
        outDefs->inputTensorAccessPattern[outputIndex], 0, 0, 512, 0, 511);
    MapDimension(
        outDefs->inputTensorAccessPattern[outputIndex],
        1,
        1,
        headsPerProgram,
        0,
        headsPerProgram - 1);
    MapDimension(
        outDefs->inputTensorAccessPattern[outputIndex], 2, 2, 1, 0, 0);
    MapDimension(
        outDefs->outputTensorAccessPattern[0],
        0,
        1,
        headsPerProgram,
        0,
        headsPerProgram - 1);
    MapDimension(
        outDefs->outputTensorAccessPattern[0], 1, 2, 1, 0, 0);

    outDefs->kernel.paramsNr = 0;
    const unsigned char* isaStart =
        &_binary___deepseek_v4_paged_sparse_attn_fp8_gaudi2_o_start;
    const unsigned char* isaEnd =
        &_binary___deepseek_v4_paged_sparse_attn_fp8_gaudi2_o_end;
    if (mode_ == LOCAL_BLOCK_TABLE) {
        isaStart =
            &_binary___deepseek_v4_paged_sparse_attn_local_fp8_gaudi2_o_start;
        isaEnd =
            &_binary___deepseek_v4_paged_sparse_attn_local_fp8_gaudi2_o_end;
    } else if (mode_ == SEQUENTIAL_BLOCK_TABLE) {
        isaStart =
            &_binary___deepseek_v4_paged_sparse_attn_sequential_fp8_gaudi2_o_start;
        isaEnd =
            &_binary___deepseek_v4_paged_sparse_attn_sequential_fp8_gaudi2_o_end;
    } else if (mode_ == PAIR_SEQUENTIAL_BLOCK_TABLE) {
        isaStart =
            &_binary___deepseek_v4_paged_sparse_attn_pair_sequential_fp8_gaudi2_o_start;
        isaEnd =
            &_binary___deepseek_v4_paged_sparse_attn_pair_sequential_fp8_gaudi2_o_end;
    } else if (mode_ == QNORM_SEQUENTIAL_BLOCK_TABLE) {
        isaStart =
            &_binary___deepseek_v4_qnorm_paged_sparse_attn_sequential_fp8_gaudi2_o_start;
        isaEnd =
            &_binary___deepseek_v4_qnorm_paged_sparse_attn_sequential_fp8_gaudi2_o_end;
    } else if (mode_ == SWA_ONLY) {
        isaStart =
            &_binary___deepseek_v4_paged_swa_attn_fp8_gaudi2_o_start;
        isaEnd =
            &_binary___deepseek_v4_paged_swa_attn_fp8_gaudi2_o_end;
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
