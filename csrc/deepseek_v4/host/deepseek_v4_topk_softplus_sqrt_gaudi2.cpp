/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <array>
#include <cstring>
#include <limits>

#include "deepseek_v4_topk_softplus_sqrt_gaudi2.hpp"

extern unsigned char
    _binary___deepseek_v4_topk_softplus_sqrt_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v4_topk_softplus_sqrt_gaudi2_o_end;

namespace {

constexpr uint64_t kExperts = 256;
constexpr uint64_t kTopK = 6;
constexpr uint64_t kLutEntries = 32;
constexpr uint64_t kLutBlockFloats = 32;
constexpr uint64_t kLutValuesPerBlock = 10;
constexpr uint64_t kLutBlocks = 4;

const float kScoreCoefficients[kLutEntries][3] = {
    {0.0207520844f, 0.0103904944f, 0.00259473834f},
    {0.0266443435f, 0.0133388287f, 0.00332959756f},
    {0.0342080774f, 0.0171214077f, 0.00427082095f},
    {0.0439157121f, 0.0219716616f, 0.00547440377f},
    {0.0563712778f, 0.0281854184f, 0.00700935286f},
    {0.0723449339f, 0.0361343294f, 0.00895822809f},
    {0.0928141718f, 0.0462784627f, 0.011414502f},
    {0.119010242f, 0.0591730098f, 0.014472557f},
    {0.152464553f, 0.0754581135f, 0.0182023326f},
    {0.195041914f, 0.0958101471f, 0.0225950583f},
    {0.248932901f, 0.120816616f, 0.0274620231f},
    {0.316554318f, 0.150721398f, 0.0322773078f},
    {0.4002804f, 0.184999648f, 0.0360087857f},
    {0.501925883f, 0.221826208f, 0.0371145225f},
    {0.621990139f, 0.25775697f, 0.0340218023f},
    {0.758907137f, 0.288201371f, 0.0261975351f},
    {0.908812134f, 0.308985848f, 0.0150713332f},
    {1.06624124f, 0.318215834f, 0.00356705723f},
    {1.22553175f, 0.316932365f, -0.00565402906f},
    {1.38210822f, 0.308100041f, -0.011493241f},
    {1.53303809f, 0.295011156f, -0.0142880276f},
    {1.6768921f, 0.280255809f, -0.0149777214f},
    {1.81329566f, 0.265471189f, -0.0144653779f},
    {1.94248433f, 0.251519339f, -0.0133872917f},
    {2.0649851f, 0.238760354f, -0.0121209556f},
    {2.18142489f, 0.227273343f, -0.0108633689f},
    {2.29242969f, 0.216998915f, -0.0097038854f},
    {2.39857829f, 0.207820276f, -0.00867375872f},
    {2.5003857f, 0.199605384f, -0.00777548702f},
    {2.59830142f, 0.192226926f, -0.00699884279f},
    {2.69271424f, 0.185570484f, -0.00632910619f},
    {2.78395954f, 0.179536823f, -0.00575102617f},
};

const std::array<float, kLutBlocks * kLutBlockFloats>& GetScoreLut()
{
    static const auto lut = []() {
        std::array<float, kLutBlocks * kLutBlockFloats> values;
        values.fill(std::numeric_limits<float>::quiet_NaN());
        for (uint64_t index = 0; index < kLutEntries; ++index) {
            const uint64_t block = index / kLutValuesPerBlock;
            const uint64_t slot = index % kLutValuesPerBlock;
            const uint64_t offset =
                block * kLutBlockFloats + slot * 3;
            for (uint64_t coefficient = 0; coefficient < 3; ++coefficient) {
                values[offset + coefficient] =
                    kScoreCoefficients[index][coefficient];
            }
        }
        return values;
    }();
    return lut;
}

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

void SetOutputShape(tpc_lib_api::Tensor& tensor, uint64_t tokens)
{
    tensor.geometry.dims = 3;
    tensor.geometry.maxSizes[0] = 1;
    tensor.geometry.maxSizes[1] = kTopK;
    tensor.geometry.maxSizes[2] = tokens;
}

}  // namespace

tpc_lib_api::GlueCodeReturn
DeepseekV4TopkSoftplusSqrtGaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(
        kernelName,
        "custom_deepseek_v4_topk_softplus_sqrt_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
DeepseekV4TopkSoftplusSqrtGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* inDefs,
    tpc_lib_api::HabanaKernelInstantiation* outDefs)
{
    constexpr unsigned kInputCount = 2;
    constexpr unsigned kOutputCount = 2;
    if (inDefs->inputTensorNr != kInputCount) {
        inDefs->inputTensorNr = kInputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (inDefs->outputTensorNr != kOutputCount) {
        inDefs->outputTensorNr = kOutputCount;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (!HasDataType(inDefs->inputTensors[0], tpc_lib_api::DATA_F32) ||
        !HasDataType(inDefs->inputTensors[1], tpc_lib_api::DATA_F32) ||
        !HasDataType(inDefs->outputTensors[0], tpc_lib_api::DATA_F32) ||
        !HasDataType(inDefs->outputTensors[1], tpc_lib_api::DATA_I32)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    const uint64_t tokens =
        inDefs->inputTensors[0].geometry.maxSizes[1];
    if (tokens == 0 ||
        !HasShape2(inDefs->inputTensors[0], kExperts, tokens) ||
        !HasShape1(inDefs->inputTensors[1], kExperts)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (!HasShape3(inDefs->outputTensors[0], 1, kTopK, tokens) ||
        !HasShape3(inDefs->outputTensors[1], 1, kTopK, tokens)) {
        SetOutputShape(inDefs->outputTensors[0], tokens);
        SetOutputShape(inDefs->outputTensors[1], tokens);
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    outDefs->indexSpaceRank = 1;
    outDefs->indexSpaceGeometry[0] = tokens;
    MapDimension(
        outDefs->inputTensorAccessPattern[0], 0, 0, 0, 0, kExperts - 1);
    MapDimension(
        outDefs->inputTensorAccessPattern[0], 1, 0, 1, 0, 0);
    MapDimension(
        outDefs->inputTensorAccessPattern[1], 0, 0, 0, 0, kExperts - 1);
    for (unsigned output = 0; output < kOutputCount; ++output) {
        MapDimension(
            outDefs->outputTensorAccessPattern[output], 0, 0, 0, 0, 0);
        MapDimension(
            outDefs->outputTensorAccessPattern[output],
            1,
            0,
            0,
            0,
            kTopK - 1);
        MapDimension(
            outDefs->outputTensorAccessPattern[output], 2, 0, 1, 0, 0);
    }

    const auto& scoreLut = GetScoreLut();
    outDefs->auxiliaryTensorNr = 1;
    auto& auxiliary = outDefs->auxiliaryTensors[0];
    auxiliary.geometry.dims = 1;
    auxiliary.geometry.maxSizes[0] = scoreLut.size();
    auxiliary.geometry.dataType = tpc_lib_api::DATA_F32;
    const unsigned auxiliarySize = scoreLut.size() * sizeof(float);
    if (auxiliary.bufferSize < auxiliarySize) {
        auxiliary.bufferSize = auxiliarySize;
        return tpc_lib_api::GLUE_INSUFFICIENT_AUX_BUFFER_SIZE;
    }
    std::memcpy(auxiliary.pData, scoreLut.data(), auxiliarySize);

    outDefs->kernel.paramsNr = 0;
    const unsigned isaSize =
        &_binary___deepseek_v4_topk_softplus_sqrt_gaudi2_o_end -
        &_binary___deepseek_v4_topk_softplus_sqrt_gaudi2_o_start;
    const unsigned providedSize = outDefs->kernel.elfSize;
    outDefs->kernel.elfSize = isaSize;
    if (providedSize < isaSize) {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    std::memcpy(
        outDefs->kernel.kernelElf,
        &_binary___deepseek_v4_topk_softplus_sqrt_gaudi2_o_start,
        isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
