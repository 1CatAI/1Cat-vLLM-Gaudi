// SPDX-License-Identifier: Apache-2.0
/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "add_rmsnorm_quant_bf16_gaudi2.hpp"
#include "dflash2_score_select_i64_bf16_f32_gaudi2.hpp"
#include "dflash2_select_path_i64_f32_gaudi2.hpp"
#include "gdn_mtp_packed_f32_gaudi2.hpp"
#include "gdn_mtp_prepared_f32_gaudi2.hpp"
#include "gdn_packed_decode_f32_gaudi2.hpp"
#include "silu_and_mul_bf16_gaudi2.hpp"
#include "silu_mul_quant_bf16_gaudi2.hpp"
#include "block_fp8_dequant_gaudi2.hpp"

extern "C" {

tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId deviceId,
    uint32_t* kernelCount,
    tpc_lib_api::GuidInfo* guids)
{
    if (kernelCount == nullptr) return tpc_lib_api::GLUE_FAILED;
    if (deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
        if (kernelCount != nullptr) {
            *kernelCount = 0;
        }
        return tpc_lib_api::GLUE_SUCCESS;
    }
    const uint32_t capacity = *kernelCount;
    *kernelCount = 9;
    if (guids == nullptr || capacity == 0) return tpc_lib_api::GLUE_SUCCESS;
    if (capacity < 9) return tpc_lib_api::GLUE_FAILED;
    if (guids != nullptr) {
        std::memset(guids, 0, 9 * sizeof(tpc_lib_api::GuidInfo));
        GdnPackedDecodeF32Gaudi2 decodeKernel;
        GdnMtpPackedF32Gaudi2 mtpKernel;
        GdnMtpPreparedF32Gaudi2 preparedMtpKernel;
        DFlash2SelectPathI64F32Gaudi2 selectKernel;
        DFlash2ScoreSelectI64Bf16F32Gaudi2 scoreSelectKernel;
        decodeKernel.GetKernelName(guids[0].name);
        mtpKernel.GetKernelName(guids[1].name);
        selectKernel.GetKernelName(guids[2].name);
        scoreSelectKernel.GetKernelName(guids[3].name);
        preparedMtpKernel.GetKernelName(guids[4].name);
        std::strcpy(guids[5].name, SiluAndMulBf16Gaudi2::name);
        std::strcpy(guids[6].name, SiluMulQuantBf16Gaudi2::name);
        std::strcpy(guids[7].name, BlockFp8DequantGaudi2::name);
        std::strcpy(guids[8].name, AddRmsNormQuantBf16Gaudi2::name);
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance)
{
    if (!params || !instance) return tpc_lib_api::GLUE_FAILED;
    char kernelName[tpc_lib_api::MAX_NODE_NAME];
    GdnPackedDecodeF32Gaudi2 kernel;
    kernel.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0) {
        return kernel.GetGcDefinitions(params, instance);
    }
    if (std::strcmp(params->guid.name, SiluAndMulBf16Gaudi2::name) == 0) {
        return SiluAndMulBf16Gaudi2{}.GetGcDefinitions(params, instance);
    }
    if (std::strcmp(params->guid.name, SiluMulQuantBf16Gaudi2::name) == 0) {
        return SiluMulQuantBf16Gaudi2{}.GetGcDefinitions(params, instance);
    }
    if (std::strcmp(params->guid.name, BlockFp8DequantGaudi2::name) == 0) {
        return BlockFp8DequantGaudi2{}.GetGcDefinitions(params, instance);
    }
    GdnMtpPackedF32Gaudi2 mtpKernel;
    mtpKernel.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0) {
        return mtpKernel.GetGcDefinitions(params, instance);
    }
    DFlash2SelectPathI64F32Gaudi2 selectKernel;
    selectKernel.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0) {
        return selectKernel.GetGcDefinitions(params, instance);
    }
    GdnMtpPreparedF32Gaudi2 preparedMtpKernel;
    preparedMtpKernel.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0) {
        return preparedMtpKernel.GetGcDefinitions(params, instance);
    }
    DFlash2ScoreSelectI64Bf16F32Gaudi2 scoreSelectKernel;
    scoreSelectKernel.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0) {
        return scoreSelectKernel.GetGcDefinitions(params, instance);
    }
    if (std::strcmp(params->guid.name, AddRmsNormQuantBf16Gaudi2::name) == 0) {
        return AddRmsNormQuantBf16Gaudi2{}.GetGcDefinitions(params, instance);
    }
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(
    tpc_lib_api::DeviceId,
    tpc_lib_api::ShapeInferenceParams*,
    tpc_lib_api::ShapeInferenceOutput*)
{
    return tpc_lib_api::GLUE_SUCCESS;
}

}  // extern "C"
