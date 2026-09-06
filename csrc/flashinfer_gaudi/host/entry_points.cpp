// SPDX-License-Identifier: Apache-2.0
/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

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
    *kernelCount = 4;
    if (guids == nullptr || capacity == 0) return tpc_lib_api::GLUE_SUCCESS;
    if (capacity < 4) return tpc_lib_api::GLUE_FAILED;
    if (guids != nullptr) {
        std::memset(guids, 0, 4 * sizeof(tpc_lib_api::GuidInfo));
        GdnPackedDecodeF32Gaudi2 kernel;
        kernel.GetKernelName(guids[0].name);
        std::strcpy(guids[1].name, SiluAndMulBf16Gaudi2::name);
        std::strcpy(guids[2].name, SiluMulQuantBf16Gaudi2::name);
        std::strcpy(guids[3].name, BlockFp8DequantGaudi2::name);
    }
    if (kernelCount != nullptr) {
        *kernelCount = 4;
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
