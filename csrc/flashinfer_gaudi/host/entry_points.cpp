/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#include <cstring>

#include "gdn_packed_decode_f32_gaudi2.hpp"

extern "C" {

tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId deviceId,
    uint32_t* kernelCount,
    tpc_lib_api::GuidInfo* guids)
{
    if (deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
        if (kernelCount != nullptr) {
            *kernelCount = 0;
        }
        return tpc_lib_api::GLUE_SUCCESS;
    }
    if (guids != nullptr) {
        GdnPackedDecodeF32Gaudi2 kernel;
        kernel.GetKernelName(guids[0].name);
    }
    if (kernelCount != nullptr) {
        *kernelCount = 1;
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance)
{
    char kernelName[tpc_lib_api::MAX_NODE_NAME];
    GdnPackedDecodeF32Gaudi2 kernel;
    kernel.GetKernelName(kernelName);
    if (std::strcmp(params->guid.name, kernelName) == 0) {
        return kernel.GetGcDefinitions(params, instance);
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

