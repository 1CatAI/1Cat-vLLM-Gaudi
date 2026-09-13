// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include "tp2_dynamic_quant_bf16_gaudi2.hpp"
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }

extern "C" tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId device, uint32_t* count, tpc_lib_api::GuidInfo* guids) {
    using namespace tpc_lib_api;
    if (!count) return GLUE_FAILED;
    const auto capacity = *count;
    *count = device == DEVICE_ID_GAUDI2 ? 1 : 0;
    if (!*count || !guids || !capacity) return GLUE_SUCCESS;
    std::memset(guids, 0, sizeof(*guids));
    std::strcpy(guids[0].name, Tp2DynamicQuantBf16Gaudi2::name);
    return GLUE_SUCCESS;
}

extern "C" tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance) {
    using namespace tpc_lib_api;
    if (!params || !instance) return GLUE_FAILED;
    if (params->deviceId != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    if (std::strcmp(params->guid.name, Tp2DynamicQuantBf16Gaudi2::name)) return GLUE_NODE_NOT_FOUND;
    return Tp2DynamicQuantBf16Gaudi2{}.GetGcDefinitions(params, instance);
}

extern "C" tpc_lib_api::GlueCodeReturn GetShapeInference(
    tpc_lib_api::DeviceId device, const tpc_lib_api::ShapeInferenceParams* params,
    tpc_lib_api::ShapeInferenceOutput* output) {
    using namespace tpc_lib_api;
    if (!params || !output || !params->inputTensors || !output->outputTensors) return GLUE_SIF_NULL_PTR;
    if (device != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    if (params->inputTensorsNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorsNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors[0] || !output->outputTensors[0] || !output->outputTensors[1]) return GLUE_SIF_NULL_PTR;
    const auto& input = params->inputTensors[0]->geometry;
    if (input.dims != 2 || input.maxSizes[1] != 1 || input.maxSizes[0] < 256 ||
        input.maxSizes[0] > 17408 || input.maxSizes[0] % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (unsigned n = 0; n < 2; ++n) {
        output->outputTensors[n]->geometry.dims = 2;
        output->outputTensors[n]->geometry.maxSizes[0] = n == 0 ? input.maxSizes[0] : 1;
        output->outputTensors[n]->geometry.maxSizes[1] = 1;
    }
    return GLUE_SUCCESS;
}

// This row kernel has a fixed contiguous layout and offers no optional
// transpose/reshape optimization. Older compiler adapters call this optional
// entry unconditionally; explicitly report that no manipulation is supported.
extern "C" tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(
    const tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::_TensorManipulationSuggestion* suggestion) {
    if (!params || !suggestion) return tpc_lib_api::GLUE_FAILED;
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
}
