// SPDX-License-Identifier: Apache-2.0
// Add the unchanged C1 consumer without replacing inherited kernel GUIDs.
#include "deepseek_v41_mhc_post_collapse_gaudi2.hpp"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }
namespace {
constexpr auto name = "custom_deepseek_v41_peer_mhc_post_collapse_gaudi2";
template<class T> T parent(const char* symbol) {
    static void* handle = [] {
        const char* path = std::getenv("VLLM_HPU_DSV41_PEER_COLLAPSE_PARENT_KERNEL");
        return path && *path ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : nullptr;
    }();
    return handle ? reinterpret_cast<T>(dlsym(handle, symbol)) : nullptr;
}
}
extern "C" {
uint64_t GetLibVersion() {
    auto fn = parent<tpc_lib_api::pfnGetLibVersion>("GetLibVersion");
    return fn ? fn() : 0;
}
tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId device, uint32_t* count,
                                         tpc_lib_api::GuidInfo* out) {
    using namespace tpc_lib_api;
    auto fn = parent<pfnGetKernelGuids>("GetKernelGuids");
    if (!fn || !count) return GLUE_FAILED;
    if (device != DEVICE_ID_GAUDI2) return fn(device, count, out);
    uint32_t inherited = 0;
    auto status = fn(device, &inherited, nullptr);
    if (status != GLUE_SUCCESS) return status;
    const auto capacity = *count;
    *count = inherited + 1;
    if (!out || !capacity) return GLUE_SUCCESS;
    if (capacity < *count) return GLUE_FAILED;
    std::memset(out, 0, sizeof(*out));
    std::strcpy(out->name, name);
    return fn(device, &inherited, out + 1);
}
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(const tpc_lib_api::HabanaKernelParams* p,
                                               tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out) return GLUE_FAILED;
    if (!std::strcmp(p->guid.name, name)) {
        if (p->deviceId != DEVICE_ID_GAUDI2) return GLUE_FAILED;
        auto parameters = *p;
        DeepseekV41MhcPostCollapseGaudi2 kernel;
        return kernel.GetGcDefinitions(&parameters, out);
    }
    auto fn = parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel");
    return fn ? fn(p, out) : GLUE_FAILED;
}
tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId device,
    const tpc_lib_api::ShapeInferenceParams* p, tpc_lib_api::ShapeInferenceOutput* out) {
    if (!std::strcmp((p->pGuid ? p->pGuid : &p->guid)->name, name)) return tpc_lib_api::GLUE_SUCCESS;
    auto fn = parent<tpc_lib_api::pfnGetShapeInference>("GetShapeInference");
    return fn ? fn(device, p, out) : tpc_lib_api::GLUE_FAILED;
}
tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(const tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::NodeDataLayouts* layouts, uint32_t* count) {
    using namespace tpc_lib_api;
    if (std::strcmp(p->guid.name, name)) {
        auto fn = parent<pfnGetSupportedDataLayout>("GetSupportedDataLayouts");
        return fn ? fn(p, layouts, count) : GLUE_FAILED;
    }
    if (!count) return GLUE_FAILED;
    *count = 1;
    if (layouts) {
        for (unsigned i = 0; i < layouts->inputTensorNr; ++i)
            std::memset(layouts->inputs[i].layout, 'x', sizeof(layouts->inputs[i].layout));
        for (unsigned i = 0; i < layouts->outputTensorNr; ++i)
            std::memset(layouts->outputs[i].layout, 'x', sizeof(layouts->outputs[i].layout));
    }
    return GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(const tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::_TensorManipulationSuggestion* suggestion) {
    if (!std::strcmp(p->guid.name, name)) return tpc_lib_api::GLUE_FAILED;
    using Function = tpc_lib_api::GlueCodeReturn(*)(const tpc_lib_api::HabanaKernelParams*,
                                                  tpc_lib_api::_TensorManipulationSuggestion*);
    auto fn = parent<Function>("GetSuggestedManipulation");
    return fn ? fn(p, suggestion) : tpc_lib_api::GLUE_FAILED;
}
}
