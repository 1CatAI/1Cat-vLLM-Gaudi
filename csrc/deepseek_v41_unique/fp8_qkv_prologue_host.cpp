// SPDX-License-Identifier: Apache-2.0
#include "tpc_kernel_lib_interface.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
extern unsigned char _binary___deepseek_v41_fp8_qkv_prologue_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_fp8_qkv_prologue_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_fp8_q_prologue_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_fp8_q_prologue_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_fp8_kv_prologue_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_fp8_kv_prologue_gaudi2_o_end;
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }
namespace {
constexpr const char* names[] = {"custom_deepseek_v41_fp8_qkv_prologue_gaudi2",
    "custom_deepseek_v41_fp8_q_prologue_gaudi2", "custom_deepseek_v41_fp8_kv_prologue_gaudi2"};
int kind(const char* name) {
    if (name) for (int i = 0; i < 3; ++i) if (!std::strcmp(name, names[i])) return i;
    return -1;
}
template<class T> T parent(const char* symbol) {
    static void* handle = [] {
        const char* path = std::getenv("VLLM_HPU_DSV41_FP8_PROLOGUE_PARENT_KERNEL");
        return path && *path ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : nullptr;
    }();
    return handle ? reinterpret_cast<T>(dlsym(handle, symbol)) : nullptr;
}
}
extern "C" {
uint64_t GetLibVersion() { auto f = parent<tpc_lib_api::pfnGetLibVersion>("GetLibVersion"); return f ? f() : 0; }
tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId d, uint32_t* count, tpc_lib_api::GuidInfo* out) {
    using namespace tpc_lib_api;
    auto f = parent<pfnGetKernelGuids>("GetKernelGuids"); if (!f || !count) return GLUE_FAILED;
    if (d != DEVICE_ID_GAUDI2) return f(d, count, out);
    uint32_t inherited = 0; auto s = f(d, &inherited, nullptr); if (s != GLUE_SUCCESS) return s;
    const auto capacity = *count; *count = inherited + 3;
    if (!out || !capacity) return GLUE_SUCCESS;
    if (capacity < *count) return GLUE_FAILED;
    for (int i = 0; i < 3; ++i) { std::memset(out+i, 0, sizeof(*out)); std::strcpy(out[i].name, names[i]); }
    return f(d, &inherited, out + 3);
}
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(const tpc_lib_api::HabanaKernelParams* p,
                                               tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out) return GLUE_FAILED;
    auto f = parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel"); if (!f) return GLUE_FAILED;
    const int mode = kind(p->guid.name);
    if (mode < 0) return f(p, out);
    const unsigned input_counts[] = {7, 4, 6}, output_counts[] = {4, 3, 1};
    if (p->deviceId != DEVICE_ID_GAUDI2 || p->inputTensorNr != input_counts[mode] ||
        p->outputTensorNr != output_counts[mode])
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto& product = p->inputTensors[0].geometry;
    const auto rows = product.maxSizes[1];
    if (product.dims != 2 || product.dataType != DATA_F32 || product.maxSizes[0] != 1792 || !rows || rows > 6)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const TensorDataType types[] = {DATA_F32, DATA_F32, DATA_F32, DATA_BF16,
        mode == 2 ? DATA_I32 : DATA_BF16, mode == 2 ? DATA_F32 : DATA_I32, DATA_F32};
    const unsigned widths[] = {1792, 1792, 1, mode == 2 ? 512u : 1280u,
        mode == 2 ? unsigned(rows) : 512u, mode == 2 ? 64u : unsigned(rows), 64};
    const unsigned dims[] = {2, 2, 2, 1, 1, mode == 2 ? 2u : 1u, 2};
    for (unsigned i = 0; i < input_counts[mode]; ++i) {
        const auto& x = p->inputTensors[i].geometry;
        if (x.dataType != types[i]) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (x.dims != dims[i] || x.maxSizes[0] != widths[i]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if ((i == 1 && x.maxSizes[1] != 1) || (i == 2 && x.maxSizes[1] != rows) ||
            (i == (mode == 2 ? 5u : 6u) && (!x.maxSizes[1] || x.maxSizes[1] > 1048576)))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->inputTensorAccessPattern[i].allRequired = true;
    }
    const TensorDataType output_types[] = {mode == 2 ? DATA_BF16 : DATA_F8_143, DATA_F32, DATA_BF16, DATA_BF16};
    const unsigned output_widths[] = {mode == 2 ? 512u : 1280u, 1, 1280, 512};
    for (unsigned i = 0; i < output_counts[mode]; ++i) {
        const auto& y = p->outputTensors[i].geometry;
        if (y.dataType != output_types[i]) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (y.dims != 2 || y.maxSizes[0] != output_widths[i] || y.maxSizes[1] != rows)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        // Q and KV own disjoint outputs. Match the qualified joint publisher's
        // conservative allRequired contract rather than declaring false tiles.
        out->outputTensorAccessPattern[i].allRequired = true;
    }
    if (!p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize != sizeof(float)) return GLUE_FAILED;
    float epsilon; std::memcpy(&epsilon, p->nodeParams.nodeParams, sizeof(epsilon));
    if (epsilon <= 0 || !std::isnormal(epsilon)) return GLUE_FAILED;
    out->indexSpaceRank = mode == 0 ? 2 : 1;
    out->indexSpaceGeometry[0] = mode == 0 ? 2 : rows;
    out->indexSpaceGeometry[1] = mode == 0 ? rows : 1;
    out->kernel.paramsNr = 1; std::memcpy(out->kernel.scalarParams, &epsilon, sizeof(epsilon));
    auto* first = mode == 0 ? &_binary___deepseek_v41_fp8_qkv_prologue_gaudi2_o_start :
        mode == 1 ? &_binary___deepseek_v41_fp8_q_prologue_gaudi2_o_start :
        &_binary___deepseek_v41_fp8_kv_prologue_gaudi2_o_start;
    auto* last = mode == 0 ? &_binary___deepseek_v41_fp8_qkv_prologue_gaudi2_o_end :
        mode == 1 ? &_binary___deepseek_v41_fp8_q_prologue_gaudi2_o_end :
        &_binary___deepseek_v41_fp8_kv_prologue_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize; out->kernel.elfSize = last - first;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, first, out->kernel.elfSize); return GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId d,
    const tpc_lib_api::ShapeInferenceParams* p, tpc_lib_api::ShapeInferenceOutput* out) {
    if (kind((p->pGuid ? p->pGuid : &p->guid)->name) >= 0) return tpc_lib_api::GLUE_SUCCESS;
    auto f = parent<tpc_lib_api::pfnGetShapeInference>("GetShapeInference"); return f ? f(d, p, out) : tpc_lib_api::GLUE_FAILED;
}
tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(const tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::NodeDataLayouts* layouts, uint32_t* count) {
    if (kind(p->guid.name) < 0) {
        auto f = parent<tpc_lib_api::pfnGetSupportedDataLayout>("GetSupportedDataLayouts");
        return f ? f(p, layouts, count) : tpc_lib_api::GLUE_FAILED;
    }
    if (!count) return tpc_lib_api::GLUE_FAILED;
    *count = 1;
    if (layouts) {
        for (unsigned i = 0; i < layouts->inputTensorNr; ++i)
            std::memset(layouts->inputs[i].layout, 'x', sizeof(layouts->inputs[i].layout));
        for (unsigned i = 0; i < layouts->outputTensorNr; ++i)
            std::memset(layouts->outputs[i].layout, 'x', sizeof(layouts->outputs[i].layout));
    }
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(const tpc_lib_api::HabanaKernelParams* p,
                                                   tpc_lib_api::_TensorManipulationSuggestion* s) {
    if (kind(p->guid.name) >= 0) return tpc_lib_api::GLUE_FAILED;
    using F = tpc_lib_api::GlueCodeReturn(*)(const tpc_lib_api::HabanaKernelParams*, tpc_lib_api::_TensorManipulationSuggestion*);
    auto f = parent<F>("GetSuggestedManipulation"); return f ? f(p, s) : tpc_lib_api::GLUE_FAILED;
}
}
