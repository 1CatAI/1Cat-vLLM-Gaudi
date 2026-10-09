// SPDX-License-Identifier: Apache-2.0
// Reuse the immutable scaled Router program with no preceding shared columns.
#include "tpc_kernel_lib_interface.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
namespace tpc_lib_api {struct _TensorManipulationSuggestion;}
namespace {
constexpr auto name="custom_deepseek_v41_router_ready_scaled_gaudi2";
constexpr auto inherited_name="custom_deepseek_v41_router_shared_scaled_gaudi2";
template<class T>T parent(const char* symbol) {
    static void* handle=[] {const char* path=std::getenv("VLLM_HPU_DSV41_ROUTER_READY_PARENT_KERNEL");
        return path&&*path?dlopen(path,RTLD_NOW|RTLD_LOCAL):nullptr;}();
    return handle?reinterpret_cast<T>(dlsym(handle,symbol)):nullptr;
}
bool selected(const char* guid) {return std::strcmp(guid,name)==0;}
}
extern "C" {
uint64_t GetLibVersion() {auto fn=parent<tpc_lib_api::pfnGetLibVersion>("GetLibVersion");return fn?fn():0;}
tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId device,uint32_t* count,
                                         tpc_lib_api::GuidInfo* out) {
    using namespace tpc_lib_api;
    auto fn=parent<pfnGetKernelGuids>("GetKernelGuids");if(!fn||!count)return GLUE_FAILED;
    if(device!=DEVICE_ID_GAUDI2)return fn(device,count,out);
    uint32_t inherited=0;auto status=fn(device,&inherited,nullptr);if(status!=GLUE_SUCCESS)return status;
    const auto capacity=*count;*count=inherited+1;
    if(!out||!capacity)return GLUE_SUCCESS;
    if(capacity<*count)return GLUE_FAILED;
    std::memset(out,0,sizeof(*out));std::strcpy(out->name,name);
    return fn(device,&inherited,out+1);
}
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(const tpc_lib_api::HabanaKernelParams* p,
                                               tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    auto fn=parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel");
    if(!p||!out||!fn)return GLUE_FAILED;
    if(!selected(p->guid.name))return fn(p,out);
    if(p->deviceId!=DEVICE_ID_GAUDI2||p->inputTensorNr!=6||p->outputTensorNr!=2)
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto& product=p->inputTensors[0].geometry;
    if(product.dims!=2||product.dataType!=DATA_F32||product.maxSizes[0]!=512||
       product.maxSizes[1]<2||product.maxSizes[1]>6)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    // The parent program already derives the Router offset from the runtime
    // tensor extent. Its installed glue restricts the old joint shapes.
    // Obtain that exact ELF/row ownership through its supported template,
    // then describe the real512-column operand, with Router at columns0..383.
    // Copy the tensor array as well: never mutate the caller's descriptors.
    std::array<Tensor,6> inputs;
    std::memcpy(inputs.data(),p->inputTensors,sizeof(inputs));
    HabanaKernelParams inherited=*p;
    inherited.inputTensors=inputs.data();
    std::strcpy(inherited.guid.name,inherited_name);
    inputs[0].geometry.maxSizes[0]=1792;
    inputs[0].geometry.minSizes[0]=1792;
    const auto status=fn(&inherited,out);
    if(status==GLUE_SUCCESS)out->inputTensorAccessPattern[0].mapping[0]={0,0,0,383,false};
    return status;
}
tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId device,
    const tpc_lib_api::ShapeInferenceParams* p,tpc_lib_api::ShapeInferenceOutput* out) {
    if(selected((p->pGuid?p->pGuid:&p->guid)->name))return tpc_lib_api::GLUE_SUCCESS;
    auto fn=parent<tpc_lib_api::pfnGetShapeInference>("GetShapeInference");return fn?fn(device,p,out):tpc_lib_api::GLUE_FAILED;
}
tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(const tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::NodeDataLayouts* layouts,uint32_t* count) {
    if(selected(p->guid.name)) {
        if(!count)return tpc_lib_api::GLUE_FAILED;
        *count=1;
        if(layouts) {
            for(unsigned i=0;i<layouts->inputTensorNr;++i)
                std::memset(layouts->inputs[i].layout,'x',sizeof(layouts->inputs[i].layout));
            for(unsigned i=0;i<layouts->outputTensorNr;++i)
                std::memset(layouts->outputs[i].layout,'x',sizeof(layouts->outputs[i].layout));
        }
        return tpc_lib_api::GLUE_SUCCESS;
    }
    auto fn=parent<tpc_lib_api::pfnGetSupportedDataLayout>("GetSupportedDataLayouts");
    return fn?fn(p,layouts,count):tpc_lib_api::GLUE_FAILED;
}
tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(const tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::_TensorManipulationSuggestion* s) {
    if(selected(p->guid.name))return tpc_lib_api::GLUE_FAILED;
    using F=tpc_lib_api::GlueCodeReturn(*)(const tpc_lib_api::HabanaKernelParams*,tpc_lib_api::_TensorManipulationSuggestion*);
    auto fn=parent<F>("GetSuggestedManipulation");return fn?fn(p,s):tpc_lib_api::GLUE_FAILED;
}
}
