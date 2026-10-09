// SPDX-License-Identifier: Apache-2.0
#include "tpc_kernel_lib_interface.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#ifndef DSV41_W2_SHARED_SCALE
#define DSV41_W2_SHARED_SCALE 0
#endif
#if DSV41_W2_SHARED_SCALE
#define FIRST _binary___deepseek_v41_w2_reduce_shared_scale_n256_gaudi2_o_start
#define LAST _binary___deepseek_v41_w2_reduce_shared_scale_n256_gaudi2_o_end
#else
#define FIRST _binary___deepseek_v41_w2_reduce_n256_gaudi2_o_start
#define LAST _binary___deepseek_v41_w2_reduce_n256_gaudi2_o_end
#endif
extern unsigned char FIRST;
extern unsigned char LAST;
extern unsigned char _binary___deepseek_v41_w2_reduce_n256_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_w2_reduce_n256_gaudi2_o_end;
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }
namespace {
constexpr auto name=DSV41_W2_SHARED_SCALE ? "custom_deepseek_v41_w2_reduce_shared_scale_n256_gaudi2" :
    "custom_deepseek_v41_w2_reduce_n256_gaudi2";
constexpr auto original="custom_deepseek_v41_expert_n256_scale_reduce_gaudi2";
template<class T> T parent(const char* symbol) {
    static void* handle=[] {
        const char* path=std::getenv(DSV41_W2_SHARED_SCALE ? "VLLM_HPU_DSV41_W2_SHARED_PARENT_KERNEL" :
                                   "VLLM_HPU_DSV41_W2_REDUCE_PARENT_KERNEL");
        return path&&*path?dlopen(path,RTLD_NOW|RTLD_LOCAL):nullptr;
    }();
    return handle?reinterpret_cast<T>(dlsym(handle,symbol)):nullptr;
}
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
    if(!p||!out)return GLUE_FAILED;
    auto fn=parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel");if(!fn)return GLUE_FAILED;
    if(std::strcmp(p->guid.name,name))return fn(p,out);
    if(p->deviceId!=DEVICE_ID_GAUDI2||p->inputTensorNr!=(DSV41_W2_SHARED_SCALE?7:4)||p->outputTensorNr!=1)
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto n=p->inputTensors[0].geometry.maxSizes[0];
    const auto slots=p->inputTensors[0].geometry.maxSizes[2];
    if(n!=5120||slots<12||slots>36||slots%6)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    HabanaKernelParams inherited=*p;std::strcpy(inherited.guid.name,original);
    inherited.inputTensorNr=4;
    const auto capacity=out->kernel.elfSize;
    auto status=fn(&inherited,out);
    if(status!=GLUE_SUCCESS&&status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
    out->indexSpaceGeometry[0]=n/256;
    out->inputTensorAccessPattern[0].mapping[0]={0,256,0,255,false};
    out->outputTensorAccessPattern[0].mapping[0]={0,256,0,255,false};
#if DSV41_W2_SHARED_SCALE
    const auto tokens=slots/6;
    for(int i=4;i<7;++i) {
        const auto& tensor=p->inputTensors[i].geometry;
        const bool scalar=i==5;
        if(tensor.dims!=2||tensor.maxSizes[0]!=(scalar?1:n)||
           tensor.maxSizes[1]!=(i==6?1:tokens)||tensor.dataType!=(i==4?DATA_BF16:DATA_F32))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto& ap=out->inputTensorAccessPattern[i];ap={};
        if(scalar)ap.mapping[0]={0,0,0,0,false};else ap.mapping[0]={0,256,0,255,false};
        if(i==6)ap.mapping[1]={1,0,0,0,false};else ap.mapping[1]={1,1,0,0,false};
    }
#endif
    const auto* first=&FIRST;
    const auto* last=&LAST;
    out->kernel.elfSize=last-first;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId device,
    const tpc_lib_api::ShapeInferenceParams* p,tpc_lib_api::ShapeInferenceOutput* out) {
    if(!std::strcmp((p->pGuid?p->pGuid:&p->guid)->name,name))return tpc_lib_api::GLUE_SUCCESS;
    auto fn=parent<tpc_lib_api::pfnGetShapeInference>("GetShapeInference");
    return fn?fn(device,p,out):tpc_lib_api::GLUE_FAILED;
}
tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(const tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::NodeDataLayouts* layouts,uint32_t* count) {
    auto fn=parent<tpc_lib_api::pfnGetSupportedDataLayout>("GetSupportedDataLayouts");
    if(!fn)return tpc_lib_api::GLUE_FAILED;
    if(std::strcmp(p->guid.name,name))return fn(p,layouts,count);
#if DSV41_W2_SHARED_SCALE
    if(!count)return tpc_lib_api::GLUE_FAILED;
    *count=1;
    if(layouts) {
        for(unsigned i=0;i<layouts->inputTensorNr;++i)
            std::memset(layouts->inputs[i].layout,'x',sizeof(layouts->inputs[i].layout));
        for(unsigned i=0;i<layouts->outputTensorNr;++i)
            std::memset(layouts->outputs[i].layout,'x',sizeof(layouts->outputs[i].layout));
    }
    return tpc_lib_api::GLUE_SUCCESS;
#else
    auto inherited=*p;std::strcpy(inherited.guid.name,original);
    return fn(&inherited,layouts,count);
#endif
}
tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(const tpc_lib_api::HabanaKernelParams* p,
                                                   tpc_lib_api::_TensorManipulationSuggestion* s) {
    if(!std::strcmp(p->guid.name,name))return tpc_lib_api::GLUE_FAILED;
    using F=tpc_lib_api::GlueCodeReturn(*)(const tpc_lib_api::HabanaKernelParams*,
                                         tpc_lib_api::_TensorManipulationSuggestion*);
    auto fn=parent<F>("GetSuggestedManipulation");return fn?fn(p,s):tpc_lib_api::GLUE_FAILED;
}
}
