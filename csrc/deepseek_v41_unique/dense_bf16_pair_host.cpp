// SPDX-License-Identifier: Apache-2.0
#include "tpc_kernel_lib_interface.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#define FIRST _binary___deepseek_v41_dense_bf16_pair_gaudi2_o_start
#define LAST _binary___deepseek_v41_dense_bf16_pair_gaudi2_o_end
extern unsigned char FIRST;
extern unsigned char LAST;
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }
namespace {
constexpr auto name="custom_deepseek_v41_dense_bf16_pair_gaudi2";

template<class T> T parent(const char* symbol) {
    static void* handle=[] {
        const char* path=std::getenv("VLLM_HPU_DSV41_DENSE_BF16_PAIR_PARENT_KERNEL");
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
    if(p->deviceId!=DEVICE_ID_GAUDI2||p->inputTensorNr!=1||p->outputTensorNr!=1)
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto& x=p->inputTensors[0].geometry;
    const auto& y=p->outputTensors[0].geometry;
    if(x.dims!=2||x.dataType!=DATA_BF16||x.maxSizes[0]!=5120||x.maxSizes[1]<2||x.maxSizes[1]>6)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if(y.dims!=2||y.dataType!=DATA_BF16||y.maxSizes[0]!=5120||y.maxSizes[1]!=x.maxSizes[1]*2)
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank=1;out->indexSpaceGeometry[0]=x.maxSizes[1];
    out->inputTensorAccessPattern[0]={};out->outputTensorAccessPattern[0]={};
    auto& in=out->inputTensorAccessPattern[0];
    in.mapping[0]={0,0,0,5119,true};in.mapping[1]={0,1,0,0,false};
    auto& dest=out->outputTensorAccessPattern[0];
    dest.mapping[0]={0,0,0,5119,true};
    dest.mapping[1]={0,1,0,static_cast<float>(x.maxSizes[1]),false};
    // Each row owns both disjoint output rows. The spanning access is a
    // conservative compiler dependency, not duplicated numerical work.
    out->kernel.paramsNr=0;
    const auto capacity=out->kernel.elfSize;
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
tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(const tpc_lib_api::HabanaKernelParams* p,
                                                   tpc_lib_api::_TensorManipulationSuggestion* s) {
    if(!std::strcmp(p->guid.name,name))return tpc_lib_api::GLUE_FAILED;
    using F=tpc_lib_api::GlueCodeReturn(*)(const tpc_lib_api::HabanaKernelParams*,
                                         tpc_lib_api::_TensorManipulationSuggestion*);
    auto fn=parent<F>("GetSuggestedManipulation");return fn?fn(p,s):tpc_lib_api::GLUE_FAILED;
}
}
