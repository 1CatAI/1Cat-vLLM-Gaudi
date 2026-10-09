// SPDX-License-Identifier: Apache-2.0
#include "tpc_kernel_lib_interface.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#define FIRST _binary___deepseek_v41_cooperative_silu_gaudi2_o_start
#define LAST _binary___deepseek_v41_cooperative_silu_gaudi2_o_end
extern unsigned char FIRST;
extern unsigned char LAST;
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }
namespace {
constexpr auto name="custom_deepseek_v41_cooperative_silu_gaudi2";
constexpr auto original="custom_deepseek_v41_expert_n256_silu_quant_gaudi2";
template<class T> T parent(const char* symbol) {
    static void* handle=[] {
        const char* path=std::getenv("VLLM_HPU_DSV41_COOPERATIVE_SILU_PARENT_KERNEL");
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
    if(p->deviceId!=DEVICE_ID_GAUDI2||p->inputTensorNr!=5||p->outputTensorNr!=3)
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto width=p->outputTensors[0].geometry.maxSizes[0];
    const auto rows=p->outputTensors[0].geometry.maxSizes[2];
    if((width!=640&&width!=1280)||rows<12||rows>36||rows%6||p->maxAvailableTpc!=24)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto& scratch=p->outputTensors[2].geometry;
    if(scratch.dims!=2||scratch.maxSizes[0]!=32||scratch.maxSizes[1]!=rows||scratch.dataType!=DATA_I32)
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    HabanaKernelParams inherited=*p;std::strcpy(inherited.guid.name,original);
    inherited.outputTensorNr=2;
    const auto capacity=out->kernel.elfSize;
    auto status=fn(&inherited,out);
    if(status!=GLUE_SUCCESS&&status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
    out->indexSpaceRank=2;out->indexSpaceGeometry[0]=width/128;out->indexSpaceGeometry[1]=rows;
    out->preferredSplitDim=1;out->kernelUsesLock=1;
    for(unsigned i=0;i<5;++i)out->inputTensorAccessPattern[i]={};
    auto& product=out->inputTensorAccessPattern[0];
    product.mapping[0]={0,0,0,static_cast<float>(width*2-1),true};
    product.mapping[1]={0,0,0,0,true};product.mapping[2]={1,1,0,0,false};
    for(unsigned i=1;i<5;++i)out->inputTensorAccessPattern[i].allRequired=true;
    for(unsigned i=0;i<3;++i)out->outputTensorAccessPattern[i]={};
    auto& quant=out->outputTensorAccessPattern[0];
    quant.mapping[0]={0,128,0,127,true};quant.mapping[1]={0,0,0,0,true};
    quant.mapping[2]={1,1,0,0,false};
    auto& scale=out->outputTensorAccessPattern[1];
    scale.mapping[0]={0,0,0,0,true};scale.mapping[1]={0,0,0,0,true};
    scale.mapping[2]={1,1,0,0,false};scale.sparseAccess=true;
    auto& coord=out->outputTensorAccessPattern[2];
    coord.mapping[0]={0,0,0,31,true};coord.mapping[1]={1,1,0,0,false};
    coord.memsetBeforeExecution=true;coord.sparseAccess=true;
    std::memset(&coord.memsetValue,0,sizeof(coord.memsetValue));
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
