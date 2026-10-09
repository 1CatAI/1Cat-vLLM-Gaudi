// SPDX-License-Identifier: Apache-2.0
#include "tpc_kernel_lib_interface.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
extern unsigned char _binary___deepseek_v41_w2_ready_scale_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_w2_ready_scale_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_w2_ready_reduce_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_w2_ready_reduce_gaudi2_o_end;
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }
namespace {
using namespace tpc_lib_api;
constexpr const char* names[]={"custom_deepseek_v41_w2_ready_scale_gaudi2",
                              "custom_deepseek_v41_w2_ready_reduce_gaudi2"};
int kind(const char* name) {
    for(int i=0;i<2;++i)if(!std::strcmp(name,names[i]))return i;
    return -1;
}
template<class T> T parent(const char* name) {
    static void* handle=[] {
        const char* path=std::getenv("VLLM_HPU_DSV41_W2_FACTOR_PARENT_KERNEL");
        return path&&*path?dlopen(path,RTLD_NOW|RTLD_LOCAL):nullptr;
    }();
    return handle?reinterpret_cast<T>(dlsym(handle,name)):nullptr;
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
    const auto capacity=*count;*count=inherited+2;
    if(!out||!capacity)return GLUE_SUCCESS;
    if(capacity<*count)return GLUE_FAILED;
    std::memset(out,0,2*sizeof(*out));for(int i=0;i<2;++i)std::strcpy(out[i].name,names[i]);
    return fn(device,&inherited,out+2);
}
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(const tpc_lib_api::HabanaKernelParams* p,
                                               tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(!p||!out)return GLUE_FAILED;
    const int op=kind(p->guid.name);
    if(op<0){auto fn=parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel");return fn?fn(p,out):GLUE_FAILED;}
    if(p->deviceId!=DEVICE_ID_GAUDI2)return GLUE_FAILED;
    if(p->inputTensorNr!=(op==0?3u:2u)||p->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto& o=p->outputTensors[0].geometry;
    if(o.dims!=3||o.maxSizes[0]!=5120||o.maxSizes[1]!=1)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if(op==0) {
        const auto& ids=p->inputTensors[0].geometry;
        const auto& sx=p->inputTensors[1].geometry;
        const auto& ch=p->inputTensors[2].geometry;
        const auto rows=o.maxSizes[2];
        if(o.dataType!=DATA_F32||rows<12||rows>36||rows%6||ids.dataType!=DATA_I32||
           ids.dims!=2||ids.maxSizes[0]!=rows||ids.maxSizes[1]!=1||sx.dataType!=DATA_F32||sx.dims!=2||
           sx.maxSizes[0]!=1||sx.maxSizes[1]!=rows||ch.dataType!=DATA_BF16||ch.dims!=3||
           ch.maxSizes[0]!=256||ch.maxSizes[1]!=20||ch.maxSizes[2]<1)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank=2;out->indexSpaceGeometry[0]=40;out->indexSpaceGeometry[1]=rows;
        out->inputTensorAccessPattern[0].mapping[0]={1,1,0,0,false};
        out->inputTensorAccessPattern[0].mapping[1]={0,0,0,0,false};
        out->inputTensorAccessPattern[1].mapping[0]={0,0,0,0,false};
        out->inputTensorAccessPattern[1].mapping[1]={1,1,0,0,false};
        out->inputTensorAccessPattern[2].allRequired=true;
    } else {
        const auto tokens=o.maxSizes[2];
        if(o.dataType!=DATA_BF16||tokens<2||tokens>6)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        for(unsigned i=0;i<2;++i) {
            const auto& t=p->inputTensors[i].geometry;
            if(t.dataType!=DATA_F32||t.dims!=3||t.maxSizes[0]!=5120||t.maxSizes[1]!=1||
               t.maxSizes[2]!=tokens*6)return GLUE_INCOMPATIBLE_INPUT_SIZE;
            out->inputTensorAccessPattern[i].mapping[0]={0,128,0,127,false};
            out->inputTensorAccessPattern[i].mapping[1]={0,0,0,0,false};
            out->inputTensorAccessPattern[i].mapping[2]={1,6,0,5,false};
        }
        out->indexSpaceRank=2;out->indexSpaceGeometry[0]=40;out->indexSpaceGeometry[1]=tokens;
    }
    out->outputTensorAccessPattern[0].mapping[0]={0,128,0,127,false};
    out->outputTensorAccessPattern[0].mapping[1]={0,0,0,0,false};
    out->outputTensorAccessPattern[0].mapping[2]={1,1,0,0,false};
    out->kernel.paramsNr=0;
    const auto* first=op==0?&_binary___deepseek_v41_w2_ready_scale_gaudi2_o_start:
                           &_binary___deepseek_v41_w2_ready_reduce_gaudi2_o_start;
    const auto* last=op==0?&_binary___deepseek_v41_w2_ready_scale_gaudi2_o_end:
                          &_binary___deepseek_v41_w2_ready_reduce_gaudi2_o_end;
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=last-first;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId device,
    const tpc_lib_api::ShapeInferenceParams* p,tpc_lib_api::ShapeInferenceOutput* out) {
    if(kind((p->pGuid?p->pGuid:&p->guid)->name)>=0)return tpc_lib_api::GLUE_SUCCESS;
    auto fn=parent<tpc_lib_api::pfnGetShapeInference>("GetShapeInference");return fn?fn(device,p,out):tpc_lib_api::GLUE_FAILED;
}
tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(const tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::NodeDataLayouts* layouts,uint32_t* count) {
    using namespace tpc_lib_api;
    if(kind(p->guid.name)<0){auto fn=parent<pfnGetSupportedDataLayout>("GetSupportedDataLayouts");
        return fn?fn(p,layouts,count):GLUE_FAILED;}
    if(!count)return GLUE_FAILED;
    *count=1;
    if(layouts){
        for(unsigned i=0;i<layouts->inputTensorNr;++i)std::memset(layouts->inputs[i].layout,'x',sizeof(layouts->inputs[i].layout));
        for(unsigned i=0;i<layouts->outputTensorNr;++i)std::memset(layouts->outputs[i].layout,'x',sizeof(layouts->outputs[i].layout));
    }
    return GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(const tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::_TensorManipulationSuggestion* suggestion) {
    if(kind(p->guid.name)>=0)return tpc_lib_api::GLUE_FAILED;
    using F=tpc_lib_api::GlueCodeReturn(*)(const tpc_lib_api::HabanaKernelParams*,
                                        tpc_lib_api::_TensorManipulationSuggestion*);
    auto fn=parent<F>("GetSuggestedManipulation");return fn?fn(p,suggestion):tpc_lib_api::GLUE_FAILED;
}
}
