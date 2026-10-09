// SPDX-License-Identifier: Apache-2.0
#include "tpc_kernel_lib_interface.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#ifndef DSV41_DENSE_BITS12
#define DSV41_DENSE_BITS12 0
#endif
#if DSV41_DENSE_BITS12
extern unsigned char _binary___deepseek_v41_dense_bits12_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_dense_bits12_gaudi2_o_end;
#else
extern unsigned char _binary___deepseek_v41_dense_source_fp8_decode_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_dense_source_fp8_decode_gaudi2_o_end;
#endif
namespace tpc_lib_api {struct _TensorManipulationSuggestion;}
namespace {
constexpr const char* names[]={
#if DSV41_DENSE_BITS12
    "custom_deepseek_v41_dense_bits12_bf16_gaudi2"
#else
    "custom_deepseek_v41_source_weight_bf16_gaudi2"
#endif
};
template<class T>T parent(const char* symbol) {
    static void* handle=[] {const char* path=std::getenv("VLLM_HPU_DSV41_SOURCE_DENSE_PARENT_KERNEL");
        return path&&*path?dlopen(path,RTLD_NOW|RTLD_LOCAL):nullptr;}();
    return handle?reinterpret_cast<T>(dlsym(handle,symbol)):nullptr;
}
int mode(const char* name) {for(int i=0;i<1;++i)if(!std::strcmp(name,names[i]))return i;return -1;}
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
    for(int i=0;i<1;++i){std::memset(out+i,0,sizeof(*out));std::strcpy(out[i].name,names[i]);}
    return fn(device,&inherited,out+1);
}
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(const tpc_lib_api::HabanaKernelParams* p,
                                               tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(!p||!out)return GLUE_FAILED;
    const int selected=mode(p->guid.name);
    if(selected<0){auto fn=parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel");return fn?fn(p,out):GLUE_FAILED;}
    if(p->deviceId!=DEVICE_ID_GAUDI2 || p->inputTensorNr!=2 || p->outputTensorNr!=1)
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto& x=p->inputTensors[0].geometry;
    const auto& scale=p->inputTensors[1].geometry;
    const auto& y=p->outputTensors[0].geometry;
    if(x.dims!=2 || x.dataType!=DATA_I8 || x.maxSizes[0]<128 || x.maxSizes[0]>16384 ||
#if DSV41_DENSE_BITS12
       x.maxSizes[0]%256 || x.maxSizes[1]<32 || x.maxSizes[1]>32768 || x.maxSizes[1]%32 ||
       scale.dims!=2 || scale.dataType!=DATA_I8 || scale.maxSizes[0]!=x.maxSizes[0]/2 ||
       scale.maxSizes[1]!=x.maxSizes[1]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
#else
       x.maxSizes[0]%128 || x.maxSizes[1]<32 || x.maxSizes[1]>32768 || x.maxSizes[1]%32 ||
       scale.dims!=2 || scale.dataType!=DATA_I16 || scale.maxSizes[0]!=x.maxSizes[0]/32 ||
       scale.maxSizes[1]!=x.maxSizes[1]/32) return GLUE_INCOMPATIBLE_INPUT_SIZE;
#endif
    if(y.dims!=2 || y.dataType!=DATA_BF16 || y.maxSizes[0]!=x.maxSizes[0] || y.maxSizes[1]!=x.maxSizes[1])
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank=2;
#if DSV41_DENSE_BITS12
    out->indexSpaceGeometry[0]=x.maxSizes[0]/256;out->indexSpaceGeometry[1]=x.maxSizes[1]/32;
    out->inputTensorAccessPattern[0].mapping[0]={0,256,0,255,false};
    out->inputTensorAccessPattern[0].mapping[1]={1,32,0,31,false};
    out->inputTensorAccessPattern[1].mapping[0]={0,128,0,127,false};
    out->inputTensorAccessPattern[1].mapping[1]={1,32,0,31,false};
#else
    out->indexSpaceGeometry[0]=x.maxSizes[0]/128;out->indexSpaceGeometry[1]=x.maxSizes[1]/32;
    out->inputTensorAccessPattern[0].mapping[0]={0,128,0,127,false};
    out->inputTensorAccessPattern[0].mapping[1]={1,32,0,31,false};
    out->inputTensorAccessPattern[1].mapping[0]={0,4,0,3,false};
    out->inputTensorAccessPattern[1].mapping[1]={1,1,0,0,false};
#endif
    out->outputTensorAccessPattern[0]=out->inputTensorAccessPattern[0];
    out->kernel.paramsNr=0;
#if DSV41_DENSE_BITS12
    auto* first=&_binary___deepseek_v41_dense_bits12_gaudi2_o_start;
    auto* last=&_binary___deepseek_v41_dense_bits12_gaudi2_o_end;
#else
    auto* first=&_binary___deepseek_v41_dense_source_fp8_decode_gaudi2_o_start;
    auto* last=&_binary___deepseek_v41_dense_source_fp8_decode_gaudi2_o_end;
#endif
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=last-first;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId device,
    const tpc_lib_api::ShapeInferenceParams* p,tpc_lib_api::ShapeInferenceOutput* out) {
    if(mode((p->pGuid?p->pGuid:&p->guid)->name)>=0)return tpc_lib_api::GLUE_SUCCESS;
    auto fn=parent<tpc_lib_api::pfnGetShapeInference>("GetShapeInference");return fn?fn(device,p,out):tpc_lib_api::GLUE_FAILED;
}
tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(const tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::NodeDataLayouts* layouts,uint32_t* count) {
    if(mode(p->guid.name)>=0) {
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
    if(mode(p->guid.name)>=0)return tpc_lib_api::GLUE_FAILED;
    using F=tpc_lib_api::GlueCodeReturn(*)(const tpc_lib_api::HabanaKernelParams*,tpc_lib_api::_TensorManipulationSuggestion*);
    auto fn=parent<F>("GetSuggestedManipulation");return fn?fn(p,s):tpc_lib_api::GLUE_FAILED;
}
}
