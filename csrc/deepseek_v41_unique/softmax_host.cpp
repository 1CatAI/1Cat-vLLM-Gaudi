// SPDX-License-Identifier: Apache-2.0
#include "tpc_kernel_lib_interface.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
extern unsigned char _binary___deepseek_v41_softmax_parts_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_softmax_parts_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_softmax_normalize_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_softmax_normalize_gaudi2_o_end;
namespace tpc_lib_api {struct _TensorManipulationSuggestion;}
namespace {
constexpr const char* names[]={"custom_deepseek_v41_softmax_parts_gaudi2",
                              "custom_deepseek_v41_softmax_normalize_gaudi2"};
template<class T>T parent(const char* symbol) {
    static void* handle=[] {const char* path=std::getenv("VLLM_HPU_DSV41_SOFTMAX_PARENT_KERNEL");
        return path&&*path?dlopen(path,RTLD_NOW|RTLD_LOCAL):nullptr;}();
    return handle?reinterpret_cast<T>(dlsym(handle,symbol)):nullptr;
}
int mode(const char* name) {for(int i=0;i<2;++i)if(!std::strcmp(name,names[i]))return i;return -1;}
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
    for(int i=0;i<2;++i){std::memset(out+i,0,sizeof(*out));std::strcpy(out[i].name,names[i]);}
    return fn(device,&inherited,out+2);
}
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(const tpc_lib_api::HabanaKernelParams* p,
                                               tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(!p||!out)return GLUE_FAILED;
    const int selected=mode(p->guid.name);
    if(selected<0){auto fn=parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel");return fn?fn(p,out):GLUE_FAILED;}
    if(p->deviceId!=DEVICE_ID_GAUDI2||p->inputTensorNr!=(selected?2u:1u)||p->outputTensorNr!=1)
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto& x=p->inputTensors[0].geometry;
    const auto columns=x.maxSizes[0],rows=x.maxSizes[1],parts=(columns+2047)/2048;
    if(x.dims!=2||x.dataType!=DATA_F32||!rows||rows>6||columns<2048||columns>131072||columns%64)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto& output=p->outputTensors[0].geometry;
    if(output.dataType!=DATA_F32)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if(!selected) {
        if(output.dims!=3||output.maxSizes[0]!=parts||output.maxSizes[1]!=2||output.maxSizes[2]!=rows)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }else {
        const auto& stats=p->inputTensors[1].geometry;
        if(stats.dims!=3||stats.maxSizes[0]!=parts||stats.maxSizes[1]!=2||stats.maxSizes[2]!=rows||
           stats.dataType!=DATA_F32)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(output.dims!=2||output.maxSizes[0]!=columns||output.maxSizes[1]!=rows)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    // Flatten row/part tasks: the installed compiler rejects extent-one
    // secondary index dimensions. Writes remain disjoint by task ownership.
    out->indexSpaceRank=1;out->indexSpaceGeometry[0]=rows*(selected?(columns+255)/256:parts);
    for(unsigned i=0;i<p->inputTensorNr;++i)out->inputTensorAccessPattern[i].allRequired=true;
    out->outputTensorAccessPattern[0].allRequired=true;out->kernel.paramsNr=0;
    auto* first=selected?&_binary___deepseek_v41_softmax_normalize_gaudi2_o_start:
                          &_binary___deepseek_v41_softmax_parts_gaudi2_o_start;
    auto* last=selected?&_binary___deepseek_v41_softmax_normalize_gaudi2_o_end:
                         &_binary___deepseek_v41_softmax_parts_gaudi2_o_end;
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
