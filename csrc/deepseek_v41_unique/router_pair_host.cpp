// SPDX-License-Identifier: Apache-2.0
#include "tpc_kernel_lib_interface.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
extern unsigned char _binary___deepseek_v41_router_pair_input_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_router_pair_input_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_router_pair_finish_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_router_pair_finish_gaudi2_o_end;
namespace tpc_lib_api {struct _TensorManipulationSuggestion;}
namespace {
constexpr const char* names[]={"custom_deepseek_v41_router_pair_input_gaudi2",
                              "custom_deepseek_v41_router_pair_finish_gaudi2"};
template<class T>T parent(const char* symbol) {
    static void* handle=[] {const char* path=std::getenv("VLLM_HPU_DSV41_ROUTER_PAIR_PARENT_KERNEL");
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
    if(p->deviceId!=DEVICE_ID_GAUDI2 || p->inputTensorNr!=(selected?6u:2u) ||
       p->outputTensorNr!=2u)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    auto shape=[](const Tensor& t,decltype(DATA_F32) type,uint64_t columns,uint64_t rows) {
        const auto& g=t.geometry;
        return g.dataType==type && g.dims==2 && g.maxSizes[0]==columns && g.maxSizes[1]==rows;
    };
    const auto rows=p->inputTensors[0].geometry.maxSizes[1]/(selected?2:1);
    if(rows<2 || rows>6)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if(!selected) {
        if(!shape(p->inputTensors[0],DATA_BF16,5120,rows) ||
           !shape(p->inputTensors[1],DATA_F32,1,rows))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(p->outputTensors[0],DATA_F8_143,5120,rows*2))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if(!shape(p->outputTensors[1],DATA_F32,1,rows))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank=2;out->indexSpaceGeometry[0]=40;out->indexSpaceGeometry[1]=rows;
        out->inputTensorAccessPattern[0].mapping[0]={0,128,0,127,false};
        out->inputTensorAccessPattern[0].mapping[1]={1,1,0,0,false};
        out->inputTensorAccessPattern[1].mapping[0]={0,0,0,0,false};
        out->inputTensorAccessPattern[1].mapping[1]={1,1,0,0,false};
        out->outputTensorAccessPattern[0].mapping[0]={0,128,0,127,false};
        out->outputTensorAccessPattern[0].mapping[1]={1,1,0,float(rows),false};
        out->outputTensorAccessPattern[1].mapping[0]={0,0,0,0,false};
        out->outputTensorAccessPattern[1].mapping[1]={1,1,0,0,false};
    }else {
        if(!shape(p->inputTensors[0],DATA_F32,1024,rows*2) ||
           !shape(p->inputTensors[4],DATA_F32,384,2) ||
           !shape(p->inputTensors[5],DATA_F32,1,rows))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for(unsigned i=1;i<4;++i) {
            const auto& g=p->inputTensors[i].geometry;
            if(g.dims!=1 || g.dataType!=(i==3?DATA_I8:DATA_F32) || g.maxSizes[0]!=(i==3?rows:384))
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
        }
        if(!shape(p->outputTensors[0],DATA_I32,6,rows) ||
           !shape(p->outputTensors[1],DATA_F32,6,rows))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank=1;out->indexSpaceGeometry[0]=rows;
        out->inputTensorAccessPattern[0].mapping[0]={0,0,0,895,false};
        out->inputTensorAccessPattern[0].mapping[1]={0,1,0,float(rows),false};
        for(unsigned i=1;i<3;++i)out->inputTensorAccessPattern[i].mapping[0]={0,0,0,383,false};
        out->inputTensorAccessPattern[3].mapping[0]={0,1,0,0,false};
        out->inputTensorAccessPattern[4].mapping[0]={0,0,0,383,false};
        out->inputTensorAccessPattern[4].mapping[1]={0,0,0,1,false};
        out->inputTensorAccessPattern[5].mapping[0]={0,0,0,0,false};
        out->inputTensorAccessPattern[5].mapping[1]={0,1,0,0,false};
        for(unsigned i=0;i<2;++i) {
            out->outputTensorAccessPattern[i].mapping[0]={0,0,0,5,false};
            out->outputTensorAccessPattern[i].mapping[1]={0,1,0,0,false};
        }
    }
    out->kernel.paramsNr=0;
    auto* first=selected?&_binary___deepseek_v41_router_pair_finish_gaudi2_o_start:
                          &_binary___deepseek_v41_router_pair_input_gaudi2_o_start;
    auto* last=selected?&_binary___deepseek_v41_router_pair_finish_gaudi2_o_end:
                         &_binary___deepseek_v41_router_pair_input_gaudi2_o_end;
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
