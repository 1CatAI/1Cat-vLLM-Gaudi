// SPDX-License-Identifier: Apache-2.0
// Additive provider: delegate every existing GUID to the locked parent binary.
#include "tpc_kernel_lib_interface.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>

extern unsigned char _binary___deepseek_v41_journal_batch_direct_gaudi2_o_start, _binary___deepseek_v41_journal_batch_direct_gaudi2_o_end;
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }

namespace {
using namespace tpc_lib_api;
constexpr const char* names[]={"custom_deepseek_v41_journal_batch_direct_gaudi2"};
int kind(const char* name) {return std::strcmp(name,names[0])? -1:0;}
template<class T> T parent(const char* name) {
    static void* handle=[] {
        const char* path=std::getenv("VLLM_HPU_DSV41_JOURNAL_BATCH_PARENT_KERNEL");
        return path && *path ? dlopen(path,RTLD_NOW|RTLD_LOCAL) : nullptr;
    }();
    return handle ? reinterpret_cast<T>(dlsym(handle,name)) : nullptr;
}

}

extern "C" {
uint64_t GetLibVersion() {
    auto fn=parent<tpc_lib_api::pfnGetLibVersion>("GetLibVersion");return fn?fn():0;
}
tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId device,uint32_t* count,
                                         tpc_lib_api::GuidInfo* out) {
    using namespace tpc_lib_api;
    auto fn=parent<pfnGetKernelGuids>("GetKernelGuids");if(!fn||!count)return GLUE_FAILED;
    if(device!=DEVICE_ID_GAUDI2)return fn(device,count,out);
    uint32_t inherited=0;auto status=fn(device,&inherited,nullptr);if(status!=GLUE_SUCCESS)return status;
    const auto capacity=*count;*count=inherited+1;
    if(!out||!capacity)return GLUE_SUCCESS;
    if(capacity<*count)return GLUE_FAILED;
    std::memset(out,0,sizeof(*out));for(unsigned i=0;i<1;++i)std::strcpy(out[i].name,names[i]);
    return fn(device,&inherited,out+1);
}
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(const tpc_lib_api::HabanaKernelParams* p,
                                               tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(!p||!out)return GLUE_FAILED;
    const int op=kind(p->guid.name);
    if(op<0){auto fn=parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel");return fn?fn(p,out):GLUE_FAILED;}
    if(p->deviceId!=DEVICE_ID_GAUDI2)return GLUE_FAILED;
    if(p->inputTensorNr!=7||p->outputTensorNr!=8)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto bytes=[](TensorDataType t)->unsigned {
        if(t==DATA_U8||t==DATA_I8)return 1;
        if(t==DATA_I16||t==DATA_BF16||t==DATA_F16)return 2;
        if(t==DATA_I32||t==DATA_F32)return 4;
        return 0;
    };
    const auto expect=[](const Tensor& t,TensorDataType type,std::initializer_list<uint64_t> sizes) {
        if(t.geometry.dataType!=type||t.geometry.dims!=sizes.size())return false;
        unsigned i=0;for(auto n:sizes)if(t.geometry.maxSizes[i++]!=n)return false;
        return true;
    };
    uint64_t width=0;int32_t byteCodes=0;
    for(unsigned i=0;i<4;++i) {
        const auto& t=p->inputTensors[i].geometry;const auto size=bytes(t.dataType);
        if(!size||t.dims!=2||!t.maxSizes[0]||t.maxSizes[1]<6 ||
           !expect(p->outputTensors[i],t.dataType,{t.maxSizes[0],6})||
           !expect(p->outputTensors[i+4],DATA_I32,{6}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(t.maxSizes[0]*size>width)width=t.maxSizes[0]*size;
        byteCodes|=(size==4?2:size==2?1:0)<<(i*2);
        out->inputTensorAccessPattern[i].allRequired=true;
        auto& saved=out->outputTensorAccessPattern[i];saved.sparseAccess=true;
        saved.mapping[0]={2,float(256/size),0,float(256/size-1),false};
        saved.mapping[1]={1,1,0,0,false};
        auto& indices=out->outputTensorAccessPattern[i+4];indices.sparseAccess=true;
        indices.mapping[0]={1,1,0,0,false};
    }
    if(!expect(p->inputTensors[4],DATA_I32,{6})||
       p->inputTensors[5].geometry.dataType!=DATA_I32||p->inputTensors[5].geometry.dims!=1||
       !p->inputTensors[5].geometry.maxSizes[0]||!expect(p->inputTensors[6],DATA_I32,{2,4}))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if(p->nodeParams.nodeParamsSize!=sizeof(int32_t)||!p->nodeParams.nodeParams ||
       *static_cast<const int32_t*>(p->nodeParams.nodeParams)!=byteCodes)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank=3;out->indexSpaceGeometry[0]=4;
    out->indexSpaceGeometry[1]=6;out->indexSpaceGeometry[2]=(width+255)/256;
    out->inputTensorAccessPattern[4].mapping[0]={1,1,0,0,false};
    out->inputTensorAccessPattern[5].allRequired=true;
    auto& config=out->inputTensorAccessPattern[6];
    config.mapping[0]={0,0,0,1,false};config.mapping[1]={0,1,0,0,false};
    out->kernel.paramsNr=1;out->kernel.scalarParams[0]=byteCodes;
    const auto* start=&_binary___deepseek_v41_journal_batch_direct_gaudi2_o_start;
    const auto* end=&_binary___deepseek_v41_journal_batch_direct_gaudi2_o_end;
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=end-start;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,start,out->kernel.elfSize);return GLUE_SUCCESS;

}
tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId device,
    const tpc_lib_api::ShapeInferenceParams* p,tpc_lib_api::ShapeInferenceOutput* out) {
    if(kind((p->pGuid?p->pGuid:&p->guid)->name)>=0)return tpc_lib_api::GLUE_SUCCESS;
    auto fn=parent<tpc_lib_api::pfnGetShapeInference>("GetShapeInference");
    return fn?fn(device,p,out):tpc_lib_api::GLUE_FAILED;
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
    using Function=tpc_lib_api::GlueCodeReturn(*)(const tpc_lib_api::HabanaKernelParams*,
                                                tpc_lib_api::_TensorManipulationSuggestion*);
    auto fn=parent<Function>("GetSuggestedManipulation");
    return fn?fn(p,suggestion):tpc_lib_api::GLUE_FAILED;
}
}
