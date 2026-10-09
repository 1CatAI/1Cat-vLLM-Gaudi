// SPDX-License-Identifier: Apache-2.0
// Additive provider: delegate every existing GUID to the locked parent binary.
#include "tpc_kernel_lib_interface.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>

extern unsigned char _binary___deepseek_v41_mla_stream_exp_publish_gaudi2_o_start, _binary___deepseek_v41_mla_stream_exp_publish_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mla_stream_exp_reuse_gaudi2_o_start, _binary___deepseek_v41_mla_stream_exp_reuse_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mla_stream_finish_publish_gaudi2_o_start, _binary___deepseek_v41_mla_stream_finish_publish_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mla_stream_finish_reuse_gaudi2_o_start, _binary___deepseek_v41_mla_stream_finish_reuse_gaudi2_o_end;
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }

namespace {
using namespace tpc_lib_api;
constexpr const char* names[] = {
    "custom_deepseek_v41_mla_stream_exp_publish_gaudi2",
    "custom_deepseek_v41_mla_stream_exp_reuse_gaudi2",
    "custom_deepseek_v41_mla_stream_finish_publish_gaudi2",
    "custom_deepseek_v41_mla_stream_finish_reuse_gaudi2"};
int kind(const char* name) {
    for (int i=0;i<4;++i) if(!std::strcmp(name,names[i])) return i;
    return -1;
}
template<class T> T parent(const char* name) {
    static void* handle=[] {
        const char* path=std::getenv("VLLM_HPU_DSV41_MLA_STREAM_PARENT_KERNEL");
        return path && *path ? dlopen(path,RTLD_NOW|RTLD_LOCAL) : nullptr;
    }();
    return handle ? reinterpret_cast<T>(dlsym(handle,name)) : nullptr;
}
bool shape(const Tensor& t,std::initializer_list<uint64_t> sizes) {
    if(t.geometry.dataType!=DATA_F32 || t.geometry.dims!=sizes.size()) return false;
    unsigned i=0;for(auto size:sizes)if(t.geometry.maxSizes[i++]!=size)return false;
    return true;
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
    const auto capacity=*count;*count=inherited+4;
    if(!out||!capacity)return GLUE_SUCCESS;
    if(capacity<*count)return GLUE_FAILED;
    std::memset(out,0,4*sizeof(*out));for(unsigned i=0;i<4;++i)std::strcpy(out[i].name,names[i]);
    return fn(device,&inherited,out+4);
}
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(const tpc_lib_api::HabanaKernelParams* p,
                                               tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(!p||!out)return GLUE_FAILED;
    const int op=kind(p->guid.name);
    if(op<0){auto fn=parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel");return fn?fn(p,out):GLUE_FAILED;}
    if(p->deviceId!=DEVICE_ID_GAUDI2)return GLUE_FAILED;
    const bool split=op%2;
    const auto expect=[&](const Tensor& t,TensorDataType type,std::initializer_list<uint64_t> dimensions) {
        if(t.geometry.dataType!=type||t.geometry.dims!=dimensions.size())return false;
        unsigned i=0;for(auto d:dimensions)if(t.geometry.maxSizes[i++]!=d)return false;
        return true;
    };
    const unsigned outputs=op<2?(split?3:2):1;
    const unsigned inputs=op<2?5:(split?5:4);
    if(p->inputTensorNr!=inputs||p->outputTensorNr!=outputs)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto& g=p->outputTensors[0].geometry;
    const auto heads=g.maxSizes[1],tokens=g.maxSizes[2];
    if(tokens<2||tokens>6||heads<1||heads>64)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if(op<2) {
        const uint64_t first=split?128:640,second=split?512:640;
        if(!shape(p->inputTensors[0],{first,heads,tokens})||
           !shape(p->inputTensors[1],{second,heads,tokens})||
           !shape(p->inputTensors[2],{640,tokens})||!shape(p->inputTensors[3],{heads})||
           !shape(p->inputTensors[4],{1})||
           !expect(p->outputTensors[0],DATA_BF16,{first,heads,tokens})||
           (split&&!expect(p->outputTensors[1],DATA_BF16,{512,heads,tokens}))||
           !shape(p->outputTensors[outputs-1],{1,heads,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank=2;out->indexSpaceGeometry[0]=heads;out->indexSpaceGeometry[1]=tokens;
        for(unsigned i=0;i<2;++i){
            auto& a=out->inputTensorAccessPattern[i];const auto width=i?second:first;
            a.mapping[0]={0,0,0,float(width-1),false};
            a.mapping[1]={0,1,0,0,false};a.mapping[2]={1,1,0,0,false};
        }
        auto& mask=out->inputTensorAccessPattern[2];
        mask.mapping[0]={0,0,0,639,false};mask.mapping[1]={1,1,0,0,false};
        out->inputTensorAccessPattern[3].mapping[0]={0,1,0,0,false};
        out->inputTensorAccessPattern[4].allRequired=true;
        for(unsigned i=0;i<outputs;++i){
            auto& a=out->outputTensorAccessPattern[i];const auto width=p->outputTensors[i].geometry.maxSizes[0];
            a.mapping[0]={0,0,0,float(width-1),false};
            a.mapping[1]={0,1,0,0,false};a.mapping[2]={1,1,0,0,false};
        }
    } else {
        const unsigned inverse=split?2:1,positions=inverse+1,phase=inverse+2;
        if(!shape(p->inputTensors[0],{512,heads,tokens})||
           (split&&!shape(p->inputTensors[1],{512,heads,tokens}))||
           !shape(p->inputTensors[inverse],{1,heads,tokens})||
           !expect(p->inputTensors[positions],DATA_I32,{tokens})||
           p->inputTensors[phase].geometry.dataType!=DATA_F32||
           p->inputTensors[phase].geometry.dims!=2||p->inputTensors[phase].geometry.maxSizes[0]!=64||
           !p->inputTensors[phase].geometry.maxSizes[1]||
           !expect(p->outputTensors[0],DATA_BF16,{512,heads,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank=3;out->indexSpaceGeometry[0]=4;
        out->indexSpaceGeometry[1]=heads;out->indexSpaceGeometry[2]=tokens;
        for(unsigned i=0;i<inverse;++i){
            auto& a=out->inputTensorAccessPattern[i];a.mapping[0]={0,128,0,127,false};
            a.mapping[1]={1,1,0,0,false};a.mapping[2]={2,1,0,0,false};
        }
        auto& a=out->inputTensorAccessPattern[inverse];a.mapping[0]={0,0,0,0,false};
        a.mapping[1]={1,1,0,0,false};a.mapping[2]={2,1,0,0,false};
        out->inputTensorAccessPattern[positions].mapping[0]={2,1,0,0,false};
        out->inputTensorAccessPattern[phase].allRequired=true;
        out->inputTensorAccessPattern[phase].sparseAccess=true;
        auto& result=out->outputTensorAccessPattern[0];result.mapping[0]={0,128,0,127,false};
        result.mapping[1]={1,1,0,0,false};result.mapping[2]={2,1,0,0,false};
    }
    out->kernel.paramsNr=0;
    const unsigned char* starts[]={
        &_binary___deepseek_v41_mla_stream_exp_publish_gaudi2_o_start,
        &_binary___deepseek_v41_mla_stream_exp_reuse_gaudi2_o_start,
        &_binary___deepseek_v41_mla_stream_finish_publish_gaudi2_o_start,
        &_binary___deepseek_v41_mla_stream_finish_reuse_gaudi2_o_start};
    const unsigned char* ends[]={
        &_binary___deepseek_v41_mla_stream_exp_publish_gaudi2_o_end,
        &_binary___deepseek_v41_mla_stream_exp_reuse_gaudi2_o_end,
        &_binary___deepseek_v41_mla_stream_finish_publish_gaudi2_o_end,
        &_binary___deepseek_v41_mla_stream_finish_reuse_gaudi2_o_end};
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=ends[op]-starts[op];
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,starts[op],out->kernel.elfSize);return GLUE_SUCCESS;
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
