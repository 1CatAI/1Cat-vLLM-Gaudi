// SPDX-License-Identifier: Apache-2.0
#include "tpc_kernel_lib_interface.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }
extern unsigned char _binary___deepseek_v41_w2_channel_decode_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_w2_channel_decode_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_w2_channel_reduce_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_w2_channel_reduce_gaudi2_o_end;
namespace {
constexpr const char* names[]={"custom_deepseek_v41_w2_channel_decode_gaudi2",
                              "custom_deepseek_v41_w2_channel_reduce_gaudi2"};
constexpr const char* originals[]={"custom_deepseek_v41_expert_w2_split_scale_gaudi2",
                                  "custom_deepseek_v41_w2_reduce_n256_gaudi2"};
int kind(const char* name) { for(int i=0;i<2;++i)if(!std::strcmp(name,names[i]))return i;return -1; }
template<class T>T parent(const char* symbol) {
    static void* handle=[] {
        const char* path=std::getenv("VLLM_HPU_DSV41_W2_CHANNELS_PARENT_KERNEL");
        return path&&*path?dlopen(path,RTLD_NOW|RTLD_LOCAL):nullptr;
    }();
    return handle?reinterpret_cast<T>(dlsym(handle,symbol)):nullptr;
}
}
extern "C" {
uint64_t GetLibVersion(){auto f=parent<tpc_lib_api::pfnGetLibVersion>("GetLibVersion");return f?f():0;}
tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId d,uint32_t* count,
                                         tpc_lib_api::GuidInfo* out) {
    using namespace tpc_lib_api;
    auto f=parent<pfnGetKernelGuids>("GetKernelGuids");if(!f||!count)return GLUE_FAILED;
    if(d!=DEVICE_ID_GAUDI2)return f(d,count,out);
    uint32_t inherited=0;auto s=f(d,&inherited,nullptr);if(s!=GLUE_SUCCESS)return s;
    const auto capacity=*count;*count=inherited+2;
    if(!out||!capacity)return GLUE_SUCCESS;
    if(capacity<*count)return GLUE_FAILED;
    for(int i=0;i<2;++i){std::memset(out+i,0,sizeof(*out));std::strcpy(out[i].name,names[i]);}
    return f(d,&inherited,out+2);
}
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(const tpc_lib_api::HabanaKernelParams* p,
                                               tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(!p||!out)return GLUE_FAILED;
    auto f=parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel");if(!f)return GLUE_FAILED;
    const int op=kind(p->guid.name);if(op<0)return f(p,out);
    if(p->deviceId!=DEVICE_ID_GAUDI2||p->inputTensorNr!=(op?4:5)||p->outputTensorNr!=1)
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    auto inherited=*p;std::strcpy(inherited.guid.name,originals[op]);
    std::vector<Tensor> inputs(p->inputTensors,p->inputTensors+p->inputTensorNr);
    std::vector<Tensor> outputs(p->outputTensors,p->outputTensors+1);
    const auto& g=op?inputs[0].geometry:outputs[0].geometry;
    if(g.dims!=5||g.maxSizes[0]!=256||g.maxSizes[3]!=20||g.maxSizes[2]!=6||
       g.maxSizes[4]<2||g.maxSizes[4]>6||(op?g.maxSizes[1]!=1:
        (!g.maxSizes[1]||g.maxSizes[1]%128||g.maxSizes[1]>5120)))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto tokens=g.maxSizes[4],slots=tokens*6;
    if(!op) {
        const auto& ids=inputs[0].geometry;
        if(ids.dims!=2||ids.maxSizes[0]!=6||ids.maxSizes[1]!=tokens)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        inputs[0].geometry.maxSizes[0]=inputs[0].geometry.minSizes[0]=slots;
        inputs[0].geometry.maxSizes[1]=inputs[0].geometry.minSizes[1]=1;
    }
    auto& virtualShape=op?inputs[0].geometry:outputs[0].geometry;
    virtualShape.dims=3;virtualShape.maxSizes[0]=5120;virtualShape.minSizes[0]=5120;
    virtualShape.maxSizes[2]=virtualShape.minSizes[2]=slots;
    virtualShape.maxSizes[3]=virtualShape.minSizes[3]=1;
    virtualShape.maxSizes[4]=virtualShape.minSizes[4]=1;
    inherited.inputTensors=inputs.data();inherited.outputTensors=outputs.data();
    const auto requested=out->kernel;
    out->kernel.elfSize=0;out->kernel.kernelElf=nullptr;
    const auto status=f(&inherited,out);out->kernel=requested;
    if(status!=GLUE_SUCCESS&&status!=GLUE_INSUFFICIENT_ELF_BUFFER)return status;
    if(op) {
        auto& a=out->inputTensorAccessPattern[0];a={};
        a.mapping[0]={0,0,0,255,false};
        a.mapping[1]={0,0,0,0,false};
        a.mapping[2]={1,0,0,5,false};
        a.mapping[3]={0,1,0,0,false};
        a.mapping[4]={1,1,0,0,false};
    } else {
        const auto original=out->outputTensorAccessPattern[0];
        if(original.mapping[2].a!=1)return GLUE_FAILED;
        auto& a=out->outputTensorAccessPattern[0];a={};
        a.mapping[0]={0,0,0,255,false};
        a.mapping[1]=original.mapping[1];
        a.mapping[2]=original.mapping[2];
        a.mapping[3]={0,1,0,0,false};
        a.mapping[4]={3,1,0,0,false};
        out->indexSpaceRank=4;out->indexSpaceGeometry[1]=6;out->indexSpaceGeometry[3]=tokens;
        auto& ids=out->inputTensorAccessPattern[0];ids={};
        ids.mapping[0]={1,1,0,0,false};ids.mapping[1]={3,1,0,0,false};
    }
    const auto* first=op?&_binary___deepseek_v41_w2_channel_reduce_gaudi2_o_start:
                         &_binary___deepseek_v41_w2_channel_decode_gaudi2_o_start;
    const auto* last=op?&_binary___deepseek_v41_w2_channel_reduce_gaudi2_o_end:
                        &_binary___deepseek_v41_w2_channel_decode_gaudi2_o_end;
    out->kernel.elfSize=last-first;
    if(requested.elfSize<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId d,
    const tpc_lib_api::ShapeInferenceParams* p,tpc_lib_api::ShapeInferenceOutput* out) {
    if(kind((p->pGuid?p->pGuid:&p->guid)->name)>=0)return tpc_lib_api::GLUE_SUCCESS;
    auto f=parent<tpc_lib_api::pfnGetShapeInference>("GetShapeInference");return f?f(d,p,out):tpc_lib_api::GLUE_FAILED;
}
tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(const tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::NodeDataLayouts* layouts,uint32_t* count) {
    auto f=parent<tpc_lib_api::pfnGetSupportedDataLayout>("GetSupportedDataLayouts");if(!f)return tpc_lib_api::GLUE_FAILED;
    if(kind(p->guid.name)<0)return f(p,layouts,count);
    if(!count)return tpc_lib_api::GLUE_FAILED;
    *count=1;
    if(layouts){
        for(unsigned i=0;i<layouts->inputTensorNr;++i)std::memset(layouts->inputs[i].layout,'x',sizeof(layouts->inputs[i].layout));
        for(unsigned i=0;i<layouts->outputTensorNr;++i)std::memset(layouts->outputs[i].layout,'x',sizeof(layouts->outputs[i].layout));
    }
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(const tpc_lib_api::HabanaKernelParams* p,
                                                   tpc_lib_api::_TensorManipulationSuggestion* s) {
    if(kind(p->guid.name)>=0)return tpc_lib_api::GLUE_FAILED;
    using F=tpc_lib_api::GlueCodeReturn(*)(const tpc_lib_api::HabanaKernelParams*,tpc_lib_api::_TensorManipulationSuggestion*);
    auto f=parent<F>("GetSuggestedManipulation");return f?f(p,s):tpc_lib_api::GLUE_FAILED;
}
}
