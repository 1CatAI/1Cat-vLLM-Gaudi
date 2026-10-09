// SPDX-License-Identifier: Apache-2.0
#include "tpc_kernel_lib_interface.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>
extern unsigned char _binary___deepseek_v41_mla_fp16_publish_gather_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mla_fp16_publish_gather_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mla_fp16_reuse_gather_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mla_fp16_reuse_gather_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mla_fp16_softmax_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mla_fp16_softmax_gaudi2_o_end;
namespace tpc_lib_api {struct _TensorManipulationSuggestion;}
namespace {
constexpr const char* names[]={"custom_deepseek_v41_mla_fp16_publish_gather_gaudi2",
                              "custom_deepseek_v41_mla_fp16_reuse_gather_gaudi2",
                              "custom_deepseek_v41_mla_fp16_softmax_gaudi2"};
template<class T>T parent(const char* symbol) {
    static void* handle=[] {const char* path=std::getenv("VLLM_HPU_DSV41_MLA_FP16_PARENT_KERNEL");
        return path&&*path?dlopen(path,RTLD_NOW|RTLD_LOCAL):nullptr;}();
    return handle?reinterpret_cast<T>(dlsym(handle,symbol)):nullptr;
}
int mode(const char* name) {for(int i=0;i<3;++i)if(!std::strcmp(name,names[i]))return i;return -1;}
}
extern "C" {
uint64_t GetLibVersion() {auto fn=parent<tpc_lib_api::pfnGetLibVersion>("GetLibVersion");return fn?fn():0;}
tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId device,uint32_t* count,
                                         tpc_lib_api::GuidInfo* out) {
    using namespace tpc_lib_api;
    auto fn=parent<pfnGetKernelGuids>("GetKernelGuids");if(!fn||!count)return GLUE_FAILED;
    if(device!=DEVICE_ID_GAUDI2)return fn(device,count,out);
    uint32_t inherited=0;auto status=fn(device,&inherited,nullptr);if(status!=GLUE_SUCCESS)return status;
    const auto capacity=*count;*count=inherited+3;
    if(!out||!capacity)return GLUE_SUCCESS;
    if(capacity<*count)return GLUE_FAILED;
    for(int i=0;i<3;++i){std::memset(out+i,0,sizeof(*out));std::strcpy(out[i].name,names[i]);}
    return fn(device,&inherited,out+3);
}
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(const tpc_lib_api::HabanaKernelParams* p,
                                               tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(!p||!out)return GLUE_FAILED;
    const int selected=mode(p->guid.name);
    if(selected<0){auto fn=parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel");return fn?fn(p,out):GLUE_FAILED;}
    if(p->deviceId!=DEVICE_ID_GAUDI2 || p->outputTensorNr!=(selected==0?4u:selected==1?3u:1u) ||
       p->inputTensorNr!=(selected==0?6u:selected==1?5u:4u))return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const unsigned output_index=selected==2?0:1;
    if(p->outputTensors[output_index].geometry.dataType!=DATA_F16)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    auto mapped=*p;
    std::vector<Tensor> inputs(p->inputTensors,p->inputTensors+p->inputTensorNr);
    std::vector<Tensor> outputs(p->outputTensors,p->outputTensors+p->outputTensorNr);
    mapped.inputTensors=inputs.data();mapped.outputTensors=outputs.data();
    outputs[output_index].geometry.dataType=DATA_F32;
    constexpr const char* old_names[]={"custom_deepseek_v41_main_batch_publish_gather_gaudi2",
        "custom_deepseek_v41_swa_only_reuse_gather_gaudi2","custom_deepseek_v41_selected_mla_softmax_gaudi2"};
    std::strcpy(mapped.guid.name,old_names[selected]);
    const auto capacity=out->kernel.elfSize;
    out->kernel.elfSize=0;  // Ask only for exact parent mappings, not its ELF.
    auto inherited=parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel");
    if(!inherited)return GLUE_FAILED;
    const auto status=inherited(&mapped,out);
    if(status!=GLUE_INSUFFICIENT_ELF_BUFFER && status!=GLUE_SUCCESS)return status;
    unsigned char* starts[]={&_binary___deepseek_v41_mla_fp16_publish_gather_gaudi2_o_start,
        &_binary___deepseek_v41_mla_fp16_reuse_gather_gaudi2_o_start,
        &_binary___deepseek_v41_mla_fp16_softmax_gaudi2_o_start};
    unsigned char* ends[]={&_binary___deepseek_v41_mla_fp16_publish_gather_gaudi2_o_end,
        &_binary___deepseek_v41_mla_fp16_reuse_gather_gaudi2_o_end,
        &_binary___deepseek_v41_mla_fp16_softmax_gaudi2_o_end};
    out->kernel.elfSize=ends[selected]-starts[selected];
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,starts[selected],out->kernel.elfSize);return GLUE_SUCCESS;
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
