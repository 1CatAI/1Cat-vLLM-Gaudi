// SPDX-License-Identifier: Apache-2.0
// Additive provider: delegate every existing GUID to the locked parent binary.
#include "tpc_kernel_lib_interface.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>

extern unsigned char _binary___deepseek_v41_qk_flat_publish_softmax_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_qk_flat_publish_softmax_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_qk_flat_reuse_softmax_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_qk_flat_reuse_softmax_gaudi2_o_end;
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }

namespace {
using namespace tpc_lib_api;
constexpr const char* names[] = {
    "custom_deepseek_v41_qk_flat_publish_softmax_gaudi2",
    "custom_deepseek_v41_qk_flat_reuse_softmax_gaudi2"};
int kind(const char* name) {
    for (int i=0;i<2;++i) if(!std::strcmp(name,names[i])) return i;
    return -1;
}
template<class T> T parent(const char* name) {
    static void* handle=[] {
        const char* path=std::getenv("VLLM_HPU_DSV41_QK_PARENT_KERNEL");
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
    const auto capacity=*count;*count=inherited+2;
    if(!out||!capacity)return GLUE_SUCCESS;
    if(capacity<*count)return GLUE_FAILED;
    std::memset(out,0,2*sizeof(*out));for(unsigned i=0;i<2;++i)std::strcpy(out[i].name,names[i]);
    return fn(device,&inherited,out+2);
}
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(const tpc_lib_api::HabanaKernelParams* p,
                                               tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(!p||!out)return GLUE_FAILED;
    const int op=kind(p->guid.name);
    if(op<0){auto fn=parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel");return fn?fn(p,out):GLUE_FAILED;}
    if(p->deviceId!=DEVICE_ID_GAUDI2||p->inputTensorNr!=5||p->outputTensorNr!=1)
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto& g=p->outputTensors[0].geometry;
    const auto heads=g.maxSizes[1],tokens=g.maxSizes[2];
    const uint64_t first=op?128:640,second=op?512:640;
    if(tokens<2||tokens>6||heads<1||heads>64||
       !shape(p->outputTensors[0],{640,heads,tokens})||
       !shape(p->inputTensors[0],{first*tokens,heads,tokens})||
       !shape(p->inputTensors[1],{second*tokens,heads,tokens})||
       !shape(p->inputTensors[2],{640,tokens})||!shape(p->inputTensors[3],{heads})||
       !shape(p->inputTensors[4],{1}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank=2;out->indexSpaceGeometry[0]=heads;out->indexSpaceGeometry[1]=tokens;
    // A diagonal couples two dimensions of the dense score matrix. The
    // current slicer aggregates this as a Cartesian tensor and asserts when
    // slices overlap. A conservative full operand requirement is correct;
    // it avoids promising an unsupported diagonal slice relation.
    for(unsigned i=0;i<2;++i)out->inputTensorAccessPattern[i].allRequired=true;
    auto& mask=out->inputTensorAccessPattern[2];
    mask.mapping[0]={0,0,0,639,false};mask.mapping[1]={1,1,0,0,false};
    out->inputTensorAccessPattern[3].mapping[0]={0,1,0,0,false};
    out->inputTensorAccessPattern[4].allRequired=true;
    auto& a=out->outputTensorAccessPattern[0];
    a.mapping[0]={0,0,0,639,false};a.mapping[1]={0,1,0,0,false};a.mapping[2]={1,1,0,0,false};
    out->kernel.paramsNr=0;
    const auto* first_elf=op?&_binary___deepseek_v41_qk_flat_reuse_softmax_gaudi2_o_start:
                            &_binary___deepseek_v41_qk_flat_publish_softmax_gaudi2_o_start;
    const auto* last_elf=op?&_binary___deepseek_v41_qk_flat_reuse_softmax_gaudi2_o_end:
                           &_binary___deepseek_v41_qk_flat_publish_softmax_gaudi2_o_end;
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=last_elf-first_elf;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first_elf,out->kernel.elfSize);return GLUE_SUCCESS;
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
