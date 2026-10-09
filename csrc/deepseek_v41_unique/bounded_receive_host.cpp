// SPDX-License-Identifier: Apache-2.0
#include "tpc_kernel_lib_interface.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
extern unsigned char _binary___deepseek_v41_bounded_peer_receive_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_bounded_peer_receive_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_future_epoch_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_future_epoch_gaudi2_o_end;
namespace tpc_lib_api {struct _TensorManipulationSuggestion;}
namespace {
constexpr auto name="custom_deepseek_v41_bounded_peer_receive_gaudi2";
template<class T>T parent(const char* symbol) {
    static void* handle=[] {const char* path=std::getenv("VLLM_HPU_DSV41_BOUNDED_RECEIVE_PARENT_KERNEL");
        return path&&*path?dlopen(path,RTLD_NOW|RTLD_LOCAL):nullptr;}();
    return handle?reinterpret_cast<T>(dlsym(handle,symbol)):nullptr;
}
constexpr auto epoch_name="custom_deepseek_v41_future_epoch_gaudi2";
bool selected(const char* guid){return std::strcmp(guid,name)==0||std::strcmp(guid,epoch_name)==0;}
tpc_lib_api::GlueCodeReturn instantiate_epoch(const tpc_lib_api::HabanaKernelParams* p,
                                            tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(p->deviceId!=DEVICE_ID_GAUDI2||p->inputTensorNr!=2||p->outputTensorNr!=1||p->nodeParams.nodeParamsSize)
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto& value=p->inputTensors[0].geometry;
    const auto& flags=p->inputTensors[1].geometry;
    const auto& result=p->outputTensors[0].geometry;
    if(value.dataType!=DATA_BF16||value.dims!=2||value.maxSizes[1]!=1||
       value.maxSizes[0]<128||value.maxSizes[0]>6*20480||value.maxSizes[0]%128||
       flags.dataType!=DATA_I32||flags.dims!=1||flags.maxSizes[0]!=(1u<<24))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if(result.dataType!=value.dataType||result.dims!=2||result.maxSizes[0]!=value.maxSizes[0]||
       result.maxSizes[1]!=1)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank=1;out->indexSpaceGeometry[0]=value.maxSizes[0]/128;
    out->inputTensorAccessPattern[0].mapping[0]={0,128,0,127,false};
    out->inputTensorAccessPattern[0].mapping[1]={0,0,0,0,false};
    out->inputTensorAccessPattern[1].allRequired=true;
    out->outputTensorAccessPattern[0]=out->inputTensorAccessPattern[0];
    out->kernel.paramsNr=0;
    const auto capacity=out->kernel.elfSize;
    out->kernel.elfSize=&_binary___deepseek_v41_future_epoch_gaudi2_o_end-
                        &_binary___deepseek_v41_future_epoch_gaudi2_o_start;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,&_binary___deepseek_v41_future_epoch_gaudi2_o_start,out->kernel.elfSize);
    return GLUE_SUCCESS;
}
}
extern "C" {
uint64_t GetLibVersion(){auto fn=parent<tpc_lib_api::pfnGetLibVersion>("GetLibVersion");return fn?fn():0;}
tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId device,uint32_t* count,tpc_lib_api::GuidInfo* out) {
    using namespace tpc_lib_api;auto fn=parent<pfnGetKernelGuids>("GetKernelGuids");
    if(!fn||!count)return GLUE_FAILED;
    if(device!=DEVICE_ID_GAUDI2)return fn(device,count,out);
    uint32_t inherited=0;auto status=fn(device,&inherited,nullptr);if(status!=GLUE_SUCCESS)return status;
    const auto capacity=*count;*count=inherited+2;if(!out||!capacity)return GLUE_SUCCESS;
    if(capacity<*count)return GLUE_FAILED;
    std::memset(out,0,2*sizeof(*out));std::strcpy(out[0].name,name);std::strcpy(out[1].name,epoch_name);
    return fn(device,&inherited,out+2);
}
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(const tpc_lib_api::HabanaKernelParams* p,
                                               tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;if(!p||!out)return GLUE_FAILED;
    if(!selected(p->guid.name)){auto fn=parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel");return fn?fn(p,out):GLUE_FAILED;}
    if(std::strcmp(p->guid.name,epoch_name)==0)return instantiate_epoch(p,out);
    if(p->deviceId!=DEVICE_ID_GAUDI2||p->inputTensorNr!=4||p->outputTensorNr!=2||
       p->nodeParams.nodeParamsSize!=3*sizeof(int))return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto& local=p->inputTensors[0].geometry;
    const auto width=local.maxSizes[0],rows=local.maxSizes[1];
    if(local.dataType!=DATA_BF16||local.dims!=2||width!=5120||rows<1||rows>6)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto& table=p->inputTensors[1].geometry;const auto group=table.maxSizes[2];
    const auto point_capacity=table.maxSizes[3];
    if(table.dataType!=DATA_BF16||table.dims!=5||table.maxSizes[0]!=width||table.maxSizes[1]!=rows||
       (group!=2&&group!=4)||point_capacity<128||point_capacity>2048||
       (point_capacity&(point_capacity-1))||table.maxSizes[4]!=2||
       width*rows*group*point_capacity*2*sizeof(uint16_t)<=(uint64_t(48)<<20))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto& flags=p->inputTensors[2].geometry;const auto& epoch=p->inputTensors[3].geometry;
    if(flags.dataType!=DATA_I32||flags.dims!=1||flags.maxSizes[0]!=(1u<<24)||
       epoch.dataType!=DATA_I32||epoch.dims!=1||(epoch.maxSizes[0]!=1&&epoch.maxSizes[0]!=(1u<<24)))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto params=static_cast<const int*>(p->nodeParams.nodeParams);
    if(params[0]<0||params[0]>=128||params[1]<0||params[1]>=int(group)||params[2]<1||params[2]>65536)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for(unsigned i=0;i<2;++i){const auto& g=p->outputTensors[i].geometry;
        if(g.dataType!=(i?DATA_I32:DATA_BF16)||g.dims!=2||g.maxSizes[0]!=(i?width/128:width)||
           g.maxSizes[1]!=rows)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;}
    out->indexSpaceRank=2;out->indexSpaceGeometry[0]=width/128;out->indexSpaceGeometry[1]=rows;
    out->inputTensorAccessPattern[0].mapping[0]={0,128,0,127,false};
    out->inputTensorAccessPattern[0].mapping[1]={1,1,0,0,false};
    // Future-written tables cannot be pulled into SRAM before readiness.
    // The full table must exceed SRAM even for two ranks and two rows.
    // These non-affine full-table mappings deliberately exceed SRAM capacity.
    // Capability must prove no hidden early copy; never assume it from shape.
    out->inputTensorAccessPattern[1].allRequired=true;
    out->inputTensorAccessPattern[2].allRequired=true;
    out->inputTensorAccessPattern[3].allRequired=true;
    out->outputTensorAccessPattern[0]=out->inputTensorAccessPattern[0];
    out->outputTensorAccessPattern[1].mapping[0]={0,1,0,0,false};
    out->outputTensorAccessPattern[1].mapping[1]={1,1,0,0,false};
    out->kernel.paramsNr=3;std::memcpy(out->kernel.scalarParams,params,3*sizeof(int));
    const auto capacity=out->kernel.elfSize;
    out->kernel.elfSize=&_binary___deepseek_v41_bounded_peer_receive_gaudi2_o_end-
                        &_binary___deepseek_v41_bounded_peer_receive_gaudi2_o_start;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,&_binary___deepseek_v41_bounded_peer_receive_gaudi2_o_start,
                out->kernel.elfSize);return GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId device,
    const tpc_lib_api::ShapeInferenceParams* p,tpc_lib_api::ShapeInferenceOutput* out) {
    if(selected((p->pGuid?p->pGuid:&p->guid)->name))return tpc_lib_api::GLUE_SUCCESS;
    auto fn=parent<tpc_lib_api::pfnGetShapeInference>("GetShapeInference");return fn?fn(device,p,out):tpc_lib_api::GLUE_FAILED;
}
tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(const tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::NodeDataLayouts* layouts,uint32_t* count) {
    if(selected(p->guid.name)) {
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
    auto fn=parent<tpc_lib_api::pfnGetSupportedDataLayout>("GetSupportedDataLayouts");return fn?fn(p,layouts,count):tpc_lib_api::GLUE_FAILED;
}
tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(const tpc_lib_api::HabanaKernelParams* p,
    tpc_lib_api::_TensorManipulationSuggestion* suggestion) {
    if(selected(p->guid.name))return tpc_lib_api::GLUE_FAILED;
    using F=tpc_lib_api::GlueCodeReturn(*)(const tpc_lib_api::HabanaKernelParams*,
                                        tpc_lib_api::_TensorManipulationSuggestion*);
    auto fn=parent<F>("GetSuggestedManipulation");
    return fn?fn(p,suggestion):tpc_lib_api::GLUE_FAILED;
}

}
