// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_mhc_mme_post_collapse_gaudi2.hpp"
#include <cstring>
#include <initializer_list>
extern unsigned char _binary___deepseek_v41_mhc_mme_post_collapse_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mhc_mme_post_collapse_gaudi2_o_end;
namespace {
bool shape(const tpc_lib_api::Tensor& t, tpc_lib_api::TensorDataType dtype, std::initializer_list<uint64_t> sizes) {
    if(t.geometry.dataType!=dtype || t.geometry.dims!=sizes.size())return false;
    unsigned d=0;for(auto size:sizes)if(t.geometry.maxSizes[d++]!=size)return false;return true;
}
}
tpc_lib_api::GlueCodeReturn DeepseekV41MhcMmePostCollapseGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(in->inputTensorNr!=5)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(in->outputTensorNr!=3)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto dims=in->inputTensors[0].geometry.dims;
    const auto tokens=in->inputTensors[0].geometry.maxSizes[1];
    const auto ranks=dims==3?in->inputTensors[0].geometry.maxSizes[2]:1;
    if(tokens<1 || tokens>6 || (dims!=2 && dims!=3) || (dims==3 && (ranks<2 || ranks>8)))return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const bool value_ok=dims==2?shape(in->inputTensors[0],DATA_BF16,{5120,tokens}):shape(in->inputTensors[0],DATA_BF16,{5120,tokens,ranks});
    if(!value_ok || !shape(in->inputTensors[1],DATA_BF16,{5120,4,tokens}) || (!shape(in->inputTensors[2],DATA_F32,{48,tokens}) && !shape(in->inputTensors[2],DATA_F32,{25,tokens})) ||
       !shape(in->inputTensors[3],DATA_F32,{3}) || !shape(in->inputTensors[4],DATA_F32,{24}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if(!shape(in->outputTensors[0],DATA_BF16,{5120,4,tokens}) || !shape(in->outputTensors[1],DATA_BF16,{5120,tokens}) ||
       !shape(in->outputTensors[2],DATA_F32,{24,tokens}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    for(unsigned i=0;i<5;++i)out->inputTensorAccessPattern[i].allRequired=true;
    for(unsigned i=0;i<2;++i) {
        auto& a=out->outputTensorAccessPattern[i];a.mapping[0]={0,128,0,127};
        if(i==0)a.mapping[1]={0,0,0,3};
        a.mapping[i==0?2:1]={1,1,0,0};
    }
    out->outputTensorAccessPattern[2].allRequired=true;
    out->indexSpaceRank=2;out->indexSpaceGeometry[0]=40;out->indexSpaceGeometry[1]=tokens;
    std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,sizeof(float));out->kernel.paramsNr=1;
    auto* first=&_binary___deepseek_v41_mhc_mme_post_collapse_gaudi2_o_start;
    auto* last=&_binary___deepseek_v41_mhc_mme_post_collapse_gaudi2_o_end;
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=last-first;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
