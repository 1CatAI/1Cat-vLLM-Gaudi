// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_expert_pair_silu_quant_gaudi2.hpp"
#include "deepseek_v41_expert_n256_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_expert_pair_silu_quant_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_expert_pair_silu_quant_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41ExpertPairSiluQuantGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(in->inputTensorNr!=5 || in->inputTensors[0].geometry.maxSizes[2]%2)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto capacity=out->kernel.elfSize;
    const auto result=DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::SiluQuant).GetGcDefinitions(in,out);
    if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
    out->indexSpaceGeometry[0]/=2;
    for(unsigned i=0;i<5;++i) {
        if(i==0) {auto& a=out->inputTensorAccessPattern[i].mapping[2];a.a=2;a.end_b=1;}
        if(i==1 || i==4){auto& a=out->inputTensorAccessPattern[i].mapping[0];a.a=2;a.end_b=1;}
    }
    for(unsigned i=0;i<2;++i){auto& a=out->outputTensorAccessPattern[i].mapping[2];a.a=2;a.end_b=1;}
    auto* first=&_binary___deepseek_v41_expert_pair_silu_quant_gaudi2_o_start;
    auto* last=&_binary___deepseek_v41_expert_pair_silu_quant_gaudi2_o_end;
    out->kernel.elfSize=last-first;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
