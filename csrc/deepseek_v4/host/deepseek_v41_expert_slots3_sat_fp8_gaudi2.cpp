// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_expert_slots3_sat_fp8_gaudi2.hpp"
#include "deepseek_v41_expert_n256_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_expert_slots3_sat_fp8_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_expert_slots3_sat_fp8_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41ExpertSlots3SatFp8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(in->inputTensorNr!=4)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(in->inputTensors[0].geometry.maxSizes[0]!=3)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto capacity=out->kernel.elfSize;
    const auto result=DeepseekV41ExpertN256Gaudi2(DeepseekV41ExpertN256Gaudi2::FP8Sat).GetGcDefinitions(in,out);
    if(result!=GLUE_SUCCESS && result!=GLUE_INSUFFICIENT_ELF_BUFFER)return result;
    out->indexSpaceGeometry[1]=1;
    auto& id=out->inputTensorAccessPattern[0].mapping[0];
    id.indexSpaceDim=1;id.a=3;id.start_b=0;id.end_b=2;
    auto& slot=out->outputTensorAccessPattern[0].mapping[2];
    slot.indexSpaceDim=1;slot.a=3;slot.start_b=0;slot.end_b=2;
    auto* first=&_binary___deepseek_v41_expert_slots3_sat_fp8_gaudi2_o_start;
    auto* last=&_binary___deepseek_v41_expert_slots3_sat_fp8_gaudi2_o_end;
    out->kernel.elfSize=last-first;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
