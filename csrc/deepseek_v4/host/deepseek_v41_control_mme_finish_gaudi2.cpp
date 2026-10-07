// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_control_mme_finish_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_control_mme_finish_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_control_mme_finish_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41ControlMmeFinishGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(in->inputTensorNr!=2)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto tokens=in->inputTensors[0].geometry.maxSizes[1];
    if(tokens<1 || tokens>6)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for(unsigned i=0;i<3;++i) {
        const auto& t=i<2?in->inputTensors[i]:in->outputTensors[0];
        if(t.geometry.dataType!=(i==0?DATA_BF16:DATA_F32) || t.geometry.dims!=2 ||
           t.geometry.maxSizes[0]!=(i==0?20480:i==1?48:25) || t.geometry.maxSizes[1]!=tokens)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        auto& access=i<2?out->inputTensorAccessPattern[i]:out->outputTensorAccessPattern[0];
        access.mapping[0]={0,0,0,float(t.geometry.maxSizes[0]-1)};
        access.mapping[1]={0,1,0,0};
    }
    out->indexSpaceRank=1;out->indexSpaceGeometry[0]=tokens;
    std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,sizeof(float));out->kernel.paramsNr=1;
    auto* first=&_binary___deepseek_v41_control_mme_finish_gaudi2_o_start;
    auto* last=&_binary___deepseek_v41_control_mme_finish_gaudi2_o_end;
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=last-first;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
