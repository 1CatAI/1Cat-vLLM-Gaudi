// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_peer_post_norm_quant_gaudi2.hpp"
#include <cmath>
#include <cstring>
extern unsigned char _binary___deepseek_v41_peer_post_norm_quant_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_peer_post_norm_quant_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41PeerPostNormQuantGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize!=2*sizeof(float))return GLUE_FAILED;
    if(in->inputTensorNr!=6)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(in->outputTensorNr!=5)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const TensorDataType types[]={DATA_BF16,DATA_BF16,DATA_F32,DATA_F32,DATA_F32,DATA_BF16};
    for(unsigned i=0;i<6;++i)if(in->inputTensors[i].geometry.dataType!=types[i])return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto& v=in->inputTensors[0].geometry;const auto& r=in->inputTensors[1].geometry;
    const auto& post=in->inputTensors[2].geometry;const auto& comb=in->inputTensors[3].geometry;
    const auto& pre=in->inputTensors[4].geometry;const auto& w=in->inputTensors[5].geometry;
    if(v.dims!=3 || v.maxSizes[0]!=5120 || v.maxSizes[1]!=1 ||
       (v.maxSizes[2]!=1 && v.maxSizes[2]!=2 && v.maxSizes[2]!=4) ||
       r.dims!=3 || r.maxSizes[0]!=5120 || r.maxSizes[1]!=4 || r.maxSizes[2]!=1 ||
       post.dims!=2 || post.maxSizes[0]!=4 || post.maxSizes[1]!=1 ||
       pre.dims!=2 || pre.maxSizes[0]!=4 || pre.maxSizes[1]!=1 ||
       comb.dims!=3 || comb.maxSizes[0]!=4 || comb.maxSizes[1]!=4 || comb.maxSizes[2]!=1 ||
       w.dims!=1 || w.maxSizes[0]!=5120)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const TensorDataType outputs[]={DATA_BF16,DATA_BF16,DATA_BF16,DATA_F8_143,DATA_F32};
    for(unsigned i=0;i<5;++i) {
        const auto& y=in->outputTensors[i].geometry;
        if(y.dataType!=outputs[i])return GLUE_INCOMPATIBLE_DATA_TYPE;
        if(i==0 ? y.dims!=3 || y.maxSizes[0]!=5120 || y.maxSizes[1]!=4 || y.maxSizes[2]!=1 :
           y.dims!=2 || y.maxSizes[0]!=(i==4?1:5120) || y.maxSizes[1]!=1)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->outputTensorAccessPattern[i].allRequired=true;
    }
    const auto* scalar=static_cast<const float*>(in->nodeParams.nodeParams);
    if(!std::isnormal(scalar[0]) || scalar[0]<=0 || scalar[1]!=1.0f/5120)return GLUE_FAILED;
    for(unsigned i=0;i<6;++i)out->inputTensorAccessPattern[i].allRequired=true;
    out->indexSpaceRank=1;out->indexSpaceGeometry[0]=4;
    out->kernel.paramsNr=2;std::memcpy(out->kernel.scalarParams,scalar,2*sizeof(float));
    auto* first=&_binary___deepseek_v41_peer_post_norm_quant_gaudi2_o_start;
    auto* last=&_binary___deepseek_v41_peer_post_norm_quant_gaudi2_o_end;
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=last-first;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
