// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_prefix_layout_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_prefix_layout_r1_i32_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefix_layout_r1_i32_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_prefix_layout_r2_i32_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefix_layout_r2_i32_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41PrefixLayoutGaudi2::GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name,ratio_==1?"custom_deepseek_v41_prefix_layout_r1_i32_gaudi2"
                            :"custom_deepseek_v41_prefix_layout_r2_i32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn DeepseekV41PrefixLayoutGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(in->inputTensorNr!=3){in->inputTensorNr=3;return GLUE_INCOMPATIBLE_INPUT_COUNT;}
    if(in->outputTensorNr!=3){in->outputTensorNr=3;return GLUE_INCOMPATIBLE_OUTPUT_COUNT;}
    for(unsigned i=0;i<3;++i)
        if(in->inputTensors[i].geometry.dataType!=DATA_I32 || in->outputTensors[i].geometry.dataType!=DATA_I32)
            return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto& s=in->inputTensors[0].geometry;
    const auto& p=in->inputTensors[1].geometry;
    const auto& b=in->inputTensors[2].geometry;
    const auto& r=in->outputTensors[0].geometry;
    const auto& a=in->outputTensors[1].geometry;
    const auto& l=in->outputTensors[2].geometry;
    const auto tokens=s.maxSizes[1];
    if(s.dims!=2 || s.maxSizes[0]!=512 || !tokens || tokens>6 || p.dims!=1 || p.maxSizes[0]!=tokens ||
       b.dims!=1 || b.maxSizes[0]<8)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if(r.dims!=2 || r.maxSizes[0]!=768 || r.maxSizes[1]!=1 || a.dims!=2 || a.maxSizes[0]!=640 ||
       a.maxSizes[1]!=tokens || l.dims!=1 || l.maxSizes[0]!=tokens)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank=3;
    out->indexSpaceGeometry[0]=out->indexSpaceGeometry[1]=1;
    out->indexSpaceGeometry[2]=tokens;
    out->inputTensorAccessPattern[0].mapping[0]={0,0,0,511};
    out->inputTensorAccessPattern[0].mapping[1]={2,1,0,0};
    out->inputTensorAccessPattern[1].mapping[0]={2,1,0,0};
    out->inputTensorAccessPattern[2].allRequired=true;
    // Only the final token writes this shared row list. Its consumer waits
    // for the complete TPC producer, including all per-query index rows.
    out->outputTensorAccessPattern[0].allRequired=true;
    out->outputTensorAccessPattern[1].mapping[0]={0,0,0,639};
    out->outputTensorAccessPattern[1].mapping[1]={2,1,0,0};
    out->outputTensorAccessPattern[2].mapping[0]={2,1,0,0};
    out->kernel.paramsNr=0;
    const auto* start=ratio_==1?&_binary___deepseek_v41_prefix_layout_r1_i32_gaudi2_o_start
                              :&_binary___deepseek_v41_prefix_layout_r2_i32_gaudi2_o_start;
    const auto* end=ratio_==1?&_binary___deepseek_v41_prefix_layout_r1_i32_gaudi2_o_end
                            :&_binary___deepseek_v41_prefix_layout_r2_i32_gaudi2_o_end;
    const auto capacity=out->kernel.elfSize;
    out->kernel.elfSize=end-start;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,start,out->kernel.elfSize);
    return GLUE_SUCCESS;
}
