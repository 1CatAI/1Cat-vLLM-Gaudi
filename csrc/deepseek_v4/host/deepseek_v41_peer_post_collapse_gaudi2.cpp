// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_peer_post_collapse_gaudi2.hpp"
#include <cmath>
#include <cstring>
extern unsigned char _binary___deepseek_v41_peer_post_collapse_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_peer_post_collapse_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41PeerPostCollapseGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(in->inputTensorNr!=5)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(in->outputTensorNr!=2)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const TensorDataType types[]={DATA_BF16,DATA_BF16,DATA_F32,DATA_F32,DATA_F32};
    for(unsigned i=0;i<5;++i)if(in->inputTensors[i].geometry.dataType!=types[i])return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto& v=in->inputTensors[0].geometry;const auto& r=in->inputTensors[1].geometry;
    const auto tokens = r.maxSizes[2];
    const auto& post=in->inputTensors[2].geometry;const auto& comb=in->inputTensors[3].geometry;
    const auto& pre=in->inputTensors[4].geometry;
    if(v.dims!=3 || v.maxSizes[0]!=5120 || v.maxSizes[1]!=tokens ||
       (v.maxSizes[2]!=1 && v.maxSizes[2]!=2 && v.maxSizes[2]!=4) ||
       r.dims!=3 || r.maxSizes[0]!=5120 || r.maxSizes[1]!=4 || (tokens<1 || tokens>6) ||
       post.dims!=2 || post.maxSizes[0]!=4 || post.maxSizes[1]!=tokens ||
       pre.dims!=2 || pre.maxSizes[0]!=4 || pre.maxSizes[1]!=tokens ||
       comb.dims!=3 || comb.maxSizes[0]!=4 || comb.maxSizes[1]!=4 || comb.maxSizes[2]!=tokens)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for(unsigned i=0;i<2;++i) {
        const auto& y=in->outputTensors[i].geometry;
        if(y.dataType!=DATA_BF16)return GLUE_INCOMPATIBLE_DATA_TYPE;
        if(i==0 ? y.dims!=3 || y.maxSizes[0]!=5120 || y.maxSizes[1]!=4 || y.maxSizes[2]!=tokens :
           y.dims!=2 || y.maxSizes[0]!=5120 || y.maxSizes[1]!=tokens)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    for(unsigned i=0;i<7;++i) {
        const auto& g=(i<5 ? in->inputTensors[i] : in->outputTensors[i-5]).geometry;
        auto& ap=(i<5 ? out->inputTensorAccessPattern[i] : out->outputTensorAccessPattern[i-5]);
        for(unsigned d=0;d<g.dims;++d) {
            if(i==0 && d==2) ap.mapping[d]={0,0,0,float(g.maxSizes[d]-1),false};
            else if((i==0 && d==1) || (i!=0 && d==g.dims-1)) ap.mapping[d]={1,1,0,0,false};
            else if(d==0 && g.dataType==DATA_BF16) ap.mapping[d]={0,128,0,127,false};
            else ap.mapping[d]={0,0,0,float(g.maxSizes[d]-1),false};
        }
    }
    out->indexSpaceRank=2;out->indexSpaceGeometry[0]=40;out->indexSpaceGeometry[1]=tokens;
    out->kernel.paramsNr=0;
    auto* first=&_binary___deepseek_v41_peer_post_collapse_gaudi2_o_start;
    auto* last=&_binary___deepseek_v41_peer_post_collapse_gaudi2_o_end;
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=last-first;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
