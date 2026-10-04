// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_kv_norm_reuse_gather_gaudi2.hpp"
#include <cstring>
#include <initializer_list>
extern unsigned char _binary___deepseek_v41_kv_norm_reuse_gather_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_kv_norm_reuse_gather_gaudi2_o_end;
namespace {
using namespace tpc_lib_api;
bool match(const Tensor& t,TensorDataType dtype,std::initializer_list<unsigned> sizes) {
    if(t.geometry.dataType!=dtype || t.geometry.dims!=sizes.size())return false;
    unsigned i=0;for(auto s:sizes){if(s && t.geometry.maxSizes[i]!=s)return false;++i;}return true;
}
void map(TensorAccessPattern& t,unsigned dim,unsigned axis,int a,int last) {
    t.mapping[dim]={axis,float(a),0,float(last),true};
}
}
tpc_lib_api::GlueCodeReturn DeepseekV41KvNormReuseGatherGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(in->inputTensorNr!=8){in->inputTensorNr=8;return GLUE_INCOMPATIBLE_INPUT_COUNT;}
    if(in->outputTensorNr!=3){in->outputTensorNr=3;return GLUE_INCOMPATIBLE_OUTPUT_COUNT;}
    if(!match(in->inputTensors[0],DATA_BF16,{512,1}) ||
       !match(in->inputTensors[1],DATA_BF16,{512}) ||
       !match(in->inputTensors[2],DATA_F32,{64,0}) ||
       !match(in->inputTensors[3],DATA_U8,{528,256}) ||
       !match(in->inputTensors[4],DATA_BF16,{512,640,1}) ||
       !match(in->inputTensors[5],DATA_F32,{640,1}) ||
       !match(in->inputTensors[6],DATA_I32,{1}) ||
       !match(in->inputTensors[7],DATA_I32,{1}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if(!match(in->outputTensors[0],DATA_BF16,{512,640,1}) ||
       !match(in->outputTensors[1],DATA_F32,{512,640,1}) ||
       !match(in->outputTensors[2],DATA_F32,{640,1}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank=2;out->indexSpaceGeometry[0]=4;out->indexSpaceGeometry[1]=640;
    // Spread the four FCD owners first, then split independent row ranges.
    // The current row's four codecs/norm tiles run on different TPCs.
    out->preferredSplitDim=1;
    for(unsigned i:{0u,2u,3u,6u,7u})out->inputTensorAccessPattern[i].allRequired=true;
    map(out->inputTensorAccessPattern[1],0,0,128,127);
    map(out->inputTensorAccessPattern[4],0,0,128,127);
    map(out->inputTensorAccessPattern[4],1,1,1,0);
    map(out->inputTensorAccessPattern[4],2,1,0,0);
    map(out->inputTensorAccessPattern[5],0,1,1,0);
    map(out->inputTensorAccessPattern[5],1,0,0,0);
    for(unsigned i=0;i<2;++i) {
        map(out->outputTensorAccessPattern[i],0,0,128,127);
        map(out->outputTensorAccessPattern[i],1,1,1,0);
        map(out->outputTensorAccessPattern[i],2,1,0,0);
    }
    map(out->outputTensorAccessPattern[2],0,1,1,0);
    map(out->outputTensorAccessPattern[2],1,0,0,0);
    struct Params{float epsilon;float inverse_width;};
    if(!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize!=sizeof(Params))return GLUE_FAILED;
    out->kernel.paramsNr=2;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,sizeof(Params));
    auto* begin=&_binary___deepseek_v41_kv_norm_reuse_gather_gaudi2_o_start;
    auto* end=&_binary___deepseek_v41_kv_norm_reuse_gather_gaudi2_o_end;
    unsigned capacity=out->kernel.elfSize;out->kernel.elfSize=end-begin;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,begin,out->kernel.elfSize);return GLUE_SUCCESS;
}
