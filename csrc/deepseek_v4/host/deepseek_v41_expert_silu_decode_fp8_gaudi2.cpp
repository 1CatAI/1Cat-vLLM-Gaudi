// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_expert_silu_decode_fp8_gaudi2.hpp"
#include <cstring>
#include <initializer_list>
extern unsigned char _binary___deepseek_v41_expert_silu_decode_fp8_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_expert_silu_decode_fp8_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41ExpertSiluDecodeFp8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(in->inputTensorNr!=8)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(in->outputTensorNr!=3)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const TensorDataType types[]={DATA_I32,DATA_I16,DATA_I16,DATA_BF16,DATA_F32,DATA_F32,DATA_BF16,DATA_F32};
    for(unsigned i=0;i<8;++i)if(in->inputTensors[i].geometry.dataType!=types[i]) {
        in->inputTensors[i].geometry.dataType=types[i];return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const auto& ids=in->inputTensors[0].geometry;
    const auto& q=in->inputTensors[1].geometry;
    const auto& s=in->inputTensors[2].geometry;
    const auto& lut=in->inputTensors[3].geometry;
    if(ids.dims!=2 || ids.maxSizes[1]!=1 || !ids.maxSizes[0] || (ids.maxSizes[0]%2 && ids.maxSizes[0]!=3) ||
       q.dims!=3 || !q.maxSizes[0] || q.maxSizes[0]%8192 || !q.maxSizes[1] || q.maxSizes[1]>20 ||
       !q.maxSizes[2] || q.maxSizes[2]>384 || s.dims!=3 || s.maxSizes[1]!=q.maxSizes[1] ||
       s.maxSizes[2]!=q.maxSizes[2] || s.maxSizes[0]!=q.maxSizes[0]/16+128 ||
       lut.dims!=1 || lut.maxSizes[0]!=128)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto n=q.maxSizes[1]*256,k=q.maxSizes[0]/64,slots=ids.maxSizes[0];
    const unsigned route_tile=slots==3?3:2;
    const auto& product=in->inputTensors[4].geometry;
    const auto& sx=in->inputTensors[5].geometry;
    const auto& channel=in->inputTensors[6].geometry;
    const auto& router=in->inputTensors[7].geometry;
    if(product.dims!=3 || product.maxSizes[0]!=k*2 || product.maxSizes[1]!=1 || product.maxSizes[2]!=slots ||
       sx.dims!=2 || sx.maxSizes[0]!=1 || sx.maxSizes[1]!=slots || channel.dims!=3 ||
       channel.maxSizes[0]!=256 || channel.maxSizes[1]*256!=k*2 || channel.maxSizes[2]!=q.maxSizes[2] ||
       router.dims!=2 || router.maxSizes[0]!=slots || router.maxSizes[1]!=1 || k>2560)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for(unsigned i=0;i<3;++i) {
        const auto& y=in->outputTensors[i].geometry;
        const auto type=i==2?DATA_F32:DATA_F8_143;
        if(y.dataType!=type)return GLUE_INCOMPATIBLE_DATA_TYPE;
        if(y.dims!=3 || y.maxSizes[0]!=(i==0?n:i==1?k:1) ||
           y.maxSizes[1]!=(i==0?k:1) || y.maxSizes[2]!=slots)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    auto map=[](auto& a,unsigned dim,unsigned index,float step,float first,float last) {
        a.mapping[dim].indexSpaceDim=index;a.mapping[dim].a=step;
        a.mapping[dim].start_b=first;a.mapping[dim].end_b=last;
    };
    out->indexSpaceRank=3;out->indexSpaceGeometry[0]=n/256;
    out->indexSpaceGeometry[1]=slots/route_tile;out->indexSpaceGeometry[2]=k/128;
    for(unsigned i:{1u,2u,3u,5u,6u})out->inputTensorAccessPattern[i].allRequired=true;
    for(unsigned i:{0u,7u}) {
        map(out->inputTensorAccessPattern[i],0,1,route_tile,0,route_tile-1);
        map(out->inputTensorAccessPattern[i],1,0,0,0,0);
    }
    auto& p=out->inputTensorAccessPattern[4];
    map(p,0,0,0,0,k*2-1);map(p,1,0,0,0,0);map(p,2,1,route_tile,0,route_tile-1);
    auto& w=out->outputTensorAccessPattern[0];
    map(w,0,0,256,0,255);map(w,1,2,128,0,127);map(w,2,1,route_tile,0,route_tile-1);
    for(unsigned i:{1u,2u}) {
        auto& a=out->outputTensorAccessPattern[i];
        map(a,0,0,0,0,i==1?k-1:0);map(a,1,0,0,0,0);map(a,2,1,route_tile,0,route_tile-1);
    }
    out->kernel.paramsNr=0;
    auto* first=&_binary___deepseek_v41_expert_silu_decode_fp8_gaudi2_o_start;
    auto* last=&_binary___deepseek_v41_expert_silu_decode_fp8_gaudi2_o_end;
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=last-first;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
