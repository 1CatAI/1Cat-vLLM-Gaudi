// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_expert_paired_decode_fp8_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_expert_paired_decode_fp8_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_expert_paired_decode_fp8_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41ExpertPairedDecodeFp8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(in->inputTensorNr!=6)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(in->outputTensorNr!=2)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const TensorDataType types[]={DATA_I32,DATA_I16,DATA_I16,DATA_I16,DATA_I16,DATA_BF16};
    for(unsigned i=0;i<6;++i)if(in->inputTensors[i].geometry.dataType!=types[i]) {
        in->inputTensors[i].geometry.dataType=types[i];return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const auto& ids=in->inputTensors[0].geometry;
    if(ids.dataType!=DATA_I32 || ids.dims!=2 || ids.maxSizes[0]!=2 || ids.maxSizes[1]!=1)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    uint64_t experts=0,n13=0,k13=0,n2=0,k2=0;
    for(unsigned i=0;i<2;++i) {
        const auto& q=in->inputTensors[1+2*i].geometry;
        const auto& s=in->inputTensors[2+2*i].geometry;
        const auto& y=in->outputTensors[i].geometry;
        if(q.dataType!=DATA_I16 || s.dataType!=DATA_I16 || y.dataType!=DATA_F8_143)return GLUE_INCOMPATIBLE_DATA_TYPE;
        if(q.dims!=3 || q.maxSizes[0]%8192 || !q.maxSizes[0] || !q.maxSizes[1] || q.maxSizes[1]>20 ||
           !q.maxSizes[2] || q.maxSizes[2]>384 || s.dims!=3 || s.maxSizes[1]!=q.maxSizes[1] ||
           s.maxSizes[2]!=q.maxSizes[2] || (s.maxSizes[0]*8!=q.maxSizes[0] && (s.maxSizes[0]-128)*16!=q.maxSizes[0]))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto n=q.maxSizes[1]*256,k=q.maxSizes[0]/64;
        if(y.dims!=3 || y.maxSizes[0]!=n || y.maxSizes[1]!=k || y.maxSizes[2]!=2)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if(i && experts!=q.maxSizes[2])return GLUE_INCOMPATIBLE_INPUT_SIZE;
        experts=q.maxSizes[2];
        if(i==0){n13=n;k13=k;}else{n2=n;k2=k;}
    }
    const auto& lut=in->inputTensors[5].geometry;
    if(lut.dims!=1 || lut.maxSizes[0]!=128 || lut.dataType!=DATA_BF16)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if(n2%n13 || k13!=n2 || n13!=k2*2 || k2%(k13/128))return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto nr=n2/n13,kt=k2/(k13/128);
    if(kt!=16 && kt!=32)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for(unsigned i=0;i<6;++i)out->inputTensorAccessPattern[i].allRequired=true;
    for(unsigned i=0;i<2;++i) {
        auto& a=out->outputTensorAccessPattern[i];
        a.mapping[0]={0,float(i==0?256:256*nr),0,float((i==0?256:256*nr)-1)};
        a.mapping[1]={2,float(i==0?128:kt),0,float((i==0?128:kt)-1)};
        a.mapping[2]={1,2,0,1};
    }
    out->indexSpaceRank=3;out->indexSpaceGeometry[0]=n13/256;
    out->indexSpaceGeometry[1]=1;out->indexSpaceGeometry[2]=k13/128;
    out->kernel.paramsNr=0;
    auto* first=&_binary___deepseek_v41_expert_paired_decode_fp8_gaudi2_o_start;
    auto* last=&_binary___deepseek_v41_expert_paired_decode_fp8_gaudi2_o_end;
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=last-first;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
