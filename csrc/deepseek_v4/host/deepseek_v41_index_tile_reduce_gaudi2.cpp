// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_index_tile_reduce_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_index_reduce_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_index_reduce_gaudi2_o_end;
using namespace tpc_lib_api;
GlueCodeReturn DeepseekV41IndexTileReduceGaudi2::GetGcDefinitions(HabanaKernelParams* p,HabanaKernelInstantiation* out) {
    if(!p||!out)return GLUE_FAILED;
    if(p->inputTensorNr!=2)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(p->outputTensorNr!=1)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& dots=p->inputTensors[0].geometry;
    const auto& weights=p->inputTensors[1].geometry;
    if(dots.dims!=3||dots.dataType!=DATA_BF16||dots.maxSizes[0]<128||dots.maxSizes[0]>2048||
       dots.maxSizes[0]%128||dots.maxSizes[1]!=32||dots.maxSizes[2]<1||dots.maxSizes[2]>64||
       weights.dims!=2||weights.dataType!=DATA_BF16||weights.maxSizes[0]!=32||weights.maxSizes[1]!=dots.maxSizes[2])
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto& scores=p->outputTensors[0].geometry;
    if(scores.dims!=2||scores.dataType!=DATA_F32||scores.maxSizes[0]!=dots.maxSizes[0]||
       scores.maxSizes[1]!=dots.maxSizes[2])return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank=2;out->indexSpaceGeometry[0]=dots.maxSizes[0]/128;
    out->indexSpaceGeometry[1]=dots.maxSizes[2];
    auto& d=out->inputTensorAccessPattern[0];
    d.mapping[0]={0,128,0,127};d.mapping[1]={0,0,0,31};d.mapping[2]={1,1,0,0};
    auto& w=out->inputTensorAccessPattern[1];
    w.mapping[0]={0,0,0,31};w.mapping[1]={1,1,0,0};
    auto& s=out->outputTensorAccessPattern[0];
    s.mapping[0]={0,128,0,127};s.mapping[1]={1,1,0,0};
    out->kernel.paramsNr=0;
    const auto size=&_binary___deepseek_v41_index_reduce_gaudi2_o_end-&_binary___deepseek_v41_index_reduce_gaudi2_o_start;
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=size;
    if(capacity<size)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,&_binary___deepseek_v41_index_reduce_gaudi2_o_start,size);return GLUE_SUCCESS;
}
