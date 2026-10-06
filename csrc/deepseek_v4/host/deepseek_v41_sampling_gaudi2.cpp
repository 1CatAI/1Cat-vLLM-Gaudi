// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_sampling_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_sampling_unpack_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_sampling_unpack_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_sampling_mask_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_sampling_mask_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_sampling_select_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_sampling_select_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41SamplingGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p,tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(mode_>2)return GLUE_FAILED;
    const unsigned ni[]={1,6,4},no[]={4,2,1};
    if(p->inputTensorNr!=ni[mode_])return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(p->outputTensorNr!=no[mode_])return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if(!p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize!=sizeof(DeepseekV41SamplingParams))return GLUE_FAILED;
    const auto params=*static_cast<const DeepseekV41SamplingParams*>(p->nodeParams.nodeParams);
    const int columns=params.columns,ranks=params.ranks,width=params.width;
    const auto batch=p->inputTensors[0].geometry.maxSizes[1];
    if(!batch || batch>64 || columns<64 || columns>4096 || columns%64)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    auto shape=[&](const auto& g,auto dtype,unsigned n) {
        return g.dataType==dtype && g.dims==2 && g.maxSizes[0]==n && g.maxSizes[1]==batch;
    };
    if(mode_==0) {
        if(ranks<2 || ranks>8 || width<64 || width>256 || width%64 || columns!=ranks*width ||
           !shape(p->inputTensors[0].geometry,DATA_F32,ranks*(3+2*width)))return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(p->outputTensors[0].geometry,DATA_F32,columns) ||
           !shape(p->outputTensors[1].geometry,DATA_I32,columns) ||
           !shape(p->outputTensors[2].geometry,DATA_F32,ranks*3) ||
           !shape(p->outputTensors[3].geometry,DATA_F32,ranks))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank=2;out->indexSpaceGeometry[0]=ranks;out->indexSpaceGeometry[1]=batch;
        out->inputTensorAccessPattern[0].mapping[0]={0,3+2*width,0,2+2*width};
        out->inputTensorAccessPattern[0].mapping[1]={1,1,0,0};
        const int spans[]={width,width,3,1};
        for(unsigned i=0;i<4;++i) {
            out->outputTensorAccessPattern[i].mapping[0]={0,spans[i],0,spans[i]-1};
            out->outputTensorAccessPattern[i].mapping[1]={1,1,0,0};
        }
    } else {
        if(mode_==1) {
            if(ranks<2 || ranks>8 || columns!=ranks*width ||
               !shape(p->inputTensors[0].geometry,DATA_F32,columns) ||
               !shape(p->inputTensors[1].geometry,DATA_F32,columns) ||
               !shape(p->inputTensors[2].geometry,DATA_F32,columns) ||
               !shape(p->inputTensors[3].geometry,DATA_F32,ranks) ||
               !shape(p->inputTensors[4].geometry,DATA_F32,4) ||
               !shape(p->inputTensors[5].geometry,DATA_F32,1))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if(!shape(p->outputTensors[0].geometry,DATA_F32,columns) ||
               !shape(p->outputTensors[1].geometry,DATA_I32,1))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        } else {
            if(!shape(p->inputTensors[0].geometry,DATA_F32,columns) ||
               !shape(p->inputTensors[1].geometry,DATA_I32,columns) ||
               !shape(p->inputTensors[2].geometry,DATA_F32,4) ||
               !shape(p->inputTensors[3].geometry,DATA_I32,1))return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if(!shape(p->outputTensors[0].geometry,DATA_I32,1))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        }
        out->indexSpaceRank=1;out->indexSpaceGeometry[0]=batch;
        for(unsigned i=0;i<ni[mode_];++i) {
            out->inputTensorAccessPattern[i].mapping[0]={0,0,0,int(p->inputTensors[i].geometry.maxSizes[0])-1};
            out->inputTensorAccessPattern[i].mapping[1]={0,1,0,0};
        }
        for(unsigned i=0;i<no[mode_];++i) {
            out->outputTensorAccessPattern[i].mapping[0]={0,0,0,int(p->outputTensors[i].geometry.maxSizes[0])-1};
            out->outputTensorAccessPattern[i].mapping[1]={0,1,0,0};
        }
    }
    out->kernel.paramsNr=3;std::memcpy(out->kernel.scalarParams,&params,sizeof(params));
    const unsigned char* starts[]={&_binary___deepseek_v41_sampling_unpack_gaudi2_o_start,
        &_binary___deepseek_v41_sampling_mask_gaudi2_o_start,&_binary___deepseek_v41_sampling_select_gaudi2_o_start};
    const unsigned char* ends[]={&_binary___deepseek_v41_sampling_unpack_gaudi2_o_end,
        &_binary___deepseek_v41_sampling_mask_gaudi2_o_end,&_binary___deepseek_v41_sampling_select_gaudi2_o_end};
    const auto available=out->kernel.elfSize;out->kernel.elfSize=ends[mode_]-starts[mode_];
    if(available<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,starts[mode_],out->kernel.elfSize);return GLUE_SUCCESS;
}
