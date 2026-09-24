// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_state_rows_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_state_rows_write_u8_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_state_rows_write_u8_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_state_rows_write_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_state_rows_write_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_state_rows_write_f32_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_state_rows_write_f32_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_state_rows_read_u8_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_state_rows_read_u8_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_state_rows_read_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_state_rows_read_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_state_rows_read_f32_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_state_rows_read_f32_gaudi2_o_end;
const char* DeepseekV41StateRowsGaudi2::names[6]={"custom_deepseek_v41_state_rows_write_u8_gaudi2","custom_deepseek_v41_state_rows_write_bf16_gaudi2","custom_deepseek_v41_state_rows_write_f32_gaudi2","custom_deepseek_v41_state_rows_read_u8_gaudi2","custom_deepseek_v41_state_rows_read_bf16_gaudi2","custom_deepseek_v41_state_rows_read_f32_gaudi2"};
using namespace tpc_lib_api;
GlueCodeReturn DeepseekV41StateRowsGaudi2::GetGcDefinitions(HabanaKernelParams* p,HabanaKernelInstantiation* out) {
    if(!p||!out||mode_>=6)return GLUE_FAILED;
    if(p->inputTensorNr!=3)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(p->outputTensorNr!=1)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const bool read=mode_>=3;
    const unsigned types[3]={DATA_U8,DATA_BF16,DATA_F32};
    const auto dtype=types[mode_%3];
    const auto& cache=p->inputTensors[0].geometry;
    const auto& rows=p->inputTensors[read?1:2].geometry;
    const auto& data=p->inputTensors[read?2:1].geometry;
    const auto& result=p->outputTensors[0].geometry;
    if(cache.dims!=2||cache.dataType!=dtype||cache.maxSizes[0]<1||cache.maxSizes[0]>32768||
       cache.maxSizes[1]<1||rows.dims!=1||rows.dataType!=DATA_I32||rows.maxSizes[0]<1||rows.maxSizes[0]>4096)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto batch=rows.maxSizes[0];
    if(read) {
        if(data.dims!=1||data.dataType!=DATA_I32||data.maxSizes[0]!=batch)return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(result.dims!=2||result.dataType!=dtype||result.maxSizes[0]!=cache.maxSizes[0]||result.maxSizes[1]!=batch)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    } else {
        if(data.dims!=2||data.dataType!=dtype||data.maxSizes[0]!=cache.maxSizes[0]||data.maxSizes[1]!=batch)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(result.dims!=1||result.dataType!=DATA_I32||result.maxSizes[0]!=batch)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->indexSpaceRank=1;out->indexSpaceGeometry[0]=batch;
    for(unsigned i=0;i<3;++i)out->inputTensorAccessPattern[i].allRequired=true;
    out->inputTensorAccessPattern[0].sparseAccess=true;
    auto& access=out->outputTensorAccessPattern[0];
    if(read){access.mapping[0]={0,0,0,float(cache.maxSizes[0]-1)};access.mapping[1]={0,1,0,0};}
    else access.mapping[0]={0,1,0,0};
    out->kernel.paramsNr=0;
unsigned char* starts[6]={&_binary___deepseek_v41_state_rows_write_u8_gaudi2_o_start,&_binary___deepseek_v41_state_rows_write_bf16_gaudi2_o_start,&_binary___deepseek_v41_state_rows_write_f32_gaudi2_o_start,&_binary___deepseek_v41_state_rows_read_u8_gaudi2_o_start,&_binary___deepseek_v41_state_rows_read_bf16_gaudi2_o_start,&_binary___deepseek_v41_state_rows_read_f32_gaudi2_o_start};
unsigned char* ends[6]={&_binary___deepseek_v41_state_rows_write_u8_gaudi2_o_end,&_binary___deepseek_v41_state_rows_write_bf16_gaudi2_o_end,&_binary___deepseek_v41_state_rows_write_f32_gaudi2_o_end,&_binary___deepseek_v41_state_rows_read_u8_gaudi2_o_end,&_binary___deepseek_v41_state_rows_read_bf16_gaudi2_o_end,&_binary___deepseek_v41_state_rows_read_f32_gaudi2_o_end};
    const auto size=ends[mode_]-starts[mode_];const auto capacity=out->kernel.elfSize;out->kernel.elfSize=size;
    if(capacity<size)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,starts[mode_],size);return GLUE_SUCCESS;
}
