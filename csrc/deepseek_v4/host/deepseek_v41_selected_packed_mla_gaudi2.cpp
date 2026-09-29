// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_selected_packed_mla_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_selected_packed_mla_gather_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_packed_mla_gather_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_selected_packed_mla_vector_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_selected_packed_mla_vector_gaudi2_o_end;
using namespace tpc_lib_api;
GlueCodeReturn DeepseekV41SelectedPackedMlaGaudi2::GetGcDefinitions(HabanaKernelParams* in,HabanaKernelInstantiation* out) {
    if(!in||!out)return GLUE_FAILED;
    if(in->inputTensorNr!=5)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(in->outputTensorNr!=3)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& swa=in->inputTensors[0].geometry;
    const auto& main=in->inputTensors[1].geometry;
    const auto& rows=in->inputTensors[2].geometry;
    const auto& ids=in->inputTensors[3].geometry;
    const auto& lengths=in->inputTensors[4].geometry;
    if(swa.dims!=2||swa.dataType!=DATA_U8||swa.maxSizes[0]!=528||!swa.maxSizes[1]||swa.maxSizes[1]>16384||
       main.dims!=2||main.dataType!=DATA_U8||main.maxSizes[0]!=288||!main.maxSizes[1]||main.maxSizes[1]>0x7fffbfffULL||
       rows.dims!=2||rows.dataType!=DATA_I32||rows.maxSizes[1]!=1||!rows.maxSizes[0]||rows.maxSizes[0]>40960||
       ids.dims!=2||ids.dataType!=DATA_I32||!ids.maxSizes[0]||ids.maxSizes[0]>640||ids.maxSizes[0]%64||
       !ids.maxSizes[1]||ids.maxSizes[1]>64||lengths.dims!=1||lengths.dataType!=DATA_I32||
       lengths.maxSizes[0]!=ids.maxSizes[1])return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto width=ids.maxSizes[0], batch=ids.maxSizes[1];
    out->indexSpaceRank=2;
    out->indexSpaceGeometry[0]=width;out->indexSpaceGeometry[1]=batch;
    auto map=[](TensorAccessPattern& p,unsigned dim,unsigned axis,int a,int last){
        p.mapping[dim]={axis,float(a),0,float(last)};
    };
    for(unsigned i=0;i<3;++i)out->inputTensorAccessPattern[i].allRequired=true;
    map(out->inputTensorAccessPattern[3],0,0,1,0);
    map(out->inputTensorAccessPattern[3],1,1,1,0);
    map(out->inputTensorAccessPattern[4],0,1,1,0);
    for(unsigned i=0;i<3;++i){
        const auto& output=in->outputTensors[i].geometry;
        if(i<2){
            if(output.dims!=3||output.dataType!=(i?DATA_F32:DATA_BF16)||output.maxSizes[0]!=512||
               output.maxSizes[1]!=width||output.maxSizes[2]!=batch)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            map(out->outputTensorAccessPattern[i],0,0,0,511);
            map(out->outputTensorAccessPattern[i],1,0,1,0);
            map(out->outputTensorAccessPattern[i],2,1,1,0);
        }else{
            if(output.dims!=2||output.dataType!=DATA_F32||output.maxSizes[0]!=width||output.maxSizes[1]!=batch)
                return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            map(out->outputTensorAccessPattern[i],0,0,1,0);
            map(out->outputTensorAccessPattern[i],1,1,1,0);
        }
    }
    out->kernel.paramsNr=0;
    const auto* first=vector_ ? &_binary___deepseek_v41_selected_packed_mla_vector_gaudi2_o_start
                             : &_binary___deepseek_v41_selected_packed_mla_gather_gaudi2_o_start;
    const auto* last=vector_ ? &_binary___deepseek_v41_selected_packed_mla_vector_gaudi2_o_end
                            : &_binary___deepseek_v41_selected_packed_mla_gather_gaudi2_o_end;
    const unsigned capacity=out->kernel.elfSize;out->kernel.elfSize=last-first;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
