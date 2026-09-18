// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_index_gaudi2.hpp"
#include <cstring>
#define ELF(name) extern unsigned char _binary___##name##_o_start, _binary___##name##_o_end;
ELF(deepseek_v41_index_scores_gaudi2)
ELF(deepseek_v41_index_threshold_gaudi2)
ELF(deepseek_v41_index_emit_gaudi2)
using namespace tpc_lib_api;
GlueCodeReturn DeepseekV41IndexGaudi2::GetGcDefinitions(HabanaKernelParams* p,HabanaKernelInstantiation* out) {
    if (!p || !out || mode_>2) return GLUE_FAILED;
    if(p->inputTensorNr!=(mode_==0?6u:mode_==1?2u:4u)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(p->outputTensorNr!=(mode_==0?2u:1u)) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if(!p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize!=3*sizeof(int)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto* param=static_cast<const int*>(p->nodeParams.nodeParams);
    if((param[0]!=1&&param[0]!=2)||param[1]<0||param[1]>1||param[2]<0||param[2]>1)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto match=[](const TensorGeometry& t, unsigned dim, unsigned width, TensorDataType type) {
        return t.dims==dim && t.maxSizes[0]==width && t.dataType==type;
    };
    if(mode_==0) {
        const auto& q=p->inputTensors[0].geometry;
        if(!match(q,3,128,DATA_BF16)||q.maxSizes[1]!=32||q.maxSizes[2]!=1 ||
           !match(p->inputTensors[1].geometry,2,32,DATA_BF16)||p->inputTensors[1].geometry.maxSizes[1]!=1 ||
           !match(p->inputTensors[2].geometry,2,68,DATA_U8) ||
           p->inputTensors[3].geometry.dataType!=DATA_I32 ||p->inputTensors[3].geometry.dims!=1 ||
           !match(p->inputTensors[4].geometry,1,1,DATA_I32) ||
           !match(p->inputTensors[5].geometry,2,2048,DATA_I32)||p->inputTensors[5].geometry.maxSizes[1]!=1)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto& score=p->outputTensors[0].geometry;
        if(score.dims!=1||score.dataType!=DATA_F32||score.maxSizes[0]%64 ||
           !match(p->outputTensors[1].geometry,1,(score.maxSizes[0]/8+63)/64*64,DATA_F32))
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    } else {
        if(p->inputTensors[0].geometry.dims!=1||p->inputTensors[0].geometry.dataType!=DATA_F32||
           !match(p->inputTensors[1].geometry,1,1,DATA_I32)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(mode_==1) {
            if(!match(p->outputTensors[0].geometry,1,50,DATA_I32)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        } else if(!match(p->inputTensors[2].geometry,2,2048,DATA_I32)||
                  !match(p->inputTensors[3].geometry,1,50,DATA_I32)||
                  !match(p->outputTensors[0].geometry,2,param[2]?2048:512,DATA_I32)||
                  p->outputTensors[0].geometry.maxSizes[1]!=1) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->indexSpaceRank=1;out->indexSpaceGeometry[0]=mode_==1?1:24;
    for(unsigned i=0;i<p->inputTensorNr;++i) {
        out->inputTensorAccessPattern[i].allRequired=true;
        out->inputTensorAccessPattern[i].sparseAccess=true;
    }
    // Runtime partitions are disjoint but are not affine functions of capacity.
    for(unsigned i=0;i<p->outputTensorNr;++i)out->outputTensorAccessPattern[i].allRequired=true;
    out->kernel.paramsNr=mode_==0?2:3;
    std::memcpy(out->kernel.scalarParams,param,out->kernel.paramsNr*sizeof(int));
    unsigned char* starts[]={&_binary___deepseek_v41_index_scores_gaudi2_o_start,
        &_binary___deepseek_v41_index_threshold_gaudi2_o_start,&_binary___deepseek_v41_index_emit_gaudi2_o_start};
    unsigned char* ends[]={&_binary___deepseek_v41_index_scores_gaudi2_o_end,
        &_binary___deepseek_v41_index_threshold_gaudi2_o_end,&_binary___deepseek_v41_index_emit_gaudi2_o_end};
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=ends[mode_]-starts[mode_];
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,starts[mode_],out->kernel.elfSize);return GLUE_SUCCESS;
}
