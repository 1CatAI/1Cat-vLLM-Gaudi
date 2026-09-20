// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_index_gaudi2.hpp"
#include <cstring>
#define ELF(name) extern unsigned char _binary___##name##_o_start, _binary___##name##_o_end;
ELF(deepseek_v41_index_scores_gaudi2)
ELF(deepseek_v41_index_threshold_gaudi2)
ELF(deepseek_v41_index_emit_gaudi2)
ELF(deepseek_v41_index_scores_decoded_gaudi2)
using namespace tpc_lib_api;
GlueCodeReturn DeepseekV41IndexGaudi2::GetGcDefinitions(HabanaKernelParams* p,HabanaKernelInstantiation* out) {
    if (!p || !out || mode_>3) return GLUE_FAILED;
    const bool score_mode=mode_==0||mode_==3;
    if(p->inputTensorNr!=(score_mode?6u:mode_==1?2u:4u)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(p->outputTensorNr!=(score_mode?2u:1u)) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if(!p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize!=3*sizeof(int)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto* param=static_cast<const int*>(p->nodeParams.nodeParams);
    if((param[0]!=1&&param[0]!=2)||param[1]<0||param[1]>1||param[2]<0||param[2]>1)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto match=[](const TensorGeometry& t, unsigned dim, unsigned width, TensorDataType type) {
        return t.dims==dim && t.maxSizes[0]==width && t.dataType==type;
    };
    const auto& position=p->inputTensors[score_mode?4:1].geometry;
    const uint64_t batch=position.maxSizes[0];
    if(position.dims!=1||position.dataType!=DATA_I32||!batch||batch>64)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto rows=[&](const TensorGeometry& t, unsigned width, TensorDataType type, bool vector=false) {
        return t.dataType==type&&t.maxSizes[0]==width&&
            (vector ? t.dims==1&&batch==1 : t.dims==2&&t.maxSizes[1]==batch);
    };
    if(score_mode) {
        const auto& q=p->inputTensors[0].geometry;
        const auto& pages=p->inputTensors[3].geometry;
        const bool vector=pages.dims==1;
        if(!match(q,3,128,DATA_BF16)||q.maxSizes[1]!=32||q.maxSizes[2]!=batch ||
           !rows(p->inputTensors[1].geometry,32,DATA_BF16) ||
           !(mode_==0 ? match(p->inputTensors[2].geometry,2,68,DATA_U8)
                      : match(p->inputTensors[2].geometry,2,128,DATA_BF16)) ||
           !rows(pages,pages.maxSizes[0],DATA_I32,vector) ||
           !rows(p->inputTensors[5].geometry,2048,DATA_I32))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto& score=p->outputTensors[0].geometry;
        if(!rows(score,score.maxSizes[0],DATA_F32,vector)||score.maxSizes[0]%64 ||
           !rows(p->outputTensors[1].geometry,(score.maxSizes[0]/8+63)/64*64,DATA_F32,vector))
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    } else {
        const auto& score=p->inputTensors[0].geometry;
        const bool vector=score.dims==1;
        if(!rows(score,score.maxSizes[0],DATA_F32,vector)||score.maxSizes[0]%64)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(mode_==1) {
            if(!rows(p->outputTensors[0].geometry,50,DATA_I32,vector))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        } else {
            if(!rows(p->inputTensors[2].geometry,2048,DATA_I32) ||
               !rows(p->outputTensors[0].geometry,param[2]?2048:512,DATA_I32))
                return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            if(mode_==2 && !rows(p->inputTensors[3].geometry,50,DATA_I32,vector))
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
        }
    }
    // Keep the proven rank-1 launch contract for a single request.  Older
    // Synapse runtimes reject a rank-2 custom TPC node whose second extent is
    // one, while a real batch still needs the request dimension in the index
    // space.  This preserves the batch-capable path without penalising B1.
    out->indexSpaceRank=batch==1?1:2;
    out->indexSpaceGeometry[0]=mode_==1?1:24;
    if(batch>1) out->indexSpaceGeometry[1]=batch;
    for(unsigned i=0;i<p->inputTensorNr;++i) {
        out->inputTensorAccessPattern[i].allRequired=true;
        out->inputTensorAccessPattern[i].sparseAccess=true;
    }
    // Runtime partitions are disjoint but are not affine functions of capacity.
    for(unsigned i=0;i<p->outputTensorNr;++i)out->outputTensorAccessPattern[i].allRequired=true;
    out->kernel.paramsNr=score_mode?2:3;
    std::memcpy(out->kernel.scalarParams,param,out->kernel.paramsNr*sizeof(int));
    unsigned char* starts[]={&_binary___deepseek_v41_index_scores_gaudi2_o_start,
        &_binary___deepseek_v41_index_threshold_gaudi2_o_start,&_binary___deepseek_v41_index_emit_gaudi2_o_start,
        &_binary___deepseek_v41_index_scores_decoded_gaudi2_o_start};
    unsigned char* ends[]={&_binary___deepseek_v41_index_scores_gaudi2_o_end,
        &_binary___deepseek_v41_index_threshold_gaudi2_o_end,&_binary___deepseek_v41_index_emit_gaudi2_o_end,
        &_binary___deepseek_v41_index_scores_decoded_gaudi2_o_end};
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=ends[mode_]-starts[mode_];
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,starts[mode_],out->kernel.elfSize);return GLUE_SUCCESS;
}
