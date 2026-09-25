// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_index_keys_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_index_keys_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_index_keys_gaudi2_o_end;
using namespace tpc_lib_api;
GlueCodeReturn DeepseekV41IndexKeysGaudi2::GetGcDefinitions(HabanaKernelParams* p,HabanaKernelInstantiation* out) {
    if(!p||!out)return GLUE_FAILED;
    if(p->inputTensorNr!=3)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(p->outputTensorNr!=1)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if(!p->nodeParams.nodeParams||p->nodeParams.nodeParamsSize!=sizeof(int))return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto* params=static_cast<const int*>(p->nodeParams.nodeParams);
    const auto& cache=p->inputTensors[0].geometry;
    const auto& pages=p->inputTensors[1].geometry;
    const auto& rows=p->inputTensors[2].geometry;
    if((params[0]!=1&&params[0]!=2)||cache.dims!=2||cache.maxSizes[0]!=68||cache.dataType!=DATA_U8||
       (pages.dims!=1&&pages.dims!=2)||pages.dataType!=DATA_I32||rows.dims!=2||rows.dataType!=DATA_I32||
       (pages.dims==2&&pages.maxSizes[1]!=rows.maxSizes[1])||
       rows.maxSizes[0]<1||rows.maxSizes[0]>2048||rows.maxSizes[1]<1||rows.maxSizes[1]>128)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto& output=p->outputTensors[0].geometry;
    if(output.dims!=3||output.dataType!=DATA_BF16||output.maxSizes[0]!=128||
       output.maxSizes[1]!=rows.maxSizes[0]||output.maxSizes[2]!=rows.maxSizes[1])return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    // Synapse 1.24 rejects a rank-2 custom TPC node when the request extent is
    // one.  Keep the request axis for real batches, but use the equivalent
    // rank-1 launch for B1.  The kernel already treats a missing second index
    // space coordinate as zero.
    out->indexSpaceRank=rows.maxSizes[1]==1?1:2;
    out->indexSpaceGeometry[0]=rows.maxSizes[0];
    if(rows.maxSizes[1]>1)out->indexSpaceGeometry[1]=rows.maxSizes[1];
    if(tiled_) {
        // Indirect source addresses remain conservative. Logical row reads
        // and decoded outputs are affine and can be sliced with the MME.
        // allRequired sources intentionally do not also claim sparseAccess:
        // that combination prevents mixed affine output inference in Synapse.
        out->inputTensorAccessPattern[0].allRequired=true;
        out->inputTensorAccessPattern[1].allRequired=true;
        auto map=[](TensorAccessPattern& access,unsigned dim,unsigned axis,
                    int coefficient,int first,int last) {
            access.mapping[dim].indexSpaceDim=axis;
            access.mapping[dim].a=coefficient;
            access.mapping[dim].start_b=first;
            access.mapping[dim].end_b=last;
        };
        const bool batched=rows.maxSizes[1]>1;
        map(out->inputTensorAccessPattern[2],0,0,1,0,0);
        map(out->inputTensorAccessPattern[2],1,batched?1:0,batched?1:0,0,0);
        map(out->outputTensorAccessPattern[0],0,0,0,0,127);
        map(out->outputTensorAccessPattern[0],1,0,1,0,0);
        map(out->outputTensorAccessPattern[0],2,batched?1:0,batched?1:0,0,0);
    } else {
        // Preserve the original conservative mapping and binary entry.
        for(unsigned i=0;i<p->inputTensorNr;++i){
            out->inputTensorAccessPattern[i].allRequired=true;
            out->inputTensorAccessPattern[i].sparseAccess=true;
        }
        out->outputTensorAccessPattern[0].allRequired=true;
    }
    out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,params,sizeof(int));
    const auto size=&_binary___deepseek_v41_index_keys_gaudi2_o_end-&_binary___deepseek_v41_index_keys_gaudi2_o_start;
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=size;
    if(capacity<size)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,&_binary___deepseek_v41_index_keys_gaudi2_o_start,size);return GLUE_SUCCESS;
}
