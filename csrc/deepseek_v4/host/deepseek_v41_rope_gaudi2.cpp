// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_rope_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_rope_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_rope_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_rope_inverse_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_rope_inverse_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_prefill_rope_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_rope_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_prefill_rope_inverse_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_rope_inverse_bf16_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41RopeGaudi2::GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, prefill_ ? (inverse_ ? "custom_deepseek_v41_prefill_rope_inverse_bf16_gaudi2"
                                         : "custom_deepseek_v41_prefill_rope_bf16_gaudi2")
                               : (inverse_ ? "custom_deepseek_v41_rope_inverse_bf16_gaudi2"
                                           : "custom_deepseek_v41_rope_bf16_gaudi2"));
    return tpc_lib_api::GLUE_SUCCESS;
}
tpc_lib_api::GlueCodeReturn DeepseekV41RopeGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(in->inputTensorNr != 3){in->inputTensorNr=3;return GLUE_INCOMPATIBLE_INPUT_COUNT;}
    if(in->outputTensorNr != 1){in->outputTensorNr=1;return GLUE_INCOMPATIBLE_OUTPUT_COUNT;}
    const auto& x=in->inputTensors[0].geometry;
    const auto& p=in->inputTensors[1].geometry;
    const auto& t=in->inputTensors[2].geometry;
    const auto& y=in->outputTensors[0].geometry;
    if(x.dataType!=DATA_BF16 || p.dataType!=DATA_I32 || t.dataType!=DATA_F32 || y.dataType!=DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if(x.dims!=3 || x.maxSizes[0]%128 || !x.maxSizes[0] || x.maxSizes[0]>512 ||
       !x.maxSizes[1] || x.maxSizes[1]>128 || !x.maxSizes[2] || x.maxSizes[2]>(prefill_ ? 8192 : 64) ||
       p.dims!=1 || p.maxSizes[0]!=x.maxSizes[2] || t.dims!=2 || t.maxSizes[0]!=64 || !t.maxSizes[1])
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if(y.dims!=3 || y.maxSizes[0]!=x.maxSizes[0] || y.maxSizes[1]!=x.maxSizes[1] || y.maxSizes[2]!=x.maxSizes[2])
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank=3;
    out->indexSpaceGeometry[0]=1;
    out->indexSpaceGeometry[1]=x.maxSizes[1];
    out->indexSpaceGeometry[2]=x.maxSizes[2];
    for(unsigned d=0;d<3;++d) {
        out->inputTensorAccessPattern[0].mapping[d]={d,1,0,d==0?float(x.maxSizes[0]-1):0};
        out->outputTensorAccessPattern[0].mapping[d]=out->inputTensorAccessPattern[0].mapping[d];
    }
    out->inputTensorAccessPattern[1].mapping[0]={2,1,0,0};
    out->inputTensorAccessPattern[2].allRequired=true;
    out->inputTensorAccessPattern[2].sparseAccess=true;
    out->kernel.paramsNr=0;
    const auto* start = prefill_ ? (inverse_ ? &_binary___deepseek_v41_prefill_rope_inverse_bf16_gaudi2_o_start
                                           : &_binary___deepseek_v41_prefill_rope_bf16_gaudi2_o_start)
                                : (inverse_ ? &_binary___deepseek_v41_rope_inverse_bf16_gaudi2_o_start
                                            : &_binary___deepseek_v41_rope_bf16_gaudi2_o_start);
    const auto* end = prefill_ ? (inverse_ ? &_binary___deepseek_v41_prefill_rope_inverse_bf16_gaudi2_o_end
                                         : &_binary___deepseek_v41_prefill_rope_bf16_gaudi2_o_end)
                              : (inverse_ ? &_binary___deepseek_v41_rope_inverse_bf16_gaudi2_o_end
                                          : &_binary___deepseek_v41_rope_bf16_gaudi2_o_end);
    const auto capacity=out->kernel.elfSize;
    out->kernel.elfSize=end-start;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,start,out->kernel.elfSize);
    return GLUE_SUCCESS;
}
