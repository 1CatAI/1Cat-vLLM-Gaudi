// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_engram_update_gaudi2.hpp"
#include <cmath>
#include <cstring>
extern unsigned char _binary___deepseek_v41_engram_update_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_engram_update_bf16_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41EngramUpdateGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out || !p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize != sizeof(float)) return GLUE_FAILED;
    if (p->inputTensorNr != 5) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& x=p->inputTensors[0].geometry; const auto& kv=p->inputTensors[1].geometry;
    const auto& q=p->inputTensors[2].geometry; const auto& k=p->inputTensors[3].geometry;
    const auto& mask=p->inputTensors[4].geometry; const auto& y=p->outputTensors[0].geometry;
    if (x.dataType!=DATA_BF16 || kv.dataType!=DATA_BF16 || q.dataType!=DATA_BF16 ||
        k.dataType!=DATA_BF16 || mask.dataType!=DATA_I8 || y.dataType!=DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto rows=x.maxSizes[2];
    if (x.dims!=3 || x.maxSizes[0]!=5120 || x.maxSizes[1]!=4 || rows<1 || rows>6 ||
        kv.dims!=2 || kv.maxSizes[0]!=25600 || kv.maxSizes[1]!=rows ||
        q.dims!=2 || q.maxSizes[0]!=5120 || q.maxSizes[1]!=4 ||
        k.dims!=2 || k.maxSizes[0]!=5120 || k.maxSizes[1]!=4 ||
        mask.dims!=1 || mask.maxSizes[0]!=rows || y.dims!=3 ||
        y.maxSizes[0]!=5120 || y.maxSizes[1]!=4 || y.maxSizes[2]!=rows) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto eps=*static_cast<const float*>(p->nodeParams.nodeParams);
    if (!(eps>0) || !std::isnormal(eps)) return GLUE_FAILED;
    out->indexSpaceRank=2; out->indexSpaceGeometry[0]=4; out->indexSpaceGeometry[1]=rows;
    auto map=[](TensorAccessPattern& pat,unsigned cols) {
        std::memset(&pat,0,sizeof(pat));pat.mapping[0].indexSpaceDim=0;pat.mapping[0].a=0;pat.mapping[0].end_b=cols-1;
    };
    map(out->inputTensorAccessPattern[0],5120);
    out->inputTensorAccessPattern[0].mapping[1].indexSpaceDim=0;out->inputTensorAccessPattern[0].mapping[1].a=1;
    out->inputTensorAccessPattern[0].mapping[2].indexSpaceDim=1;out->inputTensorAccessPattern[0].mapping[2].a=1;
    // A stream reads its key and the common value tail. Conservatively map
    // the complete KV row so the compiler cannot slice away the value tail.
    map(out->inputTensorAccessPattern[1],25600);
    out->inputTensorAccessPattern[1].mapping[1].indexSpaceDim=1;out->inputTensorAccessPattern[1].mapping[1].a=1;
    for (unsigned i=2;i<4;++i) {
        map(out->inputTensorAccessPattern[i],5120);
        out->inputTensorAccessPattern[i].mapping[1].indexSpaceDim=0;out->inputTensorAccessPattern[i].mapping[1].a=1;
    }
    map(out->inputTensorAccessPattern[4],1);
    out->inputTensorAccessPattern[4].mapping[0].indexSpaceDim=1;out->inputTensorAccessPattern[4].mapping[0].a=1;
    out->outputTensorAccessPattern[0]=out->inputTensorAccessPattern[0];
    out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,&eps,sizeof(eps));
    auto* begin=&_binary___deepseek_v41_engram_update_bf16_gaudi2_o_start;
    auto* end=&_binary___deepseek_v41_engram_update_bf16_gaudi2_o_end;
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=end-begin;
    if (capacity<out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    if (!out->kernel.kernelElf) return GLUE_FAILED;
    std::memcpy(out->kernel.kernelElf,begin,out->kernel.elfSize);return GLUE_SUCCESS;
}
