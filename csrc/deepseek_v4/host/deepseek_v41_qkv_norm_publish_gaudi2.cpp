// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_qkv_norm_publish_gaudi2.hpp"
#include <cmath>
#include <cstring>
extern unsigned char _binary___deepseek_v41_qkv_norm_publish_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_qkv_norm_publish_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41QkvNormPublishGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize!=sizeof(float)+sizeof(int))return GLUE_FAILED;
    if(in->inputTensorNr!=8)return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(in->outputTensorNr!=5)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const TensorDataType types[]={DATA_BF16,DATA_BF16,DATA_BF16,DATA_BF16,DATA_I32,DATA_F32,DATA_U8,DATA_BF16};
    for(unsigned i=0;i<8;++i)if(in->inputTensors[i].geometry.dataType!=types[i])return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto& q=in->inputTensors[0].geometry;const auto& qn=in->inputTensors[1].geometry;
    const auto& kv=in->inputTensors[2].geometry;const auto& kn=in->inputTensors[3].geometry;
    const auto& pos=in->inputTensors[4].geometry;const auto& phase=in->inputTensors[5].geometry;
    const auto& cache=in->inputTensors[6].geometry;const auto& dec=in->inputTensors[7].geometry;
    if(q.dims!=2 || q.maxSizes[0]!=1280 || q.maxSizes[1]!=1 || qn.dims!=1 || qn.maxSizes[0]!=1280 ||
       kv.dims!=2 || kv.maxSizes[0]!=512 || kv.maxSizes[1]!=1 || kn.dims!=1 || kn.maxSizes[0]!=512 ||
       pos.dims!=1 || pos.maxSizes[0]!=1 || phase.dims!=2 || phase.maxSizes[0]!=64 || !phase.maxSizes[1] ||
       phase.maxSizes[1]>1048576 || cache.dims!=2 || cache.maxSizes[0]!=528 || cache.maxSizes[1]<256 ||
       dec.dims!=2 || dec.maxSizes[0]!=512 || dec.maxSizes[1]<512)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto* raw=static_cast<const char*>(in->nodeParams.nodeParams);float epsilon;int offset;
    std::memcpy(&epsilon,raw,sizeof(epsilon));std::memcpy(&offset,raw+sizeof(epsilon),sizeof(offset));
    if(!std::isnormal(epsilon) || epsilon<=0 || (offset!=-1 && (offset<0 || offset%512 ||
       static_cast<uint64_t>(offset)+512>dec.maxSizes[1])))return GLUE_FAILED;
    const TensorDataType outputs[]={DATA_F8_143,DATA_F32,DATA_BF16,DATA_BF16,DATA_I32};
    const unsigned widths[]={1280,1,1280,512,16};
    for(unsigned i=0;i<5;++i) {
        const auto& y=in->outputTensors[i].geometry;
        if(y.dataType!=outputs[i])return GLUE_INCOMPATIBLE_DATA_TYPE;
        if(y.dims!=(i==4?1:2) || y.maxSizes[0]!=widths[i] || (i!=4 && y.maxSizes[1]!=1))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->outputTensorAccessPattern[i].allRequired=true;
    }
    for(unsigned i=0;i<8;++i)out->inputTensorAccessPattern[i].allRequired=true;
    out->indexSpaceRank=1;out->indexSpaceGeometry[0]=17;
    out->kernel.paramsNr=2;std::memcpy(out->kernel.scalarParams,raw,sizeof(float)+sizeof(int));
    auto* first=&_binary___deepseek_v41_qkv_norm_publish_gaudi2_o_start;
    auto* last=&_binary___deepseek_v41_qkv_norm_publish_gaudi2_o_end;
    const auto capacity=out->kernel.elfSize;out->kernel.elfSize=last-first;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,first,out->kernel.elfSize);return GLUE_SUCCESS;
}
