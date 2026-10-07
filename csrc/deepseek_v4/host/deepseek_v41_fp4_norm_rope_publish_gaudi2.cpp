// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_fp4_norm_rope_publish_gaudi2.hpp"
#include <cstring>
#include <initializer_list>
extern unsigned char _binary___deepseek_v41_fp4_norm_rope_publish_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_fp4_norm_rope_publish_gaudi2_o_end;
namespace {
using namespace tpc_lib_api;
bool match(const Tensor& t,TensorDataType dtype,std::initializer_list<unsigned> sizes) {
    if(t.geometry.dataType!=dtype || t.geometry.dims!=sizes.size()) return false;
    unsigned i=0;for(auto s:sizes) {if(s && t.geometry.maxSizes[i]!=s) return false;++i;}return true;
}
}
tpc_lib_api::GlueCodeReturn DeepseekV41Fp4NormRopePublishGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(in->inputTensorNr!=11){in->inputTensorNr=11;return GLUE_INCOMPATIBLE_INPUT_COUNT;}
    if(in->outputTensorNr!=1){in->outputTensorNr=1;return GLUE_INCOMPATIBLE_OUTPUT_COUNT;}
    struct Params {float epsilon;int ratio;int main_enabled;int hot_enabled;int index_enabled;};
    if(in->nodeParams.nodeParamsSize!=sizeof(Params) || !in->nodeParams.nodeParams) return GLUE_FAILED;
    const auto* p=static_cast<const Params*>(in->nodeParams.nodeParams);
    if(p->ratio!=1 && p->ratio!=2) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if(!match(in->inputTensors[0],DATA_BF16,{512,1}) ||
       !match(in->inputTensors[1],DATA_BF16,{128,1}) ||
       !match(in->inputTensors[2],DATA_BF16,{128}) ||
       !match(in->inputTensors[3],DATA_I32,{1}) ||
       !match(in->inputTensors[4],DATA_F32,{64,0}) ||
       !match(in->inputTensors[5],DATA_I32,{0}) ||
       !match(in->inputTensors[6],DATA_U8,{288,0}) ||
       !match(in->inputTensors[7],DATA_U8,{68,0}) ||
       !match(in->inputTensors[8],DATA_BF16,{512,0}) ||
       !match(in->inputTensors[9],DATA_BF16,{128,0}) ||
       !match(in->inputTensors[10],DATA_BF16,{128,0})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if(!match(in->outputTensors[0],DATA_I32,{36})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank=1;out->indexSpaceGeometry[0]=36;
    for(unsigned i=0;i<11;++i)out->inputTensorAccessPattern[i].allRequired=true;
    out->outputTensorAccessPattern[0].mapping[0]={0,1,0,0};
    out->kernel.paramsNr=5;std::memcpy(out->kernel.scalarParams,p,sizeof(Params));
    auto* begin=&_binary___deepseek_v41_fp4_norm_rope_publish_gaudi2_o_start;
    auto* end=&_binary___deepseek_v41_fp4_norm_rope_publish_gaudi2_o_end;
    const unsigned capacity=out->kernel.elfSize;out->kernel.elfSize=end-begin;
    if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,begin,out->kernel.elfSize);return GLUE_SUCCESS;
}
