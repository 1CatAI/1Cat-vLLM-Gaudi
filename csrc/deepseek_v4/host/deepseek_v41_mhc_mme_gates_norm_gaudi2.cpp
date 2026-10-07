// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_mhc_mme_gates_norm_gaudi2.hpp"
#include <cstring>
#include <initializer_list>
extern unsigned char _binary___deepseek_v41_mhc_mme_gates_norm_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mhc_mme_gates_norm_gaudi2_o_end;
namespace {
using namespace tpc_lib_api;
bool match(const Tensor& t, TensorDataType dtype, std::initializer_list<unsigned> sizes) {
    if (t.geometry.dataType != dtype || t.geometry.dims != sizes.size()) return false;
    unsigned i=0;
    for (auto s:sizes) if(t.geometry.maxSizes[i++]!=s) return false;
    return true;
}
}
tpc_lib_api::GlueCodeReturn DeepseekV41MhcMmeGatesNormGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if(in->inputTensorNr!=6) {in->inputTensorNr=6;return GLUE_INCOMPATIBLE_INPUT_COUNT;}
    if(in->outputTensorNr!=4) {in->outputTensorNr=4;return GLUE_INCOMPATIBLE_OUTPUT_COUNT;}
    const unsigned tokens=in->inputTensors[0].geometry.maxSizes[1];
    if(tokens<1 || tokens>6 || !match(in->inputTensors[0],DATA_F32,{48,tokens}) ||
       !match(in->inputTensors[1],DATA_BF16,{20480,tokens}) ||
       !match(in->inputTensors[2],DATA_BF16,{5120,tokens}) ||
       !match(in->inputTensors[3],DATA_BF16,{5120}) ||
       !match(in->inputTensors[4],DATA_F32,{3}) ||
       !match(in->inputTensors[5],DATA_F32,{24})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if(!match(in->outputTensors[0],DATA_F32,{24,tokens}) ||
       !match(in->outputTensors[1],DATA_BF16,{5120,tokens}) ||
       !match(in->outputTensors[2],DATA_F8_143,{5120,tokens}) ||
       !match(in->outputTensors[3],DATA_F32,{1,tokens})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    // Two disjoint output families; neither work point reads the other's output.
    for(unsigned i=0;i<6;++i) out->inputTensorAccessPattern[i].allRequired=true;
    for(unsigned i=0;i<4;++i) out->outputTensorAccessPattern[i].allRequired=true;
    out->indexSpaceRank=2;out->indexSpaceGeometry[0]=2;out->indexSpaceGeometry[1]=tokens;
    struct Params {float epsilon;float inverse_width;};
    std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,sizeof(Params));
    out->kernel.paramsNr=2;
    auto* begin=&_binary___deepseek_v41_mhc_mme_gates_norm_gaudi2_o_start;
    auto* end=&_binary___deepseek_v41_mhc_mme_gates_norm_gaudi2_o_end;
    const unsigned capacity=out->kernel.elfSize;
    out->kernel.elfSize=end-begin;
    if(capacity<out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,begin,out->kernel.elfSize);
    return GLUE_SUCCESS;
}
