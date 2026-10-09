// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_mhc_weighted_stats_gaudi2.hpp"
#include <cstring>
#include <cmath>
#include <initializer_list>
extern unsigned char _binary___deepseek_v41_mhc_post_weighted_stats_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mhc_post_weighted_stats_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_norm_from_weighted_stats_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_norm_from_weighted_stats_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41MhcWeightedStatsGaudi2::GetGcDefinitions(
 tpc_lib_api::HabanaKernelParams* p,tpc_lib_api::HabanaKernelInstantiation* g) {
 using namespace tpc_lib_api;
 if(p->inputTensorNr!=(finish_?3U:4U))return GLUE_INCOMPATIBLE_INPUT_COUNT;
 if(p->outputTensorNr!=(finish_?5U:3U))return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
 const auto& x=p->inputTensors[0].geometry;const auto tokens=x.maxSizes[1];
 auto check=[](const Tensor& t,TensorDataType type,std::initializer_list<uint64_t> sizes){
  if(t.geometry.dataType!=type || t.geometry.dims!=sizes.size())return false;
  unsigned d=0;for(auto n:sizes)if(t.geometry.maxSizes[d++]!=n)return false;return true;
 };
 if(!tokens || tokens>6 || x.dataType!=DATA_BF16 || x.maxSizes[0]!=5120)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(finish_) {
  if(x.dims!=2 || !check(p->inputTensors[1],DATA_BF16,{5120}) ||
     !check(p->inputTensors[2],DATA_F32,{40,2,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for(unsigned i=0;i<5;++i) {
   const bool scale=i==2 || i==4;const auto type=scale?DATA_F32:i==0?DATA_BF16:DATA_F8_143;
   if(!check(p->outputTensors[i],type,{scale?1UL:5120UL,tokens}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
   if(scale)g->outputTensorAccessPattern[i].mapping[0]={0,0,0,0,false};
   else g->outputTensorAccessPattern[i].mapping[0]={0,128,0,127,false};
   g->outputTensorAccessPattern[i].mapping[1]={1,1,0,0,false};
  }
  g->inputTensorAccessPattern[0].mapping[0]={0,128,0,127,false};
  g->inputTensorAccessPattern[0].mapping[1]={1,1,0,0,false};
  g->inputTensorAccessPattern[1].mapping[0]={0,128,0,127,false};
  g->inputTensorAccessPattern[2].mapping[0]={0,0,0,39,false};
  g->inputTensorAccessPattern[2].mapping[1]={0,0,0,1,false};
  g->inputTensorAccessPattern[2].mapping[2]={1,1,0,0,false};
  if(!p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize!=8)return GLUE_FAILED;
  const auto* params=static_cast<const float*>(p->nodeParams.nodeParams);
  if(!std::isnormal(params[0]) || params[0]<=0 || params[1]!=1.f/5120)return GLUE_FAILED;
  g->kernel.paramsNr=2;std::memcpy(g->kernel.scalarParams,params,8);
 } else {
  const auto ranks=x.dims==3?x.maxSizes[2]:1;
  if((x.dims!=2 && x.dims!=3) || (x.dims==3 && (ranks<2 || ranks>8)) ||
     !check(p->inputTensors[1],DATA_BF16,{5120,4,tokens}) ||
     !check(p->inputTensors[2],DATA_F32,{24,tokens}) || !check(p->inputTensors[3],DATA_BF16,{5120}) ||
     !check(p->outputTensors[0],DATA_BF16,{5120,4,tokens}) ||
     !check(p->outputTensors[1],DATA_BF16,{5120,tokens}) ||
     !check(p->outputTensors[2],DATA_F32,{40,2,tokens}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
  g->inputTensorAccessPattern[0].mapping[0]={0,128,0,127,false};
  g->inputTensorAccessPattern[0].mapping[1]={1,1,0,0,false};
  if(x.dims==3)g->inputTensorAccessPattern[0].mapping[2]={0,0,0,float(ranks-1),false};
  g->inputTensorAccessPattern[1].mapping[0]={0,128,0,127,false};
  g->inputTensorAccessPattern[1].mapping[1]={0,0,0,3,false};
  g->inputTensorAccessPattern[1].mapping[2]={1,1,0,0,false};
  g->inputTensorAccessPattern[2].mapping[0]={0,0,0,23,false};
  g->inputTensorAccessPattern[2].mapping[1]={1,1,0,0,false};
  g->inputTensorAccessPattern[3].mapping[0]={0,128,0,127,false};
  g->outputTensorAccessPattern[0].mapping[0]={0,128,0,127,false};
  g->outputTensorAccessPattern[0].mapping[1]={0,0,0,3,false};
  g->outputTensorAccessPattern[0].mapping[2]={1,1,0,0,false};
  g->outputTensorAccessPattern[1].mapping[0]={0,128,0,127,false};
  g->outputTensorAccessPattern[1].mapping[1]={1,1,0,0,false};
  g->outputTensorAccessPattern[2].mapping[0]={0,1,0,0,false};
  g->outputTensorAccessPattern[2].mapping[1]={0,0,0,1,false};
  g->outputTensorAccessPattern[2].mapping[2]={1,1,0,0,false};g->kernel.paramsNr=0;
 }
 g->indexSpaceRank=2;g->indexSpaceGeometry[0]=40;g->indexSpaceGeometry[1]=tokens;
 auto* first=finish_?&_binary___deepseek_v41_norm_from_weighted_stats_gaudi2_o_start:&_binary___deepseek_v41_mhc_post_weighted_stats_gaudi2_o_start;
 auto* last=finish_?&_binary___deepseek_v41_norm_from_weighted_stats_gaudi2_o_end:&_binary___deepseek_v41_mhc_post_weighted_stats_gaudi2_o_end;
 const auto capacity=g->kernel.elfSize;g->kernel.elfSize=last-first;if(capacity<g->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(g->kernel.kernelElf,first,g->kernel.elfSize);return GLUE_SUCCESS;
}
