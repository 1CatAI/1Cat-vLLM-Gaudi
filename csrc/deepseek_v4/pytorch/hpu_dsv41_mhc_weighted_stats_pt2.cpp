// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include "hpu_ops/op_backend.h"
#ifndef DSV41_MHC_STATS_SCHEMA
#define DSV41_MHC_STATS_SCHEMA "custom_deepseek_v41_mhc_post_norm_statistics_gaudi2"
#endif
namespace {
constexpr auto schema="custom_op::" DSV41_MHC_STATS_SCHEMA;
constexpr auto guid="custom_deepseek_v41_mhc_post_weighted_stats_gaudi2";
using Output=std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
habana::OutputMetaDataVector meta(const at::Stack& s) {
 TORCH_CHECK(s.size()==5,"Post statistics requires four tensors and epsilon");
 const auto x=s.at(0).toTensor(),r=s.at(1).toTensor();
 TORCH_CHECK(x.dim()==2 || x.dim()==3,"Expected row or rank-major peer tensor");
 const auto tokens=x.size(-2);
 TORCH_CHECK(tokens>=1 && tokens<=6 && x.size(-1)==5120 &&
   (x.dim()==2 || (x.size(0)>=2 && x.size(0)<=8)) && r.sizes()==at::IntArrayRef({tokens,4,5120}) &&
   s.at(2).toTensor().sizes()==at::IntArrayRef({tokens,24}) &&
   s.at(3).toTensor().sizes()==at::IntArrayRef({5120}),"Invalid post statistics geometry");
 for(int i=0;i<4;++i){const auto t=s.at(i).toTensor();
  TORCH_CHECK(t.scalar_type()==(i==2?at::kFloat:at::kBFloat16) && t.is_contiguous() &&
   t.device()==x.device() && !t.requires_grad(),"Expected matching contiguous inference operands");}
 TORCH_CHECK(std::isnormal(float(s.at(4).toDouble())) && s.at(4).toDouble()>0,"Invalid epsilon");
 return {{at::kBFloat16,r.sizes().vec()},{at::kBFloat16,{tokens,5120}},
  {at::kBFloat16,{tokens,5120}},{at::kFloat8_e4m3fn,{tokens,5120}},{at::kFloat,{tokens,1}},
  {at::kFloat8_e4m3fn,{tokens,5120}},{at::kFloat,{tokens,1}}};
}
class Stats final:public habana::OpBackend {
public:
 Stats(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("dsv41_mhc_weighted_stats"),t,{0,1,2,3,4,5,6},{},{},false){SetOutputMetaFn(meta);}
 void AddNode(synapse_helpers::graph& graph,const at::Stack& s)override {
  const auto m=meta(s);const auto tokens=s.at(0).toTensor().size(-2);
  auto partial=BuildNode(this,graph,{guid,{syn_in(0),syn_in(1),syn_in(2),syn_in(3)},
   {{m[0].shape,m[0].dtype,0},{m[1].shape,m[1].dtype,1},{{tokens,2,40},at::kFloat}}});
  float params[2]={float(s.at(4).toDouble()),1.f/5120};
#if DSV41_WEIGHTED_STATS_PREPARED
  auto parameters=BuildNode(this,graph,{"custom_deepseek_v41_weighted_row_parameters_gaudi2",
      {partial[2].get()},{{{tokens,5},at::kFloat}},params,sizeof(params)});
  auto finish=BuildNode(this,graph,{"custom_deepseek_v41_norm_prepared_weighted_stats_gaudi2",
      {partial[1].get(),syn_in(3),parameters[0].get()},
#else
  auto finish=BuildNode(this,graph,{"custom_deepseek_v41_norm_from_weighted_stats_gaudi2",{partial[1].get(),syn_in(3),partial[2].get()},
#endif
   {{m[2].shape,m[2].dtype,2},{m[3].shape,m[3].dtype,3},{m[4].shape,m[4].dtype,4},
    {m[5].shape,m[5].dtype,5},{m[6].shape,m[6].dtype,6}},params,sizeof(params)});
  syn_out(0)=std::move(partial[0]);syn_out(1)=std::move(partial[1]);
  for(unsigned i=0;i<5;++i)syn_out(i+2)=std::move(finish[i]);
 }
};
const bool registered=[] {
 habana::custom_op::registerUserCustomOp(schema,guid,[](const at::Stack& s){habana::PartialOutputMetaDataVector out;
  for(const auto& x:meta(s)){out.push_back({x.dtype,x.shape});}
  return out;},nullptr);
 habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Stats>(d,t);});return true;
}();
template<bool Meta> Output run(const at::Tensor& x,const at::Tensor& r,const at::Tensor& g,const at::Tensor& w,double eps) {
 const at::Stack s{x,r,g,w,eps};const auto m=meta(s);
 std::vector<at::Tensor> out;
 if constexpr(Meta){for(const auto& v:m)out.push_back(at::empty(v.shape,x.options().dtype(v.dtype)));}
 else {TORCH_CHECK(registered && x.device().type()==at::kHPU);auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);out=descriptor.execute(s);}
 return {out[0],out[1],out[2],out[3],out[4],out[5],out[6]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m){m.def(DSV41_MHC_STATS_SCHEMA "(Tensor value, Tensor residual, Tensor gates, Tensor norm_weight, float epsilon) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl(DSV41_MHC_STATS_SCHEMA,run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl(DSV41_MHC_STATS_SCHEMA,run<true>);}
