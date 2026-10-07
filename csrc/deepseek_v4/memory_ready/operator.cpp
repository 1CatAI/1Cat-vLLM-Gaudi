// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::private_memory_ready_post";
constexpr auto guid="private_memory_ready_post_tpc";
using Output=std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
habana::OutputMetaDataVector meta(const at::Stack& s) {
 TORCH_CHECK(s.size()==8,"Post statistics requires four tensors and epsilon");
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
 TORCH_CHECK(std::isnormal(float(s.at(5).toDouble())) && s.at(5).toDouble()>0,"Invalid epsilon");
 for(int i=4;i<5;++i){const auto t=s.at(i).toTensor();TORCH_CHECK(t.scalar_type()==at::kInt &&
   t.dim()==2 && t.size(1)==32 && s.at(6).toInt()>=0 && s.at(6).toInt()<t.size(0) && t.is_contiguous() && t.device()==x.device(),"Invalid private flag/status");}
 const auto enabled=s.at(7).toTensor();
  TORCH_CHECK(enabled.scalar_type()==at::kInt && enabled.sizes()==at::IntArrayRef({32}) && enabled.is_contiguous() && enabled.device()==s.at(0).toTensor().device(),"Invalid cold enable line");
 return {{at::kBFloat16,r.sizes().vec()},{at::kBFloat16,{tokens,5120}},
  {at::kBFloat16,{tokens,5120}},{at::kFloat8_e4m3fn,{tokens,5120}},{at::kFloat,{tokens,1}},
  {at::kFloat8_e4m3fn,{tokens,5120}},{at::kFloat,{tokens,1}},{at::kInt,{40}}};
}
class Stats final:public habana::OpBackend {
public:
 Stats(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("private_memory_ready_post"),t,{0,1,2,3,4,5,6,7},{},{},false){SetOutputMetaFn(meta);}
 void AddNode(synapse_helpers::graph& graph,const at::Stack& s)override {
  const auto m=meta(s);const auto tokens=s.at(0).toTensor().size(-2);
  int32_t line=s.at(6).toInt();
  auto partial=BuildNode(this,graph,{guid,{syn_in(0),syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(5)},
   {{m[0].shape,m[0].dtype,0},{m[1].shape,m[1].dtype,1},{{tokens,2,40},at::kFloat},{{40},at::kInt,7}},&line,sizeof(line)});
  float params[2]={float(s.at(5).toDouble()),1.f/5120};
  auto finish=BuildNode(this,graph,{"custom_deepseek_v41_norm_from_weighted_stats_gaudi2",{partial[1].get(),syn_in(3),partial[2].get()},
   {{m[2].shape,m[2].dtype,2},{m[3].shape,m[3].dtype,3},{m[4].shape,m[4].dtype,4},
    {m[5].shape,m[5].dtype,5},{m[6].shape,m[6].dtype,6}},params,sizeof(params)});
  syn_out(0)=std::move(partial[0]);syn_out(1)=std::move(partial[1]);syn_out(7)=std::move(partial[3]);
  for(unsigned i=0;i<5;++i)syn_out(i+2)=std::move(finish[i]);
 }
};
const bool registered=[] {
 habana::custom_op::registerUserCustomOp(schema,guid,[](const at::Stack& s){habana::PartialOutputMetaDataVector out;
  for(const auto& x:meta(s)){out.push_back({x.dtype,x.shape});}
  return out;},nullptr);
 habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Stats>(d,t);});return true;
}();
template<bool Meta> Output run(const at::Tensor& x,const at::Tensor& r,const at::Tensor& g,const at::Tensor& w,const at::Tensor& ready,double eps,int64_t line,const at::Tensor& enabled) {
 const at::Stack s{x,r,g,w,ready,eps,line,enabled};const auto m=meta(s);
 std::vector<at::Tensor> out;
 if constexpr(Meta){for(const auto& v:m)out.push_back(at::empty(v.shape,x.options().dtype(v.dtype)));}
 else {TORCH_CHECK(registered && x.device().type()==at::kHPU);auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);out=descriptor.execute(s);}
 return {out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m){m.def("private_memory_ready_post(Tensor value, Tensor residual, Tensor gates, Tensor norm_weight, Tensor ready, float epsilon, int flag_line, Tensor enabled) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("private_memory_ready_post",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("private_memory_ready_post",run<true>);}

namespace zero {
constexpr auto schema="custom_op::private_memory_flags_zero";
habana::OutputMetaDataVector meta(const at::Stack& s){const auto t=s.at(0).toTensor();
 TORCH_CHECK(t.scalar_type()==at::kInt && t.dim()==2 && t.size(1)==32 && t.size(0)>=2 && t.size(0)<=256 && t.is_contiguous(),"Flag root geometry");return {{at::kInt,t.sizes().vec()}};}
class Zero final:public habana::OpBackend{public:Zero(int d,c10::ScalarType t):OpBackend(d,"private_memory_flags_zero_tpc",t,{0},{},{},false){SetOutputMetaFn(meta);}
 void AddNode(synapse_helpers::graph& g,const at::Stack& s)override{auto m=meta(s)[0];syn_out(0)=std::move(BuildNode(this,g,{"private_memory_flags_zero_tpc",{syn_in(0)},{{m.shape,m.dtype,0}}})[0]);}};
const bool registered=[](){habana::custom_op::registerUserCustomOp(schema,"private_memory_flags_zero_tpc",[](const at::Stack&s){auto m=meta(s)[0];return habana::PartialOutputMetaDataVector{{m.dtype,m.shape}};},nullptr);habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Zero>(d,t);});return true;}();
template<bool Fake>at::Tensor run(const at::Tensor& t){at::Stack s{t};auto m=meta(s)[0];if constexpr(Fake)return at::empty(m.shape,t.options());TORCH_CHECK(registered);auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);return descriptor.execute(s)[0];}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m){m.def("private_memory_flags_zero(Tensor template) -> Tensor");}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("private_memory_flags_zero",zero::run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("private_memory_flags_zero",zero::run<true>);}
