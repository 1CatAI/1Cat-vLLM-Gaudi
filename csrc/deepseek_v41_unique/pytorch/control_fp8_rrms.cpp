// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
#ifndef DSV41_CONTROL_FP8_SCHEMA
#define DSV41_CONTROL_FP8_SCHEMA "custom_deepseek_v41_control_fp8_rrms_gaudi2"
#endif
#ifndef DSV41_CONTROL_FP8_PLANES
#define DSV41_CONTROL_FP8_PLANES 1
#endif
namespace {
constexpr auto schema="custom_op::" DSV41_CONTROL_FP8_SCHEMA;
struct Params {float epsilon,inverse_width;};
using Pair=std::tuple<at::Tensor,at::Tensor>;
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    TORCH_CHECK(s.size()==4,"Control FP8 requires input,weight,channel and epsilon");
    const auto x=s[0].toTensor(),w=s[1].toTensor(),c=s[2].toTensor();
    TORCH_CHECK(x.scalar_type()==at::kBFloat16 && x.dim()==2 && x.size(0)>=2 && x.size(0)<=6 &&
        x.size(1)==20480 && w.scalar_type()==at::ScalarType::Float8_e4m3fn &&
        w.sizes()==at::IntArrayRef({24*DSV41_CONTROL_FP8_PLANES,20480}) && c.scalar_type()==at::kFloat &&
        c.sizes()==at::IntArrayRef({1,24*DSV41_CONTROL_FP8_PLANES}) &&
        x.is_contiguous() && w.is_contiguous() && c.is_contiguous() &&
        x.device()==w.device() && x.device()==c.device() &&
        !x.requires_grad() && !w.requires_grad() && !c.requires_grad() &&
        s[3].toDouble()>0 && std::isnormal(float(s[3].toDouble())),"Control FP8 requires prepared C2-C6 operands");
    return {{at::kFloat,{x.size(0),24}},{at::kFloat,{x.size(0),1}}};
}
class Control:public habana::OpBackend {
public:
    Control(synDeviceId d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("dsv41_control_fp8_rrms"),
        t,{0,1},{},{},false){SetOutputMetaFn(metadata);}
    void AddNode(synapse_helpers::graph& g,const at::Stack& s) override {
        const auto m=metadata(s);const auto rows=s[0].toTensor().size(0);
        Params p{float(s[3].toDouble()),1.f/20480};
        auto stats=BuildNode(this,g,{"custom_deepseek_v41_control_statistics_gaudi2",{syn_in(0)},
            {{{rows,2,160},at::kFloat}}});
#if DSV41_CONTROL_FP8_PLANES == 2
        constexpr auto quant_guid="custom_deepseek_v41_control_pair_quant_rrms_gaudi2";
#else
        constexpr auto quant_guid="custom_deepseek_v41_control_quant_rrms_gaudi2";
#endif
        auto prepared=BuildNode(this,g,{quant_guid,{syn_in(0),stats[0].get()},
            {{{rows*DSV41_CONTROL_FP8_PLANES,20480},at::ScalarType::Float8_e4m3fn},
             {{rows*DSV41_CONTROL_FP8_PLANES,1},at::kFloat},
             {{rows,1},at::kFloat,1}},&p,sizeof(p)});
        synGEMMParams gp{false,true};
        auto product=BuildNode(this,g,{"gemm",{prepared[0].get(),syn_in(1)},
                                     {{{rows*DSV41_CONTROL_FP8_PLANES,24*DSV41_CONTROL_FP8_PLANES},at::kFloat}},
                                     &gp,sizeof(gp)});
#if DSV41_CONTROL_FP8_PLANES == 2
        auto scaled=BuildNode(this,g,{"custom_deepseek_v41_control_pair_finish_gaudi2",
            {product[0].get(),prepared[1].get(),syn_in(2)},{{m[0].shape,at::kFloat,0}}});
#else
        auto weighted=BuildNode(this,g,{"mult_fwd_f32",{product[0].get(),syn_in(2)},{{m[0].shape,at::kFloat}}});
        auto scaled=BuildNode(this,g,{"mult_fwd_f32",{weighted[0].get(),prepared[1].get()},
                                      {{m[0].shape,at::kFloat,0}}});
#endif
        syn_out(0)=std::move(scaled[0]);syn_out(1)=std::move(prepared[2]);
    }
};
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(schema,"gemm",[](const at::Stack& s) {
        const auto m=metadata(s);return habana::PartialOutputMetaDataVector{{m[0].dtype,m[0].shape},
                                                                         {m[1].dtype,m[1].shape}};
    },nullptr);
    habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Control>(d,t);});
    return true;
}();
template<bool Meta> Pair run(const at::Tensor& x,const at::Tensor& w,const at::Tensor& c,double eps) {
    const auto m=metadata({x,w,c,eps});
    if(Meta)return {at::empty(m[0].shape,x.options().dtype(at::kFloat)),
                   at::empty(m[1].shape,x.options().dtype(at::kFloat))};
    TORCH_CHECK(registered && x.device().type()==at::kHPU);
    auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    const auto out=d.execute({x,w,c,eps});return {out[0],out[1]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
 m.def(DSV41_CONTROL_FP8_SCHEMA "(Tensor input,Tensor weight,Tensor channel,float epsilon)->(Tensor,Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl(DSV41_CONTROL_FP8_SCHEMA,run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl(DSV41_CONTROL_FP8_SCHEMA,run<true>);}
