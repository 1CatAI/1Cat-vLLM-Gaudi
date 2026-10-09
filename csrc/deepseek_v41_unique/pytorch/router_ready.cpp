// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_router_ready_fp8_gaudi2";
constexpr auto guid="custom_deepseek_v41_router_ready_scaled_gaudi2";
using Pair=std::tuple<at::Tensor,at::Tensor>;
habana::OutputMetaDataVector meta(const at::Stack& s) {
    TORCH_CHECK(s.size()==7);
    const auto x=s.at(0).toTensor(),w=s.at(1).toTensor();
    const auto channel=s.at(2).toTensor(),sx=s.at(3).toTensor();
    const auto text=s.at(4).toTensor(),image=s.at(5).toTensor(),mask=s.at(6).toTensor();
    TORCH_CHECK(x.dim()==2 && x.size(0)>=2 && x.size(0)<=6 && x.size(1)==5120 &&
        x.scalar_type()==at::ScalarType::Float8_e4m3fn && w.scalar_type()==x.scalar_type() &&
        w.sizes()==at::IntArrayRef({512,5120}) && channel.scalar_type()==at::kFloat &&
        channel.sizes()==at::IntArrayRef({1,384}) && sx.scalar_type()==at::kFloat &&
        sx.sizes()==at::IntArrayRef({x.size(0),1}),"Prepared Router requires C2-C6 FFN FP8 input and checkpoint bank");
    TORCH_CHECK(text.scalar_type()==at::kFloat && text.sizes()==at::IntArrayRef({384}) &&
        image.scalar_type()==at::kFloat && image.sizes()==text.sizes() &&
        mask.scalar_type()==at::kBool && mask.sizes()==at::IntArrayRef({x.size(0)}));
    for(const auto& value:s) {
        const auto t=value.toTensor();
        TORCH_CHECK(t.device()==x.device() && t.is_contiguous() && !t.requires_grad());
    }
    return {{at::kInt,{x.size(0),6}},{at::kFloat,{x.size(0),6}}};
}
class ReadyRouter final:public habana::OpBackend {
public:
    ReadyRouter(int device,c10::ScalarType type)
        :OpBackend(device,NO_TPC+std::string("dsv41_router_ready"),type,{0,1},{},{},false) {SetOutputMetaFn(meta);}
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto out=meta(s);
        synGEMMParams params{false,true};
        auto product=BuildNode(this,graph,{"gemm",{syn_in(0),syn_in(1)},
            {{{s.at(0).toTensor().size(0),512},at::kFloat}},&params,sizeof(params)});
        auto selected=BuildNode(this,graph,{guid,
            {product.at(0).get(),syn_in(4),syn_in(5),syn_in(6),syn_in(2),syn_in(3)},
            {{out[0].shape,out[0].dtype,0},{out[1].shape,out[1].dtype,1}}});
        syn_out(0)=std::move(selected.at(0));syn_out(1)=std::move(selected.at(1));
    }
};
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(schema,guid,[](const at::Stack& s) {
        const auto out=meta(s);return habana::PartialOutputMetaDataVector{{out[0].dtype,out[0].shape},
                                                                      {out[1].dtype,out[1].shape}};
    },nullptr);
    habana::KernelRegistry().add(schema,[](synDeviceId device,c10::ScalarType type) {
        return std::make_shared<ReadyRouter>(device,type);
    });return true;
}();
template<bool Meta>Pair run(const at::Tensor& x,const at::Tensor& w,const at::Tensor& channel,const at::Tensor& sx,
                          const at::Tensor& text,const at::Tensor& image,const at::Tensor& mask) {
    const at::Stack s{x,w,channel,sx,text,image,mask};const auto out=meta(s);
    if constexpr(Meta)return {at::empty(out[0].shape,x.options().dtype(out[0].dtype)),
                             at::empty(out[1].shape,x.options().dtype(out[1].dtype))};
    TORCH_CHECK(registered && x.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto values=descriptor.execute(s);return {values.at(0),values.at(1)};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_router_ready_fp8_gaudi2(Tensor x, Tensor weight, Tensor channel, Tensor sx, Tensor text, Tensor image, Tensor mask) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_router_ready_fp8_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_router_ready_fp8_gaudi2",run<true>);}
