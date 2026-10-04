// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <cmath>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_kv_norm_reuse_mla_gaudi2";
constexpr auto ordered="custom_op::custom_deepseek_v41_kv_norm_reuse_mla_ordered_gaudi2";
struct Params{float epsilon;float inverse_width;};
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    const auto q=s.at(0).toTensor();
    TORCH_CHECK(q.dim()==3 && q.size(0)==1 && q.size(1)>=1 && q.size(1)<=64 && q.size(2)==512,
                "Fused reuse MLA requires Q [1,H1..64,512]");
    const std::vector<std::vector<int64_t>> shapes={{1,512},{512},{256,528},{1,640,512},{1,640},{1}};
    for(unsigned i=0;i<11;++i) {
        const auto t=s.at(i).toTensor();
        const auto dtype=(i==3)?at::kByte:(i==6 || i==10)?at::kInt:
                          (i==5 || i==7 || i==8 || i==9)?at::kFloat:at::kBFloat16;
        TORCH_CHECK(t.scalar_type()==dtype && t.device()==q.device() && t.is_contiguous() && !t.requires_grad(),
                    "Fused reuse MLA requires matching inference tensors");
        if(i>=1 && i<=6)TORCH_CHECK(t.sizes().vec()==shapes[i-1],"Invalid fused reuse state geometry");
        if(i==7)TORCH_CHECK(t.dim()==2 && t.size(1)==64 && t.size(0)>=1 && t.size(0)<=1048576,"Invalid phase");
        if(i==8)TORCH_CHECK(t.sizes()==at::IntArrayRef({q.size(1)}),"Invalid sink");
        if(i==9 || i==10)TORCH_CHECK(t.sizes()==at::IntArrayRef({1}),"Invalid scale/length");
    }
    TORCH_CHECK(s.at(11).toDouble()>0 && std::isnormal(float(s.at(11).toDouble())),"Invalid norm epsilon");
    return {{at::kBFloat16,q.sizes().vec()}};
}
class Kernel final:public habana::OpBackend {
public:
    Kernel(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("dsv41_kv_norm_reuse_mla"),t,{0},{},{},false){SetOutputMetaFn(metadata);}
    void AddNode(synapse_helpers::graph& g,const at::Stack& s) override {
        auto meta=metadata(s);const int64_t heads=s.at(0).toTensor().size(1);Params p{float(s.at(11).toDouble()),1.0f/512.0f};
        auto kv=BuildNode(this,g,{"custom_deepseek_v41_kv_norm_reuse_gather_gaudi2",
            {syn_in(1),syn_in(2),syn_in(7),syn_in(3),syn_in(4),syn_in(5),syn_in(6),syn_in(10)},
            {{{1,640,512},at::kBFloat16},{{1,640,512},at::kFloat},{{1,640},at::kFloat}},&p,sizeof(p)});
        synGEMMParams qk{false,true},pv{false,false};
        auto scores=BuildNode(this,g,{"batch_gemm",{syn_in(0),kv[0].get()},{{{1,heads,640},at::kFloat}},&qk,sizeof(qk)});
        auto prob=BuildNode(this,g,{"custom_deepseek_v41_selected_mla_softmax_gaudi2",
            {scores[0].get(),kv[2].get(),syn_in(8),syn_in(9)},{{{1,heads,640},at::kFloat}}});
        auto product=BuildNode(this,g,{"batch_gemm",{prob[0].get(),kv[1].get()},{{{1,heads,512},at::kFloat}},&pv,sizeof(pv)});
        syn_out(0)=std::move(BuildNode(this,g,{"cast_f32_to_bf16",{product[0].get()},{{meta[0].shape,at::kBFloat16,0}}})[0]);
    }
};
const bool registered=[] {
    for(const auto* name:{schema,ordered}) {
        habana::custom_op::registerUserCustomOp(name,"batch_gemm",[](const at::Stack& s){auto m=metadata(s);return habana::PartialOutputMetaDataVector{{m[0].dtype,m[0].shape}};},nullptr);
        habana::KernelRegistry().add(name,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Kernel>(d,t);});
    }return true;
}();
template<bool Meta,bool Ordered=false> at::Tensor run(const at::Tensor& q,const at::Tensor& raw,const at::Tensor& norm,const at::Tensor& swa,const at::Tensor& main,const at::Tensor& mask,const at::Tensor& pos,const at::Tensor& phase,const at::Tensor& sink,const at::Tensor& scale,const at::Tensor& lengths,double eps) {
    at::Stack s{q,raw,norm,swa,main,mask,pos,phase,sink,scale,lengths,eps};auto m=metadata(s);
    if(Meta)return at::empty(m[0].shape,q.options());
    TORCH_CHECK(registered && q.device().type()==at::kHPU);
    auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Ordered?ordered:schema);return d.execute(s)[0];
}
at::Tensor unwrap(const at::Tensor& t) {
    if(!at::functionalization::impl::isFunctionalTensor(t))return t;
    at::functionalization::impl::sync(t);return at::functionalization::impl::from_functional_tensor(t);
}
at::Tensor functionalize(const at::Tensor& q,const at::Tensor& raw,const at::Tensor& norm,const at::Tensor& swa,const at::Tensor& main,const at::Tensor& mask,const at::Tensor& pos,const at::Tensor& phase,const at::Tensor& sink,const at::Tensor& scale,const at::Tensor& lengths,double eps) {
    auto state=unwrap(swa);at::Stack s{unwrap(q),unwrap(raw),unwrap(norm),state,unwrap(main),unwrap(mask),
        unwrap(pos),unwrap(phase),unwrap(sink),unwrap(scale),unwrap(lengths),eps};
    static auto handle=c10::Dispatcher::singleton().findSchemaOrThrow(ordered, "");
    {at::AutoDispatchSkipFunctionalize guard;handle.callBoxed(&s);}
    at::functionalization::impl::replace_(swa,state);at::functionalization::impl::commit_update(swa);
    at::functionalization::impl::sync(swa);return s[0].toTensor();
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m){
    m.def("custom_deepseek_v41_kv_norm_reuse_mla_gaudi2(Tensor q, Tensor raw, Tensor norm, Tensor(a!) swa, Tensor main, Tensor mask, Tensor pos, Tensor phase, Tensor sink, Tensor scale, Tensor lengths, float eps) -> Tensor");
    m.def("custom_deepseek_v41_kv_norm_reuse_mla_ordered_gaudi2(Tensor q, Tensor raw, Tensor norm, Tensor swa, Tensor main, Tensor mask, Tensor pos, Tensor phase, Tensor sink, Tensor scale, Tensor lengths, float eps) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("custom_deepseek_v41_kv_norm_reuse_mla_gaudi2",run<false>);m.impl("custom_deepseek_v41_kv_norm_reuse_mla_ordered_gaudi2",run<false,true>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("custom_deepseek_v41_kv_norm_reuse_mla_gaudi2",run<true>);m.impl("custom_deepseek_v41_kv_norm_reuse_mla_ordered_gaudi2",run<true,true>);}
TORCH_LIBRARY_IMPL(custom_op,Functionalize,m){m.impl("custom_deepseek_v41_kv_norm_reuse_mla_gaudi2",functionalize);}
