// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <cmath>
#include <initializer_list>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto kSchema="custom_op::custom_deepseek_v41_qkv_projection_publish_gaudi2";
constexpr auto kOrdered="custom_op::custom_deepseek_v41_qkv_projection_publish_ordered_gaudi2";
constexpr auto kGuid="custom_deepseek_v41_qkv_norm_publish_gaudi2";
using Result=std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
struct Params { float epsilon; int offset; };
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    TORCH_CHECK(s.size()==12,"Q/KV publication requires ten tensors, epsilon and decoded offset");
    const auto q=s.at(0).toTensor(),qn=s.at(1).toTensor(),kv=s.at(2).toTensor(),kn=s.at(3).toTensor();
    const auto pos=s.at(4).toTensor(),phase=s.at(5).toTensor(),cache=s.at(6).toTensor(),decoded=s.at(7).toTensor();
    const auto w=s.at(8).toTensor(),scale=s.at(9).toTensor();
    for(int i=0;i<10;++i) {
        const auto t=s.at(i).toTensor();
        TORCH_CHECK(t.device()==q.device() && t.is_contiguous() && !t.requires_grad(),
                    "Q/KV publication requires contiguous inference operands on one device");
    }
    TORCH_CHECK(q.scalar_type()==at::kBFloat16 && q.sizes()==at::IntArrayRef({1,1280}) &&
                qn.scalar_type()==at::kBFloat16 && qn.sizes()==at::IntArrayRef({1280}) &&
                kv.scalar_type()==at::kBFloat16 && kv.sizes()==at::IntArrayRef({1,512}) &&
                kn.scalar_type()==at::kBFloat16 && kn.sizes()==at::IntArrayRef({512}),
                "Q/KV publication requires C1 BF16 [1,1280]/[1,512] and matching norms");
    TORCH_CHECK(pos.scalar_type()==at::kInt && pos.sizes()==at::IntArrayRef({1}) &&
                phase.scalar_type()==at::kFloat && phase.dim()==2 && phase.size(1)==64 &&
                phase.size(0)>0 && phase.size(0)<=1048576,"Q/KV publication requires I32 position and F32 rotary table");
    TORCH_CHECK(cache.scalar_type()==at::kByte && cache.dim()==2 && cache.size(0)>=256 && cache.size(1)==528 &&
                decoded.scalar_type()==at::kBFloat16 && decoded.dim()==2 && decoded.size(0)>=512 && decoded.size(1)==512,
                "Q/KV publication requires canonical SWA and optional decoded circular state");
    const auto epsilon=s.at(10).toDouble();const auto offset=s.at(11).toInt();
    TORCH_CHECK(epsilon>0 && std::isnormal(static_cast<float>(epsilon)) &&
                (offset==-1 || (offset>=0 && offset%512==0 && offset<=decoded.size(0)-512)),
                "Q/KV publication requires positive normal epsilon and aligned decoded offset");
    TORCH_CHECK(w.scalar_type()==at::ScalarType::Float8_e4m3fn && w.dim()==2 &&
                (w.size(0)==8192 || w.size(0)==16384) && w.size(1)==1280 &&
                scale.scalar_type()==at::kFloat && scale.sizes()==at::IntArrayRef({1,w.size(0)}),
                "Q/KV publication requires prepared FP8 Q projection with TP-local channel scales");
    return {{at::kBFloat16,{1,w.size(0)}},{at::kBFloat16,{1,512}},{at::kInt,{16}},{at::kBFloat16,{1,1280}}};
}
class Projection final:public habana::OpBackend {
 public:
    Projection(int device,c10::ScalarType dtype):OpBackend(device,NO_TPC+std::string("dsv41_qkv_projection_publish"),
          dtype,{0,1,2,3},{},{},false){SetOutputMetaFn(metadata);}
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto output=metadata(s);Params params{static_cast<float>(s.at(10).toDouble()),static_cast<int>(s.at(11).toInt())};
        auto prepared=BuildNode(this,graph,{kGuid,
            {syn_in(0),syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(5),syn_in(6),syn_in(7)},
            {{{1,1280},at::ScalarType::Float8_e4m3fn},{{1,1},at::kFloat},
             // These are public outputs even when their producer precedes
             // Q GEMM. Register the true result indices for cached/native
             // recipe replay; syn_out assignment alone loses storage mapping.
             {{1,1280},at::kBFloat16,3},{{1,512},at::kBFloat16,1},{{16},at::kInt,2}},&params,sizeof(params)});
        synGEMMParams gp{false,true};
        auto product=BuildNode(this,graph,{"gemm",{prepared[0].get(),syn_in(8)},
            {{output[0].shape,at::kFloat}},&gp,sizeof(gp)});
        auto query=BuildNode(this,graph,{"custom_deepseek_v41_q_scale_rope_gaudi2",
            {product[0].get(),syn_in(9),prepared[1].get(),syn_in(4),syn_in(5)},
            {{output[0].shape,at::kBFloat16,0}}});
        syn_out(0)=std::move(query[0]);
        syn_out(1)=std::move(prepared[3]);syn_out(2)=std::move(prepared[4]);syn_out(3)=std::move(prepared[2]);
    }
};
const bool registered=[] {
    for(const auto* name:{kSchema,kOrdered}) {
        habana::custom_op::registerUserCustomOp(name,kGuid,[](const at::Stack& s) {
            const auto all=metadata(s);habana::PartialOutputMetaDataVector output;
            for(const auto& m:all)output.push_back({m.dtype,m.shape});return output;
        },nullptr);
        habana::KernelRegistry().add(name,[](synDeviceId device,c10::ScalarType dtype){return std::make_shared<Projection>(device,dtype);});
    }
    return true;
}();
template<bool Meta,bool Ordered=false>
Result execute(const at::Tensor& q,const at::Tensor& qn,const at::Tensor& kv,const at::Tensor& kn,
    const at::Tensor& pos,const at::Tensor& phase,const at::Tensor& cache,const at::Tensor& decoded,
    const at::Tensor& weight,const at::Tensor& scale,double epsilon,int64_t offset) {
    const at::Stack s{q,qn,kv,kn,pos,phase,cache,decoded,weight,scale,epsilon,offset};const auto m=metadata(s);
    if(Meta)return {at::empty(m[0].shape,q.options()),at::empty(m[1].shape,q.options()),
                    at::empty(m[2].shape,pos.options()),at::empty(m[3].shape,q.options())};
    TORCH_CHECK(registered && q.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Ordered?kOrdered:kSchema);
    const auto out=op.execute(s);return {out[0],out[1],out[2],out[3]};
}
at::Tensor unwrap(const at::Tensor& t) {
    if(!at::functionalization::impl::isFunctionalTensor(t))return t;
    at::functionalization::impl::sync(t);return at::functionalization::impl::from_functional_tensor(t);
}
Result functionalize(const at::Tensor& q,const at::Tensor& qn,const at::Tensor& kv,const at::Tensor& kn,
    const at::Tensor& pos,const at::Tensor& phase,const at::Tensor& cache,const at::Tensor& decoded,
    const at::Tensor& weight,const at::Tensor& scale,double epsilon,int64_t offset) {
    auto c=unwrap(cache),d=unwrap(decoded);Result out;
    static auto handle=c10::Dispatcher::singleton().findSchemaOrThrow(kOrdered, "")
        .typed<Result(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
                      const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
                      double,int64_t)>();
    {at::AutoDispatchSkipFunctionalize guard;
     out=handle.call(unwrap(q),unwrap(qn),unwrap(kv),unwrap(kn),unwrap(pos),unwrap(phase),c,d,
                     unwrap(weight),unwrap(scale),epsilon,offset);}
    for(const auto& p:{std::make_pair(cache,c),std::make_pair(decoded,d)}) {
        at::functionalization::impl::replace_(p.first,p.second);
        at::functionalization::impl::commit_update(p.first);at::functionalization::impl::sync(p.first);
    }
    return out;
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_qkv_projection_publish_gaudi2(Tensor q, Tensor qnorm, Tensor kv, Tensor kvnorm, Tensor positions, Tensor phase, Tensor(a!) cache, Tensor(b!) decoded, Tensor weight, Tensor scale, float epsilon, int offset) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("custom_deepseek_v41_qkv_projection_publish_ordered_gaudi2(Tensor q, Tensor qnorm, Tensor kv, Tensor kvnorm, Tensor positions, Tensor phase, Tensor cache, Tensor decoded, Tensor weight, Tensor scale, float epsilon, int offset) -> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_qkv_projection_publish_gaudi2",execute<false>);
    m.impl("custom_deepseek_v41_qkv_projection_publish_ordered_gaudi2",execute<false,true>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_qkv_projection_publish_gaudi2",execute<true>);
    m.impl("custom_deepseek_v41_qkv_projection_publish_ordered_gaudi2",execute<true,true>);
}
TORCH_LIBRARY_IMPL(custom_op,Functionalize,m) {
    m.impl("custom_deepseek_v41_qkv_projection_publish_gaudi2",functionalize);
}
