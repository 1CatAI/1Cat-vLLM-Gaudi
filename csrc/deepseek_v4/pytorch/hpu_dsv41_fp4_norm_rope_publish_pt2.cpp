// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <cmath>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_fp4_norm_rope_publish_gaudi2";
constexpr auto ordered="custom_op::custom_deepseek_v41_fp4_norm_rope_publish_ordered_gaudi2";
constexpr auto guid="custom_deepseek_v41_fp4_norm_rope_publish_gaudi2";
struct Params {float epsilon;int ratio;int main_enabled;int hot_enabled;int index_enabled;};
habana::PartialOutputMetaDataVector metadata(const at::Stack& s) {
    const auto x=s.at(0).toTensor();const auto ratio=s.at(12).toInt();
    TORCH_CHECK((ratio==1 || ratio==2) && s.at(11).toDouble()>0 && std::isnormal(float(s.at(11).toDouble())),
                "Compressor publication requires ratio1/2 and positive normal epsilon");
    const std::vector<std::vector<int64_t>> fixed={{1,512},{1,128},{128},{1}};
    for(unsigned i=0;i<11;++i) {
        const auto t=s.at(i).toTensor();
        const auto dtype=(i==3 || i==5)?at::kInt:i==4?at::kFloat:(i==6 || i==7)?at::kByte:at::kBFloat16;
        TORCH_CHECK(t.scalar_type()==dtype && t.device()==x.device() && t.is_contiguous() && !t.requires_grad(),
                    "Invalid compressor publication inference tensor contract");
        if(i<4)TORCH_CHECK(t.sizes().vec()==fixed[i],"Invalid compressor publication source shape");
        if(i==4)TORCH_CHECK(t.dim()==2 && t.size(1)==64 && t.size(0)>=1 && t.size(0)<=1048576,"Invalid phase");
        if(i==5)TORCH_CHECK(t.dim()==1 && t.numel()>=1,"Invalid page map");
        if(i>=6)TORCH_CHECK(t.dim()==2 && t.size(0)>=1 && t.size(1)==(i==6?288:i==7?68:i==8?512:128),
                            "Invalid scheduler-owned publication state");
    }
    return {{at::kInt,{36}}};
}
const bool registered=[] {
    for(const auto* name:{schema,ordered}) habana::custom_op::registerUserCustomOp(name,guid,metadata,
        [](const at::Stack& s,size_t& size)->std::shared_ptr<void>{size=sizeof(Params);
            return std::make_shared<Params>(Params{float(s.at(11).toDouble()),int(s.at(12).toInt()),
                int(s.at(13).toBool()),int(s.at(14).toBool()),int(s.at(15).toBool())});});
    return true;
}();
template<bool Meta,bool Ordered=false>
at::Tensor run(const at::Tensor& latent,const at::Tensor& raw,const at::Tensor& norm,const at::Tensor& pos,const at::Tensor& phase,const at::Tensor& pages,const at::Tensor& main,const at::Tensor& index,const at::Tensor& decoded,const at::Tensor& hot,const at::Tensor& mirror,double eps,int64_t ratio,bool main_enabled,bool hot_enabled,bool index_enabled) {
    const at::Stack s{latent,raw,norm,pos,phase,pages,main,index,decoded,hot,mirror,eps,ratio,main_enabled,hot_enabled,index_enabled};metadata(s);
    if(Meta)return at::empty({36},latent.options().dtype(at::kInt));
    TORCH_CHECK(registered && latent.device().type()==at::kHPU);
    auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Ordered?ordered:schema);
    return d.execute(s).at(0);
}
at::Tensor unwrap(const at::Tensor& t) {
    if(!at::functionalization::impl::isFunctionalTensor(t))return t;
    at::functionalization::impl::sync(t);return at::functionalization::impl::from_functional_tensor(t);
}
at::Tensor functionalize(const at::Tensor& latent,const at::Tensor& raw,const at::Tensor& norm,const at::Tensor& pos,const at::Tensor& phase,const at::Tensor& pages,const at::Tensor& main,const at::Tensor& index,const at::Tensor& decoded,const at::Tensor& hot,const at::Tensor& mirror,double eps,int64_t ratio,bool main_enabled,bool hot_enabled,bool index_enabled) {
    auto l=unwrap(latent),r=unwrap(raw),n=unwrap(norm),p=unwrap(pos),ph=unwrap(phase),pg=unwrap(pages);
    auto m=unwrap(main),i=unwrap(index),d=unwrap(decoded),h=unwrap(hot),mi=unwrap(mirror);
    static auto handle=c10::Dispatcher::singleton().findSchemaOrThrow(ordered, "");
    at::Stack s{l,r,n,p,ph,pg,m,i,d,h,mi,eps,ratio,main_enabled,hot_enabled,index_enabled};
    {at::AutoDispatchSkipFunctionalize guard;handle.callBoxed(&s);}
    for(const auto& pair:{std::make_pair(main,m),std::make_pair(index,i),std::make_pair(decoded,d),
                         std::make_pair(hot,h),std::make_pair(mirror,mi)}) {
        at::functionalization::impl::replace_(pair.first,pair.second);
        at::functionalization::impl::commit_update(pair.first);at::functionalization::impl::sync(pair.first);
    }
    return s.at(0).toTensor();
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m){
    m.def("custom_deepseek_v41_fp4_norm_rope_publish_ordered_gaudi2(Tensor latent, Tensor raw, Tensor norm, Tensor pos, Tensor phase, Tensor pages, Tensor main, Tensor index, Tensor decoded, Tensor hot, Tensor mirror, float eps, int ratio, bool main_enabled, bool hot_enabled, bool index_enabled) -> Tensor");
    m.def("custom_deepseek_v41_fp4_norm_rope_publish_gaudi2(Tensor latent, Tensor raw, Tensor norm, Tensor pos, Tensor phase, Tensor pages, Tensor(a!) main, Tensor(b!) index, Tensor(c!) decoded, Tensor(d!) hot, Tensor(e!) mirror, float eps, int ratio, bool main_enabled, bool hot_enabled, bool index_enabled) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("custom_deepseek_v41_fp4_norm_rope_publish_gaudi2",run<false>);m.impl("custom_deepseek_v41_fp4_norm_rope_publish_ordered_gaudi2",run<false,true>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("custom_deepseek_v41_fp4_norm_rope_publish_gaudi2",run<true>);m.impl("custom_deepseek_v41_fp4_norm_rope_publish_ordered_gaudi2",run<true,true>);}
TORCH_LIBRARY_IMPL(custom_op,Functionalize,m){m.impl("custom_deepseek_v41_fp4_norm_rope_publish_gaudi2",functionalize);}
