// SPDX-License-Identifier: Apache-2.0
// The common C1 FP4 codec publishes both canonical bytes and decoded values.
#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_fp4_paged_decoded_rows_gaudi2";
constexpr auto ordered="custom_op::custom_deepseek_v41_fp4_paged_decoded_rows_ordered_gaudi2";
constexpr auto guid="custom_deepseek_v41_fp4_paged_decoded_rows_gaudi2";
void contract(const at::Stack& s) {
    const auto main=s[0].toTensor(),index=s[1].toTensor(),mv=s[2].toTensor(),iv=s[3].toTensor();
    const auto physical=s[4].toTensor(),logical=s[5].toTensor(),decoded=s[6].toTensor();
    const auto t=mv.size(0);
    TORCH_CHECK(mv.dim()==2 && t>=1 && t<=6 && mv.size(1)==512 &&
                iv.sizes()==at::IntArrayRef({t,128}) && main.dim()==2 && main.size(0)>0 &&
                main.size(0)<=0x7ffffdffLL && main.size(1)==288 &&
                index.sizes()==at::IntArrayRef({main.size(0),68}) &&
                physical.sizes()==at::IntArrayRef({t}) && logical.sizes()==physical.sizes() &&
                decoded.dim()==2 && decoded.size(0)>=512 && decoded.size(0)<=32768 && decoded.size(1)==512,
                "Paged decoded publication requires canonical C1-C6 rows and bounded logical mirror");
    for(int i=0;i<7;++i) {
        const auto a=s[i].toTensor();
        const auto type=i<2?at::kByte:(i==4||i==5)?at::kInt:at::kBFloat16;
        TORCH_CHECK(a.scalar_type()==type && a.device()==mv.device() && a.is_contiguous() &&
                    !a.requires_grad(),"Paged decoded publication dtype/device/stride mismatch");
    }
}
const bool ready=[] {
    for(auto name:{schema,ordered})habana::custom_op::registerUserCustomOp(name,guid,[](const at::Stack& s) {
        contract(s);return habana::PartialOutputMetaDataVector{{at::kInt,{s[2].toTensor().size(0),36}}};
    },nullptr);
    return true;
}();
template<bool Meta,bool Ordered> at::Tensor execute(const at::Tensor& main,const at::Tensor& index,
    const at::Tensor& mv,const at::Tensor& iv,const at::Tensor& physical,const at::Tensor& logical,
    const at::Tensor& decoded) {
    const at::Stack s{main,index,mv,iv,physical,logical,decoded};contract(s);
    if(Meta)return at::empty({mv.size(0),36},physical.options());
    TORCH_CHECK(ready && mv.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Ordered?ordered:schema);
    return descriptor.execute(s)[0];
}
at::Tensor unwrap(const at::Tensor& t) {
    if(!at::functionalization::impl::isFunctionalTensor(t))return t;
    at::functionalization::impl::sync(t);return at::functionalization::impl::from_functional_tensor(t);
}
at::Tensor functionalize(const at::Tensor& main,const at::Tensor& index,const at::Tensor& mv,
    const at::Tensor& iv,const at::Tensor& physical,const at::Tensor& logical,const at::Tensor& decoded) {
    auto a=unwrap(main),b=unwrap(index),c=unwrap(mv),d=unwrap(iv),e=unwrap(physical),f=unwrap(logical),g=unwrap(decoded);
    static auto handle=c10::Dispatcher::singleton().findSchemaOrThrow(ordered, "")
        .typed<at::Tensor(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
                         const at::Tensor&,const at::Tensor&,const at::Tensor&)>();
    at::Tensor result;
    {at::AutoDispatchSkipFunctionalize guard;result=handle.call(a,b,c,d,e,f,g);}
    for(auto pair:{std::pair<const at::Tensor*,const at::Tensor*>{&main,&a},{&index,&b},{&decoded,&g}}) {
        at::functionalization::impl::replace_(*pair.first,*pair.second);
        at::functionalization::impl::commit_update(*pair.first);
        at::functionalization::impl::sync(*pair.first);
    }
    return result;
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
 m.def("custom_deepseek_v41_fp4_paged_decoded_rows_gaudi2(Tensor(a!) main, Tensor(b!) index, Tensor main_value, Tensor index_value, Tensor physical, Tensor logical, Tensor(c!) decoded) -> Tensor");
 m.def("custom_deepseek_v41_fp4_paged_decoded_rows_ordered_gaudi2(Tensor main, Tensor index, Tensor main_value, Tensor index_value, Tensor physical, Tensor logical, Tensor decoded) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
 m.impl("custom_deepseek_v41_fp4_paged_decoded_rows_gaudi2",execute<false,false>);
 m.impl("custom_deepseek_v41_fp4_paged_decoded_rows_ordered_gaudi2",execute<false,true>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
 m.impl("custom_deepseek_v41_fp4_paged_decoded_rows_gaudi2",execute<true,false>);
 m.impl("custom_deepseek_v41_fp4_paged_decoded_rows_ordered_gaudi2",execute<true,true>);
}
TORCH_LIBRARY_IMPL(custom_op,Functionalize,m) {m.impl("custom_deepseek_v41_fp4_paged_decoded_rows_gaudi2",functionalize);}
