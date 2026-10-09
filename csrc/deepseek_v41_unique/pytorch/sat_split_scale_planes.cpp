// SPDX-License-Identifier: Apache-2.0
#define DSV41_N256_BACKEND_ONLY 1
#define DSV41_N256_SPLIT_SCALE_PLANES 1
#include "../../deepseek_v4/pytorch/hpu_dsv41_expert_n256_pt2.cpp"
namespace {
#ifndef DSV41_SPLIT_SCALE_OPERATOR
#define DSV41_SPLIT_SCALE_OPERATOR custom_deepseek_v41_expert_n256_moe_split_scale_planes_fp8_gaudi2
#endif
#define DSV41_STRINGIFY_INNER(X) #X
#define DSV41_STRINGIFY(X) DSV41_STRINGIFY_INNER(X)
constexpr auto name="custom_op::" DSV41_STRINGIFY(DSV41_SPLIT_SCALE_OPERATOR);
void split_contract(const at::Stack& stack) {
#if defined(DSV41_N256_SHARED_SCALE_INPUT) && DSV41_N256_SHARED_SCALE_INPUT
    TORCH_CHECK(stack.size()==20,"Shared scale chain requires nineteen tensors and qualification");
    const auto& raw=stack.at(16).toTensor();const auto& sx_shared=stack.at(17).toTensor();
    const auto& channel_shared=stack.at(18).toTensor();const auto& value=stack.at(0).toTensor();
    TORCH_CHECK(raw.sizes()==at::IntArrayRef({value.size(0),5120})&&raw.scalar_type()==at::kBFloat16&&
        sx_shared.sizes()==at::IntArrayRef({value.size(0),1})&&sx_shared.scalar_type()==at::kFloat&&
        channel_shared.sizes()==at::IntArrayRef({1,5120})&&channel_shared.scalar_type()==at::kFloat,
        "Shared scale chain retains BF16 raw product and row/channel FP32 scale owners");
    for(const auto& tensor:{raw,sx_shared,channel_shared})
        TORCH_CHECK(tensor.device()==value.device()&&tensor.is_contiguous()&&!tensor.requires_grad());
#else
    TORCH_CHECK(stack.size()==17,"Split scale planes expect sixteen tensors and the qualification flag");
#endif
    prequant_contract(stack);
    const auto& x=stack.at(0).toTensor();
    TORCH_CHECK(stack.back().toBool() && x.size(0)>=2 && x.size(0)<=6 &&
                stack.at(1).toTensor().size(1)==6,"Split scale planes require qualified Target C2-C6/top6");
    for(int i=0;i<2;++i) {
        const auto& source=stack.at(5+i).toTensor();
        const auto& q=stack.at(3+i).toTensor();
        const auto& groups=stack.at(12+2*i).toTensor();
        const auto& channels=stack.at(13+2*i).toTensor();
        const bool compact=source.size(2)==q.size(2)/16+128;
        const auto width=compact?source.size(2)-128:source.size(2);
        TORCH_CHECK(groups.scalar_type()==at::kShort && groups.device()==source.device() &&
            groups.sizes()==at::IntArrayRef({source.size(0),source.size(1),width}) &&
            groups.strides()==source.strides() && groups.storage_offset()==source.storage_offset() &&
            groups.is_alias_of(source) && !groups.requires_grad(),"Group scale must retain its original storage/strides");
        TORCH_CHECK(channels.scalar_type()==at::kShort && channels.device()==source.device() &&
            channels.sizes()==at::IntArrayRef({source.size(0),source.size(1),128}) &&
            !channels.requires_grad(),"Channel-code tensor must match prepared expert/N axes");
        if(compact) {
            TORCH_CHECK(channels.strides()==source.strides() && channels.is_alias_of(source) &&
                channels.storage_offset()==source.storage_offset()+width,
                "Compact channel codes must be a storage-sharing tail view");
        } else TORCH_CHECK(channels.is_contiguous(),"Legacy unused channel placeholder must be contiguous");
    }
}
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(name,kDecode,[](const at::Stack& stack) {
        split_contract(stack);return habana::PartialOutputMetaDataVector{{at::kBFloat16,moe_shape(stack,true)}};
    },nullptr);
    habana::KernelRegistry().add(name,[](synDeviceId device,c10::ScalarType type) {
        return std::make_shared<PreparedV41>(device,type,true,true,false,true,
            true,false,true,false,true,false,false,false,true,false,false,true);
    });return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x,const at::Tensor& ids,const at::Tensor& routing,
    const at::Tensor& q13,const at::Tensor& q2,const at::Tensor& s13,const at::Tensor& s2,
    const at::Tensor& lookup,const at::Tensor& c13,const at::Tensor& c2,
    const at::Tensor& quantized,const at::Tensor& sx,const at::Tensor& g13,const at::Tensor& d13,
    const at::Tensor& g2,const at::Tensor& d2,
#if defined(DSV41_N256_SHARED_SCALE_INPUT) && DSV41_N256_SHARED_SCALE_INPUT
    const at::Tensor& shared,const at::Tensor& shared_scale,const at::Tensor& shared_channel,
#endif
    bool qualified) {
    const at::Stack stack{x,ids,routing,q13,q2,s13,s2,lookup,c13,c2,quantized,sx,g13,d13,g2,d2,
#if defined(DSV41_N256_SHARED_SCALE_INPUT) && DSV41_N256_SHARED_SCALE_INPUT
        shared,shared_scale,shared_channel,
#endif
        qualified};
    split_contract(stack);
    if(Meta)return at::empty(moe_shape(stack,true),x.options());
    TORCH_CHECK(ready && x.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
#if defined(DSV41_N256_SHARED_SCALE_INPUT) && DSV41_N256_SHARED_SCALE_INPUT
    m.def(DSV41_STRINGIFY(DSV41_SPLIT_SCALE_OPERATOR) "(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, Tensor channel13, Tensor channel2, Tensor quantized, Tensor activation_scale, Tensor group13, Tensor code13, Tensor group2, Tensor code2, Tensor shared, Tensor shared_scale, Tensor shared_channel, bool qualified) -> Tensor");
#else
    m.def(DSV41_STRINGIFY(DSV41_SPLIT_SCALE_OPERATOR) "(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, Tensor channel13, Tensor channel2, Tensor quantized, Tensor activation_scale, Tensor group13, Tensor code13, Tensor group2, Tensor code2, bool qualified) -> Tensor");
#endif
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl(DSV41_STRINGIFY(DSV41_SPLIT_SCALE_OPERATOR),run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl(DSV41_STRINGIFY(DSV41_SPLIT_SCALE_OPERATOR),run<true>);}
