// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#ifndef DSV41_MHC_CARRIED_RRMS_ONLY
#define DSV41_MHC_CARRIED_RRMS_ONLY 0
#endif
namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_mhc_mme_post_collapse_gaudi2";
constexpr auto guid = "custom_deepseek_v41_mhc_mme_post_collapse_gaudi2";
using Output = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
habana::OutputMetaDataVector meta(const at::Stack& s) {
    const auto x=s.at(0).toTensor(),r=s.at(1).toTensor(),raw=s.at(2).toTensor();
    const auto scale=s.at(3).toTensor(),base=s.at(4).toTensor();
    TORCH_CHECK(x.dim()==2 || x.dim()==3,"Expected BF16 row or rank-major peer rows");
    const auto tokens=x.size(-2);
    TORCH_CHECK(tokens>=1 && tokens<=6 && x.size(-1)==5120 &&
        (x.dim()==2 || (x.size(0)>=2 && x.size(0)<=8)) &&
        r.sizes()==at::IntArrayRef({tokens,4,5120}) && ((!DSV41_MHC_CARRIED_RRMS_ONLY && raw.sizes()==at::IntArrayRef({tokens,48})) ||
        raw.sizes()==at::IntArrayRef({tokens,25})) &&
        scale.sizes()==at::IntArrayRef({3}) && base.sizes()==at::IntArrayRef({24}),"Invalid mHC MME post/collapse geometry");
    for(int i=0;i<5;++i) {
        const auto t=s.at(i).toTensor();
        TORCH_CHECK(t.scalar_type()==(i<2?at::kBFloat16:at::kFloat) && t.is_contiguous() &&
            t.device()==x.device() && !t.requires_grad(),"Expected matching contiguous inference operands");
    }
    TORCH_CHECK(s.at(5).toDouble()>0,"Expected positive RMS epsilon");
    return {{at::kBFloat16,r.sizes().vec()},{at::kBFloat16,{tokens,5120}},{at::kFloat,{tokens,24}}};
}

const bool registered=[] {
    habana::custom_op::registerUserCustomOp(schema,guid,[](const at::Stack& s) {
        const auto out=meta(s);
        return habana::PartialOutputMetaDataVector{{out[0].dtype,out[0].shape},{out[1].dtype,out[1].shape},{out[2].dtype,out[2].shape}};
    },[](const at::Stack& s,size_t& size)->std::shared_ptr<void>{size=sizeof(float);return std::make_shared<float>(static_cast<float>(s.at(5).toDouble()));});
    return true;
}();
template<bool Meta> Output execute(const at::Tensor& x, const at::Tensor& r, const at::Tensor& p,
                                  const at::Tensor& c, const at::Tensor& n,double eps) {
    const at::Stack stack{x,r,p,c,n,eps};const auto out=meta(stack);
    if (Meta) return {at::empty(out[0].shape,r.options()),at::empty(out[1].shape,x.options()),at::empty(out[2].shape,x.options().dtype(at::kFloat))};
    TORCH_CHECK(registered && x.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto result=op.execute(stack);
    return {result[0],result[1],result[2]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_mhc_mme_post_collapse_gaudi2(Tensor value, Tensor residual, Tensor raw, Tensor scale, Tensor base, float eps) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_mhc_mme_post_collapse_gaudi2",execute<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_mhc_mme_post_collapse_gaudi2",execute<true>);}
