// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_mhc_post_collapse_gaudi2";
constexpr auto guid = "custom_deepseek_v41_mhc_post_collapse_gaudi2";
using Output = std::tuple<at::Tensor, at::Tensor>;
habana::OutputMetaDataVector meta(const at::Stack& s) {
    const auto x=s.at(0).toTensor(), r=s.at(1).toTensor(), p=s.at(2).toTensor();
    const auto c=s.at(3).toTensor(), n=s.at(4).toTensor();
    TORCH_CHECK(x.dim()==2 || x.dim()==3, "mHC value must be a row or rank-major peer rows");
    const auto tokens=x.size(-2);
    TORCH_CHECK(tokens>=1 && tokens<=6 && x.size(-1)==5120 &&
        (x.dim()==2 || (x.size(0)>=2 && x.size(0)<=8)) &&
        r.sizes()==at::IntArrayRef({tokens,4,5120}) &&
        p.sizes()==at::IntArrayRef({tokens,4}) && c.sizes()==at::IntArrayRef({tokens,4,4}) &&
        n.sizes()==p.sizes(), "mHC post/collapse requires C1-C6 model-width residuals and four-stream gates");
    for (int i=0;i<5;++i) {
        const auto t=s.at(i).toTensor();
        TORCH_CHECK(t.scalar_type()==(i<2?at::kBFloat16:at::kFloat) && t.is_contiguous() &&
            t.device()==x.device() && !t.requires_grad(), "mHC post/collapse requires matching contiguous inference operands");
    }
    return {{at::kBFloat16,r.sizes().vec()},{at::kBFloat16,{tokens,5120}}};
}
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(schema,guid,[](const at::Stack& s) {
        const auto out=meta(s);
        return habana::PartialOutputMetaDataVector{{out[0].dtype,out[0].shape},{out[1].dtype,out[1].shape}};
    },nullptr);
    return true;
}();
template<bool Meta> Output execute(const at::Tensor& x, const at::Tensor& r, const at::Tensor& p,
                                  const at::Tensor& c, const at::Tensor& n) {
    const at::Stack stack{x,r,p,c,n};const auto out=meta(stack);
    if (Meta) return {at::empty(out[0].shape,r.options()),at::empty(out[1].shape,x.options())};
    TORCH_CHECK(registered && x.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto result=op.execute(stack);
    return {result[0],result[1]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_mhc_post_collapse_gaudi2(Tensor value, Tensor residual, Tensor post, Tensor comb, Tensor next_pre) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_mhc_post_collapse_gaudi2",execute<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_mhc_post_collapse_gaudi2",execute<true>);}
