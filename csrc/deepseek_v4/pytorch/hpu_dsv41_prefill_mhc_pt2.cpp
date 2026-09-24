// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"
namespace {
constexpr const char* schema="custom_op::custom_deepseek_v41_prefill_mhc_post_gaudi2";
constexpr const char* collapse_schema="custom_op::custom_deepseek_v41_prefill_mhc_collapse_gaudi2";
constexpr const char* rrms_schema="custom_op::custom_deepseek_v41_prefill_mhc_rrms_bf16_gaudi2";
constexpr const char* post_prepare_schema="custom_op::custom_deepseek_v41_prefill_mhc_post_prepare_gaudi2";
void validate(const at::Tensor& x, const at::Tensor& r, const at::Tensor& p, const at::Tensor& c) {
    TORCH_CHECK(x.scalar_type()==at::kBFloat16 && r.scalar_type()==at::kBFloat16 &&
                p.scalar_type()==at::kFloat && c.scalar_type()==at::kFloat &&
                x.dim()==2 && x.size(0)>=1 && x.size(0)<=8192 && x.size(1)==5120 &&
                r.sizes()==at::IntArrayRef({x.size(0),4,5120}) &&
                p.sizes()==at::IntArrayRef({x.size(0),4}) && c.sizes()==at::IntArrayRef({x.size(0),4,4}),
                "Prefill mHC requires BF16 [T1..8192,5120]/[T,4,5120], FP32 [T,4]/[T,4,4]");
    for (const auto& t:{x,r,p,c})
        TORCH_CHECK(t.is_contiguous() && t.device()==x.device(), "Prefill mHC requires contiguous same-device tensors");
}
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(schema,"custom_deepseek_v41_prefill_mhc_post_gaudi2",
        [](const at::Stack& stack) {
            const auto x=stack.at(0).toTensor(),r=stack.at(1).toTensor(),p=stack.at(2).toTensor(),c=stack.at(3).toTensor();
            validate(x,r,p,c);
            habana::PartialOutputMetaData out;
            out.dtype=at::kBFloat16; out.shape=r.sizes().vec();
            return habana::PartialOutputMetaDataVector{out};
        },nullptr);
    return true;
}();
void validate_collapse(const at::Tensor& r, const at::Tensor& p) {
    TORCH_CHECK(r.scalar_type()==at::kBFloat16 && p.scalar_type()==at::kFloat &&
                r.dim()==3 && r.size(0)>=1 && r.size(0)<=8192 && r.size(1)==4 && r.size(2)==5120 &&
                p.sizes()==at::IntArrayRef({r.size(0),4}),
                "Prefill mHC collapse requires BF16 [T1..8192,4,5120] and FP32 [T,4]");
    TORCH_CHECK(r.is_contiguous() && p.is_contiguous() && r.device()==p.device(),
                "Prefill mHC collapse requires contiguous same-device tensors");
}
const bool collapse_registered=[] {
    habana::custom_op::registerUserCustomOp(collapse_schema,"custom_deepseek_v41_prefill_mhc_collapse_gaudi2",
        [](const at::Stack& stack) {
            const auto r=stack.at(0).toTensor(),p=stack.at(1).toTensor();
            validate_collapse(r,p);
            habana::PartialOutputMetaData out;
            out.dtype=at::kBFloat16; out.shape={r.size(0),r.size(2)};
            return habana::PartialOutputMetaDataVector{out};
        },nullptr);
    return true;
}();
void validate_rrms(const at::Tensor& r) {
    TORCH_CHECK(r.scalar_type()==at::kBFloat16 && r.dim()==3 &&
                r.size(0)>=1 && r.size(0)<=8192 && r.size(1)==4 && r.size(2)==5120 &&
                r.is_contiguous(),
                "Prefill mHC RRMS requires contiguous BF16 [T1..8192,4,5120]");
}
const bool rrms_registered=[] {
    habana::custom_op::registerUserCustomOp(rrms_schema,"custom_deepseek_v41_prefill_mhc_rrms_bf16_gaudi2",
        [](const at::Stack& stack) {
            const auto r=stack.at(0).toTensor(); validate_rrms(r);
            habana::PartialOutputMetaData out;
            out.dtype=at::kFloat; out.shape={r.size(0),1};
            return habana::PartialOutputMetaDataVector{out};
        },nullptr);
    return true;
}();
void validate_post_prepare(const at::Tensor& x,const at::Tensor& r,const at::Tensor& p,
                           const at::Tensor& c,const at::Tensor& n) {
    validate(x,r,p,c);
    TORCH_CHECK(n.scalar_type()==at::kFloat && n.sizes()==at::IntArrayRef({x.size(0),4}) &&
                n.is_contiguous() && n.device()==x.device(),
                "Prefill mHC post-prepare requires contiguous FP32 next_pre [T,4]");
}
const bool post_prepare_registered=[] {
    habana::custom_op::registerUserCustomOp(post_prepare_schema,
        "custom_deepseek_v41_prefill_mhc_post_prepare_gaudi2",
        [](const at::Stack& stack) {
            const auto x=stack.at(0).toTensor(),r=stack.at(1).toTensor(),p=stack.at(2).toTensor();
            const auto c=stack.at(3).toTensor(),n=stack.at(4).toTensor();
            validate_post_prepare(x,r,p,c,n);
            return habana::PartialOutputMetaDataVector{
                {at::kBFloat16,r.sizes().vec()},
                {at::kBFloat16,{x.size(0),x.size(1)}},
                {at::kFloat,{x.size(0),1}}};
        },nullptr);
    return true;
}();
at::Tensor run(const at::Tensor& x,const at::Tensor& r,const at::Tensor& p,const at::Tensor& c) {
    validate(x,r,p,c); TORCH_CHECK(registered && x.device().type()==at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute({x,r,p,c}).at(0);
}
at::Tensor meta(const at::Tensor& x,const at::Tensor& r,const at::Tensor& p,const at::Tensor& c) {
    validate(x,r,p,c); return at::empty(r.sizes(),r.options());
}
at::Tensor run_collapse(const at::Tensor& r,const at::Tensor& p) {
    validate_collapse(r,p); TORCH_CHECK(collapse_registered && r.device().type()==at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(collapse_schema);
    return descriptor.execute({r,p}).at(0);
}
at::Tensor meta_collapse(const at::Tensor& r,const at::Tensor& p) {
    validate_collapse(r,p); return at::empty({r.size(0),r.size(2)},r.options());
}
at::Tensor run_rrms(const at::Tensor& r) {
    validate_rrms(r); TORCH_CHECK(rrms_registered && r.device().type()==at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(rrms_schema);
    return descriptor.execute({r}).at(0);
}
at::Tensor meta_rrms(const at::Tensor& r) {
    validate_rrms(r); return at::empty({r.size(0),1},r.options().dtype(at::kFloat));
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> run_post_prepare(
    const at::Tensor& x,const at::Tensor& r,const at::Tensor& p,
    const at::Tensor& c,const at::Tensor& n) {
    validate_post_prepare(x,r,p,c,n);
    TORCH_CHECK(post_prepare_registered && x.device().type()==at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(post_prepare_schema);
    auto outputs=descriptor.execute({x,r,p,c,n});
    TORCH_CHECK(outputs.size()==3);
    return {outputs.at(0),outputs.at(1),outputs.at(2)};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> meta_post_prepare(
    const at::Tensor& x,const at::Tensor& r,const at::Tensor& p,
    const at::Tensor& c,const at::Tensor& n) {
    validate_post_prepare(x,r,p,c,n);
    return {at::empty_like(r),at::empty_like(x),at::empty({x.size(0),1},x.options().dtype(at::kFloat))};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_prefill_mhc_post_gaudi2(Tensor value, Tensor residual, Tensor post, Tensor comb) -> Tensor");
    m.def("custom_deepseek_v41_prefill_mhc_collapse_gaudi2(Tensor residual, Tensor previous_pre) -> Tensor");
    m.def("custom_deepseek_v41_prefill_mhc_rrms_bf16_gaudi2(Tensor residual) -> Tensor");
    m.def("custom_deepseek_v41_prefill_mhc_post_prepare_gaudi2(Tensor value, Tensor residual, Tensor post, Tensor comb, Tensor next_pre) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_prefill_mhc_post_gaudi2",run);
    m.impl("custom_deepseek_v41_prefill_mhc_collapse_gaudi2",run_collapse);
    m.impl("custom_deepseek_v41_prefill_mhc_rrms_bf16_gaudi2",run_rrms);
    m.impl("custom_deepseek_v41_prefill_mhc_post_prepare_gaudi2",run_post_prepare);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_prefill_mhc_post_gaudi2",meta);
    m.impl("custom_deepseek_v41_prefill_mhc_collapse_gaudi2",meta_collapse);
    m.impl("custom_deepseek_v41_prefill_mhc_rrms_bf16_gaudi2",meta_rrms);
    m.impl("custom_deepseek_v41_prefill_mhc_post_prepare_gaudi2",meta_post_prepare);
}
