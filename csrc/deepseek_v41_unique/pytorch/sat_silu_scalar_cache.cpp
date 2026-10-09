// SPDX-License-Identifier: Apache-2.0
#define DSV41_N256_BACKEND_ONLY 1
#ifndef DSV41_N256_SILU_GUID
#define DSV41_N256_SILU_GUID "custom_deepseek_v41_silu_scalar_cache_gaudi2"
#endif
#include "../../deepseek_v4/pytorch/hpu_dsv41_expert_n256_pt2.cpp"
#ifndef DSV41_SILU_CACHE_OPERATOR
#define DSV41_SILU_CACHE_OPERATOR custom_deepseek_v41_expert_n256_moe_token_wide_silu_scalar_cache_fp8_gaudi2
#endif
#define DSV41_CACHE_STRINGIFY_INNER(X) #X
#define DSV41_CACHE_STRINGIFY(X) DSV41_CACHE_STRINGIFY_INNER(X)
namespace {
constexpr auto name = "custom_op::" DSV41_CACHE_STRINGIFY(DSV41_SILU_CACHE_OPERATOR);
const bool ready = [] {
    habana::custom_op::registerUserCustomOp(name, kDecode, [](const at::Stack& stack) {
        prequant_contract(stack);
        TORCH_CHECK(stack.at(4).toTensor().size(2)/64==640,
                    "Scalar SiLU cache requires the actual padded640 intermediate");
        TORCH_CHECK(stack.back().toBool() && stack.at(0).toTensor().size(0) >= 2 &&
                    stack.at(0).toTensor().size(0) <= 6, "SAT requires qualified C2-C6 scale planes");
        return habana::PartialOutputMetaDataVector{{at::kBFloat16, moe_shape(stack, true)}};
    }, nullptr);
    habana::KernelRegistry().add(name, [](synDeviceId device, c10::ScalarType type) {
        return std::make_shared<PreparedV41>(device, type, true, true, false, true,
            true, false, true, false, true, false, false, false, true, false, false, true);
    });
    return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& routing,
    const at::Tensor& q13, const at::Tensor& q2, const at::Tensor& s13, const at::Tensor& s2,
    const at::Tensor& lookup, const at::Tensor& c13, const at::Tensor& c2,
    const at::Tensor& quantized, const at::Tensor& sx, bool qualified) {
    const at::Stack stack{x, ids, routing, q13, q2, s13, s2, lookup, c13, c2, quantized, sx, qualified};
    prequant_contract(stack);
    TORCH_CHECK(q2.size(2)/64==640, "Scalar SiLU cache requires padded640");
    TORCH_CHECK(qualified && x.size(0) >= 2 && x.size(0) <= 6);
    if (Meta) return at::empty(moe_shape(stack, true), x.options());
    TORCH_CHECK(ready && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def(DSV41_CACHE_STRINGIFY(DSV41_SILU_CACHE_OPERATOR) "(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, Tensor channel13, Tensor channel2, Tensor quantized, Tensor activation_scale, bool qualified) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl(DSV41_CACHE_STRINGIFY(DSV41_SILU_CACHE_OPERATOR), run<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl(DSV41_CACHE_STRINGIFY(DSV41_SILU_CACHE_OPERATOR), run<true>);
}
