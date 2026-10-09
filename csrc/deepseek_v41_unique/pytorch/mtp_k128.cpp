// SPDX-License-Identifier: Apache-2.0
// Only the entry contract changes. Reuse the unchanged prepared BF16 body;
// ordinary C1 schemas and decoder/communication implementations are retained.
#define DSV41_N256_BACKEND_ONLY 1
#include "../../deepseek_v4/pytorch/hpu_dsv41_expert_n256_pt2.cpp"

namespace {
constexpr auto name = "custom_op::custom_deepseek_v41_mtp_moe_k128_bf16_gaudi2";
void draft_contract(const at::Stack& stack) {
    const auto result = moe_shape(stack);
    const auto& x = stack.at(0).toTensor();
    TORCH_CHECK(stack.size() == 9 && result[0] >= 1 && result[0] <= 6 &&
                stack.at(1).toTensor().size(1) == 3 &&
                stack.at(3).toTensor().size(0) == 128 && stack.back().toBool(),
                "Draft K128 requires C1-C6 top3/E128 with qualified normal scales");
    TORCH_CHECK(x.size(1) == 5120 && stack.at(4).toTensor().size(2) / 32 == 640,
                "Draft K128 requires the checkpoint H5120/I640 layout");
}
const bool ready = [] {
    habana::custom_op::registerUserCustomOp(name, kDecodeK128Normal, [](const at::Stack& stack) {
        draft_contract(stack);
        return habana::PartialOutputMetaDataVector{{at::kBFloat16, moe_shape(stack)}};
    }, nullptr);
    habana::KernelRegistry().add(name, [](synDeviceId device, c10::ScalarType type) {
        return std::make_shared<PreparedV41>(device, type, true, false, true);
    });
    return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& router,
    const at::Tensor& q13, const at::Tensor& q2, const at::Tensor& s13, const at::Tensor& s2,
    const at::Tensor& lookup, bool qualified) {
    const at::Stack stack{x, ids, router, q13, q2, s13, s2, lookup, qualified};
    draft_contract(stack);
    if constexpr(Meta) return at::empty(moe_shape(stack), x.options());
    TORCH_CHECK(ready && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_mtp_moe_k128_bf16_gaudi2(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, bool qualified) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) { m.impl("custom_deepseek_v41_mtp_moe_k128_bf16_gaudi2", run<false>); }
TORCH_LIBRARY_IMPL(custom_op, Meta, m) { m.impl("custom_deepseek_v41_mtp_moe_k128_bf16_gaudi2", run<true>); }
