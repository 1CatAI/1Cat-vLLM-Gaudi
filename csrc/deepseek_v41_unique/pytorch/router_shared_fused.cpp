// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
using Pair = std::tuple<at::Tensor, at::Tensor>;
const char* schemas[] = {"custom_op::custom_deepseek_v41_router_shared_scaled_gaudi2",
                        "custom_op::custom_deepseek_v41_shared_silu_full_product_gaudi2"};
habana::OutputMetaDataVector meta(const at::Stack& stack, bool shared) {
    const auto x = stack.at(0).toTensor();
    const auto rows = x.size(0);
    TORCH_CHECK(x.scalar_type() == at::kFloat && x.is_contiguous() && rows >= 2 && rows <= 6);
    for (const auto& value : stack) {
        const auto tensor = value.toTensor();
        TORCH_CHECK(tensor.is_contiguous() && tensor.device() == x.device() && !tensor.requires_grad());
    }
    if (shared) {
        const auto ids = stack.at(1).toTensor(), sx = stack.at(2).toTensor();
        const auto channel = stack.at(3).toTensor(), routing = stack.at(4).toTensor();
        TORCH_CHECK(x.dim() == 3 && x.size(1) == 1 && (x.size(2) == 1792 || x.size(2) == 3072) &&
            ids.scalar_type() == at::kInt && ids.sizes() == at::IntArrayRef({1, rows}) &&
            sx.scalar_type() == at::kFloat && sx.sizes() == at::IntArrayRef({rows, 1}) &&
            channel.scalar_type() == at::kBFloat16 && channel.dim() == 3 && channel.size(0) == 1 &&
            channel.size(1) * 256 + 512 == x.size(2) && channel.size(2) == 256 &&
            routing.scalar_type() == at::kFloat && routing.sizes() == ids.sizes(),
            "Shared epilogue requires C2-C6 full joint product and original channel scales");
        return {{at::ScalarType::Float8_e4m3fn, {rows, 1, (x.size(2) - 512) / 2}},
                {at::kFloat, {rows, 1, 1}}};
    }
    const auto text = stack.at(1).toTensor(), image = stack.at(2).toTensor(), mask = stack.at(3).toTensor();
    const auto channel = stack.at(4).toTensor(), sx = stack.at(5).toTensor();
    TORCH_CHECK(x.dim() == 2 && (x.size(1) == 1792 || x.size(1) == 3072) &&
        text.scalar_type() == at::kFloat && text.sizes() == at::IntArrayRef({384}) &&
        image.scalar_type() == at::kFloat && image.sizes() == text.sizes() &&
        mask.scalar_type() == at::kBool && mask.sizes() == at::IntArrayRef({rows}) &&
        channel.scalar_type() == at::kFloat && channel.sizes() == at::IntArrayRef({1, 384}) &&
        sx.scalar_type() == at::kFloat && sx.sizes() == at::IntArrayRef({rows, 1}),
        "Scaled Router requires a joint C2-C6 FP32 product and original biases/scales");
    return {{at::kInt, {rows, 6}}, {at::kFloat, {rows, 6}}};
}
const bool registered = [] {
    for (int i = 0; i < 2; ++i)
        habana::custom_op::registerUserCustomOp(schemas[i], schemas[i] + 11,
            [i](const at::Stack& stack) {
                const auto output = meta(stack, i == 1);
                return habana::PartialOutputMetaDataVector{{output[0].dtype, output[0].shape},
                                                          {output[1].dtype, output[1].shape}};
            }, nullptr);
    return true;
}();
template<bool Meta> Pair execute(const at::Stack& stack, bool shared) {
    const auto output = meta(stack, shared);
    const auto x = stack.at(0).toTensor();
    if constexpr (Meta) return {at::empty(output[0].shape, x.options().dtype(output[0].dtype)),
                                at::empty(output[1].shape, x.options().dtype(output[1].dtype))};
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schemas[shared ? 1 : 0]);
    auto result = op.execute(stack);
    return {result[0], result[1]};
}
template<bool Meta> Pair router(const at::Tensor& x, const at::Tensor& text, const at::Tensor& image,
    const at::Tensor& mask, const at::Tensor& channel, const at::Tensor& sx) {
    return execute<Meta>({x, text, image, mask, channel, sx}, false);
}
template<bool Meta> Pair shared(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& sx,
    const at::Tensor& channel, const at::Tensor& routing) {
    return execute<Meta>({x, ids, sx, channel, routing}, true);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_router_shared_scaled_gaudi2(Tensor product, Tensor text, Tensor image, Tensor mask, Tensor channel, Tensor sx) -> (Tensor, Tensor)");
    m.def("custom_deepseek_v41_shared_silu_full_product_gaudi2(Tensor product, Tensor ids, Tensor sx, Tensor channel, Tensor routing) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_router_shared_scaled_gaudi2", router<false>);
    m.impl("custom_deepseek_v41_shared_silu_full_product_gaudi2", shared<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_router_shared_scaled_gaudi2", router<true>);
    m.impl("custom_deepseek_v41_shared_silu_full_product_gaudi2", shared<true>);
}
