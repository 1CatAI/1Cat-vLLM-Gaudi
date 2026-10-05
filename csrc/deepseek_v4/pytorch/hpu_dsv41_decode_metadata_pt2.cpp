// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_decode_metadata_gaudi2";
habana::PartialOutputMetaDataVector metadata(const at::Stack& s) {
    const auto& p = s.at(0).toTensor();
    const auto& ids = s.at(1).toTensor();
    const auto& pages = s.at(2).toTensor();
    TORCH_CHECK(p.scalar_type() == at::kInt && ids.scalar_type() == at::kInt && pages.scalar_type() == at::kInt &&
                p.dim() == 1 && p.numel() > 0 && p.numel() <= 6 && ids.sizes() == p.sizes() &&
                pages.dim() == 1 && pages.numel() > 0 && p.device() == ids.device() && p.device() == pages.device() &&
                p.is_contiguous() && ids.is_contiguous() && pages.is_contiguous(),
                "Decode coordinates require contiguous matching I32 token rows and page table");
    return {{at::kInt, {p.numel(), 192}}, {at::kInt, {p.numel()}},
            {at::kInt, {p.numel()}}, {at::kInt, {p.numel()}}};
}
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, schema + 11, metadata, nullptr);
    return true;
}();
using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
template<bool Meta> Outputs execute(const at::Tensor& p, const at::Tensor& ids, const at::Tensor& pages) {
    const auto out = metadata({p, ids, pages});
    if (Meta) return {at::empty({p.numel(), 192}, p.options()), at::empty_like(p),
                      at::empty_like(p), at::empty_like(p)};
    TORCH_CHECK(registered && p.device().type() == at::kHPU, "Decode coordinates require HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto result = descriptor.execute({p, ids, pages});
    return {result[0], result[1], result[2], result[3]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_decode_metadata_gaudi2(Tensor positions, Tensor ids, Tensor pages) -> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) { m.impl("custom_deepseek_v41_decode_metadata_gaudi2", execute<false>); }
TORCH_LIBRARY_IMPL(custom_op, Meta, m) { m.impl("custom_deepseek_v41_decode_metadata_gaudi2", execute<true>); }
