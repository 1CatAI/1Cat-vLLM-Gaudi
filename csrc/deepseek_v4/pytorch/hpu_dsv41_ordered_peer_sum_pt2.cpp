// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_ordered_peer_sum_gaudi2";
habana::PartialOutputMetaDataVector metadata(const at::Stack& s) {
    const auto& shards = s.at(0).toTensor();
    TORCH_CHECK(shards.scalar_type() == at::kBFloat16 && shards.dim() == 2 && shards.is_contiguous() &&
                shards.size(0) >= 2 && shards.size(0) <= 8 && shards.size(1) > 0 &&
                shards.size(1) <= 32768 && shards.size(1) % 128 == 0,
                "Ordered peer sum requires contiguous BF16 [ranks, aligned width]");
    return {{at::kBFloat16, {1, shards.size(1)}}};
}
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, schema + 11, metadata, nullptr);
    return true;
}();
template<bool Meta> at::Tensor execute(const at::Tensor& shards) {
    metadata({shards});
    if (Meta) return at::empty({1, shards.size(1)}, shards.options());
    TORCH_CHECK(registered && shards.device().type() == at::kHPU, "Ordered peer sum requires HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute({shards}).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_ordered_peer_sum_gaudi2(Tensor shards) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) { m.impl("custom_deepseek_v41_ordered_peer_sum_gaudi2", execute<false>); }
TORCH_LIBRARY_IMPL(custom_op, Meta, m) { m.impl("custom_deepseek_v41_ordered_peer_sum_gaudi2", execute<true>); }
