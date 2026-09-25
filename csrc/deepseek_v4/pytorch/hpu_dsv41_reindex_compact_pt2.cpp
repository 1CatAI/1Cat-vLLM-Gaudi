// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_reindex_compact_gaudi2";
habana::PartialOutputMetaDataVector metadata(const at::Stack& s) {
    const auto pool = s.at(0).toTensor(), pos = s.at(1).toTensor();
    const auto ratio = s.at(2).toInt();
    TORCH_CHECK((ratio == 1 || ratio == 2) && pool.scalar_type() == at::kInt && pool.dim() == 2 &&
                pool.size(1) == 2048 && pool.size(0) >= 1 && pool.size(0) <= 64 &&
                pos.scalar_type() == at::kInt && pos.dim() == 1 && pos.size(0) == pool.size(0) &&
                pos.device() == pool.device() && pool.is_contiguous() && pos.is_contiguous(),
                "Reindex compaction requires I32 [B,2048]/[B], B=1..64, ratio=1/2");
    return {{at::kInt, pool.sizes().vec()}, {at::kInt, pool.sizes().vec()}, {at::kInt, pos.sizes().vec()}};
}
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, schema + 11, metadata,
        [](const at::Stack& s, size_t& bytes)->std::shared_ptr<void> {
            bytes = sizeof(int); return std::make_shared<int>(int(s.at(2).toInt()));
        });
    return true;
}();
template<bool Fake>
std::tuple<at::Tensor, at::Tensor, at::Tensor> compact(const at::Tensor& pool, const at::Tensor& pos, int64_t ratio) {
    const auto meta = metadata({pool, pos, ratio});
    if (Fake) return {at::empty(meta[0].shape, pool.options()), at::empty(meta[1].shape, pool.options()),
                      at::empty(meta[2].shape, pool.options())};
    TORCH_CHECK(registered && pool.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto out = op.execute({pool, pos, ratio});
    return {out.at(0), out.at(1), out.at(2)};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_reindex_compact_gaudi2(Tensor candidates, Tensor positions, int ratio) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {m.impl("custom_deepseek_v41_reindex_compact_gaudi2", compact<false>);}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {m.impl("custom_deepseek_v41_reindex_compact_gaudi2", compact<true>);}
