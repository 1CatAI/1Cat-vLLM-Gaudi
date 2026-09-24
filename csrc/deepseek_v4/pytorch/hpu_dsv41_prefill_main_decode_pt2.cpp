// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_prefill_main_decode_gaudi2";
struct Params { int ratio; };
habana::PartialOutputMetaDataVector metadata(const at::Stack& s) {
    TORCH_CHECK(s.size() == 4, "Paged main decode needs cache, pages, rows, ratio");
    const auto cache = s[0].toTensor(), pages = s[1].toTensor(), rows = s[2].toTensor();
    const auto ratio = s[3].toInt();
    TORCH_CHECK((ratio == 0 || ratio == 1 || ratio == 2) && cache.scalar_type() == at::kByte &&
                cache.dim() == 2 && cache.size(1) == 288 && cache.size(0) > 0 &&
                cache.size(0) <= 1048704 && pages.scalar_type() == at::kInt &&
                pages.dim() == 1 && pages.numel() > 0 && pages.numel() <= 8193 &&
                rows.scalar_type() == at::kInt && rows.dim() == 1 &&
                rows.numel() > 0 && rows.numel() <= 65536 &&
                (ratio != 0 || cache.size(0) >= rows.numel()),
                "Invalid paged main decoder geometry or dtype");
    for (unsigned i = 0; i < 3; ++i) {
        const auto t = s[i].toTensor();
        TORCH_CHECK(t.device() == cache.device() && t.is_contiguous() && !t.requires_grad(),
                    "Paged main decoder needs contiguous inference HPU tensors");
    }
    return {{at::kBFloat16, {rows.numel(), 512}}};
}
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, schema + 11, metadata,
        [](const at::Stack& s, size_t& bytes) -> std::shared_ptr<void> {
            bytes = sizeof(Params);
            return std::make_shared<Params>(Params{int(s[3].toInt())});
        });
    return true;
}();
template<bool Fake>
at::Tensor decode(const at::Tensor& cache, const at::Tensor& pages,
                  const at::Tensor& rows, int64_t ratio) {
    const auto meta = metadata({cache, pages, rows, ratio});
    if constexpr (Fake) return at::empty(meta[0].shape, cache.options().dtype(at::kBFloat16));
    TORCH_CHECK(registered && cache.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute({cache, pages, rows, ratio})[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_prefill_main_decode_gaudi2(Tensor cache, Tensor pages, Tensor rows, int ratio) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_prefill_main_decode_gaudi2", decode<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_prefill_main_decode_gaudi2", decode<true>);
}
