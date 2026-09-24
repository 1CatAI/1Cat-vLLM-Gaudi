// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_candidate_gather_f32_gaudi2";
habana::PartialOutputMetaDataVector metadata(const at::Stack& s) {
    const auto common = s.at(0).toTensor(), blocks = s.at(1).toTensor(), positions = s.at(2).toTensor();
    const auto ratio = s.at(3).toInt();
    TORCH_CHECK((ratio == 1 || ratio == 2) && common.scalar_type() == at::kFloat && common.dim() == 2 &&
                common.size(0) >= 1 && common.size(0) <= 128 && common.size(1) >= 8 &&
                common.size(1) <= 32768 && common.size(1) % 8 == 0 &&
                blocks.scalar_type() == at::kInt && blocks.dim() == 2 && blocks.size(0) == common.size(0) &&
                blocks.size(1) >= 1 && blocks.size(1) <= 2048 &&
                positions.scalar_type() == at::kInt && positions.dim() == 1 && positions.size(0) == common.size(0),
                "Candidate gather requires bounded F32 scores, I32 blocks/positions and ratio 1 or 2");
    for (unsigned i = 0; i < 3; ++i) {
        const auto t = s.at(i).toTensor();
        TORCH_CHECK(t.device() == common.device() && t.is_contiguous() && !t.requires_grad(),
                    "Candidate gather inputs must be contiguous on one device");
    }
    const std::vector<int64_t> shape = {common.size(0), blocks.size(1) * 8};
    return {{at::kFloat, shape}, {at::kInt, shape}};
}
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, schema + 11, metadata,
        [](const at::Stack& s, size_t& bytes)->std::shared_ptr<void> {
            bytes = sizeof(int); return std::make_shared<int>(int(s.at(3).toInt()));
        });
    return true;
}();
template<bool Fake> std::tuple<at::Tensor, at::Tensor> gather(
        const at::Tensor& common, const at::Tensor& blocks, const at::Tensor& positions, int64_t ratio) {
    const auto meta = metadata({common, blocks, positions, ratio});
    if (Fake) return {at::empty(meta[0].shape, common.options()),
                      at::empty(meta[1].shape, blocks.options())};
    TORCH_CHECK(registered && common.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    const auto results = op.execute({common, blocks, positions, ratio});
    return {results.at(0), results.at(1)};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_candidate_gather_f32_gaudi2(Tensor common, Tensor blocks, "
          "Tensor positions, int ratio) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {m.impl("custom_deepseek_v41_candidate_gather_f32_gaudi2", gather<false>);}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {m.impl("custom_deepseek_v41_candidate_gather_f32_gaudi2", gather<true>);}
