// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "../host/deepseek_v41_batch_mla_metadata_gaudi2.hpp"

namespace {
constexpr auto kSchema = "custom_op::custom_deepseek_v41_batch_mla_metadata_gaudi2";
habana::PartialOutputMetaDataVector metadata(const at::Stack& s) {
    const auto selected = s.at(0).toTensor(), pages = s.at(1).toTensor();
    const int64_t ratio = s.at(6).toInt(), swa_rows = s.at(7).toInt(), tile = s.at(8).toInt();
    TORCH_CHECK(selected.dim() == 2 && selected.size(1) == 512 &&
                selected.size(0) >= 1 && selected.size(0) <= 64 &&
                pages.dim() == 2 && pages.size(0) == selected.size(0) && pages.size(1) > 0 &&
                pages.size(1) <= 8192 && ratio >= 0 && ratio <= 2 && swa_rows >= 256 &&
                swa_rows % 256 == 0 && swa_rows <= 64 * 256 && (tile == 4 || tile == 8 || tile == 16),
                "Batch MLA metadata requires bounded request rows, pages, ratio and SWA slots");
    for (unsigned i = 0; i < 6; ++i) {
        const auto t = s.at(i).toTensor();
        TORCH_CHECK(t.scalar_type() == at::kInt && t.is_contiguous() &&
                    t.device() == selected.device() && !t.requires_grad() &&
                    (i < 2 || (t.dim() == 1 && t.numel() == selected.size(0))),
                    "Batch MLA metadata needs contiguous I32 tensors with one owner per row");
    }
    const int64_t width = ratio ? 640 : 128;
    return {{at::kInt, {selected.size(0), width}}, {at::kInt, {selected.size(0), width}},
            {at::kInt, {selected.size(0)}}};
}
class BatchMlaMetadata : public habana::OpBackend {
public:
    BatchMlaMetadata(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_batch_mla_metadata"), dtype, {0, 1, 2}, {}, {}, false) {
        SetOutputMetaFn([](const at::Stack& s) {
            habana::OutputMetaDataVector result;
            for (const auto& m : metadata(s)) result.push_back({m.dtype, m.shape});
            return result;
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto m = metadata(s);
        Dsv41BatchMlaMetadataParams params{int(s.at(6).toInt()), int(s.at(7).toInt()), int(s.at(8).toInt())};
        auto outputs = BuildNode(this, graph, {DeepseekV41BatchMlaMetadataGaudi2::name,
            {syn_in(0), syn_in(1), syn_in(2), syn_in(3), syn_in(4), syn_in(5)},
            {{m[0].shape, at::kInt, 0}, {m[1].shape, at::kInt, 1}, {m[2].shape, at::kInt, 2}},
            &params, sizeof(params)});
        for (unsigned i = 0; i < 3; ++i) syn_out(i) = std::move(outputs.at(i));
    }
};
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(kSchema, DeepseekV41BatchMlaMetadataGaudi2::name, metadata, nullptr);
    habana::KernelRegistry().add(kSchema, [](synDeviceId d, c10::ScalarType t) {
        return std::make_shared<BatchMlaMetadata>(d, t);
    });
    return true;
}();
template<bool Fake>
std::tuple<at::Tensor, at::Tensor, at::Tensor> execute(
        const at::Tensor& selected, const at::Tensor& pages, const at::Tensor& positions,
        const at::Tensor& slots, const at::Tensor& swa_done, const at::Tensor& main_done,
        int64_t ratio, int64_t swa_rows, int64_t tile) {
    const at::Stack s{selected, pages, positions, slots, swa_done, main_done, ratio, swa_rows, tile};
    const auto m = metadata(s);
    if (Fake) return {at::empty(m[0].shape, selected.options()), at::empty(m[1].shape, selected.options()),
                      at::empty(m[2].shape, selected.options())};
    TORCH_CHECK(registered && selected.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);
    auto result = descriptor.execute(s);
    return {result.at(0), result.at(1), result.at(2)};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_batch_mla_metadata_gaudi2(Tensor selected, Tensor pages, Tensor positions, "
          "Tensor slots, Tensor swa_done, Tensor main_done, int ratio, int swa_rows, int tile) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_batch_mla_metadata_gaudi2", execute<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_batch_mla_metadata_gaudi2", execute<true>);
}
