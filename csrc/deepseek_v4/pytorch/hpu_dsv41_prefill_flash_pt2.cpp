// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_prefill_flash_inputs_gaudi2";
constexpr auto kv_guid = "custom_deepseek_v41_prefill_flash_kv_gaudi2";
constexpr auto mask_guid = "custom_deepseek_v41_prefill_flash_mask_gaudi2";

habana::OutputMetaDataVector metadata(const at::Stack& s) {
    const auto cache = s.at(0).toTensor(), ids = s.at(1).toTensor();
    const auto lengths = s.at(2).toTensor(), sink = s.at(3).toTensor();
    TORCH_CHECK(cache.scalar_type() == at::kBFloat16 && cache.dim() == 2 &&
                cache.size(0) >= 1 && cache.size(0) <= INT32_MAX && cache.size(1) == 512 &&
                ids.scalar_type() == at::kInt && ids.dim() == 2 && ids.size(0) >= 1 && ids.size(0) <= 512 &&
                ids.size(1) >= 1 && ids.size(1) <= 640 && lengths.scalar_type() == at::kInt &&
                lengths.sizes() == at::IntArrayRef({ids.size(0)}) && sink.scalar_type() == at::kFloat &&
                sink.dim() == 1 && sink.size(0) >= 1 && sink.size(0) <= 64,
                "Flash MLA preparation requires BF16 [N,512], I32 [T1..512,K1..640]/[T] and F32 [H1..64]");
    for (const auto& t : {cache, ids, lengths, sink})
        TORCH_CHECK(t.is_contiguous() && t.device() == cache.device() && !t.requires_grad(),
                    "Flash MLA preparation requires contiguous inference tensors on one device");
    return {{at::kBFloat16, {ids.size(0), ids.size(1) + 1, 512}},
            {at::kFloat, {ids.size(0), sink.size(0), ids.size(1) + 1}}};
}

class FlashInputs final : public habana::OpBackend {
public:
    FlashInputs(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_prefill_flash_inputs"), dtype, {0, 1}, {}, {}, false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto output = metadata(s);
        int rows = s.at(0).toTensor().size(0);
        auto kv = BuildNode(this, graph, {kv_guid, {syn_in(0), syn_in(1), syn_in(2)},
                                         {{output[0].shape, at::kBFloat16, 0}}});
        auto mask = BuildNode(this, graph, {mask_guid, {syn_in(1), syn_in(2), syn_in(3)},
                                           {{output[1].shape, at::kFloat, 1}}, &rows, sizeof(rows)});
        syn_out(0) = std::move(kv[0]);
        syn_out(1) = std::move(mask[0]);
    }
};

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, kv_guid, [](const at::Stack& s) {
        const auto output = metadata(s);
        return habana::PartialOutputMetaDataVector{{output[0].dtype, output[0].shape},
                                                   {output[1].dtype, output[1].shape}};
    }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<FlashInputs>(device, dtype);
    });
    return true;
}();

template<bool Fake> std::tuple<at::Tensor, at::Tensor> run(
    const at::Tensor& cache, const at::Tensor& ids, const at::Tensor& lengths, const at::Tensor& sink) {
    const auto meta = metadata({cache, ids, lengths, sink});
    if (Fake) return {at::empty(meta[0].shape, cache.options()), at::empty(meta[1].shape, sink.options())};
    TORCH_CHECK(registered && cache.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    const auto outputs = op.execute({cache, ids, lengths, sink});
    return {outputs.at(0), outputs.at(1)};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_prefill_flash_inputs_gaudi2(Tensor cache, Tensor indices, Tensor lengths, Tensor sink) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_prefill_flash_inputs_gaudi2", run<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_prefill_flash_inputs_gaudi2", run<true>);
}
