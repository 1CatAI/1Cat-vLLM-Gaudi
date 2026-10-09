// SPDX-License-Identifier: Apache-2.0
// Weight-only source E4M3FN decoding. Activation and GEMM dtype stay BF16.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
#ifndef DSV41_DENSE_BITS12
#define DSV41_DENSE_BITS12 0
#endif
namespace {
#if DSV41_DENSE_BITS12
constexpr auto kWeight = "custom_op::custom_deepseek_v41_dense_bits12_bf16_gaudi2";
constexpr auto kDense = "custom_op::custom_deepseek_v41_dense_bits12_projection_bf16_gaudi2";
constexpr auto kGuid = "custom_deepseek_v41_dense_bits12_bf16_gaudi2";
#else
constexpr auto kWeight = "custom_op::custom_deepseek_v41_source_weight_bf16_gaudi2";
constexpr auto kDense = "custom_op::custom_deepseek_v41_source_dense_bf16_gaudi2";
constexpr auto kGuid = "custom_deepseek_v41_source_weight_bf16_gaudi2";
#endif
habana::OutputMetaDataVector metadata(const at::Stack& s, bool dense) {
    TORCH_CHECK(s.size() == (dense ? 3u : 2u), "Source weight decoding requires raw bytes and UE8M0 scales");
    const auto w = s.at(dense ? 1 : 0).toTensor(), scale = s.at(dense ? 2 : 1).toTensor();
    TORCH_CHECK(w.scalar_type() == at::kChar && w.dim() == 2 && w.size(0) >= 32 && w.size(0) <= 32768 &&
#if DSV41_DENSE_BITS12
        w.size(1) >= 256 && w.size(1) <= 16384 && w.size(0) % 32 == 0 && w.size(1) % 256 == 0 &&
        scale.scalar_type() == at::kChar && scale.sizes() == at::IntArrayRef({w.size(0), w.size(1)/2}),
        "Packed BF16 weights require I8 high-byte [N,K] and I8 low-nibble [N,K/2] planes");
#else
        w.size(1) >= 128 && w.size(1) <= 16384 && w.size(0) % 32 == 0 && w.size(1) % 128 == 0 &&
        scale.scalar_type() == at::kShort && scale.sizes() == at::IntArrayRef({w.size(0)/32, w.size(1)/32}),
        "Source weight decoding requires I8 [N,K] and I16 [N/32,K/32] source scales");
#endif
    for (const auto& item : s) {
        const auto t = item.toTensor();
        TORCH_CHECK(t.is_contiguous() && !t.requires_grad() && t.device() == w.device(),
            "Source decoder requires contiguous inference operands on one device");
    }
    if (!dense) return {{at::kBFloat16, w.sizes().vec()}};
    const auto x = s.at(0).toTensor();
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == w.size(1) &&
        x.size(0) >= 1 && x.size(0) <= 16384, "Source dense projection retains the BF16 activation contract");
    return {{at::kBFloat16, {x.size(0), w.size(0)}}};
}
class SourceDense final : public habana::OpBackend {
    bool dense_;
 public:
    SourceDense(int device, c10::ScalarType dtype, bool dense)
        : OpBackend(device, NO_TPC + std::string("dsv41_source_dense"), dtype, {0}, {}, {}, false), dense_(dense) {
        SetOutputMetaFn([dense](const at::Stack& s) { return metadata(s, dense); });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto m = metadata(s, dense_);
        const auto w = s.at(dense_ ? 1 : 0).toTensor();
        std::vector<synapse_helpers::tensor> decoded;
        if (dense_) decoded = BuildNode(this, graph, {kGuid, {syn_in(1), syn_in(2)},
            {{w.sizes().vec(), at::kBFloat16}}});
        else decoded = BuildNode(this, graph, {kGuid, {syn_in(0), syn_in(1)},
            {{w.sizes().vec(), at::kBFloat16, 0}}});
        if (!dense_) { syn_out(0) = std::move(decoded[0]); return; }
        // The decoder output is private to this BF16 MME consumer. The
        // compiler can slice/bundle the producer instead of persisting a
        // whole dequantized matrix. Placement still requires qualification.
        synGEMMParams params{false, true};
        auto product = BuildNode(this, graph, {"gemm", {syn_in(0), decoded[0].get()},
            {{m[0].shape, at::kBFloat16, 0}}, &params, sizeof(params)});
        syn_out(0) = std::move(product[0]);
    }
};
const bool ready = [] {
    for (const bool dense : {false, true}) {
        const auto name = dense ? kDense : kWeight;
        habana::custom_op::registerUserCustomOp(name, kGuid, [dense](const at::Stack& s) {
            const auto m = metadata(s, dense)[0];
            return habana::PartialOutputMetaDataVector{{m.dtype, m.shape}};
        }, nullptr);
        habana::KernelRegistry().add(name, [dense](synDeviceId d, c10::ScalarType type) {
            return std::make_shared<SourceDense>(d, type, dense);
        });
    }
    return true;
}();
template<bool Meta> at::Tensor weight_run(const at::Tensor& w, const at::Tensor& scale) {
    const at::Stack s{w, scale}; const auto m = metadata(s, false)[0];
    if (Meta) return at::empty(m.shape, w.options().dtype(m.dtype));
    TORCH_CHECK(ready && w.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kWeight);
    return op.execute(s)[0];
}
template<bool Meta> at::Tensor dense_run(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scale) {
    const at::Stack s{x, w, scale}; const auto m = metadata(s, true)[0];
    if (Meta) return at::empty(m.shape, x.options().dtype(m.dtype));
    TORCH_CHECK(ready && x.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kDense);
    return op.execute(s)[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
#if DSV41_DENSE_BITS12
    m.def("custom_deepseek_v41_dense_bits12_bf16_gaudi2(Tensor high, Tensor low) -> Tensor");
    m.def("custom_deepseek_v41_dense_bits12_projection_bf16_gaudi2(Tensor x, Tensor high, Tensor low) -> Tensor");
#else
    m.def("custom_deepseek_v41_source_weight_bf16_gaudi2(Tensor weight, Tensor scale) -> Tensor");
    m.def("custom_deepseek_v41_source_dense_bf16_gaudi2(Tensor x, Tensor weight, Tensor scale) -> Tensor");
#endif
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
#if DSV41_DENSE_BITS12
    m.impl("custom_deepseek_v41_dense_bits12_bf16_gaudi2", weight_run<false>);
    m.impl("custom_deepseek_v41_dense_bits12_projection_bf16_gaudi2", dense_run<false>);
#else
    m.impl("custom_deepseek_v41_source_weight_bf16_gaudi2", weight_run<false>);
    m.impl("custom_deepseek_v41_source_dense_bf16_gaudi2", dense_run<false>);
#endif
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
#if DSV41_DENSE_BITS12
    m.impl("custom_deepseek_v41_dense_bits12_bf16_gaudi2", weight_run<true>);
    m.impl("custom_deepseek_v41_dense_bits12_projection_bf16_gaudi2", dense_run<true>);
#else
    m.impl("custom_deepseek_v41_source_weight_bf16_gaudi2", weight_run<true>);
    m.impl("custom_deepseek_v41_source_dense_bf16_gaudi2", dense_run<true>);
#endif
}
