// SPDX-License-Identifier: Apache-2.0
// V4.1 direct indexed BF16 MAC candidate.  It consumes Q16/S16 in the
// N-major prepared layout and writes only the selected expert activations.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kFc1 = "custom_deepseek_v41_mxfp4_indexed_fc1_bf16_gaudi2";
constexpr auto kFc1Normal = "custom_deepseek_v41_mxfp4_indexed_fc1_normal_bf16_gaudi2";
constexpr auto kFc2 = "custom_deepseek_v41_mxfp4_indexed_fc2_bf16_gaudi2";
constexpr auto kFc2Normal = "custom_deepseek_v41_mxfp4_indexed_fc2_normal_bf16_gaudi2";
constexpr auto kFc1Schema = "custom_op::custom_deepseek_v41_mxfp4_indexed_fc1_bf16_gaudi2";
constexpr auto kFc2Schema = "custom_op::custom_deepseek_v41_mxfp4_indexed_fc2_bf16_gaudi2";

void contract(const at::Tensor& t, at::ScalarType dtype, const at::Device& device) {
    TORCH_CHECK(t.scalar_type() == dtype && t.device() == device && t.is_contiguous() &&
                !t.requires_grad(), "V4.1 indexed MXFP4 requires contiguous inference tensors");
}

std::vector<int64_t> output_shape(const at::Stack& stack, bool fc1) {
    const auto& x = stack.at(0).toTensor();
    const auto& ids = stack.at(1).toTensor();
    const auto& q = stack.at(2).toTensor();
    const auto& s = stack.at(3).toTensor();
    const auto& lookup = stack.at(4).toTensor();
    contract(x, at::kBFloat16, x.device());
    contract(ids, at::kInt, x.device());
    contract(q, at::kShort, x.device());
    contract(s, at::kBFloat16, x.device());
    contract(lookup, at::kBFloat16, x.device());
    TORCH_CHECK(ids.dim() == 2 && ids.size(1) == 6 && ids.size(0) > 0 && ids.size(0) <= 6,
                "V4.1 indexed MXFP4 expects [tokens,6] ordered expert IDs");
    const int64_t tokens = ids.size(0);
    const int64_t blocks = fc1 ? 18 : 40;
    const int64_t stream = fc1 ? 163840 : 36864;
    const int64_t width = fc1 ? 1152 : 5120;
    TORCH_CHECK(q.sizes() == at::IntArrayRef({384, blocks, stream}) &&
                s.sizes() == at::IntArrayRef({384, blocks, stream / 8}) &&
                lookup.sizes() == at::IntArrayRef({128}),
                "V4.1 indexed MXFP4 Q16/S16 geometry mismatch");
    if (fc1) {
        TORCH_CHECK(x.sizes() == at::IntArrayRef({tokens, 5120}),
                    "V4.1 indexed FC1 expects [tokens,5120]");
        return {tokens, 6, 2 * width};
    }
    TORCH_CHECK(x.sizes() == at::IntArrayRef({tokens, 6, 1152}),
                "V4.1 indexed FC2 expects [tokens,6,1152]");
    return {tokens, 6, width};
}

class IndexedV41 final : public habana::OpBackend {
    bool fc1_;
 public:
    IndexedV41(int device, c10::ScalarType dtype, bool fc1)
        : OpBackend(device, NO_TPC + std::string("dsv41_indexed_bf16"), dtype, {0}, {}, {}, false),
          fc1_(fc1) {
        SetOutputMetaFn([fc1](const at::Stack& stack) {
            return habana::OutputMetaDataVector{{at::kBFloat16, output_shape(stack, fc1)}};
        });
    }

    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const bool normal = stack.back().toBool();
        const char* guid = fc1_ ? (normal ? kFc1Normal : kFc1)
                                : (normal ? kFc2Normal : kFc2);
        const auto shape = output_shape(stack, fc1_);
        auto result = BuildNode(this, graph, {guid,
            {syn_in(0), syn_in(1), syn_in(2), syn_in(3), syn_in(4)},
            {{shape, at::kBFloat16, 0}}});
        syn_out(0) = std::move(result.at(0));
    }
};

const bool registered = [] {
    for (bool fc1 : {true, false}) {
        const char* schema = fc1 ? kFc1Schema : kFc2Schema;
        habana::custom_op::registerUserCustomOp(schema, fc1 ? kFc1 : kFc2,
            [fc1](const at::Stack& stack) {
                return habana::PartialOutputMetaDataVector{{at::kBFloat16, output_shape(stack, fc1)}};
            }, nullptr);
        habana::KernelRegistry().add(schema, [fc1](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<IndexedV41>(device, dtype, fc1);
        });
    }
    return true;
}();

at::Tensor execute(const at::Stack& stack, bool fc1, bool meta) {
    const auto shape = output_shape(stack, fc1);
    if (meta) return at::empty(shape, stack.at(0).toTensor().options().dtype(at::kBFloat16));
    TORCH_CHECK(registered && stack.at(0).toTensor().device().type() == at::kHPU,
                "V4.1 indexed MXFP4 requires HPU");
    const char* schema = fc1 ? kFc1Schema : kFc2Schema;
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute(stack).at(0);
}

at::Tensor fc1(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& q,
               const at::Tensor& s, const at::Tensor& lookup, bool normal) {
    return execute({x, ids, q, s, lookup, normal}, true, false);
}
at::Tensor fc2(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& q,
               const at::Tensor& s, const at::Tensor& lookup, bool normal) {
    return execute({x, ids, q, s, lookup, normal}, false, false);
}
at::Tensor fc1_meta(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& q,
                    const at::Tensor& s, const at::Tensor& lookup, bool normal) {
    return execute({x, ids, q, s, lookup, normal}, true, true);
}
at::Tensor fc2_meta(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& q,
                    const at::Tensor& s, const at::Tensor& lookup, bool normal) {
    return execute({x, ids, q, s, lookup, normal}, false, true);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_mxfp4_indexed_fc1_bf16_gaudi2(Tensor x, Tensor ids, Tensor q16, Tensor s16, Tensor lookup, bool normal_scales=False) -> Tensor");
    m.def("custom_deepseek_v41_mxfp4_indexed_fc2_bf16_gaudi2(Tensor x, Tensor ids, Tensor q16, Tensor s16, Tensor lookup, bool normal_scales=False) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_mxfp4_indexed_fc1_bf16_gaudi2", fc1);
    m.impl("custom_deepseek_v41_mxfp4_indexed_fc2_bf16_gaudi2", fc2);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_mxfp4_indexed_fc1_bf16_gaudi2", fc1_meta);
    m.impl("custom_deepseek_v41_mxfp4_indexed_fc2_bf16_gaudi2", fc2_meta);
}
