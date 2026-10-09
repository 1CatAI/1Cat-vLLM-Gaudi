// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_dspark_moe_peer_post_gaudi2";
constexpr auto guid = "custom_deepseek_v41_peer_post_collapse_gaudi2";
using Outputs = std::tuple<at::Tensor, at::Tensor>;

habana::OutputMetaDataVector meta(const at::Stack& s) {
    TORCH_CHECK(s.size() == 5, "MoE peer/post needs all peer, residual and gate operands");
    const auto peers = s[0].toTensor(), residual = s[1].toTensor();
    TORCH_CHECK(peers.dim() == 3 && (peers.size(0) == 2 || peers.size(0) == 4) &&
                peers.size(1) >= 2 && peers.size(1) <= 6 && peers.size(2) == 5120 &&
                residual.sizes() == at::IntArrayRef({peers.size(1), 4, 5120}),
                "MoE peer/post requires TP-parametric C2-C6 residual and peers");
    for (int i = 0; i < 5; ++i) {
        const auto t = s[i].toTensor();
        TORCH_CHECK(t.scalar_type() == (i < 2 ? at::kBFloat16 : at::kFloat) &&
                    t.device() == peers.device() && t.is_contiguous() && !t.requires_grad(),
                    "MoE peer/post operands must be matching contiguous inference tensors");
    }
    TORCH_CHECK(s[2].toTensor().sizes() == at::IntArrayRef({peers.size(1), 4}) &&
                s[3].toTensor().sizes() == at::IntArrayRef({peers.size(1), 4, 4}) &&
                s[4].toTensor().sizes() == at::IntArrayRef({peers.size(1), 4}),
                "MoE peer/post gate geometry changed");
    return {{at::kBFloat16, residual.sizes().vec()}, {at::kBFloat16, {peers.size(1), 5120}}};
}

class Fused final : public habana::OpBackend {
 public:
    Fused(int device, c10::ScalarType type) : OpBackend(device, guid, type, {0, 1}, {}, {}, false) {
        SetOutputMetaFn(meta);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto out = meta(s);
        auto values = BuildNode(this, graph, {guid, {syn_in(0), syn_in(1), syn_in(2), syn_in(3), syn_in(4)},
            {{out[0].shape, out[0].dtype, 0}, {out[1].shape, out[1].dtype, 1}}});
        syn_out(0) = std::move(values[0]);
        syn_out(1) = std::move(values[1]);
    }
};

const bool ready = [] {
    habana::custom_op::registerUserCustomOp(schema, guid, [](const at::Stack& s) {
        habana::PartialOutputMetaDataVector result;
        for (const auto& out : meta(s)) result.push_back({out.dtype, out.shape});
        return result;
    }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType type) {
        return std::make_shared<Fused>(device, type);
    });
    return true;
}();

template<bool Meta> Outputs run(const at::Tensor& peers, const at::Tensor& residual,
    const at::Tensor& post, const at::Tensor& comb, const at::Tensor& pre) {
    const at::Stack s{peers, residual, post, comb, pre};
    const auto out = meta(s);
    if constexpr (Meta) return {at::empty(out[0].shape, peers.options()), at::empty(out[1].shape, peers.options())};
    TORCH_CHECK(ready && peers.device().type() == at::kHPU, "MoE peer/post requires HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    const auto values = descriptor.execute(s);
    return {values[0], values[1]};
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_dspark_moe_peer_post_gaudi2(Tensor peers, Tensor residual, Tensor post, Tensor comb, Tensor pre) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_dspark_moe_peer_post_gaudi2", run<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_dspark_moe_peer_post_gaudi2", run<true>);
}
