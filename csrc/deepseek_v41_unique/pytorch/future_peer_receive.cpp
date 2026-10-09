// SPDX-License-Identifier: Apache-2.0
// Additive, disabled capability. Retain all three outputs in the replay owner:
// the persistent signal packet cannot be discarded after graph compilation.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <dlfcn.h>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_future_peer_sum_gaudi2";
constexpr auto receiver = "custom_deepseek_v41_bounded_peer_receive_gaudi2";
using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    TORCH_CHECK(stack.size() == 8);
    const auto local = stack[0].toTensor(), peers = stack[1].toTensor();
    const auto flags = stack[2].toTensor(), epoch = stack[3].toTensor();
    TORCH_CHECK(local.dim() == 2 && local.scalar_type() == at::kBFloat16 &&
                local.size(0) >= 1 && local.size(0) <= 6 && local.size(1) == 5120,
                "Future peer sum requires a bounded hidden-width packet");
    TORCH_CHECK(peers.dim() == 5 && peers.scalar_type() == at::kBFloat16 &&
                peers.size(0) == 2 && peers.size(1) >= 128 && peers.size(1) <= 2048 &&
                (peers.size(1) & (peers.size(1)-1)) == 0 &&
                (peers.size(2) == 2 || peers.size(2) == 4) &&
                peers.size(3) == local.size(0) && peers.size(4) == local.size(1),
                "Future peer table retains parity, point, rank, row and feature dimensions");
    TORCH_CHECK(peers.numel()*peers.element_size() > (int64_t(48) << 20),
                "Future-written peer payload must stay in HBM before NIC completion");
    TORCH_CHECK(flags.scalar_type() == at::kInt && flags.sizes() == at::IntArrayRef({1 << 24}) &&
                epoch.scalar_type() == at::kInt &&
                (epoch.sizes() == at::IntArrayRef({1}) || epoch.sizes() == flags.sizes()));
    const int64_t point = stack[4].toInt(), rank = stack[5].toInt(), spins = stack[6].toInt();
    TORCH_CHECK(point >= 0 && point < 128 && rank >= 0 && rank < peers.size(2) && spins >= 1 && spins <= 65536);
    for (const auto& value : {local, peers, flags, epoch}) {
        TORCH_CHECK(value.device() == local.device() && value.is_contiguous() && !value.requires_grad());
    }
    return {{at::kBFloat16, local.sizes().vec()},
            {at::kBFloat16, local.sizes().vec()}, {at::kInt, {local.size(0), 40}}};
}

class FutureSum final : public habana::OpBackend {
public:
    FutureSum(int device, c10::ScalarType type)
        : OpBackend(device, NO_TPC + std::string("dsv41_future_peer_sum"), type, {0, 1, 2}, {}, {}, false) {
        SetOutputMetaFn(metadata);
    }
    void CustomHandler(synapse_helpers::graph&, at::Stack& stack) override {
        if (!stack[7].toBool()) return;
        using SetExternal = synStatus (*)(synTensor, bool);
        const auto set = reinterpret_cast<SetExternal>(dlsym(RTLD_DEFAULT, "synTensorSetExternal"));
        TORCH_CHECK(set && GetSynOutputs().size() == 3 && IsOutputPersistent(1),
                    "Future packet must remain a persistent graph output before capture");
        TORCH_CHECK(set(GetSynOutputs()[1].ref().get(), true) == synSuccess,
                    "Persistent future packet signal rejected");
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto out = metadata(stack);
        const int64_t elements = stack[0].toTensor().numel();
        auto flat = BuildReshape(this, graph, syn_in(0), {1, elements}, at::kBFloat16);
        auto copied = BuildNode(this, graph, {"custom_deepseek_v41_bf16_identity_gaudi2", {flat.get()},
            {{{1, elements}, at::kBFloat16}}});
        auto packet = BuildReshape(this, graph, copied[0].get(), out[1].shape, at::kBFloat16, 1);
        int params[] = {int(stack[4].toInt()), int(stack[5].toInt()), int(stack[6].toInt())};
        auto sum = BuildNode(this, graph, {receiver, {packet.get(), syn_in(1), syn_in(2), syn_in(3)},
            {{out[0].shape, out[0].dtype, 0}, {out[2].shape, out[2].dtype, 2}},
            params, sizeof(params)});
        syn_out(1) = std::move(packet);
        syn_out(0) = std::move(sum[0]);
        syn_out(2) = std::move(sum[1]);
    }
};

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, receiver, [](const at::Stack& stack) {
        const auto out = metadata(stack);
        return habana::PartialOutputMetaDataVector{{out[0].dtype, out[0].shape},
                                                   {out[1].dtype, out[1].shape}, {out[2].dtype, out[2].shape}};
    }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType type) {
        return std::make_shared<FutureSum>(device, type);
    });
    return true;
}();

template<bool Meta> Outputs run(const at::Tensor& local, const at::Tensor& peers,
    const at::Tensor& flags, const at::Tensor& epoch, int64_t point, int64_t rank, int64_t spins, bool external) {
    const at::Stack stack{local, peers, flags, epoch, point, rank, spins, external};
    const auto out = metadata(stack);
    if constexpr (Meta) return {at::empty(out[0].shape, local.options()), at::empty(out[1].shape, local.options()),
                                at::empty(out[2].shape, flags.options())};
    TORCH_CHECK(registered && local.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto values = descriptor.execute(stack);
    return {values[0], values[1], values[2]};
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, module) {
    module.def("custom_deepseek_v41_future_peer_sum_gaudi2(Tensor local, Tensor peers, Tensor flags, Tensor epoch, "
               "int point, int rank, int spins, bool external) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, module) {module.impl("custom_deepseek_v41_future_peer_sum_gaudi2", run<false>);}
TORCH_LIBRARY_IMPL(custom_op, Meta, module) {module.impl("custom_deepseek_v41_future_peer_sum_gaudi2", run<true>);}
