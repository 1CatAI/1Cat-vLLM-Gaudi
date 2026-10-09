// SPDX-License-Identifier: Apache-2.0
// Isolated capability only. Serving never selects this registration.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <dlfcn.h>
#include "hpu_ops/op_backend.h"
#include "backend/helpers/create_tensor.h"
#include "habana_eager/eager_tensor.h"
#include "backend/habana_device/HPUDevice.h"
#include "synapse_common_types.h"

namespace {
constexpr auto name = "custom_op::custom_deepseek_v41_peer_signal_probe_gaudi2";

habana::OutputMetaDataVector meta(const at::Stack& stack) {
    TORCH_CHECK(stack.size() == 5, "Signal probe expects four matrices and its capability flag");
    const auto x = stack[0].toTensor(), w = stack[1].toTensor();
    const auto control = stack[2].toTensor(), cw = stack[3].toTensor();
    for (const auto& value : {x, w, control, cw}) {
        TORCH_CHECK(value.dim() == 2 && value.scalar_type() == at::kBFloat16 && value.is_contiguous() &&
                    value.device() == x.device() && !value.requires_grad(), "Signal probe uses BF16 GEMM operands");
    }
    TORCH_CHECK(x.size(0) >= 2 && x.size(0) <= 6 && x.size(1) == w.size(1) &&
                control.size(0) == 2*x.size(0) && control.size(1) == cw.size(1),
                "Signal probe preserves the C2-C6 peer projection and hi/lo control rows");
    return {{at::kBFloat16, {x.size(0), w.size(0)}}, {at::kFloat, {control.size(0), cw.size(0)}}};
}

class Probe final : public habana::OpBackend {
public:
    Probe(int device, c10::ScalarType type)
        : OpBackend(device, NO_TPC+std::string("dsv41_peer_signal_probe"), type, {0, 1}, {}, {}, false) {
        SetOutputMetaFn(meta);
    }

    void CustomHandler(synapse_helpers::graph&, at::Stack& stack) override {
        if (!stack[4].toBool()) return;
        using SetExternal = synStatus (*)(synTensor, bool);
        const auto api = reinterpret_cast<SetExternal>(dlsym(RTLD_DEFAULT, "synTensorSetExternal"));
        TORCH_CHECK(api && !GetSynOutputs().empty(), "Signal output must be allocated before node insertion");
        const auto status = api(GetSynOutputs()[0].ref().get(), true);
        TORCH_CHECK(status == synSuccess, "Persistent output signal rejected before node insertion: ", int(status));
    }

    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto outputs = meta(stack);
        synGEMMParams params{false, true};
        syn_out(0) = std::move(BuildNode(this, graph, {"gemm", {syn_in(0), syn_in(1)},
            {{outputs[0].shape, outputs[0].dtype, 0}}, &params, sizeof(params)})[0]);
        syn_out(1) = std::move(BuildNode(this, graph, {"gemm", {syn_in(2), syn_in(3)},
            {{outputs[1].shape, outputs[1].dtype, 1}}, &params, sizeof(params)})[0]);
    }
};

const bool ready = [] {
    habana::custom_op::registerUserCustomOp(name, "gemm", [](const at::Stack& stack) {
        const auto outputs = meta(stack);
        return habana::PartialOutputMetaDataVector{{outputs[0].dtype, outputs[0].shape},
                                                   {outputs[1].dtype, outputs[1].shape}};
    }, nullptr);
    habana::KernelRegistry().add(name, [](synDeviceId device, c10::ScalarType type) {
        return std::make_shared<Probe>(device, type);
    });
    return true;
}();

template<bool Meta>
std::tuple<at::Tensor, at::Tensor> run(const at::Tensor& x, const at::Tensor& w,
                                     const at::Tensor& control, const at::Tensor& cw, bool external) {
    const at::Stack stack{x, w, control, cw, external};
    const auto outputs = meta(stack);
    if (Meta) return {at::empty(outputs[0].shape, x.options()),
                     at::empty(outputs[1].shape, x.options().dtype(at::kFloat))};
    TORCH_CHECK(ready && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    auto values = descriptor.execute(stack);
    return {values[0], values[1]};
}

// A byte-exact, explicit payload producer. The existing identity glue owns
// its usual vector-wide copy; the output signal has a separate lifetime from
// the final completion of the surrounding independent-control recipe.
constexpr auto markerName = "custom_op::custom_deepseek_v41_peer_ready_identity_gaudi2";
habana::OutputMetaDataVector markerMeta(const at::Stack& stack) {
    const auto x = stack.at(0).toTensor();
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 6 &&
                x.size(1) > 0 && x.numel() <= 32768 && x.size(1) % 128 == 0 &&
                x.is_contiguous() && !x.requires_grad(), "Tensor-ready marker uses the small BF16 peer packet");
    return {{at::kBFloat16, x.sizes().vec()}};
}

class Marker final : public habana::OpBackend {
public:
    Marker(int device, c10::ScalarType type)
        : OpBackend(device, NO_TPC+std::string("dsv41_peer_ready_identity"), type, {0}, {}, {}, false) {
        SetOutputMetaFn(markerMeta);
    }
    void CustomHandler(synapse_helpers::graph&, at::Stack& stack) override {
        if (!stack.at(1).toBool()) return;
        TORCH_CHECK(IsOutputPersistent(0), "Only the final persistent peer payload can expose a tensor-ready signal");
        using SetExternal = synStatus (*)(synTensor, bool);
        const auto api = reinterpret_cast<SetExternal>(dlsym(RTLD_DEFAULT, "synTensorSetExternal"));
        TORCH_CHECK(api && !GetSynOutputs().empty() &&
                    api(GetSynOutputs()[0].ref().get(), true) == synSuccess,
                    "Tensor-ready packet must be external before node insertion");
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto shape = markerMeta(stack)[0].shape;
        const auto elements = stack[0].toTensor().numel();
        auto flat = BuildReshape(this, graph, syn_in(0), {1, elements}, at::kBFloat16);
        auto copied = BuildNode(this, graph, {
            "custom_deepseek_v41_bf16_identity_gaudi2", {flat.get()},
            {{{1, elements}, at::kBFloat16}}});
        syn_out(0) = BuildReshape(this, graph, copied[0].get(), shape, at::kBFloat16, 0);
    }
};
const bool markerReady = [] {
    habana::custom_op::registerUserCustomOp(markerName, "custom_deepseek_v41_bf16_identity_gaudi2",
        [](const at::Stack& stack) {
            return habana::PartialOutputMetaDataVector{{at::kBFloat16, markerMeta(stack)[0].shape}};
        }, nullptr);
    habana::KernelRegistry().add(markerName, [](synDeviceId device, c10::ScalarType type) {
        return std::make_shared<Marker>(device, type);
    });
    return true;
}();
template<bool Meta>
at::Tensor mark(const at::Tensor& x, bool external) {
    const at::Stack stack{x, external};
    const auto shape = markerMeta(stack)[0].shape;
    if (Meta) return at::empty(shape, x.options());
    TORCH_CHECK(markerReady && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(markerName);
    return descriptor.execute(stack)[0];
}

std::vector<int64_t> storageLayout(const at::Tensor& value) {
    TORCH_CHECK(value.device().type() == at::kHPU, "Storage-layout diagnostics require an actual HPU tensor");
    const auto [permutation, allowed] = habana_helpers::get_tensor_memory_permutation(value);
    const auto backend = habana::eager::HbEagerTensorPool::get_backend_tensor(value);
    std::vector<int64_t> result{
        int64_t(reinterpret_cast<uintptr_t>(value.storage().data_ptr().get())),
        int64_t(reinterpret_cast<uintptr_t>(backend.storage().data_ptr().get())),
        value.storage_offset(), int64_t(allowed),
        int64_t(reinterpret_cast<uintptr_t>(habana::HPUDeviceContext::get_device().get_fixed_address(
            backend.storage().data_ptr().get())))};
    result.insert(result.end(), permutation.begin(), permutation.end());
    return result;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, module) {
    module.def("custom_deepseek_v41_peer_signal_probe_gaudi2(Tensor x, Tensor w, Tensor control, Tensor cw, "
               "bool external) -> (Tensor, Tensor)");
    module.def("custom_deepseek_v41_peer_ready_identity_gaudi2(Tensor x, bool external) -> Tensor");
    module.def("deepseek_v41_peer_storage_layout(Tensor x) -> int[]");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, module) {
    module.impl("custom_deepseek_v41_peer_signal_probe_gaudi2", run<false>);
    module.impl("custom_deepseek_v41_peer_ready_identity_gaudi2", mark<false>);
    module.impl("deepseek_v41_peer_storage_layout", storageLayout);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, module) {
    module.impl("custom_deepseek_v41_peer_signal_probe_gaudi2", run<true>);
    module.impl("custom_deepseek_v41_peer_ready_identity_gaudi2", mark<true>);
}
