// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <initializer_list>
#include <cmath>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kSchema =
    "custom_op::custom_deepseek_v41_kv_norm_rope_publish_gaudi2";
constexpr auto kOrdered = "custom_op::custom_deepseek_v41_kv_norm_rope_publish_ordered_gaudi2";
constexpr auto kGuid = "custom_deepseek_v41_kv_norm_rope_publish_gaudi2";
struct Params { float epsilon; float inverse_width; int offset; };

std::vector<int64_t> shape(const at::Tensor& input,
                           const at::Tensor& weight,
                           const at::Tensor& positions,
                           const at::Tensor& phase,
                           double epsilon) {
    const auto device = input.device();
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && input.dim() == 2 &&
                input.size(0) >= 1 && input.size(0) <= 512 &&
                input.size(1) == 512 && input.is_contiguous() &&
                !input.requires_grad(), "KV norm/RoPE requires BF16 [1..512,512]");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16 &&
                weight.sizes() == at::IntArrayRef({512}) &&
                weight.device() == device && weight.is_contiguous() &&
                !weight.requires_grad(), "KV norm/RoPE requires BF16 weight [512]");
    TORCH_CHECK(positions.scalar_type() == at::kInt && positions.dim() == 1 &&
                positions.size(0) == input.size(0) && positions.device() == device &&
                positions.is_contiguous() && !positions.requires_grad(),
                "KV norm/RoPE requires one I32 position per row");
    TORCH_CHECK(phase.scalar_type() == at::kFloat && phase.dim() == 2 &&
                phase.size(0) >= 1 && phase.size(0) <= 1048576 &&
                phase.size(1) == 64 && phase.device() == device &&
                phase.is_contiguous() && !phase.requires_grad(),
                "KV norm/RoPE requires F32 phase [1..1048576,64]");
    TORCH_CHECK(epsilon > 0 && std::isnormal(static_cast<float>(epsilon)),
                "KV norm/RoPE requires a positive normal epsilon");
    return input.sizes().vec();
}

habana::PartialOutputMetaDataVector metadata(const at::Stack& stack) {
    const auto sizes = shape(stack.at(0).toTensor(),stack.at(1).toTensor(),stack.at(2).toTensor(),
                             stack.at(3).toTensor(),stack.at(6).toDouble());
    const auto x=stack.at(0).toTensor(), cache=stack.at(4).toTensor(), decoded=stack.at(5).toTensor();
    const auto offset=stack.at(7).toInt();
    TORCH_CHECK(x.size(0)==1 && cache.scalar_type()==at::kByte && cache.dim()==2 && cache.size(1)==528 &&
                cache.size(0)>=256 && decoded.scalar_type()==at::kBFloat16 && decoded.dim()==2 &&
                decoded.size(1)==512 && decoded.size(0)>=512 && (offset==-1 || (offset>=0 && offset%512==0 && offset<=decoded.size(0)-512)) && cache.device()==x.device() && decoded.device()==x.device() &&
                cache.is_contiguous() && decoded.is_contiguous() && !cache.requires_grad() && !decoded.requires_grad(),
                "KV publication requires scheduler-owned packed and decoded circular state");
    return {{at::kBFloat16,sizes},{at::kInt,{16}}};
}

const bool registered = [] {
    for (const auto* name : {kSchema, kOrdered}) {
    habana::custom_op::registerUserCustomOp(
        name, kGuid,
        [](const at::Stack& stack) { return metadata(stack); },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(Params);
            return std::make_shared<Params>(Params{
                static_cast<float>(stack.at(6).toDouble()), 1.0f / 512.0f, static_cast<int>(stack.at(7).toInt())});
        });
    }
    return true;
}();

template<bool Meta, bool Ordered = false>
std::tuple<at::Tensor,at::Tensor> run(const at::Tensor& input,const at::Tensor& weight,
    const at::Tensor& positions,const at::Tensor& phase,const at::Tensor& cache,
    const at::Tensor& decoded,double epsilon,int64_t offset) {
    const at::Stack stack{input,weight,positions,phase,cache,decoded,epsilon,offset};
    const auto meta=metadata(stack);
    if (Meta) return {at::empty(meta.at(0).shape,input.options()),
                      at::empty({16},input.options().dtype(at::kInt))};
    TORCH_CHECK(registered && input.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Ordered ? kOrdered : kSchema);
    const auto result=descriptor.execute(stack);
    return {result.at(0),result.at(1)};
}

at::Tensor unwrap(const at::Tensor& tensor) {
    if (!at::functionalization::impl::isFunctionalTensor(tensor)) return tensor;
    at::functionalization::impl::sync(tensor);
    return at::functionalization::impl::from_functional_tensor(tensor);
}
std::tuple<at::Tensor,at::Tensor> functionalize(const at::Tensor& input,const at::Tensor& weight,
    const at::Tensor& positions,const at::Tensor& phase,const at::Tensor& cache,
    const at::Tensor& decoded,double epsilon,int64_t offset) {
    auto x=unwrap(input),w=unwrap(weight),p=unwrap(positions),t=unwrap(phase),c=unwrap(cache),d=unwrap(decoded);
    static auto handle=c10::Dispatcher::singleton().findSchemaOrThrow(kOrdered, "")
        .typed<std::tuple<at::Tensor,at::Tensor>(const at::Tensor&,const at::Tensor&,const at::Tensor&,
             const at::Tensor&,const at::Tensor&,const at::Tensor&,double,int64_t)>();
    std::tuple<at::Tensor,at::Tensor> result;
    { at::AutoDispatchSkipFunctionalize guard; result=handle.call(x,w,p,t,c,d,epsilon,offset); }
    for (const auto& pair : {std::make_pair(cache,c),std::make_pair(decoded,d)}) {
        at::functionalization::impl::replace_(pair.first,pair.second);
        at::functionalization::impl::commit_update(pair.first);
        at::functionalization::impl::sync(pair.first);
    }
    return result;
}

}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_kv_norm_rope_publish_ordered_gaudi2(Tensor input, Tensor weight, Tensor positions, Tensor phase, Tensor cache, Tensor decoded, float epsilon, int offset) -> (Tensor, Tensor)");
    m.def("custom_deepseek_v41_kv_norm_rope_publish_gaudi2(Tensor input, Tensor weight, Tensor positions, Tensor phase, Tensor(a!) cache, Tensor(b!) decoded, float epsilon, int offset) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_kv_norm_rope_publish_ordered_gaudi2", run<false,true>);
    m.impl("custom_deepseek_v41_kv_norm_rope_publish_gaudi2", run<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_kv_norm_rope_publish_ordered_gaudi2", run<true,true>);
    m.impl("custom_deepseek_v41_kv_norm_rope_publish_gaudi2", run<true>);
}

TORCH_LIBRARY_IMPL(custom_op, Functionalize, m) {
    m.impl("custom_deepseek_v41_kv_norm_rope_publish_gaudi2", functionalize);
}
