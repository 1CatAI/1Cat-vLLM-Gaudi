// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto order_schema = "custom_op::custom_deepseek_v41_route_order_gaudi2";
constexpr auto pack_schema = "custom_op::custom_deepseek_v41_route_pack_gaudi2";
constexpr auto reduce_schema = "custom_op::custom_deepseek_v41_route_reduce_gaudi2";
habana::PartialOutputMetaDataVector metadata(const at::Stack& s, bool reduce) {
    TORCH_CHECK(s.size() == (reduce ? 2 : 5), "Invalid native route operand count");
    auto x=s.at(0).toTensor(), ids=s.at(reduce ? 1 : 2).toTensor();
    const int64_t r=ids.numel(), h=x.size(-1), b=r/6;
    TORCH_CHECK(r>0 && r<=384 && r%6==0 && h>0 && h<=5120 && h%256==0 &&
                ids.scalar_type()==at::kInt, "Native routing needs B1..64 top6, H256-aligned and I32 IDs");
    for (const auto& item:s) {
        auto t=item.toTensor();
        TORCH_CHECK(t.device()==x.device() && t.is_contiguous() && !t.requires_grad(),
                    "Native routing requires contiguous operands on one device");
    }
    if (reduce) {
        TORCH_CHECK(x.scalar_type()==at::kBFloat16 && x.sizes()==at::IntArrayRef({r,1,h}) && ids.dim()==1,
                    "Route reduction needs already-rounded BF16 route rows and inverse permutation");
        return {{at::kBFloat16,{b,h}}};
    }
    auto sx=s.at(1).toTensor(), route=s.at(3).toTensor();
    TORCH_CHECK(x.scalar_type()==at::kByte && x.sizes()==at::IntArrayRef({b,h}) &&
                sx.scalar_type()==at::kFloat && sx.sizes()==at::IntArrayRef({b,1}) &&
                ids.sizes()==at::IntArrayRef({1,r}) && route.scalar_type()==at::kFloat &&
                route.sizes()==at::IntArrayRef({1,r}), "Route pack geometry/dtype differs");
    TORCH_CHECK(s.at(4).toTensor().scalar_type()==at::kInt && s.at(4).toTensor().sizes()==at::IntArrayRef({r}),
                "Route inverse geometry differs");
    return {{at::kByte,{r,1,h}},{at::kFloat,{r,1,1}},{at::kInt,{1,r}},{at::kFloat,{r,1}}};
}
habana::PartialOutputMetaDataVector order_meta(const at::Stack& s) {
    auto ids=s.at(0).toTensor();const auto r=ids.numel();
    TORCH_CHECK(ids.scalar_type()==at::kInt && ids.dim()==2 && ids.size(0)==1 && r>0 && r<=384 && r%6==0 &&
                ids.is_contiguous() && !ids.requires_grad(), "Native route order needs contiguous I32 top6 IDs");
    return {{at::kInt,{r}}};
}
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(order_schema,order_schema+11,order_meta,nullptr);
    for (bool reduce:{false,true}) {
        const char* schema=reduce ? reduce_schema : pack_schema;
        habana::custom_op::registerUserCustomOp(schema,schema+11,
            [reduce](const at::Stack& s){return metadata(s,reduce);},nullptr);
    }
    return true;
}();
template<bool Fake> std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>
pack(const at::Tensor& x,const at::Tensor& sx,const at::Tensor& ids,const at::Tensor& route,const at::Tensor& inverse) {
    at::Stack s{x,sx,ids,route,inverse}; const auto meta=metadata(s,false);
    std::vector<at::Tensor> output;
    if (Fake) {for (const auto& m:meta) output.push_back(at::empty(m.shape,x.options().dtype(m.dtype)));}
    else {
        TORCH_CHECK(registered && x.device().type()==at::kHPU);
        auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(pack_schema);
        output=op.execute(s);
    }
    return {output[0],output[1],output[2],output[3]};
}
template<bool Fake> at::Tensor order(const at::Tensor& ids) {
    const auto meta=order_meta({ids});
    if (Fake) return at::empty(meta[0].shape,ids.options());
    TORCH_CHECK(registered && ids.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(order_schema);
    return op.execute({ids}).at(0);
}
template<bool Fake> at::Tensor reduce(const at::Tensor& rows,const at::Tensor& inverse) {
    at::Stack s{rows,inverse};const auto meta=metadata(s,true);
    if (Fake) return at::empty(meta[0].shape,rows.options().dtype(at::kBFloat16));
    TORCH_CHECK(registered && rows.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(reduce_schema);
    return op.execute(s).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_route_order_gaudi2(Tensor ids) -> Tensor");
    m.def("custom_deepseek_v41_route_pack_gaudi2(Tensor x, Tensor sx, Tensor ids, Tensor route, Tensor inverse) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("custom_deepseek_v41_route_reduce_gaudi2(Tensor rows, Tensor inverse) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_route_order_gaudi2",order<false>);
    m.impl("custom_deepseek_v41_route_pack_gaudi2",pack<false>);
    m.impl("custom_deepseek_v41_route_reduce_gaudi2",reduce<false>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_route_order_gaudi2",order<true>);
    m.impl("custom_deepseek_v41_route_pack_gaudi2",pack<true>);
    m.impl("custom_deepseek_v41_route_reduce_gaudi2",reduce<true>);
}
