// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_bounded_local_nucleus_gaudi2";
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    const auto values=s[0].toTensor(),ids=s[1].toTensor();
    TORCH_CHECK(values.dim()==2 && values.size(0)>=1 && values.size(0)<=6 && values.size(1)==64 &&
                values.scalar_type()==at::kFloat && ids.sizes()==values.sizes() && ids.scalar_type()==at::kInt &&
                s[6].toInt()>0 && s[6].toInt()<=131072 && s[7].toInt()>=0,"Bounded nucleus requires C1-C6 K64");
    for(int i=0;i<6;++i) {
        const auto t=s[i].toTensor();
        const int64_t columns=i<2?64:i==4?4:1;
        TORCH_CHECK(t.sizes()==at::IntArrayRef({values.size(0),columns}) &&
                    t.scalar_type()==(i==1?at::kInt:at::kFloat) && t.device()==values.device() &&
                    t.is_contiguous() && !t.requires_grad(),"Bounded nucleus input geometry or dtype mismatch");
    }
    return {{at::kInt,{values.size(0),2}},{at::kFloat,{values.size(0),s[6].toInt()}}};
}
class Nucleus final:public habana::OpBackend {
 public:
    Nucleus(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("bounded_local_nucleus"),t,{0,1},{},{},false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto m=metadata(s);const int64_t rows=s[0].toTensor().size(0);
        auto fields=BuildNode(this,graph,{"custom_deepseek_v41_bounded_nucleus_parts_gaudi2",
            {syn_in(0),syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(5)},
            {{{rows,64},at::kFloat},{m[0].shape,m[0].dtype,0}}});
        int32_t params[]={static_cast<int32_t>(s[6].toInt()),static_cast<int32_t>(s[7].toInt())};
        auto probability=BuildNode(this,graph,{"custom_deepseek_v41_bounded_local_distribution_gaudi2",
            {fields[0].get(),syn_in(1)},{{m[1].shape,m[1].dtype,1}},params,sizeof(params)});
        syn_out(0)=std::move(fields[1]);syn_out(1)=std::move(probability[0]);
    }
};
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(schema,"custom_deepseek_v41_bounded_nucleus_parts_gaudi2",
        [](const at::Stack& s) {
            habana::PartialOutputMetaDataVector out;
            for(const auto& m:metadata(s))out.push_back({m.dtype,m.shape});return out;
        },nullptr);
    habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Nucleus>(d,t);});
    return true;
}();
template<bool Meta> std::tuple<at::Tensor,at::Tensor> execute(const at::Tensor& values,const at::Tensor& ids,
    const at::Tensor& maximum,const at::Tensor& total,const at::Tensor& controls,const at::Tensor& omitted,
    int64_t columns,int64_t offset) {
    const at::Stack s{values,ids,maximum,total,controls,omitted,columns,offset};const auto m=metadata(s);
    if(Meta)return {at::empty(m[0].shape,values.options().dtype(at::kInt)),at::empty(m[1].shape,values.options())};
    TORCH_CHECK(ready && values.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);auto result=op.execute(s);
    return {result[0],result[1]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
 m.def("custom_deepseek_v41_bounded_local_nucleus_gaudi2(Tensor values, Tensor ids, Tensor maximum, Tensor total, Tensor controls, Tensor omitted, int columns, int offset) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("custom_deepseek_v41_bounded_local_nucleus_gaudi2",execute<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("custom_deepseek_v41_bounded_local_nucleus_gaudi2",execute<true>);}
