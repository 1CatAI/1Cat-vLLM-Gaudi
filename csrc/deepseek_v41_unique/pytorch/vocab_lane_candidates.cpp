// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_vocab_lane_candidates_gaudi2";
constexpr auto guid="custom_deepseek_v41_vocab_lane_candidates_gaudi2";
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    const auto scores=s[0].toTensor();
    TORCH_CHECK(scores.dim()==2 && scores.size(0)>=8 && scores.size(0)<=192 && scores.size(0)%8==0 &&
                scores.size(1)>=128 && scores.size(1)<=4096 && scores.size(1)%64==0 &&
                scores.scalar_type()==at::kFloat && scores.is_contiguous() && !scores.requires_grad(),
                "Certified lane selection requires contiguous F32 vocabulary planes");
    return {{at::kFloat,{scores.size(0),128}},{at::kInt,{scores.size(0),128}},{at::kFloat,{scores.size(0),1}}};
}
class Filter final:public habana::OpBackend {
 public:
    Filter(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("vocab_lane_candidates"),t,{0,1,2},{},{},false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto out=metadata(s);int columns=static_cast<int>(s[0].toTensor().size(1));
        auto result=BuildNode(this,graph,{guid,{syn_in(0)},
            {{out[0].shape,out[0].dtype,0},{out[1].shape,out[1].dtype,1},{out[2].shape,out[2].dtype,2}},
            &columns,sizeof(columns)});
        for(int i=0;i<3;++i)syn_out(i)=std::move(result[i]);
    }
};
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(schema,guid,[](const at::Stack& s) {
        const auto m=metadata(s);habana::PartialOutputMetaDataVector out;
        for(const auto& v:m)out.push_back({v.dtype,v.shape});return out;
    },nullptr);
    habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Filter>(d,t);});return true;
}();
template<bool Meta> std::tuple<at::Tensor,at::Tensor,at::Tensor> execute(const at::Tensor& scores) {
    const at::Stack s{scores};const auto m=metadata(s);
    if(Meta)return {at::empty(m[0].shape,scores.options()),at::empty(m[1].shape,scores.options().dtype(at::kInt)),
                    at::empty(m[2].shape,scores.options())};
    TORCH_CHECK(registered && scores.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);auto result=op.execute(s);
    return {result[0],result[1],result[2]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_vocab_lane_candidates_gaudi2(Tensor scores) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("custom_deepseek_v41_vocab_lane_candidates_gaudi2",execute<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("custom_deepseek_v41_vocab_lane_candidates_gaudi2",execute<true>);}
