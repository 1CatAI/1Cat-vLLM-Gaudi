// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_journal_copy_gaudi2";
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    const auto x=s[0].toTensor(),positions=s[1].toTensor(),pages=s[2].toTensor();
    const int64_t mode=s[3].toInt(),parameter=s[4].toInt();
    TORCH_CHECK(x.dim()>=1 && x.size(0)>=6 && x.is_contiguous() && !x.requires_grad() &&
                (x.element_size()==1 || x.element_size()==2 || x.element_size()==4) &&
                positions.sizes()==at::IntArrayRef({6}) && positions.scalar_type()==at::kInt &&
                pages.dim()==1 && pages.numel()>0 && pages.scalar_type()==at::kInt &&
                positions.device()==x.device() && pages.device()==x.device() &&
                positions.is_contiguous() && pages.is_contiguous() && mode>=0 && mode<=3 &&
                (mode==3 || (parameter>0 && (parameter&(parameter-1))==0)) &&
                (mode!=1 || parameter<=128),"Journal requires a power-of-two C6 write coordinate and byte-copy tensor");
    auto shape=x.sizes().vec();shape[0]=6;
    return {{x.scalar_type(),shape},{at::kInt,{6}}};
}
class Copy final:public habana::OpBackend {
 public:
    Copy(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("journal_copy"),t,{0,1},{},{},false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto m=metadata(s);const auto x=s[0].toTensor();
        const int64_t columns=x.numel()/x.size(0);int shift=0;
        if(s[3].toInt()!=3)while((int64_t(1)<<shift)<s[4].toInt())++shift;
        int32_t params[]={static_cast<int32_t>(s[3].toInt()),shift,static_cast<int32_t>(x.size(0)),
                          static_cast<int32_t>(columns),static_cast<int32_t>(x.element_size())};
        auto source=ReshapeHelper(graph,syn_in(0),{x.size(0),columns},x.scalar_type());
        auto outputs=BuildNode(this,graph,{"custom_deepseek_v41_journal_copy_gaudi2",
            {source.get(),syn_in(1),syn_in(2)},{{{6,columns},x.scalar_type()},{m[1].shape,m[1].dtype,1}},
            params,sizeof(params)});
        // This reshape is the public output, not an internal view. Eager
        // recipe/shape caches require its final-result binding even when
        // whole-graph functionalization happens to keep the tensor alive.
        syn_out(0)=ReshapeHelper(graph,outputs[0].get(),m[0].shape,x.scalar_type(),0);
        syn_out(1)=std::move(outputs[1]);
    }
};
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(schema,schema+11,[](const at::Stack& s) {
        habana::PartialOutputMetaDataVector out;
        for(const auto& m:metadata(s))out.push_back({m.dtype,m.shape});return out;
    },nullptr);
    habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Copy>(d,t);});
    return true;
}();
template<bool Meta> std::tuple<at::Tensor,at::Tensor> execute(const at::Tensor& x,const at::Tensor& positions,
                                                            const at::Tensor& pages,int64_t mode,int64_t parameter) {
    const at::Stack s{x,positions,pages,mode,parameter};const auto m=metadata(s);
    if(Meta)return {at::empty(m[0].shape,x.options()),at::empty(m[1].shape,x.options().dtype(at::kInt))};
    TORCH_CHECK(ready && x.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);auto result=op.execute(s);
    return {result[0],result[1]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
 m.def("custom_deepseek_v41_journal_copy_gaudi2(Tensor target, Tensor positions, Tensor pages, int mode, int parameter) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("custom_deepseek_v41_journal_copy_gaudi2",execute<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("custom_deepseek_v41_journal_copy_gaudi2",execute<true>);}
