// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr const char* names[]={"custom_deepseek_v41_weighted_score_bounds_gaudi2",
                              "custom_deepseek_v41_weighted_sparse_bins_gaudi2",
                              "custom_deepseek_v41_weighted_active_blocks_gaudi2"};
std::string schema(int kind){return std::string("custom_op::")+names[kind];}
habana::OutputMetaDataVector metadata(const at::Stack& s,bool bins) {
    const auto x=s[0].toTensor(),w=s[1].toTensor();
    TORCH_CHECK(x.dim()==2 && x.size(0)>=1 && x.size(0)<=6 && x.size(1)>=64 && x.size(1)<=131072 &&
                x.size(1)%64==0 && x.scalar_type()==at::kFloat && w.scalar_type()==at::kFloat &&
                w.sizes()==x.sizes(),"Weighted bounds require positive-mass F32 C1-C6 scores");
    for(unsigned i=0;i<(bins?4:2);++i) {
        const auto t=s[i].toTensor();
        TORCH_CHECK(t.device()==x.device() && t.is_contiguous() && !t.requires_grad());
    }
    if(bins) {
        TORCH_CHECK(s[2].toTensor().scalar_type()==at::kInt &&
                    s[2].toTensor().sizes()==at::IntArrayRef({x.size(0),1}) &&
                    s[3].toTensor().scalar_type()==at::kInt &&
                    s[3].toTensor().sizes()==at::IntArrayRef({x.size(0),(x.size(1)+2047)/2048}),"Invalid exact block bitmap");
        const auto shift=s[4].toInt();TORCH_CHECK(shift>=0 && shift<=28 && shift%4==0);
        return {{at::kFloat,{x.size(0),(x.size(1)+2047)/2048,16}}};
    }
    return {{at::kInt,{x.size(0),2,x.size(1)/64}}};
}
template<bool Bins> class Bounds final:public habana::OpBackend {
public:
    Bounds(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string(names[Bins]),t,{0},{},{},false) {
        SetOutputMetaFn([](const at::Stack& s){return metadata(s,Bins);});
    }
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto m=metadata(s,Bins);habana::NodeAttr attr;attr.guid=names[Bins];
        attr.inputs={syn_in(0),syn_in(1)};
        int32_t shift=0;
        if constexpr(Bins){attr.inputs.push_back(syn_in(2));attr.inputs.push_back(syn_in(3));
            shift=s[4].toInt();attr.params=&shift;attr.param_size=sizeof(shift);}
        attr.output_attrs={{m[0].shape,m[0].dtype,0}};
        syn_out(0)=std::move(BuildNode(this,graph,std::move(attr))[0]);
    }
};
template<bool Bins> bool register_op() {
    habana::custom_op::registerUserCustomOp(schema(Bins),names[Bins],[](const at::Stack& s){
        const auto m=metadata(s,Bins);return habana::PartialOutputMetaDataVector{{m[0].dtype,m[0].shape}};
    },nullptr);
    habana::KernelRegistry().add(schema(Bins),[](synDeviceId d,c10::ScalarType t){return std::make_shared<Bounds<Bins>>(d,t);});
    return true;
}
habana::OutputMetaDataVector active_meta(const at::Stack& s) {
    const auto b=s[0].toTensor(),p=s[1].toTensor();const auto shift=s[2].toInt();
    TORCH_CHECK(b.dim()==3 && b.size(0)>=1 && b.size(0)<=6 && b.size(1)==2 && b.size(2)>=1 &&
                b.size(2)<=2048 && b.scalar_type()==at::kInt && p.scalar_type()==at::kInt &&
                p.sizes()==at::IntArrayRef({b.size(0),1}) && b.device()==p.device() &&
                b.is_contiguous() && p.is_contiguous() && shift>=0 && shift<=28 && shift%4==0);
    return {{at::kInt,{b.size(0),(b.size(2)+31)/32}}};
}
class Active final:public habana::OpBackend {
public:
    Active(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string(names[2]),t,{0},{},{},false){SetOutputMetaFn(active_meta);}
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto m=active_meta(s);int32_t shift=s[2].toInt();
        syn_out(0)=std::move(BuildNode(this,graph,{names[2],{syn_in(0),syn_in(1)},
            {{m[0].shape,m[0].dtype,0}},&shift,sizeof(shift)})[0]);
    }
};
bool register_active() {
    habana::custom_op::registerUserCustomOp(schema(2),names[2],[](const at::Stack& s){
        const auto m=active_meta(s);return habana::PartialOutputMetaDataVector{{m[0].dtype,m[0].shape}};
    },nullptr);
    habana::KernelRegistry().add(schema(2),[](synDeviceId d,c10::ScalarType t){return std::make_shared<Active>(d,t);});
    return true;
}
const bool ready=register_op<false>() && register_op<true>() && register_active();
template<bool Meta,bool Bins> at::Tensor execute(const at::Stack& s) {
    const auto m=metadata(s,Bins);const auto x=s[0].toTensor();
    if constexpr(Meta)return at::empty(m[0].shape,x.options().dtype(m[0].dtype));
    TORCH_CHECK(ready && x.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema(Bins));return op.execute(s)[0];
}
template<bool M> at::Tensor bounds(const at::Tensor& x,const at::Tensor& w){return execute<M,false>({x,w});}
template<bool M> at::Tensor bins(const at::Tensor& x,const at::Tensor& w,const at::Tensor& p,const at::Tensor& b,int64_t shift){
    return execute<M,true>({x,w,p,b,shift});
}
template<bool M> at::Tensor active(const at::Tensor& b,const at::Tensor& p,int64_t shift) {
    const at::Stack s{b,p,shift};const auto m=active_meta(s);
    if constexpr(M)return at::empty(m[0].shape,b.options());
    TORCH_CHECK(ready && b.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema(2));return op.execute(s)[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_weighted_score_bounds_gaudi2(Tensor scores, Tensor weights) -> Tensor");
    m.def("custom_deepseek_v41_weighted_sparse_bins_gaudi2(Tensor scores, Tensor weights, Tensor prefix, Tensor bounds, int shift) -> Tensor");
    m.def("custom_deepseek_v41_weighted_active_blocks_gaudi2(Tensor bounds, Tensor prefix, int shift) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl(names[0],bounds<false>);m.impl(names[1],bins<false>);m.impl(names[2],active<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl(names[0],bounds<true>);m.impl(names[1],bins<true>);m.impl(names[2],active<true>);}
