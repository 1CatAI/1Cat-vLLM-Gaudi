// SPDX-License-Identifier: Apache-2.0
// Private C2-C6 capability: preserve canonical main cache and split reuse.
#define DSV41_MAIN_PUBLISH_SCHEMA "custom_deepseek_v41_interleaved_publish_mla_gaudi2"
#define DSV41_MAIN_REUSE_SCHEMA "custom_deepseek_v41_interleaved_reuse_mla_gaudi2"
#define DSV41_MAIN_PUBLISH_GATHER_GUID "custom_deepseek_v41_main_batch_publish_gather_gaudi2"
#define DSV41_MAIN_REUSE_GATHER_GUID "custom_deepseek_v41_swa_only_reuse_gather_gaudi2"
#define DSV41_MAIN_SPLIT_REUSE 1
#define DSV41_INTERLEAVED_LAYOUT 1
#include "../../deepseek_v4/pytorch/hpu_dsv41_shared_main_mla_pt2.cpp"

namespace interleaved_primitives {
constexpr const char* names[]={"custom_op::custom_deepseek_v41_interleaved_query_rope_gaudi2",
                              "custom_op::custom_deepseek_v41_interleaved_pv_bf16_gaudi2"};
template<unsigned Mode> habana::PartialOutputMetaDataVector output(const at::Stack& s) {
    const auto x=s.at(0).toTensor(), p=s.at(1).toTensor(), phase=s.at(2).toTensor();
    TORCH_CHECK(x.dim()==3 && x.size(0)>=2 && x.size(0)<=6 && x.size(1)>=1 && x.size(1)<=8 &&
        x.size(2)==4096 && x.scalar_type()==(Mode?at::kFloat:at::kBFloat16) &&
        p.sizes()==at::IntArrayRef({x.size(0)}) && p.scalar_type()==at::kInt &&
        phase.dim()==2 && phase.size(0)>0 && phase.size(1)==64 && phase.scalar_type()==at::kFloat,
        "Invalid interleaved RoPE primitive contract");
    for(const auto& item:s) {
        const auto t=item.toTensor();
        TORCH_CHECK(t.device()==x.device() && t.is_contiguous() && !t.requires_grad());
    }
    return {{at::kBFloat16,Mode ? std::vector<int64_t>{x.size(1),x.size(0),x.size(2)} : x.sizes().vec()}};
}
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(names[0],names[0]+11,output<0>,nullptr);
    habana::custom_op::registerUserCustomOp(names[1],names[1]+11,output<1>,nullptr);
    return true;
}();
template<bool Meta,unsigned Mode> at::Tensor execute(const at::Tensor& x,const at::Tensor& p,const at::Tensor& phase) {
    const at::Stack s{x,p,phase};
    const auto m=output<Mode>(s);
    if(Meta)return at::empty(m[0].shape,x.options().dtype(at::kBFloat16));
    TORCH_CHECK(registered && x.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(names[Mode]);
    return descriptor.execute(s)[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_interleaved_query_rope_gaudi2(Tensor x, Tensor positions, Tensor phase) -> Tensor");
    m.def("custom_deepseek_v41_interleaved_pv_bf16_gaudi2(Tensor x, Tensor positions, Tensor phase) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_interleaved_query_rope_gaudi2",interleaved_primitives::execute<false,0>);
    m.impl("custom_deepseek_v41_interleaved_pv_bf16_gaudi2",interleaved_primitives::execute<false,1>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_interleaved_query_rope_gaudi2",interleaved_primitives::execute<true,0>);
    m.impl("custom_deepseek_v41_interleaved_pv_bf16_gaudi2",interleaved_primitives::execute<true,1>);
}

namespace interleaved_matrix_oracle {
constexpr const char* schema="custom_op::custom_deepseek_v41_interleaved_matrix_gaudi2";
habana::OutputMetaDataVector meta(const at::Stack& s) {
    const auto a=s.at(0).toTensor(),b=s.at(1).toTensor();
    TORCH_CHECK(a.dim()==4 && b.dim()==4 && a.size(0)==b.size(0) &&
        (a.size(1)==b.size(1) || a.size(1)==1 || b.size(1)==1) &&
        a.scalar_type()==b.scalar_type() && (a.scalar_type()==at::kBFloat16 || a.scalar_type()==at::kFloat) &&
        a.is_contiguous() && b.is_contiguous() && a.device()==b.device());
    const auto m=a.size(s.at(2).toBool()?3:2),k=a.size(s.at(2).toBool()?2:3);
    const auto bk=b.size(s.at(3).toBool()?3:2),n=b.size(s.at(3).toBool()?2:3);
    TORCH_CHECK(k==bk);
    return {{at::kFloat,{a.size(0),std::max(a.size(1),b.size(1)),m,n}}};
}
class Matrix final:public habana::OpBackend {
public:
    Matrix(int device,c10::ScalarType type):OpBackend(device,NO_TPC+std::string("dsv41_interleaved_matrix"),
        type,{0},{},{},false){SetOutputMetaFn(meta);}
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s)override {
        const auto out=meta(s);
        synGEMMParams p{s.at(2).toBool(),s.at(3).toBool()};
        syn_out(0)=std::move(BuildNode(this,graph,{"batch_gemm",{syn_in(0),syn_in(1)},
            {{out[0].shape,at::kFloat,0}},&p,sizeof(p)})[0]);
    }
};
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(schema,"batch_gemm",[](const at::Stack& s){
        const auto out=meta(s);return habana::PartialOutputMetaDataVector{{out[0].dtype,out[0].shape}};
    },nullptr);
    habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Matrix>(d,t);});
    return true;
}();
template<bool Meta>at::Tensor execute(const at::Tensor& a,const at::Tensor& b,bool ta,bool tb) {
    const at::Stack s{a,b,ta,tb};const auto out=meta(s);
    if(Meta)return at::empty(out[0].shape,a.options().dtype(at::kFloat));
    TORCH_CHECK(registered && a.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return op.execute(s)[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_interleaved_matrix_gaudi2(Tensor a, Tensor b, bool transpose_a, bool transpose_b) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_interleaved_matrix_gaudi2",interleaved_matrix_oracle::execute<false>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_interleaved_matrix_gaudi2",interleaved_matrix_oracle::execute<true>);
}
