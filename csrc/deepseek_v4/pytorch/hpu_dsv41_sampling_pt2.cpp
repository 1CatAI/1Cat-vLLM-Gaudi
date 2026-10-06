// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <memory>
#include "hpu_ops/op_backend.h"
namespace {
constexpr const char* names[]={"custom_op::custom_deepseek_v41_sampling_unpack_gaudi2",
    "custom_op::custom_deepseek_v41_sampling_mask_gaudi2","custom_op::custom_deepseek_v41_sampling_select_gaudi2"};
struct Params { int columns,ranks,width; };
using Meta=habana::PartialOutputMetaDataVector;
Meta metadata(unsigned mode,const at::Stack& s) {
    const unsigned counts[]={1,6,4};
    TORCH_CHECK(mode<3 && s.size()==(mode==0?3:mode==1?7:4),"Invalid sampling operand count");
    const auto first=s.at(0).toTensor();
    TORCH_CHECK(first.dim()==2 && first.scalar_type()==at::kFloat && first.size(0)>0 && first.size(0)<=64,
                "Sampling requires bounded F32 rows");
    for(unsigned i=0;i<counts[mode];++i) {
        const auto t=s.at(i).toTensor();
        TORCH_CHECK(t.device()==first.device() && t.is_contiguous() && !t.requires_grad(),
                    "Sampling requires matching contiguous inference operands");
    }
    const int64_t batch=first.size(0);
    if(mode==0) {
        const auto ranks=s.at(1).toInt(),width=s.at(2).toInt();
        TORCH_CHECK(ranks>=2 && ranks<=8 && width>=64 && width<=256 && width%64==0 &&
                    first.size(1)==ranks*(3+2*width),"Invalid sampling packet layout");
        return {{at::kFloat,{batch,ranks*width}},{at::kInt,{batch,ranks*width}},
                {at::kFloat,{batch,ranks*3}},{at::kFloat,{batch,ranks}}};
    }
    const auto columns=first.size(1);
    TORCH_CHECK(columns>=64 && columns<=4096 && columns%64==0,"Invalid sampling candidate width");
    auto operand=[&](unsigned i,at::ScalarType dtype,int64_t width) {
        const auto t=s.at(i).toTensor();
        TORCH_CHECK(t.scalar_type()==dtype && t.sizes()==at::IntArrayRef({batch,width}),
                    "Invalid sampling operand shape or dtype at ",i);
    };
    if(mode==1) {
        const auto ranks=s.at(6).toInt();
        TORCH_CHECK(ranks>=2 && ranks<=8 && columns%ranks==0,"Invalid sampling rank geometry");
        operand(1,at::kFloat,columns);operand(2,at::kFloat,columns);operand(3,at::kFloat,ranks);
        operand(4,at::kFloat,4);operand(5,at::kFloat,1);
        return {{at::kFloat,{batch,columns}},{at::kInt,{batch,1}}};
    }
    operand(1,at::kInt,columns);operand(2,at::kFloat,4);operand(3,at::kInt,1);
    return {{at::kInt,{batch,1}}};
}
const bool registered=[] {
    for(unsigned mode=0;mode<3;++mode) {
        habana::custom_op::registerUserCustomOp(names[mode],names[mode]+11,
            [mode](const at::Stack& s){return metadata(mode,s);},
            [mode](const at::Stack& s,size_t& bytes)->std::shared_ptr<void> {
                const auto first=s.at(0).toTensor();
                const int ranks=mode==0?s.at(1).toInt():mode==1?s.at(6).toInt():0;
                const int width=mode==0?s.at(2).toInt():mode==1?first.size(1)/ranks:0;
                bytes=sizeof(Params);
                return std::make_shared<Params>(Params{mode==0?ranks*width:int(first.size(1)),ranks,width});
            });
    }
    return true;
}();
std::vector<at::Tensor> execute(unsigned mode,const at::Stack& s,bool fake) {
    const auto m=metadata(mode,s);const auto first=s.at(0).toTensor();
    if(fake) {
        std::vector<at::Tensor> result;
        for(const auto& out:m)result.push_back(at::empty(out.shape,first.options().dtype(out.dtype)));
        return result;
    }
    TORCH_CHECK(registered && first.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(names[mode]);
    return descriptor.execute(s);
}
template<bool Fake>
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> unpack(const at::Tensor& packet,int64_t ranks,int64_t width) {
    const auto out=execute(0,{packet,ranks,width},Fake);return {out[0],out[1],out[2],out[3]};
}
template<bool Fake>
std::tuple<at::Tensor,at::Tensor> mask(const at::Tensor& scores,const at::Tensor& prob,const at::Tensor& cdf,
    const at::Tensor& cuts,const at::Tensor& controls,const at::Tensor& total,int64_t ranks) {
    const auto out=execute(1,{scores,prob,cdf,cuts,controls,total,ranks},Fake);return {out[0],out[1]};
}
template<bool Fake>
at::Tensor select(const at::Tensor& cdf,const at::Tensor& ids,const at::Tensor& controls,const at::Tensor& greedy) {
    return execute(2,{cdf,ids,controls,greedy},Fake)[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_sampling_unpack_gaudi2(Tensor packet, int ranks, int width) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("custom_deepseek_v41_sampling_mask_gaudi2(Tensor scores, Tensor probability, Tensor cumulative, Tensor cuts, Tensor controls, Tensor total, int ranks) -> (Tensor, Tensor)");
    m.def("custom_deepseek_v41_sampling_select_gaudi2(Tensor cumulative, Tensor ids, Tensor controls, Tensor greedy) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_sampling_unpack_gaudi2",unpack<false>);
    m.impl("custom_deepseek_v41_sampling_mask_gaudi2",mask<false>);
    m.impl("custom_deepseek_v41_sampling_select_gaudi2",select<false>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_sampling_unpack_gaudi2",unpack<true>);
    m.impl("custom_deepseek_v41_sampling_mask_gaudi2",mask<true>);
    m.impl("custom_deepseek_v41_sampling_select_gaudi2",select<true>);
}
