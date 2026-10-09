// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <vector>
#include <cstring>
#include <iostream>
using namespace tpc_lib_api;
static Tensor tensor(TensorDataType type,std::initializer_list<uint64_t> dimensions) {
    Tensor value{};value.geometry.dataType=type;value.geometry.dims=dimensions.size();unsigned i=0;
    for(auto size:dimensions){value.geometry.maxSizes[i]=value.geometry.minSizes[i]=size;++i;}
    return value;
}
int main(int argc,char** argv) {
    if(argc!=2 && argc!=3)return 2;
    const unsigned tile=argc==3?std::stoul(argv[2]):128;
    if(tile!=128 && tile!=512)return 2;
    void* library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!library){std::cerr<<dlerror();return 3;}
    const auto instantiate=reinterpret_cast<GlueCodeReturn(*)(HabanaKernelParams*,HabanaKernelInstantiation*)>(
        dlsym(library,"InstantiateTpcKernel"));
    if(!instantiate)return 4;
    for(bool horizontal:{true,false})for(bool compact:{true,false})for(unsigned k:{128u,640u,1152u,2560u,5120u}) {
        const unsigned blocks=horizontal?5:20,slots=36;
        std::vector<Tensor> inputs{tensor(DATA_I32,{slots,1}),tensor(DATA_I16,{k*64,blocks,384}),
            tensor(DATA_I16,{k*(compact?4u:8u),blocks,384}),tensor(DATA_BF16,{128}),
            tensor(DATA_I16,{128,blocks,384})};
        std::vector<Tensor> outputs{tensor(DATA_F8_143,{blocks*256*(horizontal?2u:1u),k,slots/(horizontal?2u:1u)})};
        const auto original=inputs;
        for(unsigned repetition=0;repetition<2;++repetition) {
            std::vector<TensorAccessPattern> ip(inputs.size()),op(outputs.size());
            HabanaKernelParams params{};params.apiVersion=1;params.deviceId=DEVICE_ID_GAUDI2;
            params.maxAvailableTpc=24;params.inputTensorNr=inputs.size();params.outputTensorNr=outputs.size();
            params.inputTensors=inputs.data();params.outputTensors=outputs.data();
            std::strcpy(params.guid.name,tile==512?
                (horizontal?"custom_deepseek_v41_expert_w13_k512_gaudi2":"custom_deepseek_v41_expert_w2_k512_gaudi2"):
                (horizontal?"custom_deepseek_v41_expert_w13_split_scale_gaudi2":
                            "custom_deepseek_v41_expert_w2_split_scale_gaudi2"));
            HabanaKernelInstantiation result{};result.inputTensorAccessPattern=ip.data();
            result.outputTensorAccessPattern=op.data();
            auto status=instantiate(&params,&result);
            if(status!=GLUE_INSUFFICIENT_ELF_BUFFER || !result.kernel.elfSize){std::cerr<<"initial status "<<status;return 5;}
            std::vector<char> elf(result.kernel.elfSize);result.kernel.kernelElf=elf.data();
            status=instantiate(&params,&result);
            if(status!=GLUE_SUCCESS){std::cerr<<"final status "<<status;return 6;}
            if(std::memcmp(inputs.data(),original.data(),inputs.size()*sizeof(Tensor)))return 7;
            if(result.indexSpaceGeometry[2]!=(k+tile-1)/tile || ip[2].mapping[0].a!=tile*(compact?4:8) ||
               ip[4].mapping[0].a!=0 || ip[4].mapping[0].end_b!=127)return 8;
            if(tile==512 && (result.kernel.paramsNr!=1 || result.kernel.scalarParams[0]!=k ||
                ip[1].mapping[0].a!=tile*64 || op[0].mapping[1].a!=tile))return 9;
        }
    }
    std::cout<<"Affine K workpoint/full K, legacy/compact, tail bound and caller descriptors verified\n";
}
