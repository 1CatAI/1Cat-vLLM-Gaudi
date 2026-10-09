// SPDX-License-Identifier: Apache-2.0
// The validator may borrow inherited glue but must preserve caller descriptors.
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
    if(argc!=2)return 2;
    void* library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!library){std::cerr<<dlerror();return 3;}
    const auto instantiate=reinterpret_cast<GlueCodeReturn(*)(HabanaKernelParams*,HabanaKernelInstantiation*)>(
        dlsym(library,"InstantiateTpcKernel"));
    if(!instantiate)return 4;
    for(unsigned rows:{2u,5u,6u}) {
        const uint64_t slots=rows*6;
        for(bool activation:{false,true}) {
            std::vector<Tensor> inputs=activation?std::vector<Tensor>{
                tensor(DATA_BF16,{1280,1,slots}),tensor(DATA_I32,{slots,1}),tensor(DATA_F32,{1,slots}),
                tensor(DATA_BF16,{256,5,384}),tensor(DATA_F32,{slots,1})}:
                std::vector<Tensor>{tensor(DATA_I32,{slots,1}),tensor(DATA_BF16,{256,5,384})};
            std::vector<Tensor> outputs=activation?std::vector<Tensor>{
                tensor(DATA_F8_143,{640,1,slots}),tensor(DATA_F32,{1,1,slots})}:
                std::vector<Tensor>{tensor(DATA_F32,{2560,1,slots/2})};
            const auto original=inputs;
            for(unsigned invocation=0;invocation<2;++invocation) {
                std::vector<TensorAccessPattern> input_patterns(inputs.size()),output_patterns(outputs.size());
                HabanaKernelParams params{};params.apiVersion=1;params.deviceId=DEVICE_ID_GAUDI2;
                params.maxAvailableTpc=24;params.inputTensorNr=inputs.size();params.outputTensorNr=outputs.size();
                params.inputTensors=inputs.data();params.outputTensors=outputs.data();
                std::strcpy(params.guid.name,activation?"custom_deepseek_v41_expert_scaled_silu_quant_gaudi2":
                            "custom_deepseek_v41_expert_pair_channels_gaudi2");
                HabanaKernelInstantiation result{};result.inputTensorAccessPattern=input_patterns.data();
                result.outputTensorAccessPattern=output_patterns.data();
                auto status=instantiate(&params,&result);
                if(status!=GLUE_INSUFFICIENT_ELF_BUFFER || !result.kernel.elfSize)return 5;
                std::vector<char> elf(result.kernel.elfSize);result.kernel.kernelElf=elf.data();
                status=instantiate(&params,&result);
                if(status!=GLUE_SUCCESS)return 6;
                if(std::memcmp(inputs.data(),original.data(),inputs.size()*sizeof(Tensor)))return 7;
                if(!result.indexSpaceRank)return 8;
            }
        }
    }
    std::cout<<"Caller descriptors preserved across repeated real-width instantiation\n";
}
