// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace tpc_lib_api;
static Tensor tensor(TensorDataType type,std::initializer_list<uint64_t> sizes) {
    Tensor value{};value.geometry.dataType=type;value.geometry.dims=sizes.size();unsigned d=0;
    for(auto n:sizes){value.geometry.maxSizes[d]=value.geometry.minSizes[d]=n;value.permutation[d]=d;++d;}
    return value;
}
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    auto library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!library){puts(dlerror());return 2;}
    auto function=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(library,"InstantiateTpcKernel"));
    if(!function)return 2;
    for(unsigned intermediate:{1152u,1280u})for(bool down:{false,true}) {
        unsigned k=down?intermediate:5120u,n=down?5120u:intermediate*2,slots=36;
        Tensor inputs[]={tensor(DATA_I32,{slots,1}),tensor(DATA_I16,{k*64,n/256,384}),
                         tensor(DATA_I16,{k*4+128,n/256,384}),tensor(DATA_BF16,{128})};
        Tensor outputs[]={tensor(DATA_F8_143,{down?n:n*2,k,down?slots:slots/2})};
        HabanaKernelParams params{};params.apiVersion=1;params.deviceId=DEVICE_ID_GAUDI2;params.maxAvailableTpc=24;
        std::strcpy(params.guid.name,down?"custom_deepseek_v41_expert_down_n512_gaudi2":
                                         "custom_deepseek_v41_expert_token_wide_n512_gaudi2");
        params.inputTensorNr=4;params.outputTensorNr=1;params.inputTensors=inputs;params.outputTensors=outputs;
        TensorAccessPattern input_ap[4]{},output_ap[1]{};
        HabanaKernelInstantiation result{};result.inputTensorAccessPattern=input_ap;result.outputTensorAccessPattern=output_ap;
        const auto first=function(&params,&result);
        if(first!=GLUE_INSUFFICIENT_ELF_BUFFER || !result.kernel.elfSize){printf("first%u\n",unsigned(first));return 1;}
        std::vector<char> elf(result.kernel.elfSize);result.kernel.kernelElf=elf.data();
        const auto second=function(&params,&result);
        bool passed=second==GLUE_SUCCESS && result.kernel.elfSize==elf.size() && result.kernel.paramsNr==0 &&
            result.indexSpaceRank==3 && result.indexSpaceGeometry[0]==outputs[0].geometry.maxSizes[0]/512 &&
            result.indexSpaceGeometry[2]==k/128 && output_ap[0].mapping[0].a==512 &&
            output_ap[0].mapping[0].end_b==511;
        if(down)passed=passed && input_ap[1].mapping[1].a==2 && input_ap[1].mapping[1].end_b==1;
        printf("I%u down%u status%u ELF%u pass%u\n",intermediate,down,unsigned(second),result.kernel.elfSize,passed);
        if(!passed)return 1;
    }
    return 0;
}
