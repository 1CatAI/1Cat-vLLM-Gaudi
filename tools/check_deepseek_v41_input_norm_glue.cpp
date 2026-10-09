// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace tpc_lib_api;
static Tensor tensor(std::initializer_list<uint64_t> sizes) {
    Tensor value{};value.geometry.dataType=DATA_BF16;value.geometry.dims=sizes.size();unsigned d=0;
    for(auto n:sizes){value.geometry.maxSizes[d]=value.geometry.minSizes[d]=n;value.permutation[d]=d;++d;}
    return value;
}
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    auto library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!library){puts(dlerror());return 2;}
    auto function=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(library,"InstantiateTpcKernel"));
    if(!function)return 2;
    for(unsigned rows:{1u,2u,6u}) {
        Tensor inputs[]={tensor({5120,rows}),tensor({5120})},outputs[]={tensor({5120,rows})};
        HabanaKernelParams params{};params.apiVersion=1;params.deviceId=DEVICE_ID_GAUDI2;params.maxAvailableTpc=24;
        std::strcpy(params.guid.name,"custom_deepseek_v41_input_norm_bf16_gaudi2");
        params.inputTensorNr=2;params.outputTensorNr=1;params.inputTensors=inputs;params.outputTensors=outputs;
        float epsilon=1e-20f;params.nodeParams.nodeParams=&epsilon;params.nodeParams.nodeParamsSize=sizeof(epsilon);
        TensorAccessPattern input_ap[2]{},output_ap[1]{};
        HabanaKernelInstantiation result{};result.inputTensorAccessPattern=input_ap;result.outputTensorAccessPattern=output_ap;
        const auto first=function(&params,&result);
        if(first!=GLUE_INSUFFICIENT_ELF_BUFFER || !result.kernel.elfSize)return 1;
        std::vector<char> elf(result.kernel.elfSize);result.kernel.kernelElf=elf.data();
        const auto second=function(&params,&result);
        const bool passed=second==GLUE_SUCCESS && result.kernel.elfSize==elf.size() &&
            result.indexSpaceRank==1 && result.indexSpaceGeometry[0]==rows &&
            input_ap[0].mapping[1].a==1 && output_ap[0].mapping[1].a==1 &&
            result.kernel.paramsNr==2;
        printf("C%u status%u ELF%u pass%u\n",rows,unsigned(second),result.kernel.elfSize,passed);
        if(!passed)return 1;
    }
    return 0;
}
