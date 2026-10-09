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
    auto library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!library){puts(dlerror());return 2;}
    auto function=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(library,"InstantiateTpcKernel"));
    if(!function)return 2;
    for(unsigned tokens:{2u,5u,6u})for(unsigned width:{640u,1280u}) {
        const unsigned slots=tokens*6;
        Tensor input[]={tensor(DATA_F32,{width*2,1,slots}),tensor(DATA_I32,{slots,1}),
            tensor(DATA_F32,{1,slots}),tensor(DATA_BF16,{256,width*2/256,384}),tensor(DATA_F32,{slots,1})};
        Tensor output[]={tensor(DATA_F8_143,{width,1,slots}),tensor(DATA_F32,{1,1,slots}),
            tensor(DATA_I32,{32,slots})};
        HabanaKernelParams params{};params.apiVersion=1;params.deviceId=DEVICE_ID_GAUDI2;params.maxAvailableTpc=24;
        std::strcpy(params.guid.name,"custom_deepseek_v41_cooperative_silu_gaudi2");
        params.inputTensorNr=5;params.outputTensorNr=3;params.inputTensors=input;params.outputTensors=output;
        TensorAccessPattern input_ap[5]{},output_ap[3]{};
        HabanaKernelInstantiation result{};result.inputTensorAccessPattern=input_ap;result.outputTensorAccessPattern=output_ap;
        const auto first_status=function(&params,&result);
        printf("first C%u width%u status%u ELF%u\n",tokens,width,unsigned(first_status),result.kernel.elfSize);
        if(first_status!=GLUE_INSUFFICIENT_ELF_BUFFER||!result.kernel.elfSize)return 1;
        std::vector<char> elf(result.kernel.elfSize);result.kernel.kernelElf=elf.data();
        const auto status=function(&params,&result);
        const bool passed=status==GLUE_SUCCESS && result.indexSpaceRank==2 &&
            result.indexSpaceGeometry[0]==width/128 && result.indexSpaceGeometry[1]==slots &&
            result.preferredSplitDim==1 && output_ap[2].memsetBeforeExecution &&
            output_ap[2].mapping[1].a==1 && !output_ap[2].mapping[1].allRequired;
        printf("C%u width%u status%u ELF%u pass%u\n",tokens,width,unsigned(status),result.kernel.elfSize,passed);
        if(!passed)return 1;
        output[2].geometry.maxSizes[0]=31;
        if(function(&params,&result)!=GLUE_INCOMPATIBLE_OUTPUT_SIZE)return 1;
    }
    return 0;
}
