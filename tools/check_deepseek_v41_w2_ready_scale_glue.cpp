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
    for(unsigned tokens:{2u,5u,6u})for(bool reduce:{false,true}) {
        const unsigned slots=tokens*6;
        Tensor input[]={tensor(DATA_I32,{slots,1}),tensor(DATA_F32,{1,slots}),tensor(DATA_BF16,{256,20,384})};
        Tensor output[]={tensor(reduce?DATA_BF16:DATA_F32,{5120,1,reduce?tokens:slots})};
        if(reduce){input[0]=tensor(DATA_F32,{5120,1,slots});input[1]=input[0];}
        HabanaKernelParams params{};params.apiVersion=1;params.deviceId=DEVICE_ID_GAUDI2;params.maxAvailableTpc=24;
        std::strcpy(params.guid.name,reduce?"custom_deepseek_v41_w2_ready_reduce_gaudi2":
                                           "custom_deepseek_v41_w2_ready_scale_gaudi2");
        params.inputTensorNr=reduce?2:3;params.outputTensorNr=1;params.inputTensors=input;params.outputTensors=output;
        TensorAccessPattern input_ap[3]{},output_ap[1]{};
        HabanaKernelInstantiation result{};result.inputTensorAccessPattern=input_ap;result.outputTensorAccessPattern=output_ap;
        if(function(&params,&result)!=GLUE_INSUFFICIENT_ELF_BUFFER||!result.kernel.elfSize)return 1;
        std::vector<char> elf(result.kernel.elfSize);result.kernel.kernelElf=elf.data();
        const auto status=function(&params,&result);
        const bool passed=status==GLUE_SUCCESS && result.indexSpaceRank==2 &&
            result.indexSpaceGeometry[0]==40 && result.indexSpaceGeometry[1]==(reduce?tokens:slots) &&
            (reduce?input_ap[0].mapping[2].a==6:input_ap[0].mapping[0].indexSpaceDim==1);
        printf("C%u reduce%u status%u ELF%u pass%u\n",tokens,reduce,unsigned(status),result.kernel.elfSize,passed);
        if(!passed)return 1;
        if(!reduce){input[0].geometry.dims=1;if(function(&params,&result)!=GLUE_INCOMPATIBLE_INPUT_SIZE)return 1;}
    }
    return 0;
}
