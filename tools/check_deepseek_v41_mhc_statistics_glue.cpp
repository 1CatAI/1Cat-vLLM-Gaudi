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
    for(unsigned rows:{2u,5u,6u})for(bool finish:{false,true}) {
        Tensor inputs[]={tensor(DATA_F32,{48,rows}),tensor(DATA_F32,{40,rows}),
                         tensor(DATA_F32,{3}),tensor(DATA_F32,{24})};
        if(!finish)inputs[0]=tensor(DATA_BF16,{20480,rows});
        Tensor outputs[]={tensor(DATA_F32,{finish?24u:40u,rows})};
        HabanaKernelParams params{};params.apiVersion=1;params.deviceId=DEVICE_ID_GAUDI2;params.maxAvailableTpc=24;
        std::strcpy(params.guid.name,finish?"custom_deepseek_v41_mhc_statistics_finish_gaudi2":
                                            "custom_deepseek_v41_mhc_statistics_gaudi2");
        params.inputTensorNr=finish?4:1;params.outputTensorNr=1;params.inputTensors=inputs;params.outputTensors=outputs;
        TensorAccessPattern input_ap[4]{},output_ap[1]{};
        HabanaKernelInstantiation result{};result.inputTensorAccessPattern=input_ap;result.outputTensorAccessPattern=output_ap;
        const auto first=function(&params,&result);
        if(first!=GLUE_INSUFFICIENT_ELF_BUFFER || !result.kernel.elfSize)return 1;
        std::vector<char> elf(result.kernel.elfSize);result.kernel.kernelElf=elf.data();
        const auto second=function(&params,&result);
        bool passed=second==GLUE_SUCCESS && result.kernel.elfSize==elf.size() && result.kernel.paramsNr==0;
        passed=passed && (finish ? (result.indexSpaceRank==1 && result.indexSpaceGeometry[0]==rows &&
                                   input_ap[1].mapping[0].end_b==39 && output_ap[0].mapping[1].a==1):
                                  (result.indexSpaceRank==2 && result.indexSpaceGeometry[0]==40 &&
                                   result.indexSpaceGeometry[1]==rows && input_ap[0].mapping[0].a==512 &&
                                   input_ap[0].mapping[0].end_b==511 && output_ap[0].mapping[1].indexSpaceDim==1));
        printf("C%u finish%u status%u ELF%u pass%u\n",rows,finish,unsigned(second),result.kernel.elfSize,passed);
        if(!passed)return 1;
    }
    return 0;
}
