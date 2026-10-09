// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace tpc_lib_api;
static Tensor t(TensorDataType type,std::initializer_list<uint64_t> sizes) {
    Tensor x{};x.geometry.dataType=type;x.geometry.dims=sizes.size();unsigned d=0;
    for(auto n:sizes){x.geometry.maxSizes[d]=x.geometry.minSizes[d]=n;x.permutation[d]=d;++d;}return x;
}
int main(int argc,char** argv) {
    if(argc!=2)return 2;auto lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib){puts(dlerror());return 2;}
    auto f=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(lib,"InstantiateTpcKernel"));if(!f)return 2;
    for(unsigned c=2;c<=6;++c)for(unsigned h:{6u,12u}) {
        std::vector<Tensor> inputs={t(DATA_I32,{c,1}),t(DATA_I32,{3,1}),t(DATA_I32,{129280,1}),
            t(DATA_I32,{47,1}),t(DATA_U8,{256,96000000}),t(DATA_U8,{8,96000000})};
        std::vector<Tensor> outputs={t(DATA_BF16,{256,h,c}),t(DATA_I32,{3,c+1})};
        HabanaKernelParams p{};p.apiVersion=1;p.deviceId=DEVICE_ID_GAUDI2;p.maxAvailableTpc=24;
        strcpy(p.guid.name,"custom_deepseek_v41_engram_batch_gaudi2");p.inputTensorNr=6;p.inputTensors=inputs.data();
        p.outputTensorNr=2;p.outputTensors=outputs.data();TensorAccessPattern ip[6]{},op[2]{};
        HabanaKernelInstantiation r{};r.inputTensorAccessPattern=ip;r.outputTensorAccessPattern=op;
        std::vector<char> elf(1024*1024);r.kernel.kernelElf=elf.data();r.kernel.elfSize=elf.size();
        auto status=f(&p,&r);
        bool ok=status==GLUE_SUCCESS && r.indexSpaceRank==2 && r.indexSpaceGeometry[0]==h &&
            r.indexSpaceGeometry[1]==c && op[0].mapping[1].indexSpaceDim==0 && op[0].mapping[1].a==1 &&
            op[0].mapping[2].indexSpaceDim==1 && op[0].mapping[2].a==1 && op[1].allRequired;
        printf("C%u heads%u status%u pass%u\n",c,h,unsigned(status),ok);if(!ok)return 1;
    }return 0;
}
