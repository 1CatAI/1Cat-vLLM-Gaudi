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
static std::vector<char> instantiate(pfnInstantiateTpcKernel fn,HabanaKernelParams& p,
                                   HabanaKernelInstantiation& out) {
    out.kernel.elfSize=0;
    auto rc=fn(&p,&out);
    if(rc!=GLUE_INSUFFICIENT_ELF_BUFFER){printf("%s probe rc%u\n",p.guid.name,unsigned(rc));return {};}
    std::vector<char> data(out.kernel.elfSize);out.kernel.kernelElf=data.data();
    rc=fn(&p,&out);
    if(rc!=GLUE_SUCCESS){printf("%s fill rc%u\n",p.guid.name,unsigned(rc));return {};}
    return data;
}
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    auto a=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),b=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);
    if(!a||!b){puts(dlerror());return 2;}
    auto candidate=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(a,"InstantiateTpcKernel"));
    auto reference=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(b,"InstantiateTpcKernel"));
    for(uint64_t columns:{1792u,3072u})for(uint64_t rows:{2u,6u}) {
        Tensor input[]={tensor(DATA_F32,{columns,rows}),tensor(DATA_F32,{384}),tensor(DATA_F32,{384}),
            tensor(DATA_I8,{rows}),tensor(DATA_F32,{384,1}),tensor(DATA_F32,{1,rows})};
        Tensor result[]={tensor(DATA_I32,{6,rows}),tensor(DATA_F32,{6,rows})};
        HabanaKernelParams p{};p.deviceId=DEVICE_ID_GAUDI2;p.apiVersion=1;p.maxAvailableTpc=24;
        p.inputTensorNr=6;p.outputTensorNr=2;p.inputTensors=input;p.outputTensors=result;
        std::strcpy(p.guid.name,"custom_deepseek_v41_router_shared_scaled_gaudi2");
        TensorAccessPattern inAP[6]{},outAP[2]{};HabanaKernelInstantiation out{};
        out.inputTensorAccessPattern=inAP;out.outputTensorAccessPattern=outAP;
        if(instantiate(candidate,p,out).empty() || out.indexSpaceGeometry[0]!=rows ||
            inAP[0].mapping[0].start_b!=columns-512 || inAP[0].mapping[0].end_b!=columns-129 ||
            inAP[0].mapping[1].a!=1)return 1;
        const uint64_t shared=columns-512;
        Tensor sin[]={tensor(DATA_F32,{columns,1,rows}),tensor(DATA_I32,{rows,1}),tensor(DATA_F32,{1,rows}),
            tensor(DATA_BF16,{256,shared/256,1}),tensor(DATA_F32,{rows,1})};
        Tensor sout[]={tensor(DATA_F8_143,{shared/2,1,rows}),tensor(DATA_F32,{1,1,rows})};
        p.inputTensorNr=5;p.inputTensors=sin;p.outputTensors=sout;
        std::strcpy(p.guid.name,"custom_deepseek_v41_shared_silu_full_product_gaudi2");
        auto code=instantiate(candidate,p,out);
        if(code.empty() || out.indexSpaceGeometry[0]!=rows || inAP[0].mapping[0].end_b!=shared-1 ||
           inAP[0].mapping[2].a!=1)return 1;
        sin[0].geometry.maxSizes[0]=sin[0].geometry.minSizes[0]=shared;
        std::strcpy(p.guid.name,"custom_deepseek_v41_shared_silu_quant_gaudi2");
        if(code!=instantiate(reference,p,out))return 1;
        printf("columns%lu rows%lu: Router bounds and unchanged shared ELF pass\n",columns,rows);
    }
}
