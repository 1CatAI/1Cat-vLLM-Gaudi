// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <filesystem>
#include "kernel_bytes.h"
#include <vector>
#include <dlfcn.h>
#include <stdexcept>
using namespace tpc_lib_api;
const std::filesystem::path own_directory(){Dl_info info{};if(!dladdr(reinterpret_cast<void*>(&own_directory),&info)||!info.dli_fname)throw std::runtime_error("Cannot locate kernel database");return std::filesystem::path(info.dli_fname).parent_path();}
template<class F> F parent(const char* n){static void* h=dlopen((own_directory()/"libdeepseek_v4_gaudi2_base.so").c_str(),RTLD_NOW|RTLD_LOCAL);if(!h)throw std::runtime_error(dlerror());auto f=dlsym(h,n);if(!f)throw std::runtime_error(n);return reinterpret_cast<F>(f);}
const char* extra[]={"private_memory_ready_post_tpc","private_memory_flags_zero_tpc"};int which(const char*n){for(int i=0;i<2;++i)if(!strcmp(n,extra[i]))return i;return -1;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*out){auto f=parent<pfnGetKernelGuids>("GetKernelGuids");if(d!=DEVICE_ID_GAUDI2)return f(d,n,out);uint32_t count=0;auto c=f(d,&count,nullptr);if(c!=GLUE_SUCCESS)return c;auto cap=*n;*n=count+2;if(!out||!cap)return GLUE_SUCCESS;if(cap<count+2)return GLUE_FAILED;c=f(d,&count,out);if(c!=GLUE_SUCCESS)return c;for(int i=0;i<2;++i){memset(out+count+i,0,sizeof(*out));strcpy(out[count+i].name,extra[i]);}return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*p,HabanaKernelInstantiation*g){int index=which(p->guid.name);if(index<0)return parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel")(p,g);
 const auto cap=g->kernel.elfSize;
 if(index==0){if(p->inputTensorNr!=6||p->outputTensorNr!=4||p->nodeParams.nodeParamsSize!=4||!p->nodeParams.nodeParams)return GLUE_FAILED;for(unsigned i=4;i<6;++i)if(p->inputTensors[i].geometry.dataType!=DATA_I32)return GLUE_FAILED;
  auto fake=*p;fake.inputTensorNr=4;fake.outputTensorNr=3;strcpy(fake.guid.name,"custom_deepseek_v41_mhc_post_weighted_stats_gaudi2");
  auto gi=*g;TensorAccessPattern ia[4]{},oa[3]{};gi.inputTensorAccessPattern=ia;gi.outputTensorAccessPattern=oa;
  auto c=parent<pfnInstantiateTpcKernel>("InstantiateTpcKernel")(&fake,&gi);if(c!=GLUE_SUCCESS&&c!=GLUE_INSUFFICIENT_ELF_BUFFER)return c;
  auto actualIn=g->inputTensorAccessPattern;auto actualOut=g->outputTensorAccessPattern;*g=gi;g->inputTensorAccessPattern=actualIn;g->outputTensorAccessPattern=actualOut;
  for(int i=0;i<4;++i)actualIn[i]=ia[i];for(int i=0;i<3;++i)actualOut[i]=oa[i];for(int i=4;i<6;++i){memset(actualIn+i,0,sizeof(*actualIn));actualIn[i].allRequired=true;}
  g->kernel.paramsNr=1;memcpy(g->kernel.scalarParams,p->nodeParams.nodeParams,4);
  memset(actualOut+3,0,sizeof(*actualOut));actualOut[3].mapping[0].indexSpaceDim=0;actualOut[3].mapping[0].a=1;actualOut[3].mapping[0].start_b=0;actualOut[3].mapping[0].end_b=0;
 }else{if(p->inputTensorNr!=1||p->outputTensorNr!=1||p->inputTensors[0].geometry.dataType!=DATA_I32||p->outputTensors[0].geometry.dataType!=DATA_I32)return GLUE_FAILED;
  g->indexSpaceRank=1;g->indexSpaceGeometry[0]=1;g->kernel.paramsNr=0;memset(g->inputTensorAccessPattern,0,sizeof(TensorAccessPattern));memset(g->outputTensorAccessPattern,0,sizeof(TensorAccessPattern));g->inputTensorAccessPattern[0].allRequired=true;g->outputTensorAccessPattern[0].allRequired=true;
 }
 const auto* elf=index==0?gated_post_bytes:zero_flags_bytes;auto bytes=index==0?sizeof(gated_post_bytes):sizeof(zero_flags_bytes);g->kernel.elfSize=bytes;if(cap<bytes)return GLUE_INSUFFICIENT_ELF_BUFFER;memcpy(g->kernel.kernelElf,elf,bytes);return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion(){return parent<pfnGetLibVersion>("GetLibVersion")();}
extern "C" GlueCodeReturn GetShapeInference(DeviceId d,const ShapeInferenceParams*p,ShapeInferenceOutput*o){const auto*g=p->pGuid?p->pGuid:&p->guid;if(which(g->name)>=0)return GLUE_SUCCESS;return parent<pfnGetShapeInference>("GetShapeInference")(d,p,o);}
extern "C" GlueCodeReturn GetSuggestedManipulation(const HabanaKernelParams*p,void*o){if(which(p->guid.name)>=0)return GLUE_FAILED;return parent<GlueCodeReturn(*)(const HabanaKernelParams*,void*)>("GetSuggestedManipulation")(p,o);}
extern "C" GlueCodeReturn GetSupportedDataLayouts(const HabanaKernelParams*p,NodeDataLayouts*l,uint32_t*n){if(which(p->guid.name)<0)return parent<pfnGetSupportedDataLayout>("GetSupportedDataLayouts")(p,l,n);*n=1;if(l){for(unsigned i=0;i<l->inputTensorNr;++i)memset(l->inputs[i].layout,'x',sizeof(l->inputs[i].layout));for(unsigned i=0;i<l->outputTensorNr;++i)memset(l->outputs[i].layout,'x',sizeof(l->outputs[i].layout));}return GLUE_SUCCESS;}
