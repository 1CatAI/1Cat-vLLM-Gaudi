// SPDX-License-Identifier: Apache-2.0
#include <tpc_test_core_api.h>
#include <fstream>
#include <iterator>
#include <iostream>
#include <vector>
#include <stdexcept>
using namespace tpc_lib_api;using namespace tpc_tests;
void shape(Tensor& t,TensorDataType type,std::initializer_list<uint64_t> dims){t.geometry.dataType=type;t.geometry.dims=dims.size();unsigned i=0;for(auto n:dims)t.geometry.maxSizes[i++]=n;}
TensorDesc2 desc(void* data,const Tensor& t,unsigned kind){TensorDesc2 d{};d.baseAddrUnion.baseAddr=(uint64_t)data;d.configuration=kind|(0x1f<<8)|((t.geometry.dims-1)<<16);uint32_t stride=1;for(unsigned i=0;i<5;++i){d.dimDescriptors[i].size=i<t.geometry.dims?t.geometry.maxSizes[i]:1;d.dimDescriptors[i].stride=stride;stride*=d.dimDescriptors[i].size;}return d;}
int main(int argc,char** argv){
 if(argc!=5)throw std::runtime_error("elf, rank3, slots, invalid required");
 bool affine=std::stoi(argv[2]),invalid=std::stoi(argv[4]);const int slots=std::stoi(argv[3]),n=256,k=128,experts=3,blocks=n/256,words=k*64,scale_words=k*4+128;
 Tensor in[4]{},out[1]{};TensorAccessPattern ia[4]{},oa[1]{};
 shape(in[0],DATA_I32,{(uint64_t)slots,1});shape(in[1],DATA_I16,{(uint64_t)words,(uint64_t)blocks,experts});shape(in[2],DATA_I16,{(uint64_t)scale_words,(uint64_t)blocks,experts});shape(in[3],DATA_BF16,{128});shape(out[0],DATA_F8_143,{(uint64_t)(n*slots),k,1});
 HabanaKernelParams p{};p.deviceId=DEVICE_ID_GAUDI2;p.maxAvailableTpc=1;p.inputTensorNr=4;p.outputTensorNr=1;p.inputTensors=in;p.outputTensors=out;
 HabanaKernelInstantiation g{};g.inputTensorAccessPattern=ia;g.outputTensorAccessPattern=oa;g.indexSpaceRank=affine?3:2;g.indexSpaceGeometry[0]=blocks*slots;g.indexSpaceGeometry[1]=affine?1:k/128;if(affine)g.indexSpaceGeometry[2]=k/128;
 std::ifstream file(argv[1],std::ios::binary);std::vector<char> elf((std::istreambuf_iterator<char>(file)),{});g.kernel.kernelElf=elf.data();g.kernel.elfSize=elf.size();
 std::vector<int32_t> ids(slots);std::vector<uint8_t> q(words*2*blocks*experts),s(scale_words*2*blocks*experts),lut(256),decoded(n*k*slots,0xef),expected(n*k*slots);
 const uint8_t bits[]={0,48,56,60,64,68,72,76,128,176,184,188,192,196,200,204};
 for(int i=0;i<256;++i)lut[i]=bits[i%16];
 for(int e=0;e<experts;++e)for(int b=0;b<blocks;++b){int qb=(e*blocks+b)*words*2,sb=(e*blocks+b)*scale_words*2;
  for(int col=0;col<256;++col){int channel=80+col%3;s[sb+k*8+col]=channel;for(int group=0;group<k/32;++group)s[sb+group*256+col]=channel-(group+col%5+e+b)%3;}
  for(int row=0;row<k;++row)for(int col=0;col<256;col+=2){int a=(row*7+col+e*3+b)%16,c=(row*7+col+1+e*3+b)%16;q[qb+row*128+col/2]=a|(c<<4);}}
 for(int slot=0;slot<slots;++slot){int e=slot%experts;if(invalid&&slot==slots-1)e=-1;ids[slot]=e;for(int row=0;row<k;++row)for(int col=0;col<n;++col){int b=col/256,c=col%256,nibble=(row*7+c+e*3+b)%16;uint8_t value=0;if(e>=0){int sb=(e*blocks+b)*scale_words*2,delta=(s[sb+row/32*256+c]-s[sb+k*8+c])*8;value=(nibble&7)?uint8_t(bits[nibble]+delta):0;}expected[row*n*slots+slot*n+col]=value;}}
 std::vector<TensorDesc2> descriptors={desc(ids.data(),in[0],2),desc(q.data(),in[1],1),desc(s.data(),in[2],1),desc(lut.data(),in[3],1),desc(decoded.data(),out[0],0)};
 VPEStats stats;TestConfigurations cfg{};cfg.dontTestIndexSpaceMapping=true;cfg.disableShuffleIndexSpacePartition=true;auto result=RunSimulation(p,g,descriptors,stats,e_accessPatternIgnoreMode,{}, {},cfg);
 size_t differences=0;for(size_t i=0;i<decoded.size();++i)if(decoded[i]!=expected[i]){if(differences<4)std::cerr<<i<<" "<<int(decoded[i])<<" "<<int(expected[i])<<"\n";++differences;}
 std::cout<<"{\"affine\":"<<affine<<",\"slots\":"<<slots<<",\"invalid\":"<<invalid<<",\"status\":"<<result<<",\"bytes\":"<<decoded.size()<<",\"differences\":"<<differences<<",\"instructions\":"<<stats.instructionsExecuted<<"}"<<std::endl;return differences?1:0;
}
