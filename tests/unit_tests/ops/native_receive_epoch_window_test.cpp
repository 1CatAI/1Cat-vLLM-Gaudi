// SPDX-License-Identifier: Apache-2.0
#include "tools/communication/native_receive_epoch_window.h"
#include <cassert>
#include <iostream>
int main() {
 using Window=hcl::NativeReceiveEpochWindow;
 unsigned checks=0;
 auto reject=[&](size_t total,size_t split,bool active,size_t count){
  try{Window::prepare(total,split,active,count);assert(false);}catch(const std::invalid_argument&){++checks;}
 };
 reject(0,0,false,0); reject(256,256,false,1); reject(256,0,true,256);
 for(size_t total=2;total<=256;++total){
  auto full=Window::prepare(total,0,false,total);assert(full.offset==0&&!full.nextActive&&full.part==Window::Part::Full);++checks;
  for(size_t split=1;split<total;++split){
   full=Window::prepare(total,split,false,total);assert(full.offset==0&&!full.nextActive&&full.part==Window::Part::Full);++checks;
   auto prefix=Window::prepare(total,split,false,split);assert(prefix.offset==0&&prefix.nextActive&&prefix.part==Window::Part::Prefix);++checks;
   auto suffix=Window::prepare(total,split,prefix.nextActive,total-split);
   assert(suffix.offset==split&&!suffix.nextActive&&suffix.part==Window::Part::Suffix);++checks;
   for(size_t i=0;i<total-split;++i)assert(suffix.offset+i>=split&&suffix.offset+i<total);
   reject(total,split,true,total);
   for(size_t count:{size_t(0),total+1}){reject(total,split,false,count);reject(total,split,true,count);}
  }
 }
 // Production 160-call replay: split count changes no destination or phase offset.
 constexpr size_t total=39,split=31;
 bool active=false;
 for(unsigned round=0;round<160;++round){
  const auto p=Window::prepare(total,split,active,split);active=p.nextActive;
  const auto s=Window::prepare(total,split,active,total-split);active=s.nextActive;
  assert(!active&&p.offset==0&&s.offset==31);++checks;
 }
 std::cout<<checks<<" checks passed\n";
}
