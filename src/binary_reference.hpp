#pragma once
// Shared verbatim binary-association reference from learning_bench.cpp.
#include "mmio_driver.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
static int mix(int p){return ((p>>2)&7)|((p&3)<<3);}
struct CPU {
 int n,u,nw,classes,stride,rot=0,cursor=0;std::vector<uint32_t> ram;
 CPU(int nn,int uu,int ww,int cc,std::vector<uint32_t> r):n(nn),u(uu),nw(ww),classes(cc),stride(4+nn/32),ram(std::move(r)){}
 std::vector<int32_t> forward(const std::vector<uint32_t>& x,int label,bool learning){
  int words=n/32;std::vector<int32_t> p(u);
  for(int j=0;j<u;j++)for(int k=0;k<words;k++)p[j]+=__builtin_popcount(x[k]&ram[2+j*stride+4+k]);
  if(!learning)return p;
  for(auto v:p)rot=(rot+mix(v))&31;
  int lo=label*(u/classes),hi=lo+u/classes,winner=-1;
  for(int j=lo;j<hi;j++){int t=ram[2+j*stride+3]>>16;if(p[j]>std::max(0,t)&&(winner<0||p[j]>p[winner]))winner=j;}
  if(winner<0)return p;
  int base=2+winner*stride;int sw=ram[base+3]&65535;int rc=0,ic=0;
  for(int k=0;k<words;k++){rc+=__builtin_popcount(ram[base+4+k]&~x[k]);ic+=__builtin_popcount(~ram[base+4+k]&x[k]);}
  int count=std::min({sw,rc,ic}),nr=0,ni=0,a=cursor,burst=0,scan=0;
  if(!count)return p;
  while(nr<count||ni<count){
   auto& w=ram[base+4+a];uint32_t r=w&~x[a],in=~w&x[a];
   auto take=[&](uint32_t candidates){for(int b=0;b<32;b++){int bit=(rot+b)&31;if(candidates&(uint32_t(1)<<bit))return bit;}return -1;};
   if(nr<count&&r){int b=take(r);w&=~(uint32_t(1)<<b);nr++;}
   if(ni<count&&in){int b=take(in);w|=uint32_t(1)<<b;ni++;}
   rot=(rot+mix(p.back()))&31;scan++;
   if(nr==count&&ni==count)break;
   a=(a+1)%words;burst=a==0?0:(burst+1)%4;need(scan<=words*(count+1),"CPU selector terminates");
  }
  cursor=(a+1)%words;rot=(rot+std::min(3-burst,words-1-a)*mix(p.back()))&31;
  // Frozen parameters: initial .5, minimum .25, decay .125, competition .25.
  int init=int(std::floor(nw*.5+.5));int news=std::max(int(std::ceil(nw*.25)),sw-int(std::ceil(init*.125)));
  int newt=std::min(p[winner],int(std::floor(nw*.75+.5)));ram[base+3]=(uint32_t(newt)<<16)|uint32_t(news);
  return p;
 }
};
