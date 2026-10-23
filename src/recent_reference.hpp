#pragma once
// Separate task-level CPU baseline: recent labeled examples, not AKD simulation.
#include <deque>
#include <vector>
#include <cstdint>
#include <limits>
struct RecentReference {
 struct Entry {std::vector<uint32_t> x;int label;};
 std::deque<Entry> banks[2];
 int forward(const std::vector<uint32_t>& x,int label) {
  bool first=false;for(size_t k=0;k<x.size()/2;k++)first|=x[k]!=0;
  int court=first?0:1;auto& bank=banks[court];int prediction=0,best=std::numeric_limits<int>::max();
  for(auto p=bank.rbegin();p!=bank.rend();++p){int d=0;for(size_t k=0;k<x.size();k++)d+=__builtin_popcount(x[k]^p->x[k]);if(d<best){best=d;prediction=p->label;}}
  if(label!=4){bank.push_back({x,label});if(bank.size()>64)bank.pop_front();}
  return prediction;
 }
};
