#pragma once
#include <cstdint>
#include <limits>
// Half-open intervals; overflow is rejected rather than wrapping past a guard.
inline bool record_ranges_overlap(uint32_t address,uint64_t size,uint32_t low,uint64_t high){
 if(!size)return false;
 if(size>std::numeric_limits<uint64_t>::max()-address)return true;
 return uint64_t(address)<high&&uint64_t(address)+size>low;
}
