#pragma once
// Intercepts every write issued through this experiment's HardwareDriver.
// This is a harness-observation guard, not hardware access control or a claim
// about arbitrary kernel/device writers. Initial program writes precede protect.
#include "mmio_driver.hpp"
#include "record_range.hpp"
class RecordWriteGuard final: public akida::HardwareDriver {
 LabDriver inner;
 uint32_t low=0;uint64_t high=0;
 public:
 uint64_t attempts=0,guarded_writes=0;
 void protect(uint32_t a,size_t n){need(!low&&n&&a>=LabDriver::RAM+256&&uint64_t(a)+n<=uint64_t(LabDriver::RAM)+LabDriver::SIZE,"owned record guard domain");low=a;high=uint64_t(a)+n;}
 void read(uint32_t a,void* p,size_t n)const override{inner.read(a,p,n);}
 void write(uint32_t a,const void* p,size_t n)override{
  if(low){guarded_writes++;if(record_ranges_overlap(a,n,low,high)){attempts++;throw std::runtime_error("host attempted to write protected native records");}}
  inner.write(a,p,n);
 }
 const char* desc()const override{return "akd-lab record write observer; bounded PCIe";}
 uint32_t scratch_memory()const override{return inner.scratch_memory();}
 uint32_t scratch_size()const override{return inner.scratch_size();}
 uint32_t top_level_reg()const override{return inner.top_level_reg();}
 uint32_t akida_visible_memory()const override{return inner.akida_visible_memory();}
 uint32_t akida_visible_memory_size()const override{return inner.akida_visible_memory_size();}
};
