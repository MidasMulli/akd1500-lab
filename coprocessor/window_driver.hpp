#pragma once
// Own disjoint transport: card SRAM controls and separately mapped host DDR.
#include "mmio_driver.hpp"
#include "record_range.hpp"
class WindowDriver final:public akida::HardwareDriver{
 LabDriver inner;int fd=-1;void* ddr=MAP_FAILED;uint32_t low=0;uint64_t high=0;
 public:
 static constexpr uint32_t DDR=0xc0000000,EXTENT=0x400000;
 uint64_t guarded_record_attempts=0,guarded_calls=0,ddr_reads=0,ddr_writes=0;
 WindowDriver(){fd=open("/dev/akd1500_0",O_RDWR|O_CLOEXEC);need(fd>=0,"open owned DDR device");ddr=mmap(nullptr,EXTENT,PROT_READ|PROT_WRITE,MAP_SHARED,fd,DDR);need(ddr!=MAP_FAILED,"kernel-accepted host DDR mapping");}
 ~WindowDriver(){if(ddr!=MAP_FAILED)munmap(ddr,EXTENT);if(fd>=0)close(fd);}
 void protect(uint32_t a,size_t n){need(!low&&a>=DDR&&uint64_t(a)+n<=uint64_t(DDR)+EXTENT,"DDR protection extent");low=a;high=uint64_t(a)+n;}
 bool in_ddr(uint32_t a,size_t n)const{return a>=DDR&&uint64_t(a)+n<=uint64_t(DDR)+EXTENT;}
 void read(uint32_t a,void* p,size_t n)const override{if(in_ddr(a,n)){need(a%4==0&&n%4==0,"aligned DDR read");__sync_synchronize();memcpy(p,(char*)ddr+(a-DDR),n);__sync_synchronize();}else inner.read(a,p,n);}
 void write(uint32_t a,const void* p,size_t n)override{
  if(low){guarded_calls++;if(record_ranges_overlap(a,n,low,high)){guarded_record_attempts++;throw std::runtime_error("host write to protected full model");}}
  if(in_ddr(a,n)){need(a%4==0&&n%4==0,"aligned DDR write");memcpy((char*)ddr+(a-DDR),p,n);__sync_synchronize();ddr_writes++;}else inner.write(a,p,n);
 }
 const char* desc()const override{return "akd-lab disjoint card-SRAM control / host-DDR records";}
 uint32_t scratch_memory()const override{return inner.scratch_memory();}
 uint32_t scratch_size()const override{return inner.scratch_size();}
 uint32_t top_level_reg()const override{return inner.top_level_reg();}
 uint32_t akida_visible_memory()const override{return DDR;}
 uint32_t akida_visible_memory_size()const override{return EXTENT;}
};
