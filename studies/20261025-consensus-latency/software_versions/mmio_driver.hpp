#pragma once
// Own transport adapter. BrainChip's runtime is supplied separately by the user.
#include <akida/hardware_device.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
inline void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
class LabDriver final : public akida::HardwareDriver {
 int fd=-1;void* regs=MAP_FAILED;void* ram=MAP_FAILED;
 public:
 static constexpr uint32_t REG=0xfcc00000,RAM=0x20000000,SIZE=0x100000;
 mutable uint64_t reads=0,writes=0,read_bytes=0,write_bytes=0;
 mutable uint64_t dma_reads=0,dma_writes=0,dma_read_bytes=0,dma_write_bytes=0;
 size_t dma_read_min=0,dma_write_min=0;
 bool held=false;uint32_t shadow=0,queue_base=REG+0x28000;std::array<uint32_t,16> last_descriptor{};
 LabDriver(){fd=open("/dev/akd1500_0",O_RDWR|O_CLOEXEC);need(fd>=0,"open AKD device");
  regs=mmap(nullptr,SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fd,REG);
  ram=mmap(nullptr,SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fd,RAM);
  need(regs!=MAP_FAILED&&ram!=MAP_FAILED,"map known device apertures");}
 ~LabDriver(){if(regs!=MAP_FAILED)munmap(regs,SIZE);if(ram!=MAP_FAILED)munmap(ram,SIZE);if(fd>=0)close(fd);}
 volatile uint32_t* ptr(uint32_t a,size_t n)const{
  need(a%4==0&&n%4==0,"aligned MMIO");
  if(a>=REG&&uint64_t(a)+n<=uint64_t(REG)+SIZE)return (volatile uint32_t*)((char*)regs+a-REG);
  if(a>=RAM+256&&uint64_t(a)+n<=uint64_t(RAM)+SIZE)return (volatile uint32_t*)((char*)ram+a-RAM);
  throw std::runtime_error("MMIO outside known apertures");}
 void read(uint32_t a,void* dst,size_t n)const override{if(held&&a==queue_base+4&&n==4){memcpy(dst,&shadow,4);return;}auto p=ptr(a,n);__sync_synchronize();
  if(dma_read_min&&n>=dma_read_min&&a>=RAM+256&&a<RAM+SIZE){need(pread(fd,dst,n,off_t(a))==ssize_t(n),"bounded kernel DMA read failed/short; no fallback");dma_reads++;dma_read_bytes+=n;reads++;read_bytes+=n;return;}
  for(size_t i=0;i<n/4;i++){uint32_t v=p[i];memcpy((char*)dst+4*i,&v,4);}__sync_synchronize();reads++;read_bytes+=n;}
 void write(uint32_t a,const void* src,size_t n)override{if(held&&a==queue_base+4&&n==4){memcpy(&shadow,src,4);return;}auto p=ptr(a,n);__sync_synchronize();
  if(dma_write_min&&n>=dma_write_min&&a>=RAM+256&&a<RAM+SIZE){need(pwrite(fd,src,n,off_t(a))==ssize_t(n),"bounded kernel DMA write failed/short; no fallback");dma_writes++;dma_write_bytes+=n;writes++;write_bytes+=n;return;}
  for(size_t i=0;i<n/4;i++){uint32_t v;memcpy(&v,(char*)src+4*i,4);p[i]=v;}__sync_synchronize();writes++;write_bytes+=n;
  if(held&&a>=RAM+256&&a<RAM+SIZE&&(n==64||n==32)){last_descriptor.fill(0);memcpy(last_descriptor.data(),src,n);}}
 void hold_queue(){need(!held,"queue not already held");shadow=read32(queue_base+4);held=true;}
 void release_queue(){need(held,"queue held");held=false;write32(queue_base+4,shadow);}
 const char* desc()const override{return "akd1500-lab bounded PCIe MMIO";}
 uint32_t scratch_memory()const override{return RAM+256;}
 uint32_t scratch_size()const override{return SIZE-256;}
 uint32_t top_level_reg()const override{return REG;}
 uint32_t akida_visible_memory()const override{return 0;}
 uint32_t akida_visible_memory_size()const override{return 0;}
};
