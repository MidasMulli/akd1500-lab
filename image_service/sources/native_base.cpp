#include <algorithm>
#include <array>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "akida/hardware_device.h"
#include "akida/program_info.h"
#include "akida/dense.h"
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
#include "engine/dma_config_ops.h"
#include "engine/akida_program_info_generated.h"
#include "engine/flatbuffers_utils.h"

using Clock=std::chrono::steady_clock;
static auto origin=Clock::now();
static long long us(){return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-origin).count();}
extern "C" void msleep(uint32_t n){std::this_thread::sleep_for(std::chrono::milliseconds(n));}
extern "C" int64_t time_ms(){return us()/1000;}
extern "C" void kick_watchdog(){if(us()>120000000)throw std::runtime_error("process watchdog");}
extern "C" [[noreturn]] void panic(const char* fmt,...){char b[2048];va_list a;va_start(a,fmt);vsnprintf(b,sizeof(b),fmt,a);va_end(a);throw std::runtime_error(b);}
static void require(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
static std::string hexbytes(const void*p,size_t n){std::ostringstream s;s<<std::hex<<std::setfill('0');for(size_t i=0;i<n;++i)s<<std::setw(2)<<unsigned(static_cast<const uint8_t*>(p)[i]);return s.str();}
static std::vector<uint8_t> file(const std::string&p){std::ifstream f(p,std::ios::binary);require(bool(f),"open program");return {std::istreambuf_iterator<char>(f),{}};}

class Driver final:public akida::HardwareDriver {
 public:
  static constexpr uint32_t REG=0xfcc00000,SRAM=0x20000000,SIZE=0x100000,QREG=REG+0x28004;
  int fd=-1;void* regs=MAP_FAILED;void* sram=MAP_FAILED;
  mutable std::ofstream trace;mutable uint64_t accesses=0;
  bool hold=false,capture=false;uint32_t shadow=0;std::vector<std::array<uint32_t,16>> descriptors;std::vector<uint32_t> descriptor_addresses;
  explicit Driver(const std::string&p):trace(p){
    require(bool(trace),"trace open");trace<<"us,operation,address,size,data_hex\n";
    fd=open("/dev/akd1500_0",O_RDWR|O_CLOEXEC);require(fd>=0,"device open");
    regs=mmap(nullptr,SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fd,REG);
    sram=mmap(nullptr,SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fd,SRAM);
    require(regs!=MAP_FAILED&&sram!=MAP_FAILED,"mmap apertures");
  }
  ~Driver(){if(regs!=MAP_FAILED)munmap(regs,SIZE);if(sram!=MAP_FAILED)munmap(sram,SIZE);if(fd>=0)close(fd);}
  volatile uint32_t* ptr(uint32_t a,size_t n)const{
    require(a%4==0&&n%4==0,"unaligned access");
    if(a>=REG&&uint64_t(a)+n<=uint64_t(REG)+SIZE)return reinterpret_cast<volatile uint32_t*>(static_cast<uint8_t*>(regs)+(a-REG));
    if(a>=SRAM+256&&uint64_t(a)+n<=uint64_t(SRAM)+SIZE)return reinterpret_cast<volatile uint32_t*>(static_cast<uint8_t*>(sram)+(a-SRAM));
    throw std::runtime_error("out-of-aperture access");
  }
  void log(const char* op,uint32_t a,const void*p,size_t n)const{trace<<us()<<','<<op<<','<<a<<','<<n<<','<<hexbytes(p,n)<<'\n';}
  void mark(const std::string&s)const{trace<<us()<<",MARK,0,0,"<<s<<'\n';trace.flush();}
  void read(uint32_t a,void*d,size_t n)const override{
    if(hold&&a==QREG&&n==4){memcpy(d,&shadow,4);log("SHADOW_READ",a,d,n);return;}
    auto p=ptr(a,n);__sync_synchronize();for(size_t i=0;i<n/4;++i){uint32_t w=p[i];memcpy(static_cast<uint8_t*>(d)+4*i,&w,4);}__sync_synchronize();++accesses;log("READ",a,d,n);
  }
  void write(uint32_t a,const void*d,size_t n)override{
    if(hold&&a==QREG&&n==4){memcpy(&shadow,d,4);log("DEFER_WRITE",a,d,n);return;}
    auto p=ptr(a,n);__sync_synchronize();for(size_t i=0;i<n/4;++i){uint32_t w;memcpy(&w,static_cast<const uint8_t*>(d)+4*i,4);p[i]=w;}__sync_synchronize();++accesses;log("WRITE",a,d,n);
    if(capture&&a>=SRAM+256&&a<SRAM+SIZE&&n==64){std::array<uint32_t,16> v;memcpy(v.data(),d,64);descriptors.push_back(v);descriptor_addresses.push_back(a);}
  }
  void begin(){require(!hold,"already held");shadow=read32(QREG);descriptors.clear();descriptor_addresses.clear();hold=true;capture=true;mark("QUEUE_HELD");}
  void release(){require(hold,"not held");hold=false;capture=false;write32(QREG,shadow);mark("QUEUE_RELEASED");}
  const char*desc()const override{return "AKD1500 native MMIO; chip SRAM only";}
  uint32_t scratch_memory()const override{return SRAM+256;}
  uint32_t scratch_size()const override{return SIZE-256;}
  uint32_t top_level_reg()const override{return REG;}
  uint32_t akida_visible_memory()const override{return 0;}
  uint32_t akida_visible_memory_size()const override{return 0;}
};

static std::array<int,8> decode(akida::Tensor& y){
  require(dynamic_cast<akida::Dense*>(&y)!=nullptr&&y.size()==8,"dense output dimensions");
  std::array<int,8> out{};
  if(y.type()==akida::TensorType::int32){auto p=y.data<int32_t>();std::copy(p,p+8,out.begin());}
  else {require(akida::tensor_type_size(y.type())==1,"output type");auto p=y.data<uint8_t>();std::copy(p,p+8,out.begin());}
  return out;
}
template<class A>static void arrayjson(const A&a){std::cout<<'[';for(size_t i=0;i<a.size();++i){if(i)std::cout<<',';std::cout<<+a[i];}std::cout<<']';}
static akida::TensorUniquePtr fetch(akida::HardwareDevice& d){auto start=Clock::now();for(;;){auto y=d.fetch();if(y)return y;require(Clock::now()-start<std::chrono::seconds(2),"output timeout");msleep(1);}}
static void metadata(const std::string&name,const akida::ProgramInfo&p){
  std::cout<<"{\"kind\":\"program\",\"name\":\""<<name<<"\",\"input_dense\":"<<p.input_is_dense()<<",\"output_dense\":"<<p.output_is_dense()<<",\"activated\":"<<p.activation_enabled()<<",\"passes\":"<<p.number_of_passes()<<",\"can_learn\":"<<p.can_learn()<<",\"program_data_bytes\":"<<p.program_data_required_memory()<<",\"fnp2_bytes\":"<<p.fnp2_required_memory()<<"}\n";
}
