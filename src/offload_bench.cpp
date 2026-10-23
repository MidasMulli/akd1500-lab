// Host-offload experiment derived from the verified compact-output instrument.
// The original benchmark source is unchanged; both output contracts use one binary.
#include "mmio_driver.hpp"
#include <akida/dense.h>
#include "dma_events_format.h"
#include "dma_fnp_events.h"
#include "dma_cnp_events.h"
#include "engine/akida_program_info_generated.h"
#include <akida/program_info.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdarg>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <ctime>
using Clock=std::chrono::steady_clock;
static auto epoch=Clock::now();
static int64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
static int64_t cpu_ns(){timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return int64_t(t.tv_sec)*1000000000+t.tv_nsec;}
extern "C" void msleep(uint32_t x){std::this_thread::sleep_for(std::chrono::milliseconds(x));}
extern "C" int64_t time_ms(){return (ns()-std::chrono::duration_cast<std::chrono::nanoseconds>(epoch.time_since_epoch()).count())/1000000;}
extern "C" void kick_watchdog(){need(Clock::now()-epoch<std::chrono::seconds(180),"owned benchmark watchdog");}
extern "C" [[noreturn]] void panic(const char* fmt,...){char b[2048];va_list a;va_start(a,fmt);vsnprintf(b,sizeof(b),fmt,a);va_end(a);throw std::runtime_error(b);}
template<class T>static std::vector<T> load(const std::string& path){std::ifstream f(path,std::ios::binary|std::ios::ate);need(bool(f),"open fixture");auto n=f.tellg();need(n>=0&&size_t(n)%sizeof(T)==0,"fixture shape");std::vector<T> v(size_t(n)/sizeof(T));f.seekg(0);f.read((char*)v.data(),n);need(bool(f),"read fixture");return v;}
template<class T>static void dump(const std::string& path,const std::vector<T>& v){std::ofstream f(path,std::ios::binary);f.write((char*)v.data(),v.size()*sizeof(T));need(bool(f),"save raw output");}
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
int main(int argc,char**argv){
 std::vector<uint8_t> program;std::unique_ptr<LabDriver> driver;akida::HardwareDevicePtr dev;
 try{
  need(argc==8,"mode fixture-prefix output-prefix samples learning cpu-repeats reserved");
  std::string mode=argv[1],f=argv[2],out=argv[3];int count=std::stoi(argv[4]);bool learn=std::stoi(argv[5]);int repeats=std::stoi(argv[6]);
  need((mode=="cpu"||mode=="cpu-fast"||mode=="native"||mode=="resident"||mode=="transport")&&count>0&&count<=4096&&repeats>0&&repeats<=4096,"bounded arguments");
  int n,u,nw,classes;{std::ifstream m(f+".meta");m>>n>>u>>nw>>classes;need(bool(m)&&n%32==0,"metadata");}
  int activated=0,threshold=0;{std::ifstream c(f+".output");need(bool(c),"output contract fixture");c>>activated>>threshold;need(bool(c)&&activated>=0&&activated<=2,"output contract");}
  auto contract=[&](std::vector<int32_t> y){if(activated)for(auto& v:y)v=v>threshold;if(activated==2){std::vector<int32_t> reduced(classes);for(int j=0;j<u;j++)reduced[j/(u/classes)]+=y[j];return reduced;}return y;};
  auto tensor_values=[&](const akida::Tensor& t){int outputs=activated==2?classes:u;need(t.size()==size_t(outputs),"output dimensions");std::vector<int32_t> y(outputs);if(activated==1){need(t.type()==akida::TensorType::uint8,"binary output format");auto v=t.data<uint8_t>();for(int j=0;j<outputs;j++)y[j]=v[j];}else{need(t.type()==akida::TensorType::int32,"potential output format");auto v=t.data<int32_t>();std::copy(v,v+outputs,y.begin());}return y;};
  auto initial=load<uint32_t>(f+".ram");auto inputs=load<uint8_t>(f+".inputs");auto labels=load<int32_t>(f+".labels");need(int(labels.size())>=count&&inputs.size()>=size_t(count*n),"input count");
  auto cpu_inputs=std::ifstream(f+".cpu_inputs").good()?load<uint8_t>(f+".cpu_inputs"):inputs;need(cpu_inputs.size()==inputs.size(),"CPU input fixture shape");
  CPU reference(n,u,nw,classes,initial);std::vector<int32_t> scores;std::vector<int64_t> latencies,input_times,submit_times,fetch_times;uint64_t checksum=0;
  if(mode=="cpu"||mode=="cpu-fast"){
   need(mode!="cpu-fast"||(activated==1&&!learn),"early-exit CPU mask inference only");
   int64_t start=ns(),cstart=cpu_ns();
   for(int r=0;r<repeats;r++){CPU cpu(n,u,nw,classes,initial);for(int i=0;i<count;i++){
    std::vector<uint32_t> x(n/32);for(int k=0;k<n;k++)x[k/32]|=uint32_t(cpu_inputs[i*n+k]!=0)<<(k%32);
    std::vector<int32_t> y;
    if(mode=="cpu-fast"){
     y.assign(u,0);for(int j=0;j<u;j++){int v=0;for(int k=0;k<n/32&&v<=threshold;k++)v+=__builtin_popcount(x[k]&cpu.ram[2+j*cpu.stride+4+k]);y[j]=v>threshold;}
    }else y=contract(cpu.forward(x,labels[i],learn));for(auto v:y)checksum=checksum*1099511628211ULL+uint64_t(v);
    if(r==0)scores.insert(scores.end(),y.begin(),y.end());
    if(r==repeats-1&&i==count-1)reference=std::move(cpu);
   }}
   int64_t wall=ns()-start,cput=cpu_ns()-cstart;dump(out+".scores",scores);dump(out+".ram",reference.ram);
   std::cout<<"{\"status\":\"PASS\",\"mode\":\"cpu\",\"samples\":"<<count<<",\"repeats\":"<<repeats<<",\"wall_ns\":"<<wall<<",\"thread_cpu_ns\":"<<cput<<",\"checksum\":"<<checksum<<"}\n";return 0;
  }
  auto envint=[](const char* key){auto s=std::getenv(key);return s?std::stoi(s):0;};
  const int wait_us=envint("AKD_LAB_WAIT_US"),dma_read_min=envint("AKD_LAB_DMA_READ_MIN"),dma_write_min=envint("AKD_LAB_DMA_WRITE_MIN");
  need(wait_us>=0&&wait_us<=100&&dma_read_min>=0&&dma_write_min>=0,"transport controls");
  uint64_t sleeps=0;auto idle=[&](){if(wait_us){std::this_thread::sleep_for(std::chrono::microseconds(wait_us));sleeps++;}};
  need(repeats==1,"native one stream per fresh program");driver=std::make_unique<LabDriver>();dev=akida::HardwareDevice::create(driver.get());program=load<uint8_t>(f+".bin");auto info=dev->program(program.data(),program.size());
  need(info.number_of_passes()==1&&info.can_learn()&&info.activation_enabled()==(activated==1)&&info.output_is_dense(),"single-pass matched dense learner");
  if(mode=="transport"){
   auto addr=dev->scratch_alloc(2048);size_t checked=0;
   for(size_t bytes:{128u,512u,2048u}){
    std::vector<uint32_t> x(bytes/4),got(bytes/4);for(size_t i=0;i<x.size();i++)x[i]=0x5a96c30fu^uint32_t(i*2654435761u);
    driver->dma_read_min=driver->dma_write_min=0;driver->write(addr,x.data(),bytes);
    driver->dma_read_min=128;driver->read(addr,got.data(),bytes);need(got==x,"DMA read differs from MMIO write");
    for(auto& w:x)w^=0xffffffffu;driver->dma_read_min=0;driver->dma_write_min=128;driver->write(addr,x.data(),bytes);driver->read(addr,got.data(),bytes);need(got==x,"MMIO read differs from DMA write");checked+=bytes;
   }
   driver->dma_read_min=driver->dma_write_min=0;dev->scratch_free(addr);dev->unprogram();
   std::cout<<"{\"status\":\"PASS\",\"mode\":\"transport\",\"native_forwards\":0,\"bytes_per_direction_verified\":"<<checked<<",\"DMA_reads\":"<<driver->dma_reads<<",\"DMA_writes\":"<<driver->dma_writes<<",\"wait_us\":"<<wait_us<<",\"sleep_calls\":"<<sleeps<<",\"DMA_read_calls\":"<<driver->dma_reads<<",\"DMA_write_calls\":"<<driver->dma_writes<<",\"DMA_read_bytes\":"<<driver->dma_read_bytes<<",\"DMA_write_bytes\":"<<driver->dma_write_bytes<<",\"cleanup\":\"unprogrammed\"}\n";return 0;
  }
  driver->dma_read_min=dma_read_min;driver->dma_write_min=dma_write_min;
  const bool dense_input=info.input_is_dense();driver->queue_base=LabDriver::REG+(dense_input?0x28000:0x20000);
  auto sparse_tensor=[&](int index)->akida::SparseUniquePtr{akida::dma::wbuffer w;w.reserve(n*2);bool cnp=info.inputs_type()==akida::fb::IoType_cnp_sparse;need(cnp||info.inputs_type()==akida::fb::IoType_fnp_sparse,"reviewed sparse input format");for(int k=0;k<n;k++)if(inputs[index*n+k]){if(cnp){need(k<2048,"CNP feature range");w.push_back(0);w.push_back(uint32_t(k)|(1u<<16));}else{w.push_back(uint32_t(k));w.push_back(0x80000001u);}}need(!w.empty(),"nonempty sparse event fixture");if(cnp)return std::make_unique<akida::DmaCnpEvents>(info.input_dims(),std::move(w));return std::make_unique<akida::DmaFnpEvents>(info.input_dims(),std::move(w));};

  dev->toggle_learn(learn);
  if(mode=="resident"){
   // One spare descriptor makes each published producer index distinguishable
   // from the preceding index after a withheld full-ring burst.
   constexpr int batch=8,queue=batch+1;need(count%batch==0,"resident count multiple of eight");need(dev->set_batch_size(queue,!dense_input)==queue,"resident queue capacity with spare slot");
   std::vector<uint32_t> addresses;if(dense_input)for(int j=0;j<batch;j++)addresses.push_back(dev->scratch_alloc(n));
   int64_t setup=0,execution=0,drain=0,reference_time=0,resident_cpu=0;uint64_t polls=0;auto initial_reads=driver->read_bytes,initial_writes=driver->write_bytes;auto dr0=driver->dma_reads,dw0=driver->dma_writes,drb0=driver->dma_read_bytes,dwb0=driver->dma_write_bytes;
   for(int first=0;first<count;first+=batch){
    kick_watchdog();auto cpu_start=cpu_ns();auto t=ns();
    if(dense_input)for(int j=0;j<batch;j++)driver->write(addresses[j],inputs.data()+(first+j)*n,n);
    driver->hold_queue();
    for(int j=0;j<batch;j++){int32_t label=labels[first+j];bool accepted;if(dense_input)accepted=dev->enqueue(addresses[j],learn?&label:nullptr);else{auto tensor=sparse_tensor(first+j);accepted=dev->enqueue(*tensor,learn?&label:nullptr);}need(accepted,"resident enqueue");}
    auto target=driver->last_descriptor[0]>>16;need(target!=0,"captured final descriptor job");setup+=ns()-t;
    auto before_writes=driver->writes;t=ns();driver->release_queue();auto stop=Clock::now()+std::chrono::seconds(2);
    uint32_t status=0;
    while(((status=driver->read32(driver->queue_base+0x10))>>16)!=target){
     polls++;idle();
     if(Clock::now()>=stop){
      std::cerr<<"resident diagnostic: dense="<<info.input_is_dense()<<" target="<<target<<" observed="<<(status>>16)<<" descriptor_word0="<<driver->last_descriptor[0]<<" published_queue="<<driver->shadow<<"\n";
      throw std::runtime_error("resident completion deadline");
     }
    }
    execution+=ns()-t;need(driver->writes==before_writes+1,"host wrote after queue release");
    t=ns();std::vector<std::vector<int32_t>> got;
    for(int j=0;j<batch;j++){auto y=dev->fetch();need(bool(y),"resident output drain");got.push_back(tensor_values(*y));scores.insert(scores.end(),got.back().begin(),got.back().end());}
    drain+=ns()-t;resident_cpu+=cpu_ns()-cpu_start;t=ns();
    for(int j=0;j<batch;j++){
     std::vector<uint32_t> x(n/32);for(int k=0;k<n;k++)x[k/32]|=uint32_t(cpu_inputs[(first+j)*n+k]!=0)<<(k%32);
     auto want=contract(reference.forward(x,labels[first+j],learn));need(want==got[j],"resident scores differ from sequential CPU");
    }reference_time+=ns()-t;
   }
   auto transferred_reads=driver->read_bytes-initial_reads,transferred_writes=driver->write_bytes-initial_writes;auto dr=driver->dma_reads-dr0,dw=driver->dma_writes-dw0,drb=driver->dma_read_bytes-drb0,dwb=driver->dma_write_bytes-dwb0;
   if(!learn)dev->toggle_learn(true);std::vector<uint32_t> final(dev->learn_mem_size());dev->learn_mem(final.data());need(final==reference.ram,"resident full RAM differs");
   dump(out+".scores",scores);dump(out+".ram",final);for(auto a:addresses)dev->scratch_free(a);dev->unprogram();
   std::cout<<"{\"status\":\"PASS\",\"mode\":\"resident\",\"samples\":"<<count<<",\"repeats\":1,\"batch\":8,\"queue_slots\":9,\"setup_ns\":"<<setup<<",\"release_to_observed_completion_ns\":"<<execution<<",\"output_drain_ns\":"<<drain<<",\"wall_ns\":"<<setup+execution+drain<<",\"thread_cpu_ns\":"<<resident_cpu<<",\"transaction_read_bytes\":"<<transferred_reads<<",\"transaction_write_bytes\":"<<transferred_writes<<",\"wait_us\":"<<wait_us<<",\"sleep_calls\":"<<sleeps<<",\"dense_input\":"<<dense_input<<",\"transaction_DMA_reads\":"<<dr<<",\"transaction_DMA_writes\":"<<dw<<",\"transaction_DMA_read_bytes\":"<<drb<<",\"transaction_DMA_write_bytes\":"<<dwb<<",\"verification_cpu_ns\":"<<reference_time<<",\"polls\":"<<polls<<",\"host_writes_during_execution\":0,\"exact_scores_and_full_RAM\":true,\"cleanup\":\"unprogrammed\"}\n";return 0;
  }
  need(dev->set_batch_size(1,!dense_input)==1,"one sample queue");auto addr=dense_input?dev->scratch_alloc(n):0;
  int64_t wall=0,cput=0;auto initial_reads=driver->read_bytes,initial_writes=driver->write_bytes;auto dr0=driver->dma_reads,dw0=driver->dma_writes,drb0=driver->dma_read_bytes,dwb0=driver->dma_write_bytes;
  for(int i=0;i<count;i++){
   kick_watchdog();int64_t start=ns(),cstart=cpu_ns();if(dense_input)driver->write(addr,inputs.data()+i*n,n);auto loaded=ns();
   int32_t label=labels[i];bool accepted;if(dense_input)accepted=dev->enqueue(addr,learn?&label:nullptr);else{auto tensor=sparse_tensor(i);accepted=dev->enqueue(*tensor,learn?&label:nullptr);}need(accepted,"native enqueue");auto submitted=ns();akida::TensorUniquePtr y;
   auto stop=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch())){need(Clock::now()<stop,"native completion deadline");idle();}
   auto fetched=ns();int64_t ct=cpu_ns()-cstart,dt=fetched-start;wall+=dt;cput+=ct;latencies.push_back(dt);
   input_times.push_back(loaded-start);submit_times.push_back(submitted-loaded);fetch_times.push_back(fetched-submitted);
   auto values=tensor_values(*y);auto yp=values.data();scores.insert(scores.end(),values.begin(),values.end());
   std::vector<uint32_t> x(n/32);for(int k=0;k<n;k++)x[k/32]|=uint32_t(cpu_inputs[i*n+k]!=0)<<(k%32);
   auto expected=contract(reference.forward(x,label,learn));need(std::equal(expected.begin(),expected.end(),yp),"CPU/native exact potential mismatch");
   if(std::getenv("AKD_LAB_SNAPSHOTS")){
    need(learn,"per-step snapshot evidence run requires learning enabled");
    std::vector<uint32_t> captured(dev->learn_mem_size());dev->learn_mem(captured.data());
    need(captured==reference.ram,"per-step full learning RAM mismatch");dump(out+".snapshot_"+std::to_string(i)+".ram",captured);
   }
  }
  auto transferred_reads=driver->read_bytes-initial_reads,transferred_writes=driver->write_bytes-initial_writes;auto dr=driver->dma_reads-dr0,dw=driver->dma_writes-dw0,drb=driver->dma_read_bytes-drb0,dwb=driver->dma_write_bytes-dwb0;
  if(!learn)dev->toggle_learn(true); // RAM readback requires learning enabled; no forward follows.
  auto size=dev->learn_mem_size();std::vector<uint32_t> final(size);dev->learn_mem(final.data());need(final.size()==initial.size(),"native RAM length");
  // Addresses/routing auxiliary words are fixed; learned weights/thresholds must match.
  need(final==reference.ram,"complete native learning RAM differs from independent CPU model");
  dump(out+".scores",scores);dump(out+".ram",final);dump(out+".latency_ns",latencies);
  dump(out+".input_ns",input_times);dump(out+".submit_ns",submit_times);dump(out+".fetch_ns",fetch_times);
  auto reads=driver->reads,writes=driver->writes;if(dense_input)dev->scratch_free(addr);dev->unprogram();
  std::cout<<"{\"status\":\"PASS\",\"mode\":\"native\",\"samples\":"<<count<<",\"repeats\":1,\"wall_ns\":"<<wall<<",\"thread_cpu_ns\":"<<cput<<",\"transaction_read_bytes\":"<<transferred_reads<<",\"transaction_write_bytes\":"<<transferred_writes<<",\"driver_reads\":"<<reads<<",\"driver_writes\":"<<writes<<",\"dense_input\":"<<dense_input<<",\"wait_us\":"<<wait_us<<",\"sleep_calls\":"<<sleeps<<",\"transaction_DMA_reads\":"<<dr<<",\"transaction_DMA_writes\":"<<dw<<",\"transaction_DMA_read_bytes\":"<<drb<<",\"transaction_DMA_write_bytes\":"<<dwb<<",\"exact_scores_and_full_RAM\":true,\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception& e){if(dev)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
