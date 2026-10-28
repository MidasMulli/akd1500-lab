// Bounded resident explorer. Inherits the qualified dense learning primitives;
// original study actors remain unchanged. One process owns one fresh program.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
#include <poll.h>
#include <sstream>

template<class T> static void append_live(const std::string& p,const std::vector<T>& v){
 std::ofstream f(p,std::ios::binary|std::ios::app);f.write((const char*)v.data(),v.size()*sizeof(T));need(bool(f),"append raw capture");
}
static std::vector<uint8_t> fixed_live(const std::vector<uint8_t>& x,const std::vector<int8_t>& w,const std::vector<int32_t>& t){
 need(w.size()==x.size()*t.size(),"fixed dimensions");std::vector<uint8_t> y(t.size());
 for(size_t j=0;j<t.size();j++){int v=0;for(size_t k=0;k<x.size();k++)v+=x[k]*w[k*t.size()+j];y[j]=v>t[j];}return y;
}
struct LiveBank{
 int u;CPU cpu;std::vector<uint32_t> initial,infer,learn;std::vector<int32_t> threshold;
 LiveBank(int n,const std::string& p):u(n),cpu(512,n,32,2,load<uint32_t>(p+".ram")),initial(load<uint32_t>(p+".ram")),infer(load<uint32_t>(p+".infer")),learn(load<uint32_t>(p+".learn")),threshold(load<int32_t>(p+".threshold")){}
 std::vector<uint8_t> step(const std::vector<uint8_t>& x,int label,std::vector<int32_t>& p){
  std::vector<uint32_t>b(16);for(int j=0;j<512;j++)b[j/32]|=uint32_t(x[j])<<(j%32);
  p=cpu.forward(b,label<0?0:label,label>=0);std::vector<uint8_t> y(u);for(int j=0;j<u;j++)y[j]=p[j]>threshold[j];return y;
 }
};
int main(int argc,char**argv){
 // Declared before the device so storage survives exception cleanup and the
 // device destructor as well as the normal unprogram path.
 std::vector<uint8_t> program;
 std::unique_ptr<LabDriver> d;akida::HardwareDevicePtr dev;
 try{
  need(argc==4,"mode fixture output");std::string mode=argv[1],f=argv[2],out=argv[3];need(mode=="native"||mode=="cpu","mode");
  std::vector<LiveBank> banks;for(int k=0;k<3;k++)banks.emplace_back(k==2?128:64,f+"/bank"+std::to_string(k));
  std::vector<std::vector<int8_t>> w;std::vector<std::vector<int32_t>> t;
  for(int k=0;k<4;k++){w.push_back(load<int8_t>(f+"/fixed"+std::to_string(k)+".weights"));t.push_back(load<int32_t>(f+"/fixed"+std::to_string(k)+".threshold"));}
  // The SDK retains a non-owning pointer to this buffer until unprogram().
  // Keep it alive across the entire interactive session, not just program().
  akida::HardwareDeviceImpl* impl=nullptr;uint32_t external=0,source=0;
  if(mode=="native"){
   d=std::make_unique<LabDriver>();d->dma_read_min=128;dev=akida::HardwareDevice::create(d.get());
   program=load<uint8_t>(f+"/model.bin");auto info=dev->program(program.data(),program.size());
   need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&info.output_is_dense()&&info.activation_enabled(),"resident program contract");
   dev->toggle_learn(true);impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());need(dev->set_batch_size(4,false)==4,"carry buffers");impl->pipeline(false);
   external=dev->scratch_alloc(128);std::vector<uint8_t>zero(128);d->write(external,zero.data(),128);source=external;
  }
  auto capture=[&](std::vector<uint32_t> v){
   if(dev)akida::dma::dma_config_read(v.data()+2,v.size()-2,v.data(),impl->dma_config(),impl->mem(),d.get(),true);return v;
  };
  for(int k=0;k<3;k++){auto a=capture(banks[k].initial);need(a==banks[k].initial,"initial RAM");dump(out+".initial"+std::to_string(k),a);}
  auto fixed=[&](const char* tag){for(int k=0;k<4;k++){auto a=load<uint32_t>(f+"/fixed"+std::to_string(k)+".ram");auto b=capture(a);need(a==b,"fixed routing RAM");dump(out+".fixed"+std::to_string(k)+"."+tag,b);}};
  fixed("initial");std::vector<uint8_t>last(128);std::vector<int>oldm(3,-1);int count=0;auto active=Clock::now();
  std::cout<<"EXPLORER {\"type\":\"ready\",\"mode\":\""<<mode<<"\"}\n"<<std::flush;
  while(count<512&&Clock::now()-epoch<std::chrono::seconds(150)){
   pollfd pd{0,POLLIN,0};int rc=poll(&pd,1,200);need(rc>=0,"stdin poll");
   if(!rc){if(Clock::now()-active>std::chrono::seconds(45))break;continue;}
   std::string line;if(!std::getline(std::cin,line)||line=="STOP")break;
   active=Clock::now();need(line.size()<128,"bounded command");std::istringstream ss(line);int e[9];for(auto&v:e)need(bool(ss>>v),"nine fields");std::string extra;need(!(ss>>extra),"extra fields");
   int key=e[0],role=e[1],wire=e[5],block=e[6],policy=e[7],candidate=e[8];
   need(key>=0&&key<12&&policy>=0&&policy<5&&candidate>=0&&candidate<2&&role>=0&&role<5&&block>=0&&block<2,"bounded cue");
   need(wire>=0&&wire<2&&(role==1||role==4||wire==0),"outcome label contract");
   for(int k=0;k<3;k++)need(e[2+k]>=-1&&e[2+k]<2,"mode label");need(e[3]==e[4],"equal student clocks");
   need((role==1||role==4)?(e[3]==-1&&e[4]==-1):(role==2?(e[2]==-1&&e[3]<=0):e[2]==-1&&e[3]==-1&&e[4]==-1),"role ownership");
   auto x=last;
   if(role!=2){std::fill(x.begin()+32,x.begin()+64,0);for(int b=0;b<4;b++)for(int j=0;j<4;j++)x[32+b*8+((key>>b)&1)*4+j]=1;}
   std::fill(x.begin()+96,x.end(),0);for(int b=0;b<4;b++)for(int j=0;j<4;j++)x[96+b*8+((policy>>b)&1)*4+j]=1;
   std::fill(x.begin()+72,x.begin()+76,0);x[72+candidate]=1;std::fill(x.begin()+64,x.begin()+72,0);
   x[(role==1||role==4)?65:role==2?69:64]=1;x[68]=role==4;x[70]=role==3;x[71]=block;
   auto want=fixed_live(x,w[0],t[0]);
   for(int k=0;k<3;k++){std::vector<int32_t>p;want=banks[k].step(want,e[2+k],p);append_live(out+".potentials"+std::to_string(k),p);if(k==0){want=fixed_live(want,w[1],t[1]);want=fixed_live(want,w[2],t[2]);}else if(k==1)want=fixed_live(want,w[3],t[3]);}
   std::vector<uint8_t>got=want,actual_x=x;
   if(dev){
    kick_watchdog();for(int k=0;k<3;k++)if(int(e[2+k]>=0)!=oldm[k]){auto&p=e[2+k]>=0?banks[k].learn:banks[k].infer;akida::dma::dma_config_write(p.data(),p.size(),impl->dma_config(),impl->external_mem(),d.get(),true);oldm[k]=int(e[2+k]>=0);}
    // Write external identity/role only. Query responses remain carried in card SRAM.
    if(role!=2)d->write(source+32,x.data()+32,32);
    d->write(source+64,x.data()+64,12);d->write(source+96,x.data()+96,32);
    uint32_t in=source;d->hold_queue();need(dev->enqueue(in,&wire),"enqueue");auto desc=d->last_descriptor;d->release_queue();
    need(desc[1]==in&&desc[7]+32!=in&&((desc[8]>>16)&1023)==uint32_t(wire+1),"descriptor");
    akida::TensorUniquePtr y;auto end=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch()))need(Clock::now()<end,"completion timeout");
    need(y->type()==akida::TensorType::uint8&&y->size()==128,"output contract");got.assign(y->data<uint8_t>(),y->data<uint8_t>()+128);d->read(in,actual_x.data(),128);
    if(role!=2)source=desc[7]+32;
    std::vector<uint32_t>packet(40);d->read(desc[7],packet.data(),160);dump(out+".packet"+std::to_string(count),packet);
    append_live(out+".descriptors",std::vector<uint32_t>(desc.begin(),desc.end()));append_live(out+".input_addresses",std::vector<uint32_t>{in});append_live(out+".output_addresses",std::vector<uint32_t>{desc[7]});
   }
   append_live(out+".inputs",actual_x);append_live(out+".scores",got);append_live(out+".learning_modes",std::vector<int32_t>{int(e[2]>=0),int(e[3]>=0),int(e[4]>=0)});
   for(int k=0;k<3;k++){auto ram=dev?capture(banks[k].initial):banks[k].cpu.ram;append_live(out+".states"+std::to_string(k),ram);need(ram==banks[k].cpu.ram,"captured RAM differs from packed CPU");}
   need(actual_x==x&&got==want,"input or response mismatch");if(role!=2)last=got;count++;
   std::cout<<"EXPLORER {\"type\":\"event\",\"count\":"<<count<<",\"verified\":true}\n"<<std::flush;
  }
  fixed("final");if(dev){dev->scratch_free(external);dev->unprogram();}
  std::cout<<"EXPLORER {\"type\":\"closed\",\"events\":"<<count<<",\"cleanup\":\"unprogrammed\"}\n"<<std::flush;return 0;
 }catch(const std::exception&e){if(dev)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
