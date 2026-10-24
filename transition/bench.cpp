// Changed transitions from native class-addressed learning; finite SRAM chains.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
#include "engine/dma.h"
class TransitionDriver final:public akida::HardwareDriver {
 public:LabDriver io;uint32_t descriptor_address=0;
 void read(uint32_t a,void*p,size_t n)const override{io.read(a,p,n);}
 void write(uint32_t a,const void*p,size_t n)override{io.write(a,p,n);if(io.held&&n==64&&a>=LabDriver::RAM+256&&a<LabDriver::RAM+LabDriver::SIZE)descriptor_address=a;}
 const char*desc()const override{return "learned transition / bounded output dependencies";}
 uint32_t scratch_memory()const override{return io.scratch_memory();}uint32_t scratch_size()const override{return io.scratch_size();}uint32_t top_level_reg()const override{return io.top_level_reg();}
 uint32_t akida_visible_memory()const override{return 0;}uint32_t akida_visible_memory_size()const override{return 0;}
};
int main(int argc,char**argv){
 std::unique_ptr<TransitionDriver>d;akida::HardwareDevicePtr dev;std::vector<uint8_t>program;bool programmed=false;
 try{
  need(argc==6,"mode fixture arm horizon output");std::string mode=argv[1],f=argv[2],arm=argv[3],out=argv[5];int h=std::stoi(argv[4]);
  need((mode=="native"||mode=="cpu")&&(arm=="native"||arm=="frozen"||arm=="disconnected")&&(h==0||h==1||h==2||h==4||h==32),"frozen arguments");
  constexpr int n=256,u=8,events=26;auto initial=load<uint32_t>(f+"/initial.ram");auto codes=load<uint8_t>(f+"/codes"),partial=load<uint8_t>(f+"/partial");auto inputs=load<uint8_t>(f+"/"+arm+"_"+std::to_string(h)+".inputs");auto labels=load<int32_t>(f+"/"+arm+"_"+std::to_string(h)+".labels");auto old=load<int32_t>(f+"/old");
  need(initial.size()==98&&codes.size()==8*n&&partial.size()==8*n&&inputs.size()==events*n&&labels.size()==events,"fixture dimensions");
  auto packed=[&](const uint8_t*a){std::vector<uint32_t>b(n/32);for(int k=0;k<n;k++)b[k/32]|=uint32_t(a[k])<<(k%32);return b;};
  auto decode=[&](const std::vector<int32_t>&p){std::vector<uint8_t>y(n);for(int j=0;j<u;j++)if(p[j]>24)for(int k=0;k<n;k++)y[k]|=codes[j*n+k];return y;};
  CPU ref(n,u,32,u,initial);std::vector<uint8_t>expected,rollout;std::vector<uint32_t>states;std::vector<int32_t>potentials;
  for(int i=0;i<events;i++){auto p=ref.forward(packed(inputs.data()+i*n),std::max(0,labels[i]),labels[i]>=0);auto y=decode(p);expected.insert(expected.end(),y.begin(),y.end());states.insert(states.end(),ref.ram.begin(),ref.ram.end());potentials.insert(potentials.end(),p.begin(),p.end());}
  for(int start=0;start<u;start++){std::vector<uint8_t>x(partial.begin()+start*n,partial.begin()+(start+1)*n);for(int t=0;t<h;t++){x=decode(ref.forward(packed(x.data()),0,false));rollout.insert(rollout.end(),x.begin(),x.end());}}
  if(mode=="cpu"){
   // Exact-rule CPU and the stronger task-specific table both include dense decode.
   constexpr int repeats=2000;uint64_t check=0;auto t=ns(),ct=cpu_ns();
   for(int r=0;r<repeats;r++){CPU c(n,u,32,u,initial);for(int i=0;i<events;i++){auto y=decode(c.forward(packed(inputs.data()+i*n),std::max(0,labels[i]),labels[i]>=0));for(auto v:y)check=check*1099511628211ULL+v;}for(int s=0;s<u;s++){std::vector<uint8_t>x(partial.begin()+s*n,partial.begin()+(s+1)*n);for(int k=0;k<h;k++){x=decode(c.forward(packed(x.data()),0,false));for(auto v:x)check=check*1099511628211ULL+v;}}need(c.ram==ref.ram,"exact CPU repeated final RAM");}
   auto exact_wall=ns()-t,exact_cpu=cpu_ns()-ct;std::vector<int>feature_state(n,-1);for(int k=0;k<n;k++){int count=0,s=-1;for(int j=0;j<u;j++)if(codes[j*n+k]){count++;s=j;}if(count==1)feature_state[k]=s;}
   t=ns();ct=cpu_ns();for(int r=0;r<repeats;r++){auto table=old;auto step=[&](const uint8_t*x,int label){int state=-1;for(int k=0;k<n;k++)if(x[k]&&feature_state[k]>=0){state=feature_state[k];break;}need(state>=0,"table input recognized");std::vector<uint8_t>y(codes.begin()+table[state]*n,codes.begin()+(table[state]+1)*n);if(label>=0)table[state]=label;for(auto v:y)check=check*1099511628211ULL+v;return y;};for(int i=0;i<events;i++){auto y=step(inputs.data()+i*n,labels[i]);if((i>=8&&i<16)||i>=18)need(std::equal(y.begin(),y.end(),expected.begin()+i*n),"task CPU query equivalence");}for(int s=0;s<u;s++){std::vector<uint8_t>x(partial.begin()+s*n,partial.begin()+(s+1)*n);for(int k=0;k<h;k++){x=step(x.data(),-1);need(std::equal(x.begin(),x.end(),rollout.begin()+(s*h+k)*n),"task CPU rollout equivalence");}}}
   auto table_wall=ns()-t,table_cpu=cpu_ns()-ct;dump(out+".scores",expected);dump(out+".states",states);dump(out+".potentials",potentials);dump(out+".ram",ref.ram);dump(out+".rollout",rollout);
   std::cout<<"{\"status\":\"PASS\",\"mode\":\"cpu\",\"horizon\":"<<h<<",\"repeats\":"<<repeats<<",\"exact_wall_ns\":"<<exact_wall<<",\"exact_cpu_ns\":"<<exact_cpu<<",\"table_wall_ns\":"<<table_wall<<",\"table_cpu_ns\":"<<table_cpu<<",\"checksum\":"<<check<<"}\n";return 0;
  }
  d=std::make_unique<TransitionDriver>();d->io.dma_read_min=128;dev=akida::HardwareDevice::create(d.get());program=load<uint8_t>(f+"/program.bin");auto info=dev->program(program.data(),program.size());programmed=true;
  need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&info.output_is_dense()&&info.activation_enabled()&&info.output_bits()==1,"dense learner decoder");bool learning=true;dev->toggle_learn(true);
  auto capture=[&](){if(!learning)dev->toggle_learn(true);std::vector<uint32_t>v(dev->learn_mem_size());dev->learn_mem(v.data());if(!learning){dev->toggle_learn(false);static_cast<akida::HardwareDeviceImpl*>(dev.get())->pipeline(false);}return v;};need(capture()==initial,"initial visible SRAM exact");dump(out+".initial.ram",initial);
  auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());impl->pipeline(false);auto fixed=[&](const char*tag){auto p=load<uint32_t>(f+"/fixed.ram"),got=p;akida::dma::dma_config_read(got.data()+2,uint32_t(got.size()-2),got.data(),impl->dma_config(),impl->mem(),d.get(),true);dump(out+".fixed."+tag,got);need(got==p,"fixed decoder SRAM unchanged");};fixed("initial");
  need(dev->set_batch_size(1,false)==1,"serial diagnostic queue");auto input=dev->scratch_alloc(n);std::vector<uint8_t>got;std::vector<uint32_t>observed_states;std::vector<int64_t>times;
  for(int i=0;i<events;i++){
   if(i==16&&arm=="frozen"){learning=false;dev->toggle_learn(false);}
   auto t=ns();d->write(input,inputs.data()+i*n,n);auto label=labels[i];need(dev->enqueue(input,label>=0?&label:nullptr),"diagnostic enqueue");akida::TensorUniquePtr y;auto until=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch()))need(Clock::now()<until,"diagnostic completion bound");times.push_back(ns()-t);need(y->type()==akida::TensorType::uint8&&y->size()==n,"diagnostic output shape");auto p=y->data<uint8_t>();got.insert(got.end(),p,p+n);auto ram=capture();observed_states.insert(observed_states.end(),ram.begin(),ram.end());dump(out+".scores",got);dump(out+".states",observed_states);need(std::equal(p,p+n,expected.begin()+i*n),"diagnostic output differs from exact CPU");need(std::equal(ram.begin(),ram.end(),states.begin()+i*98),"diagnostic full learning SRAM differs from exact CPU");
  }
  dump(out+".latency_ns",times);auto terminal=capture();dump(out+".ram",terminal);need(terminal==ref.ram,"diagnostic terminal SRAM exact");
  std::vector<uint8_t>observed_rollout;std::vector<uint32_t>descriptors,addresses,outputs;std::vector<int64_t>quiet;
  if(h){need(dev->set_batch_size(h+1,false)==size_t(h+1),"one spare dependent descriptor");impl->pipeline(false);
   for(int s=0;s<u;s++){
    d->write(input,partial.data()+s*n,n);std::vector<uint8_t>seed_readback(n);d->read(input,seed_readback.data(),n);dump(out+".seed"+std::to_string(s),seed_readback);need(std::equal(seed_readback.begin(),seed_readback.end(),partial.begin()+s*n),"seed input SRAM readback");d->io.hold_queue();uint32_t previous=0,target=0;
    for(int k=0;k<h;k++){need(dev->enqueue(input,nullptr),"unlabelled dependent enqueue");auto a=d->io.last_descriptor,b=a;need(d->descriptor_address,"descriptor address captured");if(k)b[1]=previous+akida::dma::kOutputHeaderByteSize;for(int j=0;j<16;j++)if(j!=1)need(a[j]==b[j],"only input address edited");need(((b[8]>>16)&1023)==0,"no class is delivered in rollout");need(b[1]>=LabDriver::RAM+256&&uint64_t(b[1])+n<=uint64_t(LabDriver::RAM)+LabDriver::SIZE,"SRAM input bound");need(b[7]>=LabDriver::RAM+256&&uint64_t(b[7])+32+n<=uint64_t(LabDriver::RAM)+LabDriver::SIZE,"SRAM output bound");d->write(d->descriptor_address,b.data(),64);descriptors.insert(descriptors.end(),b.begin(),b.end());addresses.push_back(d->descriptor_address);outputs.push_back(b[7]);previous=b[7];target=b[0]>>16;}
    auto r0=d->io.reads,w0=d->io.writes;auto t0=ns(),ct0=cpu_ns();d->io.release_queue();std::this_thread::sleep_for(std::chrono::milliseconds(5));auto t1=ns(),ct1=cpu_ns();auto reads=d->io.reads-r0,writes=d->io.writes-w0;auto status=d->io.read32(LabDriver::REG+0x28010);quiet.insert(quiet.end(),{t0,t1,ct1-ct0,int64_t(reads),int64_t(writes),int64_t(target),int64_t(status)});dump(out+".quiet",quiet);need(reads==0&&writes==1&&(status>>16)==target,"quiet interval and first-wake completion");
    for(int k=0;k<h;k++){auto y=dev->fetch();need(y&&y->type()==akida::TensorType::uint8&&y->size()==n,"dependent result available");auto p=y->data<uint8_t>();observed_rollout.insert(observed_rollout.end(),p,p+n);std::vector<uint32_t>packet((32+n)/4);d->read(outputs[s*h+k],packet.data(),packet.size()*4);dump(out+".packet"+std::to_string(s*h+k),packet);}
    dump(out+".rollout",observed_rollout);need(std::equal(observed_rollout.begin()+s*h*n,observed_rollout.end(),rollout.begin()+s*h*n),"dependent trajectory differs from exact CPU");need(capture()==terminal,"query-only chain leaves full visible learning SRAM unchanged");
   }
  }else dump(out+".rollout",observed_rollout);
  dump(out+".descriptors",descriptors);dump(out+".descriptor_addresses",addresses);dump(out+".output_addresses",outputs);dump(out+".rollout.ram",capture());fixed("final");dev->scratch_free(input);dev->unprogram();programmed=false;
  std::cout<<"{\"status\":\"PASS\",\"mode\":\"native\",\"horizon\":"<<h<<",\"diagnostic_events\":26,\"rollout_forwards\":"<<8*h<<",\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception&e){if(d&&d->io.held)try{d->io.release_queue();}catch(...){}if(dev&&programmed)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
