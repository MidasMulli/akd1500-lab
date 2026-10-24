// Native output retains history; the host writes only the current-symbol half.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
#include "engine/dma.h"
class TemporalDriver final:public akida::HardwareDriver {
 public:LabDriver io;
 void read(uint32_t a,void*p,size_t n)const override{io.read(a,p,n);}
 void write(uint32_t a,const void*p,size_t n)override{io.write(a,p,n);}
 const char*desc()const override{return "native context / current-only external symbols";}
 uint32_t scratch_memory()const override{return io.scratch_memory();}uint32_t scratch_size()const override{return io.scratch_size();}uint32_t top_level_reg()const override{return io.top_level_reg();}
 uint32_t akida_visible_memory()const override{return 0;}uint32_t akida_visible_memory_size()const override{return 0;}
};
int main(int argc,char**argv){
 std::unique_ptr<TemporalDriver>d;akida::HardwareDevicePtr dev;
 try{
  need(argc==6,"cpu/native fixture arm epochs output");std::string mode=argv[1],f=argv[2],arm=argv[3],out=argv[5];int epochs=std::stoi(argv[4]);
  need((mode=="cpu"||mode=="native")&&(epochs==0||epochs==1||epochs==2||epochs==8),"bounded mode/stage");
  constexpr int n=256,u=8;auto stem=f+"/"+arm+"_e"+std::to_string(epochs);auto plan=load<int32_t>(stem+".plan");int count=plan.size()/5;need(count==8+32*epochs,"sealed event count");
  auto cues=load<uint8_t>(f+"/cues"),codes=load<uint8_t>(f+"/codes"),supplied=load<uint8_t>(stem+".inputs");auto initial=load<uint32_t>(f+"/initial.ram");need(cues.size()==6*n&&codes.size()==8*n&&initial.size()==98,"fixture shapes");
  auto advance=[&](CPU&c,const uint8_t*x,int label){std::vector<uint32_t>bits(8);for(int k=0;k<n;k++)bits[k/32]|=uint32_t(x[k])<<(k%32);auto before=c.ram;auto p=c.forward(bits,std::max(0,label),label>=0);if(label<0)for(auto v:p)c.rot=(c.rot+mix(v))&31;
   if(label>=0)for(int j=0;j<u;j++){int b=2+j*c.stride;if(!std::equal(before.begin()+b+4,before.begin()+b+c.stride,c.ram.begin()+b+4))c.ram[b+3]=(uint32_t(std::min(p[j],8))<<16)|(c.ram[b+3]&65535);}
   std::vector<uint8_t>y(n);for(int j=0;j<u;j++)if(p[j]>24)for(int k=0;k<n;k++)y[k]|=codes[j*n+k];return y;};
  CPU ref(n,u,32,u,initial);std::vector<uint8_t>want,xs0;std::vector<uint32_t>states0;std::vector<uint8_t>last(n);
  for(int i=0;i<count;i++){need(plan[5*i]==i-1&&plan[5*i+1]>=0&&plan[5*i+1]<6,"only preceding native output");auto x=last;std::copy(cues.begin()+plan[5*i+1]*n+128,cues.begin()+(plan[5*i+1]+1)*n,x.begin()+128);if(plan[5*i+3])std::fill(x.begin(),x.begin()+128,0);need(std::equal(x.begin(),x.end(),supplied.begin()+i*n),"independent CPU input");last=advance(ref,x.data(),plan[5*i+2]);want.insert(want.end(),last.begin(),last.end());xs0.insert(xs0.end(),x.begin(),x.end());states0.insert(states0.end(),ref.ram.begin(),ref.ram.end());}
  if(mode=="cpu"){
   auto table=load<int32_t>(f+"/old");int history=-1;std::vector<uint8_t>task,stateless;
   for(int i=0;i<count;i++){const uint8_t*x=cues.data()+plan[5*i+1]*n;int symbol=-1,best=-1;for(int c=0;c<5;c++){int hits=0;for(int k=128;k<n;k++)hits+=x[k]&cues[c*n+k];if(hits>best){best=hits;symbol=c;}}need(best>=12,"CPU recognizes actual current cue");int target;
    if(symbol<4){history=symbol;target=history;}else{need(history>=0,"CPU previous context");target=table[history];if(plan[5*i+2]>=0)table[history]=plan[5*i+2];}
    task.insert(task.end(),codes.begin()+target*n,codes.begin()+(target+1)*n);int st=symbol<4?symbol:4;stateless.insert(stateless.end(),codes.begin()+st*n,codes.begin()+(st+1)*n);
   }
   dump(out+".task_scores",task);dump(out+".stateless_scores",stateless);dump(out+".scores",want);dump(out+".inputs",xs0);dump(out+".states",states0);dump(out+".ram",ref.ram);std::cout<<"{\"status\":\"PASS\",\"mode\":\"cpu\",\"events\":"<<count<<"}\n";return 0;
  }
  d=std::make_unique<TemporalDriver>();d->io.dma_read_min=128;dev=akida::HardwareDevice::create(d.get());auto program=load<uint8_t>(f+"/model.bin");auto info=dev->program(program.data(),program.size());need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&info.output_is_dense(),"one resident dense model");dev->toggle_learn(true);bool learning=true;auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());need(dev->set_batch_size(2,false)==2,"two native carry buffers");impl->pipeline(false);
  auto capture=[&](){if(!learning)dev->toggle_learn(true);std::vector<uint32_t>v(dev->learn_mem_size());dev->learn_mem(v.data());if(!learning){dev->toggle_learn(false);impl->pipeline(false);}return v;};need(capture()==initial,"initial SRAM");dump(out+".initial.ram",initial);
  auto fixed=[&](const char*tag){auto p=load<uint32_t>(f+"/fixed.ram"),q=p;akida::dma::dma_config_read(q.data()+2,q.size()-2,q.data(),impl->dma_config(),impl->mem(),d.get(),true);dump(out+".fixed."+tag,q);need(q==p,"fixed decoder unchanged");};fixed("initial");
  auto external=dev->scratch_alloc(n);std::vector<uint8_t>zero(n);d->write(external,zero.data(),n);
  std::vector<uint32_t>addresses,input_addresses,descriptors,observed;std::vector<uint8_t>ys,xs;std::vector<int64_t>delays;int current_writes=0,sever_writes=0;
  for(int i=0;i<count;i++){
   if(i==8&&arm=="frozen"){learning=false;dev->toggle_learn(false);impl->pipeline(false);}
   uint32_t in=i?addresses.back()+32:external;int cue=plan[5*i+1],label=plan[5*i+2];auto t=ns();if(plan[5*i+4])std::this_thread::sleep_for(std::chrono::milliseconds(plan[5*i+4]));delays.push_back(ns()-t);
   d->write(in+128,cues.data()+cue*n+128,128);current_writes++;
   if(plan[5*i+3]){d->write(in,zero.data(),128);sever_writes++;}
   need(in>=LabDriver::RAM+256&&uint64_t(in)+n<=uint64_t(LabDriver::RAM)+LabDriver::SIZE,"SRAM bound");
   d->io.hold_queue();need(dev->enqueue(in,label>=0?&label:nullptr),"current-symbol enqueue");auto desc=d->io.last_descriptor;need(desc[1]==in&&((desc[8]>>16)&1023)==uint32_t(label>=0?label+1:0),"descriptor input/label");if(i)need(desc[7]!=addresses.back(),"preserve previous output");addresses.push_back(desc[7]);input_addresses.push_back(in);descriptors.insert(descriptors.end(),desc.begin(),desc.end());d->io.release_queue();
   akida::TensorUniquePtr y;auto stop=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch()))need(Clock::now()<stop,"completion deadline");need(y->type()==akida::TensorType::uint8&&y->size()==n,"returned code");auto p=y->data<uint8_t>();ys.insert(ys.end(),p,p+n);std::vector<uint8_t>x(n);d->read(in,x.data(),n);xs.insert(xs.end(),x.begin(),x.end());std::vector<uint32_t>packet(72);d->read(addresses.back(),packet.data(),288);dump(out+".packet"+std::to_string(i),packet);
   auto ram=capture();observed.insert(observed.end(),ram.begin(),ram.end());dump(out+".scores",ys);dump(out+".inputs",xs);dump(out+".states",observed);dump(out+".descriptors",descriptors);dump(out+".output_addresses",addresses);dump(out+".input_addresses",input_addresses);dump(out+".delays_ns",delays);
   need(std::equal(p,p+n,want.begin()+i*n),"native output / packed CPU mismatch");need(std::equal(x.begin(),x.end(),xs0.begin()+i*n),"native history / current-only input mismatch");need(std::equal(ram.begin(),ram.end(),states0.begin()+i*98),"captured learning SRAM / packed CPU mismatch");
  }
  dump(out+".ram",capture());fixed("final");dev->scratch_free(external);dev->unprogram();
  std::cout<<"{\"status\":\"PASS\",\"mode\":\"native\",\"events\":"<<count<<",\"program_calls\":1,\"weight_restore_calls\":0,\"threshold_resets\":0,\"current_half_writes\":"<<current_writes<<",\"history_ablation_writes\":"<<sever_writes<<",\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception&e){if(dev)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
