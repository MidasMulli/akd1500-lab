// Bounded native output-to-input dependencies. Vendor runtime supplied separately.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
#include "engine/dma.h"

class TapDriver final:public akida::HardwareDriver {
 public:
 LabDriver io;uint32_t descriptor_address=0;
 void read(uint32_t a,void*p,size_t n)const override{io.read(a,p,n);}
 void write(uint32_t a,const void*p,size_t n)override{io.write(a,p,n);if(io.held&&n==64&&a>=LabDriver::RAM+256&&a<LabDriver::RAM+LabDriver::SIZE)descriptor_address=a;}
 const char*desc()const override{return "bounded native feedback / recorded descriptor addresses";}
 uint32_t scratch_memory()const override{return io.scratch_memory();}
 uint32_t scratch_size()const override{return io.scratch_size();}
 uint32_t top_level_reg()const override{return io.top_level_reg();}
 uint32_t akida_visible_memory()const override{return 0;}
 uint32_t akida_visible_memory_size()const override{return 0;}
};
int main(int argc,char**argv){
 std::unique_ptr<TapDriver>d;akida::HardwareDevicePtr dev;std::vector<uint8_t>program;bool programmed=false;
 try{
  need(argc==6,"native/cpu fixture arm horizon output");std::string mode=argv[1],f=argv[2],arm=argv[3],out=argv[5];int steps=std::stoi(argv[4]);
  need((mode=="native"||mode=="cpu")&&(arm=="native"||arm=="frozen"||arm=="repeated_seed"||arm=="severed")&&(steps==1||steps==2||steps==3||steps==4||steps==5||steps==8||steps==16||steps==32),"frozen arguments");
  constexpr int n=128,u=128;bool learn=arm!="frozen";auto initial=load<uint32_t>(f+"/initial.ram");auto seed=load<uint8_t>(f+"/seed.input");auto probes=load<uint8_t>(f+"/probes.inputs");need(initial.size()==1026&&seed.size()==n&&probes.size()==4*n,"fixture shape");
  CPU ref(n,u,32,1,initial);std::vector<uint8_t>x=seed,scores,probe_scores;std::vector<uint32_t>expected_states;std::vector<int32_t>potentials;
  auto forward=[&](const std::vector<uint8_t>&a){std::vector<uint32_t>bits(n/32);for(int k=0;k<n;k++)bits[k/32]|=uint32_t(a[k])<<(k%32);auto p=ref.forward(bits,0,learn);std::vector<uint8_t>y;for(auto v:p)y.push_back(v>13);return std::make_pair(p,y);};
  for(int i=0;i<steps;i++){auto z=forward(x);potentials.insert(potentials.end(),z.first.begin(),z.first.end());scores.insert(scores.end(),z.second.begin(),z.second.end());expected_states.insert(expected_states.end(),ref.ram.begin(),ref.ram.end());x=arm=="repeated_seed"?seed:arm=="severed"?std::vector<uint8_t>(n):z.second;}
  auto expected_terminal=ref.ram;
  for(int q=0;q<4;q++){std::vector<uint8_t>a(probes.begin()+q*n,probes.begin()+(q+1)*n);auto z=forward(a);probe_scores.insert(probe_scores.end(),z.second.begin(),z.second.end());}
  if(mode=="cpu"){
   auto t=ns(),ct=cpu_ns();uint64_t checksum=0;
   for(int r=0;r<200;r++){CPU c(n,u,32,1,initial);auto a=seed;for(int i=0;i<steps;i++){std::vector<uint32_t>b(n/32);for(int k=0;k<n;k++)b[k/32]|=uint32_t(a[k])<<(k%32);auto p=c.forward(b,0,learn);for(int k=0;k<n;k++)a[k]=p[k]>13;for(auto v:a)checksum=checksum*1099511628211ULL+v;if(arm=="repeated_seed")a=seed;else if(arm=="severed")a.assign(n,0);}need(c.ram==expected_terminal,"CPU repeated final state");}
   dump(out+".scores",scores);dump(out+".ram",expected_terminal);dump(out+".probe_scores",probe_scores);dump(out+".probe_ram",ref.ram);dump(out+".reference_states",expected_states);dump(out+".potentials",potentials);
   std::cout<<"{\"status\":\"PASS\",\"mode\":\"cpu\",\"steps\":"<<steps<<",\"repeats\":200,\"wall_ns\":"<<ns()-t<<",\"cpu_ns\":"<<cpu_ns()-ct<<",\"checksum\":"<<checksum<<"}\n";return 0;
  }
  d=std::make_unique<TapDriver>();d->io.dma_read_min=128;dev=akida::HardwareDevice::create(d.get());program=load<uint8_t>(f+"/program.bin");auto info=dev->program(program.data(),program.size());programmed=true;
  need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&info.output_is_dense()&&info.activation_enabled()&&info.output_bits()==1,"dense binary resident learner/relay");
  dev->toggle_learn(true);auto capture=[&](){std::vector<uint32_t>v(dev->learn_mem_size());dev->learn_mem(v.data());return v;};need(capture()==initial,"initial SRAM exact");dump(out+".initial.ram",initial);if(!learn)dev->toggle_learn(false);
  auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());auto fixed=[&](const char*tag){auto p=load<uint32_t>(f+"/fixed.ram"),got=p;akida::dma::dma_config_read(got.data()+2,uint32_t(got.size()-2),got.data(),impl->dma_config(),impl->mem(),d.get(),true);dump(out+".fixed."+tag,got);need(got==p,"fixed relay SRAM unchanged");};fixed("initial");
  need(dev->set_batch_size(steps+1,false)==size_t(steps+1),"one spare descriptor slot");impl->pipeline(false);
  auto input=dev->scratch_alloc(n),zero=dev->scratch_alloc(n);std::vector<uint8_t>z(n,0);d->write(input,seed.data(),n);d->write(zero,z.data(),n);
  std::vector<uint32_t>descs,addresses,outputs;d->io.hold_queue();int32_t label=0;uint32_t previous=0,target=0;
  for(int i=0;i<steps;i++){
   need(dev->enqueue(input,learn?&label:nullptr),"bounded dependent enqueue");auto a=d->io.last_descriptor;need(d->descriptor_address!=0,"descriptor address captured");auto b=a;
   if(i&&arm!="repeated_seed")b[1]=(arm=="severed"&&i==1)?zero:previous+akida::dma::kOutputHeaderByteSize;
   need(b[1]>=LabDriver::RAM+256&&uint64_t(b[1])+n<=uint64_t(LabDriver::RAM)+LabDriver::SIZE,"input remains in device scratch SRAM");
   for(int k=0;k<16;k++)if(k!=1)need(a[k]==b[k],"only descriptor input pointer may change");
   need(a[7]>=LabDriver::RAM+256&&uint64_t(a[7])+32+n<=uint64_t(LabDriver::RAM)+LabDriver::SIZE,"output remains in device scratch SRAM");
   d->write(d->descriptor_address,b.data(),64);addresses.push_back(d->descriptor_address);descs.insert(descs.end(),b.begin(),b.end());outputs.push_back(b[7]);previous=b[7];target=b[0]>>16;
  }
  dump(out+".descriptors",descs);dump(out+".descriptor_addresses",addresses);dump(out+".output_addresses",outputs);
  auto r0=d->io.reads,w0=d->io.writes;auto t0=ns(),ct0=cpu_ns();d->io.release_queue();std::this_thread::sleep_for(std::chrono::milliseconds(5));auto t1=ns(),ct1=cpu_ns();auto quiet_ns=t1-t0;auto reads=d->io.reads-r0,writes=d->io.writes-w0;
  need(reads==0&&writes==1,"no driver traffic except one release during quiet interval");uint32_t status=d->io.read32(LabDriver::REG+0x28010);dump(out+".quiet",std::vector<int64_t>{t0,t1,ct1-ct0,int64_t(reads),int64_t(writes),int64_t(target),int64_t(status)});need((status>>16)==target,"all dependent steps complete at first wake");
  std::vector<uint8_t>got;for(int i=0;i<steps;i++){auto y=dev->fetch();need(y&&y->type()==akida::TensorType::uint8&&y->size()==u,"dense binary result at first fetch");auto p=y->data<uint8_t>();got.insert(got.end(),p,p+u);std::vector<uint32_t>packet((32+n)/4);d->read(outputs[i],packet.data(),packet.size()*4);dump(out+".output_packet"+std::to_string(i),packet);}
  dump(out+".scores",got);need(got==scores,"dependent outputs differ from independent C++ replay");
  if(!learn)dev->toggle_learn(true);auto terminal=capture();if(!learn)dev->toggle_learn(false);dump(out+".ram",terminal);need(terminal==expected_terminal,"dependent full terminal SRAM differs");
  r0=d->io.reads;w0=d->io.writes;t0=ns();ct0=cpu_ns();std::this_thread::sleep_for(std::chrono::milliseconds(20));t1=ns();ct1=cpu_ns();reads=d->io.reads-r0;writes=d->io.writes-w0;need(reads==0&&writes==0,"idle observation has no driver traffic");auto idle=d->io.read32(LabDriver::REG+0x28010);dump(out+".idle",std::vector<int64_t>{t0,t1,ct1-ct0,int64_t(reads),int64_t(writes),int64_t(idle)});need(idle==status,"no extra jobs after finite queue exhausted");
  if(!learn)dev->toggle_learn(true);auto idle_ram=capture();if(!learn)dev->toggle_learn(false);dump(out+".idle.ram",idle_ram);need(idle_ram==terminal,"no continued learning without submitted work");
  std::vector<uint8_t>observed_probes;for(int q=0;q<4;q++){d->write(input,probes.data()+q*n,n);need(dev->enqueue(input,learn?&label:nullptr),"post-loop query");akida::TensorUniquePtr y;auto until=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch()))need(Clock::now()<until,"query bounded completion");need(y->type()==akida::TensorType::uint8&&y->size()==u,"query shape");auto p=y->data<uint8_t>();observed_probes.insert(observed_probes.end(),p,p+u);}
  dump(out+".probe_scores",observed_probes);need(observed_probes==probe_scores,"partial cue query differs");if(!learn)dev->toggle_learn(true);auto final=capture();dump(out+".probe_ram",final);need(final==ref.ram,"post-query SRAM exact");fixed("final");dev->scratch_free(input);dev->scratch_free(zero);dev->unprogram();programmed=false;
  std::cout<<"{\"status\":\"PASS\",\"mode\":\"native\",\"steps\":"<<steps<<",\"single_seed_upload\":true,\"future_inputs_read_device_outputs\":"<<(arm=="native"||arm=="frozen"?"true":"false")<<",\"driver_reads_during_quiet\":0,\"driver_writes_after_release_during_quiet\":0,\"quiet_observation_ns\":"<<quiet_ns<<",\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception&e){if(d&&d->io.held)try{d->io.release_queue();}catch(...){}if(dev&&programmed)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
