// Own-source trace instrument; only current-symbol channels are written per event.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
#include "engine/dma.h"
int main(int argc,char**argv){
 std::unique_ptr<LabDriver>d;akida::HardwareDevicePtr dev;
 try{
  need(argc==6,"cpu/native fixture arm cycles output");std::string mode=argv[1],f=argv[2],arm=argv[3],out=argv[5];int cycles=std::stoi(argv[4]);
  need((mode=="cpu"||mode=="native")&&(cycles==1||cycles==2||cycles==4||cycles==12),"bounded mode/stage");
  constexpr int n=128,features=256,u=40,words=482;auto stem=f+"/"+arm+"_c"+std::to_string(cycles);auto plan=load<int32_t>(stem+".plan");int count=plan.size()/3;need(count==131+16*cycles+((arm=="unseen"||arm=="unseen_frozen")?128:0),"sealed event count");
  auto enc=load<int8_t>(f+"/encoder"),relay=load<int8_t>(f+"/relay");auto supplied=load<uint8_t>(stem+".inputs");auto initial=load<uint32_t>(f+"/initial.ram");need(enc.size()==n*features&&relay.size()==u*n&&initial.size()==words,"fixture shapes");
  CPU ref(features,u,32,1,initial);std::vector<uint8_t>want,xs0;std::vector<int32_t>pots;std::vector<uint32_t>states0;std::vector<uint8_t>last(n);
  for(int k=0;k<48;k++)last[k]=k%16<8;
  for(int i=0;i<count;i++){
   auto x=last;if(plan[3*i+2])std::fill(x.begin(),x.begin()+48,0);std::fill(x.begin()+48,x.begin()+64,0);int bit=plan[3*i],label=plan[3*i+1];need((bit==0||bit==1)&&(label==0||label==-1),"no context label");for(int k=0;k<8;k++)x[48+bit*8+k]=1;
   need(std::equal(x.begin(),x.end(),supplied.begin()+i*n),"independent CPU carried input");std::vector<uint32_t>bits(features/32);
   for(int j=0;j<features;j++){int v=0;for(int k=0;k<n;k++)v+=x[k]*enc[k*features+j];if(v>2)bits[j/32]|=uint32_t(1)<<(j%32);}
   auto before=ref.ram;auto p=ref.forward(bits,0,label>=0);
   if(label>=0)for(int j=0;j<u;j++){int b=2+j*ref.stride;if(!std::equal(before.begin()+b+4,before.begin()+b+ref.stride,ref.ram.begin()+b+4))ref.ram[b+3]=(uint32_t(p[j])<<16)|(ref.ram[b+3]&65535);}
   last.assign(n,0);for(int j=0;j<u;j++)if(p[j]>(j<32?27:7))for(int k=0;k<n;k++)last[k]|=relay[j*n+k];
   want.insert(want.end(),last.begin(),last.end());xs0.insert(xs0.end(),x.begin(),x.end());states0.insert(states0.end(),ref.ram.begin(),ref.ram.end());pots.insert(pots.end(),p.begin(),p.end());
  }
  dump(out+".potentials",pots);
  std::vector<int32_t>dictionary(16,-1),cpu_codes;int raw=0,allocated=0;
  for(int i=0;i<count;i++){raw=(raw>>1)|(plan[3*i]<<3);if(i>=3&&plan[3*i+1]==0&&dictionary[raw]<0)dictionary[raw]=allocated++;cpu_codes.push_back(i<3?-1:dictionary[raw]);}dump(out+".cpu",cpu_codes);
  if(mode=="cpu"){dump(out+".scores",want);dump(out+".inputs",xs0);dump(out+".states",states0);dump(out+".ram",ref.ram);std::cout<<"{\"status\":\"PASS\",\"mode\":\"cpu\",\"events\":"<<count<<"}\n";return 0;}
  d=std::make_unique<LabDriver>();d->dma_read_min=128;dev=akida::HardwareDevice::create(d.get());auto program=load<uint8_t>(f+"/model.bin");auto info=dev->program(program.data(),program.size());need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&info.output_is_dense(),"one resident dense model");dev->toggle_learn(true);bool learning=true;auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());need(dev->set_batch_size(2,false)==2,"two native carry buffers");impl->pipeline(false);
  auto capture=[&](){if(!learning)dev->toggle_learn(true);std::vector<uint32_t>v(dev->learn_mem_size());dev->learn_mem(v.data());if(!learning){dev->toggle_learn(false);impl->pipeline(false);}return v;};need(capture()==initial,"initial learning SRAM");dump(out+".initial.ram",initial);
  auto fixed=[&](const char*tag){for(int j=0;j<2;j++){auto p=load<uint32_t>(f+"/fixed"+std::to_string(j)+".ram"),q=p;akida::dma::dma_config_read(q.data()+2,q.size()-2,q.data(),impl->dma_config(),impl->mem(),d.get(),true);dump(out+".fixed"+std::to_string(j)+"."+tag,q);need(q==p,"fixed encoder/relay unchanged");}};fixed("initial");
  auto external=dev->scratch_alloc(n);std::vector<uint8_t>zero(n),seed(n);for(int k=0;k<48;k++)seed[k]=k%16<8;d->write(external,seed.data(),n);
  std::vector<uint32_t>addresses,input_addresses,descriptors,observed;std::vector<uint8_t>ys,xs;int current_writes=0,sever_writes=0,mode_changes=0;std::vector<int32_t>modes;
  for(int i=0;i<count;i++){
   bool requested=plan[3*i+1]>=0;if(learning!=requested){learning=requested;dev->toggle_learn(learning);impl->pipeline(false);mode_changes++;}modes.push_back(learning);
   uint32_t in=i?addresses.back()+32:external;int bit=plan[3*i],label=plan[3*i+1];std::vector<uint8_t>current(16);for(int k=0;k<8;k++)current[bit*8+k]=1;d->write(in+48,current.data(),16);current_writes++;
   if(plan[3*i+2]){d->write(in,zero.data(),48);sever_writes++;}
   need(in>=LabDriver::RAM+256&&uint64_t(in)+n<=uint64_t(LabDriver::RAM)+LabDriver::SIZE,"SRAM bound");
   d->hold_queue();need(dev->enqueue(in,label>=0?&label:nullptr),"current-symbol enqueue");auto desc=d->last_descriptor;need(desc[1]==in&&((desc[8]>>16)&1023)==uint32_t(label>=0?1:0),"descriptor current input / constant label");if(i)need(desc[7]!=addresses.back(),"preserve preceding output");addresses.push_back(desc[7]);input_addresses.push_back(in);descriptors.insert(descriptors.end(),desc.begin(),desc.end());d->release_queue();
   akida::TensorUniquePtr y;auto stop=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch()))need(Clock::now()<stop,"completion deadline");need(y->type()==akida::TensorType::uint8&&y->size()==n,"returned raw history / context mask");auto p=y->data<uint8_t>();ys.insert(ys.end(),p,p+n);std::vector<uint8_t>x(n);d->read(in,x.data(),n);xs.insert(xs.end(),x.begin(),x.end());std::vector<uint32_t>packet(40);d->read(addresses.back(),packet.data(),160);dump(out+".packet"+std::to_string(i),packet);
   auto ram=capture();observed.insert(observed.end(),ram.begin(),ram.end());dump(out+".scores",ys);dump(out+".inputs",xs);dump(out+".states",observed);dump(out+".descriptors",descriptors);dump(out+".output_addresses",addresses);dump(out+".input_addresses",input_addresses);dump(out+".learning_modes",modes);
   need(std::equal(p,p+n,want.begin()+i*n),"native output / packed CPU mismatch");need(std::equal(x.begin(),x.end(),xs0.begin()+i*n),"native history / current-only input mismatch");need(ram.size()==words&&std::equal(ram.begin(),ram.end(),states0.begin()+i*words),"captured native RAM / packed CPU mismatch");
  }
  dump(out+".ram",capture());fixed("final");dev->scratch_free(external);dev->unprogram();
  std::cout<<"{\"status\":\"PASS\",\"mode\":\"native\",\"events\":"<<count<<",\"program_calls\":1,\"weight_restore_calls\":0,\"threshold_resets\":0,\"current_symbol_writes\":"<<current_writes<<",\"history_ablation_writes\":"<<sever_writes<<",\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception&e){if(dev)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
