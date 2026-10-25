// Seven FNP3s, three plastic banks; uniform phase clocks, no host policy branch.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
struct B {
 int n,u,c;CPU cpu;std::vector<uint32_t>infer,learn;std::vector<int32_t>threshold;
 B(int nn,int uu,int cc,const std::string&f):n(nn),u(uu),c(cc),cpu(nn,uu,32,cc,load<uint32_t>(f+".ram")),infer(load<uint32_t>(f+".infer")),learn(load<uint32_t>(f+".learn")),threshold(load<int32_t>(f+".threshold")){}
 std::vector<uint8_t> step(const std::vector<uint8_t>&x,int label,std::vector<int32_t>&p){std::vector<uint32_t>bits(n/32);for(int j=0;j<n;j++)bits[j/32]|=uint32_t(x[j])<<(j%32);p=cpu.forward(bits,label<0?0:label,label>=0);std::vector<uint8_t>y(u);for(int j=0;j<u;j++)y[j]=p[j]>threshold[j];return y;}
};
static std::vector<uint8_t> layer(const std::vector<uint8_t>&x,const std::vector<int8_t>&w,const std::vector<int32_t>&t){need(w.size()==x.size()*t.size(),"fixed dimensions");std::vector<uint8_t>y(t.size());for(size_t j=0;j<t.size();j++){int v=0;for(size_t k=0;k<x.size();k++)v+=x[k]*w[k*t.size()+j];y[j]=v>t[j];}return y;}
int main(int argc,char**argv){
 std::unique_ptr<LabDriver>d;akida::HardwareDevicePtr dev;
 try{
  need(argc==6,"cpu/native fixture arm stage output");std::string mode=argv[1],f=argv[2],arm=argv[3],stage=argv[4],out=argv[5];need(mode=="cpu"||mode=="native","mode");auto plan=load<int32_t>(f+"/"+stage+".plan");int count=plan.size()/7;need(count>0&&count<=1024&&plan.size()==size_t(count*7),"finite plan");
  std::vector<B>banks; banks.emplace_back(256,32,1,f+"/bank0");banks.emplace_back(1024,48,2,f+"/bank1");banks.emplace_back(1024,32,2,f+"/bank2");std::vector<std::vector<int8_t>>w;std::vector<std::vector<int32_t>>t;
  for(int k=0;k<4;k++){w.push_back(load<int8_t>(f+"/fixed"+std::to_string(k)+".weights"));t.push_back(load<int32_t>(f+"/fixed"+std::to_string(k)+".threshold"));}
  std::vector<uint8_t>last(128),want,wantx;std::vector<std::vector<uint32_t>>states(3);std::vector<std::vector<int32_t>>potentials(3);
  for(int i=0;i<count;i++){
   int key=plan[7*i],teacher=plan[7*i+1];need(key>=0&&key<12&&(teacher==0||teacher==1),"raw key/phase");auto x=last;std::fill(x.begin()+64,x.begin()+68,0);x[64+teacher]=1;
   if(!teacher){std::fill(x.begin()+32,x.begin()+64,0);for(int b=0;b<4;b++)for(int c=0;c<4;c++)x[32+b*8+((key>>b)&1)*4+c]=1;}else if(plan[7*i+6]){x[80]=0;last[80]=0;}
   auto z=layer(x,w[0],t[0]);for(int k=0;k<3;k++){int label=plan[7*i+2+k];need(label>=-1&&label<banks[k].c,"constant manager/actual outcome label");std::vector<int32_t>p;auto v=banks[k].step(z,label,p);states[k].insert(states[k].end(),banks[k].cpu.ram.begin(),banks[k].cpu.ram.end());potentials[k].insert(potentials[k].end(),p.begin(),p.end());z=layer(v,w[k+1],t[k+1]);}if(!teacher)last=z;want.insert(want.end(),z.begin(),z.end());wantx.insert(wantx.end(),x.begin(),x.end());
  }
  for(int k=0;k<3;k++)dump(out+".potentials"+std::to_string(k),potentials[k]);
  if(mode=="cpu"){dump(out+".scores",want);dump(out+".inputs",wantx);for(int k=0;k<3;k++)dump(out+".states"+std::to_string(k),states[k]);std::cout<<"{\"status\":\"PASS\",\"mode\":\"cpu\",\"events\":"<<count<<"}\n";return 0;}
  d=std::make_unique<LabDriver>();d->dma_read_min=128;dev=akida::HardwareDevice::create(d.get());auto program=load<uint8_t>(f+"/model.bin");auto info=dev->program(program.data(),program.size());need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&info.output_is_dense()&&info.activation_enabled(),"one resident binary program");dev->toggle_learn(true);auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());need(dev->set_batch_size(4,false)==4,"four native carry buffers");impl->pipeline(false);
  auto capture=[&](const std::vector<uint32_t>&p){auto v=p;akida::dma::dma_config_read(v.data()+2,v.size()-2,v.data(),impl->dma_config(),impl->mem(),d.get(),true);return v;};
  auto setmode=[&](const std::vector<int>&v){for(int k=0;k<3;k++){auto &p=v[k]?banks[k].learn:banks[k].infer;akida::dma::dma_config_write(p.data(),p.size(),impl->dma_config(),impl->external_mem(),d.get(),true);}};
  std::vector<std::vector<uint32_t>>initial;for(int k=0;k<3;k++){auto a=load<uint32_t>(f+"/bank"+std::to_string(k)+".ram");auto b=capture(a);dump(out+".initial"+std::to_string(k),b);need(a==b,"all native initial RAMs");initial.push_back(a);}
  auto fixed=[&](const char*tag){for(int k=0;k<4;k++){auto a=load<uint32_t>(f+"/fixed"+std::to_string(k)+".ram");auto b=capture(a);dump(out+".fixed"+std::to_string(k)+"."+tag,b);need(a==b,"fixed encoder/policy/relay unchanged");}};fixed("initial");
  auto external=dev->scratch_alloc(128);std::vector<uint8_t>zero(128);d->write(external,zero.data(),128);uint32_t previous_query_output=external;std::vector<uint8_t>ys,xs;std::vector<uint32_t>addresses,inputs,descriptors;std::vector<std::vector<uint32_t>>observed(3);std::vector<int32_t>modes;std::vector<int>oldm(3,-1);int key_writes=0,phase_writes=0,sever_writes=0,clocks=0;
  for(int i=0;i<count;i++){
   kick_watchdog();int key=plan[7*i],teacher=plan[7*i+1],wire=plan[7*i+5];std::vector<int>requested(3);for(int k=0;k<3;k++){requested[k]=plan[7*i+2+k]>=0;modes.push_back(requested[k]);}if(requested!=oldm){setmode(requested);oldm=requested;clocks++;}
   uint32_t in=previous_query_output;
   if(!teacher){std::vector<uint8_t>raw(32);for(int b=0;b<4;b++)for(int c=0;c<4;c++)raw[b*8+((key>>b)&1)*4+c]=1;d->write(in+32,raw.data(),32);key_writes++;}
   uint32_t phase=teacher?256:1;d->write(in+64,&phase,4);phase_writes++;if(plan[7*i+6]){uint32_t v=0;d->write(in+80,&v,4);sever_writes++;}
   d->hold_queue();need(dev->enqueue(in,&wire),"raw observation / delayed actual outcome");auto desc=d->last_descriptor;need(desc[1]==in&&((desc[8]>>16)&1023)==uint32_t(wire+1),"manager constant0/actual outcome descriptor");need(desc[7]+32!=in,"output cannot overwrite source");addresses.push_back(desc[7]);inputs.push_back(in);descriptors.insert(descriptors.end(),desc.begin(),desc.end());d->release_queue();
   akida::TensorUniquePtr y;auto deadline=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch()))need(Clock::now()<deadline,"native completion deadline");need(y->type()==akida::TensorType::uint8&&y->size()==128,"native output contract");auto p=y->data<uint8_t>();ys.insert(ys.end(),p,p+128);std::vector<uint8_t>x(128);d->read(in,x.data(),128);xs.insert(xs.end(),x.begin(),x.end());if(!teacher)previous_query_output=desc[7]+32;
   std::vector<uint32_t>packet(40);d->read(addresses.back(),packet.data(),160);dump(out+".packet"+std::to_string(i),packet);
   for(int k=0;k<3;k++){auto a=capture(initial[k]);observed[k].insert(observed[k].end(),a.begin(),a.end());dump(out+".states"+std::to_string(k),observed[k]);}
   dump(out+".scores",ys);dump(out+".inputs",xs);dump(out+".descriptors",descriptors);dump(out+".output_addresses",addresses);dump(out+".input_addresses",inputs);dump(out+".learning_modes",modes);
   need(std::equal(x.begin(),x.end(),wantx.begin()+i*128),"native raw/permission input exact");need(std::equal(p,p+128,want.begin()+i*128),"native returned predictions/permission exact");
   for(int k=0;k<3;k++){size_t n=initial[k].size();if(!std::equal(observed[k].end()-n,observed[k].end(),states[k].begin()+i*n)){for(size_t j=0;j<n;j++)if(observed[k][i*n+j]!=states[k][i*n+j]){std::cerr<<"event="<<i<<" bank="<<k<<" word="<<j<<" actual="<<observed[k][i*n+j]<<" expected="<<states[k][i*n+j]<<"\n";break;}throw std::runtime_error("native SRAM differs from packed replay");}}
  }
  fixed("final");dev->scratch_free(external);dev->unprogram();std::cout<<"{\"status\":\"PASS\",\"mode\":\"native\",\"events\":"<<count<<",\"program_calls\":1,\"weight_restore_calls\":0,\"host_bank_decisions\":0,\"current_key_writes\":"<<key_writes<<",\"phase_writes\":"<<phase_writes<<",\"permission_ablation_writes\":"<<sever_writes<<",\"phase_clock_transitions\":"<<clocks<<",\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception&e){if(dev)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
