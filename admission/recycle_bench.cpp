// Native prediction-error correction with a fixed negative-evidence phase. Own source; licensed runtime is linked locally.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
static std::vector<uint8_t> fixed_layer(const std::vector<uint8_t>&x,const std::vector<int8_t>&w,const std::vector<int32_t>&t){
 need(w.size()==x.size()*t.size(),"fixed dimensions");std::vector<uint8_t>y(t.size());
 for(size_t j=0;j<t.size();j++){int v=0;for(size_t k=0;k<x.size();k++)v+=x[k]*w[k*t.size()+j];y[j]=v>t[j];}return y;
}
static void observation(std::vector<uint8_t>&x,const int32_t*e){
 int key=e[0],teacher=e[1];need(key>=0&&key<12&&(teacher==0||teacher==1),"raw observation");
 if(!teacher||e[4]){std::fill(x.begin()+32,x.begin()+64,0);for(int b=0;b<4;b++)for(int j=0;j<4;j++)x[32+b*8+((key>>b)&1)*4+j]=1;}
 std::fill(x.begin()+64,x.begin()+72,0);x[64+teacher]=1;x[68]=e[8];if(teacher)x[66+e[3]]=1;
 if(e[4]){x[84]=e[5];x[85]=e[6];}if(e[7])x[84]=x[85]=0;
}
int main(int argc,char**argv){
 std::unique_ptr<LabDriver>d;akida::HardwareDevicePtr dev;
 try{
  need(argc==6,"cpu/native fixture arm stage output");std::string mode=argv[1],f=argv[2],arm=argv[3],stage=argv[4],out=argv[5];
  need(mode=="cpu"||mode=="native","mode");auto plan=load<int32_t>(f+"/"+stage+".plan");int count=plan.size()/9;need(count>0&&count<=1024&&plan.size()==size_t(count*9),"finite plan");
  auto initial=load<uint32_t>(f+"/bank.ram"),infer=load<uint32_t>(f+"/bank.infer"),learn=load<uint32_t>(f+"/bank.learn");auto threshold=load<int32_t>(f+"/bank.threshold");CPU cpu(1024,32,32,2,initial);
  std::vector<std::vector<int8_t>>w;std::vector<std::vector<int32_t>>t;
  for(int k=0;k<4;k++){w.push_back(load<int8_t>(f+"/fixed"+std::to_string(k)+".weights"));t.push_back(load<int32_t>(f+"/fixed"+std::to_string(k)+".threshold"));}
  std::vector<uint8_t>last(128),want,wantx;std::vector<uint32_t>states;std::vector<int32_t>potentials;
  for(int i=0;i<count;i++){
   auto e=&plan[9*i];need((e[2]>=-1&&e[2]<2)&&(e[3]==0||e[3]==1),"actual label");need(!e[4]||stage=="truth","synthetic override confined to truth table");
   auto x=last;observation(x,e);if(e[7])last[84]=last[85]=0;auto z=x;
   for(int k=0;k<3;k++)z=fixed_layer(z,w[k],t[k]);std::vector<uint32_t>bits(32);for(int j=0;j<1024;j++)bits[j/32]|=uint32_t(z[j])<<(j%32);
   auto p=cpu.forward(bits,e[2]<0?0:e[2],e[2]>=0);potentials.insert(potentials.end(),p.begin(),p.end());std::vector<uint8_t>v(32);for(int j=0;j<32;j++)v[j]=p[j]>threshold[j];
   auto y=fixed_layer(v,w[3],t[3]);states.insert(states.end(),cpu.ram.begin(),cpu.ram.end());if(!e[1])last=y;
   want.insert(want.end(),y.begin(),y.end());wantx.insert(wantx.end(),x.begin(),x.end());
  }
  dump(out+".potentials",potentials);
  if(mode=="cpu"){dump(out+".scores",want);dump(out+".inputs",wantx);dump(out+".states",states);std::cout<<"{\"status\":\"PASS\",\"mode\":\"cpu\",\"events\":"<<count<<"}\n";return 0;}
  d=std::make_unique<LabDriver>();d->dma_read_min=128;dev=akida::HardwareDevice::create(d.get());auto program=load<uint8_t>(f+"/model.bin");auto info=dev->program(program.data(),program.size());
  need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&info.output_is_dense()&&info.activation_enabled(),"one resident binary program");
  dev->toggle_learn(true);auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());need(dev->set_batch_size(4,false)==4,"four native carry buffers");impl->pipeline(false);
  auto capture=[&](const std::vector<uint32_t>&p){auto v=p;akida::dma::dma_config_read(v.data()+2,v.size()-2,v.data(),impl->dma_config(),impl->mem(),d.get(),true);return v;};
  auto got=capture(initial);dump(out+".initial",got);need(got==initial,"initial native RAM");
  auto fixed=[&](const char*tag){for(int k=0;k<4;k++){auto a=load<uint32_t>(f+"/fixed"+std::to_string(k)+".ram");auto b=capture(a);dump(out+".fixed"+std::to_string(k)+"."+tag,b);need(a==b,"fixed comparator/route RAM");}};fixed("initial");
  auto external=dev->scratch_alloc(128);std::vector<uint8_t>zero(128);d->write(external,zero.data(),128);uint32_t source=external;
  std::vector<uint8_t>ys,xs;std::vector<uint32_t>observed,addr,inputs,descs;std::vector<int32_t>modes;int oldmode=-1,key_writes=0,phase_writes=0,override_writes=0,cut_writes=0;
  for(int i=0;i<count;i++){
   kick_watchdog();auto e=&plan[9*i];int teacher=e[1],wire=teacher?(e[8]?1-e[3]:e[3]):0,on=e[2]>=0;modes.push_back(on);
   if(on!=oldmode){auto &p=on?learn:infer;akida::dma::dma_config_write(p.data(),p.size(),impl->dma_config(),impl->external_mem(),d.get(),true);oldmode=on;}
   if(!teacher||e[4]){std::vector<uint8_t>raw(32);for(int b=0;b<4;b++)for(int j=0;j<4;j++)raw[b*8+((e[0]>>b)&1)*4+j]=1;d->write(source+32,raw.data(),32);key_writes++;}
   uint64_t phase=(teacher?(256u|(1u<<(16+8*e[3]))):1u)|(uint64_t(e[8])<<32);d->write(source+64,&phase,8);phase_writes++;
   if(e[4]){uint32_t pred=uint32_t(e[5])|(uint32_t(e[6])<<8);d->write(source+84,&pred,4);override_writes++;}
   if(e[7]){uint32_t z=0;d->write(source+84,&z,4);cut_writes++;}
   uint32_t in=source;d->hold_queue();need(dev->enqueue(in,&wire),"raw cue / delayed actual outcome");auto desc=d->last_descriptor;
   need(desc[1]==in&&((desc[8]>>16)&1023)==uint32_t(wire+1)&&desc[7]+32!=in,"native source and actual label descriptor");
   addr.push_back(desc[7]);inputs.push_back(in);descs.insert(descs.end(),desc.begin(),desc.end());d->release_queue();
   akida::TensorUniquePtr y;auto deadline=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch()))need(Clock::now()<deadline,"completion timeout");
   need(y->type()==akida::TensorType::uint8&&y->size()==128,"native output contract");auto p=y->data<uint8_t>();ys.insert(ys.end(),p,p+128);
   std::vector<uint8_t>x(128);d->read(in,x.data(),128);xs.insert(xs.end(),x.begin(),x.end());if(!teacher)source=desc[7]+32;
   std::vector<uint32_t>packet(40);d->read(addr.back(),packet.data(),160);dump(out+".packet"+std::to_string(i),packet);
   auto ram=capture(initial);observed.insert(observed.end(),ram.begin(),ram.end());
   dump(out+".states",observed);dump(out+".scores",ys);dump(out+".inputs",xs);dump(out+".descriptors",descs);dump(out+".output_addresses",addr);dump(out+".input_addresses",inputs);dump(out+".learning_modes",modes);
   need(std::equal(x.begin(),x.end(),wantx.begin()+i*128),"native saved prediction/raw outcome input exact");need(std::equal(p,p+128,want.begin()+i*128),"native error-gated outputs exact");
   if(!std::equal(ram.begin(),ram.end(),states.begin()+i*ram.size())){for(size_t j=0;j<ram.size();j++)if(ram[j]!=states[i*ram.size()+j]){std::cerr<<"event="<<i<<" word="<<j<<" actual="<<ram[j]<<" expected="<<states[i*ram.size()+j]<<"\n";break;}throw std::runtime_error("native learning RAM differs from packed replay");}
  }
  fixed("final");dev->scratch_free(external);dev->unprogram();std::cout<<"{\"status\":\"PASS\",\"mode\":\"native\",\"events\":"<<count<<",\"program_calls\":1,\"weight_restore_calls\":0,\"host_error_decisions\":0,\"current_key_writes\":"<<key_writes<<",\"phase_actual_writes\":"<<phase_writes<<",\"synthetic_prediction_writes\":"<<override_writes<<",\"prediction_ablation_writes\":"<<cut_writes<<",\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception&e){if(dev)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
