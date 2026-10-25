// Two co-resident native memories. Shared clocks; actor never selects a bank.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
template<class T>static void append(const std::string&p,const std::vector<T>&v){std::ofstream f(p,std::ios::binary|std::ios::app);f.write((const char*)v.data(),v.size()*sizeof(T));need(bool(f),"append raw prefix");}
struct Bank{
 int u;CPU cpu;std::vector<uint32_t>initial,infer,learn;std::vector<int32_t>threshold;
 Bank(int units,const std::string&p):u(units),cpu(1024,units,32,2,load<uint32_t>(p+".ram")),initial(load<uint32_t>(p+".ram")),infer(load<uint32_t>(p+".infer")),learn(load<uint32_t>(p+".learn")),threshold(load<int32_t>(p+".threshold")){}
 std::vector<uint8_t>step(const std::vector<uint8_t>&x,int label,std::vector<int32_t>&p){std::vector<uint32_t>b(32);for(int j=0;j<1024;j++)b[j/32]|=uint32_t(x[j])<<(j%32);p=cpu.forward(b,label<0?0:label,label>=0);std::vector<uint8_t>y(u);for(int j=0;j<u;j++)y[j]=p[j]>threshold[j];return y;}
};
static std::vector<uint8_t>fixed_layer(const std::vector<uint8_t>&x,const std::vector<int8_t>&w,const std::vector<int32_t>&t){need(w.size()==x.size()*t.size(),"fixed dimensions");std::vector<uint8_t>y(t.size());for(size_t j=0;j<t.size();j++){int v=0;for(size_t k=0;k<x.size();k++)v+=x[k]*w[k*t.size()+j];y[j]=v>t[j];}return y;}
static void observation(std::vector<uint8_t>&x,const int32_t*e){
 int key=e[0],teacher=e[1],ctx=e[9];need(key>=0&&key<12&&(teacher==0||teacher==1)&&(ctx==0||ctx==1),"raw key/context/phase");
 if(!teacher||e[5]){std::fill(x.begin()+24,x.begin()+28,0);x[24+ctx]=1;std::fill(x.begin()+32,x.begin()+64,0);for(int b=0;b<4;b++)for(int j=0;j<4;j++)x[32+b*8+((key>>b)&1)*4+j]=1;}
 std::fill(x.begin()+64,x.begin()+72,0);x[64+teacher]=1;x[68]=e[8];if(teacher)x[66+e[4]]=1;if(e[5]){x[84]=e[6];x[85]=e[7];}
}
int main(int argc,char**argv){
 std::unique_ptr<LabDriver>d;akida::HardwareDevicePtr dev;
 try{
  need(argc==6,"cpu/native fixture arm stage_stream output");std::string mode=argv[1],f=argv[2],arm=argv[3],stage=argv[4],out=argv[5];need(mode=="cpu"||mode=="native","mode");
  auto plan=load<int32_t>(f+"/"+stage+".plan");int count=plan.size()/10;need(count>0&&count<=2048&&plan.size()==size_t(count*10),"finite sealed plan");std::vector<Bank>banks;banks.emplace_back(64,f+"/bank0");banks.emplace_back(32,f+"/bank1");
  std::vector<std::vector<int8_t>>w;std::vector<std::vector<int32_t>>t;
  for(int k=0;k<5;k++){w.push_back(load<int8_t>(f+"/fixed"+std::to_string(k)+".weights"));t.push_back(load<int32_t>(f+"/fixed"+std::to_string(k)+".threshold"));}
  std::vector<uint8_t>last(128),want,wantx;std::vector<std::vector<uint32_t>>states(2);std::vector<std::vector<int32_t>>potentials(2);
  for(int i=0;i<count;i++){
   auto e=&plan[10*i];need((e[4]==0||e[4]==1)&&(!e[5]||stage.rfind("truth",0)==0),"outcome/synthetic scope");auto x=last;observation(x,e);auto z=x;
   for(int k=0;k<3;k++)z=fixed_layer(z,w[k],t[k]);
   for(int k=0;k<2;k++){need(e[2+k]>=-1&&e[2+k]<2,"learning role label");std::vector<int32_t>p;auto v=banks[k].step(z,e[2+k],p);states[k].insert(states[k].end(),banks[k].cpu.ram.begin(),banks[k].cpu.ram.end());potentials[k].insert(potentials[k].end(),p.begin(),p.end());z=fixed_layer(v,w[3+k],t[3+k]);}
   if(!e[1])last=z;want.insert(want.end(),z.begin(),z.end());wantx.insert(wantx.end(),x.begin(),x.end());
  }
  for(int k=0;k<2;k++)dump(out+".potentials"+std::to_string(k),potentials[k]);
  if(mode=="cpu"){dump(out+".inputs",wantx);dump(out+".scores",want);for(int k=0;k<2;k++)dump(out+".states"+std::to_string(k),states[k]);std::cout<<"{\"status\":\"PASS\",\"mode\":\"cpu\",\"events\":"<<count<<"}\n";return 0;}
  d=std::make_unique<LabDriver>();d->dma_read_min=128;dev=akida::HardwareDevice::create(d.get());auto program=load<uint8_t>(f+"/model.bin");auto info=dev->program(program.data(),program.size());need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&info.output_is_dense()&&info.activation_enabled(),"one resident binary program");
  dev->toggle_learn(true);auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());need(dev->set_batch_size(4,false)==4,"four native carry buffers");impl->pipeline(false);
  auto capture=[&](const std::vector<uint32_t>&p){auto v=p;akida::dma::dma_config_read(v.data()+2,v.size()-2,v.data(),impl->dma_config(),impl->mem(),d.get(),true);return v;};
  for(int k=0;k<2;k++){auto a=capture(banks[k].initial);dump(out+".initial"+std::to_string(k),a);need(a==banks[k].initial,"native initial banks");}
  auto fixed=[&](const char*tag){for(int k=0;k<5;k++){auto a=load<uint32_t>(f+"/fixed"+std::to_string(k)+".ram");auto b=capture(a);dump(out+".fixed"+std::to_string(k)+"."+tag,b);need(a==b,"fixed native routing RAM");}};fixed("initial");
  auto external=dev->scratch_alloc(128);std::vector<uint8_t>zero(128);d->write(external,zero.data(),128);uint32_t source=external;std::vector<int>oldm(2,-1);int keys=0,contexts=0,phases=0,overrides=0;
  for(int i=0;i<count;i++){
   kick_watchdog();auto e=&plan[10*i];int teacher=e[1],wire=teacher?(e[8]?1-e[4]:e[4]):0;std::vector<int32_t>modes={int(e[2]>=0),int(e[3]>=0)};
   for(int k=0;k<2;k++)if(modes[k]!=oldm[k]){auto &p=modes[k]?banks[k].learn:banks[k].infer;akida::dma::dma_config_write(p.data(),p.size(),impl->dma_config(),impl->external_mem(),d.get(),true);oldm[k]=modes[k];}
   if(!teacher||e[5]){uint32_t ctx=1u<<(8*e[9]);d->write(source+24,&ctx,4);contexts++;std::vector<uint8_t>raw(32);for(int b=0;b<4;b++)for(int j=0;j<4;j++)raw[b*8+((e[0]>>b)&1)*4+j]=1;d->write(source+32,raw.data(),32);keys++;}
   uint64_t phase=(teacher?(256u|(1u<<(16+8*e[4]))):1u)|(uint64_t(e[8])<<32);d->write(source+64,&phase,8);phases++;
   if(e[5]){uint32_t pred=uint32_t(e[6])|(uint32_t(e[7])<<8);d->write(source+84,&pred,4);overrides++;}
   uint32_t in=source;d->hold_queue();need(dev->enqueue(in,&wire),"raw context/cue and actual role label");auto desc=d->last_descriptor;need(desc[1]==in&&((desc[8]>>16)&1023)==uint32_t(wire+1)&&desc[7]+32!=in,"native source/label descriptor");d->release_queue();
   akida::TensorUniquePtr y;auto deadline=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch()))need(Clock::now()<deadline,"native completion timeout");need(y->type()==akida::TensorType::uint8&&y->size()==128,"native output contract");auto p=y->data<uint8_t>();std::vector<uint8_t>v(p,p+128),x(128);d->read(in,x.data(),128);if(!teacher)source=desc[7]+32;
   std::vector<uint32_t>packet(40);d->read(desc[7],packet.data(),160);dump(out+".packet"+std::to_string(i),packet);
   append(out+".inputs",x);append(out+".scores",v);append(out+".descriptors",std::vector<uint32_t>(desc.begin(),desc.end()));append(out+".input_addresses",std::vector<uint32_t>{in});append(out+".output_addresses",std::vector<uint32_t>{desc[7]});append(out+".learning_modes",modes);
   for(int k=0;k<2;k++){auto ram=capture(banks[k].initial);append(out+".states"+std::to_string(k),ram);if(!std::equal(ram.begin(),ram.end(),states[k].begin()+i*ram.size())){for(size_t j=0;j<ram.size();j++)if(ram[j]!=states[k][i*ram.size()+j]){std::cerr<<"event="<<i<<" bank="<<k<<" word="<<j<<" actual="<<ram[j]<<" expected="<<states[k][i*ram.size()+j]<<"\n";break;}throw std::runtime_error("native learning RAM differs from packed replay");}}
   need(std::equal(x.begin(),x.end(),wantx.begin()+i*128),"native raw/prediction input exact");need(v==std::vector<uint8_t>(want.begin()+i*128,want.begin()+(i+1)*128),"native context responses exact");
  }
  fixed("final");dev->scratch_free(external);dev->unprogram();std::cout<<"{\"status\":\"PASS\",\"mode\":\"native\",\"events\":"<<count<<",\"program_calls\":1,\"weight_restore_calls\":0,\"host_bank_selections\":0,\"host_error_decisions\":0,\"context_writes\":"<<contexts<<",\"current_key_writes\":"<<keys<<",\"phase_actual_writes\":"<<phases<<",\"synthetic_prediction_writes\":"<<overrides<<",\"cleanup\":\"unprogrammed\",\"capture_storage\":\"append-only per boundary\"}\n";
 }catch(const std::exception&e){if(dev)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
