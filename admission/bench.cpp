// Own-source two-bank input-admission instrument. Licensed runtime is external.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"

struct NativeBank {
 int n,u;CPU cpu;std::vector<uint32_t> infer,learn;
 NativeBank(int nn,int uu,const std::string& f):n(nn),u(uu),cpu(nn,uu,32,2,load<uint32_t>(f+".ram")),infer(load<uint32_t>(f+".infer")),learn(load<uint32_t>(f+".learn")){}
};
static std::vector<uint8_t> fc(const std::vector<uint8_t>&x,const std::vector<int8_t>&w,const std::vector<int32_t>&t){
 need(w.size()==x.size()*t.size(),"fixed matrix dimensions");std::vector<uint8_t>y(t.size());
 for(size_t j=0;j<t.size();j++){int v=0;for(size_t k=0;k<x.size();k++)v+=x[k]*w[k*t.size()+j];y[j]=v>t[j];}return y;
}
static std::vector<uint8_t> bank(NativeBank&b,const std::vector<uint8_t>&x,int label,bool second,std::vector<int32_t>&p){
 std::vector<uint32_t>bits(b.n/32);for(int k=0;k<b.n;k++)bits[k/32]|=uint32_t(x[k])<<(k%32);
 p=b.cpu.forward(bits,label<0?0:label,label>=0);std::vector<uint8_t>y(b.u);
 for(int j=0;j<b.u;j++){bool plastic=j<4||(j>=(second?16:24)&&j<(second?20:28));y[j]=p[j]>(plastic?27:31);}return y;
}
int main(int argc,char**argv){
 std::unique_ptr<LabDriver>d;akida::HardwareDevicePtr dev;
 try{
  need(argc==6,"cpu/native fixture arm stage output");std::string mode=argv[1],f=argv[2],arm=argv[3],stage=argv[4],out=argv[5];need(mode=="cpu"||mode=="native","mode");
  auto plan=load<int32_t>(f+"/"+stage+".plan");int count=plan.size()/4;need(count>0&&count<=1024&&plan.size()==size_t(count*4),"finite plan");
  NativeBank b0(1024,48,f+"/bank0"),b1(1024,32,f+"/bank1");std::vector<std::vector<int8_t>>w;std::vector<std::vector<int32_t>>t;
  for(int j=0;j<4;j++){w.push_back(load<int8_t>(f+"/fixed"+std::to_string(j)+".weights"));t.push_back(load<int32_t>(f+"/fixed"+std::to_string(j)+".threshold"));}
  std::vector<uint8_t>last(128),query_in,want,wantx;std::vector<uint32_t>s0,s1;std::vector<int32_t>ps0,ps1;
  for(int i=0;i<count;i++){
   int key=plan[4*i],teacher=plan[4*i+1],label=plan[4*i+2];need(key>=0&&key<12&&(teacher==0||teacher==1)&&(label>=-1&&label<=1),"plan values");auto x=teacher?query_in:last;
   if(!teacher){std::fill(x.begin()+32,x.begin()+64,0);for(int b=0;b<4;b++)for(int c=0;c<4;c++)x[32+b*8+((key>>b)&1)*4+c]=1;std::fill(x.begin()+64,x.begin()+68,0);x[64]=1;query_in=x;}else{std::fill(x.begin()+64,x.begin()+68,0);if(plan[4*i+3])std::fill(x.begin(),x.begin()+32,0);}
   auto z=fc(fc(x,w[0],t[0]),w[1],t[1]);std::vector<int32_t>p0,p1;auto y0=bank(b0,z,label,false,p0);auto y1=bank(b1,fc(y0,w[2],t[2]),label,true,p1);auto y=fc(y1,w[3],t[3]);if(!teacher)last=y;
   want.insert(want.end(),y.begin(),y.end());wantx.insert(wantx.end(),x.begin(),x.end());s0.insert(s0.end(),b0.cpu.ram.begin(),b0.cpu.ram.end());s1.insert(s1.end(),b1.cpu.ram.begin(),b1.cpu.ram.end());ps0.insert(ps0.end(),p0.begin(),p0.end());ps1.insert(ps1.end(),p1.begin(),p1.end());
  }
  dump(out+".potentials0",ps0);dump(out+".potentials1",ps1);
  if(mode=="cpu"){dump(out+".scores",want);dump(out+".inputs",wantx);dump(out+".states0",s0);dump(out+".states1",s1);std::cout<<"{\"status\":\"PASS\",\"mode\":\"cpu\",\"events\":"<<count<<"}\n";return 0;}
  d=std::make_unique<LabDriver>();d->dma_read_min=128;dev=akida::HardwareDevice::create(d.get());auto program=load<uint8_t>(f+"/model.bin");auto info=dev->program(program.data(),program.size());need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&info.output_is_dense()&&info.activation_enabled(),"one resident dense binary model");dev->toggle_learn(true);bool learning=true;auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());need(dev->set_batch_size(4,false)==4,"four nonaliasing native carry buffers");impl->pipeline(false);
  auto capture=[&](const std::vector<uint32_t>&p){auto v=p;akida::dma::dma_config_read(v.data()+2,v.size()-2,v.data(),impl->dma_config(),impl->mem(),d.get(),true);return v;};
  auto mode_set=[&](bool v){dev->toggle_learn(v);impl->pipeline(false);for(auto b:{&b0,&b1}){auto &p=v?b->learn:b->infer;akida::dma::dma_config_write(p.data(),p.size(),impl->dma_config(),impl->external_mem(),d.get(),true);}learning=v;};
  mode_set(true);auto init0=load<uint32_t>(f+"/bank0.ram"),init1=load<uint32_t>(f+"/bank1.ram");auto actual0=capture(init0),actual1=capture(init1);dump(out+".initial0",actual0);dump(out+".initial1",actual1);need(actual0==init0&&actual1==init1,"both initial bank RAMs exact");
  auto fixed=[&](const char*tag){for(int j=0;j<4;j++){auto p=load<uint32_t>(f+"/fixed"+std::to_string(j)+".ram");auto v=capture(p);dump(out+".fixed"+std::to_string(j)+"."+tag,v);need(p==v,"fixed native policy/relay RAM unchanged");}};fixed("initial");
  auto external=dev->scratch_alloc(128);std::vector<uint8_t>zero(128);d->write(external,zero.data(),128);uint32_t previous_query_output=external,query_source=external;std::vector<uint8_t>ys,xs;std::vector<uint32_t>as,is,ds,states0,states1;std::vector<int32_t>modes;int current_writes=0,phase_writes=0,sever_writes=0,mode_transitions=0;
  for(int i=0;i<count;i++){
   kick_watchdog();int key=plan[4*i],teacher=plan[4*i+1],label=plan[4*i+2];bool request=label>=0;if(request!=learning){mode_set(request);mode_transitions++;}modes.push_back(request);
   uint32_t in=teacher?query_source:previous_query_output;
   if(!teacher){std::vector<uint8_t>raw(32);for(int b=0;b<4;b++)for(int c=0;c<4;c++)raw[b*8+((key>>b)&1)*4+c]=1;d->write(in+32,raw.data(),32);current_writes++;query_source=in;}
   uint32_t phase=teacher?0:1;d->write(in+64,&phase,4);phase_writes++;if(plan[4*i+3]){d->write(in,zero.data(),32);sever_writes++;}
   d->hold_queue();need(dev->enqueue(in,label>=0?&label:nullptr),"observation/actual-outcome enqueue");auto desc=d->last_descriptor;need(desc[1]==in&&((desc[8]>>16)&1023)==uint32_t(label>=0?label+1:0),"causal input pointer and actual outcome only");need(desc[7]+32!=in,"output cannot overwrite current input");as.push_back(desc[7]);is.push_back(in);ds.insert(ds.end(),desc.begin(),desc.end());d->release_queue();
   akida::TensorUniquePtr y;auto deadline=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch()))need(Clock::now()<deadline,"forward deadline");need(y->type()==akida::TensorType::uint8&&y->size()==128,"raw native prediction and retained key");auto p=y->data<uint8_t>();ys.insert(ys.end(),p,p+128);std::vector<uint8_t>x(128);d->read(in,x.data(),128);xs.insert(xs.end(),x.begin(),x.end());if(!teacher)previous_query_output=desc[7]+32;
   auto r0=capture(init0),r1=capture(init1);states0.insert(states0.end(),r0.begin(),r0.end());states1.insert(states1.end(),r1.begin(),r1.end());
   std::vector<uint32_t>packet(40);d->read(as.back(),packet.data(),160);dump(out+".packet"+std::to_string(i),packet);
   dump(out+".scores",ys);dump(out+".inputs",xs);dump(out+".states0",states0);dump(out+".states1",states1);dump(out+".descriptors",ds);dump(out+".output_addresses",as);dump(out+".input_addresses",is);dump(out+".learning_modes",modes);
   need(std::equal(x.begin(),x.end(),wantx.begin()+i*128),"native carried key / phase differs");need(std::equal(p,p+128,want.begin()+i*128),"native prediction / packed replay differs");
   if(!std::equal(r0.begin(),r0.end(),s0.begin()+i*init0.size())||!std::equal(r1.begin(),r1.end(),s1.begin()+i*init1.size())){for(int k=0;k<2;k++){auto &a=k?r1:r0;auto &b=k?s1:s0;for(size_t j=0;j<a.size();j++)if(a[j]!=b[i*a.size()+j]){std::cerr<<"event="<<i<<" bank="<<k<<" word="<<j<<" actual="<<a[j]<<" expected="<<b[i*a.size()+j]<<"\n";break;}}throw std::runtime_error("native bank RAM / packed replay differs");}
  }
  fixed("final");dev->scratch_free(external);dev->unprogram();std::cout<<"{\"status\":\"PASS\",\"mode\":\"native\",\"events\":"<<count<<",\"program_calls\":1,\"weight_restore_calls\":0,\"host_bank_decisions\":0,\"current_key_writes\":"<<current_writes<<",\"phase_writes\":"<<phase_writes<<",\"history_ablation_writes\":"<<sever_writes<<",\"uniform_mode_transitions\":"<<mode_transitions<<",\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception&e){if(dev)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
