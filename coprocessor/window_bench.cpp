// Own native window dispatcher; never updates a connection word on the CPU.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "window_driver.hpp"
#include "fnp2_mem_conf_reg.h"
int main(int argc,char**argv){
 std::vector<uint8_t> full_info,tile0_info,tile1_info,full_data,tile_data;
 std::unique_ptr<WindowDriver> driver;akida::HardwareDevicePtr dev;bool programmed=false;
 std::string out;
 auto counters=[&](){if(driver&&!out.empty())dump(out+".guard",std::vector<uint64_t>{driver->guarded_record_attempts,driver->guarded_calls,driver->ddr_writes});};
 try{
  need(argc==5,"mode fixture-directory output-prefix count");std::string mode=argv[1],f=argv[2];out=argv[3];int count=std::stoi(argv[4]);
  need(mode=="inspect"||mode=="direct"||mode=="window"||mode=="frozen"||mode=="cpu-learn"||mode=="cpu-frozen","known window mode");need((mode=="inspect"&&count==0)||(count>=1&&count<=8),"bounded window count");
  constexpr int n=512,u=11264,tile=5632,stride=20;uint32_t record,full_addr,tile_addr;
  {std::ifstream p(f+"/dims");p>>record>>full_addr>>tile_addr;need(bool(p),"window placement dimensions");}
  auto initial=load<uint32_t>(f+"/initial.raw"),cpu_init=load<uint32_t>(f+"/cpu.raw");auto inputs=load<uint8_t>(f+"/inputs.raw");auto labels=load<int32_t>(f+"/labels.raw");
  need(initial.size()==u*stride&&cpu_init.size()==initial.size()+2&&inputs.size()==8*n&&labels.size()==8,"frozen window fixture dimensions");
  need(std::equal(initial.begin(),initial.end(),cpu_init.begin()+2),"CPU synthetic header only");
  CPU ref(n,u,32,2,cpu_init);std::vector<int32_t> pre,recall;std::vector<int64_t> times;
  bool learning=mode=="window"||mode=="direct"||mode=="cpu-learn";
  auto bits_at=[&](int i){std::vector<uint32_t>x(n/32);for(int k=0;k<n;k++)x[k/32]|=uint32_t(inputs[i*n+k])<<(k%32);return x;};
  if(mode=="cpu-learn"||mode=="cpu-frozen"){
   for(int r=0;r<5;r++){CPU cpu(n,u,32,2,cpu_init);for(int i=0;i<count;i++){auto x=bits_at(i);auto t=ns();auto y=cpu.forward(x,labels[i],learning);auto q=cpu.forward(x,0,false);times.push_back(ns()-t);if(!r){pre.insert(pre.end(),y.begin(),y.end());recall.insert(recall.end(),q.begin(),q.end());}}if(r==4)ref=std::move(cpu);}
   dump(out+".pre",pre);dump(out+".recall",recall);dump(out+".final",std::vector<uint32_t>(ref.ram.begin()+2,ref.ram.end()));dump(out+".timing",times);
   std::cout<<"{\"status\":\"PASS\",\"mode\":\"cpu\",\"physical_device_opened\":false,\"repeats\":5,\"teaching_inputs\":"<<count<<"}\n";return 0;
  }
  full_info=load<uint8_t>(f+"/full.info");tile0_info=load<uint8_t>(f+"/tile0.info");tile1_info=load<uint8_t>(f+"/tile1.info");full_data=load<uint8_t>(f+"/full.data");tile_data=load<uint8_t>(f+"/tile.data");
  driver=std::make_unique<WindowDriver>();
  need(driver->in_ddr(full_addr,full_data.size())&&driver->in_ddr(tile_addr,tile_data.size())&&uint64_t(tile_addr)+tile_data.size()<=full_addr,"disjoint external program regions");
  driver->write(full_addr,full_data.data(),full_data.size());driver->write(tile_addr,tile_data.data(),tile_data.size());
  auto full_ram=[&](){std::vector<uint32_t>a(initial.size());driver->read(record,a.data(),a.size()*4);return a;};
  need(full_ram()==initial,"external initial full records");driver->protect(record,initial.size()*4);dev=akida::HardwareDevice::create(driver.get());
  std::ofstream snaps(out+".snapshots",std::ios::binary);int forwards=0;uint32_t input=0;
  auto view=[&](int label,bool learn){
   need(!programmed,"unprogram prior view");bool is_full=label<0;auto& info_buf=is_full?full_info:(label==0?tile0_info:tile1_info);uint32_t data_addr=is_full?full_addr:tile_addr;
   auto info=dev->program_external_data(info_buf.data(),info_buf.size(),data_addr);programmed=true;int width=is_full?u:tile;
   need(info.can_learn()&&!info.learning_on_fnp3()&&info.number_of_passes()==1&&info.fnp2_required_memory()==width*stride*4,"native view contract");
   uint32_t expected=record+(is_full?0:label*tile*stride*4);need(driver->read32(akida::fnp2_memory_conf(LabDriver::REG,0))==expected,"registered native view must point into persistent bank");
   dev->toggle_learn(true);std::vector<uint32_t>a(dev->learn_mem_size());dev->learn_mem(a.data());auto all=full_ram();need(a.size()==size_t(width*stride)&&std::equal(a.begin(),a.end(),all.begin()+(is_full?0:label*tile*stride)),"API view and full bank differ");
   dev->toggle_learn(learn);need(dev->set_batch_size(1,false)==1,"one-image view");input=dev->scratch_alloc(n);
  };
  auto clear=[&](){dev->scratch_free(input);dev->unprogram();programmed=false;};
  auto forward=[&](int i,bool learn,int expected_width){
   driver->write(input,inputs.data()+i*n,n);int32_t label=labels[i];need(dev->enqueue(input,learn?&label:nullptr),"native window enqueue");akida::TensorUniquePtr y;auto stop=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch()))need(Clock::now()<stop,"native window call deadline");
   need(y->type()==akida::TensorType::int32&&y->size()==size_t(expected_width),"native view output shape");forwards++;return std::vector<int32_t>(y->data<int32_t>(),y->data<int32_t>()+expected_width);
  };
  auto capture=[&](){auto a=full_ram();snaps.write((char*)a.data(),a.size()*4);snaps.flush();dump(out+".final",a);dump(out+".expected",std::vector<uint32_t>(ref.ram.begin()+2,ref.ram.end()));counters();return a;};
  dump(out+".storage",std::vector<uint32_t>{record,uint32_t(initial.size()*4),full_addr,tile_addr,WindowDriver::DDR,WindowDriver::EXTENT});
  if(mode=="inspect"){
   for(int v:{-1,0,1}){view(v,false);clear();need(capture()==initial,"view programming changed persistent records");}
  }else{
   for(int i=0;i<count;i++){
    kick_watchdog();auto start=ns();bool direct=mode=="direct";view(direct?-1:labels[i],learning);auto actual=forward(i,learning,direct?u:tile);auto want=ref.forward(bits_at(i),labels[i],learning);
    if(!direct)want=std::vector<int32_t>(want.begin()+labels[i]*tile,want.begin()+(labels[i]+1)*tile);
    pre.insert(pre.end(),actual.begin(),actual.end());dump(out+".pre",pre);auto ram=capture();need(actual==want,"native view potentials differ");need(std::equal(ram.begin(),ram.end(),ref.ram.begin()+2),"native complete persistent learning state differs");clear();
    view(-1,false);auto q=forward(i,false,u);auto expected=ref.forward(bits_at(i),0,false);recall.insert(recall.end(),q.begin(),q.end());dump(out+".recall",recall);ram=capture();need(q==expected,"full model recall differs");need(std::equal(ram.begin(),ram.end(),ref.ram.begin()+2),"read-only recall modified records");clear();times.push_back(ns()-start);
   }
  }
  snaps.close();counters();need(driver->guarded_record_attempts==0,"host must never write live weights");dump(out+".timing",times);
  std::cout<<"{\"status\":\"PASS\",\"mode\":\""<<mode<<"\",\"forwards\":"<<forwards<<",\"records\":11264,\"window_records\":5632,\"host_record_writes\":"<<driver->guarded_record_attempts<<",\"control_storage\":\"card_SRAM\",\"record_storage\":\"host_DDR\",\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception&e){counters();if(programmed&&dev)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
