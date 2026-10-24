// Controlled lower/higher record-address diagnostic, separate from scale-up.
// Fresh near-budget record-storage proof; all harness writes are guarded.
// Prospective capacity variant of fnp2_bench.cpp; no original execution changed.
// Qualify externally stored learning records through existing runtime APIs.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "fnp2_mem_conf_reg.h"
#include "record_write_guard.hpp"

int main(int argc,char**argv){
 std::vector<uint8_t> program;
 std::unique_ptr<RecordWriteGuard> driver;akida::HardwareDevicePtr dev;bool programmed=false;
 try{
  need(argc==6,"cpu|native fixture output count learning");std::string mode=argv[1],f=argv[2],out=argv[3];int count=std::stoi(argv[4]);bool learn=std::stoi(argv[5]);
  int u;{std::ifstream meta(f+".units");meta>>u;need(bool(meta)&&u==8192,"frozen capacity cell");}
  need((mode=="cpu"||mode=="native")&&(count==8||count==64),"bounded FNP2 contract");
  auto initial=load<uint32_t>(f+".raw"),cpu_initial=load<uint32_t>(f+".cpu_ram");auto x=load<uint8_t>(f+".inputs");auto labels=load<int32_t>(f+".labels");
  need(initial.size()==size_t(u*20)&&cpu_initial.size()==size_t(u*20+2)&&x.size()==64*512&&labels.size()==64,"FNP2 fixture dimensions");
  need(std::equal(initial.begin(),initial.end(),cpu_initial.begin()+2),"explicit synthetic CPU header");
  CPU ref(512,u,32,2,cpu_initial);std::vector<int32_t> scores;std::vector<int64_t> wall,caller;
  if(mode=="cpu"){
   for(int repeat=0;repeat<5;repeat++){
    CPU cpu(512,u,32,2,cpu_initial);
    for(int i=0;i<count;i++){
     auto t=ns(),c=cpu_ns();std::vector<uint32_t> bits(16);for(int k=0;k<512;k++)bits[k/32]|=uint32_t(x[i*512+k])<<(k%32);
     auto y=cpu.forward(bits,labels[i],learn);caller.push_back(cpu_ns()-c);wall.push_back(ns()-t);
     if(!repeat)scores.insert(scores.end(),y.begin(),y.end());
    }
    if(repeat==4)ref=std::move(cpu);
   }
   dump(out+".scores",scores);dump(out+".raw",std::vector<uint32_t>(ref.ram.begin()+2,ref.ram.end()));dump(out+".latency_ns",wall);dump(out+".caller_ns",caller);
   std::cout<<"{\"status\":\"PASS\",\"mode\":\"cpu\",\"samples\":"<<count<<",\"repeats\":5,\"physical_device_opened\":false}\n";return 0;
  }
  driver=std::make_unique<RecordWriteGuard>();dev=akida::HardwareDevice::create(driver.get());program=load<uint8_t>(f+".bin");auto info=dev->program(program.data(),program.size());programmed=true;
  need(info.number_of_passes()==1&&info.can_learn()&&!info.learning_on_fnp3()&&info.fnp2_required_memory()==(u*80),"FNP2 program contract");
  dev->toggle_learn(true);need(dev->learn_mem_size()==initial.size(),"FNP2 record count");
  uint32_t address=driver->read32(akida::fnp2_memory_conf(LabDriver::REG,0));
  need(address>=LabDriver::RAM+256&&uint64_t(address)+(u*80)<=uint64_t(LabDriver::RAM)+LabDriver::SIZE,"FNP2 storage must stay in owned card SRAM");
  dump(out+".storage",std::vector<uint32_t>{address,uint32_t(u*80),LabDriver::RAM,LabDriver::SIZE});
  auto capture=[&](){std::vector<uint32_t> a(initial.size()),direct(initial.size());if(!learn)dev->toggle_learn(true);dev->learn_mem(a.data());driver->read(address,direct.data(),direct.size()*4);need(a==direct,"API and addressed storage readback differ");if(!learn)dev->toggle_learn(false);return a;};
  need(capture()==initial,"initial complete FNP2 storage");dump(out+".initial",initial);dev->toggle_learn(learn);need(dev->set_batch_size(1,false)==1,"one-image contract");auto input=dev->scratch_alloc(512);
  driver->protect(address,u*80);auto memory=dev->memory();std::ofstream snaps(out+".snapshots",std::ios::binary);
  for(int i=0;i<count;i++){
   kick_watchdog();auto t=ns(),c=cpu_ns();driver->write(input,x.data()+i*512,512);need(dev->enqueue(input,learn?&labels[i]:nullptr),"FNP2 enqueue");akida::TensorUniquePtr y;auto stop=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch()))need(Clock::now()<stop,"owned FNP2 output deadline");
   caller.push_back(cpu_ns()-c);wall.push_back(ns()-t);need(y->type()==akida::TensorType::int32&&y->size()==size_t(u),"FNP2 output dimensions");auto values=y->data<int32_t>();scores.insert(scores.end(),values,values+u);dump(out+".scores",scores);
   std::vector<uint32_t> bits(16);for(int k=0;k<512;k++)bits[k/32]|=uint32_t(x[i*512+k])<<(k%32);auto want=ref.forward(bits,labels[i],learn);
   auto ram=capture();dump(out+".guard_counts",std::vector<uint64_t>{driver->attempts,driver->guarded_writes});snaps.write((char*)ram.data(),ram.size()*4);snaps.flush();dump(out+".raw",ram);dump(out+".expected_raw",std::vector<uint32_t>(ref.ram.begin()+2,ref.ram.end()));
   need(std::equal(want.begin(),want.end(),values),"FNP2 exact potential mismatch");need(std::equal(ram.begin(),ram.end(),ref.ram.begin()+2),"FNP2 exact learning-state mismatch");
  }
  snaps.close();dump(out+".latency_ns",wall);dump(out+".caller_ns",caller);dev->scratch_free(input);dev->unprogram();programmed=false;
  std::cout<<"{\"status\":\"PASS\",\"mode\":\"native\",\"samples\":"<<count<<",\"learning\":"<<(learn?"true":"false")<<",\"records\":"<<u<<",\"storage_address\":"<<address<<",\"storage_bytes\":"<<(u*80)<<",\"memory_usage\":"<<memory.first<<",\"memory_peak\":"<<memory.second<<",\"guarded_host_record_write_attempts\":"<<driver->attempts<<",\"guarded_host_write_calls\":"<<driver->guarded_writes<<",\"storage_domain\":\"card_system_SRAM\",\"host_DDR_used\":false,\"exact_scores_and_full_records\":true,\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception&e){if(programmed&&dev)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
