// Direct bounded learning-pipeline probe using a known runtime control only.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"

int main(int argc,char**argv) {
 std::vector<uint8_t> program; std::unique_ptr<LabDriver> driver;
 akida::HardwareDevicePtr dev; bool programmed=false;
 try {
  need(argc==5,"fixture stream output pipeline-override(0|1)");
  std::string f=argv[1],s=argv[2],out=argv[3]; int override=std::stoi(argv[4]);
  need(override==0||override==1,"frozen pipeline treatment");
  constexpr int n=512,u=64,count=64,batch=8;
  auto initial=load<uint32_t>(f+".ram"); CPU reference(n,u,32,2,initial);
  auto inputs=load<uint8_t>(s+".inputs");auto labels=load<int32_t>(s+".labels");
  need(inputs.size()==count*n&&labels.size()==count,"exact queue stream");
  driver=std::make_unique<LabDriver>();dev=akida::HardwareDevice::create(driver.get());
  program=load<uint8_t>(f+".bin");auto info=dev->program(program.data(),program.size());programmed=true;
  need(info.can_learn()&&info.number_of_passes()==1&&!info.activation_enabled(),"single-pass native learner");
  dev->toggle_learn(true);need(dev->set_batch_size(9,false)==9,"nine-slot learning queue");
  dev->toggle_clock_counter(true);
  auto control_before=driver->read32(LabDriver::REG+0x28024);
  auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());impl->pipeline(bool(override));
  auto control_after=driver->read32(LabDriver::REG+0x28024);
  need(((control_before>>8)&3)==0&&((control_after>>8)&3)==uint32_t(override),"direct pipeline register readback");
  need((control_before&~uint32_t(3<<8))==(control_after&~uint32_t(3<<8)),"only selected end-mode bits changed");
  auto got_initial=dev->learn_mem();dump(out+".initial.ram",got_initial);need(got_initial==initial,"initial queue RAM");
  std::vector<uint32_t> addresses;
  for(int j=0;j<batch;j++)addresses.push_back(dev->scratch_alloc(n));
  std::vector<int32_t> scores;std::vector<uint32_t> cycles,bounds;std::vector<int64_t> times;
  std::ofstream snapshots(out+".snapshots",std::ios::binary);
  for(int first=0;first<count;first+=batch) {
   kick_watchdog();
   for(int j=0;j<batch;j++)driver->write(addresses[j],inputs.data()+(first+j)*n,n);
   driver->hold_queue();
   for(int j=0;j<batch;j++)need(dev->enqueue(addresses[j],&labels[first+j]),"queued teaching input");
   uint32_t target=driver->last_descriptor[0]>>16;need(target!=0,"captured teaching job ID");
   uint32_t c0=dev->read_clock_counter();auto t=ns();auto writes=driver->writes;driver->release_queue();
   auto deadline=Clock::now()+std::chrono::seconds(2);
   while((driver->read32(LabDriver::REG+0x28010)>>16)!=target)need(Clock::now()<deadline,"learning burst deadline");
   need(driver->writes==writes+1,"no host weights/inputs written during teaching burst");
   std::vector<std::vector<int32_t>> actual;
   for(int j=0;j<batch;j++) {
    akida::TensorUniquePtr y;deadline=Clock::now()+std::chrono::seconds(2);
    while(!(y=dev->fetch()))need(Clock::now()<deadline,"teaching output deadline");
    need(y->type()==akida::TensorType::int32&&y->size()==u,"queue output dimensions");
    auto p=y->data<int32_t>();actual.emplace_back(p,p+u);scores.insert(scores.end(),p,p+u);
   }
   times.push_back(ns()-t);auto c1=dev->read_clock_counter();bounds.push_back(c0);bounds.push_back(c1);cycles.push_back(c1-c0);
   dump(out+".scores",scores);dump(out+".cycles",cycles);dump(out+".counter_bounds",bounds);dump(out+".latency_ns",times);
   auto captured=dev->learn_mem();snapshots.write((char*)captured.data(),captured.size()*4);snapshots.flush();
   for(int j=0;j<batch;j++) {
    std::vector<uint32_t> x(n/32);for(int k=0;k<n;k++)x[k/32]|=uint32_t(inputs[(first+j)*n+k]!=0)<<(k%32);
    auto expected=reference.forward(x,labels[first+j],true);
    if(expected!=actual[j]){std::cerr<<"native mismatch event="<<first+j<<"\n";throw std::runtime_error("queued teaching output changed");}
   }
   need(captured==reference.ram,"queued teaching complete RAM differs");
  }
  snapshots.close();dump(out+".final.ram",dev->learn_mem());
  for(auto a:addresses)dev->scratch_free(a);
  impl->pipeline(false);dev->toggle_clock_counter(false);dev->unprogram();programmed=false;
  std::cout<<"{\"status\":\"PASS\",\"samples\":64,\"batch\":8,\"pipeline_override\":"<<override
   <<",\"control_before\":"<<control_before<<",\"control_after\":"<<control_after
   <<",\"all_outputs_and_burst_RAM_exact\":true,\"no_host_writes_during_released_bursts\":true,\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception& e){
  if(driver&&driver->held)try{driver->release_queue();}catch(...){}
  if(dev&&programmed)try{static_cast<akida::HardwareDeviceImpl*>(dev.get())->pipeline(false);dev->toggle_clock_counter(false);dev->unprogram();}catch(...){}
  std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;
 }
}
