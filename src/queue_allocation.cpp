// Zero-forward cleanup/allocation diagnostic. Program storage outlives cleanup.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
int main(int argc,char**argv){
 std::vector<uint8_t> program;std::unique_ptr<LabDriver> driver;akida::HardwareDevicePtr dev;bool programmed=false;
 try{
  need(argc==3,"owned-fixture output-prefix");std::string f=argv[1],out=argv[2];program=load<uint8_t>(f+".bin");
  driver=std::make_unique<LabDriver>();dev=akida::HardwareDevice::create(driver.get());auto info=dev->program(program.data(),program.size());programmed=true;
  need(info.number_of_passes()==1&&info.can_learn(),"known owned single-pass program");dev->toggle_learn(true);auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());
  std::vector<std::vector<uint32_t>> packets;
  for(int j=0;j<3;j++)packets.push_back(load<uint32_t>(f+".bank"+std::to_string(j)+".ram"));
  for(int j=0;j<3;j++)packets.push_back(load<uint32_t>(f+".fixed"+std::to_string(j)+".ram"));
  auto capture=[&](const std::vector<uint32_t>&a){std::vector<uint32_t>b(a.size());b[0]=a[0];b[1]=a[1];akida::dma::dma_config_read(b.data()+2,uint32_t(b.size()-2),b.data(),impl->dma_config(),impl->mem(),driver.get(),true);return b;};
  for(size_t j=0;j<packets.size();j++){auto a=capture(packets[j]);dump(out+".packet"+std::to_string(j)+".initial",a);need(a==packets[j],"allocation initial owned memory");}
  std::vector<uint32_t> observed;
  for(uint32_t request:{1,9,33,65,128,129}){auto got=dev->set_batch_size(request,false);observed.push_back(request);observed.push_back(uint32_t(got));dump(out+".requested_effective",observed);need(got==std::min(request,uint32_t(128)),"source-predicted runtime capacity");}
  for(size_t j=0;j<packets.size();j++){auto a=capture(packets[j]);dump(out+".packet"+std::to_string(j)+".final",a);need(a==packets[j],"queue allocation changes no native parameter memory");}
  dev->unprogram();programmed=false;dev.reset();driver.reset();
  std::cout<<"{\"status\":\"ALLOCATION_AND_CLEANUP_PASS\",\"physical_forwards\":0,\"requested129_effective\":128,\"parameter_memory_unchanged\":true,\"cleanup\":\"unprogrammed_and_handle_closed\"}\n";
 }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';if(dev&&programmed)try{dev->unprogram();}catch(...){}return 1;}
}
