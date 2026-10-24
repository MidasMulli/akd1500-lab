// Isolated compiler-produced fixed bridge; no learning or auxiliary edits.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"

int main(int argc,char**argv){
 std::vector<uint8_t> program;std::unique_ptr<LabDriver> driver;akida::HardwareDevicePtr dev;bool programmed=false;
 try{
  need(argc==4,"fixture-prefix output-prefix count");std::string f=argv[1],out=argv[2];
  int n,outputs,count,packets; int requested=std::stoi(argv[3]);std::ifstream layout(f+".layout");layout>>n>>outputs>>count>>packets;
  need(bool(layout)&&(n==64||n==512)&&outputs==4&&count==128&&packets>=3&&packets<=10&&requested>0&&requested<=count,"frozen ablation dimensions");
  std::vector<std::vector<uint32_t>> memories;
  for(int i=0;i<packets;i++){
   int col,row,id,words;layout>>col>>row>>id>>words;
   auto a=load<uint32_t>(f+".packet"+std::to_string(i)+".ram");
   need(bool(layout)&&a.size()==size_t(words),"ablation memory packet size");
   need(int(a[0]>>24)==col&&int((a[0]>>16)&255)==row&&((a[0]>>8)&15)==uint32_t(1<<id),"ablation physical target header");
   memories.push_back(a);
  }
  auto inputs=load<uint8_t>(f+".inputs");auto expected=load<int32_t>(f+".expected");
  need(inputs.size()==size_t(n*count)&&expected.size()==size_t(outputs*count),"ablation data dimensions");
  driver=std::make_unique<LabDriver>();dev=akida::HardwareDevice::create(driver.get());
  program=load<uint8_t>(f+".bin");auto info=dev->program(program.data(),program.size());programmed=true;
  need(info.number_of_passes()==1&&!info.can_learn()&&!info.activation_enabled()&&info.input_is_dense(),"ablation fixed native graph");
  auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());
  auto capture=[&](const std::vector<uint32_t>& source){
   auto got=source;
   akida::dma::dma_config_read(got.data()+2,uint32_t(got.size()-2),got.data(),impl->dma_config(),impl->mem(),driver.get(),true);
   return got;
  };
  for(int j=0;j<packets;j++){
   auto got=capture(memories[j]);dump(out+".packet"+std::to_string(j)+".initial",got);
   need(got==memories[j],"physical initial memory does not equal compiled packet");
  }
  need(dev->set_batch_size(1,false)==1,"single event ablation queue");auto address=dev->scratch_alloc(n);
  std::vector<int32_t> scores;
  for(int i=0;i<requested;i++){
   kick_watchdog();driver->write(address,inputs.data()+i*n,n);need(dev->enqueue(address,nullptr),"ablation enqueue");
   akida::TensorUniquePtr y;auto limit=Clock::now()+std::chrono::seconds(2);
   while(!(y=dev->fetch()))need(Clock::now()<limit,"ablation completion deadline");
   need(y->type()==akida::TensorType::int32&&y->size()==4,"ablation output dimensions");
   auto p=y->data<int32_t>();scores.insert(scores.end(),p,p+4);dump(out+".scores",scores);
   if(!std::equal(p,p+4,expected.data()+i*4)){
    for(int j=0;j<packets;j++)dump(out+".packet"+std::to_string(j)+".final",capture(memories[j]));
    std::cerr<<"first_divergence_event="<<i<<"\n";throw std::runtime_error("fixed bridge native output mismatch");
   }
  }
  for(int j=0;j<packets;j++){
   auto got=capture(memories[j]);dump(out+".packet"+std::to_string(j)+".final",got);
   need(got==memories[j],"fixed physical NPU memory changed");
  }
  dev->scratch_free(address);dev->unprogram();programmed=false;
  std::cout<<"{\"status\":\"PASS\",\"samples\":"<<requested<<",\"outputs_exact\":"<<requested*4<<",\"physical_memory_regions_checked\":"<<packets<<",\"initial_and_final_memory_exact\":true,\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception& e){if(dev&&programmed)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
