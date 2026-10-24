// Two independently taught populations; categorical communication remains native.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
#include "engine/akida_program_info_generated.h"

struct RelayBank {
 int col,row,id; CPU cpu; std::vector<uint32_t> infer,learn;
 RelayBank(int c,int r,int i,int n,int u,int classes,const std::string& f)
  :col(c),row(r),id(i),cpu(n,u,32,classes,load<uint32_t>(f+".ram")),
   infer(load<uint32_t>(f+".infer")),learn(load<uint32_t>(f+".learn")){}
};

int main(int argc,char**argv){
 std::vector<uint8_t> program; std::unique_ptr<LabDriver> driver;
 akida::HardwareDevicePtr dev; bool programmed=false;
 try{
  need(argc==6,"fixture-prefix stream-prefix output-prefix samples snapshot-flag");
  std::string f=argv[1],stream=argv[2],out=argv[3]; int count=std::stoi(argv[4]);
  bool snapshots=std::stoi(argv[5]); need(count>0&&count<=2048,"bounded relay count");
  int n,u,classes,nb,nf,threshold; std::ifstream layout(f+".layout");
  layout>>n>>u>>classes>>nb>>nf>>threshold;
  need(bool(layout)&&n==512&&u==64&&classes==4&&nb==2&&nf==3&&threshold==27,"frozen relay dimensions");
  std::vector<RelayBank> banks;
  for(int k=0;k<nb;k++){
   int c,r,i,w;layout>>c>>r>>i>>w;
   need(bool(layout)&&w==1282,"relay bank layout");
   banks.emplace_back(c,r,i,n,u,classes,f+".bank"+std::to_string(k));
   need(banks.back().cpu.ram.size()==size_t(w)&&banks.back().infer.size()==52&&banks.back().learn.size()==52,"relay packets");
  }
  std::vector<std::vector<uint32_t>> fixed;
  for(int j=0;j<nf;j++){
   int c,r,i,w;layout>>c>>r>>i>>w;auto a=load<uint32_t>(f+".fixed"+std::to_string(j)+".ram");
   need(bool(layout)&&a.size()==size_t(w+2),"fixed relay packet");fixed.push_back(a);
  }
  auto inputs=load<uint8_t>(stream+".inputs"); auto labels=load<int32_t>(stream+".labels");
  auto masks=load<uint8_t>(stream+".masks");
  need(inputs.size()>=size_t(count*n)&&labels.size()>=size_t(count)&&masks.size()>=size_t(count),"relay stream length");
  driver=std::make_unique<LabDriver>();dev=akida::HardwareDevice::create(driver.get());
  program=load<uint8_t>(f+".bin");auto info=dev->program(program.data(),program.size());programmed=true;
  need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&!info.activation_enabled(),"resident relay IO");
  auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());dev->toggle_learn(true);
  auto capture=[&](const std::vector<uint32_t>& expected){
   std::vector<uint32_t> got(expected.size());got[0]=expected[0];got[1]=expected[1];
   akida::dma::dma_config_read(got.data()+2,uint32_t(got.size()-2),got.data(),impl->dma_config(),impl->mem(),driver.get(),true);
   return got;
  };
  for(int j=0;j<nf;j++){
   auto got=capture(fixed[j]);dump(out+".fixed"+std::to_string(j)+".initial.ram",got);
   need(got==fixed[j],"fixed initial relay readback");
  }
  for(int k=0;k<nb;k++){
   auto got=capture(banks[k].cpu.ram);dump(out+".bank"+std::to_string(k)+".initial.ram",got);
   need(got==banks[k].cpu.ram,"relay initial bank readback");
  }
  need(dev->set_batch_size(1,false)==1,"single relay event queue");auto address=dev->scratch_alloc(n);
  dev->toggle_clock_counter(true);std::vector<uint32_t> idle;
  for(int us:{0,0,5000,50000}){if(us)std::this_thread::sleep_for(std::chrono::microseconds(us));idle.push_back(dev->read_clock_counter());}
  int current_mask=-1;uint64_t polls=0,mode_changes=0;int64_t wall=0,caller=0;
  std::vector<int32_t> outputs;std::vector<int64_t> times,caller_times;std::vector<uint64_t> write_counts;
  std::vector<uint32_t> bounds,cycles,config_bounds,control_bounds;
  for(int i=0;i<count;i++){
   kick_watchdog();int mask=masks[i],label=labels[i];need(mask>=0&&mask<4&&label>=0&&label<4,"relay teaching event");
   if(mask!=current_mask){
    for(int k=0;k<nb;k++){
     auto& packet=(mask&(1<<k))?banks[k].learn:banks[k].infer;
     akida::dma::dma_config_write(packet.data(),packet.size(),impl->dma_config(),impl->external_mem(),driver.get(),true);
    }
    current_mask=mask;mode_changes++;
   }
   auto q0=dev->read_config_clock_counter(),c0=dev->read_clock_counter(),control0=driver->read32(LabDriver::REG+0x28024);
   auto start=ns(),cstart=cpu_ns();driver->write(address,inputs.data()+i*n,n);
   need(dev->enqueue(address,&label),"relay enqueue");auto after_submit=driver->writes;
   akida::TensorUniquePtr y;auto deadline=Clock::now()+std::chrono::seconds(2);
   while(!(y=dev->fetch())){polls++;need(Clock::now()<deadline,"relay completion deadline");std::this_thread::sleep_for(std::chrono::microseconds(50));}
   auto elapsed=ns()-start,spent=cpu_ns()-cstart;wall+=elapsed;caller+=spent;times.push_back(elapsed);caller_times.push_back(spent);
   auto c1=dev->read_clock_counter(),q1=dev->read_config_clock_counter(),control1=driver->read32(LabDriver::REG+0x28024);
   bounds.push_back(c0);bounds.push_back(c1);cycles.push_back(c1-c0);config_bounds.push_back(q0);config_bounds.push_back(q1);
   control_bounds.push_back(control0);control_bounds.push_back(control1);
   // Fetch may acknowledge/drain an output; no host intermediate activation is injected.
   write_counts.push_back(driver->writes-after_submit);
   need(y->type()==akida::TensorType::int32&&y->size()==size_t(classes),"four destination supports");
   auto yp=y->data<int32_t>();std::vector<int32_t> actual(yp,yp+classes);
   outputs.insert(outputs.end(),actual.begin(),actual.end());dump(out+".scores",outputs);dump(out+".latency_ns",times);
   std::vector<uint32_t> x(n/32);for(int j=0;j<n;j++)x[j/32]|=uint32_t(inputs[i*n+j]!=0)<<(j%32);
   auto p=banks[0].cpu.forward(x,label,bool(mask&1));std::vector<int32_t> support(classes);
   for(int j=0;j<u;j++)support[j/16]+=p[j]>threshold;
   int identity=int(std::max_element(support.begin(),support.end())-support.begin());
   x.assign(n/32,0);for(int j=identity*128;j<(identity+1)*128;j++)x[j/32]|=uint32_t(1)<<(j%32);
   p=banks[1].cpu.forward(x,label,bool(mask&2));std::fill(support.begin(),support.end(),0);
   for(int j=0;j<u;j++)support[j/16]+=p[j]>threshold;
   for(int k=0;k<nb;k++){
    auto got=capture(banks[k].cpu.ram);
    if(snapshots)dump(out+".snapshot_"+std::to_string(i)+"_bank"+std::to_string(k)+".ram",got);
    if(got!=banks[k].cpu.ram){std::cerr<<"sample="<<i<<" bank="<<k<<" mask="<<mask<<"\n";throw std::runtime_error("relay per-step RAM mismatch");}
   }
   if(actual!=support){std::cerr<<"sample="<<i<<" mask="<<mask<<" actual=";for(auto q:actual)std::cerr<<q<<",";std::cerr<<" expected=";for(auto q:support)std::cerr<<q<<",";std::cerr<<"\n";throw std::runtime_error("relay final output mismatch");}
  }
  idle.push_back(dev->read_clock_counter());std::this_thread::sleep_for(std::chrono::milliseconds(50));idle.push_back(dev->read_clock_counter());
  dump(out+".idle_counter",idle);dump(out+".cycles",cycles);dump(out+".counter_bounds",bounds);dump(out+".config_bounds",config_bounds);
  dump(out+".control_bounds",control_bounds);dump(out+".caller_ns",caller_times);
  for(int k=0;k<nb;k++)dump(out+".bank"+std::to_string(k)+".ram",capture(banks[k].cpu.ram));
  for(int j=0;j<nf;j++){
   auto got=capture(fixed[j]);dump(out+".fixed"+std::to_string(j)+".final.ram",got);
   need(got==fixed[j],"fixed relay communication changed");
  }
  dump(out+".writes_after_submit",write_counts);dev->scratch_free(address);dev->toggle_clock_counter(false);dev->unprogram();programmed=false;
  std::cout<<"{\"status\":\"PASS\",\"samples\":"<<count<<",\"plastic_banks\":2,\"learning_neurons\":128,\"mode_changes\":"<<mode_changes<<",\"wall_ns\":"<<wall<<",\"thread_cpu_ns\":"<<caller<<",\"polls\":"<<polls<<",\"intermediate_inputs_from_host\":0,\"every_step_RAM_and_output_exact\":true,\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';if(dev&&programmed)try{dev->unprogram();}catch(...){}return 1;}
}
