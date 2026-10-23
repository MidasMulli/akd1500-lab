#define main inherited_probe_main
#define kick_watchdog inherited_short_watchdog
#include "native_base.cpp"
#undef kick_watchdog
#undef main
#include "dma_engine_ops.h"
#include <cstdlib>
#include <ctime>
static int64_t thread_ns(){timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return int64_t(t.tv_sec)*1000000000+t.tv_nsec;}
extern "C" void kick_watchdog(){if(us()>480000000)throw std::runtime_error("owned service session watchdog");}

int main(int argc,char**argv){
 std::unique_ptr<Driver>dr;akida::HardwareDevicePtr dev;std::vector<uint8_t> p;
 try{
  require(argc==7,"program trace plan primary-inference primary-learning secondary-inference");
  p=file(argv[1]);dr=std::make_unique<Driver>(argv[2]);dev=akida::HardwareDevice::create(dr.get());auto info=dev->program(p.data(),p.size());
  require(info.number_of_passes()==1&&info.activation_enabled()&&info.input_is_dense()&&info.output_is_dense()&&info.can_learn(),"single-pass dense native learner");
  auto*impl=dynamic_cast<akida::HardwareDeviceImpl*>(dev.get());require(impl,"impl");
  std::vector<std::pair<akida::hw::Ident,size_t>>all;std::ifstream plan(argv[3]);unsigned c,r,i,n;
  while(plan>>c>>r>>i>>n)all.push_back({{c,r,i},n});require(all.size()==6&&all[0].second==1282&&all[1].second==1282,"two memory blocks, four fixed");
  std::vector<std::vector<uint32_t>>modes;
  for(int z=4;z<7;z++){auto b=file(argv[z]);require(b.size()==208,"52-word mode packet");std::vector<uint32_t>w(52);memcpy(w.data(),b.data(),208);require((w[0]&255)==8,"register-only packet");modes.push_back(w);}
  auto mode=[&](int k){auto&w=modes[k];akida::dma::dma_config_write(w.data(),w.size(),impl->dma_config(),impl->external_mem(),dr.get(),true);};
  auto snap=[&](int id){dr->mark("snapshot_"+std::to_string(id));std::cout<<"{\"kind\":\"snapshot\",\"id\":"<<id<<",\"words\":[";
   for(size_t f=0;f<all.size();f++){auto who=all[f].first;auto count=all[f].second;std::vector<uint32_t>w(count);akida::dma::format_config_header(w.data(),who,akida::dma::Target::FnpWeights,count-2,0);akida::dma::dma_config_read(w.data()+2,count-2,w.data(),impl->dma_config(),impl->mem(),dr.get(),true);if(f)std::cout<<',';arrayjson(w);}std::cout<<"]}\n"<<std::flush;};
  dev->toggle_learn(true);mode(1);require(dev->set_batch_size(3,false)==3,"fixed three-slot SDK queue");impl->pipeline(false);
  std::array<uint32_t,2>slabs{dev->scratch_alloc(128),dev->scratch_alloc(128)};auto initial=dev->scratch_alloc(64);
  std::array<uint8_t,128>guard;guard.fill(165);for(auto a:slabs)dr->write(a,guard.data(),guard.size());
  std::array<uint8_t,64>init{};init[3]=1;dr->write(initial,init.data(),64);uint32_t carry=initial;
  snap(-1);auto allocation=impl->mem()->report();
  std::cout<<"{\"kind\":\"ready\",\"initial_address\":"<<initial<<",\"slabs\":";arrayjson(slabs);std::cout<<",\"allocated_bytes\":"<<allocation.first<<",\"peak_bytes\":"<<allocation.second<<"}\n"<<std::flush;
  bool pending_observation=false;bool transition_learning=true,history_learning=true;size_t commands=0;std::string cmd;
  while(std::cin>>cmd){
   kick_watchdog();require(++commands<=2000,"command budget");
   if(cmd=="EXIT")break;
   if(cmd=="S"){int id;std::cin>>id;snap(id);continue;}
   if(cmd=="R"){int id;std::cin>>id;require(!pending_observation,"new request between completed observations only");
    dr->mark("request_start_"+std::to_string(id));dr->write(initial,init.data(),64);carry=initial;
    std::cout<<"{\"kind\":\"request_start\",\"id\":"<<id<<",\"address\":"<<initial<<",\"bytes\":";arrayjson(init);std::cout<<"}\n"<<std::flush;continue;}
   if(cmd=="M"){int a,b;std::cin>>a>>b;require((a==0||a==1)&&(b==0||b==1),"mode bits");require(!pending_observation,"mode changes between actions only");dev->toggle_learn(true);mode(a?1:0);if(!b)mode(2);transition_learning=a;history_learning=b;std::cout<<"{\"kind\":\"mode\",\"transition\":"<<a<<",\"history\":"<<b<<"}\n"<<std::flush;continue;}
   require(cmd=="Q"||cmd=="O","command type");int id;std::cin>>id;int count=cmd=="Q"?2:1;int outcome=-1;
   if(cmd=="Q")require(!pending_observation,"query requires completed prior observation");
   else{
    require(pending_observation,"observation requires native choice");std::cin>>outcome;require(outcome>=0&&outcome<4,"observed next state");
    std::array<uint8_t,4>before{},value{};dr->read(carry+24,before.data(),4);require(std::all_of(before.begin(),before.end(),[](uint8_t b){return b==0;}),"route output has empty observation field");value[outcome]=1;
    dr->mark("external_observation_"+std::to_string(id));dr->write(carry+24,value.data(),4);
    std::cout<<"{\"kind\":\"injection\",\"id\":"<<id<<",\"address\":"<<carry+24<<",\"observed_state\":"<<outcome<<",\"before\":";arrayjson(before);std::cout<<",\"bytes\":";arrayjson(value);std::cout<<"}\n"<<std::flush;
   }
   dr->mark("chain_setup_"+std::to_string(id));
   int first=carry==slabs[0]+32?1:0;std::array<std::array<uint8_t,128>,2>before{},after{};std::array<uint32_t,2>used{},original{};
   for(int j=0;j<count;j++){used[j]=slabs[(first+j)%2];dr->read(used[j],before[j].data(),128);}
   auto input=carry;dr->begin();
   for(int j=0;j<count;j++){
    int32_t label=cmd=="Q"?1:0;require(dev->enqueue(input,&label),"enqueue");require(dr->descriptors.size()==size_t(j+1),"one descriptor");auto d=dr->descriptors.back();require(d[1]==input,"native feedback pointer");original[j]=d[7];d[7]=used[j];
    dr->capture=false;dr->write(dr->descriptor_addresses.back(),d.data(),64);dr->capture=true;dr->descriptors.back()=d;input=used[j]+32;
   }
   dr->release();dr->mark("NO_HOST_IO_BEGIN_"+std::to_string(id));auto accesses=dr->accesses;const char* wait_env=std::getenv("AKD_IMAGE_WAIT_US");int wait_us=wait_env?std::stoi(wait_env):50;require(wait_us==50||wait_us==20000,"admitted completion wait");auto start=us();auto cstart=thread_ns();std::this_thread::sleep_for(std::chrono::microseconds(wait_us));require(accesses==dr->accesses,"host I/O during protected execution interval");auto elapsed=us()-start;
   dr->mark("NO_HOST_IO_END_"+std::to_string(id));dr->mark("first_completion_poll_"+std::to_string(id));
   uint32_t completion=0;int completion_polls=0;do{completion=dr->read32(Driver::QREG-4+0x10)>>16;completion_polls++;if(completion==dr->descriptors.back()[0]>>16)break;require(us()-start<2000000,"bounded inference completion");std::this_thread::sleep_for(std::chrono::microseconds(50));}while(true);auto wait_total_us=us()-start;auto wait_cpu_ns=thread_ns()-cstart;
   for(int j=0;j<count;j++){dr->read(used[j],after[j].data(),128);require(std::all_of(after[j].begin()+64,after[j].end(),[](uint8_t b){return b==165;}),"guard corrupted");}
   auto memory=impl->mem()->report();require(memory==allocation,"SRAM allocation grew");
   std::cout<<"{\"kind\":\"chain\",\"id\":"<<id<<",\"phase\":\""<<cmd<<"\",\"n\":"<<count<<",\"initial_address\":"<<carry<<",\"observed_state\":"<<outcome<<",\"transition_learning\":"<<transition_learning<<",\"history_learning\":"<<history_learning<<",\"silent_us\":"<<elapsed<<",\"wait_us\":"<<wait_us<<",\"completion_polls\":"<<completion_polls<<",\"wait_total_us\":"<<wait_total_us<<",\"wait_thread_cpu_ns\":"<<wait_cpu_ns<<",\"first_completion\":"<<completion<<",\"allocated_bytes\":"<<memory.first<<",\"peak_bytes\":"<<memory.second<<",\"descriptors\":[";
   for(int j=0;j<count;j++){if(j)std::cout<<',';std::cout<<"{\"address\":"<<dr->descriptor_addresses[j]<<",\"original_output\":"<<original[j]<<",\"words\":";arrayjson(dr->descriptors[j]);std::cout<<'}';}
   std::cout<<"],\"slots\":[";for(int j=0;j<count;j++){if(j)std::cout<<',';std::cout<<"{\"address\":"<<used[j]<<",\"before\":";arrayjson(before[j]);std::cout<<",\"after\":";arrayjson(after[j]);std::cout<<'}';}std::cout<<"]}\n"<<std::flush;
   dr->mark("POST_COMPLETE_SDK_DRAIN_"+std::to_string(id));for(int j=0;j<count;j++)dr->write(original[j],after[j].data(),64);for(int j=0;j<count;j++){auto y=fetch(*dev);require(y->size()==32&&y->type()==akida::TensorType::uint8,"drain format");}
   carry=used[count-1]+32;pending_observation=cmd=="Q";
  }
  snap(-2);dev->scratch_free(initial);for(auto a:slabs)dev->scratch_free(a);dev->unprogram();dr->mark("UNPROGRAMMED");std::cout<<"{\"kind\":\"final\",\"status\":\"PASS\",\"cleanup\":\"unprogrammed\"}\n"<<std::flush;return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';if(dr){dr->hold=false;dr->capture=false;dr->mark("FAILURE");}if(dev){try{dev->unprogram();std::cerr<<"cleanup: unprogrammed\n";}catch(...){std::cerr<<"cleanup failed\n";}}return 1;}
}
