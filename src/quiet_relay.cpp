// Qualified six-FNP graph, with queued recognition repair and a blocked host thread.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
#include <sys/resource.h>
struct RelayBank {
 int col,row,id; CPU cpu; std::vector<uint32_t> infer,learn;
 RelayBank(int c,int r,int i,int n,int u,int classes,const std::string& f)
  :col(c),row(r),id(i),cpu(n,u,32,classes,load<uint32_t>(f+".ram")),
   infer(load<uint32_t>(f+".infer")),learn(load<uint32_t>(f+".learn")){}
};

int main(int argc,char**argv){
 std::vector<uint8_t> program;std::unique_ptr<LabDriver> driver;akida::HardwareDevicePtr dev;bool programmed=false;
 try{
  need(argc==6,"fixture stream output chunk sleep-ms");std::string f=argv[1],s=argv[2],out=argv[3];
  int chunk=std::stoi(argv[4]),sleep_ms=std::stoi(argv[5]);
  need((chunk==8&&(sleep_ms==0||sleep_ms==20))||(chunk==32&&sleep_ms==100)||(chunk==128&&sleep_ms==250),"frozen quiet treatments");
  need(chunk<=32,"128-event learning stage remains unqualified; refuse before hardware access");
  int n,u,classes,nb,nf,threshold;std::ifstream layout(f+".layout");layout>>n>>u>>classes>>nb>>nf>>threshold;
  need(bool(layout)&&n==512&&u==512&&classes==4&&nb==3&&nf==3&&threshold==27,"qualified quiet graph");
  std::vector<RelayBank> banks;
  for(int k=0;k<3;k++){int c,r,id,words;layout>>c>>r>>id>>words;need(bool(layout)&&words==10242,"quiet bank layout");banks.emplace_back(c,r,id,n,u,4,f+".bank"+std::to_string(k));}
  std::vector<std::vector<uint32_t>> fixed;
  for(int k=0;k<3;k++){int c,r,id,words;layout>>c>>r>>id>>words;auto a=load<uint32_t>(f+".fixed"+std::to_string(k)+".ram");need(bool(layout)&&a.size()==size_t(words),"quiet fixed packet");fixed.push_back(a);}
  auto x=load<uint8_t>(s+".inputs");auto labels=load<int32_t>(s+".labels");auto masks=load<uint8_t>(s+".masks");
  need(x.size()==768*512&&labels.size()==768&&masks.size()==768,"quiet full stream");for(int i=512;i<640;i++)need(masks[i]==1,"recognition-only repair");
  driver=std::make_unique<LabDriver>();dev=akida::HardwareDevice::create(driver.get());program=load<uint8_t>(f+".bin");auto info=dev->program(program.data(),program.size());programmed=true;
  need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&!info.activation_enabled(),"quiet resident IO");
  auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());dev->toggle_learn(true);
  auto capture=[&](const std::vector<uint32_t>&a){std::vector<uint32_t>got(a.size());got[0]=a[0];got[1]=a[1];akida::dma::dma_config_read(got.data()+2,uint32_t(got.size()-2),got.data(),impl->dma_config(),impl->mem(),driver.get(),true);return got;};
  for(int k=0;k<3;k++){auto a=capture(fixed[k]);dump(out+".fixed"+std::to_string(k)+".initial",a);need(a==fixed[k],"quiet fixed initial");auto b=capture(banks[k].cpu.ram);dump(out+".bank"+std::to_string(k)+".initial",b);need(b==banks[k].cpu.ram,"quiet bank initial");}
  std::array<std::ofstream,3> snapshots;for(int k=0;k<3;k++)snapshots[k].open(out+".bank"+std::to_string(k)+".snapshots",std::ios::binary);
  std::ofstream scores(out+".scores",std::ios::binary);std::vector<int32_t> boundary;std::vector<int64_t> timings,activity;std::vector<uint32_t> counters;
  auto capture_banks=[&](int i){for(int k=0;k<3;k++){auto a=capture(banks[k].cpu.ram);snapshots[k].write((char*)a.data(),a.size()*4);need(a==banks[k].cpu.ram,"quiet complete native bank RAM");}boundary.push_back(i);};
  auto reference=[&](int i,const std::vector<int32_t>&got){std::vector<uint32_t> bits(16);for(int j=0;j<512;j++)bits[j/32]|=uint32_t(x[i*512+j]!=0)<<(j%32);std::vector<int32_t> support(4);
   for(int k=0;k<3;k++){auto p=banks[k].cpu.forward(bits,labels[i],bool(masks[i]&(1<<k)));support.assign(4,0);for(int j=0;j<512;j++)support[j/128]+=p[j]>27;bits.assign(16,0);for(int c=0;c<4;c++)if(support[c])for(int j=c*128;j<(c+1)*128;j++)bits[j/32]|=uint32_t(1)<<(j%32);}
   need(got==support,"quiet native output exact");
  };
  auto fetch=[&](){akida::TensorUniquePtr y;auto end=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch()))need(Clock::now()<end,"quiet output fetch deadline");need(y->type()==akida::TensorType::int32&&y->size()==4,"quiet output format");auto p=y->data<int32_t>();std::vector<int32_t>got(p,p+4);scores.write((char*)got.data(),16);return got;};
  int mask=-1;auto set_mode=[&](int wanted){if(mask==wanted)return;for(int k=0;k<3;k++){auto&a=(wanted&(1<<k))?banks[k].learn:banks[k].infer;akida::dma::dma_config_write(a.data(),a.size(),impl->dma_config(),impl->external_mem(),driver.get(),true);}mask=wanted;};
  need(dev->set_batch_size(1,false)==1,"quiet serial prefix queue");auto address=dev->scratch_alloc(512);dev->toggle_clock_counter(true);
  auto serial=[&](int i){kick_watchdog();set_mode(masks[i]);driver->write(address,x.data()+i*512,512);need(dev->enqueue(address,&labels[i]),"quiet prefix enqueue");auto got=fetch();reference(i,got);capture_banks(i);};
  for(int i=0;i<512;i++)serial(i);
  set_mode(1);need(dev->set_batch_size(chunk+1,false)==size_t(chunk+1),"quiet requested queue capacity");
  std::vector<uint32_t>addresses;for(int j=0;j<chunk;j++)addresses.push_back(dev->scratch_alloc(512));
  for(int first=512;first<640;first+=chunk){
   kick_watchdog();for(int j=0;j<chunk;j++)driver->write(addresses[j],x.data()+(first+j)*512,512);
   driver->hold_queue();for(int j=0;j<chunk;j++)need(dev->enqueue(addresses[j],&labels[first+j]),"quiet held enqueue");
   uint32_t target=driver->last_descriptor[0]>>16;need(target!=0,"quiet target job ID");
   auto c0=dev->read_clock_counter();auto before_reads=driver->reads,before_writes=driver->writes;struct rusage r0{},r1{};getrusage(RUSAGE_THREAD,&r0);
   auto t0=ns(),p0=cpu_ns();driver->release_queue();uint64_t polls=0;uint32_t observed=0;
   if(sleep_ms)std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
   auto interval_reads=driver->reads-before_reads,interval_writes=driver->writes-before_writes;
   observed=driver->read32(LabDriver::REG+0x28010)>>16;
   if(!sleep_ms){auto limit=Clock::now()+std::chrono::seconds(2);while(observed!=target){polls++;need(Clock::now()<limit,"quiet polled completion deadline");observed=driver->read32(LabDriver::REG+0x28010)>>16;}}
   auto spent=cpu_ns()-p0,elapsed=ns()-t0;getrusage(RUSAGE_THREAD,&r1);auto c1=dev->read_clock_counter();
   timings.insert(timings.end(),{first,chunk,sleep_ms,elapsed,spent});activity.insert(activity.end(),{int64_t(interval_reads),int64_t(interval_writes),r1.ru_nvcsw-r0.ru_nvcsw,r1.ru_nivcsw-r0.ru_nivcsw,int64_t(polls)});counters.insert(counters.end(),{c0,c1,target,observed});
   dump(out+".released_intervals_ns",timings);dump(out+".host_activity",activity);dump(out+".counter_jobs",counters);
   need(observed==target,"quiet completion must precede first wake check");need(interval_writes==1,"quiet release is the only interval host write");if(sleep_ms)need(interval_reads==0&&r1.ru_nvcsw>r0.ru_nvcsw,"quiet interval has no device polling and host thread blocks");
   for(int j=0;j<chunk;j++){auto got=fetch();reference(first+j,got);}capture_banks(first+chunk-1);
  }
  for(auto a:addresses)dev->scratch_free(a);need(dev->set_batch_size(1,false)==1,"quiet serial tail queue");
  for(int i=640;i<768;i++)serial(i);
  for(int k=0;k<3;k++){snapshots[k].close();auto a=capture(banks[k].cpu.ram);dump(out+".bank"+std::to_string(k)+".final",a);need(a==banks[k].cpu.ram,"quiet final bank state");auto b=capture(fixed[k]);dump(out+".fixed"+std::to_string(k)+".final",b);need(b==fixed[k],"quiet final fixed RAM");}
  scores.close();dump(out+".boundary_indices",boundary);dev->scratch_free(address);dev->toggle_clock_counter(false);dev->unprogram();programmed=false;
  std::cout<<"{\"status\":\"QUIET_NATIVE_PASS\",\"forwards\":768,\"chunk\":"<<chunk<<",\"sleep_ms\":"<<sleep_ms<<",\"snapshot_boundaries\":"<<boundary.size()<<",\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';if(driver&&driver->held)try{driver->release_queue();}catch(...){}if(dev&&programmed)try{dev->unprogram();}catch(...){}return 1;}
}
