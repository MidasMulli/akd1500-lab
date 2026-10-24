// Native-counter characterization. Reuse the independently verified CPU rule.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"

int main(int argc, char** argv) {
 std::vector<uint8_t> program;
 std::unique_ptr<LabDriver> driver;
 akida::HardwareDevicePtr dev;
 bool programmed = false;
 try {
  need(argc == 6, "fixture stream output samples kind(learner|fixed)");
  std::string f=argv[1], stream=argv[2], out=argv[3], kind=argv[5];
  bool learner=kind=="learner"; need(learner||kind=="fixed", "known kind");
  int count=std::stoi(argv[4]), n,u,nw,classes,outputs;
  std::ifstream meta(f+".meta"); meta>>n>>u>>nw>>classes;
  need(bool(meta)&&n==512&&count>0&&count<=1024, "bounded metal fixture");
  outputs=learner?u:2;
  auto inputs=load<uint8_t>(stream+".inputs");
  auto labels=load<int32_t>(stream+".labels");
  auto modes=load<uint8_t>(stream+".modes");
  auto expected_fixed=learner?std::vector<int32_t>{}:load<int32_t>(stream+".expected");
  need(inputs.size()==size_t(count*n)&&labels.size()==size_t(count)&&modes.size()==size_t(count), "stream dimensions");
  if(!learner) need(expected_fixed.size()==size_t(count*outputs), "fixed reference dimensions");
  std::unique_ptr<CPU> reference;
  std::vector<uint32_t> initial;
  if(learner) { initial=load<uint32_t>(f+".ram"); reference=std::make_unique<CPU>(n,u,nw,classes,initial); }
  driver=std::make_unique<LabDriver>(); dev=akida::HardwareDevice::create(driver.get());
  program=load<uint8_t>(f+".bin"); auto info=dev->program(program.data(),program.size()); programmed=true;
  need(info.number_of_passes()==1&&info.can_learn()==learner&&info.input_is_dense()&&!info.activation_enabled(), "native metal IO");
  auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());
  auto capture=[&]() {
   std::vector<uint32_t> got(initial.size()); got[0]=initial[0]; got[1]=initial[1];
   akida::dma::dma_config_read(got.data()+2,uint32_t(got.size()-2),got.data(),impl->dma_config(),impl->mem(),driver.get(),true);
   return got;
  };
  if(learner) { auto got=capture(); dump(out+".initial.ram",got); need(got==initial,"initial full RAM"); }
  dev->toggle_clock_counter(true);
  std::vector<uint32_t> idle; std::vector<int64_t> idle_times;
  for(int delay_us : {0,0,5000,50000}) {
   if(delay_us) std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
   idle_times.push_back(ns()); idle.push_back(dev->read_clock_counter());
  }
  dump(out+".idle_counter",idle); dump(out+".idle_ns",idle_times);
  auto address=dev->scratch_alloc(n);
  std::vector<int32_t> scores;
  std::vector<uint32_t> bounds, cycles, config_bounds;
  std::vector<int64_t> times, caller;
  int mode=-1; size_t effective_infer=0,effective_learn=0;
  std::ofstream snapshots(out+".snapshots",std::ios::binary);
  auto values=[&](akida::TensorUniquePtr y) {
   need(y&&y->type()==akida::TensorType::int32&&y->size()==size_t(outputs),"native output shape");
   auto v=y->data<int32_t>(); return std::vector<int32_t>(v,v+outputs);
  };
  auto wait_output=[&]() {
   akida::TensorUniquePtr y; auto limit=Clock::now()+std::chrono::seconds(2);
   while(!(y=dev->fetch())) need(Clock::now()<limit,"owned completion deadline");
   return values(std::move(y));
  };
  for(int i=0;i<count;i++) {
   kick_watchdog(); need(modes[i]<=1&&(!modes[i]||learner),"mode contract");
   if(mode!=modes[i]) {
    if(learner) dev->toggle_learn(modes[i]);
    auto effective=dev->set_batch_size(9,false);
    if(modes[i]) effective_learn=effective; else effective_infer=effective;
    need(dev->set_batch_size(1,false)==1,"serial queue"); mode=modes[i];
   }
   uint32_t q0=dev->read_config_clock_counter(), c0=dev->read_clock_counter();
   auto t0=ns(), p0=cpu_ns();
   driver->write(address,inputs.data()+i*n,n);
   need(dev->enqueue(address,modes[i]?&labels[i]:nullptr),"serial enqueue");
   auto got=wait_output(); auto elapsed=ns()-t0, spent=cpu_ns()-p0;
   uint32_t c1=dev->read_clock_counter(),q1=dev->read_config_clock_counter();
   bounds.push_back(c0); bounds.push_back(c1); cycles.push_back(c1-c0);
   config_bounds.push_back(q0); config_bounds.push_back(q1);
   times.push_back(elapsed); caller.push_back(spent);
   scores.insert(scores.end(),got.begin(),got.end());
   std::vector<int32_t> want;
   if(learner) {
    std::vector<uint32_t> bits(n/32);
    for(int k=0;k<n;k++) bits[k/32]|=uint32_t(inputs[i*n+k]!=0)<<(k%32);
    want=reference->forward(bits,labels[i],modes[i]);
    auto ram=capture(); snapshots.write((char*)ram.data(),ram.size()*sizeof(uint32_t)); snapshots.flush();
    need(ram==reference->ram,"every-step complete native learning RAM");
   } else want=std::vector<int32_t>(expected_fixed.begin()+i*outputs,expected_fixed.begin()+(i+1)*outputs);
   if(got!=want) {
    std::cerr<<"output mismatch at event "<<i<<"\n";
    dump(out+".scores",scores); dump(out+".counter_bounds",bounds);
    throw std::runtime_error("exact native output mismatch");
   }
  }
  snapshots.close();
  dump(out+".scores",scores); dump(out+".counter_bounds",bounds); dump(out+".cycles",cycles);
  dump(out+".config_bounds",config_bounds); dump(out+".latency_ns",times); dump(out+".caller_ns",caller);
  // Queue experiment uses the same frozen eight-input suffix, no teaching.
  if(learner) dev->toggle_learn(false);
  need(count>=8&&dev->set_batch_size(9,false)==9,"frozen resident queue capacity");
  std::vector<uint32_t> addresses;
  for(int j=0;j<8;j++) { addresses.push_back(dev->scratch_alloc(n)); driver->write(addresses[j],inputs.data()+(count-8+j)*n,n); }
  std::vector<int32_t> serial,burst; std::vector<uint32_t> burst_cycles;
  std::vector<int64_t> burst_times;
  for(int repeat=0;repeat<3;repeat++) {
   uint32_t start=dev->read_clock_counter(); auto t=ns();
   for(int j=0;j<8;j++) { need(dev->enqueue(addresses[j],nullptr),"frozen serial suffix enqueue"); auto got=wait_output(); serial.insert(serial.end(),got.begin(),got.end()); }
   burst_cycles.push_back(dev->read_clock_counter()-start); burst_times.push_back(ns()-t);
   driver->hold_queue();
   for(int j=0;j<8;j++) need(dev->enqueue(addresses[j],nullptr),"held burst enqueue");
   uint32_t target=driver->last_descriptor[0]>>16; need(target!=0,"captured burst job ID");
   start=dev->read_clock_counter(); t=ns(); auto writes=driver->writes;
   driver->release_queue(); auto deadline=Clock::now()+std::chrono::seconds(2);
   while((driver->read32(LabDriver::REG+0x28010)>>16)!=target) need(Clock::now()<deadline,"resident burst completion deadline");
   burst_cycles.push_back(dev->read_clock_counter()-start); burst_times.push_back(ns()-t);
   need(driver->writes==writes+1,"no host writes during released burst");
   for(int j=0;j<8;j++) { auto got=wait_output(); burst.insert(burst.end(),got.begin(),got.end()); }
  }
  need(serial==burst,"serialized and burst outputs identical");
  // Recheck suffix against already-verified serial probe outputs.
  for(int repeat=0;repeat<3;repeat++) for(int j=0;j<8;j++) for(int k=0;k<outputs;k++)
   need(serial[(repeat*8+j)*outputs+k]==scores[(count-8+j)*outputs+k],"burst reference suffix");
  dump(out+".serial_suffix",serial); dump(out+".burst_suffix",burst);
  dump(out+".burst_cycles",burst_cycles); dump(out+".burst_latency_ns",burst_times);
  idle.push_back(dev->read_clock_counter()); std::this_thread::sleep_for(std::chrono::milliseconds(50)); idle.push_back(dev->read_clock_counter());
  dump(out+".idle_counter",idle);
  if(learner) { auto final=capture(); dump(out+".final.ram",final); need(final==reference->ram,"frozen suffix changes no native memory"); }
  for(auto a:addresses) dev->scratch_free(a);
  dev->scratch_free(address); dev->toggle_clock_counter(false); dev->unprogram(); programmed=false;
  std::cout<<"{\"status\":\"PASS\",\"kind\":\""<<kind<<"\",\"events\":"<<count<<",\"extra_frozen_forwards\":48,\"outputs\":"<<outputs
   <<",\"effective_inference_queue_request9\":"<<effective_infer<<",\"effective_learning_queue_request9\":"<<effective_learn
   <<",\"initial_idle_delta\":"<<uint32_t(idle[3]-idle[0])<<",\"final_idle_delta\":"<<uint32_t(idle[5]-idle[4])
   <<",\"every_output_exact\":true,\"every_learning_RAM_exact\":"<<(learner?"true":"null")<<",\"cleanup\":\"unprogrammed\"}\n";
 } catch(const std::exception& e) {
  if(driver&&driver->held) try { driver->release_queue(); } catch(...) {}
  if(dev&&programmed) try { dev->toggle_clock_counter(false); dev->unprogram(); } catch(...) {}
  std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1;
 }
}
