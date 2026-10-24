// Equal-rate co-running actor: 3072 events /5.12s, persistent state, no observers.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
int main(int argc,char**argv){
 std::vector<uint8_t>program;std::unique_ptr<LabDriver>driver;akida::HardwareDevicePtr dev;bool programmed=false;
 try{
  need(argc==4,"native/cpu/task fixture output");std::string mode=argv[1],f=argv[2],out=argv[3];need(mode=="native"||mode=="cpu"||mode=="task","bounded mode");bool native=mode=="native";
  constexpr int total=3072,batch=32,rate=600;auto x=load<uint8_t>(f+"/native.inputs");auto initial=load<uint32_t>(f+"/reduced.ram");need(x.size()==768*512&&initial.size()==322,"sealed fixture");
  CPU cpu(512,16,32,1,initial);std::vector<int>table(16,-1);std::vector<uint32_t>addresses;
  if(native){
   driver=std::make_unique<LabDriver>();driver->dma_read_min=128;dev=akida::HardwareDevice::create(driver.get());program=load<uint8_t>(f+"/reduced.bin");auto info=dev->program(program.data(),program.size());programmed=true;need(info.number_of_passes()==1&&info.can_learn()&&!info.activation_enabled(),"native contract");dev->toggle_learn(true);
   std::vector<uint32_t>got(dev->learn_mem_size());dev->learn_mem(got.data());need(got==initial,"initial complete SRAM");need(dev->set_batch_size(33,false)==33,"bounded spare-slot queue");for(int j=0;j<32;j++)addresses.push_back(dev->scratch_alloc(512));
  }
  std::cout<<"{\"kind\":\"ready\",\"mode\":\""<<mode<<"\",\"events\":3072,\"rate\":600}"<<std::endl;std::string go;std::getline(std::cin,go);need(go=="GO","start handshake");
  std::vector<int32_t>scores;std::vector<int64_t>timings;int late=0;int64_t start=ns(),cp=cpu_ns();
  std::cout<<"{\"kind\":\"started\",\"start_ns\":"<<start<<"}"<<std::endl;
  for(int first=0;first<total;first+=batch){
   kick_watchdog();auto begin=ns();int64_t due=start+int64_t(first+batch)*1000000000/rate;
   uint32_t target=0;int32_t zero=0;
   if(native){
    for(int j=0;j<batch;j++)driver->write(addresses[j],x.data()+((first+j)%768)*512,512);
    driver->hold_queue();for(int j=0;j<batch;j++)need(dev->enqueue(addresses[j],&zero),"constant teacher queue");target=driver->last_descriptor[0]>>16;driver->release_queue();
   }else for(int j=0;j<batch;j++){
    int i=(first+j)%768;
    if(mode=="task"){
     int context=-1;for(int k=0;k<16;k++)if(x[i*512+k*10]){context=k;break;}need(context>=0,"context decoder");int candidate=x[i*512+176]?0:1;int y=table[context]==candidate;scores.push_back(y);if(x[i*512+160])table[context]=candidate;
    }else{
     std::vector<uint32_t>bits(16);for(int k=0;k<512;k++)bits[k/32]|=uint32_t(x[i*512+k])<<(k%32);auto y=cpu.forward(bits,0,true);int support=0;for(auto a:y)support+=a>13;scores.push_back(support);
    }
   }
   if(ns()<due)std::this_thread::sleep_until(Clock::time_point(std::chrono::nanoseconds(due)));
   if(native){
    need((driver->read32(LabDriver::REG+0x28010)>>16)==target,"all work completed by scheduled wake");
    for(int j=0;j<batch;j++){auto y=dev->fetch();need(bool(y)&&y->type()==akida::TensorType::int32&&y->size()==1,"ready native prediction support");scores.push_back(y->data<int32_t>()[0]);}
   }
   auto end=ns();if(end>due+2000000)late++;timings.insert(timings.end(),{first,begin,end,due});
  }
  auto cput=cpu_ns()-cp,wall=ns()-start;std::vector<uint32_t>final;
  if(native){final.resize(dev->learn_mem_size());dev->learn_mem(final.data());for(auto a:addresses)dev->scratch_free(a);dev->unprogram();programmed=false;
   // Full independent replay is after the service measurement, never co-run.
   for(int i=0;i<total;i++){int k=i%768;std::vector<uint32_t>bits(16);for(int j=0;j<512;j++)bits[j/32]|=uint32_t(x[k*512+j])<<(j%32);auto y=cpu.forward(bits,0,true);int support=0;for(auto a:y)support+=a>13;need(scores[i]==support,"continuous native output replay");}need(final==cpu.ram,"continuous native final SRAM");
  }else final=cpu.ram;
  dump(out+".scores",scores);dump(out+".ram",final);dump(out+".batch_ns",timings);
  std::cout<<"{\"kind\":\"complete\",\"status\":\"PASS\",\"mode\":\""<<mode<<"\",\"events\":3072,\"rate\":600,\"wall_ns\":"<<wall<<",\"thread_cpu_ns\":"<<cput<<",\"late_batches_over_2ms\":"<<late<<",\"native_mode_changes\":0}"<<std::endl;
 }catch(const std::exception&e){if(driver&&driver->held)try{driver->release_queue();}catch(...){}if(dev&&programmed)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
