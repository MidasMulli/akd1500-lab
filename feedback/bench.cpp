// Native input-gated plasticity: constant class0, no per-event mode/weight writes.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"

int main(int argc,char**argv){
 std::vector<uint8_t>program;std::unique_ptr<LabDriver>driver;akida::HardwareDevicePtr dev;bool programmed=false;
 try{
  need(argc==10,"mode fixture arm output count raw/reduced batch sleep-ms repeats");std::string mode=argv[1],f=argv[2],arm=argv[3],out=argv[4],variant=argv[6];int count=std::stoi(argv[5]),batch=std::stoi(argv[7]),sleep=std::stoi(argv[8]),repeats=std::stoi(argv[9]);
  bool native=mode=="native",raw=variant=="raw";need((native||mode=="cpu"||mode=="task")&&(raw||variant=="reduced")&&(arm=="native"||arm=="frozen"||arm=="disconnected")&&(count==3||count==128||count==768)&&(batch==1||batch==32)&&(sleep==0||sleep==5||sleep==20)&&repeats>=1&&repeats<=100,"frozen arguments");
  need(batch==1||count==768,"burst full stream only");auto x=load<uint8_t>(f+"/"+arm+".inputs");auto initial=load<uint32_t>(f+"/"+variant+".ram");need(x.size()==768*512&&initial.size()==322,"fixed task shape");
  std::vector<int32_t>scores,bounds;std::vector<int64_t>times,callers,intervals;std::vector<uint32_t>final;std::ofstream snaps(out+".snapshots",std::ios::binary);
  if(!native){
   for(int r=0;r<repeats;r++){
    CPU cpu(512,16,32,1,initial);std::vector<int>table(16,-1);
    for(int i=0;i<count;i++){
     kick_watchdog();auto start=ns(),cp=cpu_ns();bool learning=!(arm=="frozen"&&i>=128);std::vector<int32_t>y;
     if(mode=="task"){
      int ctx=-1;for(int j=0;j<16;j++)if(x[i*512+j*10]){ctx=j;break;}need(ctx>=0,"context encoder");int candidate=x[i*512+176]?0:1;bool gate=x[i*512+160];y={int(table[ctx]==candidate)};if(learning&&gate)table[ctx]=candidate;
     }else{
      std::vector<uint32_t>bits(16);for(int k=0;k<512;k++)bits[k/32]|=uint32_t(x[i*512+k])<<(k%32);y=cpu.forward(bits,0,learning);if(!raw){int support=0;for(auto a:y)support+=a>13;y={support};}
     }
     callers.push_back(cpu_ns()-cp);times.push_back(ns()-start);
     if(!r){scores.insert(scores.end(),y.begin(),y.end());if(mode!="task"){snaps.write((char*)cpu.ram.data(),cpu.ram.size()*4);bounds.push_back(i);}}
    }final=cpu.ram;
   }
  }else{
   need(repeats==1,"one native trajectory per program");driver=std::make_unique<LabDriver>();driver->dma_read_min=128;dev=akida::HardwareDevice::create(driver.get());program=load<uint8_t>(f+"/"+variant+".bin");auto info=dev->program(program.data(),program.size());programmed=true;
   need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&!info.activation_enabled(),"single resident local-learning path");dev->toggle_learn(true);
   auto capture=[&](){std::vector<uint32_t>a(dev->learn_mem_size());dev->learn_mem(a.data());return a;};need(capture()==initial,"initial local RAM");dump(out+".initial.ram",initial);
   auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());auto fixed=[&](const char*tag){
    int number;std::ifstream l(f+"/"+variant+".layout");l>>number;need(bool(l)&&number>=0&&number<=4,"fixed packet count");
    for(int j=0;j<number;j++){auto src=load<uint32_t>(f+"/"+variant+".packet"+std::to_string(j)+".ram"),got=src;akida::dma::dma_config_read(got.data()+2,uint32_t(got.size()-2),got.data(),impl->dma_config(),impl->mem(),driver.get(),true);dump(out+".packet"+std::to_string(j)+"."+tag,got);need(got==src,"fixed readout SRAM");}
   };fixed("initial");
   need(dev->set_batch_size(batch==1?1:batch+1,false)==size_t(batch==1?1:batch+1),"queue spare slot contract");std::vector<uint32_t>addresses;for(int j=0;j<batch;j++)addresses.push_back(dev->scratch_alloc(512));
   CPU cpu(512,16,32,1,initial);bool learning=true;int mode_changes=0;int32_t zero=0;
   for(int first=0;first<count;first+=batch){
    kick_watchdog();bool learn=!(arm=="frozen"&&first>=128);if(learning!=learn){dev->toggle_learn(learn);learning=learn;mode_changes++;}
    auto start=ns(),cp=cpu_ns();
    for(int j=0;j<batch;j++)driver->write(addresses[j],x.data()+(first+j)*512,512);
    if(batch>1)driver->hold_queue();for(int j=0;j<batch;j++)need(dev->enqueue(addresses[j],learning?&zero:nullptr),"constant-class enqueue");
    uint64_t rr=driver->reads,ww=driver->writes;uint32_t target=batch>1?driver->last_descriptor[0]>>16:0;
    if(batch>1)driver->release_queue();if(sleep)std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
    auto rd=driver->reads-rr,wr=driver->writes-ww;
    if(sleep){need(rd==0&&wr==1,"no driver polling/writes during blocked interval except release");need((driver->read32(LabDriver::REG+0x28010)>>16)==target,"all queued work done by first wake");}
    std::vector<int32_t>got;auto deadline=Clock::now()+std::chrono::seconds(2);
    for(int j=0;j<batch;j++){
     akida::TensorUniquePtr y;while(!(y=dev->fetch()))need(Clock::now()<deadline,"bounded completion");int outputs=raw?16:1;need(y->type()==akida::TensorType::int32&&y->size()==size_t(outputs),"native result size");auto p=y->data<int32_t>();got.insert(got.end(),p,p+outputs);
    }
    callers.push_back(cpu_ns()-cp);times.push_back(ns()-start);intervals.insert(intervals.end(),{first,batch,sleep,int64_t(rd),int64_t(wr)});scores.insert(scores.end(),got.begin(),got.end());dump(out+".scores",scores);
    for(int j=0;j<batch;j++){
     int i=first+j;std::vector<uint32_t>bits(16);for(int k=0;k<512;k++)bits[k/32]|=uint32_t(x[i*512+k])<<(k%32);auto y=cpu.forward(bits,0,learn);
     if(!raw){int support=0;for(auto a:y)support+=a>13;y={support};}need(std::equal(y.begin(),y.end(),got.begin()+j*(raw?16:1)),"native prediction / CPU exact");
    }
    if(!learning)dev->toggle_learn(true);auto state=capture();if(!learning)dev->toggle_learn(false);snaps.write((char*)state.data(),state.size()*4);snaps.flush();dump(out+".ram",state);bounds.push_back(first+batch-1);need(state==cpu.ram,"native full SRAM / CPU exact");
   }
   if(!learning)dev->toggle_learn(true);final=capture();fixed("final");for(auto a:addresses)dev->scratch_free(a);dev->unprogram();programmed=false;need(arm!="native"||mode_changes==0,"native arm changed modes");
  }
  snaps.close();dump(out+".scores",scores);dump(out+".ram",final);dump(out+".boundaries",bounds);dump(out+".latency_ns",times);dump(out+".caller_ns",callers);dump(out+".blocked_intervals",intervals);
  std::cout<<"{\"status\":\"PASS\",\"mode\":\""<<mode<<"\",\"count\":"<<count<<",\"batch\":"<<batch<<",\"sleep_ms\":"<<sleep<<",\"constant_teacher_class\":0,\"native_mode_changes\":0,\"cleanup\":\"complete\"}\n";
 }catch(const std::exception&e){if(driver&&driver->held)try{driver->release_queue();}catch(...){}if(dev&&programmed)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
