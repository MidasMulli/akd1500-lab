// Finite hardware-generated replay; captured learned weights are inference-only.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
int main(int argc,char**argv){
 std::unique_ptr<LabDriver>d;akida::HardwareDevicePtr dev;bool programmed=false;std::vector<uint8_t>program;
 try{
  need(argc==4,"fixture horizon output");std::string f=argv[1],out=argv[3];int h=std::stoi(argv[2]);need(h==1||h==2||h==4||h==8,"finite replay horizons");f+="/h"+std::to_string(h);program=load<uint8_t>(f+".bin");auto xs=load<uint8_t>(f+".inputs"),want=load<uint8_t>(f+".expected");need(xs.size()==2048&&want.size()==2048,"eight partial cues");d=std::make_unique<LabDriver>();d->dma_read_min=128;dev=akida::HardwareDevice::create(d.get());auto info=dev->program(program.data(),program.size());programmed=true;
  need(info.number_of_passes()==size_t(h)&&!info.can_learn()&&info.input_is_dense()&&info.output_is_dense()&&info.activation_enabled()&&info.output_bits()==1,"finite dense inference replay");need(dev->set_batch_size(1,false)==1,"single submitted input");static_cast<akida::HardwareDeviceImpl*>(dev.get())->pipeline(false);auto addr=dev->scratch_alloc(256);std::vector<uint8_t>ys;std::vector<uint32_t>descs;std::vector<int64_t>quiet;
  dump(out+".replay_registers",std::vector<uint32_t>{d->read32(LabDriver::REG+0x28070),d->read32(LabDriver::REG+0x28074)});
  for(int s=0;s<8;s++){
   d->write(addr,xs.data()+s*256,256);d->hold_queue();need(dev->enqueue(addr,nullptr),"one seed enqueue");auto desc=d->last_descriptor;descs.insert(descs.end(),desc.begin(),desc.end());auto target=desc[0]>>16;auto r0=d->reads,w0=d->writes;auto t0=ns(),ct0=cpu_ns();d->release_queue();std::this_thread::sleep_for(std::chrono::milliseconds(5));auto t1=ns(),ct1=cpu_ns();auto reads=d->reads-r0,writes=d->writes-w0;auto status=d->read32(LabDriver::REG+0x28010);quiet.insert(quiet.end(),{int64_t(t0),t1,ct1-int64_t(ct0),int64_t(reads),int64_t(writes),int64_t(target),int64_t(status)});dump(out+".quiet",quiet);need(reads==0&&writes==1&&(status>>16)==target,"finite replay complete at first quiet wake");auto y=dev->fetch();need(y&&y->type()==akida::TensorType::uint8&&y->size()==256,"replay terminal output");auto p=y->data<uint8_t>();ys.insert(ys.end(),p,p+256);dump(out+".scores",ys);std::vector<uint32_t>packet(72);d->read(desc[7],packet.data(),288);dump(out+".packet"+std::to_string(s),packet);need(std::equal(p,p+256,want.begin()+s*256),"replay terminal state differs from sealed composition");
  }
  dump(out+".descriptors",descs);dev->scratch_free(addr);dev->unprogram();programmed=false;std::cout<<"{\"status\":\"PASS\",\"passes_per_seed\":"<<h<<",\"seeds\":8,\"host_input_descriptors_per_seed\":1,\"learning_enabled\":false,\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception&e){if(d&&d->held)try{d->release_queue();}catch(...){}if(dev&&programmed)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
