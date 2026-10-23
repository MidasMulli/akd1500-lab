// Compiler-routed multi-bank mechanism instrument. Runtime supplied separately.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
#include "engine/akida_program_info_generated.h"

struct Bank {int col,row,id,words;CPU cpu;std::vector<uint32_t> infer,learn;
 Bank(int c,int r,int i,int w,int n,int u,const std::string& f):col(c),row(r),id(i),words(w),cpu(n,u,32,2,load<uint32_t>(f+".ram")),infer(load<uint32_t>(f+".infer")),learn(load<uint32_t>(f+".learn")){}
};
int main(int argc,char**argv){
 std::vector<uint8_t> program;std::unique_ptr<LabDriver> driver;akida::HardwareDevicePtr dev;bool programmed=false;
 try{
  need(argc==5||argc==6,"fixture-prefix output-prefix samples learning-mask [compiler-inference]");bool compiler_inference=argc==6;need(!compiler_inference||std::string(argv[5])=="compiler-inference","diagnostic mode");std::string f=argv[1],out=argv[2];int count=std::stoi(argv[3]),mask=std::stoi(argv[4]);
  int n,u,classes,nb,nf;std::ifstream layout(f+".layout");layout>>n>>u>>classes>>nb>>nf;need(bool(layout)&&n==512&&u==64&&classes==2&&nb>=2&&nb<=7&&count>0&&count<=256&&mask>=0&&mask<(1<<nb),"frozen network dimensions");
  std::vector<Bank> banks;for(int k=0;k<nb;k++){int c,r,i,w;layout>>c>>r>>i>>w;need(bool(layout)&&w==1282,"bank layout");banks.emplace_back(c,r,i,w,n,u,f+".bank"+std::to_string(k));need(banks.back().cpu.ram.size()==size_t(w)&&banks.back().infer.size()==52&&banks.back().learn.size()==52,"bank packet dimensions");}
  std::vector<std::vector<uint32_t>> fixed;
  for(int j=0;j<nf;j++){int c,r,i,w;layout>>c>>r>>i>>w;auto a=load<uint32_t>(f+".fixed"+std::to_string(j)+".ram");need(bool(layout)&&a.size()==size_t(w+2),"fixed packet layout");fixed.push_back(a);}
  if(compiler_inference){need(mask==0,"compiler diagnostic is frozen");for(int k=0;k<nb;k++)banks[k].cpu.ram=load<uint32_t>(f+".compiler-bank"+std::to_string(k)+".ram");}
  int dual_rail=0;layout>>dual_rail;need(dual_rail==0||dual_rail==1,"encoder flag");
  auto inputs=load<uint8_t>(f+".inputs");auto labels=load<int32_t>(f+".labels");need(inputs.size()>=size_t(count*n)&&labels.size()>=size_t(count),"network input count");
  std::cerr<<"stage=open\n";driver=std::make_unique<LabDriver>();dev=akida::HardwareDevice::create(driver.get());program=load<uint8_t>(f+(compiler_inference?".compiler.bin":".bin"));std::cerr<<"stage=program\n";auto info=dev->program(program.data(),program.size());programmed=true;std::cerr<<"IO passes="<<info.number_of_passes()<<" learn="<<info.can_learn()<<" input_dense="<<info.input_is_dense()<<" output_dense="<<info.output_is_dense()<<" activated="<<info.activation_enabled()<<"\n";need(info.number_of_passes()==1&&info.can_learn()==!compiler_inference&&info.input_is_dense()&&(info.outputs_type()==akida::fb::IoType_cnp_sparse||info.output_is_dense())&&!info.activation_enabled(),"single-pass network IO");
  std::cerr<<"stage=configure_modes\n";auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());if(!compiler_inference)dev->toggle_learn(true);
  for(int k=0;k<nb&&!compiler_inference;k++){auto& packet=(mask&(1<<k))?banks[k].learn:banks[k].infer;akida::dma::dma_config_write(packet.data(),packet.size(),impl->dma_config(),impl->external_mem(),driver.get(),true);}
  auto capture=[&](const std::vector<uint32_t>& expected){std::vector<uint32_t> got(expected.size());got[0]=expected[0];got[1]=expected[1];akida::dma::dma_config_read(got.data()+2,uint32_t(got.size()-2),got.data(),impl->dma_config(),impl->mem(),driver.get(),true);return got;};
  std::cerr<<"stage=fixed_readback\n";std::vector<std::vector<uint32_t>> fixed_initial;
  for(int j=0;j<nf;j++){std::cerr<<"fixed="<<j<<" words="<<fixed[j].size()<<"\n";auto got=capture(fixed[j]);dump(out+".fixed"+std::to_string(j)+".initial.ram",got);need(got==fixed[j],"fixed initial compiler/readback mismatch");fixed_initial.push_back(got);}
  std::cerr<<"stage=bank_readback\n";for(int k=0;k<nb;k++){auto got=capture(banks[k].cpu.ram);dump(out+".bank"+std::to_string(k)+".initial.ram",got);need(got==banks[k].cpu.ram,"learning initial readback mismatch");}
  need(dev->set_batch_size(1,false)==1,"network queue");auto addr=dev->scratch_alloc(n);int64_t wall=0,cput=0;uint64_t polls=0;std::vector<int32_t> scores;std::vector<int64_t> latency;
  for(int i=0;i<count;i++){
   kick_watchdog();auto start=ns(),cs=cpu_ns();driver->write(addr,inputs.data()+i*n,n);int32_t label=labels[i];need(dev->enqueue(addr,compiler_inference?nullptr:&label),"network enqueue");auto deadline=Clock::now()+std::chrono::seconds(2);akida::TensorUniquePtr y;
   while(!(y=dev->fetch())){polls++;need(Clock::now()<deadline,"network completion deadline");std::this_thread::sleep_for(std::chrono::microseconds(50));}
   auto dt=ns()-start;wall+=dt;cput+=cpu_ns()-cs;latency.push_back(dt);need(y->type()==akida::TensorType::int32&&y->size()==2,"network output dimensions");auto yp=y->data<int32_t>();std::vector<int32_t> actual(yp,yp+2);scores.insert(scores.end(),actual.begin(),actual.end());dump(out+".scores",scores);dump(out+".latency_ns",latency);
   std::vector<uint32_t> x(n/32);for(int k=0;k<n;k++)x[k/32]|=uint32_t(inputs[i*n+k]!=0)<<(k%32);std::vector<int32_t> responses;
   for(int k=0;k<nb;k++){
    auto p=banks[k].cpu.forward(x,label,bool(mask&(1<<k)));responses.clear();for(auto v:p)responses.push_back(v>15);
    auto got=capture(banks[k].cpu.ram);dump(out+".snapshot_"+std::to_string(i)+"_bank"+std::to_string(k)+".ram",got);
    if(got!=banks[k].cpu.ram){for(size_t w=2;w<got.size();w++)if(got[w]!=banks[k].cpu.ram[w]){std::cerr<<"sample="<<i<<" bank="<<k<<" word="<<w<<" actual="<<got[w]<<" expected="<<banks[k].cpu.ram[w]<<"\n";break;}throw std::runtime_error("network per-bank learning RAM mismatch");}
    if(k+1<nb){x.assign(n/32,0);for(int j=0;j<u;j++)for(int b=0;b<8;b++){int z=j*8+b;bool v=dual_rail?(b<4?responses[j]:!responses[j]):responses[j];x[z/32]|=uint32_t(v)<<(z%32);}}
   }
   std::vector<int32_t> expected(2);for(int j=0;j<u;j++)expected[j/32]+=responses[j];need(actual==expected,"network final output mismatch");
  }
  for(int k=0;k<nb;k++)dump(out+".bank"+std::to_string(k)+".ram",capture(banks[k].cpu.ram));
  for(int j=0;j<nf;j++){std::cerr<<"fixed="<<j<<" words="<<fixed[j].size()<<"\n";auto got=capture(fixed[j]);dump(out+".fixed"+std::to_string(j)+".final.ram",got);need(got==fixed_initial[j],"fixed encoder/readout RAM changed");}
  dev->scratch_free(addr);dev->unprogram();std::cout<<"{\"status\":\"PASS\",\"samples\":"<<count<<",\"banks\":"<<nb<<",\"learning_mask\":"<<mask<<",\"configured_learning_neurons\":"<<nb*u<<",\"wall_ns\":"<<wall<<",\"thread_cpu_ns\":"<<cput<<",\"polls\":"<<polls<<",\"all_step_RAM_and_final_outputs_exact\":true,\"fixed_RAM_unchanged\":true,\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';if(dev&&programmed)try{dev->unprogram();}catch(...){}return 1;}
}
