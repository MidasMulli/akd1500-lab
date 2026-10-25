// Timed inherited consensus circuit. No event-loop disk writes or CPU teacher decode.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
#include <numeric>

struct TimedBank {
 int u; CPU cpu; std::vector<uint32_t> initial,infer,learn; std::vector<int32_t> threshold;
 TimedBank(int units,int classes,const std::string&p):u(units),cpu(512,units,32,classes,load<uint32_t>(p+".ram")),initial(load<uint32_t>(p+".ram")),infer(load<uint32_t>(p+".infer")),learn(load<uint32_t>(p+".learn")),threshold(load<int32_t>(p+".threshold")){}
 std::vector<uint8_t> step(const std::vector<uint8_t>&x,int label){
  std::vector<uint32_t>b(16);for(int j=0;j<512;j++)b[j/32]|=uint32_t(x[j])<<(j%32);
  auto p=cpu.forward(b,label<0?0:label,label>=0);std::vector<uint8_t>y(u);
  for(int j=0;j<u;j++)y[j]=p[j]>threshold[j];return y;
 }
};
struct SparseFixed {
 std::vector<std::vector<std::pair<int,int>>> columns;std::vector<int32_t> thresholds;
 SparseFixed(const std::vector<int8_t>&w,const std::vector<int32_t>&t):columns(t.size()),thresholds(t){
  need(w.size()%t.size()==0,"fixed dimensions");for(size_t k=0;k<w.size()/t.size();k++)for(size_t j=0;j<t.size();j++)if(w[k*t.size()+j])columns[j].push_back({int(k),w[k*t.size()+j]});
 }
 std::vector<uint8_t> operator()(const std::vector<uint8_t>&x)const{
  std::vector<uint8_t>y(columns.size());for(size_t j=0;j<columns.size();j++){int sum=0;for(auto term:columns[j])sum+=x[term.first]*term.second;y[j]=sum>thresholds[j];}return y;
 }
};
struct Dictionary {
 std::array<int,12>a,b,m;
 Dictionary(){a.fill(-1);b.fill(-1);m.fill(-1);}
 __attribute__((noinline)) int agree(int key,int first,int second){
  if(first>=0&&first==second)m[key]=first;return m[key];
 }
 __attribute__((noinline)) int step(const int32_t*e){
  int key=e[0],role=e[1];if(role==1){int wire=e[5];(wire/2?b:a)[key]=wire%2;}
  if(role==2&&a[key]>=0&&a[key]==b[key])m[key]=a[key];return m[key];
 }
};
static void observe(std::vector<uint8_t>&x,const int32_t*e){
 if(e[1]!=2){std::fill(x.begin()+32,x.begin()+64,0);for(int b=0;b<4;b++)for(int j=0;j<4;j++)x[32+b*8+((e[0]>>b)&1)*4+j]=1;}
 std::fill(x.begin()+64,x.begin()+72,0);x[(e[1]==1||e[1]==4)?65:e[1]==2?69:64]=1;
 x[68]=e[1]==4;x[70]=e[1]==3;x[71]=e[6];
}
int main(int argc,char**argv){
 std::unique_ptr<LabDriver>d;akida::HardwareDevicePtr dev;
 try{
  need(argc==6,"mode fixture stage verification out");std::string mode=argv[1],f=argv[2],stage=argv[3],verification=argv[4],out=argv[5];
  need(mode=="native"||mode=="replica"||mode=="dictionary","mode");need(verification=="all"||verification=="final","verification");
  auto plan=load<int32_t>(f+"/"+stage+".plan");int count=plan.size()/7;need(count>0&&count<=8192&&plan.size()==size_t(count*7),"sealed finite plan");
  auto expected=load<uint8_t>(f+"/"+stage+".scores");auto expected_inputs=load<uint8_t>(f+"/"+stage+".inputs");auto expected_dict=load<int32_t>(f+"/"+stage+".dict");
  need(expected.size()==size_t(count*128)&&expected_inputs.size()==expected.size()&&expected_dict.size()==size_t(count),"reference sizes");
  std::vector<TimedBank>banks;banks.emplace_back(128,4,f+"/bank0");banks.emplace_back(64,2,f+"/bank1");banks.emplace_back(128,2,f+"/bank2");
  std::vector<SparseFixed>fixed;std::vector<std::vector<uint32_t>>want_states(3),want_final(3),states(3),final(3);
  for(int k=0;k<4;k++)fixed.emplace_back(load<int8_t>(f+"/fixed"+std::to_string(k)+".weights"),load<int32_t>(f+"/fixed"+std::to_string(k)+".threshold"));
  for(int k=0;k<3;k++){want_final[k]=load<uint32_t>(f+"/"+stage+".final"+std::to_string(k));if(verification=="all"){want_states[k]=load<uint32_t>(f+"/"+stage+".states"+std::to_string(k));states[k].reserve(want_states[k].size());}}
  std::vector<int64_t>calibration;calibration.reserve(10000);for(int i=0;i<10000;i++){auto a=ns(),b=ns();calibration.push_back(b-a);}dump(out+".clock_ns",calibration);
  int64_t setup=0,final_readback=0,final_check=0,cleanup=0;uint32_t source=0,external=0;std::array<int,3>oldm={-1,-1,-1};uint64_t execution_writes=0,total_polls=0;
  // The runtime retains views into the supplied program; keep its bytes alive
  // through every enqueue, just as in the qualified parent benchmark.
  std::vector<uint8_t>program;
  auto capture=[&](const std::vector<uint32_t>&p){auto v=p;akida::dma::dma_config_read(v.data()+2,v.size()-2,v.data(),static_cast<akida::HardwareDeviceImpl*>(dev.get())->dma_config(),static_cast<akida::HardwareDeviceImpl*>(dev.get())->mem(),d.get(),true);return v;};
  auto fixed_check=[&](){for(int k=0;k<4;k++){auto p=load<uint32_t>(f+"/fixed"+std::to_string(k)+".ram");need(capture(p)==p,"native fixed route RAM unchanged");}};
  if(mode=="native"){
   auto s=ns();d=std::make_unique<LabDriver>();d->dma_read_min=128;dev=akida::HardwareDevice::create(d.get());program=load<uint8_t>(f+"/model.bin");auto info=dev->program(program.data(),program.size());
   need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&info.output_is_dense()&&info.activation_enabled(),"unchanged single resident program");dev->toggle_learn(true);
   need(dev->set_batch_size(4,false)==4,"four carry buffers");static_cast<akida::HardwareDeviceImpl*>(dev.get())->pipeline(false);
   for(auto&bank:banks)need(capture(bank.initial)==bank.initial,"native initial RAM");fixed_check();external=dev->scratch_alloc(128);std::vector<uint8_t>zero(128);d->write(external,zero.data(),128);source=external;setup=ns()-s;
  }
  std::vector<std::array<int64_t,14>>timing(count);std::vector<uint8_t>scores;std::vector<int32_t>answers;scores.reserve(count*128);answers.reserve(count);std::vector<uint8_t>last(128);Dictionary dictionary;
  int64_t cpu_start=cpu_ns();
  for(int i=0;i<count;i++){
   kick_watchdog();const auto e=&plan[i*7];auto&tt=timing[i];std::vector<uint8_t>y;int answer=-1;auto s=ns();tt[9]=s;
   if(mode=="native"){
    auto before_reads=d->read_bytes,before_writes=d->write_bytes;
    for(int k=0;k<3;k++){int next=e[2+k]>=0;if(next!=oldm[k]){auto&p=next?banks[k].learn:banks[k].infer;akida::dma::dma_config_write(p.data(),p.size(),static_cast<akida::HardwareDeviceImpl*>(dev.get())->dma_config(),static_cast<akida::HardwareDeviceImpl*>(dev.get())->external_mem(),d.get(),true);oldm[k]=next;}}
    auto a=ns();tt[0]=a-s;
    if(e[1]!=2){std::array<uint8_t,32>raw{};for(int b=0;b<4;b++)for(int j=0;j<4;j++)raw[b*8+((e[0]>>b)&1)*4+j]=1;d->write(source+32,raw.data(),32);}
    uint64_t phase=uint64_t(1)<<(8*((e[1]==1||e[1]==4)?1:e[1]==2?5:0));if(e[1]==4)phase|=uint64_t(1)<<32;if(e[1]==3)phase|=uint64_t(1)<<48;if(e[6])phase|=uint64_t(1)<<56;d->write(source+64,&phase,8);
    auto b=ns();tt[1]=b-a;d->hold_queue();uint32_t in=source;int wire=e[5];need(dev->enqueue(in,&wire),"native enqueue");auto desc=d->last_descriptor;auto target=desc[0]>>16;need(target&&((desc[8]>>16)&1023)==uint32_t(wire+1),"descriptor job/class");
    auto c=ns();tt[2]=c-b;auto bw=d->writes;d->release_queue();auto released=ns();tt[3]=released-c;
    auto deadline=Clock::now()+std::chrono::seconds(2);uint64_t polls=0;
    while((d->read32(LabDriver::REG+0x28010)>>16)!=target){polls++;need(Clock::now()<deadline,"observed completion timeout");}
    auto completed=ns();tt[4]=completed-released;tt[13]=polls+1;total_polls+=polls+1;execution_writes+=d->writes-bw-1;
    auto tensor=dev->fetch();need(bool(tensor)&&tensor->type()==akida::TensorType::uint8&&tensor->size()==128,"completed output fetch");auto p=tensor->data<uint8_t>();y.assign(p,p+128);if(e[1]!=2)source=desc[7]+32;
    auto fetched=ns();tt[5]=fetched-completed;tt[10]=fetched;tt[11]=d->read_bytes-before_reads;tt[12]=d->write_bytes-before_writes;
   }else if(mode=="replica"){
    auto x=last;observe(x,e);auto a=ns();tt[1]=a-s;
    auto z=fixed[0](x);for(int k=0;k<3;k++){z=banks[k].step(z,e[2+k]);if(k==0){z=fixed[1](z);z=fixed[2](z);}else if(k==1)z=fixed[3](z);}if(e[1]!=2)last=z;y=std::move(z);auto b=ns();tt[6]=b-a;tt[10]=b;
   }else{
    answer=dictionary.step(e);auto a=ns();tt[6]=a-s;tt[10]=a;
   }
   auto a=ns();
   if(verification=="all"&&mode!="dictionary")for(int k=0;k<3;k++){auto v=mode=="native"?capture(banks[k].initial):banks[k].cpu.ram;states[k].insert(states[k].end(),v.begin(),v.end());}
   auto b=ns();tt[7]=b-a;
   if(mode=="dictionary"){need(answer==expected_dict[i],"CPU dictionary semantics");answers.push_back(answer);}
   else{
    need(std::equal(y.begin(),y.end(),expected.begin()+i*128),"exact output bytes");scores.insert(scores.end(),y.begin(),y.end());
    if(verification=="all")for(int k=0;k<3;k++)need(std::equal(states[k].end()-banks[k].initial.size(),states[k].end(),want_states[k].begin()+i*banks[k].initial.size()),"exact per-event learning RAM");
   }
   tt[8]=ns()-b;
  }
  auto thread_cpu=cpu_ns()-cpu_start;
  if(mode!="dictionary"){
   auto a=ns();for(int k=0;k<3;k++)final[k]=mode=="native"?capture(banks[k].initial):banks[k].cpu.ram;auto b=ns();final_readback=b-a;
   for(int k=0;k<3;k++)need(final[k]==want_final[k],"exact final learning RAM");if(mode=="native")fixed_check();final_check=ns()-b;
  }
  if(mode=="native"){auto a=ns();need(execution_writes==0,"no host writes during native execution");dev->scratch_free(external);dev->unprogram();cleanup=ns()-a;}
  uint64_t checksum=0,agreed_checksum=0;int64_t batch_time=0,agreed_time=0;uint64_t agreed_operations=0;constexpr int batch_repeats=512;
  if(mode=="dictionary"){
   auto a=ns();for(int r=0;r<batch_repeats;r++){Dictionary q;for(int i=0;i<count;i++)checksum=checksum*1099511628211ULL+uint64_t(q.step(&plan[i*7])+1);}batch_time=ns()-a;
   // A separate batch times the resident agreement+store kernel without one
   // clock call per tiny update. Proposals come from source evidence, not truth.
   Dictionary replay;std::vector<std::array<int,3>>agreed;
   for(int i=0;i<count;i++){auto e=&plan[i*7];replay.step(e);if(e[1]==2&&replay.a[e[0]]>=0&&replay.a[e[0]]==replay.b[e[0]])agreed.push_back({e[0],replay.a[e[0]],replay.b[e[0]]});}
   need(!agreed.empty(),"agreed update batch");uint64_t want=0;for(int r=0;r<4096;r++)for(auto q:agreed)want=want*1099511628211ULL+uint64_t(q[1]+1);
   Dictionary resident;a=ns();for(int r=0;r<4096;r++)for(auto q:agreed)agreed_checksum=agreed_checksum*1099511628211ULL+uint64_t(resident.agree(q[0],q[1],q[2])+1);agreed_time=ns()-a;
   need(want==agreed_checksum,"batched agreed CPU updates checksum");agreed_operations=4096*agreed.size();
  }
  std::vector<int64_t>flat;flat.reserve(count*14);for(auto&row:timing)flat.insert(flat.end(),row.begin(),row.end());dump(out+".timing",flat);
  if(mode=="dictionary")dump(out+".answers",answers);else{dump(out+".scores",scores);for(int k=0;k<3;k++){dump(out+".final"+std::to_string(k),final[k]);if(verification=="all")dump(out+".states"+std::to_string(k),states[k]);}}
  std::cout<<"{\"status\":\"PASS\",\"mode\":\""<<mode<<"\",\"events\":"<<count<<",\"setup_ns\":"<<setup<<",\"final_readback_ns\":"<<final_readback<<",\"final_check_ns\":"<<final_check<<",\"cleanup_ns\":"<<cleanup<<",\"thread_cpu_with_audit_ns\":"<<thread_cpu<<",\"polls\":"<<total_polls<<",\"host_writes_during_execution\":"<<execution_writes<<",\"exact_scores_and_final_RAM\":true,\"dictionary_batch_repeats\":"<<(mode=="dictionary"?batch_repeats:0)<<",\"dictionary_batch_ns\":"<<batch_time<<",\"checksum\":"<<checksum<<",\"agreed_update_batch_ns\":"<<agreed_time<<",\"agreed_update_batch_operations\":"<<agreed_operations<<",\"agreed_update_batch_checksum\":"<<agreed_checksum<<",\"device_opened\":"<<(mode=="native"?"true":"false")<<",\"cleanup\":\""<<(mode=="native"?"unprogrammed":"device_not_opened")<<"\"}\n";
 }catch(const std::exception&e){if(dev)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
