// Pong native responses: each reply follows the actual physical forward.
#include "binary_reference.hpp"
#include <akida/dense.h>
#include <akida/program_info.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <cstdarg>
using Clock=std::chrono::steady_clock;
static auto epoch=Clock::now();
static int64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
extern "C" void msleep(uint32_t x){std::this_thread::sleep_for(std::chrono::milliseconds(x));}
extern "C" int64_t time_ms(){return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-epoch).count();}
extern "C" void kick_watchdog(){need(Clock::now()-epoch<std::chrono::seconds(175),"owned stream watchdog");}
extern "C" [[noreturn]] void panic(const char* fmt,...){char b[2048];va_list a;va_start(a,fmt);vsnprintf(b,sizeof(b),fmt,a);va_end(a);throw std::runtime_error(b);}
template<class T>static std::vector<T> load(const std::string& p){std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f),"open fixture");auto n=f.tellg();need(n>=0&&size_t(n)%sizeof(T)==0,"fixture shape");std::vector<T> v(size_t(n)/sizeof(T));f.seekg(0);f.read((char*)v.data(),n);need(bool(f),"read fixture");return v;}
template<class T>static void dump(const std::string& p,const std::vector<T>& v){std::ofstream f(p,std::ios::binary);f.write((char*)v.data(),v.size()*sizeof(T));need(bool(f),"save raw state");}
template<class T>static void append(std::ofstream& f,const std::vector<T>& v){f.write((char*)v.data(),v.size()*sizeof(T));f.flush();need(bool(f),"append raw evidence");}
template<class T>static void array(const std::vector<T>& v){std::cout<<'[';for(size_t k=0;k<v.size();k++){if(k)std::cout<<',';std::cout<<v[k];}std::cout<<']';}
int main(int argc,char**argv){
 std::unique_ptr<LabDriver> driver;akida::HardwareDevicePtr device;
 try{
  need(argc==4,"fixture-prefix output-prefix maximum-forwards");std::string f=argv[1],out=argv[2];int cap=std::stoi(argv[3]);need(cap>0&&cap<=4096,"bounded stream cap");
  int n,u,nw,classes;{std::ifstream m(f+".meta");m>>n>>u>>nw>>classes;need(bool(m)&&n==1024&&u==160&&nw==64&&classes==5,"reviewed stream shape");}
  auto initial=load<uint32_t>(f+".ram");CPU reference(n,u,nw,classes,initial);auto before=initial;auto program=load<uint8_t>(f+".bin");
  driver=std::make_unique<LabDriver>();device=akida::HardwareDevice::create(driver.get());auto info=device->program(program.data(),program.size());
  need(info.number_of_passes()==1&&info.can_learn()&&!info.activation_enabled(),"single-pass potential learner");device->toggle_learn(true);need(device->set_batch_size(1,false)==1,"one-sample queue");auto address=device->scratch_alloc(n);
  std::vector<uint32_t> captured(device->learn_mem_size());device->learn_mem(captured.data());need(captured==initial,"fresh native initial RAM");
  std::ofstream inputs(f+".inputs",std::ios::binary),labels(f+".labels",std::ios::binary),native(out+".scores",std::ios::binary),cpu(out.substr(0,out.find_last_of('/'))+"/cpu.scores",std::ios::binary);
  std::cout<<"{\"kind\":\"ready\",\"inputs\":"<<n<<",\"neurons\":"<<u<<"}\n"<<std::flush;
  int count=0,last=-1;std::string line;
  while(std::getline(std::cin,line)){
   kick_watchdog();if(line=="X")break;std::istringstream command(line);std::string op,hex;int id,label;command>>op>>id>>label>>hex;
   need(bool(command)&&op=="F"&&id>last&&label>=0&&label<classes&&hex.size()==size_t(2*n)&&count<cap,"bounded forward command");last=id;
   std::vector<uint8_t> x(n);std::vector<uint32_t> packed(n/32);
   for(int k=0;k<n;k++){need(hex[2*k]=='0'&&(hex[2*k+1]=='0'||hex[2*k+1]=='1'),"binary input hex");x[k]=hex[2*k+1]-'0';packed[k/32]|=uint32_t(x[k])<<(k%32);}
   auto start=ns();driver->write(address,x.data(),n);int32_t target=label;need(device->enqueue(address,&target),"native enqueue");akida::TensorUniquePtr result;auto deadline=Clock::now()+std::chrono::seconds(2);
   while(!(result=device->fetch()))need(Clock::now()<deadline,"native completion deadline");auto duration=ns()-start;
   need(result->type()==akida::TensorType::int32&&result->size()==size_t(u),"native response shape");std::vector<int32_t> got(result->data<int32_t>(),result->data<int32_t>()+u);auto want=reference.forward(packed,label,true);need(got==want,"native reference scores");
   device->learn_mem(captured.data());need(captured==reference.ram,"native reference full RAM");if(label==classes-1)need(captured==before,"protected query changed RAM");
   need(std::equal(initial.begin()+2+(u-u/classes)*(4+n/32),initial.end(),captured.begin()+2+(u-u/classes)*(4+n/32)),"query-class RAM changed");
   std::vector<int> changed;int edits=0,stride=4+n/32;for(int j=0;j<u;j++){int base=2+j*stride;bool differs=false;for(int k=0;k<stride;k++){differs|=before[base+k]!=captured[base+k];if(k>=4)edits+=__builtin_popcount(before[base+k]^captured[base+k]);}if(differs)changed.push_back(j);}
   append(inputs,x);append(labels,std::vector<int32_t>{target});append(native,got);append(cpu,want);dump(out+".snapshot_"+std::to_string(count)+".ram",captured);before=captured;
   std::cout<<"{\"kind\":\"forward\",\"id\":"<<id<<",\"index\":"<<count<<",\"label\":"<<label<<",\"transaction_ns\":"<<duration<<",\"host_monotonic_ns\":"<<ns()<<",\"connection_bit_edits\":"<<edits<<",\"changed_records\":";array(changed);std::cout<<",\"scores\":";array(got);std::cout<<"}\n"<<std::flush;count++;
  }
  dump(out+".ram",captured);dump(out.substr(0,out.find_last_of('/'))+"/cpu.ram",reference.ram);device->scratch_free(address);device->unprogram();
  std::cout<<"{\"kind\":\"closed\",\"forwards\":"<<count<<",\"cleanup\":\"unprogrammed\"}\n"<<std::flush;return 0;
 }catch(const std::exception& e){if(device)try{device->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
