// Same image decisions, exact-overlap output versus packed task alternatives.
// No Akida runtime or device access. Prototype packing is resident setup work.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
using Bits=std::array<uint64_t,16>;
using Clock=std::chrono::steady_clock;
static int64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
static int64_t cpu_ns(){timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return int64_t(t.tv_sec)*1000000000+t.tv_nsec;}
static void need(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
template<class T>static std::vector<T> load(const std::string& p){std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f),"input file");auto n=f.tellg();need(n>=0&&size_t(n)%sizeof(T)==0,"input size");std::vector<T> a(size_t(n)/sizeof(T));f.seekg(0);f.read((char*)a.data(),n);need(bool(f),"read input");return a;}
template<class T>static void save(const std::string& p,const std::vector<T>& a){std::ofstream f(p,std::ios::binary);f.write((const char*)a.data(),a.size()*sizeof(T));need(bool(f),"save output");}
static Bits pack(const uint8_t* x,bool pooled){
 Bits b{};
 if(pooled){for(int r=0;r<16;r++)for(int c=0;c<16;c++){
  int i=64*r+2*c,k=16*r+c;bool bit=x[i]||x[i+1]||x[i+32]||x[i+33];b[k/64]|=uint64_t(bit)<<(k%64);
 }}else for(int k=0;k<1024;k++)b[k/64]|=uint64_t(x[k]!=0)<<(k%64);
 return b;
}
int main(int argc,char** argv){try{
 need(argc==4,"data-prefix output-prefix raw|compacted");std::string data=argv[1],out=argv[2],kind=argv[3];
 need(kind=="raw"||kind=="compacted","representation");bool pooled=kind=="compacted";int words=pooled?4:16;
 auto templates=load<uint8_t>(data+".templates"),inputs=load<uint8_t>(data+".inputs");
 need(templates.size()==3200*1024&&inputs.size()%1024==0,"frozen bank dimensions");int count=int(inputs.size()/1024);
 need(count==1000,"frozen fresh-test count");std::vector<Bits> bank;std::vector<int32_t> bias(3200);
 for(int j=0;j<3200;j++){bank.push_back(pack(templates.data()+j*1024,pooled));for(int k=0;k<words;k++)bias[j]+=__builtin_popcountll(bank[j][k]);}
 std::vector<int32_t> scores(3200),observed;
 auto classify=[&](const uint8_t* x,int method){
  Bits q=pack(x,pooled);int winner=0,best=2048;
  for(int j=0;j<3200;j++){
   int value=0;
   if(method==0){for(int k=0;k<words;k++)value+=__builtin_popcountll(q[k]&bank[j][k]);scores[j]=value;value=bias[j]-2*value;}
   else if(method==1){for(int k=0;k<words;k++){value+=__builtin_popcountll(q[k]^bank[j][k]);if(value>=best)break;}}
   else{
#if defined(__aarch64__)
    for(int k=0;k<words;k+=2){auto a=vld1q_u8((const uint8_t*)(q.data()+k));auto b=vld1q_u8((const uint8_t*)(bank[j].data()+k));value+=vaddvq_u8(vcntq_u8(veorq_u8(a,b)));if(value>=best)break;}
#else
    throw std::runtime_error("NEON arm requires AArch64");
#endif
   }
   if(value<best){best=value;winner=j;}
  }
  return winner/320;
 };
 std::array<std::vector<int32_t>,3> predictions;std::array<std::vector<int64_t>,3> wall,cpu;
 for(int method=0;method<3;method++)for(int i=0;i<16;i++)classify(inputs.data()+i*1024,method);
 for(int rep=0;rep<3;rep++)for(int offset=0;offset<3;offset++){
  int method=(rep+offset)%3;
  for(int i=0;i<count;i++){
   auto start=ns(),c0=cpu_ns();int pred=classify(inputs.data()+i*1024,method);auto spent=cpu_ns()-c0,elapsed=ns()-start;
   predictions[method].push_back(pred);wall[method].push_back(elapsed);cpu[method].push_back(spent);
   if(method==0&&rep==0)observed.insert(observed.end(),scores.begin(),scores.end());
  }
 }
 need(predictions[0]==predictions[1]&&predictions[1]==predictions[2],"all CPU methods same decisions");
 for(int method=0;method<3;method++){
  auto prefix=out+".m"+std::to_string(method);save(prefix+".predictions",predictions[method]);save(prefix+".wall_ns",wall[method]);save(prefix+".cpu_ns",cpu[method]);
 }
 save(out+".overlap_scores",observed);
 std::cout<<"{\"status\":\"CPU_ONLY_PASS\",\"kind\":\""<<kind<<"\",\"images\":"<<count<<",\"representations\":1,\"methods\":3,\"repetitions\":3}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
