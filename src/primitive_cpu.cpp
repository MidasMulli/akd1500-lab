// CPU-native implementations of the frozen binary-overlap/two-count primitive.
// No Akida runtime or physical-device access. Dense input packing is timed.
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
using Bits=std::array<uint64_t,8>;using Result=std::array<int32_t,2>;
static void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class T>static std::vector<T> load(const std::string& p){std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f),"CPU fixture open");auto n=f.tellg();need(n>=0&&size_t(n)%sizeof(T)==0,"CPU fixture shape");std::vector<T>a(size_t(n)/sizeof(T));f.seekg(0);f.read((char*)a.data(),n);need(bool(f),"CPU fixture read");return a;}
template<class T>static void save(const std::string&p,const std::vector<T>&a){std::ofstream f(p,std::ios::binary);f.write((const char*)a.data(),a.size()*sizeof(T));need(bool(f),"CPU output write");}
static int64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static int64_t cpu_ns(){timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return int64_t(t.tv_sec)*1000000000+t.tv_nsec;}
static Bits pack(const uint8_t*x,int&active){Bits q{};active=0;
#if defined(__aarch64__)
 const uint8_t shifts[16]={1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};auto mask=vld1q_u8(shifts);
 for(int i=0;i<512;i+=16){auto a=vld1q_u8(x+i);active+=vaddvq_u8(a);auto b=vandq_u8(vcgtq_u8(a,vdupq_n_u8(0)),mask);uint64_t v=uint64_t(vaddv_u8(vget_low_u8(b)))|(uint64_t(vaddv_u8(vget_high_u8(b)))<<8);q[i/64]|=v<<(i%64);}
#else
 for(int i=0;i<512;i++){active+=x[i]!=0;q[i/64]|=uint64_t(x[i]!=0)<<(i%64);}
#endif
 return q;
}
struct Bank{
 int u;std::vector<Bits> words;std::array<std::vector<uint16_t>,512> index;
 Bank(int width,const std::vector<uint8_t>&w):u(width),words(width){need(w.size()==size_t(u*512),"CPU weight shape");for(int j=0;j<u;j++){int count=0;for(int i=0;i<512;i++){need(w[j*512+i]<=1,"binary weights");if(w[j*512+i]){words[j][i/64]|=uint64_t(1)<<(i%64);index[i].push_back(j);count++;}}need(count==32,"32 connections");}}
 __attribute__((noinline)) Result run(const uint8_t*x,int method)const{
  asm volatile("" ::: "memory");int active;auto q=pack(x,active);Result out{};
  if(method==2){if(active<=15)return out;method=active<=64?1:0;}
  if(method==0){
   for(int j=0;j<u;j++){int value=0;
#if defined(__aarch64__)
    auto sum=vdupq_n_u8(0);for(int k=0;k<8;k+=2){auto a=vld1q_u8((const uint8_t*)(q.data()+k));auto b=vld1q_u8((const uint8_t*)(words[j].data()+k));sum=vaddq_u8(sum,vcntq_u8(vandq_u8(a,b)));}value=vaddlvq_u8(sum);
#else
    for(int k=0;k<8;k++)value+=__builtin_popcountll(q[k]&words[j][k]);
#endif
    out[j/(u/2)]+=value>15;
   }
  }else{
   std::vector<uint16_t> counts(u,0);
   for(int k=0;k<8;k++){uint64_t a=q[k];while(a){int bit=__builtin_ctzll(a);for(auto j:index[k*64+bit])counts[j]++;a&=a-1;}}
   for(int j=0;j<u;j++)out[j/(u/2)]+=counts[j]>15;
  }
  return out;
 }
};
int main(int argc,char**argv){try{
 need(argc==4,"fixture-prefix output-prefix width");std::string f=argv[1],out=argv[2];int u=std::stoi(argv[3]);need(u==128||u==256||u==512||u==1024||u==2048,"CPU frozen width");
 auto x=load<uint8_t>(f+".inputs");auto w=load<uint8_t>(f+".weights");auto want=load<int32_t>(f+".expected");need(x.size()==160*512&&want.size()==320,"CPU input dimensions");for(auto b:x)need(b<=1,"binary inputs");Bank bank(u,w);
 std::array<std::vector<int>,5> groups;const int ds[5]={0,8,64,256,512};
 for(int i=0;i<160;i++){int active=0;for(int k=0;k<512;k++)active+=x[i*512+k];auto at=std::find(ds,ds+5,active);need(at!=ds+5,"CPU density");groups[at-ds].push_back(i);}
 for(auto&g:groups)need(g.size()==32,"balanced density bank");
 std::vector<int32_t> scores;
 for(int method=0;method<3;method++)for(int i=0;i<160;i++){auto y=bank.run(x.data()+i*512,method);need(y[0]==want[i*2]&&y[1]==want[i*2+1],"exact CPU outputs");scores.insert(scores.end(),y.begin(),y.end());}
 std::vector<int64_t> wall(45),cpu(45);uint64_t checksum=0;
 for(int repeat=0;repeat<3;repeat++)for(int offset=0;offset<3;offset++){
  int method=(offset+repeat)%3;
  for(int d=0;d<5;d++){
   auto t=ns(),ct=cpu_ns();for(int r=0;r<256;r++)for(int i:groups[d]){auto y=bank.run(x.data()+i*512,method);checksum=checksum*1099511628211ULL+uint64_t(y[0]+y[1]*2049);}
   auto cp=cpu_ns()-ct,elapsed=ns()-t;int at=repeat*15+method*5+d;wall[at]=elapsed;cpu[at]=cp;
  }
 }
 save(out+".scores",scores);save(out+".wall_ns",wall);save(out+".cpu_ns",cpu);
 std::cout<<"{\"status\":\"CPU_EXACT_PASS\",\"width\":"<<u<<",\"timed_calls_per_density_sample\":8192,\"temporal_batches\":3,\"methods\":3,\"checksum\":"<<checksum<<",\"neon\":"
#if defined(__aarch64__)
 <<"true"
#else
 <<"false"
#endif
 <<",\"physical_device_opened\":false}\n";
 }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
