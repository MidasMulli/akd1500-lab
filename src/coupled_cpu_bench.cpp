// CPU comparator for the exact two-bank teaching stream; no physical device open.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include <array>

int main(int argc,char**argv){
 try{
  need(argc==6,"fixture stream output samples repeats");
  std::string f=argv[1],s=argv[2],out=argv[3];int count=std::stoi(argv[4]),repeats=std::stoi(argv[5]);
  need(count==656&&repeats>0&&repeats<=256,"bounded exact coupled CPU stream");
  auto input=load<uint8_t>(s+".inputs");auto labels=load<int32_t>(s+".labels");auto masks=load<uint8_t>(s+".masks");
  need(input.size()==size_t(count*512)&&labels.size()==size_t(count)&&masks.size()==size_t(count),"CPU stream shapes");
  auto a0=load<uint32_t>(f+".bank0.ram"),b0=load<uint32_t>(f+".bank1.ram");
  need(a0.size()==1282&&b0.size()==1282,"CPU native RAM shapes");
  std::vector<int32_t> outputs;std::vector<int64_t> times,caller;uint64_t checksum=0;
  for(int repeat=0;repeat<repeats;repeat++){
   CPU a(512,64,32,4,a0),b(512,64,32,4,b0);int64_t start=0,cstart=0;
   for(int i=0;i<count;i++){
    kick_watchdog();need(masks[i]<4&&labels[i]>=0&&labels[i]<4,"CPU shared teacher");
    if(i==512){start=ns();cstart=cpu_ns();}
    std::vector<uint32_t> x(16);for(int k=0;k<512;k++)x[k/32]|=uint32_t(input[i*512+k]!=0)<<(k%32);
    auto p=a.forward(x,labels[i],masks[i]&1);std::array<int,4> supports{};
    for(int j=0;j<64;j++)supports[j/16]+=p[j]>27;
    auto identity=std::max_element(supports.begin(),supports.end())-supports.begin();
    std::fill(x.begin(),x.end(),0);for(int k=identity*4;k<(identity+1)*4;k++)x[k]=0xffffffff;
    p=b.forward(x,labels[i],masks[i]&2);supports.fill(0);
    for(int j=0;j<64;j++)supports[j/16]+=p[j]>27;
    for(auto v:supports){checksum=checksum*1099511628211ULL+uint64_t(v);if(repeat==0)outputs.push_back(v);}
    if(i==639){times.push_back(ns()-start);caller.push_back(cpu_ns()-cstart);}
   }
   if(repeat==repeats-1){dump(out+".bank0.ram",a.ram);dump(out+".bank1.ram",b.ram);}
  }
  dump(out+".scores",outputs);dump(out+".probe_wall_ns",times);dump(out+".probe_cpu_ns",caller);
  std::cout<<"{\"status\":\"PASS\",\"physical_device_opened\":false,\"samples_per_repeat\":656,\"timed_test_events_per_repeat\":128,\"repeats\":"<<repeats<<",\"checksum\":"<<checksum<<"}\n";
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
 return 0;
}
