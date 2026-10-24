// Documented PCIe environment: raw RGB -> native HRC -> CNP -> FNP learner.
// Host reference independently executes all spatial and learning operations.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include <arm_neon.h>

struct Front {
 int width;
 std::vector<int8_t> w0,w1;
 std::vector<int32_t> t0,t1;
 std::vector<uint8_t> a,b;
 Front(const std::string& f,int w):width(w),w0(load<int8_t>(f+".w0")),w1(load<int8_t>(f+".w1")),t0(load<int32_t>(f+".t0")),t1(load<int32_t>(f+".t1")),a(16*16*w),b(512){
  need(w0.size()==size_t(27*w)&&w1.size()==size_t(9*w*8)&&t0.size()==size_t(w)&&t1.size()==8,"front shape");
  for(auto v:w0)need(v>=-3&&v<=3,"HRC 16bit accumulator bound");
  for(auto v:w1)need(v>=-7&&v<=7,"CNP 16bit accumulator bound");
 }
 // Exact, preallocated, NEON SIMD over eight output channels. Sparse variant
 // skips zero input contributions. Includes padding, ReLU and FC axon packing.
 std::vector<uint32_t> forward(const uint8_t* x,bool sparse) {
  for(int y=0;y<16;y++)for(int z=0;z<16;z++)for(int j=0;j<width;j+=8){
   int16x8_t acc=vdupq_n_s16(0);
   for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++){
    int iy=2*y+ky,ix=2*z+kx;if(iy>=32||ix>=32)continue;
    for(int c=0;c<3;c++){
     int value=x[(iy*32+ix)*3+c];if(sparse&&!value)continue;
     auto weights=vmovl_s8(vld1_s8(&w0[((ky*3+kx)*3+c)*width+j]));
     acc=vmlaq_n_s16(acc,weights,int16_t(value));
    }
   }
   int16_t v[8];vst1q_s16(v,acc);
   for(int k=0;k<8;k++)a[(y*16+z)*width+j+k]=v[k]>t0[j+k];
  }
  for(int y=0;y<8;y++)for(int z=0;z<8;z++){
   int16x8_t acc=vdupq_n_s16(0);
   for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++){
    int iy=2*y+ky,ix=2*z+kx;if(iy>=16||ix>=16)continue;
    for(int c=0;c<width;c++){
     int value=a[(iy*16+ix)*width+c];if(sparse&&!value)continue;
     auto weights=vmovl_s8(vld1_s8(&w1[(((2-ky)*3+2-kx)*width+c)*8]));
     acc=vmlaq_n_s16(acc,weights,int16_t(value));
    }
   }
   int16_t v[8];vst1q_s16(v,acc);
   for(int k=0;k<8;k++)b[(k*8+z)*8+y]=v[k]>t1[k];
  }
  std::vector<uint32_t> bits(16);
  for(int i=0;i<512;i++)bits[i/32]|=uint32_t(b[i])<<(i%32);
  return bits;
 }
};

int main(int argc,char**argv){
 std::vector<uint8_t> program; // outlives the runtime (which retains its pointer).
 std::unique_ptr<LabDriver> driver;akida::HardwareDevicePtr dev;bool programmed=false;
 try {
  need(argc==8,"mode fixture output count learn batch repeats");
  std::string mode=argv[1],f=argv[2],out=argv[3];int count=std::stoi(argv[4]);bool learn=std::stoi(argv[5]);int batch=std::stoi(argv[6]),repeats=std::stoi(argv[7]);
  need((mode=="cpu-dense"||mode=="cpu-sparse"||mode=="native")&&count>=8&&count<=96&&count%8==0&&(batch==1||batch==8)&&repeats>=1&&repeats<=5,"bounded arguments");
  int side,width,c,n,u,nw,classes,available;{std::ifstream m(f+".meta");m>>side>>width>>c>>n>>u>>nw>>classes>>available;need(bool(m)&&side==32&&n==512&&u==128&&available==96,"shape contract");}
  Front front(f,width);auto x=load<uint8_t>(f+".inputs"),features=load<uint8_t>(f+".features");auto labels=load<int32_t>(f+".labels");auto initial=load<uint32_t>(f+".ram");
  need(x.size()==96*3072&&features.size()==96*512&&initial.size()==2562,"fixture sizes");
  CPU reference(n,u,nw,classes,initial);std::vector<int32_t> scores;std::vector<int64_t> timings,callers;uint64_t checksum=0;
  if(mode!="native"){
   for(int r=0;r<repeats;r++){
    CPU cpu(n,u,nw,classes,initial);
    for(int i=0;i<count;i++){
     auto t=ns(),cp=cpu_ns();auto bits=front.forward(x.data()+i*3072,mode=="cpu-sparse");auto y=cpu.forward(bits,labels[i],learn);callers.push_back(cpu_ns()-cp);timings.push_back(ns()-t);
     need(std::equal(front.b.begin(),front.b.end(),features.data()+i*512),"independent CPU CNN feature contract");
     for(auto value:y)checksum=checksum*1099511628211ULL+uint64_t(value);
     if(!r)scores.insert(scores.end(),y.begin(),y.end());
    }
    if(r==repeats-1)reference=std::move(cpu);
   }
   dump(out+".scores",scores);dump(out+".ram",reference.ram);dump(out+".latency_ns",timings);dump(out+".caller_ns",callers);
   std::cout<<"{\"status\":\"PASS\",\"mode\":\""<<mode<<"\",\"samples\":"<<count<<",\"repeats\":"<<repeats<<",\"checksum\":"<<checksum<<",\"physical_device_opened\":false}\n";return 0;
  }
  need(repeats==1,"one native stream per fresh program");
  driver=std::make_unique<LabDriver>();dev=akida::HardwareDevice::create(driver.get());program=load<uint8_t>(f+".bin");auto info=dev->program(program.data(),program.size());programmed=true;
  need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&!info.activation_enabled(),"native image/learner interface");
  dev->toggle_learn(learn);need(dev->set_batch_size(batch==8?9:1,false)==size_t(batch==8?9:1),"queue capacity");
  auto capture=[&](){if(!learn)dev->toggle_learn(true);std::vector<uint32_t> ram(dev->learn_mem_size());dev->learn_mem(ram.data());if(!learn)dev->toggle_learn(false);return ram;};
  need(capture()==initial,"initial full learner RAM");dump(out+".initial.ram",initial);
  std::vector<uint32_t> addresses;for(int j=0;j<batch;j++)addresses.push_back(dev->scratch_alloc(3072));
  auto rb=driver->read_bytes,wb=driver->write_bytes;std::ofstream snaps(out+".snapshots",std::ios::binary);
  for(int first=0;first<count;first+=batch){
   kick_watchdog();auto t=ns(),cp=cpu_ns();
   for(int j=0;j<batch;j++){
    driver->write(addresses[j],x.data()+(first+j)*3072,3072);
    need(dev->enqueue(addresses[j],learn?&labels[first+j]:nullptr),"enqueue image");
   }
   std::vector<int32_t> got;auto deadline=Clock::now()+std::chrono::seconds(2);
   for(int j=0;j<batch;j++){
    akida::TensorUniquePtr y;while(!(y=dev->fetch()))need(Clock::now()<deadline,"owned output deadline");
    need(y->type()==akida::TensorType::int32&&y->size()==128,"native potential shape");auto v=y->data<int32_t>();got.insert(got.end(),v,v+128);
   }
   callers.push_back(cpu_ns()-cp);timings.push_back(ns()-t);
   scores.insert(scores.end(),got.begin(),got.end());dump(out+".scores",scores);
   for(int j=0;j<batch;j++){
    auto bits=front.forward(x.data()+(first+j)*3072,true);need(std::equal(front.b.begin(),front.b.end(),features.data()+(first+j)*512),"CPU image features");
    auto want=reference.forward(bits,labels[first+j],learn);
    if(!std::equal(want.begin(),want.end(),got.begin()+j*128)){std::cerr<<"mismatch image "<<first+j<<"\n";throw std::runtime_error("native end-to-end score mismatch");}
   }
   // Diagnostics are outside the timed interval and identical at each boundary.
   auto ram=capture();snaps.write((char*)ram.data(),ram.size()*4);snaps.flush();need(ram==reference.ram,"native complete learner RAM mismatch");
  }
  auto reads=driver->read_bytes-rb,writes=driver->write_bytes-wb;snaps.close();auto final=capture();dump(out+".ram",final);dump(out+".latency_ns",timings);dump(out+".caller_ns",callers);
  for(auto a:addresses)dev->scratch_free(a);dev->unprogram();programmed=false;
  std::cout<<"{\"status\":\"PASS\",\"mode\":\"native\",\"samples\":"<<count<<",\"repeats\":1,\"batch\":"<<batch<<",\"read_bytes_including_diagnostics\":"<<reads<<",\"write_bytes_including_diagnostics\":"<<writes<<",\"exact_scores_and_full_RAM\":true,\"cleanup\":\"unprogrammed\"}\n";
 }catch(const std::exception& e){if(programmed&&dev)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
