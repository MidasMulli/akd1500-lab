// Resident spatial fabric -> native learner. No registered host weight memory.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
#include "hardware_device_impl.h"
#include "dma_config_mem_rw.h"
#include <arm_neon.h>
#include <omp.h>

static int64_t process_ns(){timespec t{};clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t);return int64_t(t.tv_sec)*1000000000+t.tv_nsec;}
struct Spatial {
 int side,w,c,h,hs,ps,n;std::vector<int8_t>w0,w1,w2;std::vector<int32_t>t0,t1,t2;
 std::vector<uint8_t>a,b,features;std::vector<int16_t>z;
 Spatial(const std::string& f,int ss,int ww,int cc,int nn):side(ss),w(ww),c(cc),h(ss/2),hs(ss/4),ps(ss/8),n(nn),w0(load<int8_t>(f+".w0")),w1(load<int8_t>(f+".w1")),w2(load<int8_t>(f+".w2")),t0(load<int32_t>(f+".t0")),t1(load<int32_t>(f+".t1")),t2(load<int32_t>(f+".t2")),a(h*h*w),b(ps*ps*c),features(n),z(hs*hs*c){
  need(n==ps*ps*8&&w==32&&c==512&&w0.size()==size_t(27*w)&&w1.size()==size_t(9*w*c)&&w2.size()==size_t(c*8),"bounded convolution shapes");
  for(auto v:w0)need(v>=-3&&v<=3,"HRC accumulation bound");for(auto v:w1)need(v>=-7&&v<=7,"CNP accumulation bound");for(auto v:w2)need(v>=-1&&v<=1,"projection bound");
 }
 std::vector<uint32_t> forward(const uint8_t*x,bool sparse,bool parallel){
  #pragma omp parallel for if(parallel) schedule(static)
  for(int q=0;q<h*h*(w/8);q++){
   int j=(q%(w/8))*8,y=q/(w/8)/h,xx=q/(w/8)%h;int16x8_t acc=vdupq_n_s16(0);
   for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++){
    int iy=2*y+ky,ix=2*xx+kx;if(iy>=side||ix>=side)continue;
    for(int k=0;k<3;k++){int v=x[(iy*side+ix)*3+k];if(sparse&&!v)continue;acc=vmlaq_n_s16(acc,vmovl_s8(vld1_s8(&w0[((ky*3+kx)*3+k)*w+j])),int16_t(v));}
   }
   int16_t v[8];vst1q_s16(v,acc);for(int k=0;k<8;k++)a[(y*h+xx)*w+j+k]=v[k]>t0[j+k];
  }
  #pragma omp parallel for if(parallel) schedule(static)
  for(int q=0;q<hs*hs*(c/8);q++){
   int j=(q%(c/8))*8,y=q/(c/8)/hs,xx=q/(c/8)%hs;int16x8_t acc=vdupq_n_s16(0);
   for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++){
    int iy=2*y+ky,ix=2*xx+kx;if(iy>=h||ix>=h)continue;
    for(int k=0;k<w;k++){int v=a[(iy*h+ix)*w+k];if(sparse&&!v)continue;acc=vmlaq_n_s16(acc,vmovl_s8(vld1_s8(&w1[(((2-ky)*3+2-kx)*w+k)*c+j])),int16_t(v));}
   }
   vst1q_s16(&z[(y*hs+xx)*c+j],acc);
  }
  #pragma omp parallel for if(parallel) schedule(static)
  for(int q=0;q<ps*ps*c;q++){
   int j=q%c,y=q/c/ps,xx=q/c%ps;int v=-32768;
   for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++)v=std::max(v,int(z[((2*y+dy)*hs+2*xx+dx)*c+j]));b[q]=v>t1[j];
  }
  for(int y=0;y<ps;y++)for(int xx=0;xx<ps;xx++){
   int16x8_t acc=vdupq_n_s16(0);
   for(int k=0;k<c;k++){int v=b[(y*ps+xx)*c+k];if(sparse&&!v)continue;acc=vmlaq_n_s16(acc,vmovl_s8(vld1_s8(&w2[k*8])),int16_t(v));}
   int16_t v[8];vst1q_s16(v,acc);for(int j=0;j<8;j++)features[(j*ps+xx)*ps+y]=v[j]>t2[j];
  }
  std::vector<uint32_t>bits(n/32);for(int k=0;k<n;k++)bits[k/32]|=uint32_t(features[k])<<(k%32);return bits;
 }
};

#include "extent_blas.hpp"

int main(int argc,char**argv){
 std::vector<uint8_t>program;std::unique_ptr<LabDriver>driver;akida::HardwareDevicePtr dev;bool programmed=false;
 try{
  need(argc==7,"mode fixture output count learning-arm repeats");std::string mode=argv[1],f=argv[2],out=argv[3];int count=std::stoi(argv[4]),arm=std::stoi(argv[5]),repeats=std::stoi(argv[6]);
  bool native=mode=="native",parallel=mode=="cpu-parallel",sparse=mode!="cpu-dense",blas=mode=="cpu-blas1"||mode=="cpu-blas4";
  need((native||mode=="cpu-dense"||mode=="cpu-sparse"||parallel||blas)&&(count==8||count==40||count==64)&&(arm==0||arm==1)&&repeats>=1&&repeats<=5,"bounded study arguments");
  int side,w,c,n,u,nw,classes,available;{std::ifstream m(f+".meta");m>>side>>w>>c>>n>>u>>nw>>classes>>available;need(bool(m)&&(side==32||side==64||side==96||side==128)&&n==(side/8)*(side/8)*8&&u==128&&available==64,"fixture shape");}
  omp_set_num_threads(parallel?4:1);omp_set_dynamic(0);
  int input_bytes=side*side*3;Spatial front(f,side,w,c,n);auto x=load<uint8_t>(f+".inputs"),features=load<uint8_t>(f+".features");auto labels=load<int32_t>(f+".labels");auto initial=load<uint32_t>(f+".ram");
  std::unique_ptr<MatrixFront> matrix;if(blas)matrix=std::make_unique<MatrixFront>(front,mode=="cpu-blas4"?4:1);
  need(x.size()==64*input_bytes&&features.size()==64*n&&initial.size()==size_t(2+u*(4+n/32)),"fixture sizes");
  std::vector<int32_t>scores;std::vector<int64_t>latencies,callers,observers;std::vector<uint32_t>final;
  if(!native){
   for(int r=0;r<repeats;r++){
    CPU cpu(n,u,nw,classes,initial);std::ofstream snaps;if(r==0)snaps.open(out+".snapshots",std::ios::binary);
    for(int i=0;i<count;i++){
     kick_watchdog();bool learn=arm&&i>=8&&i<40;auto start=ns(),cp=process_ns();auto bits=blas?matrix->forward(x.data()+i*input_bytes):front.forward(x.data()+i*input_bytes,sparse,parallel);auto y=cpu.forward(bits,labels[i],learn);callers.push_back(process_ns()-cp);latencies.push_back(ns()-start);
     need(std::equal(front.features.begin(),front.features.end(),features.data()+i*n),"independent C++ CNN features");
     if(r==0){scores.insert(scores.end(),y.begin(),y.end());snaps.write((char*)cpu.ram.data(),cpu.ram.size()*4);}
    }
    if(r==repeats-1)final=cpu.ram;
   }
  }else{
   need(repeats==1,"one stream per physical session");driver=std::make_unique<LabDriver>();driver->dma_read_min=128;
   need(driver->akida_visible_memory()==0&&driver->akida_visible_memory_size()==0,"no external host DDR weight registration");
   dev=akida::HardwareDevice::create(driver.get());program=load<uint8_t>(f+".bin");auto info=dev->program(program.data(),program.size());programmed=true;
   need(info.number_of_passes()==1&&info.can_learn()&&info.input_is_dense()&&!info.activation_enabled(),"native resident graph contract");
   auto impl=static_cast<akida::HardwareDeviceImpl*>(dev.get());
   auto capture_fixed=[&](bool end){
    std::ifstream l(f+".layout");int count;l>>count;need(bool(l)&&count>=4&&count<=64,"fixed packet count");
    for(int j=0;j<count;j++){
     int col,row,id,words;l>>col>>row>>id>>words;auto source=load<uint32_t>(f+".packet"+std::to_string(j)+".ram");need(bool(l)&&source.size()==size_t(words),"fixed packet size");
     need((source[0]>>24)==uint32_t(col)&&((source[0]>>16)&255)==uint32_t(row)&&((source[0]>>8)&15)==uint32_t(1<<id),"fixed target identity");auto got=source;
     akida::dma::dma_config_read(got.data()+2,uint32_t(got.size()-2),got.data(),impl->dma_config(),impl->mem(),driver.get(),true);
     dump(out+".packet"+std::to_string(j)+(end?".final":".initial"),got);need(got==source,"fixed NPU configuration RAM readback differs");
    }
   };
   bool learning=false;dev->toggle_learn(false);
   auto capture=[&](){if(!learning)dev->toggle_learn(true);std::vector<uint32_t>r(dev->learn_mem_size());dev->learn_mem(r.data());if(!learning)dev->toggle_learn(false);return r;};
   need(capture()==initial,"initial local FNP3 RAM");dump(out+".initial.ram",initial);capture_fixed(false);
   need(dev->set_batch_size(1,false)==1,"one event queue");auto addr=dev->scratch_alloc(input_bytes);CPU cpu(n,u,nw,classes,initial);std::ofstream snaps(out+".snapshots",std::ios::binary);
   for(int i=0;i<count;i++){
    kick_watchdog();bool learn=arm&&i>=8&&i<40;if(learning!=learn){dev->toggle_learn(learn);learning=learn;}
    auto start=ns(),cp=process_ns();driver->write(addr,x.data()+i*input_bytes,input_bytes);need(dev->enqueue(addr,learn?&labels[i]:nullptr),"native raw image enqueue");
    akida::TensorUniquePtr y;auto deadline=Clock::now()+std::chrono::seconds(2);while(!(y=dev->fetch()))need(Clock::now()<deadline,"owned completion deadline");
    need(y->type()==akida::TensorType::int32&&y->size()==128,"returned potential shape");std::vector<int32_t>got(y->data<int32_t>(),y->data<int32_t>()+128);callers.push_back(process_ns()-cp);latencies.push_back(ns()-start);
    scores.insert(scores.end(),got.begin(),got.end());dump(out+".scores",scores);
    auto observe=ns();auto bits=front.forward(x.data()+i*input_bytes,true,false);need(std::equal(front.features.begin(),front.features.end(),features.data()+i*n),"CPU independent native front contract");auto expected=cpu.forward(bits,labels[i],learn);
    auto r=capture();snaps.write((char*)r.data(),r.size()*4);snaps.flush();dump(out+".ram",r);
    need(got==expected,"native potential mismatch");need(r==cpu.ram,"native complete learning RAM mismatch");observers.push_back(ns()-observe);
   }
   final=capture();capture_fixed(true);dev->scratch_free(addr);dev->unprogram();programmed=false;
  }
  dump(out+".scores",scores);dump(out+".ram",final);dump(out+".latency_ns",latencies);dump(out+".caller_ns",callers);dump(out+".observer_ns",observers);
  std::cout<<"{\"status\":\"PASS\",\"mode\":\""<<mode<<"\",\"samples\":"<<count<<",\"learning_arm\":"<<arm<<",\"cpu_threads\":"<<((parallel||mode=="cpu-blas4")?4:1)<<",\"repeats\":"<<repeats<<",\"physical_device_opened\":"<<(native?"true":"false")<<",\"registered_host_weight_memory_bytes\":0,\"cleanup\":\"complete\"}\n";
 }catch(const std::exception&e){if(programmed&&dev)try{dev->unprogram();}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
