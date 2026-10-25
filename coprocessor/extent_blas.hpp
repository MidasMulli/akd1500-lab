// Strong CPU comparator: explicit installed OpenBLAS ILP64, CPU only.
// CBLAS ABI: OpenBLAS v0.3.27 cblas.h. Integer-valued FP32 sums stay exact
// within 2^24; every feature and learning-memory word is independently checked.
#include <dlfcn.h>
#include <cstdlib>
struct MatrixFront {
 Spatial& f;void* lib;
 using SGEMM=void(*)(int,int,int,int64_t,int64_t,int64_t,float,const float*,int64_t,const float*,int64_t,float,float*,int64_t);
 SGEMM gemm;std::vector<float>w0,w1,w2,i0,i1,i2,z0,z1,z2;
 MatrixFront(Spatial& a,int threads):f(a),w0(27*f.w),w1(9*f.w*f.c),w2(f.c*8),i0(f.h*f.h*27),i1(f.hs*f.hs*9*f.w),i2(f.ps*f.ps*f.c),z0(f.h*f.h*f.w),z1(f.hs*f.hs*f.c),z2(f.ps*f.ps*8){
  const char* path=std::getenv("AKD_OPENBLAS_LIBRARY");need(path&&*path,"explicit CPU OpenBLAS library path");
  lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);need(lib!=nullptr,"OpenBLAS load failed");
  gemm=(SGEMM)dlsym(lib,"scipy_cblas_sgemm64_");
  auto config=(char*(*)())dlsym(lib,"scipy_openblas_get_config64_");auto set=(void(*)(int))dlsym(lib,"scipy_openblas_set_num_threads64_");auto get=(int(*)())dlsym(lib,"scipy_openblas_get_num_threads64_");
  need(gemm&&config&&set&&get,"explicit ILP64 OpenBLAS symbols");need(std::string(config()).find("USE64BITINT")!=std::string::npos,"ILP64 required");set(threads);need(get()==threads,"BLAS thread count");
  std::cerr<<config()<<" threads="<<get()<<"\n";
  for(size_t k=0;k<w0.size();k++)w0[k]=f.w0[k];for(size_t k=0;k<w2.size();k++)w2[k]=f.w2[k];
  for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++)for(int k=0;k<f.w;k++)for(int j=0;j<f.c;j++)w1[((ky*3+kx)*f.w+k)*f.c+j]=f.w1[(((2-ky)*3+2-kx)*f.w+k)*f.c+j];
 }
 ~MatrixFront(){if(lib)dlclose(lib);}
 void mul(int m,int n,int k,const std::vector<float>&a,const std::vector<float>&b,std::vector<float>&c){gemm(101,111,111,m,n,k,1,a.data(),k,b.data(),n,0,c.data(),n);}
 std::vector<uint32_t> forward(const uint8_t*x){
  for(int y=0;y<f.h;y++)for(int xx=0;xx<f.h;xx++)for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++)for(int k=0;k<3;k++){
   int iy=2*y+ky,ix=2*xx+kx;i0[(y*f.h+xx)*27+(ky*3+kx)*3+k]=iy<f.side&&ix<f.side?x[(iy*f.side+ix)*3+k]:0;
  }
  mul(f.h*f.h,f.w,27,i0,w0,z0);for(int q=0;q<f.h*f.h*f.w;q++)f.a[q]=z0[q]>f.t0[q%f.w];
  for(int y=0;y<f.hs;y++)for(int xx=0;xx<f.hs;xx++)for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++)for(int k=0;k<f.w;k++){
   int iy=2*y+ky,ix=2*xx+kx;i1[(y*f.hs+xx)*9*f.w+(ky*3+kx)*f.w+k]=iy<f.h&&ix<f.h?f.a[(iy*f.h+ix)*f.w+k]:0;
  }
  mul(f.hs*f.hs,f.c,9*f.w,i1,w1,z1);
  for(int y=0;y<f.ps;y++)for(int xx=0;xx<f.ps;xx++)for(int k=0;k<f.c;k++){
   float v=-32768;for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++)v=std::max(v,z1[((2*y+dy)*f.hs+2*xx+dx)*f.c+k]);i2[(y*f.ps+xx)*f.c+k]=v>f.t1[k];
  }
  mul(f.ps*f.ps,8,f.c,i2,w2,z2);
  for(int y=0;y<f.ps;y++)for(int xx=0;xx<f.ps;xx++)for(int k=0;k<8;k++)f.features[(k*f.ps+xx)*f.ps+y]=z2[(y*f.ps+xx)*8+k]>f.t2[k];
  std::vector<uint32_t> bits(f.n/32);for(int k=0;k<f.n;k++)bits[k/32]|=uint32_t(f.features[k])<<(k%32);return bits;
 }
};
