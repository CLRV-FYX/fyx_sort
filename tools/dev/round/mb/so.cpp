// so: fyx (serial) vs crumsort / fluxsort, batch-copied like the suite.
// usage: ./so N dist...   (dist numbers as in bench_core Dist)
#include "fyx_sort.hpp"
#include "core/bench_core.hpp"
#include <cstdio>
extern "C" {
void crum_i32(void*,size_t); void crum_u32(void*,size_t); void crum_i64(void*,size_t); void crum_u64(void*,size_t); void crum_f32(void*,size_t); void crum_f64(void*,size_t);
void flux_i32(void*,size_t); void flux_u32(void*,size_t); void flux_i64(void*,size_t); void flux_u64(void*,size_t); void flux_f32(void*,size_t); void flux_f64(void*,size_t);
}
using namespace fb;
template<class T> void sc(int m,T* p,size_t n){
  typedef void(*F)(void*,size_t);
  F c,f;
  if(std::is_same<T,int32_t>::value){c=crum_i32;f=flux_i32;} else if(std::is_same<T,uint32_t>::value){c=crum_u32;f=flux_u32;}
  else if(std::is_same<T,int64_t>::value){c=crum_i64;f=flux_i64;} else if(std::is_same<T,uint64_t>::value){c=crum_u64;f=flux_u64;}
  else if(std::is_same<T,float>::value){c=crum_f32;f=flux_f32;} else {c=crum_f64;f=flux_f64;}
  (m==0?c:f)(p,n);
}
template<class T> void go(const char* nm,size_t n,int d){
  auto in=make_input<T>(n,(Dist)d,20261002); size_t batch=std::max<size_t>(1,(1u<<18)/n);
  std::vector<T> w(n*batch); double b[3]={1e9,1e9,1e9};
  int R = n>=1000000?7:15;
  for(int r=0;r<R;r++) for(int m=0;m<3;m++){
    for(size_t k=0;k<batch;k++) std::copy(in.begin(),in.end(),w.begin()+k*n);
    auto t0=Clock::now();
    for(size_t k=0;k<batch;k++){ T*p=w.data()+k*n; if(m<2) sc<T>(m,p,n); else { fyx::Options o; o.parallel=fyx::Tri::Off; fyx::sort(p,n,o);} }
    b[m]=std::min(b[m],seconds_between(t0,Clock::now())/batch);
    for(size_t k=1;k<n;k++) if(w[k]<w[k-1]){ std::printf("UNSORTED m=%d\n",m); break; }
  }
  std::printf("%s n=%zu %-12s crum %8.1f flux %8.1f fyx %8.1f ratio %.3f\n",nm,n,dist_name((Dist)d),b[0]*1e6,b[1]*1e6,b[2]*1e6,std::min(b[0],b[1])/b[2]);
  std::fflush(stdout);
}
int main(int c,char**v){ size_t n=atol(v[1]);
  for(int i=2;i<c;i++){ int d=atoi(v[i]); go<int32_t>("i32",n,d); go<uint32_t>("u32",n,d); go<float>("f32",n,d); go<int64_t>("i64",n,d); go<uint64_t>("u64",n,d); go<double>("f64",n,d);} }
