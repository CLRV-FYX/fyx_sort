// op: organ-pipe breakdown (prescan alone vs full serial sort), batch-copied.
#define FYX_ENABLE_TEST_HOOKS 1
#include "fyx_sort.hpp"
#include "core/bench_core.hpp"
#include <cstdio>
using namespace fb;
template<class T> void go(const char* nm,size_t n,int d){
  auto in=make_input<T>(n,(Dist)d,20261002); size_t batch=std::max<size_t>(1,(1u<<18)/n);
  std::vector<T> w(n*batch); double b[2]={1e9,1e9}; volatile unsigned sink=0; int disp=0; size_t st=0;
  for(int r=0;r<15;r++) for(int m=0;m<2;m++){
    for(size_t k=0;k<batch;k++) std::copy(in.begin(),in.end(),w.begin()+k*n);
    auto t0=Clock::now();
    for(size_t k=0;k<batch;k++){ T*p=w.data()+k*n; if(m==0){ unsigned pr; sink=sink+fyx::detail::vqsort_small_prescan(p,n,&st,&pr);} else { fyx::Options o; o.parallel=fyx::Tri::Off; fyx::sort(p,n,o); disp=(int)fyx::detail::test_last_dispatch();} }
    b[m]=std::min(b[m],seconds_between(t0,Clock::now())/batch);
  }
  std::printf("%s n=%zu %-12s prescan %7.1f (settled %zu) sort %7.1f disp %d\n",nm,n,dist_name((Dist)d),b[0]*1e6,st,b[1]*1e6,disp);
}
int main(int c,char**v){ size_t n=atol(v[1]); for(int i=2;i<c;i++){int d=atoi(v[i]); go<int64_t>("i64",n,d); go<uint64_t>("u64",n,d); go<double>("f64",n,d); go<float>("f32",n,d);} }
