// ab: bench_core distributions, new header vs ab_old.o (namespace fyx_old), interleaved in one process.
// usage: ./ab N dist...
#include "fyx_sort.hpp"
#include "core/bench_core.hpp"
#include <cstdio>
void old_sort(std::int32_t*, std::size_t); void old_sort(std::uint32_t*, std::size_t); void old_sort(std::int64_t*, std::size_t);
void old_sort(std::uint64_t*, std::size_t); void old_sort(float*, std::size_t); void old_sort(double*, std::size_t);
using namespace fb;
template<class T> void go(const char* nm,size_t n,int d){ auto in=make_input<T>(n,(Dist)d,20261002); size_t batch=std::max<size_t>(1,(1u<<18)/n); std::vector<T> w(n*batch); double b[2]={1e9,1e9};
  for(int r=0;r<(n>=1000000?9:21);r++) for(int m=0;m<2;m++){ for(size_t k=0;k<batch;k++) std::copy(in.begin(),in.end(),w.begin()+k*n); auto t0=Clock::now();
    for(size_t k=0;k<batch;k++){ if(m==0) old_sort(w.data()+k*n,n); else { fyx::Options o; o.parallel=fyx::Tri::Off; fyx::sort(w.data()+k*n,n,o);} } b[m]=std::min(b[m],seconds_between(t0,Clock::now())/batch); }
  printf("%s n=%zu %-12s old %9.1f new %9.1f new/old %.2f\n",nm,n,dist_name((Dist)d),b[0]*1e6,b[1]*1e6,b[1]/b[0]); fflush(stdout); }
int main(int c,char**v){ size_t n=atol(v[1]); for(int i=2;i<c;i++){int d=atoi(v[i]); go<int32_t>("i32",n,d); go<uint32_t>("u32",n,d); go<float>("f32",n,d); go<int64_t>("i64",n,d); go<uint64_t>("u64",n,d); go<double>("f64",n,d);} }
