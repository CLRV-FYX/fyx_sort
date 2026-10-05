// nr: reverse-like inputs that are not reversed, new header vs old (A/B builds).
#include "fyx_sort.hpp"
#include "core/bench_core.hpp"
#include <cstdio>
#include <random>
using namespace fb;
template<class T> std::vector<T> mk(size_t n,int s){ std::vector<T> v(n); for(size_t j=0;j<n;j++) v[j]=(T)(n-j); std::mt19937_64 g(7);
  if(s==0) for(int k=0;k<8;k++) std::swap(v[g()%n],v[g()%n]);                 // rev + 8 far swaps
  if(s==1){ for(size_t j=0;j<n/2;j++) v[j]=(T)(n-2*j); for(size_t j=n/2;j<n;j++) v[j]=(T)(n-2*(j-n/2)+1); } // two interleaving descending halves
  if(s==2) std::sort(v.begin()+n*9/10,v.end());                                 // rev + ascending 10% tail
  if(s==3){ for(size_t j=0;j<n;j++) v[j]=(T)(j<n/2? 2*(n-j): 2*j); v[0]=(T)(4*n); } // V
  if(s==4) for(size_t j=0;j+1<n;j+=100) std::swap(v[j],v[j+1]);               // local perturbation
  if(s==5) std::swap(v[n/2],v[n/2+1]);                                          // single middle break
  return v; }
const char* SN[]={"rev+8swaps","2desc","rev+asc10%","V","rev+local","rev+midbrk"};
template<class T> void go(const char* nm,size_t n,int s){ auto in=mk<T>(n,s); size_t batch=std::max<size_t>(1,(1u<<18)/n); std::vector<T> w(n*batch); double b=1e9;
  for(int r=0;r<(n>=1000000?7:15);r++){ for(size_t k=0;k<batch;k++) std::copy(in.begin(),in.end(),w.begin()+k*n); auto t0=Clock::now();
    for(size_t k=0;k<batch;k++){ fyx::Options o; o.parallel=fyx::Tri::Off; fyx::sort(w.data()+k*n,n,o);} b=std::min(b,seconds_between(t0,Clock::now())/batch);
    for(size_t k=1;k<n;k++) if(w[k]<w[k-1]){printf("UNSORTED\n");break;} }
  printf("%s n=%zu %-11s %9.1f\n",nm,n,SN[s],b*1e6); }
int main(int c,char**v){ size_t n=atol(v[1]); for(int s=0;s<6;s++){ go<int32_t>("i32",n,s); go<uint64_t>("u64",n,s); go<double>("f64",n,s);} }
