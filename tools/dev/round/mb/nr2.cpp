// nr2: reverse-like non-reversed inputs, new vs old header interleaved in one process.
#include "fyx_sort.hpp"
#include "core/bench_core.hpp"
#include <cstdio>
#include <random>
void old_sort(std::int32_t*, std::size_t); void old_sort(std::uint32_t*, std::size_t); void old_sort(std::int64_t*, std::size_t);
void old_sort(std::uint64_t*, std::size_t); void old_sort(float*, std::size_t); void old_sort(double*, std::size_t);
using namespace fb;
template<class T> std::vector<T> mk(size_t n,int s){ std::vector<T> v(n); for(size_t j=0;j<n;j++) v[j]=(T)(n-j); std::mt19937_64 g(7);
  if(s==0) for(int k=0;k<8;k++) std::swap(v[g()%n],v[g()%n]);
  if(s==1){ for(size_t j=0;j<n/2;j++) v[j]=(T)(n-2*j); for(size_t j=n/2;j<n;j++) v[j]=(T)(n-2*(j-n/2)+1); }
  if(s==2) std::sort(v.begin()+n*9/10,v.end());
  if(s==3){ for(size_t j=0;j<n;j++) v[j]=(T)(j<n/2? 2*(n-j): 2*j); v[0]=(T)(4*n); }
  if(s==4) for(size_t j=0;j+1<n;j+=100) std::swap(v[j],v[j+1]);
  if(s==5) std::swap(v[n/2],v[n/2+1]);
  if(s==6) {}   // plain reverse
  return v; }
const char* SN[]={"rev+8swaps","2desc","rev+asc10%","V","rev+local","rev+midbrk","reverse"};
template<class T> void go(const char* nm,size_t n,int s){ auto in=mk<T>(n,s); size_t batch=std::max<size_t>(1,(1u<<18)/n); static size_t OFF = getenv("OFF") ? atol(getenv("OFF")) : 0; std::vector<T> wbuf(n*batch + 4096/sizeof(T)); T* wp = wbuf.data() + OFF/sizeof(T); struct { T* p; T* data(){return p;} T& operator[](size_t i){return p[i];} T* begin(){return p;} } w{wp}; double b[2]={1e9,1e9};
  for(int r=0;r<(n>=1000000?9:21);r++) for(int m=0;m<2;m++){ for(size_t k=0;k<batch;k++) std::copy(in.begin(),in.end(),w.begin()+k*n); auto t0=Clock::now();
    for(size_t k=0;k<batch;k++){ if(m==0) old_sort(w.data()+k*n,n); else { fyx::Options o; o.parallel=fyx::Tri::Off; fyx::sort(w.data()+k*n,n,o);} } b[m]=std::min(b[m],seconds_between(t0,Clock::now())/batch);
    for(size_t k=1;k<n;k++) if(w[k]<w[k-1]){printf("UNSORTED\n");break;} }
  printf("%s n=%zu %-11s old %9.1f new %9.1f new/old %.2f\n",nm,n,SN[s],b[0]*1e6,b[1]*1e6,b[1]/b[0]); fflush(stdout); }
int main(int c,char**v){ size_t n=atol(v[1]); for(int s=0;s<7;s++){ go<int32_t>("i32",n,s); go<uint64_t>("u64",n,s); go<double>("f64",n,s);} }
