// kq: kv16 all-equivalent proof variants (16-byte records, comparator on the u64 key), batch-copied.
#include "fyx_sort.hpp"
#include "core/bench_core.hpp"
#include "hwy/contrib/sort/vqsort.h"
#include <cstdio>
using namespace fb;
template<class Comp> __attribute__((noinline)) bool v_fwd4(const KV* p,size_t n,Comp comp){ const KV ref=p[0]; size_t i=1;
  for(;i+2048<=n;i+=2048){ unsigned b0=0,b1=0,b2=0,b3=0; for(size_t k=0;k<2048;k+=4){ b0|=comp(p[i+k],ref)|comp(ref,p[i+k]); b1|=comp(p[i+k+1],ref)|comp(ref,p[i+k+1]); b2|=comp(p[i+k+2],ref)|comp(ref,p[i+k+2]); b3|=comp(p[i+k+3],ref)|comp(ref,p[i+k+3]); } if(b0|b1|b2|b3) return false; }
  unsigned b=0; for(;i<n;i++) b|=comp(p[i],ref)|comp(ref,p[i]); return !b; }
template<class Comp> __attribute__((noinline)) bool v_4s(const KV* p,size_t n,Comp comp){ const KV ref=p[0]; size_t q=(n-1)/4/512*512;
  for(size_t i=0;i<q;i+=512){ unsigned b=0; for(int t=0;t<4;t++){ const KV* s=p+1+t*q+i; for(size_t k=0;k<512;k++) b|=comp(s[k],ref)|comp(ref,s[k]); } if(b) return false; }
  unsigned b=0; for(size_t i=1+4*q;i<n;i++) b|=comp(p[i],ref)|comp(ref,p[i]); return !b; }
int main(int c,char**v){ size_t n=atol(v[1]); auto in=make_input<KV>(n,Dist::AllEqual,20261002); size_t batch=std::max<size_t>(1,(1u<<18)/n); std::vector<KV> w(n*batch);
  const char* M[]={"hwy","fyxsort","fyx_equiv","fwd4","4s"}; double b[5]={1e9,1e9,1e9,1e9,1e9}; volatile int sink=0; KVLess comp;
  for(int r=0;r<(n>=1000000?11:31);r++) for(int m=0;m<5;m++){ for(size_t k=0;k<batch;k++) std::copy(in.begin(),in.end(),w.begin()+k*n); auto t0=Clock::now();
    for(size_t k=0;k<batch;k++){ KV* p=w.data()+k*n;
      if(m==0) hwy::VQSort(reinterpret_cast<hwy::K64V64*>(p),n,hwy::SortAscending());
      else if(m==1){ fyx::Options o; o.parallel=fyx::Tri::Off; fyx::sort(p,n,comp,o);} else if(m==2) sink=sink+fyx::detail::kv16_all_equiv_avx512(p,n,comp);
      else if(m==3) sink=sink+v_fwd4(p,n,comp); else sink=sink+v_4s(p,n,comp); }
    b[m]=std::min(b[m],seconds_between(t0,Clock::now())/batch); }
  printf("kv16 n=%zu",n); for(int m=0;m<5;m++) printf("  %s %.1f",M[m],b[m]*1e6); printf("\n"); }
