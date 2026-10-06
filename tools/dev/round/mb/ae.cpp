// ae: all-equal inputs, batch-copied like the suite: hwy vqsort vs fyx and its scan kernels.
// build: g++ ... -I<repo> -I<repo>/bench/suite -I/tmp/w/tp/highway ae.cpp /tmp/w/hobj/*.o
#include "fyx_sort.hpp"
#include "core/bench_core.hpp"
#include "hwy/contrib/sort/vqsort.h"
#include <cstdio>
using namespace fb;
// candidate: ymm, ternlog x |= v ^ b, one accumulator per stream, NS streams
template<class T,int NS,int SBV> __attribute__((noinline,target("avx512f,avx512vl"))) bool cand(const T* p,size_t n){
  constexpr size_t L=32/sizeof(T), SB=SBV*L; __m256i b = sizeof(T)==8? _mm256_set1_epi64x(*(const long long*)p) : _mm256_set1_epi32(*(const int*)p);
  { __m256i x=_mm256_xor_si256(_mm256_loadu_si256((const __m256i*)p),b); if(!_mm256_testz_si256(x,x)) return false; }
  const size_t q=n/NS/SB*SB; __m256i acc[NS]; for(int t=0;t<NS;t++) acc[t]=_mm256_setzero_si256();
  for(size_t i=0;i<q;i+=SB){ for(int t=0;t<NS;t++) for(size_t k=0;k<SB;k+=L) acc[t]=_mm256_ternarylogic_epi32(acc[t],_mm256_loadu_si256((const __m256i*)(p+t*q+i+k)),b,0xF6);
    __m256i x=acc[0]; for(int t=1;t<NS;t++) x=_mm256_or_si256(x,acc[t]); if(!_mm256_testz_si256(x,x)) return false; }
  __m256i x=_mm256_setzero_si256(); size_t i=NS*q; for(;i+L<=n;i+=L) x=_mm256_or_si256(x,_mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(p+i)),b));
  if(i<n) x=_mm256_or_si256(x,_mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(p+n-L)),b)); return _mm256_testz_si256(x,x); }
// aligned: scalar-free head (one unaligned vector), then 64B-aligned loads, 2 accumulators, check per 16 vectors
template<class T,bool Z> __attribute__((noinline,target("avx512f,avx512vl"))) bool alg(const T* p,size_t n){
  if constexpr (Z) {
    constexpr size_t L=64/sizeof(T); __m512i b = sizeof(T)==8? _mm512_set1_epi64(*(const long long*)p) : _mm512_set1_epi32(*(const int*)p);
    { __m512i x=_mm512_xor_si512(_mm512_loadu_si512(p),b); if(_mm512_test_epi64_mask(x,x)) return false; }
    size_t i=((reinterpret_cast<uintptr_t>(p)+64)&~uintptr_t(63))-reinterpret_cast<uintptr_t>(p); i/=sizeof(T);
    __m512i a0=_mm512_setzero_si512(),a1=a0;
    for(;i+16*L<=n;i+=16*L){ for(size_t k=0;k<16*L;k+=2*L){ a0=_mm512_ternarylogic_epi32(a0,_mm512_load_si512(p+i+k),b,0xF6); a1=_mm512_ternarylogic_epi32(a1,_mm512_load_si512(p+i+k+L),b,0xF6);}
      __m512i x=_mm512_or_si512(a0,a1); if(_mm512_test_epi64_mask(x,x)) return false; }
    __m512i x=_mm512_or_si512(a0,a1); for(;i+L<=n;i+=L) x=_mm512_ternarylogic_epi32(x,_mm512_load_si512(p+i),b,0xF6);
    if(i<n) x=_mm512_ternarylogic_epi32(x,_mm512_loadu_si512(p+n-L),b,0xF6); return !_mm512_test_epi64_mask(x,x);
  } else {
    constexpr size_t L=32/sizeof(T); __m256i b = sizeof(T)==8? _mm256_set1_epi64x(*(const long long*)p) : _mm256_set1_epi32(*(const int*)p);
    { __m256i x=_mm256_xor_si256(_mm256_loadu_si256((const __m256i*)p),b); if(!_mm256_testz_si256(x,x)) return false; }
    size_t i=((reinterpret_cast<uintptr_t>(p)+32)&~uintptr_t(31))-reinterpret_cast<uintptr_t>(p); i/=sizeof(T);
    __m256i a0=_mm256_setzero_si256(),a1=a0;
    for(;i+16*L<=n;i+=16*L){ for(size_t k=0;k<16*L;k+=2*L){ a0=_mm256_ternarylogic_epi32(a0,_mm256_load_si256((const __m256i*)(p+i+k)),b,0xF6); a1=_mm256_ternarylogic_epi32(a1,_mm256_load_si256((const __m256i*)(p+i+k+L)),b,0xF6);}
      __m256i x=_mm256_or_si256(a0,a1); if(!_mm256_testz_si256(x,x)) return false; }
    __m256i x=_mm256_or_si256(a0,a1); for(;i+L<=n;i+=L) x=_mm256_ternarylogic_epi32(x,_mm256_load_si256((const __m256i*)(p+i)),b,0xF6);
    if(i<n) x=_mm256_ternarylogic_epi32(x,_mm256_loadu_si256((const __m256i*)(p+n-L)),b,0xF6); return _mm256_testz_si256(x,x);
  }
}
// single stream, 4 accumulators
template<class T,int SBV> __attribute__((noinline,target("avx512f,avx512vl"))) bool cand1(const T* p,size_t n){
  constexpr size_t L=32/sizeof(T); __m256i b = sizeof(T)==8? _mm256_set1_epi64x(*(const long long*)p) : _mm256_set1_epi32(*(const int*)p);
  __m256i a0=_mm256_setzero_si256(),a1=a0,a2=a0,a3=a0; size_t i=0;
  for(;i+SBV*L<=n;i+=SBV*L){ for(size_t k=0;k<SBV*L;k+=4*L){ a0=_mm256_ternarylogic_epi32(a0,_mm256_loadu_si256((const __m256i*)(p+i+k)),b,0xF6); a1=_mm256_ternarylogic_epi32(a1,_mm256_loadu_si256((const __m256i*)(p+i+k+L)),b,0xF6);
     a2=_mm256_ternarylogic_epi32(a2,_mm256_loadu_si256((const __m256i*)(p+i+k+2*L)),b,0xF6); a3=_mm256_ternarylogic_epi32(a3,_mm256_loadu_si256((const __m256i*)(p+i+k+3*L)),b,0xF6);}
    __m256i x=_mm256_or_si256(_mm256_or_si256(a0,a1),_mm256_or_si256(a2,a3)); if(!_mm256_testz_si256(x,x)) return false; }
  __m256i x=_mm256_or_si256(_mm256_or_si256(a0,a1),_mm256_or_si256(a2,a3)); for(;i+L<=n;i+=L) x=_mm256_or_si256(x,_mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(p+i)),b));
  if(i<n) x=_mm256_or_si256(x,_mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(p+n-L)),b)); return _mm256_testz_si256(x,x); }

template<class T> void go(const char* nm,size_t n){ auto in=make_input<T>(n,Dist::AllEqual,20261002); size_t batch=std::max<size_t>(1,(1u<<18)/n); std::vector<T> w(n*batch);
  const char* M[]={"hwy","fyx","vall","algZ","memcmp","c4s8","algZ","algY"}; double b[8]; for(auto&x:b)x=1e9; volatile int sink=0;
  for(int r=0;r<31;r++) for(int m=0;m<8;m++){ for(size_t k=0;k<batch;k++) std::copy(in.begin(),in.end(),w.begin()+k*n); auto t0=Clock::now();
    for(size_t k=0;k<batch;k++){ T* p=w.data()+k*n;
      if(m==0) hwy::VQSort(p,n,hwy::SortAscending()); else if(m==1){ fyx::Options o; o.parallel=fyx::Tri::Off; fyx::sort(p,n,o);}
      else if(m==2) sink=sink+fyx::detail::isa_avx512::vall_equal(p,n) ; else if(m==3) sink=sink+alg<T,true>(p,n);
      else if(m==4) sink=sink+(std::memcmp(p,p+1,(n-1)*sizeof(T))==0); else if(m==5) sink=sink+cand<T,4,8>(p,n); else if(m==6) sink=sink+alg<T,true>(p,n); else sink=sink+alg<T,false>(p,n); }
    b[m]=std::min(b[m],seconds_between(t0,Clock::now())/batch); }
  printf("%s n=%zu",nm,n); for(int m=0;m<8;m++) printf("  %s %.2f",M[m],b[m]*1e6); printf("\n"); }
int main(int c,char**v){ size_t n=atol(v[1]); go<int32_t>("i32",n); go<uint32_t>("u32",n); go<float>("f32",n); go<int64_t>("i64",n); go<uint64_t>("u64",n); go<double>("f64",n);}
