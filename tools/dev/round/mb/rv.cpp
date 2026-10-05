// rv: in-place reverse kernels on uint64, batch-copied (cold-ish like the suite).
#include "fyx_sort.hpp"
#include "core/bench_core.hpp"
#include <cstdio>
#include <immintrin.h>
using namespace fb;
__attribute__((noinline)) void r_std(uint64_t* p,size_t n){ std::reverse(p,p+n); }
__attribute__((noinline,target("avx2"))) void r_ymm(uint64_t* p,size_t n){ size_t l=0,r=n; while(r-l>=8){ __m256i a=_mm256_loadu_si256((__m256i*)(p+l)), b=_mm256_loadu_si256((__m256i*)(p+r-4));
  _mm256_storeu_si256((__m256i*)(p+l),_mm256_permute4x64_epi64(b,0x1B)); _mm256_storeu_si256((__m256i*)(p+r-4),_mm256_permute4x64_epi64(a,0x1B)); l+=4; r-=4;} std::reverse(p+l,p+r); }
__attribute__((noinline,target("avx2"))) void r_ymm4(uint64_t* p,size_t n){ size_t l=0,r=n; while(r-l>=32){ __m256i a[4],b[4]; for(int k=0;k<4;k++){a[k]=_mm256_loadu_si256((__m256i*)(p+l+4*k)); b[k]=_mm256_loadu_si256((__m256i*)(p+r-4-4*k));}
  for(int k=0;k<4;k++){ _mm256_storeu_si256((__m256i*)(p+l+4*k),_mm256_permute4x64_epi64(b[k],0x1B)); _mm256_storeu_si256((__m256i*)(p+r-4-4*k),_mm256_permute4x64_epi64(a[k],0x1B));} l+=16; r-=16;} std::reverse(p+l,p+r); }
// four streams: pairs split at quarter points (l from 0 and n/4, r from n and 3n/4)
__attribute__((noinline,target("avx2"))) void r_4s(uint64_t* p,size_t n){ size_t h=n/2/32*32/2; // per stream pair count (multiple of 16)
  for(size_t i=0;i<h;i+=8){ for(int s=0;s<2;s++){ size_t l=s*h+i, r=n-s*h-i; for(int k=0;k<2;k++){ __m256i a=_mm256_loadu_si256((__m256i*)(p+l+4*k)), b=_mm256_loadu_si256((__m256i*)(p+r-4-4*k));
   _mm256_storeu_si256((__m256i*)(p+l+4*k),_mm256_permute4x64_epi64(b,0x1B)); _mm256_storeu_si256((__m256i*)(p+r-4-4*k),_mm256_permute4x64_epi64(a,0x1B));}}}
  std::reverse(p+2*h,p+n-2*h); }
int main(int c,char**v){ size_t n=atol(v[1]); auto in=make_input<uint64_t>(n,Dist::Reverse,20261002); size_t batch=std::max<size_t>(1,(1u<<18)/n); std::vector<uint64_t> w(n*batch);
 void(*F[4])(uint64_t*,size_t)={r_std,r_ymm,r_ymm4,r_4s}; const char* nm[4]={"std","ymm","ymm4","4s"}; double b[4]={1e9,1e9,1e9,1e9};
 for(int r=0;r<15;r++) for(int m=0;m<4;m++){ for(size_t k=0;k<batch;k++) std::copy(in.begin(),in.end(),w.begin()+k*n); auto t0=Clock::now(); for(size_t k=0;k<batch;k++) F[m](w.data()+k*n,n); b[m]=std::min(b[m],seconds_between(t0,Clock::now())/batch);
   for(size_t k=1;k<n;k++) if(w[k]<w[k-1]){printf("BAD %s\n",nm[m]);break;} }
 for(int m=0;m<4;m++) printf("%s %.1f  ",nm[m],b[m]*1e6); printf("\n"); }
