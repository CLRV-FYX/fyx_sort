#include "../fyx_sort.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
using namespace fyx::detail;
template<class Ops, class Fn> bool test_ops(const char* name, std::mt19937_64& rng, Fn sortfn){
  using Key = typename Ops::Key;
  for(std::size_t n=0;n<=64;++n){
    for(int t=0;t<300;++t){
      std::vector<Key> a(n);
      for(std::size_t i=0;i<n;i++){
        int m=t%4; Key v;
        if(m==0) v=(Key)rng();
        else if(m==1) v=(Key)(rng()%3);
        else if(m==2) v=(Key)((rng()&1)?std::numeric_limits<Key>::max():0);
        else v=(Key)(rng()%256);
        a[i]=v;
      }
      std::vector<Key> b=a;
      sortfn(a.data(), n);
      std::sort(b.begin(),b.end());
      if(a!=b){ printf("%s MISMATCH n=%zu t=%d\n",name,n,t); return false; }
    }
  }
  printf("  %-16s OK (n=0..64)\n",name);
  return true;
}
int main(){
  std::mt19937_64 rng(999);
  bool ok=true;
  printf("avx512=%d avx2=%d sse42=%d\n",(int)use_avx512(),(int)use_avx2(),(int)use_sse42());
#if FYX_HAS_AVX512_CODE
  if(use_avx512()){
    ok&=test_ops<isa_avx512::Avx512Ops32>("Avx512Ops32",rng,[](uint32_t*p,size_t n){isa_avx512::network_sort_keys<isa_avx512::Avx512Ops32>(p,n);});
    ok&=test_ops<isa_avx512::Avx512Ops64>("Avx512Ops64",rng,[](uint64_t*p,size_t n){isa_avx512::network_sort_keys<isa_avx512::Avx512Ops64>(p,n);});
  }
#endif
#if FYX_HAS_AVX2_CODE
  if(use_avx2()){
    ok&=test_ops<isa_avx2::Avx2Ops32>("Avx2Ops32",rng,[](uint32_t*p,size_t n){isa_avx2::network_sort_keys<isa_avx2::Avx2Ops32>(p,n);});
    ok&=test_ops<isa_avx2::Avx2Ops64>("Avx2Ops64",rng,[](uint64_t*p,size_t n){isa_avx2::network_sort_keys<isa_avx2::Avx2Ops64>(p,n);});
  }
#endif
#if FYX_HAS_SSE42_CODE
  if(use_sse42()){
    ok&=test_ops<isa_sse42::Sse42Ops32>("Sse42Ops32",rng,[](uint32_t*p,size_t n){isa_sse42::network_sort_keys<isa_sse42::Sse42Ops32>(p,n);});
    ok&=test_ops<isa_sse42::Sse42Ops64>("Sse42Ops64",rng,[](uint64_t*p,size_t n){isa_sse42::network_sort_keys<isa_sse42::Sse42Ops64>(p,n);});
  }
#endif
  ok&=test_ops<ScalarOps<uint32_t>>("ScalarOps32",rng,[](uint32_t*p,size_t n){network_sort_keys<ScalarOps<uint32_t>>(p,n);});
  ok&=test_ops<ScalarOps<uint64_t>>("ScalarOps64",rng,[](uint64_t*p,size_t n){network_sort_keys<ScalarOps<uint64_t>>(p,n);});
  printf(ok?"ALL NETWORK TESTS PASS\n":"FAILURES\n");
  return ok?0:1;
}
