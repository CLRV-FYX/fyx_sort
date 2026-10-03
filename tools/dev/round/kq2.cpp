#include "../../../fyx_sort.hpp"
#include "../../../bench/suite/core/bench_core.hpp"
#include <cstdio>
using namespace fb;
int main(int c,char**v){ size_t n=atol(v[1]);
  for(int d: {4,5,6,14,15}){ auto base=make_input<KV>(n,(Dist)d,42); double b[3]={1e9,1e9,1e9};
    for(int r=0;r<11;r++){
      auto x=base; auto t0=Clock::now(); struct F{bool operator()(size_t,size_t){return true;} bool verify(size_t,size_t,bool){return true;} void seam(size_t){}} fn; fyx::detail::kv16_vqsort(x.data(),n,true,false,fn); auto t1=Clock::now(); b[0]=std::min(b[0],seconds_between(t0,t1));
      x=base; t0=Clock::now(); fyx::detail::trivial_field_kv16_vqsort<uint64_t>(x.data(),n,KVLess{},8,false); t1=Clock::now(); b[1]=std::min(b[1],seconds_between(t0,t1));
      auto y=base; fyx::Options o; o.parallel=fyx::Tri::Off; t0=Clock::now(); fyx::sort(y.data(),n,KVLess{},o); t1=Clock::now(); b[2]=std::min(b[2],seconds_between(t0,t1));
    }
    printf("%-14s raw %7.3f chk %7.3f fyx %7.3f\n",dist_name((Dist)d),b[0]*1e3,b[1]*1e3,b[2]*1e3); } }
