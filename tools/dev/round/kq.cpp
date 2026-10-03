#ifndef HDR
#define HDR "../../../fyx_sort.hpp"
#endif
#include HDR
#include "../../../bench/suite/core/bench_core.hpp"
#include <cstdio>
using namespace fb;
int main(int c,char**v){ size_t n=atol(v[1]);
  for(int d=0; d<(int)Dist::Count; ++d){ auto base=make_input<KV>(n,(Dist)d,42); double b[2]={1e9,1e9}; bool okk=true;
    for(int r=0;r<11;r++){
      auto x=base; auto t0=Clock::now(); struct F{bool operator()(size_t,size_t){return true;} bool verify(size_t,size_t,bool){return true;} void seam(size_t){}} fn; bool ok=fyx::detail::kv16_vqsort(x.data(),n,true,false,fn); auto t1=Clock::now(); b[0]=std::min(b[0],seconds_between(t0,t1));
      if(!ok||!std::is_sorted(x.begin(),x.end(),KVLess{})) okk=false;
      if(r==0){ uint64_t s1=0,s2=0; for(auto&e:x) s1+=e.val*0x9E3779B97F4A7C15ull^e.key; for(auto&e:base) s2+=e.val*0x9E3779B97F4A7C15ull^e.key; if(s1!=s2) okk=false; }
      auto y=base; fyx::Options o; o.parallel=fyx::Tri::Off; t0=Clock::now(); fyx::sort(y.data(),n,KVLess{},o); t1=Clock::now(); b[1]=std::min(b[1],seconds_between(t0,t1));
    }
    printf("%-14s kvq %7.3f fyx %7.3f %s\n",dist_name((Dist)d),b[0]*1e3,b[1]*1e3,okk?"":"WRONG"); } }
