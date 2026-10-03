#ifndef HDR
#define HDR "../../../fyx_sort.hpp"
#endif
#include HDR
#include "../../../bench/suite/core/bench_core.hpp"
#include <cstdio>
using namespace fb;
int main(int c,char**v){ size_t n=atol(v[1]); int only=c>2?atoi(v[2]):-1;
  for(int d=0; d<(int)Dist::Count; ++d){ if(only>=0&&d!=only) continue; auto base=make_input<KV>(n,(Dist)d,42); double b[2]={1e9,1e9};
    for(int r=0;r<5;r++) for(int m=0;m<2;m++){ auto x=base; fyx::Options o; o.parallel=m?fyx::Tri::On:fyx::Tri::Off; auto t0=Clock::now(); fyx::sort(x.data(),n,KVLess{},o); auto t1=Clock::now(); b[m]=std::min(b[m],seconds_between(t0,t1)); if(!std::is_sorted(x.begin(),x.end(),KVLess{})) printf("BAD\n"); }
    printf("%-14s ser %7.3f par %7.3f\n",dist_name((Dist)d),b[0]*1e3,b[1]*1e3); } }
