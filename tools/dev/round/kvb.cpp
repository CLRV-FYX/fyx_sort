#ifndef HDR
#define HDR "../../../fyx_sort.hpp"
#endif
#include HDR
#include "../../../bench/suite/core/bench_core.hpp"
#include <cstdio>
#include <algorithm>
#ifndef LINEX
#define LINEX 0
#endif
using namespace fb;
template<class T> void run(size_t n,int reps,int only){
  for(int d=0; d<(int)Dist::Count; ++d){ if(only>=0&&d!=only) continue;
    auto base=make_input<T>(n,(Dist)d,42);
    double bf=1e9,bs=1e9;
    for(int r=0;r<reps;++r){
      auto v=base; auto t0=Clock::now();
      fyx::Options o; o.parallel=fyx::Tri::Off;
      if constexpr (std::is_same_v<T,KV>) fyx::sort(v.data(),n,Less<T>{},o); else fyx::sort(v.data(),n,o);
      auto t1=Clock::now(); bf=std::min(bf,seconds_between(t0,t1));
      auto w=base; t0=Clock::now(); std::sort(w.begin(),w.end(),Less<T>{}); t1=Clock::now(); bs=std::min(bs,seconds_between(t0,t1));
      for(size_t i=0;i<n;i++){ if(Less<T>{}(v[i],w[i])||Less<T>{}(w[i],v[i])){printf("MISMATCH %s\n",dist_name((Dist)d));break;} }
    }
    printf("%-14s fyx %8.3fms std %8.3fms line %d\n",dist_name((Dist)d),bf*1e3,bs*1e3,(int)(LINEX));
  }
}
int main(int c,char**v){ size_t n=atol(v[1]); std::string k=v[2]; int reps=atoi(v[3]); int only=c>4?atoi(v[4]):-1;
  if(k=="kv") run<KV>(n,reps,only); else run<std::string>(n,reps,only); }
