// vqh: hwy::VQSort (K64V64) vs fyx serial on kv16 bench inputs, min of 9, interleaved.
// build: g++ -std=c++17 -O3 -march=native -pthread -I$HWY vqh.cpp $HWY_OBJS -o vqh
#include "hwy/contrib/sort/vqsort.h"
#include "../../../fyx_sort.hpp"
#include "../../../bench/suite/core/bench_core.hpp"
#include <cstdio>
using namespace fb;
int main(int c,char**v){ size_t n=atol(v[1]); int only=c>2?atoi(v[2]):-1;
  for(int d=0; d<(int)Dist::Count; ++d){ if(only>=0&&d!=only) continue; auto base=make_input<KV>(n,(Dist)d,42); double b[2]={1e9,1e9};
    for(int r=0;r<9;r++) for(int m=0;m<2;m++){ auto x=base; auto t0=Clock::now();
      if(m==0) hwy::VQSort(reinterpret_cast<hwy::K64V64*>(x.data()), n, hwy::SortAscending());
      else { fyx::Options o; o.parallel=fyx::Tri::Off; fyx::sort(x.data(),n,KVLess{},o); }
      b[m]=std::min(b[m],seconds_between(t0,Clock::now())); if(!std::is_sorted(x.begin(),x.end(),KVLess{})) printf("BAD m%d\n",m); }
    printf("%-14s vq %8.3f fyx %8.3f  ratio %.3f\n",dist_name((Dist)d),b[0]*1e3,b[1]*1e3,b[0]/b[1]); } }
