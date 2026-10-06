// fs: far_swaps route costs (each candidate on a fresh copy).
#include "fyx_sort.hpp"
#include "core/bench_core.hpp"
#include <cstdio>
using namespace fb; namespace fd=fyx::detail;
template<class T> void run(size_t n){ auto in=make_input<T>(n,Dist::FarSwaps,20261002); std::vector<T> w;
  auto tm=[&](const char* nm, auto f){ double b=1e9; int ok=0; for(int r=0;r<7;r++){ w=in; auto t0=Clock::now(); ok=f(w.data()); b=std::min(b,seconds_between(t0,Clock::now())); } printf("  %s %.1f(%d)",nm,b*1e6,ok); };
  printf("%s n=%zu",type_name<T>(),n);
  std::uint32_t pos[64]; size_t D=fd::descent_positions(in.data(),1,n,false,pos,64); printf(" D=%zu",D);
  tm("fewruns",[&](T* p){ return (int)fd::try_few_runs_merge(p,n,false,1);});
  tm("proof",[&](T* p){ return (int)fd::try_proof_structured_sort(p,n,false);});
  tm("sparse",[&](T* p){ return (int)fd::try_sparse_outlier_repair(p,n,1,false);});
  printf(" local=%d headinv=%d",(int)fd::descents_look_local(in.data(),n,pos[0],false,96,4),(int)fd::head_inversions_within(in.data(),n,false,256,8));
  tm("binsert",[&](T* p){ return (int)fd::budgeted_insertion_repair(p,n,false,n/4+64);});
  tm("sort",[&](T* p){ fyx::Options o; o.parallel=fyx::Tri::Off; fyx::sort(p,p+n,o); return 1;});
  printf("\n"); }
int main(int c,char**v){ size_t n=atol(v[1]); run<int32_t>(n); run<uint32_t>(n); run<int64_t>(n); run<uint64_t>(n); run<float>(n); run<double>(n); }
