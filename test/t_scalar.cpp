#include "../fyx_sort.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
int main(){
  std::mt19937_64 rng(12345);
  for(unsigned N : {2u,4u,8u,16u,32u,64u}){
    for(int trial=0; trial<3000; ++trial){
      std::vector<uint32_t> a(N);
      for(auto&x : a) x = (trial%3==0)? (uint32_t)(rng()%4) : (uint32_t)rng();
      std::vector<uint32_t> b=a;
      fyx::detail::bitonic_scalar_n(a.data(), N);
      std::sort(b.begin(),b.end());
      if(a!=b){ printf("MISMATCH N=%u trial=%d\n",N,trial); return 1; }
    }
  }
  printf("scalar bitonic OK N=2..64\n");
  // heapsort + insertion sort
  for(int n=0;n<300;++n) for(int t=0;t<20;++t){
    std::vector<int> a(n); for(auto&x:a) x=(int)(rng()%50);
    std::vector<int> b=a,c=a;
    fyx::detail::heap_sort(a.begin(),a.end(),fyx::less{});
    fyx::detail::insertion_sort(c.begin(),c.end(),fyx::less{});
    std::sort(b.begin(),b.end());
    if(a!=b){printf("heap MISMATCH n=%d\n",n);return 1;}
    if(c!=b){printf("ins MISMATCH n=%d\n",n);return 1;}
  }
  printf("heap+insertion OK\n");
  return 0;
}
