// Small-range numeric dispatch (AVX-512 vector quicksort fast path below
// kVqsortMinN) and integer stable_sort delegation.  Every check compares with
// std::sort / std::stable_sort, so it is valid with or without AVX-512.
#include "../fyx_sort.hpp"
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cstring>
static long fails=0, checks=0;
template<class T, class C> void chk(std::vector<T> v, C c, const char* what, size_t n, int d){
  auto ref=v; std::sort(ref.begin(),ref.end(),c); fyx::sort(v.data(),v.size(),c); ++checks;
  bool ok = v.size()==ref.size();
  for(size_t i=0;ok&&i<v.size();i++) ok = !(c(v[i],ref[i])||c(ref[i],v[i]));
  if(!ok){ if(fails<10) printf("FAIL %s n=%zu d=%d\n",what,n,d); ++fails; }
}
template<class T> void go(const char* tn){ std::mt19937_64 g(42);
 for(size_t n=0;n<20000; n = n<300? n+1 : (size_t)(n*1.37)){
  for(int d=0; d<9; d++){ std::vector<T> v(n);
   for(size_t i=0;i<n;i++){ uint64_t r=g(); if constexpr(std::is_floating_point<T>::value) v[i]=(T)((int64_t)(r>>3))/(T)1e6; else { T x; std::memcpy(&x,&r,sizeof(T)); v[i]=x; } }
   if(d==1) { std::sort(v.begin(),v.end()); }
   if(d==2) { std::sort(v.rbegin(),v.rend()); }
   if(d==3) { for(auto&x:v) x=(T)(g()%5); }
   if(d==4 && n) { std::sort(v.begin(),v.end()); std::rotate(v.begin(), v.begin()+g()%n, v.end()); }
   if(d==5) { for(auto&x:v) x=(T)7; }
   if(d==6 && n) { std::sort(v.begin(),v.end()); std::reverse(v.begin()+n/2,v.end()); }
   if(d==8 && n > 70) { std::sort(v.begin(),v.end()); for(size_t k=0;k<n/1000+1;k++){ size_t i=g()%(n-66); std::swap(v[i], v[i+1+g()%64]); } }
   if constexpr(std::is_floating_point<T>::value){ if(d==7) for(auto&x:v) if(g()%3==0) x = (g()&1)? (T)0.0 : -(T)0.0; }
   chk(v,std::less<T>(),tn,n,d); chk(v,std::greater<T>(),tn,n,d); chk(v,fyx::less{},tn,n,d);
   if constexpr(std::is_integral<T>::value){ auto a=v, b=v; fyx::stable_sort(a.data(),a.size()); std::stable_sort(b.begin(),b.end()); ++checks; if(a!=b){ if(fails<10) printf("FAIL stable %s n=%zu d=%d\n",tn,n,d); ++fails; }
     auto c2=v, d2=v; fyx::stable_sort(c2.data(),c2.size(),std::greater<T>()); std::stable_sort(d2.begin(),d2.end(),std::greater<T>()); ++checks; if(c2!=d2){ if(fails<10) printf("FAIL stable> %s n=%zu d=%d\n",tn,n,d); ++fails; } }
  }}
}
int main(){ go<int32_t>("i32"); go<uint32_t>("u32"); go<int64_t>("i64"); go<uint64_t>("u64"); go<float>("f32"); go<double>("f64");
 // NaN: must not crash, non-NaN prefix sorted and multiset preserved
 std::mt19937_64 g(5); for(size_t n: {10ul,100ul,1000ul,5000ul,15000ul}){ std::vector<double> v(n); for(auto&x:v) x = (g()%10==0)? NAN : (double)(g()%1000); auto cnt=std::count_if(v.begin(),v.end(),[](double x){return std::isnan(x);}); fyx::sort(v.data(),n); ++checks; if(std::count_if(v.begin(),v.end(),[](double x){return std::isnan(x);})!=cnt){printf("NaN count FAIL n=%zu\n",n);++fails;} }
 printf("t_small checks=%ld fails=%ld\n",checks,fails); return fails!=0; }
