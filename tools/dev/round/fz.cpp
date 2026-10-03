#ifndef HDR
#define HDR "../../../fyx_sort.hpp"
#endif
#define FYX_ENABLE_TEST_HOOKS 1
#include HDR
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cstring>
struct KV { uint64_t v; uint64_t k; };
struct L { bool operator()(const KV&a,const KV&b)const{return a.k<b.k;} };
struct G { bool operator()(const KV&a,const KV&b)const{return a.k>b.k;} };
struct X { bool operator()(const KV&a,const KV&b)const{return a.k!=b.k? a.k<b.k : a.v<b.v;} };
struct KVs { uint64_t k; int64_t v; };           // key first, signed payload
struct Ls { bool operator()(const KVs&a,const KVs&b)const{return a.k<b.k;} };
struct KVi { int64_t v; int64_t k; };            // signed key
struct Li { bool operator()(const KVi&a,const KVi&b)const{return a.k<b.k;} };
struct R32 { uint32_t k; uint32_t id; };
struct LR { bool operator()(const R32&a,const R32&b)const{return a.k<b.k;} };
static std::mt19937_64 g(99);
template<class T,class C> bool same(const std::vector<T>& a, const std::vector<T>& b, C c){
  for(size_t i=1;i<b.size();i++) if(c(b[i],b[i-1])) return false;
  auto key=[](const T&x){ uint64_t h=0; std::memcpy(&h,&x,8); uint64_t h2=0; std::memcpy(&h2,(const char*)&x+sizeof(T)-8,8); return h*31+h2;};
  std::vector<uint64_t> x,y; for(auto&e:a)x.push_back(key(e)); for(auto&e:b)y.push_back(key(e));
  std::sort(x.begin(),x.end()); std::sort(y.begin(),y.end()); return x==y; }
template<class T,class C> void shape(std::vector<T>& v, int s, C less){
  size_t n=v.size(); if(n<4) return;
  switch(s){
   case 1: std::sort(v.begin(),v.end(),less); break;
   case 2: std::sort(v.begin(),v.end(),less); std::reverse(v.begin(),v.end()); break;
   case 3: { size_t m=n/2; std::sort(v.begin(),v.begin()+m,less); std::sort(v.begin()+m,v.end(),less);} break;
   case 4: { size_t R=2+g()%9; for(size_t r=0;r<R;r++){ size_t a=r*n/R,b=(r+1)*n/R; std::sort(v.begin()+a,v.begin()+b,less);} } break;
   case 5: { std::sort(v.begin(),v.end(),less); size_t t=n-n/(5+g()%20); std::shuffle(v.begin()+t,v.end(),g);} break;
   case 6: { std::sort(v.begin(),v.end(),less); std::rotate(v.begin(),v.begin()+g()%n,v.end());} break;
   case 7: { std::sort(v.begin(),v.end(),less); for(int k=0;k<20;k++) std::swap(v[g()%n],v[g()%n]);} break;
   case 8: { size_t R=200+g()%2000; for(size_t a=0;a<n;a+=R) std::sort(v.begin()+a,v.begin()+std::min(n,a+R),less);} break;
   default: break;
  }
}
template<class T,class C> bool chk(const std::vector<T>& v, C c, const char* nm, int it, int s){
  for(int par=0;par<2;par++){
    fyx::Options o; o.parallel= par? fyx::Tri::On : fyx::Tri::Off;
    auto a=v; fyx::sort(a.data(),a.size(),c,o);
    if(!same(v,a,c)){ std::printf("FAIL %s it=%d s=%d par=%d n=%zu disp=%d\n",nm,it,s,par,v.size(),(int)fyx::detail::test_last_dispatch()); return false; }
  }
  return true;
}
int main(){
  const int N = std::getenv("FZN") ? std::atoi(std::getenv("FZN")) : 300;
  for(int it=0;it<N;++it){
    size_t n= (it%4==0)? 1+g()%3000 : (it%4==1)? 3000+g()%60000 : 100000+g()%400000;
    uint64_t mod = it%6==0? 16 : it%6==1? 256 : it%6==2? 3000 : it%6==3? (1ull<<20) : it%6==4? 2 : ~0ull;
    bool sparse = g()%2;
    std::vector<uint64_t> pal(4096); for(auto&x:pal) x=g();
    auto key=[&](){ uint64_t r=g()%mod; return (sparse&&mod<=4096)? pal[r] : r; };
    int s=g()%9;
    std::vector<KV> v(n); for(size_t i=0;i<n;i++) v[i]={i,key()}; shape(v,s,L{});
    if(!chk(v,L{},"L",it,s)||!chk(v,G{},"G",it,s)||!chk(v,X{},"X",it,s)) return 1;
    std::vector<KVs> vs(n); for(size_t i=0;i<n;i++) vs[i]={key(),(int64_t)i}; shape(vs,s,Ls{});
    if(!chk(vs,Ls{},"Ls",it,s)) return 1;
    std::vector<KVi> vi(n); for(size_t i=0;i<n;i++) vi[i]={(int64_t)i,(int64_t)key()}; shape(vi,s,Li{});
    if(!chk(vi,Li{},"Li",it,s)) return 1;
    std::vector<R32> w(n); for(size_t i=0;i<n;i++) w[i]={uint32_t(key()),uint32_t(i)}; shape(w,s,LR{});
    if(!chk(w,LR{},"R32",it,s)) return 1;
    std::vector<int64_t> z(n); for(auto&x:z) x=int64_t(key()); shape(z,s,std::less<int64_t>{});
    for(int par=0;par<2;par++){ fyx::Options o; o.parallel= par? fyx::Tri::On : fyx::Tri::Off; auto a=z; fyx::sort(a.data(),n,o); auto r=z; std::sort(r.begin(),r.end()); if(a!=r){std::printf("FAIL i64 it=%d s=%d par=%d\n",it,s,par);return 1;} }
  }
  std::printf("fuzz ok\n");
}
