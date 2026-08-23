#include "../fyx_sort.hpp"
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <functional>
using namespace fyx::detail;
static int fails=0;
template<class T, class C>
void chk(const char* tag, std::vector<T> v, C comp){
    std::vector<T> ref=v; std::stable_sort(ref.begin(),ref.end(),
        [&](const T&a,const T&b){return comp(a,b);});
    std::vector<T> g=v; pdqsort(g.begin(),g.end(),comp);
    // compare as multisets under comp: check sorted + same elements
    bool ok=std::is_sorted(g.begin(),g.end(),comp);
    std::vector<T> s1=g, s2=v;
    std::sort(s1.begin(),s1.end(),comp); std::sort(s2.begin(),s2.end(),comp);
    if(s1!=s2) ok=false;
    if(!ok){printf("  FAIL %s n=%zu\n",tag,v.size()); fails++;}
}
int main(){
    std::mt19937_64 rng(4242);
    size_t sizes[]={0,1,2,3,5,10,23,24,25,50,64,65,100,257,1000,5000,50000,200000};
    for(size_t n:sizes){
        std::vector<int> a(n); for(auto&x:a)x=(int)(rng()%1000000);
        chk("rand",a,std::less<int>());
        chk("rand-greater",a,std::greater<int>());
        std::sort(a.begin(),a.end()); chk("sorted",a,std::less<int>());
        std::reverse(a.begin(),a.end()); chk("rev",a,std::less<int>());
        std::vector<int> e(n,7); chk("equal",e,std::less<int>());
        std::vector<int> d(n); for(auto&x:d)x=(int)(rng()%3); chk("fewdistinct",d,std::less<int>());
        // organ pipe
        std::vector<int> o(n); for(size_t i=0;i<n;i++) o[i]=(int)std::min(i,n-i);
        chk("organpipe",o,std::less<int>());
        // sorted with a few swaps
        std::vector<int> t(n); for(size_t i=0;i<n;i++) t[i]=(int)i;
        for(int k=0;k<(int)n/50;k++) if(n>1) std::swap(t[rng()%n],t[rng()%n]);
        chk("nearsorted",t,std::less<int>());
        // strings (non-trivial type -> branchy path)
        if(n<=5000){ std::vector<std::string> s(n);
            for(size_t i=0;i<n;i++) s[i]=std::to_string(rng()%100000);
            chk("string",s,std::less<std::string>()); }
    }
    // adversarial: killer pattern for median-of-3
    { size_t n=100000; std::vector<int> k(n);
      for(size_t i=0;i<n;i++) k[i]=(i%2)?(int)(i/2):(int)(n/2+i/2);
      chk("median3killer",k,std::less<int>()); }
    printf(fails?"PDQ FAILURES=%d\n":"ALL PDQSORT TESTS PASS\n",fails);
    return fails?1:0;
}
