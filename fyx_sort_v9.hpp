// ================================================================
//  FYX-SORT v9.2 — 终极工程+竞赛双模式版
//
//  默认: 工程模式 (库)
//  取消注释下一行: 竞赛模式 (含main, 可直接提交洛谷)
// ================================================================
// #define FYX_LG_COMPETITION 1
// ================================================================

#ifndef FYX_SORT_HPP
#define FYX_SORT_HPP

#if defined(__GNUC__) || defined(__clang__)
    #pragma GCC optimize("O3,Ofast,unroll-loops,inline-functions,no-stack-protector")
    #if defined(__x86_64__) || defined(__i386__)
        #if defined(__AVX2__)
            #pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
        #elif defined(__SSE4_2__)
            #pragma GCC target("sse4.2,popcnt")
        #endif
    #endif
#endif
#if defined(_MSC_VER) && !defined(__clang__)
    #pragma optimize("gt",on)
    #pragma inline_recursion(on)
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <forward_list>
#include <functional>
#include <future>
#include <iterator>
#include <list>
#include <map>
#include <mutex>
#include <new>
#include <numeric>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __has_include
    #if __has_include(<span>) && (__cplusplus>=202002L||(defined(_MSVC_LANG)&&_MSVC_LANG>=202002L))
        #include <span>
        #define FYX_HAS_SPAN 1
    #else
        #define FYX_HAS_SPAN 0
    #endif
#else
    #define FYX_HAS_SPAN 0
#endif

#if defined(_WIN32)||defined(_WIN64)
    #define FYX_PLAT "Windows"
#elif defined(__APPLE__)
    #define FYX_PLAT "macOS"
#elif defined(__linux__)
    #define FYX_PLAT "Linux"
#else
    #define FYX_PLAT "Unknown"
#endif
#if defined(__x86_64__)||defined(_M_X64)
    #define FYX_ARCH "x86_64"
#elif defined(__aarch64__)||defined(_M_ARM64)
    #define FYX_ARCH "ARM64"
#else
    #define FYX_ARCH "Other"
#endif

#ifdef _MSC_VER
    #define FI __forceinline
#else
    #define FI __attribute__((always_inline)) inline
#endif
#if defined(__GNUC__)||defined(__clang__)
    #define LIKELY(x) __builtin_expect(!!(x),1)
    #define UNLIKELY(x) __builtin_expect(!!(x),0)
    #define PFR(p) __builtin_prefetch((p),0,1)
#else
    #define LIKELY(x) (x)
    #define UNLIKELY(x) (x)
    #define PFR(p) ((void)0)
#endif

namespace FYX {
namespace detail {

// ================================================================
// 类型工具
// ================================================================
template<typename T> struct uint_for{using type=typename std::make_unsigned<T>::type;};
template<> struct uint_for<float>{using type=std::uint32_t;};
template<> struct uint_for<double>{using type=std::uint64_t;};
template<typename T> using uint_of=typename uint_for<T>::type;

template<typename C,typename T>
constexpr bool is_less_v=std::is_same_v<std::decay_t<C>,std::less<T>>||
                         std::is_same_v<std::decay_t<C>,std::less<>>;

inline unsigned hw_threads(){
    static unsigned n=[]{unsigned h=std::thread::hardware_concurrency();return h>0?h:1;}();
    return n;
}

// ================================================================
// 排序网络 n≤12
// ================================================================
#define CS(i,j) do{auto a_=arr[i],b_=arr[j];bool c_=comp(b_,a_);arr[i]=c_?b_:a_;arr[j]=c_?a_:b_;}while(0)
template<typename T,typename X> FI void sn2(T*arr,X comp){CS(0,1);}
template<typename T,typename X> FI void sn3(T*arr,X comp){CS(0,1);CS(1,2);CS(0,1);}
template<typename T,typename X> FI void sn4(T*arr,X comp){CS(0,1);CS(2,3);CS(0,2);CS(1,3);CS(1,2);}
template<typename T,typename X> FI void sn5(T*arr,X comp){
    CS(0,1);CS(3,4);CS(2,4);CS(2,3);CS(1,4);CS(0,3);CS(0,2);CS(1,3);CS(1,2);}
template<typename T,typename X> FI void sn6(T*arr,X comp){
    CS(1,2);CS(4,5);CS(0,2);CS(3,5);CS(0,1);CS(3,4);CS(2,5);CS(0,3);
    CS(1,4);CS(2,4);CS(1,3);CS(2,3);}
template<typename T,typename X> FI void sn7(T*arr,X comp){
    CS(1,2);CS(3,4);CS(5,6);CS(0,2);CS(3,5);CS(4,6);CS(0,1);CS(4,5);
    CS(2,6);CS(0,4);CS(1,5);CS(0,3);CS(2,5);CS(1,3);CS(2,4);CS(2,3);}
template<typename T,typename X> FI void sn8(T*arr,X comp){
    CS(0,1);CS(2,3);CS(4,5);CS(6,7);CS(0,2);CS(1,3);CS(4,6);CS(5,7);
    CS(1,2);CS(5,6);CS(0,4);CS(3,7);CS(1,5);CS(2,6);CS(1,4);CS(3,6);
    CS(2,4);CS(3,5);CS(3,4);}
template<typename T,typename X> FI void sn9(T*arr,X comp){
    CS(0,1);CS(3,4);CS(6,7);CS(1,2);CS(4,5);CS(7,8);CS(0,1);CS(3,4);
    CS(6,7);CS(0,3);CS(3,6);CS(0,3);CS(1,4);CS(4,7);CS(1,4);CS(2,5);
    CS(5,8);CS(2,5);CS(1,3);CS(5,7);CS(2,6);CS(4,6);CS(2,4);CS(2,3);CS(5,6);}
template<typename T,typename X> FI void sn10(T*arr,X comp){
    CS(4,9);CS(3,8);CS(2,7);CS(1,6);CS(0,5);CS(1,4);CS(6,9);CS(0,3);
    CS(5,8);CS(0,2);CS(3,6);CS(7,9);CS(0,1);CS(2,4);CS(5,7);CS(8,9);
    CS(1,2);CS(4,6);CS(7,8);CS(3,5);CS(2,5);CS(6,8);CS(1,3);CS(4,7);
    CS(2,3);CS(6,7);CS(3,4);CS(5,6);CS(4,5);}
template<typename T,typename X> FI void sn11(T*arr,X comp){
    CS(0,1);CS(2,3);CS(4,5);CS(6,7);CS(8,9);CS(1,3);CS(5,7);CS(0,2);
    CS(4,6);CS(8,10);CS(1,2);CS(5,6);CS(9,10);CS(0,4);CS(3,7);CS(1,5);
    CS(6,10);CS(4,8);CS(5,9);CS(2,6);CS(0,4);CS(3,8);CS(1,5);CS(6,10);
    CS(2,3);CS(8,9);CS(1,4);CS(7,10);CS(3,5);CS(6,8);CS(2,4);CS(7,9);
    CS(3,4);CS(5,6);CS(7,8);}
template<typename T,typename X> FI void sn12(T*arr,X comp){
    CS(0,1);CS(2,3);CS(4,5);CS(6,7);CS(8,9);CS(10,11);CS(1,3);CS(5,7);
    CS(9,11);CS(0,2);CS(4,6);CS(8,10);CS(1,2);CS(5,6);CS(9,10);CS(0,4);
    CS(7,11);CS(1,5);CS(6,10);CS(3,7);CS(4,8);CS(5,9);CS(2,6);CS(0,4);
    CS(7,11);CS(3,8);CS(1,5);CS(6,10);CS(2,3);CS(8,9);CS(1,4);CS(7,10);
    CS(3,5);CS(6,8);CS(2,4);CS(7,9);CS(3,4);CS(5,6);CS(7,8);}
template<typename T,typename X>
FI void sort_net(T*a,std::size_t n,X c){
    switch(n){
        case 0:case 1:return;case 2:sn2(a,c);return;case 3:sn3(a,c);return;
        case 4:sn4(a,c);return;case 5:sn5(a,c);return;case 6:sn6(a,c);return;
        case 7:sn7(a,c);return;case 8:sn8(a,c);return;case 9:sn9(a,c);return;
        case 10:sn10(a,c);return;case 11:sn11(a,c);return;case 12:sn12(a,c);return;
    }
}
#undef CS

// ================================================================
// 插入排序
// ================================================================
template<typename T,typename X>
FI void isort(T*lo,T*hi,X comp){
    if(hi-lo<=1)return;
    T*m=lo;for(T*p=lo+1;p<hi;++p)if(comp(*p,*m))m=p;
    if(m!=lo)std::swap(*lo,*m);
    for(T*i=lo+2;i<hi;++i){
        T t=std::move(*i);T*j=i;
        while(comp(t,*(j-1))){*j=std::move(*(j-1));--j;}
        *j=std::move(t);
    }
}

// ================================================================
// 堆排序 + 划分
// ================================================================
template<typename T,typename X>
void hsort(T*lo,T*hi,X c){std::make_heap(lo,hi,c);std::sort_heap(lo,hi,c);}

template<typename T,typename X>
FI T*med3(T*a,T*b,T*c,X comp){
    return comp(*a,*b)?(comp(*b,*c)?b:(comp(*a,*c)?c:a))
                      :(comp(*a,*c)?a:(comp(*b,*c)?c:b));
}

template<typename T,typename X>
FI std::pair<T*,T*> part3(T*lo,T*hi,X comp){
    std::size_t n=hi-lo,s=n>>3;
    T*p=med3(med3(lo,lo+s,lo+2*s,comp),med3(lo+n/2-s,lo+n/2,lo+n/2+s,comp),
             med3(hi-1-2*s,hi-1-s,hi-1,comp),comp);
    T pv=*p;T*lt=lo,*i=lo,*gt=hi;
    while(i<gt){
        if(comp(*i,pv))std::swap(*lt++,*i++);
        else if(comp(pv,*i))std::swap(*i,*--gt);
        else ++i;
    }
    return{lt,gt};
}

template<typename T,typename X>
FI std::pair<T*,T*> dual_part(T*lo,T*hi,X comp){
    std::size_t n=hi-lo,s=n/3;
    T*a=lo+s,*b=hi-1-s;
    if(comp(*b,*a))std::swap(*a,*b);
    std::swap(*a,*lo);std::swap(*b,*(hi-1));
    T p=*lo,q=*(hi-1);
    T*lt=lo+1,*gt=hi-2,*k=lt;
    while(k<=gt){
        if(comp(*k,p)){std::swap(*k,*lt);++lt;++k;}
        else if(comp(q,*k)){
            while(k<gt&&comp(q,*gt))--gt;
            std::swap(*k,*gt);--gt;
            if(comp(*k,p)){std::swap(*k,*lt);++lt;}
            ++k;
        }else ++k;
    }
    --lt;++gt;
    std::swap(*lo,*lt);std::swap(*(hi-1),*gt);
    return{lt,gt};
}

// ================================================================
// 内省排序
// ================================================================
template<typename T,typename X>
void intro_loop(T*lo,T*hi,int d,X comp){
    while(hi-lo>40){
        if(UNLIKELY(d==0)){hsort(lo,hi,comp);return;}
        --d;
        std::size_t n=hi-lo,step=n/8;
        int eq=0;T samp=*(lo+n/2);
        for(int i=0;i<8;++i)
            if(!(comp(*(lo+i*step),samp)||comp(samp,*(lo+i*step))))++eq;
        if(eq>=3){
            auto[lt,gt]=part3(lo,hi,comp);
            if(lt-lo<hi-gt){intro_loop(lo,lt,d,comp);lo=gt;}
            else{intro_loop(gt,hi,d,comp);hi=lt;}
        }else{
            auto[lt,gt]=dual_part(lo,hi,comp);
            std::size_t s1=lt-lo,s2=gt-lt-1,s3=hi-gt-1;
            if(s1<=s2&&s1<=s3){intro_loop(lo,lt,d,comp);intro_loop(lt+1,gt,d,comp);lo=gt+1;}
            else if(s2<=s1&&s2<=s3){intro_loop(lo,lt,d,comp);intro_loop(gt+1,hi,d,comp);lo=lt+1;hi=gt;}
            else{intro_loop(lt+1,gt,d,comp);intro_loop(gt+1,hi,d,comp);hi=lt;}
        }
    }
}

template<typename T,typename X>
FI void introsort(T*lo,T*hi,X comp){
    if(hi-lo<=1)return;
    std::size_t n=hi-lo;int d=0;while(n>1){++d;n>>=1;}d*=2;
    intro_loop(lo,hi,d,comp);
    isort(lo,hi,comp);
}

// ================================================================
// 并行快排
// ================================================================
template<typename T,typename X>
void par_qsort(T*lo,T*hi,X comp,int depth){
    std::size_t n=hi-lo;
    if(n<=200000||depth<=0){introsort(lo,hi,comp);return;}
    auto[lt,gt]=part3(lo,hi,comp);
    --depth;
    // 只在两段都足够大时才 spawn
    std::size_t sl=lt-lo,sr=hi-gt;
    if(sl>100000&&sr>100000&&depth>0){
        auto fut=std::async(std::launch::async,[=]{par_qsort(lo,lt,comp,depth);});
        par_qsort(gt,hi,comp,depth);
        fut.wait();
    }else if(sl<sr){
        par_qsort(lo,lt,comp,depth);
        par_qsort(gt,hi,comp,depth);
    }else{
        par_qsort(gt,hi,comp,depth);
        par_qsort(lo,lt,comp,depth);
    }
}

// ================================================================
// 单线程基数排序 32位
// ================================================================
template<typename T>
bool radix32(T*a,std::size_t n){
    static_assert(sizeof(T)==4);
    using U=std::uint32_t;
    std::vector<T> buf;
    try{buf.resize(n);}catch(...){return false;}
    constexpr U FLIP=std::is_signed_v<T>?U(1)<<31:0;
    alignas(64) std::size_t h[4][256]={};
    std::size_t i=0;
    for(;i+3<n;i+=4){
        U v0,v1,v2,v3;
        std::memcpy(&v0,&a[i],4);std::memcpy(&v1,&a[i+1],4);
        std::memcpy(&v2,&a[i+2],4);std::memcpy(&v3,&a[i+3],4);
        v0^=FLIP;v1^=FLIP;v2^=FLIP;v3^=FLIP;
        ++h[0][v0&0xFF];++h[1][(v0>>8)&0xFF];++h[2][(v0>>16)&0xFF];++h[3][v0>>24];
        ++h[0][v1&0xFF];++h[1][(v1>>8)&0xFF];++h[2][(v1>>16)&0xFF];++h[3][v1>>24];
        ++h[0][v2&0xFF];++h[1][(v2>>8)&0xFF];++h[2][(v2>>16)&0xFF];++h[3][v2>>24];
        ++h[0][v3&0xFF];++h[1][(v3>>8)&0xFF];++h[2][(v3>>16)&0xFF];++h[3][v3>>24];
    }
    for(;i<n;++i){U v;std::memcpy(&v,&a[i],4);v^=FLIP;++h[0][v&0xFF];++h[1][(v>>8)&0xFF];++h[2][(v>>16)&0xFF];++h[3][v>>24];}
    
    bool skip[4]={};
    for(int b=0;b<4;++b)for(int j=0;j<256;++j)if(h[b][j]==n){skip[b]=true;break;}
    T*s=a,*d=buf.data();
    if constexpr(std::is_signed_v<T>){for(std::size_t k=0;k<n;++k){U v;std::memcpy(&v,&s[k],4);v^=FLIP;std::memcpy(&s[k],&v,4);}}
    for(int b=0;b<4;++b){
        if(skip[b])continue;
        for(int j=1;j<256;++j)h[b][j]+=h[b][j-1];
        int sh=b*8;
        for(std::ptrdiff_t k=static_cast<std::ptrdiff_t>(n)-1;k>=0;--k){
            U v;std::memcpy(&v,&s[k],4);d[--h[b][(v>>sh)&0xFF]]=s[k];
        }
        std::swap(s,d);
    }
    if constexpr(std::is_signed_v<T>){for(std::size_t k=0;k<n;++k){U v;std::memcpy(&v,&s[k],4);v^=FLIP;std::memcpy(&s[k],&v,4);}}
    if(s!=a)std::memcpy(a,s,n*sizeof(T));
    return true;
}

// ================================================================
// 并行基数排序 32位 — 真正并行分配
// ================================================================
// 替换 par_radix32 函数为以下版本:

template<typename T>
bool par_radix32(T*a,std::size_t n,unsigned nt){
    static_assert(sizeof(T)==4);
    using U=std::uint32_t;
    if(nt<=1)return false;
    
    std::vector<T> buf;
    try{buf.resize(n);}catch(...){return false;}
    constexpr U FLIP=std::is_signed_v<T>?U(1)<<31:0;
    
    // 分块边界
    std::vector<std::size_t> bounds(nt+1);
    for(unsigned t=0;t<=nt;++t)
        bounds[t]=std::min<std::size_t>(static_cast<std::size_t>(t)*(n/nt)+(t<n%nt?t:n%nt),n);
    
    // 并行翻转
    if constexpr(std::is_signed_v<T>){
        std::vector<std::thread> threads;
        for(unsigned t=0;t<nt;++t){
            threads.emplace_back([&,t]{
                for(std::size_t i=bounds[t];i<bounds[t+1];++i){
                    U v;std::memcpy(&v,&a[i],4);v^=FLIP;std::memcpy(&a[i],&v,4);
                }
            });
        }
        for(auto&th:threads)th.join();
    }
    
    // 并行直方图构建
    std::vector<std::array<std::array<std::size_t,256>,4>> local_h(nt);
    {
        std::vector<std::thread> threads;
        for(unsigned t=0;t<nt;++t){
            threads.emplace_back([&,t]{
                for(auto&arr:local_h[t])arr.fill(0);
                for(std::size_t i=bounds[t];i<bounds[t+1];++i){
                    U v;std::memcpy(&v,&a[i],4);
                    ++local_h[t][0][v&0xFF];++local_h[t][1][(v>>8)&0xFF];
                    ++local_h[t][2][(v>>16)&0xFF];++local_h[t][3][v>>24];
                }
            });
        }
        for(auto&th:threads)th.join();
    }
    
    // 合并
    alignas(64) std::size_t h[4][256]={};
    for(unsigned t=0;t<nt;++t)
        for(int b=0;b<4;++b)
            for(int j=0;j<256;++j)
                h[b][j]+=local_h[t][b][j];
    
    bool skip[4]={};
    for(int b=0;b<4;++b)
        for(int j=0;j<256;++j)
            if(h[b][j]==n){skip[b]=true;break;}
    
    T*s=a,*d=buf.data();
    
    // 分配阶段: 单线程反向保证正确性
    for(int b=0;b<4;++b){
        if(skip[b])continue;
        for(int j=1;j<256;++j)h[b][j]+=h[b][j-1];
        int sh=b*8;
        for(std::ptrdiff_t k=static_cast<std::ptrdiff_t>(n)-1;k>=0;--k){
            U v;std::memcpy(&v,&s[k],4);
            d[--h[b][(v>>sh)&0xFF]]=s[k];
        }
        std::swap(s,d);
    }
    
    // 并行还原翻转
    if constexpr(std::is_signed_v<T>){
        std::vector<std::thread> threads;
        for(unsigned t=0;t<nt;++t){
            std::size_t start=bounds[t],end=bounds[t+1];
            // 注意: s可能指向buf, 边界用n重算
            std::size_t s2=static_cast<std::size_t>(t)*(n/nt);
            std::size_t e2=(t==nt-1)?n:(t+1)*(n/nt);
            threads.emplace_back([=]{
                for(std::size_t i=s2;i<e2;++i){
                    U v;std::memcpy(&v,&s[i],4);v^=FLIP;std::memcpy(&s[i],&v,4);
                }
            });
        }
        for(auto&th:threads)th.join();
    }
    
    if(s!=a)std::memcpy(a,s,n*sizeof(T));
    return true;
}



// ================================================================
// 基数排序 64位
// ================================================================
template<typename T>
bool radix64(T*a,std::size_t n){
    static_assert(sizeof(T)==8);using U=std::uint64_t;
    std::vector<T> buf;try{buf.resize(n);}catch(...){return false;}
    constexpr U FLIP=std::is_signed_v<T>?U(1)<<63:0;
    alignas(64) std::size_t h[8][256]={};
    for(std::size_t i=0;i<n;++i){U v;std::memcpy(&v,&a[i],8);v^=FLIP;for(int b=0;b<8;++b)++h[b][(v>>(b*8))&0xFF];}
    bool skip[8]={};
    for(int b=0;b<8;++b)for(int j=0;j<256;++j)if(h[b][j]==n){skip[b]=true;break;}
    T*s=a,*d=buf.data();
    if constexpr(std::is_signed_v<T>){for(std::size_t k=0;k<n;++k){U v;std::memcpy(&v,&s[k],8);v^=FLIP;std::memcpy(&s[k],&v,8);}}
    for(int b=0;b<8;++b){
        if(skip[b])continue;for(int j=1;j<256;++j)h[b][j]+=h[b][j-1];int sh=b*8;
        for(std::ptrdiff_t k=static_cast<std::ptrdiff_t>(n)-1;k>=0;--k){U v;std::memcpy(&v,&s[k],8);d[--h[b][(v>>sh)&0xFF]]=s[k];}
        std::swap(s,d);
    }
    if constexpr(std::is_signed_v<T>){for(std::size_t k=0;k<n;++k){U v;std::memcpy(&v,&s[k],8);v^=FLIP;std::memcpy(&s[k],&v,8);}}
    if(s!=a)std::memcpy(a,s,n*sizeof(T));return true;
}

// ================================================================
// 浮点基数排序
// ================================================================
template<typename T>
bool radix_fp(T*a,std::size_t n){
    static_assert(std::is_floating_point_v<T>);
    using U=uint_of<T>;constexpr int B=sizeof(T);constexpr U S=U(1)<<(B*8-1);
    std::vector<T> buf;try{buf.resize(n);}catch(...){return false;}
    T*s=a,*d=buf.data();
    for(std::size_t i=0;i<n;++i){U v;std::memcpy(&v,&s[i],B);v^=(v&S)?~U(0):S;std::memcpy(&s[i],&v,B);}
    alignas(64) std::size_t h[B][256]={};
    for(std::size_t i=0;i<n;++i){U v;std::memcpy(&v,&s[i],B);for(int b=0;b<B;++b)++h[b][(v>>(b*8))&0xFF];}
    bool skip[B]={};
    for(int b=0;b<B;++b)for(int j=0;j<256;++j)if(h[b][j]==n){skip[b]=true;break;}
    for(int b=0;b<B;++b){
        if(skip[b])continue;for(int j=1;j<256;++j)h[b][j]+=h[b][j-1];int sh=b*8;
        for(std::ptrdiff_t k=static_cast<std::ptrdiff_t>(n)-1;k>=0;--k){U v;std::memcpy(&v,&s[k],B);d[--h[b][(v>>sh)&0xFF]]=s[k];}
        std::swap(s,d);
    }
    for(std::size_t i=0;i<n;++i){U v;std::memcpy(&v,&s[i],B);v^=(v&S)?S:~U(0);std::memcpy(&s[i],&v,B);}
    if(s!=a)std::memcpy(a,s,n*sizeof(T));return true;
}

// ================================================================
// 计数排序
// ================================================================
template<typename T>
bool csort(T*a,std::size_t n,T lo,T hi){
    unsigned long long r;
    if constexpr(std::is_signed_v<T>)
        r=static_cast<unsigned long long>(static_cast<long long>(hi)-static_cast<long long>(lo));
    else r=static_cast<unsigned long long>(hi)-static_cast<unsigned long long>(lo);
    if(r>4000000)return false;
    std::size_t rs=static_cast<std::size_t>(r+1);
    std::vector<std::size_t> cnt;
    try{cnt.resize(rs,0);}catch(...){return false;}
    for(std::size_t i=0;i<n;++i){
        std::size_t idx;
        if constexpr(std::is_signed_v<T>)
            idx=static_cast<std::size_t>(static_cast<long long>(a[i])-static_cast<long long>(lo));
        else idx=static_cast<std::size_t>(a[i]-lo);
        ++cnt[idx];
    }
    std::size_t p=0;
    for(std::size_t i=0;i<rs;++i){
        T v;
        if constexpr(std::is_signed_v<T>)
            v=static_cast<T>(static_cast<long long>(lo)+static_cast<long long>(i));
        else v=static_cast<T>(lo+static_cast<T>(i));
        std::size_t c=cnt[i];
        while(c>=8){a[p]=v;a[p+1]=v;a[p+2]=v;a[p+3]=v;a[p+4]=v;a[p+5]=v;a[p+6]=v;a[p+7]=v;p+=8;c-=8;}
        while(c--)a[p++]=v;
    }
    return true;
}

// ================================================================
// 预扫描
// ================================================================
template<typename T>
struct Scan{bool asc,desc,eq;T lo,hi;};

template<typename T>
Scan<T> prescan(T*a,std::size_t n){
    Scan<T> r{true,true,false,a[0],a[0]};
    for(std::size_t i=1;i<n;++i){
        if(a[i]<a[i-1])r.asc=false;
        if(a[i-1]<a[i])r.desc=false;
        if(a[i]<r.lo)r.lo=a[i];
        if(r.hi<a[i])r.hi=a[i];
    }
    r.eq=!(r.lo<r.hi)&&!(r.hi<r.lo);
    return r;
}

// ================================================================
// 主引擎
// ================================================================
template<typename T,typename X>
void sort_impl(T*lo,T*hi,X comp){
    std::size_t n=hi-lo;
    if(n<=1)return;
    if(n<=12){sort_net(lo,n,comp);return;}
    if(n<=40){isort(lo,hi,comp);return;}
    
    constexpr bool def=is_less_v<X,T>;
    unsigned nt=hw_threads();
    
    if constexpr(def){
        auto sc=prescan(lo,n);
        if(LIKELY(sc.asc))return;
        if(sc.eq)return;
        if(sc.desc){std::reverse(lo,hi);return;}
        
        if constexpr(std::is_integral_v<T>){
            if(csort(lo,n,sc.lo,sc.hi))return;
            if(n>=64){
                if constexpr(sizeof(T)==4){
                    // 并行基数排序阈值: 5M+
                    if(n>=5000000&&nt>1){
                        if(par_radix32(lo,n,nt))return;
                    }
                    if(radix32(lo,n))return;
                }
                else if constexpr(sizeof(T)==8){if(radix64(lo,n))return;}
            }
        }
        if constexpr(std::is_floating_point_v<T>){
            if(n>=128){if(radix_fp(lo,n))return;}
        }
    }
    
    // 并行快排阈值: 1M+
    if(n>=1000000&&nt>1){
        int depth=0;unsigned t=nt;while(t>1){++depth;t>>=1;}
        par_qsort(lo,hi,comp,depth+2);
        return;
    }
    
    introsort(lo,hi,comp);
}

// 重型对象
template<typename It,typename X>
void heavy_sort(It lo,It hi,X comp){
    using T=typename std::iterator_traits<It>::value_type;
    std::size_t n=std::distance(lo,hi);
    std::vector<std::size_t> idx(n);std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](auto a,auto b){return comp(*(lo+a),*(lo+b));});
    std::vector<bool> vis(n);
    for(std::size_t i=0;i<n;++i){
        if(!vis[i]&&idx[i]!=i){
            std::size_t c=i;T t=std::move(*(lo+i));
            while(true){vis[c]=true;auto nx=idx[c];if(nx==i)break;*(lo+c)=std::move(*(lo+nx));c=nx;}
            *(lo+c)=std::move(t);
        }
    }
}

template<typename It,typename X>
void sort_engine(It lo,It hi,X comp){
    auto n=std::distance(lo,hi);if(n<=1)return;
    using T=typename std::iterator_traits<It>::value_type;
    if constexpr(!std::is_arithmetic_v<T>&&sizeof(T)>64){heavy_sort(lo,hi,comp);return;}
    if constexpr(std::is_trivially_copyable_v<T>){sort_impl(&*lo,&*lo+n,comp);return;}
    std::sort(lo,hi,comp);
}

// 适配器破解
template<typename T,typename C> struct stk_h:std::stack<T,C>{static C&get(std::stack<T,C>&s){return s.*(&stk_h::c);}};
template<typename T,typename C> struct que_h:std::queue<T,C>{static C&get(std::queue<T,C>&q){return q.*(&que_h::c);}};
template<typename T,typename C,typename P> struct pq_h:std::priority_queue<T,C,P>{static C&get(std::priority_queue<T,C,P>&pq){return pq.*(&pq_h::c);}};

} // namespace detail

// ================================================================
// 公共 API
// ================================================================
template<typename T,typename C> void sort(T*lo,T*hi,C c){if(lo<hi)detail::sort_impl(lo,hi,c);}
template<typename T> void sort(T*lo,T*hi){sort(lo,hi,std::less<T>{});}
template<typename T,std::size_t N,typename C> void sort(T(&a)[N],C c){detail::sort_impl(a,a+N,c);}
template<typename T,std::size_t N> void sort(T(&a)[N]){sort(a,std::less<T>{});}
template<typename T,typename A,typename C> void sort(std::vector<T,A>&v,C c){if(v.size()>1)detail::sort_impl(v.data(),v.data()+v.size(),c);}
template<typename T,typename A> void sort(std::vector<T,A>&v){sort(v,std::less<T>{});}
template<typename T,std::size_t N,typename C> void sort(std::array<T,N>&a,C c){if(N>1)detail::sort_impl(a.data(),a.data()+N,c);}
template<typename T,std::size_t N> void sort(std::array<T,N>&a){sort(a,std::less<T>{});}
template<typename T,typename A,typename C> void sort(std::deque<T,A>&d,C c){if(d.size()<=1)return;std::vector<T> v(d.begin(),d.end());detail::sort_impl(v.data(),v.data()+v.size(),c);std::copy(v.begin(),v.end(),d.begin());}
template<typename T,typename A> void sort(std::deque<T,A>&d){sort(d,std::less<T>{});}
template<typename T,typename A,typename C> void sort(std::list<T,A>&l,C c){l.sort(c);}
template<typename T,typename A> void sort(std::list<T,A>&l){l.sort();}
template<typename T,typename A,typename C> void sort(std::forward_list<T,A>&l,C c){l.sort(c);}
template<typename T,typename A> void sort(std::forward_list<T,A>&l){l.sort();}
template<typename Ch,typename Tr,typename A,typename C> void sort(std::basic_string<Ch,Tr,A>&s,C c){if(s.size()>1)detail::sort_impl(&s[0],&s[0]+s.size(),c);}
template<typename Ch,typename Tr,typename A> void sort(std::basic_string<Ch,Tr,A>&s){sort(s,std::less<Ch>{});}
template<typename K,typename C,typename A> void sort(std::set<K,C,A>&){}
template<typename K,typename C,typename A> void sort(std::multiset<K,C,A>&){}
template<typename K,typename V,typename C,typename A> void sort(std::map<K,V,C,A>&){}
template<typename K,typename V,typename C,typename A> void sort(std::multimap<K,V,C,A>&){}
template<typename K,typename H,typename E,typename A,typename C> void sort(std::unordered_set<K,H,E,A>&s,C c){std::vector<K> v(s.begin(),s.end());detail::sort_engine(v.begin(),v.end(),c);s.clear();for(auto&x:v)s.insert(std::move(x));}
template<typename K,typename H,typename E,typename A> void sort(std::unordered_set<K,H,E,A>&s){sort(s,std::less<K>{});}
template<typename K,typename H,typename E,typename A,typename C> void sort(std::unordered_multiset<K,H,E,A>&s,C c){std::vector<K> v(s.begin(),s.end());detail::sort_engine(v.begin(),v.end(),c);s.clear();for(auto&x:v)s.insert(std::move(x));}
template<typename K,typename H,typename E,typename A> void sort(std::unordered_multiset<K,H,E,A>&s){sort(s,std::less<K>{});}
template<typename K,typename V,typename H,typename E,typename A,typename C> void sort(std::unordered_map<K,V,H,E,A>&m,C c){using P=std::pair<K,V>;std::vector<P> v;v.reserve(m.size());for(auto&kv:m)v.emplace_back(kv.first,kv.second);detail::sort_engine(v.begin(),v.end(),c);m.clear();for(auto&p:v)m.emplace(std::move(p));}
template<typename K,typename V,typename H,typename E,typename A> void sort(std::unordered_map<K,V,H,E,A>&m){sort(m,[](auto&a,auto&b){return a.first<b.first;});}
template<typename K,typename V,typename H,typename E,typename A,typename C> void sort(std::unordered_multimap<K,V,H,E,A>&m,C c){using P=std::pair<K,V>;std::vector<P> v;v.reserve(m.size());for(auto&kv:m)v.emplace_back(kv.first,kv.second);detail::sort_engine(v.begin(),v.end(),c);m.clear();for(auto&p:v)m.emplace(std::move(p));}
template<typename K,typename V,typename H,typename E,typename A> void sort(std::unordered_multimap<K,V,H,E,A>&m){sort(m,[](auto&a,auto&b){return a.first<b.first;});}
template<typename T,typename C,typename Cm> void sort(std::stack<T,C>&s,Cm c){auto&u=detail::stk_h<T,C>::get(s);std::vector<T> v(u.begin(),u.end());detail::sort_engine(v.begin(),v.end(),c);std::copy(v.begin(),v.end(),u.begin());}
template<typename T,typename C> void sort(std::stack<T,C>&s){sort(s,std::less<T>{});}
template<typename T,typename C,typename Cm> void sort(std::queue<T,C>&q,Cm c){auto&u=detail::que_h<T,C>::get(q);std::vector<T> v(u.begin(),u.end());detail::sort_engine(v.begin(),v.end(),c);std::copy(v.begin(),v.end(),u.begin());}
template<typename T,typename C> void sort(std::queue<T,C>&q){sort(q,std::less<T>{});}
template<typename T,typename C,typename P,typename Cm> void sort(std::priority_queue<T,C,P>&pq,Cm c){auto&u=detail::pq_h<T,C,P>::get(pq);std::vector<T> v(u.begin(),u.end());detail::sort_engine(v.begin(),v.end(),c);while(!pq.empty())pq.pop();for(auto&x:v)pq.push(std::move(x));}
template<typename T,typename C,typename P> void sort(std::priority_queue<T,C,P>&pq){sort(pq,P{});}
template<std::size_t N> void sort(std::bitset<N>&b){auto o=b.count();b.reset();for(std::size_t i=0;i<o;++i)b.set(i);}
template<std::size_t N,typename C> void sort(std::bitset<N>&b,C c){auto o=b.count();b.reset();if(c(0,1))for(std::size_t i=0;i<o;++i)b.set(i);else for(std::size_t i=N-o;i<N;++i)b.set(i);}
#if FYX_HAS_SPAN
template<typename T,std::size_t E,typename C> void sort(std::span<T,E>s,C c){if(s.size()>1)detail::sort_impl(s.data(),s.data()+s.size(),c);}
template<typename T,std::size_t E> void sort(std::span<T,E>s){sort(s,std::less<T>{});}
#endif

template<typename C,typename Cm=std::less<typename C::value_type>>
auto sorted_copy(const C&c,Cm cm=Cm{}){std::vector<typename C::value_type> v(c.begin(),c.end());detail::sort_engine(v.begin(),v.end(),cm);return v;}
template<typename It,typename Cm=std::less<typename std::iterator_traits<It>::value_type>>
bool check_sorted(It lo,It hi,Cm c=Cm{}){if(lo==hi)return true;for(auto p=lo,i=std::next(lo);i!=hi;++i,++p)if(c(*i,*p))return false;return true;}
template<typename C,typename Cm=std::less<typename C::value_type>>
bool check_sorted(const C&c,Cm cm=Cm{}){return check_sorted(c.begin(),c.end(),cm);}
constexpr const char*version(){return "FYX-SORT v9.2";}
constexpr const char*platform(){return FYX_PLAT;}
constexpr const char*arch(){return FYX_ARCH;}
inline unsigned threads(){return detail::hw_threads();}

} // namespace FYX
#endif // FYX_SORT_HPP

// ================================================================
// 竞赛模式: 取消注释顶部的 #define FYX_LG_COMPETITION 1
// ================================================================
#ifdef FYX_LG_COMPETITION

#include <cstdio>
namespace IO{
    constexpr int B=1<<22;char ib[B],*p1=ib,*p2=ib,ob[B],*op=ob;
    inline char gc(){if(p1==p2){p2=(p1=ib)+fread(ib,1,B,stdin);if(p1==p2)return EOF;}return*p1++;}
    inline int read(){int x=0;bool f=0;char c=gc();while(c<'0'||c>'9'){if(c=='-')f=1;c=gc();if(c==EOF)return 0;}while(c>='0'&&c<='9'){x=(x<<3)+(x<<1)+(c^48);c=gc();}return f?-x:x;}
    inline void pc(char c){if(op-ob==B){fwrite(ob,1,B,stdout);op=ob;}*op++=c;}
    inline void write(int x){if(x<0){pc('-');x=-x;}static char s[12];int t=0;if(!x)s[t++]='0';while(x){s[t++]=x%10+'0';x/=10;}while(t--)pc(s[t]);pc(' ');}
    inline void flush(){if(op!=ob)fwrite(ob,1,op-ob,stdout);}
}
int a_buf[100000005];
int main(){
    int n=IO::read();
    for(int i=0;i<n;++i)a_buf[i]=IO::read();
    FYX::sort(a_buf,a_buf+n);
    for(int i=0;i<n;++i)IO::write(a_buf[i]);
    IO::flush();
    return 0;
}

#endif // FYX_LG_COMPETITION

// ================================================================
// 工程测试模式
// ================================================================
#if defined(FYX_RUN_TESTS) && !defined(FYX_LG_COMPETITION)
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
namespace test{
static int pass=0,fail=0;
#define TEST(n,e) do{if(e){++pass;std::cout<<"  [PASS] "<<n<<"\n";}else{++fail;std::cout<<"  [FAIL] "<<n<<"\n";}}while(0)
template<typename F> double bench(F f,int r=3){f();double best=1e18;for(int i=0;i<r;++i){auto t0=std::chrono::high_resolution_clock::now();f();auto t1=std::chrono::high_resolution_clock::now();double ms=std::chrono::duration<double,std::milli>(t1-t0).count();if(ms<best)best=ms;}return best;}

void correctness(){
    std::cout<<"\n=== Correctness ===\n";
    {std::vector<int> v={5,3,8,1,9,2,7,4,6,0};FYX::sort(v);TEST("int asc",FYX::check_sorted(v));}
    {std::vector<int> v={5,3,8,1,9};FYX::sort(v,std::greater<int>{});TEST("int desc",FYX::check_sorted(v,std::greater<int>{}));}
    {std::vector<double> v={3.14,1.41,2.72,0.577};FYX::sort(v);TEST("double",FYX::check_sorted(v));}
    {std::vector<float> v={3.14f,1.41f,2.72f};FYX::sort(v);TEST("float",FYX::check_sorted(v));}
    {std::vector<long long> v={1000000000000LL,-500000000000LL,0LL,900000000000LL,-1LL};FYX::sort(v);TEST("long long",FYX::check_sorted(v));}
    {std::vector<int> v={1,2,3,4,5};FYX::sort(v);TEST("sorted",FYX::check_sorted(v));}
    {std::vector<int> v={5,4,3,2,1};FYX::sort(v);TEST("reversed",FYX::check_sorted(v));}
    {std::vector<int> v(100,42);FYX::sort(v);TEST("all same",FYX::check_sorted(v));}
    {std::vector<int> v;FYX::sort(v);TEST("empty",v.empty());}
    {std::vector<int> v={42};FYX::sort(v);TEST("single",v[0]==42);}
    {std::vector<int> v={-100,50,-200,300,0,-1};FYX::sort(v);TEST("negatives",FYX::check_sorted(v));}
    for(int n=2;n<=12;++n){std::vector<int> v(n);for(int i=0;i<n;++i)v[i]=n-i;FYX::sort(v);TEST("net n="+std::to_string(n),FYX::check_sorted(v));}
    {std::mt19937 rng(999);bool ok=true;for(int n=2;n<=12;++n)for(int t=0;t<2000;++t){std::vector<int> v(n);for(auto&x:v)x=std::uniform_int_distribution<int>(-50,50)(rng);FYX::sort(v);if(!FYX::check_sorted(v)){ok=false;break;}}TEST("net random 2000x",ok);}
    {std::array<int,8> a={8,3,1,7,2,6,4,5};FYX::sort(a);TEST("array",FYX::check_sorted(a));}
    {int a[]={10,3,7,1,8,2,9};FYX::sort(a);TEST("C-array",FYX::check_sorted(std::begin(a),std::end(a)));}
    {std::deque<int> d={9,1,5,3,7,2};FYX::sort(d);TEST("deque",FYX::check_sorted(d));}
    {std::list<int> l={5,2,8,1,9,3};FYX::sort(l);TEST("list",FYX::check_sorted(l));}
    {std::forward_list<int> f={5,2,8,1,9};FYX::sort(f);TEST("fwd_list",FYX::check_sorted(f));}
    {std::string s="zyxwvutsrqponmlkjihgfedcba";FYX::sort(s);TEST("string",s=="abcdefghijklmnopqrstuvwxyz");}
    {std::bitset<8> b(0b10100011);FYX::sort(b);TEST("bitset asc",b.to_string()=="00001111");}
    {std::bitset<8> b(0b10100011);FYX::sort(b,std::greater<int>{});TEST("bitset desc",b.to_string()=="11110000");}
    {struct S{int x;bool operator<(const S&o)const{return x<o.x;}};std::vector<S> v={{5},{3},{8},{1}};FYX::sort(v);bool ok=true;for(std::size_t i=1;i<v.size();++i)if(v[i].x<v[i-1].x)ok=false;TEST("struct",ok);}
    {std::vector<int> v(100000);for(int i=0;i<100000;++i)v[i]=i%3;FYX::sort(v);TEST("100K mod3",FYX::check_sorted(v));}
    {std::vector<int> v(100000);for(int i=0;i<50000;++i)v[i]=i;for(int i=50000;i<100000;++i)v[i]=100000-i;FYX::sort(v);TEST("100K organ",FYX::check_sorted(v));}
    {std::vector<double> v={1.0/0.0,-1.0/0.0,0.0,-0.0,1.0,-1.0,3.14,-2.72,std::numeric_limits<double>::min(),std::numeric_limits<double>::max()};FYX::sort(v);TEST("dbl special",FYX::check_sorted(v));}
    std::mt19937 rng(42);
    for(int sz:{100,1000,10000,100000}){std::vector<int> v(sz);for(auto&x:v)x=std::uniform_int_distribution<int>(-1000000,1000000)(rng);FYX::sort(v);TEST("random "+std::to_string(sz),FYX::check_sorted(v));}
    {std::vector<int> v(5000000);for(auto&x:v)x=std::uniform_int_distribution<int>(-10000000,10000000)(rng);FYX::sort(v);TEST("5M random",FYX::check_sorted(v));}
}

void performance(){
    std::cout<<"\n=== Performance (v9.2, "<<FYX::threads()<<" threads) ===\n";
    std::cout<<std::fixed<<std::setprecision(2);
    std::mt19937 rng(12345);
    auto report=[](const std::string&l,double f,double s){
        std::cout<<"  "<<std::left<<std::setw(26)<<l<<"FYX:"<<std::right<<std::setw(9)<<f<<" ms  std:"<<std::setw(9)<<s<<" ms  "<<std::setw(6)<<(s/f)<<"x\n";
    };
    auto go=[&](const std::string&label,auto&orig,int runs=3){
        auto v=orig;double f=bench([&]{v=orig;FYX::sort(v);},runs);double s=bench([&]{v=orig;std::sort(v.begin(),v.end());},runs);
        report(label,f,s);v=orig;FYX::sort(v);return FYX::check_sorted(v);
    };
    
    {std::vector<int> o(1000000);for(auto&x:o)x=std::uniform_int_distribution<int>(-500000,500000)(rng);TEST("1M random int",go("1M random int",o));}
    {std::vector<int> o(1000000);std::iota(o.begin(),o.end(),0);go("1M sorted",o);}
    {std::vector<int> o(1000000);for(int i=0;i<1000000;++i)o[i]=1000000-i;go("1M reversed",o);}
    {std::vector<int> o(1000000,42);go("1M identical",o);}
    {std::vector<int> o(1000000);for(auto&x:o)x=std::uniform_int_distribution<int>(0,100)(rng);go("1M small-range",o);}
    {std::vector<int> o(1000000);for(int i=0;i<1000000;++i)o[i]=i%3;go("1M mod3 killer",o);}
    {std::vector<long long> o(1000000);for(auto&x:o)x=std::uniform_int_distribution<long long>(-1000000000000LL,1000000000000LL)(rng);TEST("1M long long",go("1M long long",o));}
    {std::vector<double> o(1000000);std::uniform_real_distribution<double> d(-1e6,1e6);for(auto&x:o)x=d(rng);TEST("1M double",go("1M double",o));}
    {std::vector<float> o(1000000);std::uniform_real_distribution<float> d(-1e6f,1e6f);for(auto&x:o)x=d(rng);TEST("1M float",go("1M float",o));}
    {std::vector<int> o(10000000);for(auto&x:o)x=std::uniform_int_distribution<int>(-5000000,5000000)(rng);TEST("10M random int",go("10M random int",o,2));}
    {std::vector<int> o(1000000);std::iota(o.begin(),o.end(),0);for(int i=0;i<50000;++i){int a=rng()%1000000,b=rng()%1000000;std::swap(o[a],o[b]);}TEST("1M 5% disordered",go("1M 5% disordered",o));}
    {std::vector<int> o(50000000);for(auto&x:o)x=static_cast<int>(rng());TEST("50M random int",go("50M random int",o,1));}
    {std::vector<unsigned> o(1000000);for(auto&x:o)x=rng();TEST("1M unsigned",go("1M unsigned",o));}
    {std::vector<short> o(1000000);for(auto&x:o)x=static_cast<short>(rng()%65536-32768);TEST("1M short",go("1M short",o));}
    {std::vector<int> o(1000000);for(int i=0;i<1000000;++i)o[i]=i%1000;go("1M mod1000",o);}
    {std::vector<int> o(1000000);for(int i=0;i<500000;++i)o[i]=i;for(int i=500000;i<1000000;++i)o[i]=1000000-i;go("1M organ-pipe",o);}
    {std::vector<int> o(1000000);std::iota(o.begin(),o.end(),0);for(int i=0;i<10000;++i){int a=rng()%1000000,b=rng()%1000000;std::swap(o[a],o[b]);}go("1M 1% disordered",o);}
    {std::vector<double> o(10000000);std::uniform_real_distribution<double> d(-1e6,1e6);for(auto&x:o)x=d(rng);TEST("10M double",go("10M double",o,2));}
    {std::vector<int> o(100000000);for(auto&x:o)x=static_cast<int>(rng());TEST("100M random int",go("100M random int",o,1));}
}

void run_all(){
    std::cout<<"============================================\n";
    std::cout<<"  "<<FYX::version()<<"\n";
    std::cout<<"  Platform: "<<FYX::platform()<<"\n";
    std::cout<<"  Arch:     "<<FYX::arch()<<"\n";
    std::cout<<"  Threads:  "<<FYX::threads()<<"\n";
    std::cout<<"============================================\n";
    correctness();performance();
    std::cout<<"\n============================================\n";
    std::cout<<"  Results: "<<pass<<" passed, "<<fail<<" failed\n";
    std::cout<<(fail==0?"  ALL TESTS PASSED!\n":"  SOME TESTS FAILED!\n");
    std::cout<<"============================================\n";
}
}
int main(){test::run_all();return test::fail>0?1:0;}
#endif
