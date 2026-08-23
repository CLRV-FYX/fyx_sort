#include "../fyx_sort.hpp"
#include <cstdio>
#include <vector>
#include <atomic>
#include <thread>
#include <random>
#include <algorithm>
using namespace fyx::detail;
#if FYX_ENABLE_PARALLEL
// 1. Deque single-thread semantics
static bool test_deque_serial(){
    WorkStealingDeque d(4);
    Task t;
    if(d.pop(t)) return false;              // empty pop
    for(int i=0;i<1000;i++){                // forces several growths
        Task x; x.fn=(void(*)(void*))(uintptr_t)0x1000; x.arg=(void*)(uintptr_t)i;
        if(!d.push(x)) return false;
    }
    for(int i=999;i>=0;i--){                // LIFO order
        if(!d.pop(t)) return false;
        if((int)(uintptr_t)t.arg != i) return false;
    }
    if(d.pop(t)) return false;
    return true;
}
// 2. Concurrent steal: every pushed task taken exactly once
static bool test_deque_concurrent(){
    const int N=200000;
    WorkStealingDeque d(6);
    std::vector<std::atomic<int>> seen(N);
    for(int i=0;i<N;i++) seen[i].store(0);
    std::atomic<bool> go{false}, done{false};
    std::atomic<int> stolen{0}, popped{0};
    const int NT=4;
    std::vector<std::thread> th;
    for(int k=0;k<NT;k++) th.emplace_back([&]{
        while(!go.load()) ;
        Task t;
        while(!done.load(std::memory_order_acquire)){
            StealStatus s=d.steal(t);
            if(s==StealStatus::Success){ seen[(int)(uintptr_t)t.arg].fetch_add(1); stolen.fetch_add(1); }
        }
        Task t2; while(d.steal(t2)==StealStatus::Success){ seen[(int)(uintptr_t)t2.arg].fetch_add(1); stolen.fetch_add(1);} 
    });
    go.store(true);
    for(int i=0;i<N;i++){
        Task x; x.fn=(void(*)(void*))(uintptr_t)0x1000; x.arg=(void*)(uintptr_t)i;
        while(!d.push(x)) ;
        Task t;
        if((i&3)==0 && d.pop(t)){ seen[(int)(uintptr_t)t.arg].fetch_add(1); popped.fetch_add(1); }
    }
    { Task t; while(d.pop(t)){ seen[(int)(uintptr_t)t.arg].fetch_add(1); popped.fetch_add(1);} }
    done.store(true,std::memory_order_release);
    for(auto&x:th) x.join();
    { Task t; while(d.steal(t)==StealStatus::Success){ seen[(int)(uintptr_t)t.arg].fetch_add(1); stolen.fetch_add(1);} }
    for(int i=0;i<N;i++) if(seen[i].load()!=1){ printf("  task %d taken %d times\n",i,seen[i].load()); return false; }
    printf("  stolen=%d popped=%d total=%d\n",stolen.load(),popped.load(),stolen.load()+popped.load());
    return true;
}
// 3. fork_join correctness under nesting
static void rec_sum(long long lo, long long hi, long long* out){
    if(hi-lo<=1000){ long long s=0; for(long long i=lo;i<hi;i++) s+=i; *out=s; return; }
    long long mid=(lo+hi)/2, a=0,b=0;
    fork_join([&]{rec_sum(lo,mid,&a);},[&]{rec_sum(mid,hi,&b);});
    *out=a+b;
}
#endif
int main(){
#if FYX_ENABLE_PARALLEL
    printf("pool workers=%u available=%d\n", global_pool().nworkers(), (int)parallel_available());
    printf("deque serial     : %s\n", test_deque_serial()?"OK":"FAIL");
    printf("deque concurrent : %s\n", test_deque_concurrent()?"OK":"FAIL");
    long long got=0; rec_sum(0,10000000,&got);
    long long want=0; for(long long i=0;i<10000000;i++) want+=i;
    printf("fork_join nested : %s (%lld)\n", got==want?"OK":"FAIL", got);
#else
    printf("parallel disabled\n");
#endif
    return 0;
}
