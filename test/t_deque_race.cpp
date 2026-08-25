// Targets the Chase-Lev last-element race: owner pop() vs thief steal()
// contending for a deque holding exactly one task. Correct behaviour: exactly
// one of them wins, never both, never neither.
#include "../fyx_sort.hpp"
#include <cstdio>
#include <atomic>
#include <thread>
#include <vector>
#include <cstdlib>
using namespace fyx::detail;
int main(){
    const long long ROUNDS = 500000;
    WorkStealingDeque d(8);
    std::atomic<bool> stop{false};
    std::atomic<long long> stolen{0};
    std::atomic<long long> round_id{0};
    std::atomic<long long> both{0}, neither{0};
    // per-round claim flags
    std::vector<std::atomic<int>> claim(ROUNDS);
    for(auto&c:claim) c.store(0);
    const int NT=3;
    std::vector<std::thread> th;
    for(int k=0;k<NT;k++) th.emplace_back([&]{
        Task t;
        while(!stop.load(std::memory_order_acquire)){
            if(d.steal(t)==StealStatus::Success){
                long long got=(long long)(uintptr_t)t.arg;
                if(got<0||got>=ROUNDS){ printf("BAD arg from steal: %lld\n",got); std::abort(); }
                if(claim[got].fetch_add(1)!=0) both.fetch_add(1);
                stolen.fetch_add(1);
            }
        }
    });
    long long popped=0;
    for(long long r=0;r<ROUNDS;r++){
        
        round_id.store(r);
        Task x; x.fn=(void(*)(void*))(uintptr_t)0x1000; x.arg=(void*)(uintptr_t)r;
        while(!d.push(x)) ;
        // immediately race pop against the thieves for this single element
        Task t;
        bool got=d.pop(t);
        if(got){
            if((long long)(uintptr_t)t.arg!=r){ printf("BAD value from pop\n"); return 1; }
            if(claim[r].fetch_add(1)!=0) both.fetch_add(1);
            popped++;
        }
        // drain any leftover so each round starts empty
        while(d.pop(t)){ if(claim[(long long)(uintptr_t)t.arg].fetch_add(1)!=0) both.fetch_add(1); popped++; }
    }
    stop.store(true,std::memory_order_release);
    for(auto&x:th) x.join();
    { Task t; while(d.steal(t)==StealStatus::Success) stolen.fetch_add(1); }
    printf("rounds=%lld popped=%lld stolen=%lld  double-claims=%lld\n",
           ROUNDS,popped,stolen.load(),both.load());
    printf("conservation: popped+stolen = %lld (must equal rounds=%lld) : %s\n",
           popped+stolen.load(), ROUNDS, (popped+stolen.load()==ROUNDS)?"OK":"FAIL");
    printf("no double-claim: %s\n", both.load()==0?"OK":"FAIL");
    return (both.load()==0 && popped+stolen.load()==ROUNDS)?0:1;
}
