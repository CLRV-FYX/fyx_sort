// Times the individual parallel/serial radix kernels against the whole
// fyx::sort dispatch on the same input, to see what the detector chain costs.
#include "../../fyx_sort.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <chrono>
#include <algorithm>
using namespace std::chrono;
static double now(){return duration<double>(steady_clock::now().time_since_epoch()).count();}
template <class F> double bench(F f,int reps){double b=1e30;for(int r=0;r<reps;++r){double t0=now();f();double t1=now();b=std::min(b,t1-t0);}return b;}
int main(int argc, char** argv){
    std::size_t N = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 8000000;
    const char* want = argc > 2 ? argv[2] : "int32";
    std::mt19937_64 rng(12345);
    const bool i32  = std::strcmp(want, "int32") == 0;
    const bool i64  = std::strcmp(want, "int64") == 0;
    auto run = [&](auto mk) {
        using T = decltype(mk(0));
        std::vector<T> base(N), work(N);
        for (std::size_t i = 0; i < N; ++i) base[i] = mk(i);
        auto ref = base; std::sort(ref.begin(), ref.end());
        fyx::Options off; off.parallel = fyx::Tri::Off;
        fyx::Options on;  on.parallel  = fyx::Tri::On;
        double tsort_s = bench([&]{ work = base; fyx::sort(work.data(), work.size(), off); }, 7);
        double tsort_p = bench([&]{ work = base; fyx::sort(work.data(), work.size(), on); }, 7);
        double hp_s=1e30, hp_p=1e30, w_p=1e30, r_p=1e30, r_s=1e30;
        { auto v=base; if (fyx::detail::try_serial_radix_high_prefix_sort(v.data(), v.size(), false) && v==ref)
            hp_s = bench([&]{ work=base; fyx::detail::try_serial_radix_high_prefix_sort(work.data(), work.size(), false); }, 7); }
        { auto v=base; if (fyx::detail::try_parallel_radix_high_prefix_sort(v.data(), v.size(), false) && v==ref)
            hp_p = bench([&]{ work=base; fyx::detail::try_parallel_radix_high_prefix_sort(work.data(), work.size(), false); }, 7); }
        { auto v=base; if (fyx::detail::try_parallel_radix32_wide_sort(v.data(), v.size(), false) && v==ref)
            w_p = bench([&]{ work=base; fyx::detail::try_parallel_radix32_wide_sort(work.data(), work.size(), false); }, 7); }
        { auto v=base; if (fyx::detail::try_parallel_radix_sort(v.data(), v.size(), false, true) && v==ref)
            r_p = bench([&]{ work=base; fyx::detail::try_parallel_radix_sort(work.data(), work.size(), false, true); }, 7); }
        { auto v=base; if (fyx::detail::radix_sort(v.data(), v.size()) && v==ref)
            r_s = bench([&]{ work=base; fyx::detail::radix_sort(work.data(), work.size()); }, 7); }
        printf("%-6s n=%-9zu sort ser=%.5f par=%.5f | hpfx ser=%.5f par=%.5f | wide32 par=%.5f | lsdradix ser=%.5f par=%.5f\n",
               want, N, tsort_s, tsort_p, hp_s, hp_p, w_p, r_s, r_p);
        fflush(stdout);
    };
    if (i32)      run([&](std::size_t){ return std::int32_t(rng()); });
    else if (i64) run([&](std::size_t){ return std::int64_t(rng()); });
    else          run([&](std::size_t){ return double(std::int64_t(rng()) >> 11) * 0.25; });
    // fill with random data in the generator above
    return 0;
}
