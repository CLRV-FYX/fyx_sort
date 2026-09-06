// Which low-cardinality routine actually handles an input, and what each one
// costs.  Timed with the copy excluded, one binary, -O3.
#include "../../fyx_sort.hpp"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <chrono>
#include <algorithm>
using namespace std::chrono;
static double now(){return duration<double>(steady_clock::now().time_since_epoch()).count();}
template <class F> double bench(F f,int reps){double b=1e30;for(int r=0;r<reps;++r){double t0=now();f();double t1=now();b=std::min(b,t1-t0);}return b;}
template <class T, class Gen>
void probe(const char* tag, Gen gen, std::size_t N) {
    std::vector<T> base(N), work(N);
    for (std::size_t i = 0; i < N; ++i) base[i] = gen(i);
    auto ref = base; std::sort(ref.begin(), ref.end());
    fyx::Options off; off.parallel = fyx::Tri::Off;
    fyx::Options on;  on.parallel  = fyx::Tri::On;
    // each measurement warms the copy first, then times only the sort
    auto t = [&](auto call) {
        auto v = base;
        if (!call(v.data(), v.size()) || v != ref) return 1e30;
        double b = 1e30;
        for (int r = 0; r < 9; ++r) {
            work = base;
            double t0 = now(); call(work.data(), work.size()); double t1 = now();
            b = std::min(b, t1 - t0);
        }
        return b;
    };
    auto tsort = [&](const fyx::Options& o) {
        double b = 1e30;
        for (int r = 0; r < 9; ++r) {
            work = base;
            double t0 = now(); fyx::sort(work.data(), work.size(), o); double t1 = now();
            b = std::min(b, t1 - t0);
        }
        return b;
    };
    double tsort_s = tsort(off), tsort_p = tsort(on);
    double t_is  = t([](T* p, std::size_t n){ return fyx::detail::try_integer_sparse_count_sort(p, n, false); });
    double t_rs  = t([](T* p, std::size_t n){ return fyx::detail::try_radix_key_sparse_count_sort(p, n, false); });
    double t_ip  = t([](T* p, std::size_t n){ return fyx::detail::try_integer_sparse_count_sort_parallel(p, n, false); });
    double t_rp  = t([](T* p, std::size_t n){ return fyx::detail::try_radix_key_sparse_count_sort_parallel(p, n, false); });
    double t_sa  = t([](T* p, std::size_t n){ return fyx::detail::sample_arithmetic_sparse_count_sort(p, p + n, fyx::less{}); });
    double t_lc  = t([](T* p, std::size_t n){ return fyx::detail::try_low_cardinality_count_sort(p, p + n, fyx::less{}); });
    // eight separate buffers: printf would otherwise see the same pointer
    // eight times and print the last value in every column
    static char bufs[8][16]; int nb = 0;
    auto fmt = [&](double x){ char* b = bufs[nb++ & 7]; if (x > 1e29) std::snprintf(b,16,"   --   "); else std::snprintf(b,16,"%.5f",x); return b; };
    printf("%-24s sort ser=%s par=%s | int=%s radix=%s intP=%s radixP=%s sample=%s lowcard=%s\n",
           tag, fmt(tsort_s), fmt(tsort_p), fmt(t_is), fmt(t_rs), fmt(t_ip), fmt(t_rp), fmt(t_sa), fmt(t_lc));
    fflush(stdout);
}
int main(int argc, char** argv){
    const std::size_t N = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
    std::mt19937_64 rng(7);
    probe<std::int32_t>("int32 mod8 periodic",   [](std::size_t i){ return std::int32_t(i % 8); }, N);
    probe<std::int64_t>("int64 mod8 periodic",   [](std::size_t i){ return std::int64_t(i % 8); }, N);
    probe<std::int32_t>("int32 lowcard16",       [&](std::size_t){ return std::int32_t(rng() % 16); }, N);
    probe<std::int64_t>("int64 lowcard16",       [&](std::size_t){ return std::int64_t(rng() % 16); }, N);
    probe<std::int64_t>("int64 lowcard16 spread",[&](std::size_t){ return std::int64_t(rng() % 16) << 24; }, N);
    return 0;
}
