// tools/dev/timer.cpp -- fyx::sort throughput, serial against parallel.
//
// Best of `reps`, the copy out of the timed region, sortedness asserted, one
// size and one element type per run so the numbers stay comparable.
//
//     g++ -std=c++17 -O3 -march=native -DNDEBUG -pthread -I. tools/dev/timer.cpp -o /tmp/timer
//     /tmp/timer 1000000
#include "../../fyx_sort.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <chrono>
#include <algorithm>
using namespace std::chrono;
static double now(){ return duration<double>(steady_clock::now().time_since_epoch()).count(); }

template <class T>
void run(const char* name, std::size_t n, int reps) {
    std::mt19937_64 rng(12345);
    std::vector<T> base(n), v(n);
    for (std::size_t i = 0; i < n; ++i) base[i] = T(std::uint32_t(rng()));
    fyx::Options off; off.parallel = fyx::Tri::Off;
    fyx::Options on;  on.parallel  = fyx::Tri::On;
    double bestOff = 1e30, bestOn = 1e30, copyT = 1e30;
    bool ok = true;
    for (int r = 0; r <= reps; ++r) {
        double c0 = now(); std::copy(base.begin(), base.end(), v.begin()); double c1 = now();
        if (r) copyT = std::min(copyT, c1 - c0);
        double t0 = now(); fyx::sort(v.data(), v.size(), off); double t1 = now();
        if (r) bestOff = std::min(bestOff, t1 - t0);
        if (!std::is_sorted(v.begin(), v.end())) ok = false;
        std::copy(base.begin(), base.end(), v.begin());
        double t2 = now(); fyx::sort(v.data(), v.size(), on); double t3 = now();
        if (r) bestOn = std::min(bestOn, t3 - t2);
        if (!std::is_sorted(v.begin(), v.end())) ok = false;
    }
    std::printf("%-7s n=%-9zu ser=%.5f  par=%.5f  (gain %.2fx, copy %.5f) %s\n",
                name, n, bestOff, bestOn, bestOff / bestOn, copyT, ok ? "" : "NOT SORTED");
    std::fflush(stdout);
}
int main(int argc, char** argv) {
    std::size_t n = argc > 1 ? std::strtoull(argv[1], 0, 10) : 1000000;
    run<std::int32_t>("int32", n, 5);
    run<std::int64_t>("int64", n, 5);
    run<double>("double", n, 5);
    return 0;
}
