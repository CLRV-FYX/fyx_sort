// Splits the radix sort into its four stages.  Build against the instrumented
// header (see instrument.py):
//
//   python3 tools/dev/instrument.py /tmp/fyxinst
//   g++ -std=c++17 -O2 -march=native -DNDEBUG -pthread \
//       -DFYX_HDR='"/tmp/fyxinst/fyx_sort.hpp"' tools/dev/radix_timer.cpp -o /tmp/rt
//   /tmp/rt
#include FYX_HDR
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

using namespace fyx::detail;

template <class T>
static void run(const char* name, std::vector<T> base, int reps) {
    double best = 1e9, bh = 0, be = 0, bd = 0;
    int passes = 0;
    for (int r = 0; r < reps; ++r) {
        auto v = base;
        RadixTimers::reset();
        (void)radix_sort<T>(v.data(), v.size());
        RadixTimers& t = RadixTimers::get();
        const double per = t.passes ? t.scat / t.passes : 0.0;
        if (per < best) { best = per; bh = t.hist; be = t.enc; bd = t.dec; passes = t.passes; }
        if (!std::is_sorted(v.begin(), v.end())) { std::printf("  !! NOT SORTED: %s\n", name); return; }
    }
    std::printf("  %-16s %2d passes  best %0.5f s/pass   hist %.5f  encode %.5f  decode %.5f\n",
                name, passes, best, bh, be, bd);
}

int main() {
    const std::size_t n = 1000000;
    const int reps = 7;
    std::mt19937_64 g(3);
    { std::vector<std::int32_t> v(n); for (auto& x : v) x = (std::int32_t)g();          run("int32 32-bit", v, reps); }
    { std::vector<std::int32_t> v(n); for (auto& x : v) x = (std::int32_t)(g() & 0x7FF); run("int32 11-bit", v, reps); }
    { std::vector<std::int64_t> v(n); for (auto& x : v) x = (std::int64_t)g();          run("int64 64-bit", v, reps); }
    { std::vector<double> v(n);       for (auto& x : v) x = (double)(g() & 0xFFFFFFF) / 7.0; run("double", v, reps); }
    return 0;
}
