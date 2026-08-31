// tools/dev/correct.cpp -- sortedness and stability of the radix paths.
//
// Exercises the wide high-prefix kernel the way the benchmarks do not: with
// ties in the high bits, with 64-bit keys, with floats and negatives, serial
// and parallel, and checks the result element by element against std::sort.
//
//     g++ -std=c++17 -O2 -march=native -pthread -I. tools/dev/correct.cpp -o /tmp/correct
#include "../../fyx_sort.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

static int g_fail = 0;
template <class T>
void check(const char* tag, std::vector<T> v, bool verbose = false) {
    std::vector<T> ref = v;
    std::sort(ref.begin(), ref.end());
    fyx::Options off; off.parallel = fyx::Tri::Off;
    fyx::Options on;  on.parallel  = fyx::Tri::On;
    auto a = v; fyx::sort(a.data(), a.size(), off);
    auto b = v; fyx::sort(b.data(), b.size(), on);
    bool ok = (a == ref) && (b == ref);
    if (!ok) {
        ++g_fail;
        std::size_t i = 0;
        while (i < v.size() && a[i] == ref[i]) ++i;
        std::printf("  MISMATCH %-28s n=%zu at %zu\n", tag, v.size(), i);
    } else if (verbose) {
        std::printf("  ok %-28s n=%zu\n", tag, v.size());
    }
    std::fflush(stdout);
}
int main() {
    std::mt19937_64 rng(2024);
    const std::size_t N = 1u << 20;
    std::printf("radix correctness, N=%zu\n", N);
    for (int mode = 0; mode < 6; ++mode) {
        std::vector<std::int32_t> v(N);
        for (std::size_t i = 0; i < N; ++i) {
            const std::uint32_t r = std::uint32_t(rng());
            switch (mode) {
                case 0: v[i] = std::int32_t(r); break;                       // uniform
                case 1: v[i] = std::int32_t(r & 0x0000ffff); break;          // many ties up high
                case 2: v[i] = std::int32_t(r & 0x000000ff); break;          // very few prefixes
                case 3: v[i] = std::int32_t(r) - (std::int32_t)(r & 0xffff); break;
                case 4: v[i] = 42; break;                                    // all equal
                case 5: v[i] = std::int32_t(r) | 1; break;                   // odd only
            }
        }
        char tag[64]; std::snprintf(tag, sizeof tag, "int32 mode=%d", mode);
        check(tag, v, true);
    }
    { std::vector<std::uint32_t> v(N); for (auto& x : v) x = std::uint32_t(rng()) & 0xffff; check("uint32 narrow", v, true); }
    { std::vector<std::uint32_t> v(N); for (auto& x : v) x = std::uint32_t(rng());           check("uint32 full", v, true); }
    { std::vector<std::int64_t>  v(N); for (auto& x : v) x = std::int64_t(rng());            check("int64 full", v, true); }
    { std::vector<std::int64_t>  v(N); for (auto& x : v) x = std::int64_t(rng() & 0xffffff); check("int64 narrow", v, true); }
    { std::vector<std::uint64_t> v(N); for (auto& x : v) x = rng();                          check("uint64 full", v, true); }
    { std::vector<double> v(N); for (auto& x : v) x = double(std::int64_t(rng() & 0xffffff)) * 0.5; check("double narrow", v, true); }
    { std::vector<double> v(N); for (auto& x : v) { std::int64_t b = std::int64_t(rng()); std::double_t d; std::memcpy(&d, &b, 8);
                                                    if (std::isnan(d)) d = 0; x = double(d); } check("double full", v, true); }
    { std::vector<float> v(N); for (auto& x : v) x = float(std::int64_t(rng() & 0xfffff)) * 0.25f; check("float narrow", v, true); }
    { std::vector<float> v(N); for (auto& x : v) { std::uint32_t b = std::uint32_t(rng()); float f; std::memcpy(&f, &b, 4);
                                                   if (std::isnan(f)) f = 0; x = f; } check("float full", v, true); }
    // sizes that straddle the avx512/write-combining scatter crossover (8 MB)
    for (std::size_t n : {std::size_t(1) << 19, std::size_t(1) << 20, std::size_t(1500000),
                          std::size_t(1) << 21, std::size_t(1) << 22}) {
        std::vector<std::int32_t> v(n); for (auto& x : v) x = std::int32_t(rng());
        char tag[64]; std::snprintf(tag, sizeof tag, "int32 n=%zu (crossover)", n);
        check(tag, v, true);
        std::vector<double> w(n); for (auto& x : w) x = double(std::int64_t(rng() >> 3));
        std::snprintf(tag, sizeof tag, "double n=%zu (crossover)", n);
        check(tag, w, true);
    }
    std::printf(g_fail ? "FAILURES: %d\n" : "all correct (%d failures)\n", g_fail);
    return g_fail ? 1 : 0;
}
