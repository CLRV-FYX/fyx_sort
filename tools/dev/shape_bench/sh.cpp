// Per-shape single-thread ns/elem: fyx (HDR) or xss (-DXSS).  One process per
// contender (same-binary timing skews).  Usage: sh N [dist] -> "shape ns" lines.
#ifdef XSS
#include "x86simdsort-static-incl.h"
#else
#include HDR
#endif
#include "../../../bench/suite/core/bench_core.hpp"
#include <cstdio>
#include <vector>
#include <chrono>
typedef TY T;
int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20000;
    const int only = argc > 2 ? std::atoi(argv[2]) : -1;
    const std::size_t B = std::max<std::size_t>(1, 65536 / n);   // batch small n
    for (int d = 0; d < 16; ++d) {
        if (only >= 0 && d != only) continue;
        double best = 1e30;
        const int rounds = n >= 1000000 ? 7 : n >= 100000 ? 41 : 15;
        for (int r = 0; r < rounds; ++r) {
            std::vector<std::vector<T>> v(B);
            for (std::size_t b = 0; b < B; ++b) v[b] = fb::make_input<T>(n, (fb::Dist)d, 1000 + r * 131 + b);
            auto t0 = std::chrono::steady_clock::now();
            for (std::size_t b = 0; b < B; ++b) {
#ifdef XSS
                x86simdsortStatic::qsort(v[b].data(), n, false, false);
#else
                fyx::sort(v[b].begin(), v[b].end());
#endif
            }
            double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(n * B);
            best = std::min(best, ns);
            for (std::size_t b = 0; b < B; ++b)
                if (!std::is_sorted(v[b].begin(), v[b].end(), [](T a, T c){ return a < c; })) { std::printf("UNSORTED %d\n", d); return 1; }
        }
        std::printf("%-14s %.3f\n", fb::dist_name((fb::Dist)d), best);
    }
}
