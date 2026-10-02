// In-process A/B for small n: same data, alternating order, median per-sort
// cycles.  ./ab N DIST  -> fyx and xss median ns/elem.
#include HDR
#include "x86simdsort-static-incl.h"
#include "../../../bench/suite/core/bench_core.hpp"
#include <cstdio>
#include <algorithm>
typedef TY T;
int main(int, char** v) {
    const std::size_t n = std::strtoull(v[1], 0, 10); const int d = std::atoi(v[2]);
    const int S = n > 50000 ? 8 : 64, R = n > 50000 ? 12 : 40;   // distinct inputs, passes
    std::vector<std::vector<T>> src(S);
    for (int i = 0; i < S; ++i) src[i] = fb::make_input<T>(n, (fb::Dist)d, 500 + i);
    std::vector<T> w(n);
    std::vector<double> ta, tb;
    for (int r = 0; r < R; ++r) for (int i = 0; i < S; ++i) for (int k = 0; k < 2; ++k) {
        const bool fyx_turn = ((r + i + k) & 1) == 0;
        std::copy(src[i].begin(), src[i].end(), w.begin());
        auto t0 = std::chrono::steady_clock::now();
        if (fyx_turn) fyx::sort(w.data(), w.data() + n); else x86simdsortStatic::qsort(w.data(), n, false, false);
        double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / n;
        (fyx_turn ? ta : tb).push_back(ns);
    }
    std::nth_element(ta.begin(), ta.begin() + ta.size() / 2, ta.end());
    std::nth_element(tb.begin(), tb.begin() + tb.size() / 2, tb.end());
    std::printf("%-14s n=%zu fyx=%.3f xss=%.3f ratio=%.2f %s\n", fb::dist_name((fb::Dist)d), n, ta[ta.size()/2], tb[tb.size()/2],
                tb[tb.size()/2] / ta[ta.size()/2], tb[tb.size()/2] < ta[ta.size()/2] ? "LOSE" : "");
}
