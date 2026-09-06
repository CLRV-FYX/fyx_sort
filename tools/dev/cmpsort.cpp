#include "../../fyx_sort.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 8000000;
    std::vector<std::int32_t> base(n), w(n), ref(n);
    std::mt19937 rng(4242);
    for (std::size_t i = 0; i < n; ++i) base[i] = std::int32_t(rng());
    ref = base; std::sort(ref.begin(), ref.end());
    auto bench = [&](const char* name, auto&& fn) {
        double best = 1e30;
        for (int r = 0; r < 3; ++r) {
            w = base;
            auto t0 = std::chrono::steady_clock::now();
            fn(w.data(), n);
            double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            best = std::min(best, dt);
            if (w != ref) std::printf("%s WRONG\n", name);
        }
        std::printf("%-14s n=%9zu  %8.5f s\n", name, n, best);
    };
    bench("sort(radix)",  [](std::int32_t* q, std::size_t m) { fyx::sort(q, m); });
    bench("sample_sort",  [](std::int32_t* q, std::size_t m) { fyx::stable_sort(q, m); });
    return 0;
}
