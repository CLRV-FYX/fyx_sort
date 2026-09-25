// Interleaved serial vs parallel fyx::sort on PSS target shapes.
// Same buffer, warm pages, 5 rounds, report median. Not a canonical matrix.
#include "fyx_sort.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <class T, class Make>
static void row(const char* name, std::size_t n, Make make) {
    std::vector<T> src(n), a(n), b(n);
    make(src);
    // warm
    a = src;
    fyx::Options on; on.parallel = fyx::Tri::On;
    fyx::Options off; off.parallel = fyx::Tri::Off;
    fyx::sort(a.data(), n, on);
    a = src;
    fyx::sort(a.data(), n, off);

    std::vector<double> ts, tp;
    for (int r = 0; r < 5; ++r) {
        a = src;
        auto t0 = clk::now();
        fyx::sort(a.data(), n, off);
        ts.push_back(std::chrono::duration<double>(clk::now() - t0).count());
        b = src;
        auto t1 = clk::now();
        fyx::sort(b.data(), n, on);
        tp.push_back(std::chrono::duration<double>(clk::now() - t1).count());
        if (r == 0) {
            std::vector<T> ref = src;
            std::sort(ref.begin(), ref.end());
            if (a != ref || b != ref) {
                std::fprintf(stderr, "UNSORTED %s n=%zu\n", name, n);
                std::exit(1);
            }
        }
    }
    const double ms = median(ts), mp = median(tp);
    std::printf("%-10s n=%8zu  ser=%8.4fms  par=%8.4fms  par/ser=%.2f\n",
                name, n, ms * 1e3, mp * 1e3, mp / ms);
}

int main() {
    const std::size_t n = std::size_t(1) << 20; // 1M: parallel PSS min is 256K
    std::printf("load: check uptime yourself; n=%zu int64\n", n);

    row<std::int64_t>("random", n, [&](std::vector<std::int64_t>& a) {
        std::mt19937_64 rng(1);
        for (auto& x : a) x = static_cast<std::int64_t>(rng());
    });
    row<std::int64_t>("concat2", n, [&](std::vector<std::int64_t>& a) {
        std::mt19937_64 rng(2);
        const std::size_t m = n / 2;
        for (auto& x : a) x = static_cast<std::int64_t>(rng() >> 2);
        std::sort(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(m));
        std::sort(a.begin() + static_cast<std::ptrdiff_t>(m), a.end());
    });
    row<std::int64_t>("runs3", n, [&](std::vector<std::int64_t>& a) {
        std::mt19937_64 rng(3);
        const std::size_t s = n / 3;
        auto fill = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t i = lo; i < hi; ++i) a[i] = static_cast<std::int64_t>(rng() % (1ull << 40));
            std::sort(a.begin() + static_cast<std::ptrdiff_t>(lo),
                      a.begin() + static_cast<std::ptrdiff_t>(hi));
        };
        fill(0, s); fill(s, 2 * s); fill(2 * s, n);
    });
    row<std::int64_t>("runs8", n, [&](std::vector<std::int64_t>& a) {
        std::mt19937_64 rng(4);
        const std::size_t s = n / 8;
        for (unsigned r = 0; r < 8; ++r) {
            const std::size_t lo = r * s, hi = (r == 7) ? n : (r + 1) * s;
            for (std::size_t i = lo; i < hi; ++i) a[i] = static_cast<std::int64_t>(rng() % (1ull << 40));
            std::sort(a.begin() + static_cast<std::ptrdiff_t>(lo),
                      a.begin() + static_cast<std::ptrdiff_t>(hi));
        }
    });
    row<std::int64_t>("rotated", n, [&](std::vector<std::int64_t>& a) {
        for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<std::int64_t>(i);
        std::rotate(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n / 3), a.end());
    });
    row<std::int64_t>("zigzag", n, [&](std::vector<std::int64_t>& a) {
        for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<std::int64_t>(i);
        std::reverse(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n / 2));
    });
    return 0;
}
