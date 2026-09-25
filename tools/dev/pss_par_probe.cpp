// Correctness probe for parallel capped natural-merge. Not part of the suite.
#include "fyx_sort.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

static bool same(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b) {
    return a == b;
}

template <class Make>
static bool check(const char* name, std::size_t n, Make make) {
    std::vector<std::int64_t> a(n), ref(n);
    make(a);
    ref = a;
    std::sort(ref.begin(), ref.end());
    fyx::Options o;
    o.parallel = fyx::Tri::On;
    fyx::sort(a.data(), n, o);
    if (!same(a, ref)) {
        std::fprintf(stderr, "FAIL %s n=%zu\n", name, n);
        return false;
    }
    std::printf("ok %s n=%zu\n", name, n);
    return true;
}

int main() {
    bool ok = true;
    const std::size_t ns[] = {std::size_t(4096), std::size_t(1) << 18, std::size_t(1) << 20};
    for (std::size_t n : ns) {
        ok &= check("random", n, [&](std::vector<std::int64_t>& a) {
            std::mt19937_64 rng(n * 17 + 3);
            for (auto& x : a) x = static_cast<std::int64_t>(rng());
        });
        ok &= check("concat2", n, [&](std::vector<std::int64_t>& a) {
            std::mt19937_64 rng(n * 19 + 1);
            const std::size_t m = n / 2;
            for (std::size_t i = 0; i < m; ++i) a[i] = static_cast<std::int64_t>(rng() >> 2);
            for (std::size_t i = m; i < n; ++i) a[i] = static_cast<std::int64_t>(rng() >> 2);
            std::sort(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(m));
            std::sort(a.begin() + static_cast<std::ptrdiff_t>(m), a.end());
        });
        ok &= check("runs3", n, [&](std::vector<std::int64_t>& a) {
            std::mt19937_64 rng(n * 23 + 7);
            const std::size_t s = n / 3;
            auto fill = [&](std::size_t lo, std::size_t hi, std::int64_t bias) {
                for (std::size_t i = lo; i < hi; ++i)
                    a[i] = bias + static_cast<std::int64_t>(rng() % (1ull << 40));
                std::sort(a.begin() + static_cast<std::ptrdiff_t>(lo),
                          a.begin() + static_cast<std::ptrdiff_t>(hi));
            };
            fill(0, s, 0);
            fill(s, 2 * s, 0);
            fill(2 * s, n, 0);
        });
        ok &= check("runs8", n, [&](std::vector<std::int64_t>& a) {
            std::mt19937_64 rng(n * 29 + 11);
            const std::size_t s = n / 8;
            for (unsigned r = 0; r < 8; ++r) {
                const std::size_t lo = r * s;
                const std::size_t hi = (r == 7) ? n : (r + 1) * s;
                for (std::size_t i = lo; i < hi; ++i)
                    a[i] = static_cast<std::int64_t>(rng() % (1ull << 40));
                std::sort(a.begin() + static_cast<std::ptrdiff_t>(lo),
                          a.begin() + static_cast<std::ptrdiff_t>(hi));
            }
        });
        ok &= check("rotated", n, [&](std::vector<std::int64_t>& a) {
            for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<std::int64_t>(i);
            std::rotate(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n / 3), a.end());
        });
        ok &= check("desc_concat2", n, [&](std::vector<std::int64_t>& a) {
            std::mt19937_64 rng(n * 41 + 3);
            const std::size_t m = n / 2;
            for (auto& x : a) x = static_cast<std::int64_t>(rng() >> 2);
            std::sort(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(m), std::greater<std::int64_t>());
            std::sort(a.begin() + static_cast<std::ptrdiff_t>(m), a.end(), std::greater<std::int64_t>());
        });
        // force descending comparator
        {
            std::vector<std::int64_t> a(n), ref(n);
            std::mt19937_64 rng(n * 43);
            const std::size_t m = n / 2;
            for (auto& x : a) x = static_cast<std::int64_t>(rng() >> 2);
            std::sort(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(m), std::greater<std::int64_t>());
            std::sort(a.begin() + static_cast<std::ptrdiff_t>(m), a.end(), std::greater<std::int64_t>());
            ref = a;
            std::sort(ref.begin(), ref.end(), std::greater<std::int64_t>());
            fyx::Options o;
            o.parallel = fyx::Tri::On;
            fyx::sort(a.data(), n, std::greater<std::int64_t>(), o);
            if (a != ref) {
                std::fprintf(stderr, "FAIL greater concat2 n=%zu\n", n);
                ok = false;
            } else {
                std::printf("ok greater_concat2 n=%zu\n", n);
            }
        }
        ok &= check("neg_runs3", n, [&](std::vector<std::int64_t>& a) {
            std::mt19937_64 rng(n * 31);
            const std::size_t s = n / 3;
            for (unsigned r = 0; r < 3; ++r) {
                const std::size_t lo = r * s;
                const std::size_t hi = (r == 2) ? n : (r + 1) * s;
                for (std::size_t i = lo; i < hi; ++i)
                    a[i] = -static_cast<std::int64_t>(rng() % 1000000);
                std::sort(a.begin() + static_cast<std::ptrdiff_t>(lo),
                          a.begin() + static_cast<std::ptrdiff_t>(hi));
            }
        });
    }
    std::puts(ok ? "ALL OK" : "SOME FAILED");
    return ok ? 0 : 1;
}
