// Capped natural-merge (PSS) shapes: serial + parallel, int64/int32/double.
#include "../fyx_sort.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

static int checks = 0, fails = 0;
#define CHECK(c, m) do { ++checks; if (!(c)) { std::printf("FAIL %s\n", m); ++fails; } } while (0)

template <class T, class Comp>
static void once(const char* tag, std::vector<T> a, Comp comp, fyx::Tri par) {
    std::vector<T> ref = a;
    std::sort(ref.begin(), ref.end(), comp);
    fyx::Options o;
    o.parallel = par;
    fyx::sort(a.data(), a.size(), comp, o);
    CHECK(a == ref, tag);
}

template <class T>
static void shapes(std::size_t n) {
    std::mt19937_64 rng(n * 91 + sizeof(T));
    auto rnd = [&]() -> T {
        if constexpr (std::is_floating_point<T>::value)
            return static_cast<T>(static_cast<int64_t>(rng() >> 3) * 1e-6);
        else
            return static_cast<T>(rng() >> 2);
    };
    for (int mode = 0; mode < 2; ++mode) {
        fyx::Tri par = mode ? fyx::Tri::On : fyx::Tri::Off;
        const char* ms = mode ? "par" : "ser";
        {
            std::vector<T> a(n);
            for (auto& x : a) x = rnd();
            once<T>( (std::string("random ")+ms).c_str(), a, std::less<T>{}, par);
        }
        {
            std::vector<T> a(n);
            for (auto& x : a) x = rnd();
            const auto m = n / 2;
            std::sort(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(m));
            std::sort(a.begin() + static_cast<std::ptrdiff_t>(m), a.end());
            once<T>((std::string("concat2 ")+ms).c_str(), a, std::less<T>{}, par);
        }
        {
            std::vector<T> a(n);
            const auto s = n / 8;
            for (unsigned r = 0; r < 8; ++r) {
                auto lo = r * s, hi = (r == 7) ? n : (r + 1) * s;
                for (auto i = lo; i < hi; ++i) a[i] = rnd();
                std::sort(a.begin() + static_cast<std::ptrdiff_t>(lo),
                          a.begin() + static_cast<std::ptrdiff_t>(hi));
            }
            once<T>((std::string("runs8 ")+ms).c_str(), a, std::less<T>{}, par);
        }
        {
            std::vector<T> a(n);
            for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<T>(i);
            std::rotate(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n / 3), a.end());
            once<T>((std::string("rotated ")+ms).c_str(), a, std::less<T>{}, par);
        }
        {
            std::vector<T> a(n);
            for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<T>(i);
            std::reverse(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n / 2));
            once<T>((std::string("zigzag ")+ms).c_str(), a, std::less<T>{}, par);
        }
        {
            std::vector<T> a(n);
            for (auto& x : a) x = rnd();
            const auto m = n / 2;
            std::sort(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(m), std::greater<T>());
            std::sort(a.begin() + static_cast<std::ptrdiff_t>(m), a.end(), std::greater<T>());
            once<T>((std::string("greater_concat2 ")+ms).c_str(), a, std::greater<T>{}, par);
        }
    }
}

int main() {
    for (std::size_t n : {std::size_t(4096), std::size_t(1) << 18, std::size_t(1) << 20}) {
        shapes<std::int64_t>(n);
        shapes<std::int32_t>(n);
        if (n <= (std::size_t(1) << 18)) shapes<double>(n);
    }
    std::printf("t_pss checks=%d fails=%d\n", checks, fails);
    return fails ? 1 : 0;
}
