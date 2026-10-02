// Standard library: std::sort (FB_STD_MODE=0), std::stable_sort (1),
// std::sort(std::execution::par) (2), std::stable_sort(par) (3).
#include "common.hpp"
#ifndef FB_STD_MODE
#  define FB_STD_MODE 0
#endif
#if FB_STD_MODE >= 2
#  include <execution>
#endif
struct Algo {
#if FB_STD_MODE == 0
    static constexpr const char* name = "std_sort";
    static constexpr bool parallel = false, stable = false; static constexpr int suites = fb::kU;
#elif FB_STD_MODE == 1
    static constexpr const char* name = "std_stable";
    static constexpr bool parallel = false, stable = true; static constexpr int suites = fb::kS;
#elif FB_STD_MODE == 2
    static constexpr const char* name = "std_par";
    static constexpr bool parallel = true, stable = false; static constexpr int suites = fb::kU;
#else
    static constexpr const char* name = "std_par_stable";
    static constexpr bool parallel = true, stable = true; static constexpr int suites = fb::kS;
#endif
    template <class T> static bool sort(T* p, std::size_t n) {
#if FB_STD_MODE == 0
        std::sort(p, p + n, fb::Less<T>{});
#elif FB_STD_MODE == 1
        std::stable_sort(p, p + n, fb::Less<T>{});
#elif FB_STD_MODE == 2
        std::sort(std::execution::par, p, p + n, fb::Less<T>{});
#else
        std::stable_sort(std::execution::par, p, p + n, fb::Less<T>{});
#endif
        return true;
    }
};
#include "../core/bench_main.hpp"
