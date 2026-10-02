// IPS4o: sequential (FB_PAR=0) and parallel std::thread backend (FB_PAR=1).
#include "common.hpp"
#include "ips4o.hpp"
#include <functional>
#ifndef FB_PAR
#  define FB_PAR 0
#endif
struct Algo {
    static constexpr const char* name = FB_PAR ? "ips4o_par" : "ips4o";
    static constexpr bool parallel = FB_PAR != 0, stable = false; static constexpr int suites = fb::kU;
    template <class T> static bool sort(T* p, std::size_t n) {
#if FB_PAR
        if constexpr (fb::is_record_v<T>) ips4o::parallel::sort(p, p + n, fb::Less<T>{});
        else ips4o::parallel::sort(p, p + n, std::less<T>());
#else
        if constexpr (fb::is_record_v<T>) ips4o::sort(p, p + n, fb::Less<T>{});
        else ips4o::sort(p, p + n, std::less<T>());
#endif
        return true;
    }
};
#include "../core/bench_main.hpp"
