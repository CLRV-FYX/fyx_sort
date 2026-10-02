// fyx::sort serial (FB_FYX_MODE=0), parallel (1), fyx::stable_sort (2).
// Built twice by the runner: "native" (best ISA flags) and "portable"
// (no -march / no /arch) -- the latter is fyx's least favourable setting.
#include "fyx_sort.hpp"
#include "common.hpp"
#ifndef FB_FYX_MODE
#  define FB_FYX_MODE 0
#endif
struct Algo {
#if FB_FYX_MODE == 0
    static constexpr const char* name = "fyx";
    static constexpr bool parallel = false, stable = false;
    static constexpr int suites = fb::kU;
#elif FB_FYX_MODE == 1
    static constexpr const char* name = "fyx_par";
    static constexpr bool parallel = true, stable = false;
    static constexpr int suites = fb::kU;
#else
    static constexpr const char* name = "fyx_stable";
    static constexpr bool parallel = false, stable = true;
    static constexpr int suites = fb::kS;
#endif
    template <class T> static bool sort(T* p, std::size_t n) {
#if FB_FYX_MODE == 2
        if constexpr (fb::is_record_v<T>) fyx::stable_sort(p, n, fb::Less<T>{});
        else fyx::stable_sort(p, n);
#else
        fyx::Options o;
        o.parallel = FB_FYX_MODE == 1 ? fyx::Tri::On : fyx::Tri::Off;
        if constexpr (fb::is_record_v<T>) fyx::sort(p, n, fb::Less<T>{}, o);
        else fyx::sort(p, n, o);
#endif
        return true;
    }
};
#include "../core/bench_main.hpp"
