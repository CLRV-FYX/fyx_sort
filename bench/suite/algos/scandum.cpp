// scandum crumsort (unstable) / fluxsort / quadsort (stable) via a C wrapper.
#include "common.hpp"
extern "C" {
void fb_scandum_i32(void*, std::size_t); void fb_scandum_u32(void*, std::size_t);
void fb_scandum_i64(void*, std::size_t); void fb_scandum_u64(void*, std::size_t);
void fb_scandum_f32(void*, std::size_t); void fb_scandum_f64(void*, std::size_t);
void fb_scandum_rec(void*, std::size_t);
}
#ifndef FB_SCANDUM
#  define FB_SCANDUM 0
#endif
struct Algo {
#if FB_SCANDUM == 0
    static constexpr const char* name = "crumsort"; static constexpr bool parallel = false, stable = false; static constexpr int suites = fb::kU;
#elif FB_SCANDUM == 1
    static constexpr const char* name = "fluxsort"; static constexpr bool parallel = false, stable = true; static constexpr int suites = fb::kBoth;
#else
    static constexpr const char* name = "quadsort"; static constexpr bool parallel = false, stable = true; static constexpr int suites = fb::kS;
#endif
    template <class T> static bool sort(T* p, std::size_t n) {
        if constexpr (std::is_same_v<T, std::int32_t>) fb_scandum_i32(p, n);
        else if constexpr (std::is_same_v<T, std::uint32_t>) fb_scandum_u32(p, n);
        else if constexpr (std::is_same_v<T, std::int64_t>) fb_scandum_i64(p, n);
        else if constexpr (std::is_same_v<T, std::uint64_t>) fb_scandum_u64(p, n);
        else if constexpr (std::is_same_v<T, float>) fb_scandum_f32(p, n);
        else if constexpr (std::is_same_v<T, double>) fb_scandum_f64(p, n);
        else if constexpr (std::is_same_v<T, fb::Rec>) fb_scandum_rec(p, n);
        else return false;
        return true;
    }
};
#include "../core/bench_main.hpp"
