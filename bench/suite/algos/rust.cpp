// Rust std sorts through the staticlib built from rust_sorts.rs.
// FB_RUST_STABLE=0: sort_unstable (ipnsort); 1: sort (driftsort).
#include "common.hpp"
#ifndef FB_RUST_STABLE
#  define FB_RUST_STABLE 0
#endif
#define FB_DECL(kind, sfx, T) extern "C" void fb_rust_##kind##_##sfx(T*, std::size_t);
#define FB_ALL(kind) FB_DECL(kind, i32, std::int32_t) FB_DECL(kind, u32, std::uint32_t) \
    FB_DECL(kind, i64, std::int64_t) FB_DECL(kind, u64, std::uint64_t) FB_DECL(kind, f32, float) \
    FB_DECL(kind, f64, double) FB_DECL(kind, kv, fb::KV) FB_DECL(kind, rec, fb::Rec)
FB_ALL(unstable)
FB_ALL(stable)
#if FB_RUST_STABLE
#  define FB_K stable
#else
#  define FB_K unstable
#endif
#define FB_CAT2(a, b, c) a##b##c
#define FB_CAT(a, b, c) FB_CAT2(a, b, c)
#define FB_CALL(sfx) FB_CAT(fb_rust_, FB_K, _##sfx)(p, n)
struct Algo {
    static constexpr const char* name = FB_RUST_STABLE ? "rust_stable" : "rust_unstable";
    static constexpr bool parallel = false, stable = FB_RUST_STABLE != 0;
    static constexpr int suites = FB_RUST_STABLE ? fb::kS : fb::kU;
    template <class T> static bool sort(T* p, std::size_t n) {
        if constexpr (std::is_same_v<T, std::int32_t>) FB_CALL(i32);
        else if constexpr (std::is_same_v<T, std::uint32_t>) FB_CALL(u32);
        else if constexpr (std::is_same_v<T, std::int64_t>) FB_CALL(i64);
        else if constexpr (std::is_same_v<T, std::uint64_t>) FB_CALL(u64);
        else if constexpr (std::is_same_v<T, float>) FB_CALL(f32);
        else if constexpr (std::is_same_v<T, double>) FB_CALL(f64);
        else if constexpr (std::is_same_v<T, fb::KV>) FB_CALL(kv);
        else if constexpr (std::is_same_v<T, fb::Rec>) FB_CALL(rec);
        else return false;
        return true;
    }
};
#include "../core/bench_main.hpp"
