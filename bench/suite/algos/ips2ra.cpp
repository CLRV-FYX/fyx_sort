// IPS2Ra (in-place parallel super scalar radix sort): FB_PAR=0/1.
// Radix sort needs an unsigned key extractor: signed ints flip the sign bit,
// floats use the usual order-preserving bit transform, records use `key`.
#include "common.hpp"
#include "ips2ra.hpp"
#include <cstring>
#ifndef FB_PAR
#  define FB_PAR 0
#endif
namespace {
struct Extract {
    std::uint32_t operator()(std::int32_t x) const { return static_cast<std::uint32_t>(x) ^ 0x80000000u; }
    std::uint32_t operator()(std::uint32_t x) const { return x; }
    std::uint64_t operator()(std::int64_t x) const { return static_cast<std::uint64_t>(x) ^ 0x8000000000000000ull; }
    std::uint64_t operator()(std::uint64_t x) const { return x; }
    std::uint32_t operator()(float x) const {
        std::uint32_t u; std::memcpy(&u, &x, 4);
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }
    std::uint64_t operator()(double x) const {
        std::uint64_t u; std::memcpy(&u, &x, 8);
        return (u & 0x8000000000000000ull) ? ~u : (u | 0x8000000000000000ull);
    }
    std::uint64_t operator()(const fb::KV& x) const { return x.key; }
    std::uint32_t operator()(const fb::Rec& x) const { return x.key; }
};
template <class T> struct ExtractT {
    auto operator()(const T& x) const { return Extract{}(x); }
};
}
struct Algo {
    static constexpr const char* name = FB_PAR ? "ips2ra_par" : "ips2ra";
    static constexpr bool parallel = FB_PAR != 0, stable = false; static constexpr int suites = fb::kU;
    template <class T> static bool sort(T* p, std::size_t n) {
        if constexpr (std::is_same_v<T, std::string>) {
            return false;
        } else {
#if FB_PAR
            ips2ra::parallel::sort(p, p + n, ExtractT<T>{});
#else
            ips2ra::sort(p, p + n, ExtractT<T>{});
#endif
            return true;
        }
    }
};
#include "../core/bench_main.hpp"
