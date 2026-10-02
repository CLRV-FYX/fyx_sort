// fyx benchmark suite -- shared data generation, timing, verification, CSV.
//
// Every algorithm is compiled into its OWN executable (one translation unit
// per competitor).  That keeps each library's preferred compiler flags,
// prevents ODR clashes between differently-compiled copies of fyx, and lets a
// competitor that fails to build on some platform drop out without taking the
// others down.  All executables generate bit-identical inputs from the same
// seed (portable splitmix64; no std::*_distribution, whose output differs
// between standard libraries).
//
// An algorithm translation unit defines:
//
//   struct Algo {
//       static constexpr const char* name = "...";
//       static constexpr bool parallel = ...;   // uses more than one thread
//       static constexpr bool stable   = ...;   // guarantees stability
//       template <class T> static bool sort(T* p, std::size_t n);   // false = n/a
//   };
//   #include "bench_main.hpp"
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace fb {

// 16-byte key/value record.  Field order {val, key} matches hwy::K64V64 so
// vqsort can sort it natively (its key-value layout); every other algorithm
// uses the comparator on `key` only.
struct KV {
    std::uint64_t val;
    std::uint64_t key;
};
inline bool operator==(const KV& a, const KV& b) { return a.key == b.key && a.val == b.val; }
struct KVLess {
    bool operator()(const KV& a, const KV& b) const noexcept { return a.key < b.key; }
};

// 8-byte record for the stability suite: comparator on key, id = input index.
struct Rec {
    std::uint32_t key;
    std::uint32_t id;
};
inline bool operator==(const Rec& a, const Rec& b) { return a.key == b.key && a.id == b.id; }
struct RecLess {
    bool operator()(const Rec& a, const Rec& b) const noexcept { return a.key < b.key; }
};

template <class T> struct Less { bool operator()(const T& a, const T& b) const { return a < b; } };
template <> struct Less<KV> : KVLess {};
template <> struct Less<Rec> : RecLess {};

// ---------------------------------------------------------------------------
// type registry
// ---------------------------------------------------------------------------
template <class T> constexpr const char* type_name();
template <> constexpr const char* type_name<std::int32_t>()  { return "i32"; }
template <> constexpr const char* type_name<std::uint32_t>() { return "u32"; }
template <> constexpr const char* type_name<std::int64_t>()  { return "i64"; }
template <> constexpr const char* type_name<std::uint64_t>() { return "u64"; }
template <> constexpr const char* type_name<float>()         { return "f32"; }
template <> constexpr const char* type_name<double>()        { return "f64"; }
template <> constexpr const char* type_name<std::string>()   { return "str"; }
template <> constexpr const char* type_name<KV>()            { return "kv16"; }
template <> constexpr const char* type_name<Rec>()           { return "rec8"; }

// ---------------------------------------------------------------------------
// portable deterministic RNG
// ---------------------------------------------------------------------------
struct SplitMix {
    std::uint64_t s;
    explicit SplitMix(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    std::uint64_t below(std::uint64_t bound) { return bound ? next() % bound : 0; }
    double unit() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
};

inline std::uint64_t hash_mix(std::uint64_t h, std::uint64_t v) {
    SplitMix m(h ^ (v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2)));
    return m.next();
}
inline std::uint64_t hash_str(const char* s) {
    std::uint64_t h = 1469598103934665603ull;
    for (; *s; ++s) h = (h ^ static_cast<unsigned char>(*s)) * 1099511628211ull;
    return h;
}

// ---------------------------------------------------------------------------
// distributions
// ---------------------------------------------------------------------------
enum class Dist {
    Random, Sorted, Reverse, NearlySorted, FewUnique16, FewUnique256, AllEqual,
    OrganPipe, SortedRuns, Rotated, Concat2, BlockSwap, FarSwaps, SortedTail,
    Zipf, SqrtUnique, Count
};
inline const char* dist_name(Dist d) {
    switch (d) {
        case Dist::Random:       return "random";
        case Dist::Sorted:       return "sorted";
        case Dist::Reverse:      return "reverse";
        case Dist::NearlySorted: return "nearly_sorted";
        case Dist::FewUnique16:  return "few_unique16";
        case Dist::FewUnique256: return "few_unique256";
        case Dist::AllEqual:     return "all_equal";
        case Dist::OrganPipe:    return "organ_pipe";
        case Dist::SortedRuns:   return "sorted_runs";
        case Dist::Rotated:      return "rotated";
        case Dist::Concat2:      return "concat2";
        case Dist::BlockSwap:    return "block_swap";
        case Dist::FarSwaps:     return "far_swaps";
        case Dist::SortedTail:   return "sorted_tail";
        case Dist::Zipf:         return "zipf";
        case Dist::SqrtUnique:   return "sqrt_unique";
        default:                 return "?";
    }
}

// Raw 64-bit draws; cardinality is decided here so every type sees the same
// duplicate structure.
inline std::vector<std::uint64_t> raw_draws(std::size_t n, Dist d, SplitMix& rng) {
    std::vector<std::uint64_t> r(n);
    std::uint64_t card = 0;
    if (d == Dist::FewUnique16) card = 16;
    else if (d == Dist::FewUnique256) card = 256;
    else if (d == Dist::SqrtUnique) card = static_cast<std::uint64_t>(std::sqrt(static_cast<double>(n))) + 1;
    if (d == Dist::AllEqual) {
        const std::uint64_t v = rng.next();
        for (auto& x : r) x = v;
    } else if (d == Dist::Zipf) {
        // Heavy-skewed: value = floor(n^u) for uniform u -> many small repeats.
        const double ln = std::log(static_cast<double>(n) + 1.0);
        std::vector<std::uint64_t> palette(1024);
        for (auto& p : palette) p = rng.next();
        for (auto& x : r) {
            const auto rank = static_cast<std::uint64_t>(std::exp(rng.unit() * ln));
            x = rank < palette.size() ? palette[rank] : hash_mix(0x5eedull, rank);
        }
    } else if (card) {
        std::vector<std::uint64_t> palette(card);
        for (auto& p : palette) p = rng.next();
        for (auto& x : r) x = palette[rng.below(card)];
    } else {
        for (auto& x : r) x = rng.next();
    }
    return r;
}

template <class T> T convert(std::uint64_t r, std::size_t index);
template <> inline std::int32_t  convert<std::int32_t>(std::uint64_t r, std::size_t) { return static_cast<std::int32_t>(static_cast<std::uint32_t>(r >> 32)); }
template <> inline std::uint32_t convert<std::uint32_t>(std::uint64_t r, std::size_t) { return static_cast<std::uint32_t>(r >> 32); }
template <> inline std::int64_t  convert<std::int64_t>(std::uint64_t r, std::size_t) { return static_cast<std::int64_t>(r); }
template <> inline std::uint64_t convert<std::uint64_t>(std::uint64_t r, std::size_t) { return r; }
template <> inline float convert<float>(std::uint64_t r, std::size_t) {
    return static_cast<float>(static_cast<std::int32_t>(r >> 32)) * (1.0f / 65536.0f);
}
template <> inline double convert<double>(std::uint64_t r, std::size_t) {
    return static_cast<double>(static_cast<std::int64_t>(r) >> 11) * (1.0 / 1048576.0);
}
template <> inline std::string convert<std::string>(std::uint64_t r, std::size_t) {
    SplitMix m(r);
    const std::size_t len = 4 + static_cast<std::size_t>(r % 29);   // 4..32 chars
    std::string s(len, 'a');
    std::uint64_t bits = m.next();
    for (std::size_t i = 0; i < len; ++i) {
        if (i % 12 == 11) bits = m.next();
        s[i] = static_cast<char>('a' + bits % 26);
        bits /= 26;
    }
    return s;
}
template <> inline KV convert<KV>(std::uint64_t r, std::size_t i) { return KV{static_cast<std::uint64_t>(i), r}; }
template <> inline Rec convert<Rec>(std::uint64_t r, std::size_t i) {
    return Rec{static_cast<std::uint32_t>(r >> 32), static_cast<std::uint32_t>(i)};
}

// Reassign the payload index after shaping so records always carry their
// final input position (makes stable-reference equality meaningful).
template <class T> inline void renumber(std::vector<T>&) {}
template <> inline void renumber<KV>(std::vector<KV>& v) { for (std::size_t i = 0; i < v.size(); ++i) v[i].val = i; }
template <> inline void renumber<Rec>(std::vector<Rec>& v) {
    for (std::size_t i = 0; i < v.size(); ++i) v[i].id = static_cast<std::uint32_t>(i);
}

template <class T>
std::vector<T> make_input(std::size_t n, Dist d, std::uint64_t seed) {
    std::uint64_t h = hash_mix(seed, hash_str(type_name<T>()));
    h = hash_mix(h, n);
    h = hash_mix(h, static_cast<std::uint64_t>(d));
    SplitMix rng(h);
    const auto raw = raw_draws(n, d, rng);
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = convert<T>(raw[i], i);
    const Less<T> less{};
    auto sort_all = [&] { std::stable_sort(v.begin(), v.end(), less); };
    switch (d) {
        case Dist::Sorted: sort_all(); break;
        case Dist::Reverse: sort_all(); std::reverse(v.begin(), v.end()); break;
        case Dist::NearlySorted: {
            sort_all();
            if (n > 70) {
                const std::size_t swaps = n / 1000 + 1;
                for (std::size_t k = 0; k < swaps; ++k) {
                    const std::size_t i = rng.below(n - 66);
                    std::swap(v[i], v[i + 1 + rng.below(64)]);
                }
            }
            break;
        }
        case Dist::OrganPipe:
            sort_all();
            std::reverse(v.begin() + static_cast<std::ptrdiff_t>(n / 2), v.end());
            break;
        case Dist::SortedRuns: {
            const std::size_t run = std::max<std::size_t>(16, static_cast<std::size_t>(std::sqrt(static_cast<double>(n))));
            for (std::size_t b = 0; b < n; b += run)
                std::stable_sort(v.begin() + static_cast<std::ptrdiff_t>(b),
                                 v.begin() + static_cast<std::ptrdiff_t>(std::min(n, b + run)), less);
            break;
        }
        case Dist::Rotated:
            sort_all();
            std::rotate(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(n / 3), v.end());
            break;
        case Dist::Concat2:
            std::stable_sort(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(n / 2), less);
            std::stable_sort(v.begin() + static_cast<std::ptrdiff_t>(n / 2), v.end(), less);
            break;
        case Dist::BlockSwap:
            sort_all();
            if (n >= 20) std::swap_ranges(v.begin() + static_cast<std::ptrdiff_t>(n / 10),
                                          v.begin() + static_cast<std::ptrdiff_t>(2 * (n / 10)),
                                          v.begin() + static_cast<std::ptrdiff_t>(7 * (n / 10)));
            break;
        case Dist::FarSwaps:
            sort_all();
            for (int k = 0; k < 8 && n > 1; ++k) std::swap(v[rng.below(n)], v[rng.below(n)]);
            break;
        case Dist::SortedTail:
            std::stable_sort(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(n - n / 10), less);
            break;
        default: break;
    }
    renumber(v);
    return v;
}

// ---------------------------------------------------------------------------
// timing helpers
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;
inline double seconds_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

}  // namespace fb
