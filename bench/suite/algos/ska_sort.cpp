// skarupke/ska_sort (in-place American-flag radix sort with std::sort fallback).
#include "common.hpp"
#include "ska_sort.hpp"
struct Algo {
    static constexpr const char* name = "ska_sort";
    static constexpr bool parallel = false, stable = false; static constexpr int suites = fb::kU;
    template <class T> static bool sort(T* p, std::size_t n) {
        if constexpr (std::is_same_v<T, fb::KV>) ska_sort(p, p + n, [](const fb::KV& x) { return x.key; });
        else if constexpr (std::is_same_v<T, fb::Rec>) ska_sort(p, p + n, [](const fb::Rec& x) { return x.key; });
        else ska_sort(p, p + n);
        return true;
    }
};
#include "../core/bench_main.hpp"
