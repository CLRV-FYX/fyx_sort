// google/highway vqsort (runtime-dispatched SIMD quicksort; uses its best
// available target on the host).  kv16's {val, key} layout is hwy::K64V64.
#include "common.hpp"
#include "hwy/contrib/sort/vqsort.h"
struct Algo {
    static constexpr const char* name = "vqsort";
    static constexpr bool parallel = false, stable = false; static constexpr int suites = fb::kU;
    template <class T> static bool sort(T* p, std::size_t n) {
        if constexpr (fb::is_num_v<T>) {
            hwy::VQSort(p, n, hwy::SortAscending());
            return true;
        } else if constexpr (std::is_same_v<T, fb::KV>) {
            static_assert(sizeof(fb::KV) == sizeof(hwy::K64V64), "layout");
            hwy::VQSort(reinterpret_cast<hwy::K64V64*>(p), n, hwy::SortAscending());
            return true;
        } else {
            return false;
        }
    }
};
#include "../core/bench_main.hpp"
