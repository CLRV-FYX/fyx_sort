// orlp/pdqsort (branchless variant auto-selected for arithmetic + std::less).
#include "common.hpp"
#include "pdqsort.h"
#include <functional>
struct Algo {
    static constexpr const char* name = "pdqsort";
    static constexpr bool parallel = false, stable = false; static constexpr int suites = fb::kU;
    template <class T> static bool sort(T* p, std::size_t n) {
        if constexpr (fb::is_record_v<T>) pdqsort(p, p + n, fb::Less<T>{});
        else pdqsort(p, p + n, std::less<T>());
        return true;
    }
};
#include "../core/bench_main.hpp"
