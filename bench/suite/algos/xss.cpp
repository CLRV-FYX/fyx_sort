// intel/x86-simd-sort, header-only static build (AVX-512 / AVX2 picked by
// the compile flags -- the runner builds it with the host's native ISA).
#include "common.hpp"
#include "x86simdsort-static-incl.h"
struct Algo {
    static constexpr const char* name = "xss";
    static constexpr bool parallel = false, stable = false; static constexpr int suites = fb::kU;
    template <class T> static bool sort(T* p, std::size_t n) {
        if constexpr (fb::is_num_v<T>) {
            x86simdsortStatic::qsort(p, n, false, false);
            return true;
        } else {
            return false;
        }
    }
};
#include "../core/bench_main.hpp"
