// Boost.Sort family.  FB_BOOST_MODE:
//   0 spreadsort (hybrid radix, serial, unstable)
//   1 block_indirect_sort (parallel, unstable)
//   2 sample_sort (parallel, stable)
//   3 parallel_stable_sort (parallel, stable)
//   4 spinsort (serial, stable)
//   5 flat_stable_sort (serial, stable)
//   6 pdqsort (serial, unstable)
#include "common.hpp"
#include <boost/sort/sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <boost/sort/spreadsort/integer_sort.hpp>
#ifndef FB_BOOST_MODE
#  define FB_BOOST_MODE 0
#endif
struct Algo {
#if FB_BOOST_MODE == 0
    static constexpr const char* name = "boost_spreadsort"; static constexpr bool parallel = false, stable = false; static constexpr int suites = fb::kU;
#elif FB_BOOST_MODE == 1
    static constexpr const char* name = "boost_block_indirect"; static constexpr bool parallel = true, stable = false; static constexpr int suites = fb::kU;
#elif FB_BOOST_MODE == 2
    static constexpr const char* name = "boost_sample_sort"; static constexpr bool parallel = true, stable = true; static constexpr int suites = fb::kS;
#elif FB_BOOST_MODE == 3
    static constexpr const char* name = "boost_parallel_stable"; static constexpr bool parallel = true, stable = true; static constexpr int suites = fb::kS;
#elif FB_BOOST_MODE == 4
    static constexpr const char* name = "boost_spinsort"; static constexpr bool parallel = false, stable = true; static constexpr int suites = fb::kS;
#elif FB_BOOST_MODE == 5
    static constexpr const char* name = "boost_flat_stable"; static constexpr bool parallel = false, stable = true; static constexpr int suites = fb::kS;
#else
    static constexpr const char* name = "boost_pdqsort"; static constexpr bool parallel = false, stable = false; static constexpr int suites = fb::kU;
#endif
    template <class T> static bool sort(T* p, std::size_t n) {
        const fb::Less<T> less{};
#if FB_BOOST_MODE == 0
        if constexpr (std::is_same_v<T, fb::KV>) {
            boost::sort::spreadsort::integer_sort(p, p + n,
                [](const fb::KV& x, unsigned s) { return x.key >> s; }, less);
        } else if constexpr (std::is_same_v<T, fb::Rec>) {
            boost::sort::spreadsort::integer_sort(p, p + n,
                [](const fb::Rec& x, unsigned s) { return x.key >> s; }, less);
        } else {
            boost::sort::spreadsort::spreadsort(p, p + n);
        }
#elif FB_BOOST_MODE == 1
        boost::sort::block_indirect_sort(p, p + n, less);
#elif FB_BOOST_MODE == 2
        boost::sort::sample_sort(p, p + n, less);
#elif FB_BOOST_MODE == 3
        boost::sort::parallel_stable_sort(p, p + n, less);
#elif FB_BOOST_MODE == 4
        boost::sort::spinsort(p, p + n, less);
#elif FB_BOOST_MODE == 5
        boost::sort::flat_stable_sort(p, p + n, less);
#else
        boost::sort::pdqsort(p, p + n, less);
#endif
        return true;
    }
};
#include "../core/bench_main.hpp"
