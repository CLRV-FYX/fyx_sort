// bench_matrix.cpp -- local reproduction of the user's comparison matrix.
//
// Types: int32 / int64 / double / std::string
// Sizes: 1M / 5M (configurable)
// Dist : random / sorted / reverse / nearlysorted / lowcard16 / lowcard256 /
//        allequal / zigzag (organ pipe) / sawtooth / mod8
//
// Algorithms (compiled in depending on what is available):
//   fyx      -- fyx::sort with Options.parallel = Off   (serial path)
//   fyxpar   -- fyx::sort with Options.parallel = On    (parallel path)
//   std      -- std::sort
//   ips4o    -- sequential IPS4o          (-DFYX_MATRIX_HAVE_IPS4O)
//   ips4opar -- parallel IPS4o w/ oneTBB  (-DFYX_MATRIX_HAVE_IPS4O_PARALLEL)
//   pdq      -- orlp/pdqsort              (-DFYX_MATRIX_HAVE_PDQ)
//   vqsort   -- google/highway vqsort     (-DFYX_MATRIX_HAVE_VQSORT)
//   xss      -- x86-simd-sort             (-DFYX_MATRIX_HAVE_XSS)
//
// Copy time is excluded: the work vector is refilled before each timed run.
// Best-of-R is reported.
//
// Build:
//   g++ -std=c++17 -O3 -march=native -DNDEBUG -pthread -I. bench/bench_matrix.cpp -o /tmp/bm
//
#include "../fyx_sort.hpp"

#ifdef FYX_MATRIX_HAVE_IPS4O
#  include <ips4o.hpp>
#endif
#ifdef FYX_MATRIX_HAVE_PDQ
#  include "pdqsort.h"
#endif
#ifdef FYX_MATRIX_HAVE_VQSORT
#  include "hwy/contrib/sort/vqsort.h"
#  include "hwy/aligned_allocator.h"
#endif
#ifdef FYX_MATRIX_HAVE_XSS
#  include "x86simdsort.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------------------
// infrastructure
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

enum class Algo { Fyx, FyxPar, Std, Ips4o, Ips4oPar, Pdq, Vqsort, Xss, Count };

static const char* algo_name(Algo a) {
    switch (a) {
        case Algo::Fyx:      return "fyx";
        case Algo::FyxPar:   return "fyxpar";
        case Algo::Std:      return "std";
        case Algo::Ips4o:    return "ips4o";
        case Algo::Ips4oPar: return "ips4opar";
        case Algo::Pdq:      return "pdqsort";
        case Algo::Vqsort:   return "vqsort";
        case Algo::Xss:      return "xss";
        default:             return "?";
    }
}

static bool g_have[static_cast<int>(Algo::Count)] = {
    true,                       // Fyx
#if FYX_ENABLE_PARALLEL
    true,                       // FyxPar
#else
    false,
#endif
    true,                       // Std
#ifdef FYX_MATRIX_HAVE_IPS4O
    true, false,
#else
    false, false,
#endif
#ifdef FYX_MATRIX_HAVE_PDQ
    true,
#else
    false,
#endif
#ifdef FYX_MATRIX_HAVE_VQSORT
    true,
#else
    false,
#endif
#ifdef FYX_MATRIX_HAVE_XSS
    true,
#else
    false,
#endif
};
#ifdef FYX_MATRIX_HAVE_IPS4O_PARALLEL
// enabled below
#endif

// ---------------------------------------------------------------------------
// data generation
// ---------------------------------------------------------------------------
enum class Dist {
    Random, Sorted, Reverse, NearlySorted, LowCard16, LowCard256,
    AllEqual, Zigzag, Sawtooth, Mod8, Rotated, Concat, BlockSwap, FarSwap, Count
};
static const char* dist_name(Dist d) {
    switch (d) {
        case Dist::Random:       return "random";
        case Dist::Sorted:       return "sorted";
        case Dist::Reverse:      return "reverse";
        case Dist::NearlySorted: return "nearlysorted";
        case Dist::LowCard16:    return "lowcard16";
        case Dist::LowCard256:   return "lowcard256";
        case Dist::AllEqual:     return "allequal";
        case Dist::Zigzag:       return "zigzag";
        case Dist::Sawtooth:     return "sawtooth";
        case Dist::Mod8:         return "mod8";
        case Dist::Rotated:      return "rotated";
        case Dist::Concat:       return "concat2";
        case Dist::BlockSwap:    return "blockswap";
        case Dist::FarSwap:      return "farswap";
        default:                 return "?";
    }
}

static std::mt19937_64 g_rng(0xF00DBAADULL);

static std::string random_string(std::size_t len) {
    static const char* alphabet = "abcdefghijklmnopqrstuvwxyz";
    std::string s(len, 'a');
    for (std::size_t i = 0; i < len; ++i) s[i] = alphabet[g_rng() % 26];
    return s;
}

template <class T>
static std::vector<T> make_data(std::size_t n, Dist d);

// --- structural reshaping applied on top of the sorted base ---------------
template <class T>
static void reshape(std::vector<T>& v, Dist d) {
    const std::size_t n = v.size();
    if (n < 16) return;
    if (d == Dist::Rotated) {
        std::rotate(v.begin(), v.begin() + n / 3, v.end());
    } else if (d == Dist::Concat) {
        // two independently sorted halves, each a permutation of disjoint
        // value ranges -> a single merge pass must rebuild the order
        std::size_t mid = n / 2;
        std::vector<T> a(v.begin(), v.begin() + mid), b(v.begin() + mid, v.end());
        for (std::size_t i = 0; i < a.size(); ++i) a[i] = v[2 * i];
        for (std::size_t i = 0; i < b.size(); ++i) b[i] = v[2 * i + 1 < n ? 2 * i + 1 : n - 1];
        std::copy(a.begin(), a.end(), v.begin());
        std::copy(b.begin(), b.end(), v.begin() + mid);
    } else if (d == Dist::BlockSwap) {
        const std::size_t bs = 128;
        const std::size_t nb = n / bs;
        for (std::size_t k = 0; k < 20 && nb > 4; ++k) {
            std::size_t i = g_rng() % (nb - 1);
            std::size_t j = i + 1 + g_rng() % 4;
            if (j >= nb) j = nb - 1;
            for (std::size_t t = 0; t < bs; ++t) std::swap(v[i * bs + t], v[j * bs + t]);
        }
    } else if (d == Dist::FarSwap) {
        std::size_t swaps = n / 1000 + 1;
        for (std::size_t k = 0; k < swaps; ++k) {
            std::size_t i = g_rng() % n;
            std::size_t j = g_rng() % n;
            std::swap(v[i], v[j]);
        }
    }
}


template <>
std::vector<std::int32_t> make_data<std::int32_t>(std::size_t n, Dist d) {
    std::vector<std::int32_t> v(n);
    switch (d) {
        case Dist::Random:
            for (auto& x : v) x = static_cast<std::int32_t>(g_rng());
            break;
        case Dist::Sorted:
        case Dist::Reverse:
        case Dist::NearlySorted:
        case Dist::Zigzag:
        case Dist::Sawtooth:
        case Dist::Rotated:
        case Dist::Concat:
        case Dist::BlockSwap:
        case Dist::FarSwap:
            for (auto& x : v) x = static_cast<std::int32_t>(g_rng());
            std::sort(v.begin(), v.end());
            break;
        case Dist::LowCard16:
            for (auto& x : v) x = static_cast<std::int32_t>(g_rng() % 16);
            break;
        case Dist::LowCard256:
            for (auto& x : v) x = static_cast<std::int32_t>(g_rng() % 256);
            break;
        case Dist::AllEqual:
            for (auto& x : v) x = 42;
            break;
        case Dist::Mod8:
            for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::int32_t>(i % 8);
            break;
        default: break;
    }
    if (d == Dist::Reverse) std::reverse(v.begin(), v.end());
    if (d == Dist::NearlySorted) {
        // sorted data with sparse long-ish displacements: n/1000 swaps with a
        // partner up to 64 positions away.
        std::size_t swaps = n / 1000 + 1;
        for (std::size_t k = 0; k < swaps; ++k) {
            std::size_t i = g_rng() % (n - 66);
            std::size_t j = i + 1 + g_rng() % 64;
            std::swap(v[i], v[j]);
        }
    }
    if (d == Dist::Zigzag) {
        // organ pipe: descending head then ascending tail (bench_final shape)
        std::reverse(v.begin(), v.begin() + n / 2);
    }
    if (d == Dist::Sawtooth) {
        // local sawtooth: sorted, then every pair swapped in place
        for (std::size_t i = 0; i + 1 < n; i += 2) std::swap(v[i], v[i + 1]);
    }
    reshape(v, d);
    return v;
}

template <>
std::vector<std::int64_t> make_data<std::int64_t>(std::size_t n, Dist d) {
    if (d == Dist::Random) {
        std::vector<std::int64_t> v(n);
        for (auto& x : v) x = static_cast<std::int64_t>(g_rng());
        return v;
    }
    // Structured distributions reuse the 32-bit shape so the ordering
    // structure (runs, low cardinality, ...) is identical.
    auto s32 = make_data<std::int32_t>(n, d);
    std::vector<std::int64_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::int64_t>(s32[i]) * 4096;
    return v;
}

template <>
std::vector<double> make_data<double>(std::size_t n, Dist d) {
    auto s32 = make_data<std::int32_t>(n, d);
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<double>(s32[i]);
    if (d == Dist::Random) {
        for (auto& x : v) x = static_cast<double>(static_cast<std::int64_t>(g_rng() % (1ULL << 40))) / 1024.0;
    }
    return v;
}

template <>
std::vector<std::string> make_data<std::string>(std::size_t n, Dist d) {
    std::vector<std::string> v(n);
    switch (d) {
        case Dist::LowCard16:
            for (auto& s : v) s = random_string(4 + g_rng() % 12);
            for (auto& s : v) s = "k" + std::to_string(g_rng() % 16);
            break;
        case Dist::LowCard256:
            for (auto& s : v) s = "k" + std::to_string(g_rng() % 256);
            break;
        case Dist::AllEqual:
            for (auto& s : v) s = "constant-string-value";
            break;
        default:
            for (auto& s : v) s = random_string(16);
            break;
    }
    if (d == Dist::Sorted || d == Dist::Reverse || d == Dist::NearlySorted ||
        d == Dist::Zigzag || d == Dist::Sawtooth || d == Dist::Rotated ||
        d == Dist::Concat || d == Dist::BlockSwap || d == Dist::FarSwap) {
        std::sort(v.begin(), v.end());
    }
    if (d == Dist::Reverse) std::reverse(v.begin(), v.end());
    if (d == Dist::NearlySorted) {
        std::size_t swaps = n / 1000 + 1;
        for (std::size_t k = 0; k < swaps; ++k) {
            std::size_t i = g_rng() % (n - 66);
            std::size_t j = i + 1 + g_rng() % 64;
            std::swap(v[i], v[j]);
        }
    }
    if (d == Dist::Zigzag) std::reverse(v.begin(), v.begin() + n / 2);
    if (d == Dist::Sawtooth)
        for (std::size_t i = 0; i + 1 < n; i += 2) std::swap(v[i], v[i + 1]);
    reshape(v, d);
    return v;
}

// ---------------------------------------------------------------------------
// sorting driver per algorithm
// ---------------------------------------------------------------------------
template <class T>
static void run_algo(Algo a, T* p, std::size_t n) {
    switch (a) {
        case Algo::Fyx: {
            fyx::Options o;
            o.parallel = fyx::Tri::Off;
            fyx::sort(p, n, o);
            break;
        }
        case Algo::FyxPar: {
            fyx::Options o;
            o.parallel = fyx::Tri::On;
            fyx::sort(p, n, o);
            break;
        }
        case Algo::Std:
            std::sort(p, p + n);
            break;
#ifdef FYX_MATRIX_HAVE_IPS4O
        case Algo::Ips4o:
            ips4o::sort(p, p + n);
            break;
#  ifdef FYX_MATRIX_HAVE_IPS4O_PARALLEL
        case Algo::Ips4oPar:
            ips4o::parallel::sort(p, p + n);
            break;
#  endif
#endif
#ifdef FYX_MATRIX_HAVE_PDQ
        case Algo::Pdq:
            pdqsort(p, p + n);
            break;
#endif
#ifdef FYX_MATRIX_HAVE_VQSORT
        case Algo::Vqsort:
            if constexpr (std::is_arithmetic_v<T>)
                hwy::VQSort(p, n, hwy::SortAscending());
            break;
#endif
#ifdef FYX_MATRIX_HAVE_XSS
        case Algo::Xss:
            if constexpr (std::is_arithmetic_v<T>)
                x86simdsort::qsort(p, n);
            break;
#endif
        default: break;
    }
}

// ---------------------------------------------------------------------------
// verification
// ---------------------------------------------------------------------------
template <class T>
static bool verify(const std::vector<T>& got, const std::vector<T>& base) {
    if (!std::is_sorted(got.begin(), got.end())) return false;
    if (got.size() != base.size()) return false;
    std::vector<T> a = got, b = base;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
}

// ---------------------------------------------------------------------------
// benchmark one (type, size, dist) triple
// ---------------------------------------------------------------------------
struct Result { double t; bool ok; };

template <class T>
static Result bench_one(const std::vector<T>& base, Algo a, int reps) {
    std::vector<T> work(base.size());
    double best = std::numeric_limits<double>::infinity();
    bool ok = true;
    for (int r = 0; r < reps; ++r) {
        work = base;                       // copy excluded from timing
        auto t0 = Clock::now();
        run_algo(a, work.data(), work.size());
        auto t1 = Clock::now();
        best = std::min(best, secs(t0, t1));
        if (r == reps - 1) ok = verify(work, base);
    }
    return Result{best, ok};
}

static const char* g_type_name = "";

template <class T>
static void run_case(const char* tname, std::size_t n, Dist d, int reps) {
    g_type_name = tname;
    std::vector<T> base = make_data<T>(n, d);
    double times[static_cast<int>(Algo::Count)];
    bool   oks[static_cast<int>(Algo::Count)] = {false};
    for (int i = 0; i < static_cast<int>(Algo::Count); ++i) times[i] = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < static_cast<int>(Algo::Count); ++i) {
        if (!g_have[i]) continue;
        Result r = bench_one<T>(base, static_cast<Algo>(i), reps);
        times[i] = r.t;
        oks[i] = r.ok;
    }
    // fastest available competitor (excluding fyx variants)
    double best_other = std::numeric_limits<double>::infinity();
    const char* best_other_name = "-";
    for (int i = 0; i < static_cast<int>(Algo::Count); ++i) {
        if (!g_have[i]) continue;
        if (i == static_cast<int>(Algo::Fyx) || i == static_cast<int>(Algo::FyxPar)) continue;
        if (times[i] < best_other) { best_other = times[i]; best_other_name = algo_name(static_cast<Algo>(i)); }
    }
    std::printf("| %-7s | %8zu | %-13s |", tname, n, dist_name(d));
    for (int i = 0; i < static_cast<int>(Algo::Count); ++i) {
        if (!g_have[i]) continue;
        std::printf(" %10.6f%s", times[i], oks[i] ? "" : "!");
    }
    double fyx_par = times[static_cast<int>(Algo::FyxPar)];
    double fyx_ser = times[static_cast<int>(Algo::Fyx)];
    std::printf(" | %-8s %9.2fx %9.2fx |", best_other_name,
                best_other / fyx_par, best_other / fyx_ser);
    std::printf("\n");
}

static void print_header() {
    std::printf("| type    |        n | dist          |");
    for (int i = 0; i < static_cast<int>(Algo::Count); ++i) {
        if (!g_have[i]) continue;
        std::printf(" %11s", algo_name(static_cast<Algo>(i)));
    }
    std::printf(" | best-other  par/ser gain |\n");
    std::printf("|---------|----------|---------------|");
    for (int i = 0; i < static_cast<int>(Algo::Count); ++i) {
        if (!g_have[i]) continue;
        std::printf("------------");
    }
    std::printf(" |--------------------------|\n");
}

int main(int argc, char** argv) {
    std::vector<std::size_t> sizes = {1000000, 5000000};
    int reps = 5;
    const char* filter = "";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--size=", 0) == 0) {
            sizes.clear();
            std::string rest = a.substr(7);
            std::size_t pos = 0;
            while (pos < rest.size()) {
                std::size_t comma = rest.find(',', pos);
                if (comma == std::string::npos) comma = rest.size();
                sizes.push_back(std::stoull(rest.substr(pos, comma - pos)));
                pos = comma + 1;
            }
        } else if (a.rfind("--reps=", 0) == 0) {
            reps = std::atoi(a.c_str() + 7);
        } else if (a.rfind("--filter=", 0) == 0) {
            filter = argv[i] + 9;
        } else {
            std::fprintf(stderr, "usage: %s [--size=1000000,5000000] [--reps=5] [--filter=substr]\n", argv[0]);
            return 2;
        }
    }
#ifdef FYX_MATRIX_HAVE_IPS4O_PARALLEL
    g_have[static_cast<int>(Algo::Ips4oPar)] = true;
#endif

    std::printf("# bench_matrix: best of %d, copy excluded, %s build\n", reps,
#if defined(NDEBUG)
                "NDEBUG"
#else
                "debug"
#endif
    );
    print_header();

    for (std::size_t n : sizes) {
        for (int di = 0; di < static_cast<int>(Dist::Count); ++di) {
            Dist d = static_cast<Dist>(di);
            std::string key = std::string(dist_name(d));
            if (*filter && key.find(filter) == std::string::npos) continue;
            run_case<std::int32_t>("int32", n, d, reps);
            run_case<std::int64_t>("int64", n, d, reps);
            run_case<double>("double", n, d, reps);
            if (n > 1000000) continue;                 // strings: keep memory sane
            run_case<std::string>("string", n, d, reps);
        }
        std::fflush(stdout);
    }
    std::printf("\n(! = verification failed)\n");
    return 0;
}
