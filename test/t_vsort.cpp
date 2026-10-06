// AVX-512 vectorised quicksort tests (parts/10b_vsort.hpp).
//
// The kernel is only reachable on a machine that has AVX-512, so every check
// here is written to be *correct* on a machine that does not: the kernel-level
// checks are skipped when `vqsort_usable` says no, and the dispatch-level
// checks compare against std::sort, which is the right answer whichever kernel
// the dispatcher picked.
//
// What is covered:
//   1. the kernel itself, on the shapes a quicksort can degenerate on --
//      one value owning most of the range (the pivot-is-the-minimum case),
//      two values, sorted, reverse, and sizes on both sides of the leaf;
//   2. the entry point on the same shapes, ascending and descending, for every
//      type the kernel claims;
//   3. floating point: the kernel must decline a range holding a NaN or a -0,
//      and `fyx::sort` must still produce the documented total order
//      (-NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN) for those;
//   4. stability of the *result* for numeric input: an unstable kernel is only
//      allowed here because equal scalars are indistinguishable, so the output
//      must equal std::sort's element for element.
#define FYX_ENABLE_TEST_HOOKS 1
#include "../fyx_sort.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>

namespace fd = fyx::detail;

static int failures = 0;
static int checks   = 0;

#define CHECK(cond, msg) do {                                          \
    ++checks;                                                          \
    if (!(cond)) { std::printf("  FAIL: %s\n", msg); ++failures; }     \
} while (0)

static std::mt19937_64 rng(0x5EED1234ULL);

// ---------------------------------------------------------------------------
// shapes
// ---------------------------------------------------------------------------
enum class Shape { Random, Sorted, Reverse, AllEqual, TwoValues, MostlyMin,
                   MostlyMax, FewDistinct, Ramp, Sawtooth };

static const char* shape_name(Shape s) {
    switch (s) {
        case Shape::Random:      return "random";
        case Shape::Sorted:      return "sorted";
        case Shape::Reverse:     return "reverse";
        case Shape::AllEqual:    return "all-equal";
        case Shape::TwoValues:   return "two values";
        case Shape::MostlyMin:   return "90% minimum";
        case Shape::MostlyMax:   return "90% maximum";
        case Shape::FewDistinct: return "few distinct";
        case Shape::Ramp:        return "ramp";
        default:                 return "sawtooth";
    }
}

template <class T>
static T from_u64(std::uint64_t u) {
    if constexpr (std::is_floating_point<T>::value) {
        // finite, spread over a wide exponent range, never NaN and never -0
        const double m = static_cast<double>(u % 1000003u) - 500000.0;
        const int    e = static_cast<int>(u % 61u) - 30;
        const T      v = static_cast<T>(std::ldexp(m, e));
        return v == T(0) ? T(1) : v;
    } else {
        return static_cast<T>(u);
    }
}

template <class T>
static std::vector<T> make(Shape s, std::size_t n) {
    std::vector<T> v(n);
    const T lo = std::is_floating_point<T>::value ? T(-1e30) : (std::numeric_limits<T>::lowest)();
    const T hi = std::is_floating_point<T>::value ? T(1e30)  : (std::numeric_limits<T>::max)();
    for (std::size_t i = 0; i < n; ++i) {
        switch (s) {
            case Shape::Random:      v[i] = from_u64<T>(rng());                       break;
            case Shape::Sorted:
            case Shape::Reverse:     v[i] = from_u64<T>(i * 7919u);                   break;
            case Shape::AllEqual:    v[i] = from_u64<T>(42);                          break;
            case Shape::TwoValues:   v[i] = (rng() & 1u) ? lo : hi;                   break;
            case Shape::MostlyMin:   v[i] = (rng() % 10u) ? lo : from_u64<T>(rng());  break;
            case Shape::MostlyMax:   v[i] = (rng() % 10u) ? hi : from_u64<T>(rng());  break;
            case Shape::FewDistinct: v[i] = from_u64<T>(rng() % 5u);                  break;
            case Shape::Ramp:        v[i] = from_u64<T>(i % 8u);                      break;
            default:                 v[i] = from_u64<T>((i % 2u) ? i : n - i);        break;
        }
    }
    if (s == Shape::Sorted)  std::sort(v.begin(), v.end());
    if (s == Shape::Reverse) { std::sort(v.begin(), v.end()); std::reverse(v.begin(), v.end()); }
    return v;
}

// ---------------------------------------------------------------------------
// 1 + 2: the kernel, and the entry point, on every shape and size
// ---------------------------------------------------------------------------
template <class T>
static void run_type(const char* tname) {
    static const std::size_t sizes[] = {
        0, 1, 2, 3, 15, 17, 31, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
        300, 511, 512, 513, 1000,
        16383, 16384, 16385, 70000
    };
    static const Shape shapes[] = {
        Shape::Random, Shape::Sorted, Shape::Reverse, Shape::AllEqual,
        Shape::TwoValues, Shape::MostlyMin, Shape::MostlyMax,
        Shape::FewDistinct, Shape::Ramp, Shape::Sawtooth
    };

    for (std::size_t n : sizes) {
        for (Shape s : shapes) {
            const std::vector<T> src = make<T>(s, n);
            std::vector<T> want = src;
            std::sort(want.begin(), want.end());

            // 1. the kernel on its own, where it exists
            if (fd::vqsort_usable<T>(n)) {
                std::vector<T> v = src;
                fd::vqsort_serial(v.data(), v.size());
                if (v != want) {
                    std::printf("  FAIL: %s kernel n=%zu shape=%s\n", tname, n, shape_name(s));
                    ++failures;
                }
                ++checks;

                std::vector<T> p = src;
                unsigned depth = 0;
                for (unsigned w = 4; w > 1; w >>= 1) ++depth;
                fd::vqsort_parallel_rec(p.data(), p.size(), fd::vqsort_budget(p.size()), depth);
                if (p != want) {
                    std::printf("  FAIL: %s parallel kernel n=%zu shape=%s\n", tname, n, shape_name(s));
                    ++failures;
                }
                ++checks;
            }

            // 2. the entry point, both orders, serial and parallel
            for (int par = 0; par < 2; ++par) {
                fyx::Options o;
                o.parallel = par ? fyx::Tri::On : fyx::Tri::Off;

                std::vector<T> asc = src;
                fyx::sort(asc.data(), asc.size(), o);
                if (asc != want) {
                    std::printf("  FAIL: %s sort n=%zu shape=%s par=%d\n", tname, n, shape_name(s), par);
                    ++failures;
                }
                ++checks;

                std::vector<T> desc_want = want;
                std::reverse(desc_want.begin(), desc_want.end());
                std::vector<T> desc = src;
                fyx::sort(desc.data(), desc.size(), std::greater<T>(), o);
                if (desc != desc_want) {
                    std::printf("  FAIL: %s descending n=%zu shape=%s par=%d\n", tname, n, shape_name(s), par);
                    ++failures;
                }
                ++checks;

                std::vector<T> st = src;
                fyx::stable_sort(st.data(), st.size());
                if (st != want) {
                    std::printf("  FAIL: %s stable n=%zu shape=%s par=%d\n", tname, n, shape_name(s), par);
                    ++failures;
                }
                ++checks;
            }
        }
    }
    std::printf("  %-8s shapes x sizes ok\n", tname);
}

// ---------------------------------------------------------------------------
// 3: floating point the hardware compare cannot order
// ---------------------------------------------------------------------------
template <class T>
static void run_float_edge(const char* tname) {
    const T qnan = std::numeric_limits<T>::quiet_NaN();
    const T inf  = std::numeric_limits<T>::infinity();
    const T nzero = -T(0);

    // the kernel must refuse a range that holds either of them
    {
        std::vector<T> v(70000);
        for (auto& x : v) x = from_u64<T>(rng());
        // (a build without the kernel refuses everything, which is safe)
        CHECK(!fd::vqsort_usable<T>(v.size()) || fd::vqsort_range_clean(v.data(), v.size()),
              "clean range accepted");

        std::vector<T> with_nan = v;
        with_nan[with_nan.size() / 3] = qnan;
        CHECK(!fd::vqsort_range_clean(with_nan.data(), with_nan.size()), "NaN range refused");

        std::vector<T> with_nzero = v;
        with_nzero[with_nzero.size() - 1] = nzero;
        CHECK(!fd::vqsort_range_clean(with_nzero.data(), with_nzero.size()), "-0 range refused");

        std::vector<T> tail_nan = v;
        tail_nan[tail_nan.size() - 2] = qnan;      // inside the masked tail
        CHECK(!fd::vqsort_range_clean(tail_nan.data(), tail_nan.size()), "NaN in tail refused");
    }

    // and the entry point must still produce the documented total order
    for (std::size_t n : {std::size_t(1000), std::size_t(70000)}) {
        std::vector<T> v(n);
        for (std::size_t i = 0; i < n; ++i) {
            switch (i % 7) {
                case 0:  v[i] = qnan;      break;
                case 1:  v[i] = -qnan;     break;
                case 2:  v[i] = nzero;     break;
                case 3:  v[i] = T(0);      break;
                case 4:  v[i] = inf;       break;
                case 5:  v[i] = -inf;      break;
                default: v[i] = from_u64<T>(rng()); break;
            }
        }
        for (int par = 0; par < 2; ++par) {
            fyx::Options o;
            o.parallel = par ? fyx::Tri::On : fyx::Tri::Off;
            std::vector<T> s = v;
            fyx::sort(s.data(), s.size(), o);

            // radix total order: keys must be non-decreasing
            using RT = fd::RadixTraits<T>;
            bool ordered = true;
            for (std::size_t i = 1; i < s.size(); ++i)
                if (RT::encode(s[i]) < RT::encode(s[i - 1])) ordered = false;
            CHECK(ordered, "float total order after sort");

            // and the multiset must be preserved, bit for bit
            std::vector<typename RT::Key> a(v.size()), b(s.size());
            for (std::size_t i = 0; i < v.size(); ++i) a[i] = RT::encode(v[i]);
            for (std::size_t i = 0; i < s.size(); ++i) b[i] = RT::encode(s[i]);
            std::sort(a.begin(), a.end());
            CHECK(a == b, "float multiset preserved");
        }
    }
    std::printf("  %-8s NaN / -0 edge cases ok\n", tname);
}

// ---------------------------------------------------------------------------
// 4: the leaf network on every size it serves, and the small-n prescan
// ---------------------------------------------------------------------------
template <class T>
static void run_leaf_and_prescan(const char* tname) {
#if FYX_HAS_AVX512_CODE
    if (!fd::use_avx512() || !fd::vqsort_kernel_supported_v<T>) return;
    for (std::size_t n = 1; n <= 520; ++n) {
        for (int rep = 0; rep < 6; ++rep) {
            std::vector<T> v(n + 16);
            for (auto& x : v) x = from_u64<T>(rep % 3 == 0 ? rng() % 5 : rng());
            std::vector<T> want(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(n));
            std::sort(want.begin(), want.end());
            const std::vector<T> guard(v.begin() + static_cast<std::ptrdiff_t>(n), v.end());
            fd::isa_avx512::vnet_sort<T>(v.data(), n);
            const bool ok = std::equal(want.begin(), want.end(), v.begin()) &&
                            std::equal(guard.begin(), guard.end(), v.begin() + static_cast<std::ptrdiff_t>(n));
            if (!ok) {
                std::printf("  FAIL: %s leaf n=%zu rep=%d\n", tname, n, rep);
                ++failures;
            }
            ++checks;
        }
    }
    for (std::size_t n = 1; n <= 300; n += (n < 80 ? 1 : 37)) {
        for (int rep = 0; rep < 8; ++rep) {
            std::vector<T> v(n);
            for (auto& x : v) x = from_u64<T>(rep < 2 ? 7 : rng() % 1000);
            if (rep == 2 || rep == 4) std::sort(v.begin(), v.end());
            if (rep == 3 || rep == 5) std::sort(v.begin(), v.end(), std::greater<T>());
            if constexpr (std::is_floating_point<T>::value) {
                if (rep == 4) v[rng() % n] = std::numeric_limits<T>::quiet_NaN();
                if (rep == 5) v[rng() % n] = T(-0.0);
            }
            unsigned up = 0, dn = 0;
            for (std::size_t j = 0; j + 1 < n; ++j) { up |= v[j] < v[j + 1]; dn |= v[j + 1] < v[j]; }
            bool bad = false;
            if constexpr (std::is_floating_point<T>::value)
                for (T x : v) bad |= (x != x) || (x == T(0) && std::signbit(x));
            const unsigned got = fd::vqsort_small_prescan(v.data(), n);
            // Bit 3 (cleanliness left unchecked) only after both directions
            // were seen in a clean prefix.
            const bool ok = (got & 8u)
                ? ((got & 7u) == 3u && up && dn)
                : (((got & 4u) != 0) == bad && (bad || (got & 3u) == (up | (dn << 1))));
            if (!ok) {
                std::printf("  FAIL: %s prescan n=%zu rep=%d got=%u\n", tname, n, rep, got);
                ++failures;
            }
            ++checks;
        }
    }
    // Past the ymm threshold: four interleaved quarter streams, the
    // sequential settled / prior rebuilt from per-stream flags.  Breaks in
    // every quarter, at the stream junctions and inside verified chunks;
    // prior must be the exact order bits of the pairs before settled.
    // Pair flags as the scan defines them: < up, > down, neither with
    // differing bits (NaN, +-0) both -- exact in key order without a NaN / -0
    // screen, so bit 2 may stay clear on unclean floats.
    auto pf = [](T x, T y) -> unsigned {
        if (x < y) return 1u;
        if (y < x) return 2u;
        return std::memcmp(&x, &y, sizeof(T)) != 0 ? 3u : 0u;
    };
    {
        const std::size_t thr = fd::isa_avx512::kPrescanYmmBytes / sizeof(T);
        for (std::size_t n : {thr + 1, thr + 37, 3 * thr + 5}) {
            const std::size_t q = (n - 1) / 4;
            std::vector<std::size_t> spots = {0, 1, 63, q - 1, q, q + 1, q + 17, 2 * q - 1, 2 * q,
                                              2 * q + 300, 3 * q - 2, 3 * q, 3 * q + 5, 4 * q - 1,
                                              4 * q, n - 3, n - 2, q / 2, q + q / 3, 3 * q + q / 2};
            for (int r = 0; r < 6; ++r) spots.push_back(rng() % (n - 1));
            for (std::size_t b : spots) {
                if (b + 1 >= n) continue;
                for (int mode = 0; mode < 8; ++mode) {
                    if (mode >= 5 && !std::is_floating_point<T>::value) continue;
                    std::vector<T> v(n);
                    // ascending with equal runs; mode 1/3 descending base
                    // (from_u64 is not monotone for floats: exact small
                    // integers crossing +0 instead)
                    for (std::size_t j = 0; j < n; ++j)
                        v[j] = std::is_floating_point<T>::value
                                   ? static_cast<T>(static_cast<double>(j / 3) - static_cast<double>(n / 6))
                                   : from_u64<T>(j / 3);
                    if (mode == 1 || mode == 3) std::reverse(v.begin(), v.end());
                    if (mode <= 1) std::swap(v[b], v[b + 1 < n ? b + 1 : b]);   // one local break (may be equal)
                    if (mode == 0 || mode == 1) { v[b + 1] = v[0 + (mode == 0 ? 0 : n - 1)]; }
                    if (mode == 2 || mode == 3) std::reverse(v.begin() + static_cast<std::ptrdiff_t>(b), v.end());   // organ at b
                    if (mode == 4) std::fill(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(b + 1), v[b + 1]);   // flat prefix, then ascending
                    if constexpr (std::is_floating_point<T>::value)
                    {
                        if (b % 3 == 0 && mode == 4) v[n - 1] = T(-0.0);
                        // -0 prefix before +0 / positive keys
                        if (mode == 5) std::fill(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(b + 1), T(-0.0));
                        if (mode == 6) v[b] = std::numeric_limits<T>::quiet_NaN();
                        if (mode == 7) {   // descending, -0 tail after +0 keys
                            std::reverse(v.begin(), v.end());
                            std::fill(v.begin() + static_cast<std::ptrdiff_t>(b), v.end(), T(-0.0));
                        }
                    }
                    unsigned up = 0, dn = 0;
                    std::size_t m = n;   // first pair index where both directions are seen
                    for (std::size_t j = 0; j + 1 < n; ++j) {
                        const unsigned f = pf(v[j], v[j + 1]);
                        up |= f & 1u;
                        dn |= f >> 1;
                        if (up && dn && m == n) m = j;
                    }
                    bool bad = false;
                    if constexpr (std::is_floating_point<T>::value)
                        for (T x : v) bad |= (x != x) || (x == T(0) && std::signbit(x));
                    std::size_t settled = ~std::size_t(0);
                    unsigned prior = 99;
                    const unsigned got = fd::vqsort_small_prescan(v.data(), n, &settled, &prior);
                    bool ok;
                    if (m == n) {
                        ok = (got & 8u) == 0 && ((got & 4u) == 0 || bad) &&
                             ((got & 4u) != 0 || (got & 3u) == (up | (dn << 1)));
                    } else {
                        ok = (got & 3u) == 3u && ((got & 4u) == 0 || bad) && !((got & 4u) && (got & 8u)) &&
                             settled <= m && m - settled < 1024;
                        if (ok) {
                            unsigned pu = 0, pd = 0;
                            for (std::size_t j = 0; j < settled; ++j) { const unsigned f = pf(v[j], v[j + 1]); pu |= f & 1u; pd |= f >> 1; }
                            ok = prior == (pu | (pd << 1));
                        }
                    }
                    if (!ok) {
                        std::printf("  FAIL: %s prescan big n=%zu b=%zu mode=%d got=%u settled=%zu m=%zu prior=%u\n",
                                    tname, n, b, mode, got, settled, m, prior);
                        ++failures;
                    }
                    ++checks;
                }
            }
        }
    }
    std::printf("  %-8s leaf + prescan ok\n", tname);
#else
    (void)tname;
#endif
}

// ---------------------------------------------------------------------------
// 5: sparse-descent repairs (also a regression test: the sorted-affix weapon
//    once sorted its middle ascending under std::greater when NaN was present) of the small path (two-run merge, rotation,
//    extract-merge with outlier popping, bounded insertion), ascending and
//    descending, with NaN / -0 mixed into float keys.  Checked against the
//    radix total order and the bit-exact multiset.
// ---------------------------------------------------------------------------
template <class T>
static void run_sparse_descent_shapes(const char* tname) {
    using RT  = fd::RadixTraits<T>;
    using Key = typename RT::Key;
    auto gen = [&](std::size_t i) -> T {
        if (std::is_floating_point<T>::value && i % 97 == 5) {
            switch (rng() % 3) {
                case 0:  return std::numeric_limits<T>::quiet_NaN();
                case 1:  return -T(0);
                default: return T(0);
            }
        }
        if (i % 13 == 0) return from_u64<T>(rng() % 8);         // duplicates
        return from_u64<T>(rng());
    };
    auto asc = [](std::vector<T>& v, std::size_t lo, std::size_t hi) {
        std::sort(v.begin() + lo, v.begin() + hi,
                  [](const T& a, const T& b) { return RT::encode(a) < RT::encode(b); });
    };
    int shapes_run = 0;
    for (std::size_t n : {std::size_t(600), std::size_t(1500), std::size_t(5000),
                          std::size_t(20000), std::size_t(100000)}) {
        for (int shape = 0; shape < 12; ++shape) {
            std::vector<T> v(n);
            for (std::size_t i = 0; i < n; ++i) v[i] = gen(i);
            switch (shape) {
                case 0: asc(v, 0, n / 2); asc(v, n / 2, n); break;               // concat2
                case 1: asc(v, 0, n / 3); asc(v, n / 3, n); break;               // uneven two runs
                case 2: asc(v, 0, n); std::rotate(v.begin(), v.begin() + n / 3, v.end()); break;
                case 3: asc(v, 0, n - n / 20); break;                            // 5% tail
                case 4: asc(v, 0, n - n / 3); break;                             // 33% tail
                case 5: asc(v, 0, n);                                            // far swaps
                        for (int k = 0; k < 8; ++k) std::swap(v[rng() % n], v[rng() % n]);
                        break;
                case 6: asc(v, 0, n);                                            // adjacent outlier pairs
                        for (std::size_t k = 0; k + 2 < n; k += n / 7 + 1) {
                            v[k] = v[n - 1]; v[k + 1] = v[n - 1];
                        }
                        break;
                case 7: asc(v, 0, n);                                            // local swaps
                        for (std::size_t k = 0; k + 9 < n; k += 61) std::swap(v[k], v[k + 9]);
                        break;
                case 9: asc(v, 0, n / 2); asc(v, n / 2, n);                      // organ pipe
                        std::reverse(v.begin() + n / 2, v.end());
                        break;
                case 10: asc(v, 0, n);                                           // block swap
                        std::swap_ranges(v.begin() + n / 10, v.begin() + 2 * n / 10, v.begin() + 7 * n / 10);
                        break;
                case 11: asc(v, 0, n / 4); asc(v, n / 4, n / 2); asc(v, n / 2, n); // 3 runs, middle down
                        std::reverse(v.begin() + n / 4, v.begin() + n / 2);
                        break;
                default: asc(v, 0, n);                                           // near-equal halves
                        for (std::size_t i = 0; i < n; ++i) v[i] = from_u64<T>(i % (n / 2) / 4);
                        break;
            }
            for (int dir = 0; dir < 2; ++dir) {
                std::vector<T> s = v;
                if (dir) std::reverse(s.begin(), s.end());                       // mirrored shape
                if (dir) fyx::sort(s.begin(), s.end(), std::greater<T>());
                else     fyx::sort(s.begin(), s.end());
                bool ordered = true;
                for (std::size_t i = 1; i < n; ++i) {
                    const Key a = RT::encode(s[i - 1]), b = RT::encode(s[i]);
                    if (std::is_floating_point<T>::value) {
                        // with NaN only the non-NaN subsequence order is defined
                        if (s[i - 1] != s[i - 1] || s[i] != s[i]) continue;
                        if (a != b && s[i - 1] == s[i]) continue;                // -0 / +0
                    }
                    if (dir ? (a < b) : (b < a)) {
                        if (ordered) std::printf("    bad at %zu: %.17g %.17g\n", i, double(s[i - 1]), double(s[i]));
                        ordered = false;
                    }
                }
                if (!ordered) std::printf("    [%s n=%zu shape=%d dir=%d dispatch=%d]\n", tname, n, shape, dir, int(fd::test_last_dispatch()));
                CHECK(ordered, "sparse-descent shape ordered");
                std::vector<Key> ka(n), kb(n);
                for (std::size_t i = 0; i < n; ++i) { ka[i] = RT::encode(v[i]); kb[i] = RT::encode(s[i]); }
                std::sort(ka.begin(), ka.end());
                std::sort(kb.begin(), kb.end());
                CHECK(ka == kb, "sparse-descent multiset preserved");
                ++shapes_run;
            }
        }
    }
    std::printf("  %-8s sparse-descent repairs ok (%d cases)\n", tname, shapes_run);
}

// ---------------------------------------------------------------------------
// Newer small-n paths: few-distinct counting, bitwise all-equal, the NaN/-0
// screening fused into the first vq partition, and the sparse-outlier repair.
// Every case is checked for order (radix total order; loose around NaN and
// signed zeros exactly as above) and for an unchanged multiset.
// ---------------------------------------------------------------------------
template <class T>
static bool sorted_ok(const std::vector<T>& orig, const std::vector<T>& s, bool dir) {
    using RT  = fd::RadixTraits<T>;
    using Key = typename RT::Key;
    const std::size_t n = s.size();
    for (std::size_t i = 1; i < n; ++i) {
        const Key a = RT::encode(s[i - 1]), b = RT::encode(s[i]);
        if (std::is_floating_point<T>::value) {
            if (s[i - 1] != s[i - 1] || s[i] != s[i]) continue;
            if (a != b && s[i - 1] == s[i]) continue;
        }
        if (dir ? (a < b) : (b < a)) return false;
    }
    std::vector<Key> ka(n), kb(n);
    for (std::size_t i = 0; i < n; ++i) { ka[i] = RT::encode(orig[i]); kb[i] = RT::encode(s[i]); }
    std::sort(ka.begin(), ka.end());
    std::sort(kb.begin(), kb.end());
    return ka == kb;
}

template <class T>
static void check_both_dirs(const std::vector<T>& v, const char* tname, const char* what, std::size_t n, int k) {
    for (int dir = 0; dir < 2; ++dir) {
        std::vector<T> s = v;
        if (dir) fyx::sort(s.begin(), s.end(), std::greater<T>());
        else     fyx::sort(s.begin(), s.end());
        const bool ok = sorted_ok(v, s, dir != 0);
        if (!ok) std::printf("    [%s %s n=%zu k=%d dir=%d dispatch=%d]\n", tname, what, n, k, dir, int(fd::test_last_dispatch()));
        CHECK(ok, what);
    }
}

template <class T>
static T special_or(std::uint64_t u, int which) {
    if (std::is_floating_point<T>::value) {
        if (which == 1) return std::numeric_limits<T>::quiet_NaN();
        if (which == 2) return -T(0);
        if (which == 3) return T(0);
    }
    return from_u64<T>(u);
}

template <class T>
static void run_new_small_paths(const char* tname) {
    int cases = 0;
    // 1. few distinct keys (<= 32 counts; 33+ must decline), rare extra key.
    for (std::size_t n : {std::size_t(1000), std::size_t(1023), std::size_t(1024), std::size_t(1500),
                          std::size_t(5000), std::size_t(70000)}) {
        for (int K : {1, 2, 3, 5, 16, 17, 31, 32, 33, 40}) {
            for (int variant = 0; variant < 4; ++variant) {
                std::vector<T> vals(K);
                for (int j = 0; j < K; ++j) vals[j] = from_u64<T>(rng());
                if (std::is_floating_point<T>::value && variant >= 2 && K >= 3) {
                    vals[0] = -T(0); vals[1] = T(0);
                    if (variant == 3) vals[2] = std::numeric_limits<T>::quiet_NaN();
                }
                std::vector<T> v(n);
                for (std::size_t i = 0; i < n; ++i) v[i] = vals[rng() % K];
                if (variant == 1) v[rng() % n] = from_u64<T>(rng());         // one key the sample misses
                check_both_dirs(v, tname, "few-distinct", n, K);
                ++cases;
            }
        }
    }
    // 2. bitwise all-equal and near misses (one key off at head / middle / tail).
    for (std::size_t n = 1; n <= 70; ++n) {
        for (int where = 0; where < 4; ++where) {
            std::vector<T> v(n, from_u64<T>(12345));
            if (where == 1) v[0] = from_u64<T>(7);
            if (where == 2) v[n / 2] = from_u64<T>(7);
            if (where == 3) v[n - 1] = from_u64<T>(7);
            check_both_dirs(v, tname, "all-equal", n, where);
            ++cases;
        }
    }
    for (std::size_t n : {std::size_t(1000), std::size_t(4097), std::size_t(100003)}) {
        for (int where = 0; where < 5; ++where) {
            std::vector<T> v(n, special_or<T>(99, where == 4 ? 2 : 0));       // all -0 (float)
            if (where == 1) v[0] = from_u64<T>(7);
            if (where == 2) v[n / 2 + 1] = from_u64<T>(7);
            if (where == 3) v[n - 2] = from_u64<T>(7);
            check_both_dirs(v, tname, "all-equal", n, where);
            ++cases;
        }
    }
    // 3. random keys with one / a few NaN, -0, +0 (the fused screening path,
    //    on both sides of the leaf size and past the first partition).
    for (std::size_t n : {std::size_t(300), std::size_t(600), std::size_t(1000), std::size_t(5000),
                          std::size_t(20000), std::size_t(100000), std::size_t(300000)}) {
        for (int mix = 0; mix < 6; ++mix) {
            std::vector<T> v(n);
            for (std::size_t i = 0; i < n; ++i) v[i] = from_u64<T>(rng());
            switch (mix) {
                case 1: v[rng() % n] = special_or<T>(0, 1); break;
                case 2: v[rng() % n] = special_or<T>(0, 2); break;
                case 3: v[n - 1] = special_or<T>(0, 2); v[0] = special_or<T>(0, 1); break;
                case 4: for (int j = 0; j < 9; ++j) v[rng() % n] = special_or<T>(0, 1 + j % 3); break;
                case 5: v[n / 2] = special_or<T>(0, 3); break;
                default: break;
            }
            check_both_dirs(v, tname, "screened-vq", n, mix);
            ++cases;
        }
    }
    // 4. sparse outliers: far swaps, adjacent pairs, clustered outliers,
    //    heavy duplicates, special keys among the outliers.
    auto asc = [](std::vector<T>& v) {
        std::sort(v.begin(), v.end(), [](const T& a, const T& b) {
            return fd::RadixTraits<T>::encode(a) < fd::RadixTraits<T>::encode(b); });
    };
    for (std::size_t n : {std::size_t(300), std::size_t(513), std::size_t(1000), std::size_t(1500),
                          std::size_t(2047), std::size_t(3000), std::size_t(4096), std::size_t(5000),
                          std::size_t(20000)}) {
        for (int k : {1, 2, 4, 8, 15, 16, 17, 31, 40}) {
            for (int kind = 0; kind < 5; ++kind) {
                std::vector<T> v(n);
                for (std::size_t i = 0; i < n; ++i)
                    v[i] = kind == 3 ? from_u64<T>(rng() % 5) : from_u64<T>(rng());
                asc(v);
                for (int j = 0; j < k; ++j) {
                    const std::size_t a = rng() % n, b = rng() % n;
                    switch (kind) {
                        case 1: if (a + 1 < n) std::swap(v[a], v[a + 1]); break;            // adjacent
                        case 2: if (a + 3 < n) { v[a] = v[n - 1]; v[a + 1] = v[n - 1]; v[a + 2] = v[0]; } break;
                        case 4: v[a] = special_or<T>(rng(), 1 + j % 3); break;               // NaN / -0 / +0
                        default: std::swap(v[a], v[b]); break;
                    }
                }
                check_both_dirs(v, tname, "sparse-outliers", n, k * 10 + kind);
                ++cases;
            }
        }
    }
    // 5. few monotone runs (galloping merges): 2..9 runs of uneven lengths,
    //    disjoint or interleaved value ranges, ties, descending runs, specials;
    //    and many well-spaced runs (the runs-like gate in front of the vq).
    for (std::size_t n : {std::size_t(700), std::size_t(1000), std::size_t(3000), std::size_t(20000),
                          std::size_t(70000)}) {
        for (int R = 2; R <= 9; ++R) {
            for (int kind = 0; kind < 4; ++kind) {
                std::vector<T> v(n);
                for (std::size_t i = 0; i < n; ++i) {
                    const int sp = (kind == 3 && i % 41 == 7) ? 1 + int(i % 3) : 0;
                    v[i] = special_or<T>(kind == 1 ? rng() % 7 : rng(), sp);
                }
                std::vector<std::size_t> cut{0, n};
                for (int r = 1; r < R; ++r) cut.push_back(40 + rng() % (n - 80));
                std::sort(cut.begin(), cut.end());
                if (kind == 2) asc(v);                      // disjoint ranges, runs then permuted
                for (int r = 0; r < R; ++r) {
                    auto b0 = v.begin() + cut[r], b1 = v.begin() + cut[r + 1];
                    std::sort(b0, b1, [](const T& x, const T& y) {
                        return fd::RadixTraits<T>::encode(x) < fd::RadixTraits<T>::encode(y); });
                    if (r % 3 == 2) std::reverse(b0, b1);
                }
                if (kind == 2 && R >= 3)                     // block swap of two runs
                    std::rotate(v.begin() + cut[1], v.begin() + cut[2], v.begin() + cut[3]);
                check_both_dirs(v, tname, "few-runs", n, R * 10 + kind);
                ++cases;
            }
        }
        for (std::size_t run : {std::size_t(16), std::size_t(17), std::size_t(31), std::size_t(64), std::size_t(300)}) {
            std::vector<T> v(n);
            for (std::size_t i = 0; i < n; ++i) v[i] = from_u64<T>(rng());
            for (std::size_t b0 = 0; b0 < n; b0 += run) {
                auto e = v.begin() + std::min(n, b0 + run);
                std::sort(v.begin() + b0, e, [](const T& x, const T& y) {
                    return fd::RadixTraits<T>::encode(x) < fd::RadixTraits<T>::encode(y); });
            }
            check_both_dirs(v, tname, "sorted-runs", n, int(run));
            ++cases;
        }
    }
    // 6. moderate cardinality (hash counting): 33..2500 distinct keys incl.
    //    NaN / -0 / +0 among them, skewed (zipf-like) and near the table cap.
    for (std::size_t n : {std::size_t(4096), std::size_t(20000), std::size_t(70000), std::size_t(300000)}) {
        for (int K : {33, 100, 256, 447, 1000, 1500, 2048, 2049, 2500}) {
            for (int variant = 0; variant < 3; ++variant) {
                std::vector<T> vals(K);
                for (int j = 0; j < K; ++j) vals[j] = from_u64<T>(rng());
                if (std::is_floating_point<T>::value) {
                    vals[0] = -T(0); vals[1] = T(0); vals[2] = std::numeric_limits<T>::quiet_NaN();
                }
                std::vector<T> v(n);
                for (std::size_t i = 0; i < n; ++i) {
                    std::size_t r = rng() % K;
                    if (variant == 1) r = std::min<std::size_t>(r, rng() % K);       // skew
                    v[i] = vals[r];
                }
                if (variant == 2) {                                          // long tail
                    for (std::size_t i = 0; i < n; i += 7) v[i] = from_u64<T>(rng());
                }
                check_both_dirs(v, tname, "hash-count", n, K * 10 + variant);
                ++cases;
            }
        }
    }
    // 7. hash counting entry points directly: gate boundaries (n/24, n/32,
    //    n/640 for 4-byte keys), single-copy keys at both ends (the run fill
    //    trims only the final run), block-clustered inputs that pass the
    //    block sample but hold many keys (must decline untouched).
    for (std::size_t n : {std::size_t(4096), std::size_t(20000), std::size_t(200000)}) {
        for (int K : {150, 200, 300, 333, 400, 600, 700, 800, 1300}) {
            for (int variant = 0; variant < 3; ++variant) {
                std::vector<T> vals(K);
                for (int j = 0; j < K; ++j) vals[j] = from_u64<T>(rng());
                if (std::is_floating_point<T>::value && K > 3) {
                    vals[0] = -T(0); vals[1] = T(0); vals[2] = std::numeric_limits<T>::quiet_NaN();
                }
                std::vector<T> v(n);
                if (variant == 2) {
                    // 64-key blocks of one value each, values all distinct
                    for (std::size_t i = 0; i < n; ++i) v[i] = from_u64<T>((i / 64) * 0x9E3779B97F4A7C15ull + 1);
                } else {
                    for (std::size_t i = 0; i < n; ++i) v[i] = vals[rng() % K];
                    if (variant == 1) {          // extreme keys present exactly once
                        v[0] = std::numeric_limits<T>::lowest();
                        v[n - 1] = std::numeric_limits<T>::max();
                    }
                }
                for (int dir = 0; dir < 2; ++dir) {
                    std::vector<T> s = v;
                    const bool took = fd::try_hash_count_sort(s.data(), n, dir != 0);
                    const bool ok = took ? sorted_ok(v, s, dir != 0) : (std::memcmp(s.data(), v.data(), n * sizeof(T)) == 0);
                    if (!ok) std::printf("    [%s hash-direct n=%zu K=%d var=%d dir=%d took=%d]\n", tname, n, K, variant, dir, int(took));
                    CHECK(ok, "hash-direct");
                }
                check_both_dirs(v, tname, "hash-gate", n, K * 10 + variant);
                ++cases;
            }
        }
    }
    std::printf("  %-8s new small-n paths ok (%d cases)\n", tname, cases);
}


// Few-valued ranges through the vq directly: leaf screen (one / two values
// rewritten from counts, three or more fall through to the network) and the
// uniform-sample exit (single-valued ranges above leaf size), at sizes
// around the leaf and partition boundaries, with values placed so the
// minimum / maximum / a middle value dominate in turn.
template <class T>
static void run_few_valued(const char* tname) {
    if (!fd::vqsort_usable<T>(100000)) return;
    std::mt19937_64 g(11);
    char msg[128];
    for (std::size_t n : {64ul, 65ul, 100ul, 255ul, 256ul, 257ul, 511ul, 512ul, 513ul, 700ul,
                          1500ul, 4100ul, 30000ul, 200000ul}) {
        for (int k : {1, 2, 3, 5, 40}) {
            for (int skew = 0; skew < 3; ++skew) {
                std::vector<T> vals(static_cast<std::size_t>(k));
                for (auto& x : vals) x = from_u64<T>(g());
                std::vector<T> v(n);
                for (auto& x : v) {
                    std::size_t idx = static_cast<std::size_t>(g() % static_cast<std::uint64_t>(k));
                    // skew 1: 90% the first value; skew 2: one value is rare
                    if (skew == 1 && g() % 10 != 0) idx = 0;
                    if (skew == 2 && k > 1 && idx == 0 && g() % 50 != 0) idx = 1;
                    x = vals[idx];
                }
                std::vector<T> ref = v;
                std::sort(ref.begin(), ref.end());
                fd::vqsort_serial(v.data(), v.size());
                std::snprintf(msg, sizeof msg, "few-valued vq %s n=%zu k=%d skew=%d", tname, n, k, skew);
                CHECK(v == ref, msg);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bitwise all-equal scan: both paths (aligned zmm below kPrescanYmmBytes,
// four aligned ymm streams past it), element-misaligned starts, and one
// differing element at the head, around the alignment point, at stream
// starts / ends and in the tail.
// ---------------------------------------------------------------------------
template <class T>
static void run_all_equal_scan(const char* tname) {
#if FYX_HAS_AVX512_CODE
    if (!fd::use_avx512()) return;
    const std::size_t thr = fd::isa_avx512::kPrescanYmmBytes / sizeof(T);
    for (std::size_t n : {std::size_t(1), std::size_t(15), std::size_t(17), std::size_t(300), std::size_t(4099),
                          thr - 1, thr, thr + 77, 2 * thr + 3}) {
        for (std::size_t off = 0; off < 4; ++off) {
            std::vector<T> buf(n + 8);
            T* p = buf.data() + off;
            const T base = from_u64<T>(12345);
            std::fill(p, p + n, base);
            bool ok = fd::isa_avx512::vall_equal(p, n);
            const std::size_t q = n / 4;
            std::vector<std::size_t> spots = {0, 1, n / 2, n - 1, 15, 16, 17, 31, 32, 33, q - 1, q, q + 1,
                                              2 * q, 3 * q - 1, 3 * q, 4 * q - 1, 4 * q, n - 2, n - 9};
            for (int r = 0; r < 8; ++r) spots.push_back(rng() % n);
            for (std::size_t j : spots) {
                if (j >= n || n < 2) continue;   // one key is all-equal whatever it is
                T alt = from_u64<T>(777);
                if constexpr (std::is_floating_point<T>::value)
                    if (j % 2) { std::fill(p, p + n, T(0.0)); alt = T(-0.0); }   // bitwise, not ==
                const T keep = p[j];
                p[j] = alt;
                ok = ok && !fd::isa_avx512::vall_equal(p, n);
                p[j] = keep;
                ok = ok && fd::isa_avx512::vall_equal(p, n);
                std::fill(p, p + n, base);
            }
            if (!ok) {
                std::printf("  FAIL: %s vall_equal n=%zu off=%zu\n", tname, n, off);
                ++failures;
            }
            ++checks;
        }
    }
    std::printf("  %-8s all-equal scan ok\n", tname);
#else
    (void)tname;
#endif
}

// ---------------------------------------------------------------------------
// Checked tail reverse (few-run merge's descending tail): the ymm kernel's
// three outcomes and undo, and whole organ pipes with +-0 / NaN tails
// against the key order (radix totalOrder: -0 before +0, NaN last).
// ---------------------------------------------------------------------------
template <class T>
static void run_reverse_tail(const char* tname) {
    constexpr bool fp = std::is_floating_point<T>::value;
    using RT = fd::RadixTraits<T>;
    auto bits_eq = [](const std::vector<T>& a, const std::vector<T>& b) {
        return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
    };
    for (std::size_t n : {std::size_t(200), std::size_t(1000), std::size_t(4097), std::size_t(70001)}) {
        for (std::size_t lo : {std::size_t(0), std::size_t(7), n / 2}) {
            for (int desc = 0; desc < 2; ++desc) {
                for (int mode = 0; mode < 6; ++mode) {
                    if (mode >= 3 && !fp) continue;
                    std::vector<T> v(n);
                    for (std::size_t j = 0; j < n; ++j) v[j] = from_u64<T>(rng() % 1000);
                    // tail non-increasing in sort order (non-decreasing if desc)
                    std::sort(v.begin() + static_cast<std::ptrdiff_t>(lo), v.end());
                    if (!desc) std::reverse(v.begin() + static_cast<std::ptrdiff_t>(lo), v.end());
                    const std::size_t m = n - lo;
                    const std::size_t b = lo + rng() % (m - 1);
                    if (mode == 1) std::swap(v[b], v[b + 1]);                                      // maybe a break
                    if (mode == 2) v[lo + (rng() % 2 ? 0 : m - 1)] = desc ? from_u64<T>(5000) : from_u64<T>(0);   // end break
                    if constexpr (fp) {
                        if (mode == 3) {   // a run of +0 then -0 (key order valid for the ascending sort)
                            const std::size_t z = std::min<std::size_t>(m / 2, 40);
                            for (std::size_t k = 0; k < z; ++k) v[n - 1 - k] = (k < z / 2) == !desc ? T(-0.0) : T(0.0);
                            for (std::size_t k = lo; k + z < n; ++k) v[k] = desc ? -std::fabs(v[k]) - 1 : std::fabs(v[k]) + 1;
                            if (desc) std::sort(v.begin() + static_cast<std::ptrdiff_t>(lo), v.end() - static_cast<std::ptrdiff_t>(z));
                            else std::sort(v.begin() + static_cast<std::ptrdiff_t>(lo), v.end() - static_cast<std::ptrdiff_t>(z), std::greater<T>());
                        }
                        if (mode == 4) v[desc ? n - 1 : lo] = std::numeric_limits<T>::quiet_NaN();   // NaN at the key-max end
                        if (mode == 5) v[b] = std::numeric_limits<T>::quiet_NaN();                    // NaN inside
                    }
                    // key-order verdict
                    bool keyok = true;
                    for (std::size_t k = lo; k + 1 < n; ++k) {
                        const auto a = RT::encode(v[k]), c = RT::encode(v[k + 1]);
                        if (desc ? (c < a) : (a < c)) keyok = false;
                    }
                    const std::vector<T> orig = v;
                    {   // no-undo instance: same verdict; else a permutation
                        std::vector<T> u = v;
                        const int ru = fd::reverse_tail_checked<T, false>(u.data(), lo, n, desc != 0);
                        std::vector<T> su = u, so = orig;
                        auto kl = [](T a, T c) { return RT::encode(a) < RT::encode(c); };
                        std::sort(su.begin(), su.end(), kl);
                        std::sort(so.begin(), so.end(), kl);
                        bool okp = bits_eq(su, so) && std::equal(u.begin(), u.begin() + static_cast<std::ptrdiff_t>(lo), orig.begin());
                        if (ru == 0) {
                            std::vector<T> want = orig;
                            std::reverse(want.begin() + static_cast<std::ptrdiff_t>(lo), want.end());
                            okp = okp && keyok && bits_eq(u, want);
                        } else {
                            okp = okp && (ru == 2 ? (fp || !fd::use_avx512()) : !keyok);
                        }
                        if (!okp) {
                            std::printf("  FAIL: %s reverse_tail no-undo n=%zu lo=%zu desc=%d mode=%d r=%d\n", tname, n, lo, desc, mode, ru);
                            ++failures;
                        }
                        ++checks;
                    }
                    const int r = fd::reverse_tail_checked(v.data(), lo, n, desc != 0);
                    bool ok;
                    if (r == 0) {
                        std::vector<T> want = orig;
                        std::reverse(want.begin() + static_cast<std::ptrdiff_t>(lo), want.end());
                        ok = keyok && bits_eq(v, want);
                    } else {
                        ok = bits_eq(v, orig) && (r == 2 ? (fp || !fd::use_avx512()) : !keyok);
                    }
                    if (!ok) {
                        std::printf("  FAIL: %s reverse_tail n=%zu lo=%zu desc=%d mode=%d r=%d keyok=%d\n", tname, n, lo,
                                    desc, mode, r, static_cast<int>(keyok));
                        ++failures;
                    }
                    ++checks;
                }
            }
        }
    }
    // Whole sorts of reverse-like ranges (the dispatcher's fused reverse:
    // exact reverses, breaks near either end and mid-way, equal runs,
    // +-0 / NaN), ascending and descending, bitwise against key order.
    for (std::size_t n : {std::size_t(20000), std::size_t(100000), std::size_t(300001)}) {
        for (int shape = 0; shape < 8; ++shape) {
            if (shape >= 6 && !fp) continue;
            for (int desc = 0; desc < 2; ++desc) {
                if (desc && shape == 7) continue;   // NaN placement is specified for the ascending order
                std::vector<T> v(n);
                for (std::size_t j = 0; j < n; ++j) v[j] = from_u64<T>(rng() % (shape == 1 ? 50 : 1000000));
                auto kl = [](T a, T c) { return RT::encode(a) < RT::encode(c); };
                if (shape == 6) for (std::size_t j = 0; j < n; j += 5) v[j] = (j / 5) % 2 ? T(0.0) : T(-0.0);
                if (shape == 7) for (std::size_t j = 0; j < n; j += 1001) v[j] = std::numeric_limits<T>::quiet_NaN();
                std::sort(v.begin(), v.end(), kl);
                if (!desc) std::reverse(v.begin(), v.end());   // input reversed w.r.t. the sort order
                if (shape == 2) std::swap(v[n / 2], v[n / 2 + 1]);
                if (shape == 3) std::swap(v[3], v[n - 5]);
                if (shape == 4) std::swap(v[n - 2], v[n - 1]);
                if (shape == 5) for (int k = 0; k < 8; ++k) std::swap(v[rng() % n], v[rng() % n]);
                std::vector<T> want = v;
                std::sort(want.begin(), want.end(), kl);
                if (desc) std::reverse(want.begin(), want.end());
                fyx::Options o;
                o.parallel = fyx::Tri::Off;
                if (desc) fyx::sort(v.data(), n, std::greater<T>(), o);
                else fyx::sort(v.data(), n, o);
                // equal keys (+-0 aside, which key order separates) may permute: compare keys
                bool ok = v.size() == want.size();
                for (std::size_t j = 0; ok && j < n; ++j)
                    ok = RT::encode(v[j]) == RT::encode(want[j]) || (v[j] != v[j] && want[j] != want[j]) ||
                         (desc && v[j] == want[j]);   // std::greater leaves +-0 unordered
                if (!ok) {
                    std::printf("  FAIL: %s fused reverse sort n=%zu shape=%d desc=%d\n", tname, n, shape, desc);
                    ++failures;
                }
                ++checks;
            }
        }
    }
    if constexpr (fp) {
        // Whole organ pipes: ascending (key order) front, reversed back half
        // holding the top keys incl. +-0 / NaN -- bitwise against key order.
        for (std::size_t n : {std::size_t(1000), std::size_t(100000), std::size_t(300001)}) {
            for (int mode = 0; mode < 3; ++mode) {
                std::vector<T> v(n);
                for (std::size_t j = 0; j < n; ++j) v[j] = from_u64<T>(rng() % 100000) - T(50000);
                if (mode >= 1) for (std::size_t j = 0; j < n; j += 7) v[j] = (j / 7) % 2 ? T(0.0) : T(-0.0);
                if (mode == 2) for (std::size_t j = 3; j < n; j += 997) v[j] = std::numeric_limits<T>::quiet_NaN();
                std::sort(v.begin(), v.end(), [](T a, T c) { return RT::encode(a) < RT::encode(c); });
                const std::vector<T> want = v;
                std::reverse(v.begin() + static_cast<std::ptrdiff_t>(n / 2), v.end());
                fyx::Options o;
                o.parallel = fyx::Tri::Off;
                fyx::sort(v.data(), n, o);
                if (!bits_eq(v, want)) {
                    std::printf("  FAIL: %s organ +-0/NaN n=%zu mode=%d\n", tname, n, mode);
                    ++failures;
                }
                ++checks;
            }
        }
    }
    std::printf("  %-8s checked tail reverse ok\n", tname);
}

int main() {
    std::printf("t_vsort: AVX-512 vectorised quicksort\n");
    std::printf("  kernel present for int32=%d, usable at 70000=%d\n",
                static_cast<int>(fd::vqsort_kernel_supported_v<std::int32_t>),
                static_cast<int>(fd::vqsort_usable<std::int32_t>(70000)));

    run_type<std::int32_t>("int32");
    run_type<std::uint32_t>("uint32");
    run_type<std::int64_t>("int64");
    run_type<std::uint64_t>("uint64");
    run_type<float>("float");
    run_type<double>("double");
    run_all_equal_scan<std::int32_t>("int32");
    run_all_equal_scan<std::uint32_t>("uint32");
    run_all_equal_scan<std::int64_t>("int64");
    run_all_equal_scan<std::uint64_t>("uint64");
    run_all_equal_scan<float>("float");
    run_all_equal_scan<double>("double");
    run_reverse_tail<std::int32_t>("int32");
    run_reverse_tail<std::uint32_t>("uint32");
    run_reverse_tail<std::int64_t>("int64");
    run_reverse_tail<std::uint64_t>("uint64");
    run_reverse_tail<float>("float");
    run_reverse_tail<double>("double");

    run_float_edge<float>("float");
    run_float_edge<double>("double");

    run_leaf_and_prescan<std::int32_t>("int32");
    run_leaf_and_prescan<std::uint32_t>("uint32");
    run_leaf_and_prescan<std::int64_t>("int64");
    run_leaf_and_prescan<std::uint64_t>("uint64");
    run_leaf_and_prescan<float>("float");
    run_leaf_and_prescan<double>("double");

    run_sparse_descent_shapes<std::int32_t>("int32");
    run_sparse_descent_shapes<std::uint64_t>("uint64");
    run_sparse_descent_shapes<float>("float");
    run_sparse_descent_shapes<double>("double");

    run_new_small_paths<std::int32_t>("int32");
    run_new_small_paths<std::uint32_t>("uint32");
    run_new_small_paths<std::int64_t>("int64");
    run_new_small_paths<std::uint64_t>("uint64");
    run_new_small_paths<float>("float");
    run_new_small_paths<double>("double");

    run_few_valued<std::int32_t>("int32");
    run_few_valued<std::uint32_t>("uint32");
    run_few_valued<std::int64_t>("int64");
    run_few_valued<std::uint64_t>("uint64");
    run_few_valued<float>("float");
    run_few_valued<double>("double");

    std::printf("checks=%d failures=%d\n", checks, failures);
    if (failures) { std::printf("VSORT TEST FAILURES=%d\n", failures); return 1; }
    std::printf("ALL VSORT TESTS PASS\n");
    return 0;
}
