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
        0, 1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000,
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
        CHECK(fd::vqsort_range_clean(v.data(), v.size()), "clean range accepted");

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

    run_float_edge<float>("float");
    run_float_edge<double>("double");

    std::printf("checks=%d failures=%d\n", checks, failures);
    if (failures) { std::printf("VSORT TEST FAILURES=%d\n", failures); return 1; }
    std::printf("ALL VSORT TESTS PASS\n");
    return 0;
}
