// Adaptive-order weapon tests (parts/12b): natural-run merge, dirty-patch
// merge and displacement-patch merge.
//
// The weapons are structure detectors, so the tests do three things for every
// shape:
//   1. the public entry point must produce the same multiset as std::sort, in
//      order (for floating point: in the documented IEEE totalOrder, verified
//      through the radix key);
//   2. each weapon called directly must either produce that same result or
//      decline without touching the data;
//   3. weapons that are supposed to decline on high-entropy input must keep
//      declining -- that is what keeps the dispatcher cheap on random data.
#define FYX_ENABLE_TEST_HOOKS 1
#include "../fyx_sort.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace fyx;

static int failures = 0;
static int checks   = 0;

#define CHECK(cond, msg) do {                                          \
    ++checks;                                                          \
    if (!(cond)) { std::printf("  FAIL: %s\n", msg); ++failures; }     \
} while (0)

static std::mt19937_64 rng(0xABCDEF123456789ULL);

// ---------------------------------------------------------------------------
// verification helpers
// ---------------------------------------------------------------------------
template <class T>
static bool same_multiset(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    std::vector<T> x = a, y = b;
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    return x == y;
}

/// Floating point is verified through the radix key so that -0 / +0 and NaN
/// placement match what fyx documents (std::sort cannot express that order).
static bool double_ok(const std::vector<double>& got, const std::vector<double>& src) {
    using RT = fyx::detail::RadixTraits<double>;
    std::vector<std::uint64_t> a(got.size()), b(src.size());
    for (std::size_t i = 0; i < got.size(); ++i) a[i] = RT::encode(got[i]);
    for (std::size_t i = 0; i < src.size(); ++i) b[i] = RT::encode(src[i]);
    std::sort(b.begin(), b.end());
    return a == b;
}

// ---------------------------------------------------------------------------
// shape generators (all built on top of a sorted base so the disorder is
// controlled and reproducible)
// ---------------------------------------------------------------------------
template <class T>
static std::vector<T> shape(const char* name, std::size_t n) {
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<T>(i);
    const std::string s(name);
    if (s == "sorted")        return v;
    if (s == "reversed")      { std::reverse(v.begin(), v.end()); return v; }
    if (s == "allequal")      { for (auto& x : v) x = static_cast<T>(7); return v; }
    if (s == "rotate")        { std::rotate(v.begin(), v.begin() + n / 3, v.end()); return v; }
    if (s == "rotate_back")   { std::rotate(v.begin(), v.begin() + 2 * n / 3, v.end()); return v; }
    // two ascending runs that interleave: even values then odd values
    if (s == "concat2") {
        std::vector<T> a, b;
        for (std::size_t i = 0; i < n; ++i) {
            if (i % 2 == 0) a.push_back(static_cast<T>(i));
            else            b.push_back(static_cast<T>(i));
        }
        std::copy(a.begin(), a.end(), v.begin());
        std::copy(b.begin(), b.end(), v.begin() + a.size());
        return v;
    }
    // organ pipe: descending head, ascending tail
    if (s == "organ") { std::reverse(v.begin(), v.begin() + n / 2); return v; }
    // three runs: asc, desc, asc
    if (s == "three_runs") {
        std::reverse(v.begin() + n / 3, v.begin() + 2 * n / 3);
        return v;
    }
    // local swaps (displacement <= 64)
    if (s == "local_swap") {
        for (std::size_t k = 0; k + 1 < n / 500; ++k) {
            std::size_t i = rng() % (n - 66);
            std::swap(v[i], v[i + 1 + rng() % 64]);
        }
        return v;
    }
    // long-distance swaps
    if (s == "far_swap") {
        for (std::size_t k = 0; k + 1 < n / 500; ++k)
            std::swap(v[rng() % n], v[rng() % n]);
        return v;
    }
    // whole 128-element blocks moved a few blocks away
    if (s == "block_move") {
        const std::size_t bs = 128, nb = n / bs;
        for (int k = 0; k < 16 && nb > 4; ++k) {
            std::size_t i = rng() % (nb - 1);
            std::size_t j = std::min(nb - 1, i + 1 + rng() % 4);
            for (std::size_t t = 0; t < bs; ++t) std::swap(v[i * bs + t], v[j * bs + t]);
        }
        return v;
    }
    // one long segment spliced to the front
    if (s == "splice") {
        std::rotate(v.begin(), v.begin() + n / 7, v.begin() + n / 7 + n / 9);
        return v;
    }
    // random permutation (high entropy: every weapon must decline)
    if (s == "random") {
        std::shuffle(v.begin(), v.end(), rng);
        return v;
    }
    std::printf("  (unknown shape %s)\n", name);
    return v;
}

/// std::string gets the same shapes through fixed-width keys, so the string
/// order is exactly the numeric order of the underlying values.
template <>
std::vector<std::string> shape<std::string>(const char* name, std::size_t n) {
    const auto keys = shape<std::int64_t>(name, n);
    std::vector<std::string> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "key_%020lld", static_cast<long long>(keys[i]));
        v[i] = buf;
    }
    return v;
}

// ---------------------------------------------------------------------------
// 1. public entry point, every shape x type
// ---------------------------------------------------------------------------
template <class T>
static void check_shapes(const char* tname, std::size_t n,
                         bool (*extra)(const std::vector<T>&, const std::vector<T>&)) {
    const char* shapes[] = {"sorted", "reversed", "allequal", "rotate", "rotate_back",
                            "concat2", "organ", "three_runs", "local_swap", "far_swap",
                            "block_move", "splice", "random"};
    for (const char* s : shapes) {
        rng.seed(0xABCDEF123456789ULL);
        const std::vector<T> base = shape<T>(s, n);
        for (int mode = 0; mode < 2; ++mode) {
            std::vector<T> v = base;
            fyx::Options o;
            o.parallel = mode ? fyx::Tri::On : fyx::Tri::Off;
            fyx::sort(v, o);
            const bool ok = extra ? extra(v, base)
                                  : (std::is_sorted(v.begin(), v.end()) && same_multiset(v, base));
            if (!ok) std::printf("  FAIL: %s %s (%s) sort\n", tname, s, mode ? "par" : "ser");
            ++checks;
            if (!ok) ++failures;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. weapons called directly
// ---------------------------------------------------------------------------
template <class T, class Gen>
static void check_weapons_direct(const char* tname, std::size_t n, Gen gen) {
    const char* shapes[] = {"sorted", "reversed", "allequal", "rotate", "concat2", "organ",
                            "three_runs", "local_swap", "far_swap", "block_move", "splice", "random"};
    for (const char* s : shapes) {
        rng.seed(0xABCDEF123456789ULL);
        const std::vector<T> base = gen(s, n);
        const std::vector<T> expect = [&] { std::vector<T> e = base; std::sort(e.begin(), e.end()); return e; }();
        const char* wnames[] = {"natural_run", "dirty_patch", "displacement_patch"};
        for (int w = 0; w < 3; ++w) {
            std::vector<T> v = base;
            bool fired = false;
            switch (w) {
                case 0: fired = fyx::detail::try_natural_run_merge_adaptive(v.data(), v.size(), fyx::less{}); break;
                case 1: fired = fyx::detail::try_dirty_patch_merge_adaptive(v.data(), v.size(), fyx::less{}); break;
                default: fired = fyx::detail::try_displacement_patch_merge_adaptive(v.data(), v.size(), fyx::less{}); break;
            }
            ++checks;
            if (fired && v != expect) {
                std::printf("  FAIL: %s %s weapon %s returned a wrong permutation\n", tname, s, wnames[w]);
                ++failures;
            }
            // A weapon that declines must leave the input untouched.
            ++checks;
            if (!fired && v != base) {
                std::printf("  FAIL: %s %s weapon %s mutated the input while declining\n", tname, s, wnames[w]);
                ++failures;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. cheap rejection on high-entropy input
// ---------------------------------------------------------------------------
static void check_random_declines() {
    const std::size_t n = 1u << 20;
    std::vector<std::int32_t> v(n);
    for (auto& x : v) x = static_cast<std::int32_t>(rng());
    {
        auto w = v;
        CHECK(!fyx::detail::try_natural_run_merge_adaptive(w.data(), w.size(), fyx::less{}),
              "natural-run merge declines on random input");
        CHECK(w == v, "natural-run merge leaves random input untouched");
    }
    {
        auto w = v;
        CHECK(!fyx::detail::try_dirty_patch_merge_adaptive(w.data(), w.size(), fyx::less{}),
              "dirty-patch merge declines on random input");
        CHECK(w == v, "dirty-patch merge leaves random input untouched");
    }
    {
        auto w = v;
        CHECK(!fyx::detail::try_displacement_patch_merge_adaptive(w.data(), w.size(), fyx::less{}),
              "displacement-patch merge declines on random input");
        CHECK(w == v, "displacement-patch merge leaves random input untouched");
    }
}

// ---------------------------------------------------------------------------
// 4. floating point: NaN / -0 / +0 keep the documented total order
// ---------------------------------------------------------------------------
static void check_float_order() {
    const std::size_t n = 200000;
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<double>(i);
    v[n / 4] = std::numeric_limits<double>::quiet_NaN();
    v[n / 2] = -0.0;
    v[n / 2 + 1] = 0.0;
    v[n - 1] = -std::numeric_limits<double>::infinity();
    // rotate so the structured weapons engage
    std::rotate(v.begin(), v.begin() + n / 3, v.end());
    for (int k = 0; k < 200; ++k) std::swap(v[rng() % n], v[rng() % n]);

    std::vector<double> src = v;
    for (int mode = 0; mode < 2; ++mode) {
        std::vector<double> w = v;
        fyx::Options o;
        o.parallel = mode ? fyx::Tri::On : fyx::Tri::Off;
        fyx::sort(w, o);
        CHECK(double_ok(w, src), "double NaN/-0/+0 total order preserved (rotate + swaps)");
    }
}

// ---------------------------------------------------------------------------
// 5. descending and custom comparators
// ---------------------------------------------------------------------------
struct Rec {
    int key;
    int payload;
    bool operator==(const Rec& o) const { return key == o.key && payload == o.payload; }
};

static void check_comparators() {
    const std::size_t n = 300000;
    rng.seed(0xABCDEF123456789ULL);
    std::vector<Rec> base(n);
    for (std::size_t i = 0; i < n; ++i) base[i] = Rec{static_cast<int>(i), static_cast<int>(i)};
    std::rotate(base.begin(), base.begin() + n / 3, base.end());
    for (std::size_t k = 0; k < n / 500; ++k) {
        std::size_t i = rng() % (n - 66);
        std::swap(base[i], base[i + 1 + rng() % 64]);
    }
    auto by_key = [](const Rec& a, const Rec& b) { return a.key < b.key; };
    auto expected = base;
    std::sort(expected.begin(), expected.end(), by_key);

    for (int mode = 0; mode < 2; ++mode) {
        auto v = base;
        fyx::Options o;
        o.parallel = mode ? fyx::Tri::On : fyx::Tri::Off;
        fyx::sort(v, by_key, o);
        CHECK(v == expected, "custom comparator with rotated + locally swapped payloads");
    }

    // descending
    std::vector<int> d(n);
    for (std::size_t i = 0; i < n; ++i) d[i] = static_cast<int>(i);
    std::rotate(d.begin(), d.begin() + n / 5, d.end());
    for (std::size_t k = 0; k < n / 500; ++k) std::swap(d[rng() % n], d[rng() % n]);
    auto src = d;
    for (int mode = 0; mode < 2; ++mode) {
        auto v = d;
        fyx::Options o;
        o.parallel = mode ? fyx::Tri::On : fyx::Tri::Off;
        fyx::sort(v, std::greater<int>{}, o);
        auto ref = src;
        std::sort(ref.begin(), ref.end(), std::greater<int>{});
        CHECK(v == ref, "std::greater with rotated + far-swapped input");
    }
}

// ---------------------------------------------------------------------------
// 6. odd sizes around the thresholds
// ---------------------------------------------------------------------------
static void check_sizes() {
    for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(1023),
                          std::size_t(1024), std::size_t(1025), std::size_t(4097),
                          std::size_t(65537)}) {
        rng.seed(0xABCDEF123456789ULL);
        std::vector<std::int64_t> base(n);
        for (std::size_t i = 0; i < n; ++i) base[i] = static_cast<std::int64_t>(i);
        if (n > 4) {
            std::rotate(base.begin(), base.begin() + n / 3, base.end());
            for (std::size_t k = 0; k + 1 < n / 200; ++k)
                std::swap(base[rng() % n], base[rng() % n]);
        }
        auto ref = base;
        std::sort(ref.begin(), ref.end());
        for (int mode = 0; mode < 2; ++mode) {
            auto v = base;
            fyx::Options o;
            o.parallel = mode ? fyx::Tri::On : fyx::Tri::Off;
            fyx::sort(v, o);
            if (v != ref) {
                std::printf("  FAIL: odd size n=%zu (%s)\n", n, mode ? "par" : "ser");
                ++failures;
            }
            ++checks;
        }
    }
}

int main() {
    std::printf("adaptive: public entry point, every shape\n");
    check_shapes<std::int32_t>("int32", 300000, nullptr);
    check_shapes<std::int64_t>("int64", 300000, nullptr);
    check_shapes<double>("double", 300000, double_ok);
    check_shapes<std::string>("string", 200000, nullptr);

    std::printf("adaptive: weapons called directly\n");
    check_weapons_direct<std::int32_t>("int32", 300000, shape<std::int32_t>);
    check_weapons_direct<std::int64_t>("int64", 300000, shape<std::int64_t>);
    check_weapons_direct<std::string>("string", 200000, shape<std::string>);

    std::printf("adaptive: cheap rejection on high-entropy input\n");
    check_random_declines();

    std::printf("adaptive: floating point total order\n");
    check_float_order();

    std::printf("adaptive: comparators and sizes\n");
    check_comparators();
    check_sizes();

    std::printf("checks=%d failures=%d\n", checks, failures);
    if (failures == 0) { std::printf("ALL ADAPTIVE TESTS PASS\n"); return 0; }
    std::printf("ADAPTIVE TEST FAILURES=%d\n", failures);
    return 1;
}
