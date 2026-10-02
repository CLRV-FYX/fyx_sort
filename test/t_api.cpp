// Public-API conformance test for fyx_sort.hpp
// Verifies fyx::sort / stable_sort / partial_sort / nth_element against the
// standard library across types, sizes, distributions and comparators, plus
// every overload shape and the C ABI.
#include "../fyx_sort.hpp"

#include <cstdio>
#include <vector>
#include <array>
#include <string>
#include <deque>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>

using namespace fyx;

static int failures = 0;
static int checks   = 0;

#define CHECK(cond, msg) do {                                          \
    ++checks;                                                          \
    if (!(cond)) { printf("  FAIL: %s\n", msg); ++failures; }          \
} while (0)

struct NonDefaultMoveOnly {
    inline static int live = 0;
    int key;
    int index;
    NonDefaultMoveOnly() = delete;
    NonDefaultMoveOnly(int k, int i) : key(k), index(i) { ++live; }
    NonDefaultMoveOnly(const NonDefaultMoveOnly&) = delete;
    NonDefaultMoveOnly& operator=(const NonDefaultMoveOnly&) = delete;
    NonDefaultMoveOnly(NonDefaultMoveOnly&& other) noexcept
        : key(other.key), index(other.index) { ++live; }
    NonDefaultMoveOnly& operator=(NonDefaultMoveOnly&&) noexcept = default;
    ~NonDefaultMoveOnly() { --live; }
};

// --------------------------------------------------------------------------
// Generic sort-correctness harness.
//   gen(i, rng)       -> value for position i
//   expectedLess      -> sort ascending and compare to std::sort
//   expectedGreater   -> sort descending and compare to std::sort(greater)
// --------------------------------------------------------------------------
template <class T>
bool elem_eq(const T& a, const T& b) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(a) && std::isnan(b)) return true;   // both NaN -> equal
        if (a == T(0) && b == T(0))           return true;   // -0 / +0
        return a == b;
    } else {
        return a == b;
    }
}
template <class T>
bool vec_eq(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) if (!elem_eq(a[i], b[i])) return false;
    return true;
}

template <class T, class Gen>
void check_sort_family(const char* name, std::size_t n, Gen gen,
                       bool with_nan = false) {
    std::mt19937_64 rng(0x1234 + n);
    std::vector<T> a(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = gen(i, rng);

    // ---- ascending, container overload ----
    {
        std::vector<T> ref = a;
        std::sort(ref.begin(), ref.end());
        std::vector<T> got = a;
        fyx::sort(got);
        CHECK(vec_eq(got, ref), (std::string(name) + " container ascending").c_str());
    }
    // ---- ascending, iterator pair ----
    {
        std::vector<T> ref = a;
        std::sort(ref.begin(), ref.end());
        std::vector<T> got = a;
        fyx::sort(got.begin(), got.end());
        CHECK(vec_eq(got, ref), (std::string(name) + " iterator ascending").c_str());
    }
    // ---- ascending, pointer + length ----
    {
        std::vector<T> ref = a;
        std::sort(ref.begin(), ref.end());
        std::vector<T> got = a;
        fyx::sort(got.data(), got.size());
        CHECK(vec_eq(got, ref), (std::string(name) + " ptr+len ascending").c_str());
    }
    // ---- ascending, Options (parallel on) ----
    {
        std::vector<T> ref = a;
        std::sort(ref.begin(), ref.end());
        std::vector<T> got = a;
        Options o; o.parallel = Tri::On;
        fyx::sort(got, o);
        CHECK(vec_eq(got, ref), (std::string(name) + " options parallel ascending").c_str());
    }
    // ---- ascending, Options (serial) ----
    {
        std::vector<T> ref = a;
        std::sort(ref.begin(), ref.end());
        std::vector<T> got = a;
        Options o; o.parallel = Tri::Off;
        fyx::sort(got.begin(), got.end(), o);
        CHECK(vec_eq(got, ref), (std::string(name) + " options serial ascending").c_str());
    }
    // ---- descending, custom comparator ----
    {
        std::vector<T> ref = a;
        std::sort(ref.begin(), ref.end(), std::greater<T>());
        std::vector<T> got = a;
        fyx::sort(got, std::greater<T>());
        CHECK(vec_eq(got, ref), (std::string(name) + " descending").c_str());
    }
    (void)with_nan;
}

// A comparator that is NOT a default "<" so the radix path must be skipped.
struct abs_less {
    template <class T> bool operator()(const T& a, const T& b) const {
        return std::abs(a) < std::abs(b);
    }
};

int main() {
    printf("== fyx::sort public API ==\n");

    // Sizes spanning the network / radix / pdqsort boundaries.
    std::size_t sizes[] = {0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64,
                           65, 100, 255, 256, 1000, 4096, 50000, 200000};

    for (std::size_t n : sizes) {
        // int32
        check_sort_family<int32_t>("i32", n, [](std::size_t, std::mt19937_64& r){
            return static_cast<int32_t>(r() % 1000000) - 500000; });
        // uint32
        check_sort_family<uint32_t>("u32", n, [](std::size_t, std::mt19937_64& r){
            return static_cast<uint32_t>(r()); });
        // int64
        check_sort_family<int64_t>("i64", n, [](std::size_t, std::mt19937_64& r){
            return static_cast<int64_t>(r()) - 5000000000LL; });
        // float (finite + inf / -inf; no NaN, which std::sort treats as UB)
        check_sort_family<float>("f32", n, [](std::size_t i, std::mt19937_64& r){
            int m = static_cast<int>(i % 7);
            if (m == 0) return std::numeric_limits<float>::infinity();
            if (m == 1) return -std::numeric_limits<float>::infinity();
            int v = static_cast<int>(r() % 4001) - 2000;
            return static_cast<float>(v) / 7.0f; });
        // double
        check_sort_family<double>("f64", n, [](std::size_t, std::mt19937_64& r){
            return static_cast<double>(static_cast<int64_t>(r())) / 1000.0; });
    }

    // ---- non-numeric types force the comparison (pdqsort) path ----
    printf("  strings\n");
    {
        std::vector<std::string> a = {"pear", "apple", "banana", "apple", "cherry", "", "zebra"};
        std::vector<std::string> ref = a; std::sort(ref.begin(), ref.end());
        fyx::sort(a);
        CHECK(a == ref, "string container");
        fyx::sort(a.begin(), a.end(), std::greater<std::string>());
        std::sort(ref.begin(), ref.end(), std::greater<std::string>());
        CHECK(a == ref, "string descending");
    }
    printf("  custom (abs) comparator on int\n");
    {
        std::vector<int> a = {3, -7, 2, -2, 5, -1, 0, -9};
        std::vector<int> ref = a; std::sort(ref.begin(), ref.end(), abs_less{});
        fyx::sort(a, abs_less{});
        CHECK(a == ref, "abs comparator");
    }
    printf("  C array + pointer/iterator pairs\n");
    {
        int a[50];
        std::mt19937_64 rng(7);
        for (int& x : a) x = static_cast<int>(rng() % 1000);
        int ref[50]; std::copy(a, a + 50, ref); std::sort(ref, ref + 50);
        fyx::sort(a, a + 50);
        CHECK(std::equal(a, a + 50, ref), "C array iterator pair");
        int b[50]; std::copy(ref, ref + 50, b); std::sort(b, b + 50);
        fyx::sort(b, 50);
        CHECK(std::equal(b, b + 50, ref), "C array ptr+len");
    }
    printf("  std::array / std::deque\n");
    {
        std::array<int, 37> ar; std::mt19937_64 rng(9);
        for (int& x : ar) x = static_cast<int>(rng() % 500);
        auto ref = ar; std::sort(ref.begin(), ref.end());
        fyx::sort(ar);
        CHECK(ar == ref, "std::array");
        std::deque<int> dq(ar.begin(), ar.end());
        std::deque<int> dref = dq; std::sort(dref.begin(), dref.end());
        fyx::sort(dq.begin(), dq.end());   // non-contiguous -> comparison path
        CHECK(dq == dref, "std::deque iterators");
    }

    // ---- stable_sort: numeric ascending (radix, stable) ----
    printf("  stable_sort numeric ascending\n");
    {
        std::mt19937_64 rng(11);
        for (std::size_t n : {0u,1u,2u,10u,100u,1000u,50000u}) {
            std::vector<int> a(n);
            for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<int>(rng() % 20); // many dups
            std::vector<int> ref = a; std::stable_sort(ref.begin(), ref.end());
            fyx::stable_sort(a);
            CHECK(a == ref, (std::string("stable i32 n=") + std::to_string(n)).c_str());
        }
    }
    printf("  stable_sort stability (key,index) pairs\n");
    {
        using P = std::pair<int, int>;
        std::mt19937_64 rng(13);
        auto cmp = [](const P& a, const P& b) { return a.first < b.first; };
        for (std::size_t n : {0u,1u,5u,50u,2000u}) {
            std::vector<P> a(n);
            for (std::size_t i = 0; i < n; ++i) a[i] = {static_cast<int>(rng()%15), static_cast<int>(i)};
            std::vector<P> ref = a; std::stable_sort(ref.begin(), ref.end(), cmp);
            fyx::stable_sort(a, cmp);
            CHECK(a == ref, (std::string("stable pairs n=") + std::to_string(n)).c_str());
        }
        // descending stable
        auto gcmp = [](const P& a, const P& b) { return a.first > b.first; };
        {
            std::vector<P> a(500);
            for (std::size_t i = 0; i < 500; ++i) a[i] = {static_cast<int>(i%15), static_cast<int>(i)};
            std::vector<P> ref = a; std::stable_sort(ref.begin(), ref.end(), gcmp);
            fyx::stable_sort(a, gcmp);
            CHECK(a == ref, "stable pairs descending");
        }
        {
            std::vector<P> a = {{3, 0}, {2, 1}, {2, 2}, {1, 3}};
            std::vector<P> ref = a;
            std::stable_sort(ref.begin(), ref.end(), cmp);
            fyx::stable_sort(a, cmp);
            CHECK(a == ref, "strict-reverse fast path does not reverse equal keys");
        }
    }

    // stable_sort's fallback must support move-only, non-default-constructible
    // values when a custom comparator selects the generic merge path.
    printf("  stable_sort move-only/non-default type\n");
    CHECK(NonDefaultMoveOnly::live == 0, "move-only test starts without live objects");
    {
        std::vector<NonDefaultMoveOnly> values;
        for (int i = 0; i < 1000; ++i) values.emplace_back((i * 37) % 19, i);
        fyx::stable_sort(values.begin(), values.end(),
                         [](const auto& a, const auto& b) { return a.key < b.key; });
        bool ok = true;
        for (std::size_t i = 1; i < values.size(); ++i) {
            if (values[i].key < values[i - 1].key ||
                (values[i].key == values[i - 1].key && values[i].index < values[i - 1].index)) {
                ok = false;
                break;
            }
        }
        CHECK(ok, "stable_sort move-only/non-default values");
        CHECK(NonDefaultMoveOnly::live == static_cast<int>(values.size()),
              "stable_sort releases scratch objects after successful merge");
    }
    CHECK(NonDefaultMoveOnly::live == 0, "successful stable_sort destroys each object once");
    {
        // Two individually sorted, overlapping runs exercise a large stable
        // merge and verify that equal keys retain their original order.
        std::vector<NonDefaultMoveOnly> values;
        constexpr int half = 4096;
        for (int i = 0; i < half; ++i) values.emplace_back(1000 + i, i);
        for (int i = 0; i < half; ++i) values.emplace_back(i, half + i);
        fyx::stable_sort(values.begin(), values.end(),
                         [](const auto& a, const auto& b) { return a.key < b.key; });
        bool ok = true;
        for (std::size_t i = 1; i < values.size(); ++i) {
            if (values[i].key < values[i - 1].key ||
                (values[i].key == values[i - 1].key && values[i].index < values[i - 1].index)) {
                ok = false;
                break;
            }
        }
        CHECK(ok, "stable merge preserves stability for move-only values");
        CHECK(NonDefaultMoveOnly::live == static_cast<int>(values.size()),
              "stable merge releases scratch objects");
    }
    CHECK(NonDefaultMoveOnly::live == 0, "stable merge destroys each object once");
    {
        // Many overlapping natural runs cross several recursive partition levels.
        constexpr int count = 8192;
        constexpr int run_count = 65;
        std::vector<NonDefaultMoveOnly> values;
        for (int run = 0; run < run_count; ++run) {
            const int begin = count * run / run_count;
            const int end = count * (run + 1) / run_count;
            const int key_base = (run_count - run - 1) * (count / run_count) / 2;
            for (int i = begin; i < end; ++i)
                values.emplace_back(key_base + i - begin, i);
        }
        fyx::stable_sort(values.begin(), values.end(),
                         [](const auto& a, const auto& b) { return a.key < b.key; });
        bool ok = true;
        for (std::size_t i = 1; i < values.size(); ++i) {
            if (values[i].key < values[i - 1].key ||
                (values[i].key == values[i - 1].key && values[i].index < values[i - 1].index)) {
                ok = false;
                break;
            }
        }
        CHECK(ok, "stable merge preserves order across many natural runs");
        CHECK(NonDefaultMoveOnly::live == static_cast<int>(values.size()),
              "many-run stable merge releases scratch objects");
    }
    CHECK(NonDefaultMoveOnly::live == 0, "many-run stable merge destroys each object once");
    {
        std::vector<NonDefaultMoveOnly> values;
        for (int i = 0; i < 8192; ++i) values.emplace_back(8191 - i, i);
        fyx::stable_sort(values.begin(), values.end(),
                         [](const auto& a, const auto& b) { return a.key < b.key; });
        bool sorted = true;
        for (std::size_t i = 1; i < values.size(); ++i)
            if (values[i].key < values[i - 1].key) sorted = false;
        CHECK(sorted, "stable_sort orders a strict descending move-only range");
        CHECK(NonDefaultMoveOnly::live == static_cast<int>(values.size()),
              "descending move-only range releases scratch objects");
    }
    CHECK(NonDefaultMoveOnly::live == 0, "descending move-only range destroys each object once");
    {
        std::vector<NonDefaultMoveOnly> values;
        constexpr int half = 4096;
        for (int i = 0; i < half; ++i) values.emplace_back(1000 + i, i);
        for (int i = 0; i < half; ++i) values.emplace_back(i, half + i);
        bool caught = false;
        try {
            fyx::stable_sort(values.begin(), values.end(), [](const auto& a, const auto& b) {
                if (a.key == 5 && b.key == 1000) throw 11; // a merge-only comparison
                return a.key < b.key;
            });
        } catch (int) {
            caught = true;
        }
        CHECK(caught, "stable merge propagates comparator exceptions");
        CHECK(NonDefaultMoveOnly::live == static_cast<int>(values.size()),
              "stable merge releases partially constructed scratch on exception");
    }
    CHECK(NonDefaultMoveOnly::live == 0, "stable merge exception path destroys each object once");
    {
        std::vector<NonDefaultMoveOnly> values;
        for (int i = 0; i < 128; ++i) values.emplace_back((i * 37) % 19, i);
        int calls = 0;
        bool caught = false;
        try {
            fyx::stable_sort(values.begin(), values.end(), [&](const auto& a, const auto& b) {
                if (++calls == 5) throw 7; // during the first merge, after scratch construction
                return a.key < b.key;
            });
        } catch (int) {
            caught = true;
        }
        CHECK(caught, "stable_sort propagates comparator exceptions");
        CHECK(NonDefaultMoveOnly::live == static_cast<int>(values.size()),
              "stable_sort releases partially constructed scratch on exception");
    }
    CHECK(NonDefaultMoveOnly::live == 0, "exception path destroys each object once");

    // ---- partial_sort ----
    // Validate the contract, including permutation preservation at k==0 and
    // k==n. The tail order is unspecified, so only the sorted smallest prefix
    // is compared with a fully sorted reference.
    printf("  partial_sort\n");
    {
        auto check_partial = [&](const std::vector<int>& got, const std::vector<int>& orig,
                                 const std::vector<int>& sorted, std::size_t k,
                                 const char* tag) {
            const std::size_t nn = got.size();
            CHECK(orig.size() == nn, (std::string(tag) + " size preserved").c_str());
            if (k > nn || orig.size() != nn) return;
            // The output must be a permutation, including the k==0 and k==n
            // boundaries; the tail's order is deliberately unspecified.
            std::vector<int> g = got, o = orig;
            std::sort(g.begin(), g.end()); std::sort(o.begin(), o.end());
            CHECK(g == o, (std::string(tag) + " multiset preserved").c_str());
            CHECK(std::is_sorted(got.begin(), got.begin() + static_cast<std::ptrdiff_t>(k)),
                  (std::string(tag) + " first k sorted").c_str());
            CHECK(std::equal(got.begin(), got.begin() + static_cast<std::ptrdiff_t>(k), sorted.begin()),
                  (std::string(tag) + " first k are smallest").c_str());
            if (k > 0 && k < nn) {
                bool partition = true;
                for (std::size_t i = k; i < nn; ++i)
                    if (got[i] < got[k - 1]) { partition = false; break; }
                CHECK(partition, (std::string(tag) + " partition").c_str());
            }
        };
        std::mt19937_64 rng(17);
        for (std::size_t n : {0u,1u,2u,10u,100u,5000u}) {
            const std::vector<std::size_t> ks = {std::size_t(0), std::size_t(1), n/2,
                                                  (n ? n-1 : std::size_t(0)), n};
            for (std::size_t k : ks) {
                if (k > n) continue;
                std::vector<int> orig(n);
                for (std::size_t i = 0; i < n; ++i) orig[i] = static_cast<int>(rng() % 100000);
                std::vector<int> sorted = orig; std::sort(sorted.begin(), sorted.end());
                std::vector<int> a = orig;
                fyx::partial_sort(a.begin(), a.begin()+static_cast<std::ptrdiff_t>(k), a.end());
                const std::string tag = std::string("partial_sort n=") + std::to_string(n) +
                                        " k=" + std::to_string(k);
                check_partial(a, orig, sorted, k, tag.c_str());
                // container + count form, starting from the same original input
                std::vector<int> b = orig;
                fyx::partial_sort(b, k);
                check_partial(b, orig, sorted, k, "partial_sort container+count");
            }
        }
    }

    // ---- nth_element ----
    printf("  nth_element\n");
    {
        auto check_nth = [&](const std::vector<int>& got, const std::vector<int>& orig,
                             const std::vector<int>& sorted, std::size_t m, const char* tag) {
            std::vector<int> g = got, o = orig;
            std::sort(g.begin(), g.end()); std::sort(o.begin(), o.end());
            CHECK(g == o, (std::string(tag) + " multiset preserved").c_str());
            CHECK(got[m] == sorted[m], (std::string(tag) + " nth value").c_str());
            bool partition = true;
            for (std::size_t i = 0; i < m; ++i) if (got[i] > got[m]) partition = false;
            for (std::size_t i = m + 1; i < got.size(); ++i) if (got[i] < got[m]) partition = false;
            CHECK(partition, (std::string(tag) + " partition").c_str());
        };
        std::mt19937_64 rng(19);
        for (std::size_t n : {0u,1u,2u,3u,10u,77u,1000u,20000u}) {
            const std::vector<std::size_t> ms = {std::size_t(0), std::size_t(1), n/2,
                                                  (n ? n-1 : std::size_t(0))};
            for (std::size_t m : ms) {
                if (m >= n) continue;
                std::vector<int> orig(n);
                for (std::size_t i = 0; i < n; ++i) orig[i] = static_cast<int>(rng() % 100000);
                std::vector<int> sorted = orig; std::sort(sorted.begin(), sorted.end());
                std::vector<int> got = orig;
                fyx::nth_element(got.begin(), got.begin()+static_cast<std::ptrdiff_t>(m), got.end());
                check_nth(got, orig, sorted, m, "nth_element iterator");
                // container + count form
                std::vector<int> gc = orig; fyx::nth_element(gc, m);
                check_nth(gc, orig, sorted, m, "nth_element container+count");
            }
        }
    }

    // ---- NaN handling for floats (radix totalOrder) ----
    printf("  float NaN/-0/+0 ordering\n");
    {
        std::vector<float> v = {1.0f,-1.0f,0.0f,-0.0f,
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
            -std::numeric_limits<float>::denorm_min(),
            std::numeric_limits<float>::denorm_min(),
            3.14159f, -2.71828f, 1e30f, -1e30f};
        std::vector<float> g = v; fyx::sort(g);
        bool ok = true; size_t i = 0;
        for (; i < g.size() && g[i] == g[i]; ++i) if (i && g[i-1] > g[i]) ok = false;
        for (size_t j = i; j < g.size(); ++j) if (g[j] == g[j]) ok = false;  // NaN at end
        CHECK(ok, "float NaN at end + finite nondecreasing");
        std::vector<float> z = {0.0f,-0.0f,0.0f,-0.0f};
        fyx::sort(z);
        bool zok = std::signbit(z[0]) && std::signbit(z[1]) && !std::signbit(z[2]) && !std::signbit(z[3]);
        CHECK(zok, "-0 before +0");
    }

    // ---- C ABI ----
    printf("  C ABI fyx_sort_int32\n");
    {
        std::mt19937_64 rng(23);
        std::vector<int32_t> a(1000);
        for (auto& x : a) x = static_cast<int32_t>(rng() % 5000);
        std::vector<int32_t> ref = a; std::sort(ref.begin(), ref.end());
        int rc = fyx_sort_int32(a.data(), a.size());
        CHECK(rc == 0, "fyx_sort_int32 return code");
        CHECK(a == ref, "fyx_sort_int32 result");
    }
    {
        std::vector<double> a(500); std::mt19937_64 rng(29);
        for (auto& x : a) x = static_cast<double>(rng()%1000)/7.0;
        std::vector<double> ref = a; std::sort(ref.begin(), ref.end());
        CHECK(fyx_sort_double(a.data(), a.size()) == 0, "fyx_sort_double rc");
        CHECK(a == ref, "fyx_sort_double result");
    }

    // ---- Options.gpu path (falls back to CPU when no GPU backend) ----------
    printf("  Options.gpu=true (CPU fallback)\n");
    {
        std::mt19937_64 rng(41);
        for (std::size_t n : {0u, 1u, 50u, 1000u, 50000u}) {
            std::vector<int> a(n);
            for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<int>(rng() % 100000);
            std::vector<int> ref = a; std::sort(ref.begin(), ref.end());
            Options og; og.gpu = true; og.parallel = Tri::Off;
            fyx::sort(a, og);
            CHECK(a == ref, (std::string("gpu-option i32 n=") + std::to_string(n)).c_str());
            std::vector<double> d(n);
            for (std::size_t i = 0; i < n; ++i) d[i] = static_cast<double>(rng() % 1000) / 7.0;
            std::vector<double> dr = d; std::sort(dr.begin(), dr.end());
            fyx::sort(d, og);
            CHECK(d == dr, (std::string("gpu-option f64 n=") + std::to_string(n)).c_str());
        }
    }

    printf("checks=%d failures=%d\n", checks, failures);
    if (failures) { printf("API TEST FAILURES=%d\n", failures); return 1; }
    printf("ALL API TESTS PASS\n");
    return 0;
}
