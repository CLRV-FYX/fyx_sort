// Sample-sort (Module C) conformance + sanity test.
// Validates the Eytzinger classifier against brute force, and checks that
// fyx::sort on strings/structs (which now route through sample_sort) matches
// std::sort for both high- and low-distinct data (guard on/off paths).
#include "../fyx_sort.hpp"
#include <cstdio>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>

using namespace fyx;
using namespace fyx::detail;

static int failures = 0;
#define CHECK(c, m) do { if(!(c)){ printf("  FAIL: %s\n", m); ++failures; } } while(0)

// Brute-force bucket of x given sorted splitters: number of splitters <= x.
template <class T, class Comp>
unsigned brute_bucket(const T& x, const std::vector<T>& s, Comp comp) {
    unsigned b = 0;
    for (std::size_t i = 0; i < s.size(); ++i)
        if (!comp(x, s[i])) ++b;   // s[i] <= x
    return b;
}

int main() {
    std::mt19937_64 rng(20240823);

    // ---- 1. classifier tree vs brute force -------------------------------
    printf("classifier tree vs brute force\n");
    {
        const unsigned m = 255;
        std::vector<int> s(m);
        for (unsigned i = 0; i < m; ++i) s[i] = static_cast<int>(rng() % 1000000);
        std::sort(s.begin(), s.end());
        std::vector<int> tree(2 * m + 1);
        build_classifier_tree(tree, s.data(), 1, 0, (int)m - 1, std::less<int>());
        int ok = 1;
        for (int t = 0; t < 20000; ++t) {
            int x = static_cast<int>(rng() % 2000000) - 1000000;
            unsigned a = classify_bucket(x, tree.data(), m, std::less<int>());
            unsigned au = classify_bucket_256(x, tree.data(), std::less<int>());
            unsigned b = brute_bucket(x, s, std::less<int>());
            if (a != b || au != b) { ok = 0; break; }
        }
        CHECK(ok, "classify matches brute force (int)");
    }
    {
        const unsigned m = 255;
        std::vector<std::string> s(m);
        for (unsigned i = 0; i < m; ++i) s[i] = std::to_string(rng() % 100000);
        std::sort(s.begin(), s.end());
        std::vector<std::string> tree(2 * m + 1);
        build_classifier_tree(tree, s.data(), 1, 0, (int)m - 1, std::less<std::string>());
        int ok = 1;
        for (int t = 0; t < 20000; ++t) {
            std::string x = std::to_string(rng() % 200000);
            unsigned a = classify_bucket(x, tree.data(), m, std::less<std::string>());
            unsigned au = classify_bucket_256(x, tree.data(), std::less<std::string>());
            unsigned b = brute_bucket(x, s, std::less<std::string>());
            if (a != b || au != b) { ok = 0; break; }
        }
        CHECK(ok, "classify matches brute force (string)");
    }

    // ---- 2. fyx::sort on strings: high + low distinct --------------------
    auto check_str = [&](const char* tag, const std::vector<std::string>& in) {
        std::vector<std::string> ref = in; std::sort(ref.begin(), ref.end());
        std::vector<std::string> got = in; fyx::sort(got);
        CHECK(got == ref, tag);
    };
    printf("string sorting (sample path)\n");
    {
        // high-distinct: long random strings
        std::vector<std::string> v(300000);
        for (auto& s : v) { s.resize(16); for (char& c : s) c = (char)('a' + (rng() % 26)); }
        check_str("high-distinct strings", v);
    }
    {
        // low-distinct: pick from a tiny alphabet (exercises the guard -> pdqsort)
        std::vector<std::string> v(300000);
        const char* alpha = "abc";
        for (auto& s : v) { s.resize(6); for (char& c : s) c = alpha[rng() % 3]; }
        check_str("low-distinct strings (guard)", v);
    }
    {
        // already sorted / reverse / all-equal
        std::vector<std::string> a(100000), b(100000), c(100000);
        for (size_t i = 0; i < a.size(); ++i) {
            std::string s = std::to_string(i % 50);
            a[i] = s; b[i] = std::to_string(100000 - i); c[i] = "x";
        }
        check_str("sorted strings", a);
        check_str("reverse strings", b);
        check_str("all-equal strings", c);
    }

    // ---- 3. structs + custom comparator ----------------------------------
    printf("struct sorting (sample path)\n");
    {
        struct Rec { int k; int v;
            bool operator==(const Rec& o) const { return k == o.k && v == o.v; } };
        auto cmp = [](const Rec& a, const Rec& b) {
            if (a.k != b.k) return a.k < b.k;
            return a.v < b.v;
        };
        std::vector<Rec> v(300000);
        for (auto& r : v) { r.k = static_cast<int>(rng() % 1000); r.v = static_cast<int>(rng() % 1000); }
        std::vector<Rec> ref = v; std::sort(ref.begin(), ref.end(), cmp);
        fyx::sort(v, cmp);
        CHECK(v == ref, "struct custom comparator");
        // descending
        auto gcmp = [](const Rec& a, const Rec& b) {
            if (a.k != b.k) return a.k > b.k;
            return a.v > b.v;
        };
        std::vector<Rec> v2 = v; std::vector<Rec> ref2 = v2;
        std::sort(ref2.begin(), ref2.end(), gcmp);
        fyx::sort(v2, gcmp);
        CHECK(v2 == ref2, "struct custom comparator descending");
    }

    // ---- 4. performance sketch: sample sort vs pdqsort on strings --------
    printf("perf: fyx::sort vs std::sort on strings\n");
    {
        const std::size_t n = 2'000'000;
        std::vector<std::string> v(n);
        for (auto& s : v) { s.resize(16); for (char& c : s) c = (char)('a' + (rng() % 26)); }
        std::vector<std::string> a = v, b = v;
        auto t0 = std::chrono::steady_clock::now(); fyx::sort(a);
        auto t1 = std::chrono::steady_clock::now(); std::sort(b.begin(), b.end());
        auto t2 = std::chrono::steady_clock::now();
        double tf = std::chrono::duration<double>(t1 - t0).count();
        double ts = std::chrono::duration<double>(t2 - t1).count();
        printf("  fyx::sort=%.3fs  std::sort=%.3fs  speedup=%.2fx\n", tf, ts, ts / tf);
        CHECK(a == b, "string perf reference matches");
    }

    printf(failures ? "SAMPLE SORT FAILURES=%d\n" : "ALL SAMPLE SORT TESTS PASS\n", failures);
    return failures ? 1 : 0;
}
