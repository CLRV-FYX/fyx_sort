// Local FYX vs IPS4o comparison harness.
// Build sequential IPS4o with e.g.:
//   g++ -std=c++17 -O3 -march=native -DNDEBUG -DFYX_DISABLE_PARALLEL \
//       -I. -I/path/to/ips4o/include bench/bench_ips4o_compare.cpp -o /tmp/bench_ips4o_seq
// For FYX parallel-only sanity, compile a separate FYX-only benchmark; upstream
// IPS4o parallel needs TBB headers in this environment.
#include "../fyx_sort.hpp"
#include <ips4o.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

struct Rec {
    int key;
    int payload;
    bool operator==(const Rec& o) const { return key == o.key && payload == o.payload; }
};

struct RecByKey {
    bool operator()(const Rec& a, const Rec& b) const { return a.key < b.key; }
};

static bool same_rec_multiset(const std::vector<Rec>& a, const std::vector<Rec>& b) {
    if (a.size() != b.size()) return false;
    auto total = [](const Rec& x, const Rec& y) {
        return x.payload < y.payload || (x.payload == y.payload && x.key < y.key);
    };
    std::vector<Rec> ca = a, cb = b;
    std::sort(ca.begin(), ca.end(), total);
    std::sort(cb.begin(), cb.end(), total);
    return ca == cb;
}

template <class T, class Comp>
static bool sorted_and_permutation_equiv(const std::vector<T>& got, const std::vector<T>& base, Comp comp) {
    if (!std::is_sorted(got.begin(), got.end(), comp)) return false;
    if constexpr (std::is_same<T, Rec>::value) {
        return same_rec_multiset(got, base);
    } else {
        std::vector<T> a = got, b = base;
        auto total = [](const T& x, const T& y) { return x < y; };
        std::sort(a.begin(), a.end(), total);
        std::sort(b.begin(), b.end(), total);
        return a == b;
    }
}

template <class T, class Comp>
static double time_one(const std::vector<T>& base, Comp comp,
                       const char* algo, bool& ok) {
    std::vector<T> v = base;
    auto t0 = std::chrono::steady_clock::now();
    if (std::string(algo) == "fyx") {
        fyx::Options o;
#ifdef FYX_BENCH_FYX_PARALLEL
        o.parallel = fyx::Tri::On;
#else
        o.parallel = fyx::Tri::Off;
#endif
        fyx::sort(v, comp, o);
    } else if (std::string(algo) == "ips4o") {
#ifdef FYX_BENCH_IPS4O_PARALLEL
        ips4o::parallel::sort(v.begin(), v.end(), comp);
#else
        ips4o::sort(v.begin(), v.end(), comp);
#endif
    } else {
        std::sort(v.begin(), v.end(), comp);
    }
    auto t1 = std::chrono::steady_clock::now();
    ok = sorted_and_permutation_equiv(v, base, comp);
    return std::chrono::duration<double>(t1 - t0).count();
}

template <class T, class Comp>
static void run_case(const char* name, const std::vector<T>& base, Comp comp, int repeats = 3) {
    double best_fyx = std::numeric_limits<double>::infinity();
    double best_ips = std::numeric_limits<double>::infinity();
    double best_std = std::numeric_limits<double>::infinity();
    bool ok_fyx = true, ok_ips = true, ok_std = true;
    for (int r = 0; r < repeats; ++r) {
        bool ok = false;
        double t = time_one(base, comp, "fyx", ok);   best_fyx = std::min(best_fyx, t); ok_fyx = ok_fyx && ok;
        t = time_one(base, comp, "ips4o", ok);        best_ips = std::min(best_ips, t); ok_ips = ok_ips && ok;
        t = time_one(base, comp, "std", ok);          best_std = std::min(best_std, t); ok_std = ok_std && ok;
    }
    std::printf("| %-30s | %9zu | %8.4f | %8.4f | %8.4f | %8.2fx | %8.2fx | %s/%s/%s |\n",
                name, base.size(), best_fyx, best_ips, best_std,
                best_ips / best_fyx, best_std / best_fyx,
                ok_fyx ? "OK" : "BAD", ok_ips ? "OK" : "BAD", ok_std ? "OK" : "BAD");
}

static std::string random_string(std::mt19937_64& rng, std::size_t len) {
    std::string s(len, 'a');
    for (char& c : s) c = static_cast<char>('a' + (rng() % 26));
    return s;
}

int main() {
    std::mt19937_64 rng(0x20260826ULL);
#ifdef FYX_BENCH_FYX_PARALLEL
#  ifdef FYX_BENCH_IPS4O_PARALLEL
    std::printf("FYX(parallel=On) vs IPS4o(parallel) comparison; best of 3, copy time excluded.\n");
#  else
    std::printf("FYX(parallel=On) vs IPS4o(sequential) comparison; best of 3, copy time excluded.\n");
#  endif
#else
#  ifdef FYX_BENCH_IPS4O_PARALLEL
    std::printf("FYX(sequential) vs IPS4o(parallel) comparison; best of 3, copy time excluded.\n");
#  else
    std::printf("FYX(sequential) vs IPS4o(sequential) comparison; best of 3, copy time excluded.\n");
#  endif
#endif
    std::printf("| case                           |         n |      fyx |    ips4o |      std | ips/fyx  | std/fyx  | ok f/i/s |\n");
    std::printf("|--------------------------------|-----------|----------|----------|----------|----------|----------|----------|\n");

    {
        std::vector<std::int32_t> v(5'000'000);
        for (auto& x : v) x = static_cast<std::int32_t>(rng());
        run_case("i32 random", v, std::less<std::int32_t>{});
    }
    {
        std::vector<std::int32_t> v(5'000'000);
        for (auto& x : v) x = static_cast<std::int32_t>(rng() % 16) - 8;
        run_case("i32 low distinct 16", v, std::less<std::int32_t>{});
    }
    {
        std::vector<std::int64_t> keys(256);
        for (std::size_t i = 0; i < keys.size(); ++i)
            keys[i] = (static_cast<std::int64_t>(i) - 128) * 1000000000037LL;
        std::vector<std::int64_t> v(3'000'000);
        for (auto& x : v) x = keys[rng() % keys.size()];
        run_case("i64 sparse distinct 256", v, std::less<std::int64_t>{});
    }
    {
        std::vector<std::string> v(1'000'000);
        for (auto& s : v) s = random_string(rng, 16);
        run_case("string random len16", v, std::less<std::string>{});
    }
    {
        std::vector<std::string> keys(64);
        for (std::size_t i = 0; i < keys.size(); ++i) keys[i] = "key_" + std::to_string(i);
        std::vector<std::string> v(1'000'000);
        for (auto& s : v) s = keys[rng() % keys.size()];
        run_case("string low distinct 64", v, std::less<std::string>{});
    }
    {
        std::vector<Rec> v(1'000'000);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = {static_cast<int>(rng() % 64), static_cast<int>(i)};
        run_case("struct key distinct 64", v, RecByKey{});
    }
    {
        std::vector<Rec> v(1'000'000);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = {static_cast<int>(rng() % 1'000'000), static_cast<int>(i)};
        run_case("struct key random 1M", v, RecByKey{});
    }
}
