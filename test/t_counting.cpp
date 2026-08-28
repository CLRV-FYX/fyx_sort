// Counting-sort fast path conformance tests (weapon two), explicit
// parallel sample-sort dispatch sanity (weapon three), and top-level input
// profiling/dispatch checks (weapon seven).
#define FYX_ENABLE_TEST_HOOKS 1
#include "../fyx_sort.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  FAIL: %s\n", m); ++failures; } } while (0)

template <class T, class Comp = std::less<T>>
static void check_against_std(const char* tag, std::vector<T> v, Comp comp = Comp{}) {
    std::vector<T> ref = v;
    std::sort(ref.begin(), ref.end(), comp);
    fyx::sort(v, comp);
    CHECK(v == ref, tag);
}

int main() {
    std::mt19937_64 rng(0xC0FFEE);

    std::printf("counting: dense integer ranges\n");
    {
        std::vector<int> v(300000);
        for (auto& x : v) x = static_cast<int>(rng() % 17) - 8;
        check_against_std("i32 dense low range", v);
        check_against_std("i32 dense low range descending", v, std::greater<int>{});
    }
    {
        std::vector<unsigned char> v(200000);
        for (auto& x : v) x = static_cast<unsigned char>(rng() & 0xFFu);
        check_against_std("u8 full 256 range", v);
    }

    std::printf("counting: sparse low-cardinality integers\n");
    {
        std::vector<std::int64_t> keys(256);
        for (std::size_t i = 0; i < keys.size(); ++i)
            keys[i] = (static_cast<std::int64_t>(i) - 128) * 1000000000037LL;
        std::vector<std::int64_t> v(350000);
        for (auto& x : v) x = keys[rng() % keys.size()];
        check_against_std("i64 sparse 256 distinct", v);
    }
    {
        // 257 distinct must remain correct even when the compressed counter
        // declines and falls back to the normal path.
        std::vector<std::int64_t> v(120000);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = static_cast<std::int64_t>(i % 257) * 999999937LL - 1234567LL;
        std::shuffle(v.begin(), v.end(), rng);
        check_against_std("i64 sparse 257 distinct fallback", v);
    }
#if FYX_ENABLE_PARALLEL
    {
        std::vector<std::uint32_t> keys;
        keys.reserve(256);
        while (keys.size() < 256) {
            const std::uint32_t x = static_cast<std::uint32_t>(rng());
            if (std::find(keys.begin(), keys.end(), x) == keys.end()) keys.push_back(x);
        }
        std::vector<std::uint32_t> v(350000);
        for (auto& x : v) x = keys[rng() % keys.size()];
        const bool ok_path = fyx::detail::try_radix_key_rank16_count_sort_parallel(v.data(), v.size(), false);
        CHECK(ok_path, "parallel uint32 rank16 count path accepted sparse 256-way lowcard");
        CHECK(std::is_sorted(v.begin(), v.end()), "parallel uint32 rank16 count output");
    }
#endif

    std::printf("counting: strings and payload structs\n");
    {
        std::vector<std::string> alpha;
        for (int i = 0; i < 64; ++i) alpha.push_back("key_" + std::to_string(i));
        std::vector<std::string> v(180000);
        for (auto& s : v) s = alpha[rng() % alpha.size()];
        check_against_std("string 64 distinct", v);
        check_against_std("string 64 distinct descending", v, std::greater<std::string>{});
        auto natural = [](const std::string& a, const std::string& b) { return a < b; };
        check_against_std("string 64 distinct custom natural comparator", v, natural);
#if FYX_ENABLE_PARALLEL
        std::vector<std::string> pv(300000);
        for (auto& s : pv) s = alpha[rng() % alpha.size()];
        const bool ok_parallel = fyx::detail::try_string_value_count_sort_parallel(pv.data(), pv.size(), natural, false);
        CHECK(ok_parallel, "parallel string value-count accepted custom comparator lowcard");
        CHECK(std::is_sorted(pv.begin(), pv.end(), natural), "parallel string value-count output");
#endif
    }
    {
        struct Rec {
            int key;
            int payload;
            bool operator==(const Rec& o) const { return key == o.key && payload == o.payload; }
        };
        auto by_key = [](const Rec& a, const Rec& b) { return a.key < b.key; };
        std::vector<Rec> v(50000);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = {static_cast<int>(rng() % 31), static_cast<int>(i)};

        std::vector<Rec> got = v;
        fyx::sort(got, by_key);
        CHECK(std::is_sorted(got.begin(), got.end(), by_key), "struct by-key sorted");
        std::vector<int> seen(v.size(), 0);
        for (const Rec& r : got) if (r.payload >= 0 && static_cast<std::size_t>(r.payload) < seen.size()) ++seen[r.payload];
        CHECK(std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; }), "struct payload permutation preserved");

        std::vector<Rec> ref = v;
        std::stable_sort(ref.begin(), ref.end(), by_key);
        fyx::stable_sort(v, by_key);
        CHECK(v == ref, "stable struct by-key low cardinality");
    }
    {
        struct Rec {
            int key;
            int payload;
            bool operator==(const Rec& o) const { return key == o.key && payload == o.payload; }
        };
        auto by_key = [](const Rec& a, const Rec& b) { return a.key < b.key; };
        std::vector<Rec> v(180000);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = {static_cast<int>(rng() % 100000), static_cast<int>(i)};
        std::vector<Rec> got = v;
        fyx::sort(got, by_key);
        CHECK(std::is_sorted(got.begin(), got.end(), by_key), "struct by-key high distinct sorted");
        std::vector<int> seen(v.size(), 0);
        for (const Rec& r : got) if (r.payload >= 0 && static_cast<std::size_t>(r.payload) < seen.size()) ++seen[r.payload];
        CHECK(std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; }), "struct by-key high distinct payload permutation");
        auto by_key_desc = [](const Rec& a, const Rec& b) { return a.key > b.key; };
        got = v;
        fyx::sort(got, by_key_desc);
        CHECK(std::is_sorted(got.begin(), got.end(), by_key_desc), "struct by-key high distinct descending sorted");
    }
    {
        auto abs_less = [](int a, int b) { return std::abs(a) < std::abs(b); };
        std::vector<int> v(60000);
        for (auto& x : v) x = static_cast<int>(rng() % 200000) - 100000;
        std::vector<int> ref = v;
        std::sort(ref.begin(), ref.end());
        fyx::Options o;
        o.parallel = fyx::Tri::On;
        fyx::sort(v, abs_less, o);
        CHECK(std::is_sorted(v.begin(), v.end(), abs_less), "parallel merge custom abs comparator sorted");
        std::sort(v.begin(), v.end());
        CHECK(v == ref, "parallel merge custom abs comparator permutation");
    }

    std::printf("input profile top-level dispatch\n");
    {
        namespace fd = fyx::detail;
        std::vector<int> v(4096);
        std::iota(v.begin(), v.end(), 0);
        fd::test_reset_dispatch();
        fyx::sort(v);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::ProfileSorted,
              "profile sorted direct return");
        CHECK(std::is_sorted(v.begin(), v.end()), "profile sorted output");
    }
    {
        namespace fd = fyx::detail;
        std::vector<int> v(4096);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int>(v.size() - i);
        fd::test_reset_dispatch();
        fyx::sort(v);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::ProfileReverse,
              "profile reverse direct reverse");
        CHECK(std::is_sorted(v.begin(), v.end()), "profile reverse output");
    }
    {
        namespace fd = fyx::detail;
        std::vector<int> v(4096, 7);
        fd::test_reset_dispatch();
        fyx::sort(v);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::ProfileAllEqual,
              "profile all-equal direct return");
        CHECK(std::all_of(v.begin(), v.end(), [](int x) { return x == 7; }),
              "profile all-equal output");
    }
    {
        namespace fd = fyx::detail;
        std::vector<int> v(200000);
        for (auto& x : v) x = static_cast<int>(rng() % 17) - 8;
        fd::test_reset_dispatch();
        fyx::Options o;
        o.parallel = fyx::Tri::Off;
        fyx::sort(v, o);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::LowCardinality,
              "profile low-cardinality chooses counting");
        CHECK(std::is_sorted(v.begin(), v.end()), "profile low-cardinality output");
    }
    {
        namespace fd = fyx::detail;
        std::vector<int> v(200000);
        for (auto& x : v) x = static_cast<int>(rng());
        fd::test_reset_dispatch();
        fyx::Options o;
        o.parallel = fyx::Tri::Off;
        fyx::sort(v.begin(), v.end(), o);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::Radix,
              "vector iterator numeric random uses pointer radix dispatch");
        CHECK(std::is_sorted(v.begin(), v.end()), "vector iterator numeric random output");
    }
    {
        namespace fd = fyx::detail;
        std::vector<int> v(200000);
        for (auto& x : v) x = static_cast<int>(rng());
        auto less_lambda = [](int a, int b) { return a < b; };
        fd::test_reset_dispatch();
        fyx::Options o;
        o.parallel = fyx::Tri::Off;
        fyx::sort(v.begin(), v.end(), less_lambda, o);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::Radix,
              "custom numeric less lambda recovers guarded radix path");
        CHECK(std::is_sorted(v.begin(), v.end(), less_lambda),
              "custom numeric less lambda guarded radix output");
    }
    {
        namespace fd = fyx::detail;
        std::vector<float> keys(16);
        for (std::size_t i = 0; i < keys.size(); ++i)
            keys[i] = static_cast<float>(static_cast<int>(i) - 8) * 1.25f;
        std::vector<float> v(200000);
        for (auto& x : v) x = keys[rng() % keys.size()];
        fd::test_reset_dispatch();
        fyx::Options o;
        o.parallel = fyx::Tri::Off;
        fyx::sort(v.begin(), v.end(), o);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::LowCardinality,
              "vector iterator float low-cardinality uses radix-key sparse count");
        CHECK(std::is_sorted(v.begin(), v.end()), "vector iterator float low-cardinality output");
    }
#if FYX_ENABLE_PARALLEL
    {
        namespace fd = fyx::detail;
        std::vector<float> keys(256);
        for (std::size_t i = 0; i < keys.size(); ++i)
            keys[i] = static_cast<float>(static_cast<int>(i) - 128) * 0.5f;
        std::vector<float> v((std::size_t(1) << 20) + 17);
        for (auto& x : v) x = keys[rng() % keys.size()];
        fd::test_reset_dispatch();
        fyx::Options o;
        o.parallel = fyx::Tri::On;
        fyx::sort(v.begin(), v.end(), o);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::LowCardinality,
              "parallel float low-cardinality uses rank/direct-map count");
        CHECK(std::is_sorted(v.begin(), v.end()), "parallel float low-cardinality output");
    }
#endif
#if FYX_ENABLE_PARALLEL
    {
        std::uniform_real_distribution<float> dist(-1000000.0f, 1000000.0f);
        std::vector<float> keys;
        keys.reserve(256);
        while (keys.size() < 256) {
            const float x = dist(rng);
            if (std::find(keys.begin(), keys.end(), x) == keys.end()) keys.push_back(x);
        }
        std::vector<float> v((std::size_t(1) << 20) + 17);
        for (auto& x : v) x = keys[rng() % keys.size()];
        const bool ok_path = fyx::detail::try_radix_key_rank16_count_sort_parallel(v.data(), v.size(), false);
        CHECK(ok_path, "parallel float rank16 count path accepted arbitrary 256-way lowcard");
        CHECK(std::is_sorted(v.begin(), v.end()), "parallel float rank16 count output");
    }
#endif
#if FYX_ENABLE_PARALLEL
    {
        std::vector<std::int32_t> v((std::size_t(1) << 20) + 17);
        for (auto& x : v) x = static_cast<std::int32_t>(rng() % 256) - 128;
        const bool ok_path = fyx::detail::try_integer_range_count_sort_parallel(v.data(), v.size(), false);
        CHECK(ok_path, "parallel int32 dense-range count path accepted 256-way lowcard");
        CHECK(std::is_sorted(v.begin(), v.end()), "parallel int32 dense-range count output");
    }
#endif
    {
        namespace fd = fyx::detail;
        std::vector<std::int32_t> keys = {
            std::numeric_limits<std::int32_t>::min(), -1234567890, -77777777, -1024,
            -7, 0, 3, 19, 1024, 65537, 77777777, 123456789, 987654321,
            1357913579, 1900000000, std::numeric_limits<std::int32_t>::max()
        };
        std::vector<std::int32_t> v((std::size_t(1) << 20) + 17);
        for (auto& x : v) x = keys[rng() % keys.size()];
        fd::test_reset_dispatch();
        fyx::Options o;
        o.parallel = fyx::Tri::On;
        fyx::sort(v.begin(), v.end(), o);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::LowCardinality,
              "parallel int32 sparse low-cardinality uses sparse count");
        CHECK(std::is_sorted(v.begin(), v.end()), "parallel int32 sparse low-cardinality output");
    }
    {
        namespace fd = fyx::detail;
        std::vector<std::int32_t> v((std::size_t(1) << 20) + 17);
        for (auto& x : v) x = static_cast<std::int32_t>(rng());
        fd::test_reset_dispatch();
        fyx::Options o;
        o.parallel = fyx::Tri::On;
        fyx::sort(v.begin(), v.end(), o);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::Radix,
              "parallel int32 high-entropy uses chunked radix dispatch");
        CHECK(std::is_sorted(v.begin(), v.end()), "parallel int32 high-entropy output");
    }
    {
        namespace fd = fyx::detail;
        std::vector<std::uint64_t> v((std::size_t(1) << 20) + 17);
        for (auto& x : v) x = rng();
        fd::test_reset_dispatch();
        fyx::Options o;
        o.parallel = fyx::Tri::On;
        fyx::sort(v.begin(), v.end(), o);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::Radix,
              "parallel uint64 high-entropy uses chunked radix dispatch");
        CHECK(std::is_sorted(v.begin(), v.end()), "parallel uint64 high-entropy output");
    }
    {
        namespace fd = fyx::detail;
        std::vector<double> v((std::size_t(1) << 20) + 17);
        for (auto& x : v) {
            const std::int64_t raw = static_cast<std::int64_t>(rng());
            x = static_cast<double>(raw) * 0.25;
        }
        fd::test_reset_dispatch();
        fyx::Options o;
        o.parallel = fyx::Tri::On;
        fyx::sort(v.begin(), v.end(), o);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::Radix,
              "parallel double high-entropy uses radix bucket dispatch");
        CHECK(std::is_sorted(v.begin(), v.end()), "parallel double high-entropy output");
    }
    {
        namespace fd = fyx::detail;
        std::vector<float> keys(16);
        for (std::size_t i = 0; i < keys.size(); ++i)
            keys[i] = static_cast<float>(static_cast<int>(i) - 8) * 0.5f;
        std::vector<float> v((std::size_t(1) << 20) + 17);
        for (auto& x : v) x = keys[rng() % keys.size()];
        fd::test_reset_dispatch();
        fyx::Options o;
        o.parallel = fyx::Tri::On;
        fyx::sort(v.begin(), v.end(), o);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::LowCardinality,
              "parallel float low-cardinality uses radix-key sparse count");
        CHECK(std::is_sorted(v.begin(), v.end()), "parallel float low-cardinality output");
    }
    {
        namespace fd = fyx::detail;
        std::vector<int> v(200000);
        std::iota(v.begin(), v.end(), 0);
        for (std::size_t i = 1000; i + 1 < v.size(); i += 4096)
            std::swap(v[i], v[i + 1]);
        fd::test_reset_dispatch();
        fyx::Options o;
        o.parallel = fyx::Tri::Off;
        fyx::sort(v, o);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::PartialPdq,
              "profile partially-sorted chooses pdqsort");
        CHECK(std::is_sorted(v.begin(), v.end()), "profile partially-sorted output");
    }
    {
        namespace fd = fyx::detail;
        std::vector<int> v(8192);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = static_cast<int>((i % 2 == 0) ? (i / 2) : (v.size() - i / 2));
        fd::test_reset_dispatch();
        fyx::sort(v);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::PartialPdq,
              "interleaved zigzag runs use linear two-run merge");
        CHECK(std::is_sorted(v.begin(), v.end()), "interleaved zigzag output");
    }
    {
        namespace fd = fyx::detail;
        std::vector<std::string> v(8192);
        for (std::size_t i = 0; i < v.size(); ++i) {
            const std::size_t key = (i % 2 == 0) ? (i / 2) : (v.size() - i / 2);
            v[i] = std::to_string(1000000000u + static_cast<unsigned>(key));
        }
        fd::test_reset_dispatch();
        fyx::sort(v);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::PartialPdq,
              "string interleaved zigzag runs use two-run merge");
        CHECK(std::is_sorted(v.begin(), v.end()), "string interleaved zigzag output");
    }
    {
        namespace fd = fyx::detail;
        struct Rec {
            int key;
            int payload;
            bool operator==(const Rec& o) const { return key == o.key && payload == o.payload; }
        };
        auto by_key = [](const Rec& a, const Rec& b) { return a.key < b.key; };
        std::vector<Rec> v(200000);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = {static_cast<int>(rng() % 1000000), static_cast<int>(i)};
        fd::test_reset_dispatch();
        fyx::Options o;
        o.parallel = fyx::Tri::Off;
        fyx::sort(v, by_key, o);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::Radix,
              "profile high-entropy struct-key chooses comparator-key radix");
        CHECK(std::is_sorted(v.begin(), v.end(), by_key), "profile high-entropy struct-key output");
    }

    std::printf("string MSD lexicographic edge cases\n");
    {
        namespace fd = fyx::detail;
        std::vector<std::string> v(30000);
        for (auto& s : v) {
            const std::size_t len = static_cast<std::size_t>(rng() % 9);
            s.resize(len);
            for (char& c : s) c = static_cast<char>(rng() & 0xFFu);
        }
        check_against_std("string MSD random bytes", v, std::less<std::string>{});
        check_against_std("string MSD random bytes descending", v, std::greater<std::string>{});

        auto less_lambda = [](const std::string& a, const std::string& b) { return a < b; };
        fd::test_reset_dispatch();
        fyx::Options o;
        o.parallel = fyx::Tri::Off;
        fyx::sort(v.begin(), v.end(), less_lambda, o);
        CHECK(fd::test_last_dispatch() == fd::DispatchDecision::Radix,
              "custom string less lambda recovers guarded MSD path");
        CHECK(std::is_sorted(v.begin(), v.end(), less_lambda),
              "custom string less lambda guarded MSD output");
    }

    std::printf("radix monotone detector with NaN\n");
    {
        std::vector<float> v(100);
        v[0] = std::numeric_limits<float>::quiet_NaN();
        for (std::size_t i = 1; i < v.size(); ++i) v[i] = static_cast<float>(i);
        fyx::sort(v);
        bool ok = true;
        for (std::size_t i = 1; i + 1 < v.size(); ++i) if (v[i - 1] > v[i]) ok = false;
        CHECK(ok && v.back() != v.back(), "float NaN not hidden by monotone fast path");
    }

    std::printf("parallel sample sort explicit dispatch\n");
    {
        std::vector<std::string> v(300000);
        for (auto& s : v) {
            s.resize(20);
            for (char& c : s) c = static_cast<char>('a' + (rng() % 26));
        }
        std::vector<std::string> ref = v;
        std::sort(ref.begin(), ref.end());
        fyx::Options o;
        o.parallel = fyx::Tri::On;
        fyx::sort(v, o);
        CHECK(v == ref, "parallel sample strings high distinct");
    }

    std::printf(failures ? "COUNTING TEST FAILURES=%d\n" : "ALL COUNTING TESTS PASS\n", failures);
    return failures ? 1 : 0;
}
