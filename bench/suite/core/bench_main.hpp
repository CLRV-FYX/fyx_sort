// fyx benchmark suite -- main() shared by every algorithm executable.
// Include after defining `struct Algo` (see bench_core.hpp).
#pragma once
#include "bench_core.hpp"

#include <exception>
#include <fstream>
#include <functional>
#include <sstream>

#if defined(FB_CONFIG_PORTABLE) && FB_CONFIG_PORTABLE
#  define FB_CONFIG_NAME "portable"
#else
#  define FB_CONFIG_NAME "native"
#endif

namespace fb {

enum Suite : int { kUnstable = 1, kStable = 2 };

struct Args {
    std::vector<std::size_t> sizes{100000, 1000000};
    std::vector<std::string> suites{"unstable", "stable"};
    std::vector<std::string> types;     // empty = all applicable
    std::vector<std::string> dists;     // empty = all
    std::size_t max_str = 1000000;      // strings / records above this are skipped
    std::size_t max_kv = 100000000;
    std::size_t batch_target = 1u << 18;
    std::uint64_t seed = 20261002;
    int round = 0;
    std::string out;
    bool info = false;
};

inline std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(item);
    return out;
}
inline bool contains(const std::vector<std::string>& v, const std::string& s) {
    return v.empty() || std::find(v.begin(), v.end(), s) != v.end();
}

template <class A, class T>
bool algo_supports() {
    std::vector<T> tiny = make_input<T>(64, Dist::Random, 1);
    try {
        return A::sort(tiny.data(), tiny.size());
    } catch (...) {
        return false;
    }
}

template <class T>
bool verify_unstable(std::vector<T>& out, const std::vector<T>& ref) {
    if constexpr (std::is_same_v<T, KV> || std::is_same_v<T, Rec>) {
        const Less<T> less{};
        if (!std::is_sorted(out.begin(), out.end(), less)) return false;
        // Canonicalise ties (payload order is unspecified for unstable sorts).
        auto full = [](const T& a, const T& b) {
            if constexpr (std::is_same_v<T, KV>) return a.key != b.key ? a.key < b.key : a.val < b.val;
            else return a.key != b.key ? a.key < b.key : a.id < b.id;
        };
        std::sort(out.begin(), out.end(), full);
        return out == ref;
    } else {
        return out == ref;
    }
}

template <class T>
std::vector<T> make_reference(const std::vector<T>& input, bool stable) {
    std::vector<T> ref = input;
    if (stable) {
        std::stable_sort(ref.begin(), ref.end(), Less<T>{});
    } else if constexpr (std::is_same_v<T, KV>) {
        std::sort(ref.begin(), ref.end(), [](const KV& a, const KV& b) {
            return a.key != b.key ? a.key < b.key : a.val < b.val;
        });
    } else if constexpr (std::is_same_v<T, Rec>) {
        std::sort(ref.begin(), ref.end(), [](const Rec& a, const Rec& b) {
            return a.key != b.key ? a.key < b.key : a.id < b.id;
        });
    } else {
        std::sort(ref.begin(), ref.end(), Less<T>{});
    }
    return ref;
}

template <class A, class T>
void run_type(const Args& args, const char* suite, bool stable, std::FILE* out) {
    const char* tname = type_name<T>();
    if (!contains(args.types, tname)) return;
    if (!algo_supports<A, T>()) return;

    // Untimed warm-up: thread pools, lazy dispatch tables, page faults.
    {
        std::vector<T> warm = make_input<T>(1u << 16, Dist::Random, args.seed ^ 0xabcdefull);
        A::sort(warm.data(), warm.size());
    }

    for (std::size_t n : args.sizes) {
        if ((std::is_same_v<T, std::string>) && n > args.max_str) continue;
        if ((std::is_same_v<T, KV> || std::is_same_v<T, Rec>) && n > args.max_kv) continue;
        for (int di = 0; di < static_cast<int>(Dist::Count); ++di) {
            const Dist d = static_cast<Dist>(di);
            if (!contains(args.dists, dist_name(d))) continue;
            const std::vector<T> input = make_input<T>(n, d, args.seed);
            std::vector<T> ref = make_reference(input, stable);

            const std::size_t batch = std::max<std::size_t>(1, args.batch_target / std::max<std::size_t>(n, 1));
            std::vector<T> work;
            work.reserve(n * batch);
            for (std::size_t b = 0; b < batch; ++b) work.insert(work.end(), input.begin(), input.end());

            bool ok = true;
            double secs = -1.0;
            // Several timed repetitions per round (fresh copy of the input each
            // time, untimed), median taken: single-shot timings of mid-size
            // inputs were dominated by noise.  Bounded so huge inputs stay 1x.
            const std::size_t elems = n * batch;
            const std::size_t reps = std::max<std::size_t>(1, std::min<std::size_t>(5, (std::size_t(1) << 23) / std::max<std::size_t>(elems, 1)));
            try {
                std::vector<double> times;
                for (std::size_t rep = 0; rep < reps; ++rep) {
                    if (rep) {
                        for (std::size_t b = 0; b < batch; ++b)
                            std::copy(input.begin(), input.end(), work.begin() + static_cast<std::ptrdiff_t>(b * n));
                    }
                    const auto t0 = Clock::now();
                    for (std::size_t b = 0; b < batch; ++b) A::sort(work.data() + b * n, n);
                    const auto t1 = Clock::now();
                    times.push_back(seconds_between(t0, t1) / static_cast<double>(batch));
                }
                std::sort(times.begin(), times.end());
                secs = times[times.size() / 2];
                for (std::size_t b = 0; b < batch && ok; ++b) {
                    std::vector<T> got(work.begin() + static_cast<std::ptrdiff_t>(b * n),
                                       work.begin() + static_cast<std::ptrdiff_t>((b + 1) * n));
                    ok = stable ? (got == ref) : verify_unstable(got, ref);
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "%s %s %s n=%zu: exception: %s\n", A::name, suite, tname, n, e.what());
                ok = false;
            } catch (...) {
                std::fprintf(stderr, "%s %s %s n=%zu: unknown exception\n", A::name, suite, tname, n);
                ok = false;
            }
            std::fprintf(out, "%s,%s,%s,%zu,%s,%s,%d,%.9g,%d,%zu\n", suite, tname, dist_name(d), n,
                         A::name, FB_CONFIG_NAME, args.round, secs, ok ? 1 : 0, batch);
            std::fflush(out);
        }
    }
}

template <class A>
int bench_main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto val = [&](const char* key) -> const char* {
            const std::size_t k = std::strlen(key);
            return s.compare(0, k, key) == 0 ? s.c_str() + k : nullptr;
        };
        if (const char* v = val("--sizes=")) { a.sizes.clear(); for (auto& x : split(v)) a.sizes.push_back(std::stoull(x)); }
        else if (const char* v2 = val("--suites=")) a.suites = split(v2);
        else if (const char* v3 = val("--types=")) a.types = split(v3);
        else if (const char* v4 = val("--dists=")) a.dists = split(v4);
        else if (const char* v5 = val("--max-str=")) a.max_str = std::stoull(v5);
        else if (const char* v6 = val("--max-kv=")) a.max_kv = std::stoull(v6);
        else if (const char* v7 = val("--batch-target=")) a.batch_target = std::stoull(v7);
        else if (const char* v8 = val("--seed=")) a.seed = std::stoull(v8);
        else if (const char* v9 = val("--round=")) a.round = std::atoi(v9);
        else if (const char* v10 = val("--out=")) a.out = v10;
        else if (s == "--info") a.info = true;
        else { std::fprintf(stderr, "unknown argument: %s\n", s.c_str()); return 2; }
    }

    if (a.info) {
        std::printf("name=%s\nconfig=%s\nparallel=%d\nstable=%d\nsuites=%d\nthreads=%u\ntypes=",
                    A::name, FB_CONFIG_NAME, A::parallel ? 1 : 0, A::stable ? 1 : 0, A::suites,
                    std::thread::hardware_concurrency());
        const char* sep = "";
        auto t = [&](bool ok, const char* nm) { if (ok) { std::printf("%s%s", sep, nm); sep = ","; } };
        t(algo_supports<A, std::int32_t>(), "i32");
        t(algo_supports<A, std::uint32_t>(), "u32");
        t(algo_supports<A, std::int64_t>(), "i64");
        t(algo_supports<A, std::uint64_t>(), "u64");
        t(algo_supports<A, float>(), "f32");
        t(algo_supports<A, double>(), "f64");
        t(algo_supports<A, std::string>(), "str");
        t(algo_supports<A, KV>(), "kv16");
        t(algo_supports<A, Rec>(), "rec8");
        std::printf("\n");
        return 0;
    }

    std::FILE* out = stdout;
    if (!a.out.empty()) {
        out = std::fopen(a.out.c_str(), "w");
        if (!out) { std::fprintf(stderr, "cannot open %s\n", a.out.c_str()); return 2; }
    }
    if ((A::suites & kUnstable) && contains(a.suites, "unstable")) {
        run_type<A, std::int32_t>(a, "unstable", false, out);
        run_type<A, std::uint32_t>(a, "unstable", false, out);
        run_type<A, std::int64_t>(a, "unstable", false, out);
        run_type<A, std::uint64_t>(a, "unstable", false, out);
        run_type<A, float>(a, "unstable", false, out);
        run_type<A, double>(a, "unstable", false, out);
        run_type<A, std::string>(a, "unstable", false, out);
        run_type<A, KV>(a, "unstable", false, out);
    }
    if ((A::suites & kStable) && A::stable && contains(a.suites, "stable")) {
        run_type<A, Rec>(a, "stable", true, out);
        run_type<A, KV>(a, "stable", true, out);
        run_type<A, std::int32_t>(a, "stable", true, out);
        run_type<A, double>(a, "stable", true, out);
        run_type<A, std::string>(a, "stable", true, out);
    }
    std::fprintf(out, "#done\n");
    if (out != stdout) std::fclose(out);
    return 0;
}

}  // namespace fb

int main(int argc, char** argv) { return fb::bench_main<Algo>(argc, argv); }
