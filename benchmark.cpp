// benchmark.cpp
// FYX-SORT v2.0 vs IPS⁴o — Full Benchmark
// Usage: define -DFYX_MAIN and include fyx_sort.hpp
// IPS⁴o is downloaded during CI (not stored in repo)

#include "fyx_sort.hpp"
#include "ips4o.hpp"  // will be downloaded by CI

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <fstream>

std::mt19937_64 global_rng(20260118);

// ----------------------------
// 数据生成器
// ----------------------------

template<typename T>
struct DataGenerator {
    virtual T generate() = 0;
    virtual ~DataGenerator() = default;
};

struct RandInt : DataGenerator<int> {
    std::uniform_int_distribution<int> dist{INT_MIN / 2, INT_MAX / 2};
    int generate() override { return dist(global_rng); }
};

struct RandDouble : DataGenerator<double> {
    std::uniform_real_distribution<double> dist{-1e9, 1e9};
    double generate() override { return dist(global_rng); }
};

struct RandString : DataGenerator<std::string> {
    std::uniform_int_distribution<size_t> len_dist{1, 16};
    std::uniform_int_distribution<char> char_dist{'a', 'z'};
    std::string generate() override {
        size_t len = len_dist(global_rng);
        std::string s(len, ' ');
        for (char& c : s) c = char_dist(global_rng);
        return s;
    }
};

struct LargeObj {
    int key;
    char pad[252];
    bool operator<(const LargeObj& o) const { return key < o.key; }
    bool operator==(const LargeObj& o) const { return key == o.key; }
};

struct RandLarge : DataGenerator<LargeObj> {
    std::uniform_int_distribution<int> dist;
    LargeObj generate() override { return {dist(global_rng), {}}; }
};

// ----------------------------
// Benchmark 核心
// ----------------------------

template<typename T>
struct Result {
    std::string type;
    std::string distribution;
    size_t n;
    double fyx_ms = 0.0;
    double ips_ms = 0.0;
    bool correct = true;
};

template<typename T, typename Gen>
Result<T> run_benchmark(
    const std::string& type,
    const std::string& dist,
    size_t n,
    Gen& gen,
    int rounds = 3
) {
    Result<T> res{type, dist, n, 0.0, 0.0, true};

    for (int r = 0; r < rounds; ++r) {
        std::vector<T> data(n), ref(n);
        for (size_t i = 0; i < n; ++i) {
            T val = gen.generate();
            data[i] = val;
            ref[i] = val;
        }

        // Reference sort
        std::sort(ref.begin(), ref.end());

        // FYX-SORT
        auto a = data;
        auto t0 = std::chrono::high_resolution_clock::now();
        fyx::sort(a);
        auto t1 = std::chrono::high_resolution_clock::now();
        res.fyx_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // IPS⁴o
        auto b = data;
        t0 = std::chrono::high_resolution_clock::now();
        ips4o::sort(b.begin(), b.end());
        t1 = std::chrono::high_resolution_clock::now();
        res.ips_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Verify correctness
        if (a != ref || b != ref) res.correct = false;
    }

    res.fyx_ms /= rounds;
    res.ips_ms /= rounds;
    return res;
}

// ----------------------------
// 主函数
// ----------------------------

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                        FYX-SORT v2.0 vs IPS⁴o Benchmark                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════╝\n\n";

    std::vector<Result<void*>> results;

    // === Int: random ===
    for (size_t n : {100000, 1000000, 10000000}) {
        RandInt gen;
        results.push_back(run_benchmark<int>("int", "random", n, gen));
    }

    // === Int: sorted ===
    auto sorted_gen = [i = 0]() mutable { return i++; };
    results.push_back(Result<int>{"int", "sorted", 10000000, 0, 0, true});
    for (int r = 0; r < 3; ++r) {
        std::vector<int> a(10000000);
        std::generate(a.begin(), a.end(), sorted_gen);
        auto t0 = std::chrono::high_resolution_clock::now();
        fyx::sort(a);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.back().fyx_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        // IPS⁴o
        std::generate(a.begin(), a.end(), [i = 0]() mutable { return i++; });
        t0 = std::chrono::high_resolution_clock::now();
        ips4o::sort(a.begin(), a.end());
        t1 = std::chrono::high_resolution_clock::now();
        results.back().ips_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (!fyx::is_sorted(a)) results.back().correct = false;
    }
    results.back().fyx_ms /= 3;
    results.back().ips_ms /= 3;

    // === Int: reverse ===
    auto reverse_gen = [i = 10000000]() mutable { return i--; };
    results.push_back(Result<int>{"int", "reverse", 10000000, 0, 0, true});
    for (int r = 0; r < 3; ++r) {
        std::vector<int> a(10000000);
        std::generate(a.begin(), a.end(), reverse_gen);
        auto t0 = std::chrono::high_resolution_clock::now();
        fyx::sort(a);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.back().fyx_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        // IPS⁴o
        std::generate(a.begin(), a.end(), [i = 10000000]() mutable { return i--; });
        t0 = std::chrono::high_resolution_clock::now();
        ips4o::sort(a.begin(), a.end());
        t1 = std::chrono::high_resolution_clock::now();
        results.back().ips_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (!fyx::is_sorted(a)) results.back().correct = false;
    }
    results.back().fyx_ms /= 3;
    results.back().ips_ms /= 3;

    // === Double ===
    for (size_t n : {100000, 1000000}) {
        RandDouble gen;
        results.push_back(run_benchmark<double>("double", "random", n, gen));
    }

    // === String ===
    for (size_t n : {50000, 200000}) {
        RandString gen;
        results.push_back(run_benchmark<std::string>("string", "short", n, gen));
    }

    // === Large Object ===
    for (size_t n : {10000, 50000}) {
        RandLarge gen;
        results.push_back(run_benchmark<LargeObj>("Large(256B)", "random", n, gen));
    }

    // === Stable Sort Test ===
    struct StableItem { int key, order; };
    auto stable_cmp = [](const StableItem& a, const StableItem& b) { return a.key < b.key; };
    results.push_back(Result<StableItem>{"stable_test", "random", 500000, 0, 0, true});
    for (int r = 0; r < 3; ++r) {
        std::vector<StableItem> a(500000);
        std::uniform_int_distribution<int> key_dist(0, 99);
        for (int i = 0; i < 500000; ++i) {
            a[i] = {key_dist(global_rng), i};
        }
        auto ref = a;
        std::stable_sort(ref.begin(), ref.end(), stable_cmp);

        // FYX stable
        auto fyxa = a;
        auto t0 = std::chrono::high_resolution_clock::now();
        fyx::stable_sort(fyxa, stable_cmp);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.back().fyx_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // std::stable_sort (IPS⁴o has no stable version)
        auto stda = a;
        t0 = std::chrono::high_resolution_clock::now();
        std::stable_sort(stda.begin(), stda.end(), stable_cmp);
        t1 = std::chrono::high_resolution_clock::now();
        results.back().ips_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (fyxa != ref || stda != ref) results.back().correct = false;
        // Check stability
        for (size_t i = 1; i < fyxa.size(); ++i) {
            if (fyxa[i].key == fyxa[i-1].key && fyxa[i].order < fyxa[i-1].order)
                results.back().correct = false;
        }
    }
    results.back().fyx_ms /= 3;
    results.back().ips_ms /= 3;

    // ----------------------------
    // 输出结果
    // ----------------------------

    std::cout << std::setw(15) << "Type"
              << " | " << std::setw(12) << "Distribution"
              << " | " << std::setw(10) << "Size"
              << " | " << std::setw(10) << "FYX (ms)"
              << " | " << std::setw(10) << "IPS⁴o (ms)"
              << " | " << std::setw(8) << "Speedup"
              << " | Correct\n";
    std::cout << std::string(90, '-') << "\n";

    for (auto& r : results) {
        if (r.n == 0) continue;
        double speedup = r.ips_ms / r.fyx_ms;
        std::cout << std::setw(15) << r.type
                  << " | " << std::setw(12) << r.distribution
                  << " | " << std::setw(10) << r.n
                  << " | " << std::fixed << std::setprecision(2) << std::setw(10) << r.fyx_ms
                  << " | " << std::setw(10) << r.ips_ms
                  << " | " << std::setw(8) << speedup << "x"
                  << " | " << (r.correct ? "✅" : "❌") << "\n";
    }

    // Save to Markdown
    std::ofstream md("benchmark.md");
    md << "# FYX-SORT v2.0 vs IPS⁴o Benchmark\n\n";
    md << "| Type | Distribution | Size | FYX (ms) | IPS⁴o (ms) | Speedup | Correct |\n";
    md << "|------|--------------|------|----------|------------|---------|---------|\n";
    for (auto& r : results) {
        if (r.n == 0) continue;
        double speedup = r.ips_ms / r.fyx_ms;
        md << "| " << r.type
           << " | " << r.distribution
           << " | " << r.n
           << " | " << std::fixed << std::setprecision(2) << r.fyx_ms
           << " | " << r.ips_ms
           << " | " << speedup << "x"
           << " | " << (r.correct ? "PASS" : "FAIL") << " |\n";
    }
    md.close();

    std::cout << "\nBenchmark report saved to benchmark.md\n";
    return 0;
}

// 此文件AI完全
