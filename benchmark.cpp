// benchmark.cpp
// FYX-SORT v2.0 vs std::sort —— 安全、无外部依赖
#include "fyx_sort.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>

std::mt19937_64 rng(20260118);

template<typename T>
struct Generator {
    virtual T generate() = 0;
    virtual ~Generator() = default;
};

struct RandInt : Generator<int> {
    std::uniform_int_distribution<int> dist{INT_MIN / 2, INT_MAX / 2};
    int generate() override { return dist(rng); }
};

struct RandDouble : Generator<double> {
    std::uniform_real_distribution<double> dist{-1e9, 1e9};
    double generate() override { return dist(rng); }
};

struct Large { int key; char pad[252]; bool operator<(const Large& o) const { return key < o.key; } };
struct RandLarge : Generator<Large> {
    std::uniform_int_distribution<int> dist;
    Large generate() override { return {dist(rng), {}}; }
};

template<typename T>
struct Result {
    std::string type;
    std::string dist;
    size_t n;
    double fyx_ms = 0.0;
    double std_ms = 0.0;
    bool correct = true;
};

template<typename T, typename Gen>
Result<T> run_benchmark(const std::string& type, const std::string& dist, size_t n, Gen& gen, int rounds = 3) {
    Result<T> res{type, dist, n, 0.0, 0.0, true};
    for (int r = 0; r < rounds; ++r) {
        std::vector<T> a(n), b(n), ref(n);
        for (size_t i = 0; i < n; ++i) {
            T val = gen.generate();
            a[i] = val;
            b[i] = val;
            ref[i] = val;
        }
        std::sort(ref.begin(), ref.end());

        auto t0 = std::chrono::high_resolution_clock::now();
        fyx::sort(a);
        auto t1 = std::chrono::high_resolution_clock::now();
        res.fyx_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        std::sort(b.begin(), b.end());
        t1 = std::chrono::high_resolution_clock::now();
        res.std_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (a != ref || b != ref) res.correct = false;
    }
    res.fyx_ms /= rounds;
    res.std_ms /= rounds;
    return res;
}

int main() {
    std::cout << "FYX-SORT v2.0 vs std::sort Benchmark\n\n";

    std::vector<Result<void*>> results;

    // int random
    for (size_t n : {100000, 1000000, 10000000}) {
        RandInt gen;
        results.push_back(run_benchmark<int>("int", "random", n, gen));
    }

    // sorted & reverse
    results.push_back(Result<int>{"int", "sorted", 10000000, 0, 0, true});
    for (int r = 0; r < 3; ++r) {
        std::vector<int> a(10000000);
        std::generate(a.begin(), a.end(), [i = 0]() mutable { return i++; });
        auto t0 = std::chrono::high_resolution_clock::now();
        fyx::sort(a);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.back().fyx_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (!fyx::is_sorted(a)) results.back().correct = false;
    }
    results.back().fyx_ms /= 3;

    results.push_back(Result<int>{"int", "reverse", 10000000, 0, 0, true});
    for (int r = 0; r < 3; ++r) {
        std::vector<int> a(10000000);
        std::generate(a.begin(), a.end(), [i = 10000000]() mutable { return i--; });
        auto t0 = std::chrono::high_resolution_clock::now();
        fyx::sort(a);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.back().fyx_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (!fyx::is_sorted(a)) results.back().correct = false;
    }
    results.back().fyx_ms /= 3;

    // double
    for (size_t n : {100000, 1000000}) {
        RandDouble gen;
        results.push_back(run_benchmark<double>("double", "random", n, gen));
    }

    // Large
    for (size_t n : {10000, 50000}) {
        RandLarge gen;
        results.push_back(run_benchmark<Large>("Large(256B)", "random", n, gen));
    }

    // Output
    std::cout << std::setw(15) << "Type"
              << " | " << std::setw(12) << "Distribution"
              << " | " << std::setw(10) << "Size"
              << " | " << std::setw(10) << "FYX (ms)"
              << " | " << std::setw(10) << "std (ms)"
              << " | " << std::setw(8) << "Speedup"
              << " | Correct\n";
    std::cout << std::string(90, '-') << "\n";

    for (auto& r : results) {
        if (r.n == 0) continue;
        double speedup = r.std_ms / r.fyx_ms;
        std::cout << std::setw(15) << r.type
                  << " | " << std::setw(12) << r.dist
                  << " | " << std::setw(10) << r.n
                  << " | " << std::fixed << std::setprecision(2) << std::setw(10) << r.fyx_ms
                  << " | " << std::setw(10) << r.std_ms
                  << " | " << std::setw(8) << speedup << "x"
                  << " | " << (r.correct ? "✅" : "❌") << "\n";
    }

    // Save Markdown
    std::ofstream md("benchmark.md");
    md << "# FYX-SORT v2.0 vs std::sort Benchmark\n\n";
    md << "| Type | Distribution | Size | FYX (ms) | std (ms) | Speedup | Correct |\n";
    md << "|------|--------------|------|----------|----------|---------|---------|\n";
    for (auto& r : results) {
        if (r.n == 0) continue;
        double speedup = r.std_ms / r.fyx_ms;
        md << "| " << r.type
           << " | " << r.dist
           << " | " << r.n
           << " | " << std::fixed << std::setprecision(2) << r.fyx_ms
           << " | " << r.std_ms
           << " | " << speedup << "x"
           << " | " << (r.correct ? "PASS" : "FAIL") << " |\n";
    }
    md.close();

    std::cout << "\nReport saved to benchmark.md\n";
    return 0;
}
