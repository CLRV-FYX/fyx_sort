// test_fyx_sort.cpp
// FYX-SORT vs IPS⁴o Benchmark (IPS⁴o 通过 CI 动态下载)
#include "fyx_sort.hpp"

// 注意：ips4o.hpp 将由 CI 下载到当前目录
#include "ips4o.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <cassert>

// 大对象
struct Large {
    int key;
    char payload[252];
    bool operator<(const Large& o) const { return key < o.key; }
    bool operator==(const Large& o) const { return key == o.key; }
};

std::mt19937_64 rng(12345);

template<typename T, typename Generator>
void run_benchmark(const char* name, size_t n, Generator gen, int rounds = 3) {
    double fyx_time = 0.0, ips_time = 0.0;
    bool correct = true;

    for (int r = 0; r < rounds; ++r) {
        std::vector<T> data(n), ref(n);
        for (size_t i = 0; i < n; ++i) {
            data[i] = gen();
            ref[i] = data[i];
        }

        // FYX
        auto t1 = std::chrono::high_resolution_clock::now();
        fyx::sort(data.data(), data.data() + data.size());
        auto t2 = std::chrono::high_resolution_clock::now();
        fyx_time += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // Reference
        std::sort(ref.begin(), ref.end());

        // IPS4o
        std::vector<T> data2(n);
        for (size_t i = 0; i < n; ++i) data2[i] = gen();
        t1 = std::chrono::high_resolution_clock::now();
        ips4o::sort(data2.begin(), data2.end());
        t2 = std::chrono::high_resolution_clock::now();
        ips_time += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // 验证
        if (!fyx::is_sorted(data)) correct = false;
        if (data != ref || data2 != ref) correct = false;
    }

    fyx_time /= rounds;
    ips_time /= rounds;
    double speedup = ips_time / fyx_time;

    std::cout << std::setw(20) << name
              << " | " << std::setw(10) << n
              << " | " << std::fixed << std::setprecision(2)
              << std::setw(10) << fyx_time << " ms"
              << " | " << std::setw(10) << ips_time << " ms"
              << " | " << std::setw(8) << speedup << "x"
              << " | " << (correct ? "PASS" : "FAIL")
              << "\n";
}

std::string random_string(size_t max_len) {
    static std::uniform_int_distribution<size_t> len_dist(1, max_len);
    static std::uniform_int_distribution<char> char_dist('a', 'z');
    size_t len = len_dist(rng);
    std::string s(len, ' ');
    for (char& c : s) c = char_dist(rng);
    return s;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           FYX-SORT vs IPS⁴o Benchmark (Auto Download)          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "CPU Threads: " << fyx::detail::Config::threads() << "\n";
#ifdef FYX_AVX2
    std::cout << "SIMD: AVX2\n";
#elif defined(FYX_SSE42)
    std::cout << "SIMD: SSE4.2\n";
#endif
    std::cout << "\n";

    std::cout << std::setw(20) << "Type"
              << " | " << std::setw(10) << "Size"
              << " | " << std::setw(10) << "FYX"
              << " | " << std::setw(10) << "IPS⁴o"
              << " | " << std::setw(8) << "Speedup"
              << " | Result\n";
    std::cout << std::string(80, '-') << "\n";

    // 整数
    for (size_t n : {100000, 1000000, 10000000}) {
        run_benchmark<int>("int", n, [&]() { return static_cast<int>(rng()); });
    }
    std::cout << std::string(80, '-') << "\n";

    // 浮点
    for (size_t n : {100000, 1000000}) {
        run_benchmark<double>("double", n, [&]() {
            return std::uniform_real_distribution<double>(-1e9, 1e9)(rng);
        });
    }
    std::cout << std::string(80, '-') << "\n";

    // 大对象
    for (size_t n : {10000, 50000}) {
        run_benchmark<Large>("Large (256B)", n, [&]() {
            Large obj;
            obj.key = static_cast<int>(rng());
            return obj;
        });
    }
    std::cout << std::string(80, '-') << "\n";

    // 特殊场景
    run_benchmark<int>("sorted", 10000000, [i = 0]() mutable { return i++; });
    run_benchmark<int>("reverse", 10000000, [i = 10000000]() mutable { return i--; });

    std::cout << "\nBenchmark completed.\n";
    return 0;
}


// 此文件完全由AI编写
