// Real measured benchmark: fyx::sort vs std::sort on this machine.
// Throughput = n * sizeof(T) / seconds.  Speedup = std_time / fyx_time.
// Compile: g++ -std=c++17 -O3 -march=native -pthread bench_api.cpp -o bench_api
#include "../fyx_sort.hpp"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using namespace fyx;

template <class T>
double best_seconds(T* p, std::size_t n, int iters, bool use_fyx) {
    double best = 1e30;
    std::vector<T> copy(n);
    for (int it = 0; it < iters; ++it) {
        std::copy_n(p, n, copy.data());
        auto t0 = std::chrono::steady_clock::now();
        if (use_fyx) fyx::sort(copy.data(), n);
        else         std::sort(copy.data(), copy.data() + n);
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        if (s < best) best = s;
    }
    return best;
}

template <class T>
void bench(const char* name, std::size_t n, std::uint64_t seed) {
    std::vector<T> data(n);
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < n; ++i) {
        // uniform-ish over the type's range without producing NaN for floats
        if constexpr (std::is_floating_point_v<T>)
            data[i] = static_cast<T>(static_cast<std::int64_t>(rng() % 2000001) - 1000000) / 1000.0;
        else
            data[i] = static_cast<T>(rng());
    }
    int iters = n >= (1u << 24) ? 5 : 20;
    double tf = best_seconds(data.data(), n, iters, true);
    double ts = best_seconds(data.data(), n, iters, false);
    double bytes = static_cast<double>(n) * sizeof(T);
    double gbf = bytes / tf / 1e9;
    double gbs = bytes / ts / 1e9;
    printf("| %-7s | %9.1f MB | %8.3f s | %8.3f s | %6.2f GB/s | %6.2f GB/s | %5.2fx |\n",
           name, bytes / 1e6, tf, ts, gbf, gbs, ts / tf);
}

int main() {
    printf("## fyx::sort vs std::sort  (single-thread, -O3 -march=native, Ice Lake-SP, 2 vCPU)\n\n");
    printf("| type   | size       | fyx (s)   | std (s)   | fyx GB/s  | std GB/s  | speedup |\n");
    printf("|--------|------------|-----------|-----------|-----------|-----------|---------|\n");
    bench<std::int32_t >("int32", 1u << 20, 1);
    bench<std::int32_t >("int32", 1u << 24, 2);
    bench<std::int32_t >("int32", 1u << 27, 3);   // 100M -> 400 MB
    bench<std::uint32_t>("uint32", 1u << 24, 4);
    bench<std::int64_t >("int64", 1u << 24, 5);
    bench<std::uint64_t>("uint64", 1u << 24, 6);
    bench<float        >("float", 1u << 24, 7);
    bench<double       >("double", 1u << 24, 8);
    printf("\n(GB/s = input bytes / time. speedup = std_time / fyx_time. Best of several runs.)\n");
    return 0;
}
