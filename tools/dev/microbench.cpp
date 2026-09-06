// What the machine can actually do: memory bandwidth and the clock.  Needed to
// tell "bandwidth bound" from "stall bound" before touching a kernel.
//
//   g++ -std=c++17 -O2 -march=native -DNDEBUG tools/dev/microbench.cpp -o /tmp/mb && /tmp/mb
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
template <class F> static double best(F f, int reps = 7) {
    double b = 1e9;
    for (int r = 0; r < reps; ++r) { const double t = now(); (void)f(); const double e = now(); b = std::min(b, e - t); }
    return b;
}

int main() {
    for (std::size_t bytes : {(size_t)4 << 20, (size_t)64 << 20}) {
        const std::size_t n = bytes / 4;
        std::uint32_t* src = (std::uint32_t*)aligned_alloc(64, bytes);
        std::uint32_t* dst = (std::uint32_t*)aligned_alloc(64, bytes);
        for (std::size_t i = 0; i < n; ++i) src[i] = (std::uint32_t)i;
        const double mc = best([&] { std::memcpy(dst, src, bytes); return 0; });
        const double nt = best([&] {
            for (std::size_t i = 0; i < n; i += 16) {
                __m512i v = _mm512_load_si512(reinterpret_cast<const __m512i*>(src + i));
                _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i), v);
            }
            _mm_sfence(); return 0;
        });
        std::printf("%6zu MiB:  memcpy %.5f s (%5.2f GB/s)   NT-write %.5f s (%5.2f GB/s)\n",
                    bytes >> 20, mc, bytes / mc / 1e9, nt, bytes / nt / 1e9);
        free(src); free(dst);
    }
    // clock: a dependent multiply chain is ~3 cycles per iteration on Ice Lake
    volatile std::uint64_t sink = 0;
    const double lc = best([&] {
        std::uint64_t s = 12345;
        for (std::uint64_t i = 0; i < 200000000ull; ++i)
            s = s * 6364136223846793005ull + 1442695040888963407ull;
        sink = s; return 0;
    }, 5);
    std::printf("clock: %.5f s for 200M dependent imul+add -> ~%.2f GHz (if 3 cycles/iter)\n", lc, 0.6 / lc);
    (void)sink;
    return 0;
}
