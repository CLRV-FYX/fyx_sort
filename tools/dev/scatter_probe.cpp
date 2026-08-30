// Which scatter is actually fastest at this working set?
//
// fyx's radix scatter buffers one cache line per bucket and flushes it with
// non-temporal stores.  That is the right shape for data that dwarfs the cache;
// for a 4 MB sort it may not be.  Compare, in one harness:
//
//   naive   dst[off[b]++] = x
//   wcb     fyx::detail::radix_scatter_pass (line buffer + NT flush)
//   simd    AVX-512CD conflict -> rank -> vpscatterdd straight to the destination
//
//   g++ -std=c++17 -O3 -march=native -DNDEBUG -I. tools/dev/scatter_probe.cpp -o /tmp/sp && /tmp/sp
#include "fyx_sort.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <random>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
template <class F> static double best(F f, int reps = 7) {
    double b = 1e9;
    for (int r = 0; r < reps; ++r) { const double t = now(); (void)f(); const double e = now(); b = std::min(b, e - t); }
    return b;
}

static void pass_naive(const std::uint32_t* src, std::size_t n, std::uint32_t* dst,
                       std::uint32_t* off, unsigned shift) {
    for (std::size_t i = 0; i < n; ++i) { const unsigned b = (src[i] >> shift) & 255u; dst[off[b]++] = src[i]; }
}

__attribute__((target("avx512f,avx512cd")))
static void pass_simd(const std::uint32_t* src, std::size_t n, std::uint32_t* dst,
                      std::uint32_t* off, unsigned shift) {
    const __m512i vmask  = _mm512_set1_epi32(255);
    const __m512i ridx   = _mm512_setr_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    const __m512i vshift = _mm512_set1_epi32((int)shift);
    const __m512i one    = _mm512_set1_epi32(1);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512i k    = _mm512_loadu_si512(src + i);
        const __m512i b    = _mm512_and_si512(_mm512_srlv_epi32(k, vshift), vmask);
        const __m512i cf   = _mm512_conflict_epi32(b);
        const __m512i rank = _mm512_popcnt_epi32(cf);
        const __m512i base = _mm512_i32gather_epi32(b, (const void*)off, 4);
        _mm512_i32scatter_epi32(dst, _mm512_add_epi32(base, rank), k, 4);
        const __m512i br  = _mm512_permutexvar_epi32(ridx, b);
        const __m512i crr = _mm512_permutexvar_epi32(ridx, _mm512_conflict_epi32(br));
        const __m512i cnt = _mm512_add_epi32(_mm512_add_epi32(_mm512_popcnt_epi32(cf),
                                                              _mm512_popcnt_epi32(crr)), one);
        const __mmask16 last = _mm512_cmpeq_epi32_mask(_mm512_add_epi32(rank, one), cnt);
        // the cursor advances by this block's count -- add, do not overwrite
        _mm512_mask_i32scatter_epi32(off, last, b, _mm512_add_epi32(base, cnt), 4);
    }
    for (; i < n; ++i) { const unsigned b = (src[i] >> shift) & 255u; dst[off[b]++] = src[i]; }
}

int main() {
    const std::size_t n = 1u << 20;
    const unsigned shift = 8;
    std::uint32_t* src = (std::uint32_t*)aligned_alloc(64, n * 4);
    std::uint32_t* dst = (std::uint32_t*)aligned_alloc(64, n * 4);
    std::uint32_t* ref = (std::uint32_t*)aligned_alloc(64, n * 4);
    std::mt19937 g(11);
    for (std::size_t i = 0; i < n; ++i) src[i] = g();

    std::uint32_t hist[256], start[256], off[256];
    std::memset(hist, 0, sizeof hist);
    for (std::size_t i = 0; i < n; ++i) ++hist[(src[i] >> shift) & 255u];
    std::uint32_t sum = 0;
    for (unsigned b = 0; b < 256; ++b) { start[b] = sum; sum += hist[b]; }

    // fyx's scatter needs its write-combining scratch
    using Scratch = fyx::detail::RadixScatterScratch<std::uint32_t>;
    constexpr std::size_t kPerLine = Scratch::kPerLine;
    Scratch sc;
    std::uint32_t* raw = (std::uint32_t*)aligned_alloc(64, (2 * 256 * kPerLine + kPerLine + 64) * 4);
    const std::uintptr_t a0 = reinterpret_cast<std::uintptr_t>(raw);
    const std::uintptr_t m  = (64 - (a0 & 63)) & 63;
    sc.line = reinterpret_cast<std::uint32_t*>(a0 + m);
    sc.head = sc.line + 256 * kPerLine;
    const bool can_stream = fyx::detail::have_nt_stores();

    // Interleave the variants and take the best of many rounds: measured
    // back to back, whichever runs first pays for cold pages and looks twice
    // as slow as it is.
    auto one = [&](int which, std::uint32_t* out) -> double {
        std::memcpy(off, start, sizeof start);
        const double t0 = now();
        if (which == 0)      pass_naive(src, n, out, off, shift);
        else if (which == 1) pass_simd (src, n, out, off, shift);
        else {
            std::size_t o[256];
            for (unsigned b = 0; b < 256; ++b) o[b] = start[b];
            fyx::detail::radix_scatter_pass<std::uint32_t>(src, n, out, shift, o, sc, can_stream);
        }
        const double t1 = now();
        return t1 - t0;
    };
    const int rounds = 15;
    for (int r = 0; r < 3; ++r) for (int w = 0; w < 3; ++w) (void)one(w, dst);   // warm up
    double t[3] = {1e9, 1e9, 1e9};
    for (int r = 0; r < rounds; ++r)
        for (int w = 0; w < 3; ++w) t[w] = std::min(t[w], one(w, dst));
    const double t_naive = t[0], t_simd = t[1], t_wcb = t[2];

    // correctness, measured off the clock
    std::memcpy(off, start, sizeof start);
    pass_naive(src, n, ref, off, shift);
    std::memcpy(off, start, sizeof start);
    pass_simd(src, n, dst, off, shift);
    const bool simd_same = std::memcmp(ref, dst, n * 4) == 0;
    {
        std::size_t o[256];
        for (unsigned b = 0; b < 256; ++b) o[b] = start[b];
        fyx::detail::radix_scatter_pass<std::uint32_t>(src, n, dst, shift, o, sc, can_stream);
    }
    const bool wcb_same = std::memcmp(ref, dst, n * 4) == 0;

    std::printf("1M uint32, one 8-bit scatter pass (%.2f GB/s of traffic each way)\n", n * 4 / 1e6);
    std::printf("  naive scatter   %.5f s   %5.1f cycles/elem\n", t_naive, t_naive * 2.46e9 / n);
    std::printf("  simd  conflict  %.5f s   %5.1f cycles/elem   %s\n",
                t_simd, t_simd * 2.46e9 / n, simd_same ? "matches" : "MISMATCH");
    std::printf("  wcb + NT flush  %.5f s   %5.1f cycles/elem   %s\n",
                t_wcb, t_wcb * 2.46e9 / n, wcb_same ? "matches" : "MISMATCH");
    free(src); free(dst); free(ref); free(raw);
    return 0;
}
