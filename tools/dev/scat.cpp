// tools/dev/scat.cpp -- what a radix scatter pass actually costs.
//
// Four strategies over the same input, each verified against a stable
// counting sort:
//
//   naive    dst[offset[digit]++] = key, one store per element
//   wcb-nt   per-bucket one-line buffer, flushed with a non-temporal store
//   wcb-wb   the same buffer, flushed with a normal (write-back) store
//   avx512   conflict/rank/gather/scatter, 16 lanes per step
//
// Offsets are initialised to real bucket starts (n/B per bucket), which is
// what the kernel passes in; leaving them at zero makes every bucket write to
// the same place and flatters every strategy.
//
//     g++ -std=c++17 -O3 -march=native -DNDEBUG -I. tools/dev/scat.cpp -o /tmp/scat
//     /tmp/scat 1000000 12
#include "../../fyx_sort.hpp"
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
using namespace std::chrono;
static double now(){ return duration<double>(steady_clock::now().time_since_epoch()).count(); }
static const std::size_t CL = 64;

template <unsigned Bits>
void scat_naive(const std::uint32_t* src, std::size_t n, std::uint32_t* dst,
                unsigned shift, std::size_t* off) {
    constexpr std::uint32_t mask = (1u << Bits) - 1;
    for (std::size_t i = 0; i < n; ++i) { std::uint32_t k = src[i]; dst[off[(k >> shift) & mask]++] = k; }
}

template <unsigned Bits, bool Stream>
void scat_wcb(const std::uint32_t* src, std::size_t n, std::uint32_t* dst,
              unsigned shift, std::size_t* off) {
    constexpr std::size_t B = std::size_t(1) << Bits, per = 16;
    static std::vector<std::uint32_t> line(B * per), fill(B), need(B);
    for (std::size_t b = 0; b < B; ++b) {
        fill[b] = 0;
        std::uintptr_t a = (std::uintptr_t)(dst + off[b]);
        need[b] = (std::uint32_t)(((CL - (a & (CL - 1))) & (CL - 1)) / 4);
    }
    std::size_t remaining = 0;
    for (std::size_t b = 0; b < B; ++b) remaining += need[b];
    std::size_t i = 0;
    for (; i < n; ++i) {
        std::uint32_t k = src[i]; std::size_t b = (k >> shift) & (B - 1);
        if (remaining && need[b]) { dst[off[b]++] = k; --need[b]; --remaining; continue; }
        std::uint32_t* L = line.data() + b * per; std::uint32_t f = fill[b]; L[f] = k;
        if (f + 1 == per) {
            if (Stream) _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + off[b]), _mm512_loadu_si512(L));
            else        _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + off[b]), _mm512_loadu_si512(L));
            off[b] += per; fill[b] = 0;
        } else fill[b] = f + 1;
    }
    for (std::size_t b = 0; b < B; ++b)
        if (fill[b]) std::memcpy(dst + off[b], line.data() + b * per, fill[b] * 4);
}

template <unsigned Bits>
void scat_avx512(const std::uint32_t* src, std::size_t n, std::uint32_t* dst,
                 unsigned shift, std::uint32_t* off) {
    constexpr std::uint32_t mask = (1u << Bits) - 1;
    const __m512i vmask = _mm512_set1_epi32(int(mask)), vone = _mm512_set1_epi32(1);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512i k   = _mm512_loadu_si512(reinterpret_cast<const void*>(src + i));
        __m512i idx = _mm512_and_si512(_mm512_srli_epi32(k, shift), vmask);
        __m512i rnk = _mm512_popcnt_epi32(_mm512_conflict_epi32(idx));
        __m512i pos = _mm512_add_epi32(_mm512_i32gather_epi32(idx, off, 4), rnk);
        _mm512_i32scatter_epi32(dst, pos, k, 4);
        _mm512_i32scatter_epi32(off, idx, _mm512_add_epi32(pos, vone), 4);
    }
    for (; i < n; ++i) { std::uint32_t k = src[i]; std::uint32_t b = (k >> shift) & mask; dst[off[b]++] = k; }
}

template <class F> double bench(F f, int reps) {
    double b = 1e30; for (int r = 0; r < reps; ++r) { double t0 = now(); f(); double t1 = now(); b = std::min(b, t1 - t0); } return b;
}
int main(int argc, char** argv) {
    std::size_t n = argc > 1 ? std::strtoull(argv[1], 0, 10) : 1000000;
    int bits = argc > 2 ? std::atoi(argv[2]) : 12;
    const std::size_t B = std::size_t(1) << bits;
    const unsigned shift = 32 - unsigned(bits);
    std::mt19937_64 rng(7);
    std::vector<std::uint32_t> src(n), dst(n), ref(n), chk(n);
    for (std::size_t i = 0; i < n; ++i) src[i] = std::uint32_t(rng());
    // Bucket sizes are Poisson, not uniform: the starts have to come from a
    // real counting pass or the buckets overflow into each other.
    std::vector<std::size_t> start(B + 1), off(B); std::vector<std::uint32_t> off32(B);
    for (std::size_t b = 0; b <= B; ++b) start[b] = 0;
    for (std::size_t i = 0; i < n; ++i) ++start[std::size_t((src[i] >> shift) & (B - 1)) + 1];
    for (std::size_t b = 0; b < B; ++b) start[b + 1] += start[b];
    {   auto s = src;
        std::stable_sort(s.begin(), s.end(), [&](std::uint32_t a, std::uint32_t b){ return (a >> shift) < (b >> shift); });
        for (std::size_t b = 0, w = 0; b < B; ++b) {
            std::size_t c = start[b + 1] - start[b];
            for (std::size_t j = 0; j < c; ++j) ref[start[b] + j] = s[w++];
        } }
    auto prep = [&]{ for (std::size_t b = 0; b < B; ++b) { off[b] = start[b]; off32[b] = std::uint32_t(start[b]); } };
#define DO(F, OFF) switch (bits) { case 9: F<9>(src.data(),n,dst.data(),shift,OFF); break; \
        case 10: F<10>(src.data(),n,dst.data(),shift,OFF); break; case 11: F<11>(src.data(),n,dst.data(),shift,OFF); break; \
        case 12: F<12>(src.data(),n,dst.data(),shift,OFF); break; case 13: F<13>(src.data(),n,dst.data(),shift,OFF); break; \
        case 14: F<14>(src.data(),n,dst.data(),shift,OFF); break; case 15: F<15>(src.data(),n,dst.data(),shift,OFF); break; \
        case 16: F<16>(src.data(),n,dst.data(),shift,OFF); break; }
#define DOW(F, OFF) switch (bits) { case 9: F<9,true>(src.data(),n,dst.data(),shift,OFF); break; \
        case 10: F<10,true>(src.data(),n,dst.data(),shift,OFF); break; case 11: F<11,true>(src.data(),n,dst.data(),shift,OFF); break; \
        case 12: F<12,true>(src.data(),n,dst.data(),shift,OFF); break; case 13: F<13,true>(src.data(),n,dst.data(),shift,OFF); break; \
        case 14: F<14,true>(src.data(),n,dst.data(),shift,OFF); break; case 15: F<15,true>(src.data(),n,dst.data(),shift,OFF); break; \
        case 16: F<16,true>(src.data(),n,dst.data(),shift,OFF); break; }
#define DOB(F, OFF) switch (bits) { case 9: F<9,false>(src.data(),n,dst.data(),shift,OFF); break; \
        case 10: F<10,false>(src.data(),n,dst.data(),shift,OFF); break; case 11: F<11,false>(src.data(),n,dst.data(),shift,OFF); break; \
        case 12: F<12,false>(src.data(),n,dst.data(),shift,OFF); break; case 13: F<13,false>(src.data(),n,dst.data(),shift,OFF); break; \
        case 14: F<14,false>(src.data(),n,dst.data(),shift,OFF); break; case 15: F<15,false>(src.data(),n,dst.data(),shift,OFF); break; \
        case 16: F<16,false>(src.data(),n,dst.data(),shift,OFF); break; }
    prep(); DO(scat_naive, off.data())   printf("  naive  ok=%d\n", (int)(dst == ref));
    prep(); DOW(scat_wcb, off.data())    printf("  wcb-nt ok=%d\n", (int)(dst == ref));
    prep(); DOB(scat_wcb, off.data())    printf("  wcb-wb ok=%d\n", (int)(dst == ref));
    prep(); DO(scat_avx512, off32.data()) printf("  avx512 ok=%d\n", (int)(dst == ref));
    double tN  = bench([&]{ prep(); DO(scat_naive, off.data());    }, 7);
    double tNT = bench([&]{ prep(); DOW(scat_wcb, off.data());     }, 7);
    double tWB = bench([&]{ prep(); DOB(scat_wcb, off.data());     }, 7);
    double tA  = bench([&]{ prep(); DO(scat_avx512, off32.data()); }, 7);
    double tC  = bench([&]{ std::memcpy(chk.data(), src.data(), n * 4); }, 7);
    printf("n=%zu bits=%d   naive=%.5f  wcb-nt=%.5f  wcb-wb=%.5f  avx512=%.5f  memcpy=%.5f\n"
           "                ns/elem  %.2f       %.2f        %.2f         %.2f\n",
           n, bits, tN, tNT, tWB, tA, tC, tN*1e9/n, tNT*1e9/n, tWB*1e9/n, tA*1e9/n);
    return 0;
}
