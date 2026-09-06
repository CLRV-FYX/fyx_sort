// tools/dev/hist.cpp -- how fast can one sweep fill a radix histogram?
//
//   bank4   four banks, unrolled by four, increments independent  (shipping)
//   bank2   two banks
//   avx512  vpconflictd + vpopcntd + gather + scatter, 16 lanes   (in the
//           header since 09_radix, but never called anywhere)
//
//     g++ -std=c++17 -O3 -march=native -DNDEBUG -I. tools/dev/hist.cpp -o /tmp/hist
//     /tmp/hist 1000000 12
#include "../../fyx_sort.hpp"
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <array>
using namespace std::chrono;
static double now(){ return duration<double>(steady_clock::now().time_since_epoch()).count(); }
using fyx::detail::RadixTraits;
using fyx::detail::prefetch_stream;

template <unsigned Bits, unsigned Banks>
void bank(const std::uint32_t* src, std::size_t n, unsigned shift, std::uint32_t* out) {
    constexpr std::size_t B = std::size_t(1) << Bits;
    std::vector<std::uint32_t> bk(static_cast<std::size_t>(Banks) * B, 0);
    std::size_t i = 0;
    for (; i + Banks <= n; i += Banks) {
        prefetch_stream(src, i, n);
        for (unsigned j = 0; j < Banks; ++j)
            ++bk[std::size_t(j) * B + ((src[i + j] >> shift) & (B - 1))];
    }
    for (; i < n; ++i) ++bk[(src[i] >> shift) & (B - 1)];
    for (std::size_t d = 0; d < B; ++d) {
        std::uint32_t s = 0; for (unsigned j = 0; j < Banks; ++j) s += bk[std::size_t(j) * B + d];
        out[d] = s;
    }
}
template <class F> double bench(F f, int reps) {
    double b = 1e30; for (int r = 0; r < reps; ++r) { double t0 = now(); f(); double t1 = now(); b = std::min(b, t1 - t0); } return b;
}
int main(int argc, char** argv) {
    std::size_t n = argc > 1 ? std::strtoull(argv[1], 0, 10) : 1000000;
    int bits = argc > 2 ? std::atoi(argv[2]) : 12;
    const std::size_t B = std::size_t(1) << bits;
    const unsigned shift = 32 - unsigned(bits);
    std::mt19937_64 rng(5);
    std::vector<std::uint32_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = std::uint32_t(rng());
    std::vector<std::uint32_t> a(B), b(B), c(B);
    bank<12,4>(v.data(), n, shift, a.data());
    bank<12,2>(v.data(), n, shift, b.data());
    std::fill(c.begin(), c.end(), 0);
    fyx::detail::isa_avx512_hist::histogram_u32_avx512(v.data(), n, shift, c.data());
    printf("n=%zu bits=%d  bank4==bank2: %d   bank4==avx512: %d\n", n, bits,
           (int)(a == b), (int)(a == c));
    double t4 = bench([&]{ bank<12,4>(v.data(), n, shift, a.data()); }, 9);
    double t2 = bench([&]{ bank<12,2>(v.data(), n, shift, b.data()); }, 9);
    double ta = bench([&]{ std::fill(c.begin(), c.end(), 0);
                           fyx::detail::isa_avx512_hist::histogram_u32_avx512(v.data(), n, shift, c.data()); }, 9);
    double tc = bench([&]{ std::fill(c.begin(), c.end(), 0); }, 9);
    printf("  bank4=%.5f  bank2=%.5f  avx512=%.5f  (zeroing=%.5f)   ns/elem %.2f / %.2f / %.2f\n",
           t4, t2, ta, tc, (t4-tc)*1e9/n, (t2-tc)*1e9/n, (ta-tc)*1e9/n);
    return 0;
}
