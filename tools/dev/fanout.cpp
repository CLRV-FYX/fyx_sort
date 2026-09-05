// Directly test the stream-fanout hypothesis: one histogram + one WCB
// scatter over the same 8M keys, varying only the digit width (2^Bits dests).
#include "../../fyx_sort.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
using fyx::detail::radix_count_key_pass_banked_wide;
using fyx::detail::radix_scatter_key_pass_wide;
using Key = std::uint32_t;

template <unsigned Bits>
double run(const Key* src, std::size_t n, Key* dst) {
    constexpr std::size_t NB = std::size_t(1) << Bits;
    std::vector<std::uint32_t> cnt(NB);
    std::vector<std::size_t>   off(NB);
    double best = 1e30;
    for (int r = 0; r < 3; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        radix_count_key_pass_banked_wide<Key, Bits>(src, n, 0, cnt.data());
        std::size_t s = 0;
        for (std::size_t d = 0; d < NB; ++d) { off[d] = s; s += cnt[d]; }
        radix_scatter_key_pass_wide<Key, Bits>(src, n, dst, 0, off.data(), true);
        double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        best = std::min(best, dt);
    }
    return best / double(n) * 1e9;
}
template <unsigned... Bs>
void all(const Key* s, std::size_t n, Key* d) {
    ((std::printf("Bits %2u  %6u streams   %6.3f ns/elem\n", Bs,
                  1u << Bs, run<Bs>(s, n, d))), ...);
}
int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 8000000;
    std::vector<Key> src(n), dst(n);
    std::mt19937 rng(999);
    for (std::size_t i = 0; i < n; ++i) src[i] = Key(rng());
    all<8, 9, 10, 11, 12, 13>(src.data(), n, dst.data());
    return 0;
}
