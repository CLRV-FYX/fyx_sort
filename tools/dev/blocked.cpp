// STATUS: FAILED, KEPT AS EVIDENCE (8-bit variant). Output is WRONG at every thread count, so it is not a race -- the bug is in the per-bucket pass wiring, and nobody has found it yet. It is also ~59% slower than fyx::sort, which is the part that matters. Do not use.
// Experiment: cache-blocked radix for 4-byte keys, corrected design.
//   pass 1  : 8-bit LSD over the whole array -> 256 coarse buckets (~128 KB each)
//   finish  : per coarse bucket, three 8-bit passes (shifts 8/16/24) confined to
//             that block + one scratch block, so the per-block fixed cost is a
//             16 KB WCB flush against 128 KB of data.
// All key<->value handling goes through the library's encode/decode passes.
#include "../../fyx_sort.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using fyx::detail::radix_count_key_pass_banked_wide;
using fyx::detail::radix_count_value_pass_banked_wide;
using fyx::detail::radix_scatter_encode_pass_wide;
using fyx::detail::radix_scatter_key_decode_pass_wide;
using fyx::detail::radix_scatter_key_pass_wide;

using Key = std::uint32_t;
static const unsigned CB = 8, PB = 8;
static const std::size_t NCOARSE = std::size_t(1) << CB;
static const std::size_t NPB     = std::size_t(1) << PB;

static void finish_range(const Key* keys, const std::size_t* starts,
                         std::size_t b0, std::size_t b1,
                         std::int32_t* out, Key* scratch) {
    std::vector<std::uint32_t> cnt(NPB);
    std::vector<std::size_t>   off(NPB);
    for (std::size_t b = b0; b < b1; ++b) {
        const std::size_t lo = starts[b], hi = starts[b + 1], nb = hi - lo;
        if (nb == 0) continue;
        const Key* src = keys + lo;
        radix_count_key_pass_banked_wide<Key, PB>(src, nb, CB, cnt.data());
        std::size_t s = 0;
        for (std::size_t d = 0; d < NPB; ++d) { off[d] = s; s += cnt[d]; }
        radix_scatter_key_pass_wide<Key, PB>(src, nb, scratch, CB, off.data(), false);

        radix_count_key_pass_banked_wide<Key, PB>(scratch, nb, CB + PB, cnt.data());
        s = 0;
        for (std::size_t d = 0; d < NPB; ++d) { off[d] = s; s += cnt[d]; }
        radix_scatter_key_pass_wide<Key, PB>(scratch, nb, const_cast<Key*>(src),
                                             CB + PB, off.data(), false);

        radix_count_key_pass_banked_wide<Key, PB>(src, nb, CB + 2 * PB, cnt.data());
        s = 0;
        for (std::size_t d = 0; d < NPB; ++d) { off[d] = s; s += cnt[d]; }
        radix_scatter_key_decode_pass_wide<std::int32_t, Key, PB>(
            src, nb, out + lo, CB + 2 * PB, off.data(), false);
    }
}

static void blocked_sort(std::int32_t* p, std::size_t n, int nthreads) {
    std::vector<Key>           keys(n);
    std::vector<std::size_t>   starts(NCOARSE + 1), off(NCOARSE);
    std::vector<std::uint32_t> cnt(NCOARSE);

    radix_count_value_pass_banked_wide<std::int32_t, CB>(p, n, 0, cnt.data());
    std::size_t s = 0;
    for (std::size_t d = 0; d < NCOARSE; ++d) { starts[d] = s; s += cnt[d]; }
    starts[NCOARSE] = s;
    std::memcpy(off.data(), starts.data(), NCOARSE * sizeof(std::size_t));
    radix_scatter_encode_pass_wide<std::int32_t, Key, CB>(
        p, n, keys.data(), 0, off.data(), true);

    std::size_t maxb = 0;
    for (std::size_t b = 0; b < NCOARSE; ++b)
        maxb = std::max(maxb, starts[b + 1] - starts[b]);
    std::vector<Key> scratch(std::size_t(nthreads) * (maxb + 64));

    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; ++t) {
        std::size_t b0 = (NCOARSE * std::size_t(t)) / std::size_t(nthreads);
        std::size_t b1 = (NCOARSE * std::size_t(t + 1)) / std::size_t(nthreads);
        th.emplace_back(finish_range, keys.data(), starts.data(), b0, b1, p,
                        scratch.data() + std::size_t(t) * (maxb + 64));
    }
    for (auto& x : th) x.join();
}

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 8000000;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    std::vector<std::int32_t> base(n), w1(n), w2(n), ref(n);
    std::mt19937 rng(12345);
    for (std::size_t i = 0; i < n; ++i) base[i] = std::int32_t(rng());
    ref = base; std::sort(ref.begin(), ref.end());

    w1 = base; blocked_sort(w1.data(), n, 2);
    std::printf("blocked(2thr) correct: %s\n", w1 == ref ? "yes" : "NO");
    w1 = base; blocked_sort(w1.data(), n, 1);
    std::printf("blocked(1thr) correct: %s\n", w1 == ref ? "yes" : "NO");

    auto bench = [&](const char* name, auto&& fn) {
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            w2 = base;
            auto t0 = std::chrono::steady_clock::now();
            fn(w2.data(), n);
            double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            best = std::min(best, dt);
            if (w2 != ref) std::printf("%s WRONG\n", name);
        }
        std::printf("%-10s n=%9zu  %8.5f s\n", name, n, best);
    };
    bench("fyx::sort", [](std::int32_t* q, std::size_t m) { fyx::sort(q, m); });
    bench("blocked",   [](std::int32_t* q, std::size_t m) { blocked_sort(q, m, 2); });
    bench("blocked1",  [](std::int32_t* q, std::size_t m) { blocked_sort(q, m, 1); });
    return 0;
}
