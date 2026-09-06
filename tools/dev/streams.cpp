// Does the cost of a scatter pass depend on how many output streams it keeps
// open?  It does not -- see NOTES.md.  Kept so the negative result can be
// re-checked when the kernel changes.
#include FYX_HDR
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
using namespace fyx::detail;
int main() {
    const std::size_t n = 1000000;
    std::mt19937_64 g(7);
    std::printf("%-9s %-8s %-12s %-12s\n", "streams", "passes", "best/pass", "cycles/elem");
    for (unsigned N : {256u, 128u, 64u, 32u, 16u, 8u}) {
        std::vector<std::int32_t> v(n);
        const std::uint32_t mask = N - 1;
        for (auto& x : v) {                      // every byte takes N values only
            std::uint32_t out = 0;
            for (int p = 0; p < 4; ++p) out |= (std::uint32_t)(g() & mask) << (p * 8);
            x = (std::int32_t)out;
        }
        double best = 1e9; int passes = 0;
        for (int r = 0; r < 7; ++r) {
            auto w = v;
            RadixTimers::reset();
            (void)radix_sort<std::int32_t>(w.data(), w.size());
            RadixTimers& t = RadixTimers::get();
            const double per = t.passes ? t.scat / t.passes : 0.0;
            if (per < best) { best = per; passes = t.passes; }
            if (!std::is_sorted(w.begin(), w.end())) { std::printf("  !! unsorted N=%u\n", N); break; }
        }
        std::printf("%-9u %-8d %-12.5f %-12.1f\n", N, passes, best, best * 2.5e9 / n);
    }
    return 0;
}
