// dispatch decision + time per shape: dd N DIST
#define FYX_ENABLE_TEST_HOOKS 1
#include HDR
#include "../../../bench/suite/core/bench_core.hpp"
#include <cstdio>
typedef TY T;
int main(int, char** v) { std::size_t n = std::strtoull(v[1], 0, 10); int d = std::atoi(v[2]);
  double best = 1e30; int dec = -1;
  for (int r = 0; r < 20; ++r) { auto x = fb::make_input<T>(n, (fb::Dist)d, 77 + r);
    auto t = std::chrono::steady_clock::now(); fyx::sort(x.begin(), x.end());
    best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count());
    dec = int(fyx::detail::test_last_dispatch()); }
  std::printf("%s n=%zu dist=%d: %.1fus dispatch=%d\n", fb::dist_name((fb::Dist)d), n, d, best, dec); }
