Round-to-round dev tools (kept in-tree so a sandbox reset does not lose them).

- `kvb.cpp`  : `./kvb N kv|str reps [dist]` serial fyx vs std::sort per distribution (bench_core inputs).
- `pb.cpp`   : `./pb N` kv16 serial vs parallel fyx per distribution.
- `fz.cpp`   : randomized correctness fuzz (records with key/greater/lexicographic comparators,
               32-bit records, int64; serial and parallel).  `FZN=40 ./fz` for a short run.
- `run_tests.sh TAG "FLAGS" [tests...]` : build+run test/t_*.cpp with -Wall -Wextra -Werror, 2 jobs.

Build: `g++ -std=c++17 -O3 -march=native -pthread -DHDR='"../../../fyx_sort.hpp"' X.cpp -o X`

## Dead ends (round notes)
- Dictionary counting sort for <=64-class record keys (count -> stable scatter -> copy+proof):
  floor ~0.38 ms at 100k kv16 on the Ice Lake box (1.6 MB memcpy alone 0.096 ms, 16-way
  scatter ~0.2 ms) vs in-place kvq 0.30 ms; 1M 6.2 vs 3.7 ms.  Reverted.
- all_equal: the compare-to-reference scan is already at the bandwidth floor (51.5 us / 100k
  vs std::is_sorted 55.7 us); adjacent-pair proof is slower.  Suite gaps there are harness noise.
