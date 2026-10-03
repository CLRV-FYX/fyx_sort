Round-to-round dev tools (kept in-tree so a sandbox reset does not lose them).

- `kvb.cpp`  : `./kvb N kv|str reps [dist]` serial fyx vs std::sort per distribution (bench_core inputs).
- `pb.cpp`   : `./pb N` kv16 serial vs parallel fyx per distribution.
- `fz.cpp`   : randomized correctness fuzz (records with key/greater/lexicographic comparators,
               32-bit records, int64; serial and parallel).  `FZN=40 ./fz` for a short run.
- `run_tests.sh TAG "FLAGS" [tests...]` : build+run test/t_*.cpp with -Wall -Wextra -Werror, 2 jobs.

Build: `g++ -std=c++17 -O3 -march=native -pthread -DHDR='"../../../fyx_sort.hpp"' X.cpp -o X`
