# FYX vs IPS4o local comparison

`bench_ips4o_compare.cpp` compares this repository's `fyx::sort` with an unpacked IPS4o checkout. It reports the best of 3 runs and excludes input-copy time.

## Build

Sequential IPS4o / sequential FYX:

```bash
g++ -std=c++17 -O3 -march=native -DNDEBUG -DFYX_DISABLE_PARALLEL \
  -I. -I/path/to/ips4o/include bench/bench_ips4o_compare.cpp -o /tmp/bench_ips4o_seq
```

FYX parallel enabled, IPS4o sequential:

```bash
g++ -std=c++17 -O3 -march=native -DNDEBUG -DFYX_BENCH_FYX_PARALLEL \
  -I. -I/path/to/ips4o/include bench/bench_ips4o_compare.cpp -pthread -o /tmp/bench_fyx_parallel
```

FYX parallel vs IPS4o parallel with real TBB:

```bash
g++ -std=c++17 -O3 -march=native -DNDEBUG \
  -DFYX_BENCH_FYX_PARALLEL -DFYX_BENCH_IPS4O_PARALLEL \
  -I. -I/path/to/oneTBB/include -I/path/to/ips4o/include \
  bench/bench_ips4o_compare.cpp -pthread \
  -L/path/to/oneTBB/lib -Wl,-rpath,/path/to/oneTBB/lib -ltbb -latomic \
  -o /tmp/bench_ips4o_parallel
```

Upstream IPS4o includes `tbb/concurrent_queue.h` even for the sequential entry point. In an environment without TBB, a minimal compatibility header can compile the sequential path, but it must not be used for a fair parallel IPS4o benchmark.

## Snapshot: 2 vCPU Intel Xeon, GCC 12.2, `-O3 -march=native`

### FYX sequential vs IPS4o sequential

| case | n | FYX | IPS4o | std::sort | FYX vs IPS4o |
|---|---:|---:|---:|---:|---:|
| i32 random | 5,000,000 | 0.0688s | 0.1299s | 0.4043s | 1.89x faster |
| i32 low distinct 16 | 5,000,000 | 0.0072s | 0.0184s | 0.1268s | 2.57x faster |
| i64 sparse distinct 256 | 3,000,000 | 0.0160s | 0.0197s | 0.1156s | 1.24x faster |
| string random len16 | 1,000,000 | 0.1540s | 0.1970s | 0.3528s | 1.28x faster |
| string low distinct 64 | 1,000,000 | 0.0212s | 0.0652s | 0.2063s | 3.07x faster |
| struct key distinct 64 | 1,000,000 | 0.0048s | 0.0052s | 0.0313s | 1.08x faster |
| struct key random 1M | 1,000,000 | 0.0138s | 0.0278s | 0.0747s | 2.01x faster |

### FYX parallel vs IPS4o parallel

This was run with real oneTBB and linked with `-ltbb -latomic`.

| case | n | FYX parallel | IPS4o parallel | std::sort | FYX vs IPS4o |
|---|---:|---:|---:|---:|---:|
| i32 random | 5,000,000 | 0.0585s | 0.0687s | 0.4032s | 1.17x faster |
| i32 low distinct 16 | 5,000,000 | 0.0073s | 0.0108s | 0.1261s | 1.48x faster |
| i64 sparse distinct 256 | 3,000,000 | 0.0103s | 0.0114s | 0.1157s | 1.10x faster |
| string random len16 | 1,000,000 | 0.0941s | 0.1115s | 0.3650s | 1.18x faster |
| string low distinct 64 | 1,000,000 | 0.0209s | 0.0378s | 0.2060s | 1.81x faster |
| struct key distinct 64 | 1,000,000 | 0.0040s | 0.0050s | 0.0316s | 1.27x faster |
| struct key random 1M | 1,000,000 | 0.0127s | 0.0142s | 0.0752s | 1.12x faster |

Interpretation: on this 2-vCPU matrix, FYX now leads both IPS4o sequential and real IPS4o parallel across all tracked cases. Keep this benchmark in CI/manual release checks; any future optimization must preserve these seven distributions, especially both low-distinct and high-distinct struct-key custom-comparator payloads.
