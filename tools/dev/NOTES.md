# Measurements that are expensive to repeat

Everything here was measured on the sandbox: 2 vCPU Xeon Ice Lake-SP (AVX-512),
GCC 12.2, `-O3 -march=native`, ~2.5 GHz.  Absolute numbers will differ on other
machines; the structural conclusions should not.

## What the machine can do

| workload (4 MiB working set) | rate |
|---|---|
| `memcpy` | **16.9 GB/s** |
| non-temporal write | **18.2 GB/s** |
| `memcpy`, 64 MiB | 11.1 GB/s |
| clock (200M dependent imul+add) | ~2.5 GHz |

`tools/dev/microbench.cpp` reproduces this.  It matters because a kernel that
runs at 1.5 GB/s is *not* bandwidth bound, and no amount of cache tuning will
help it.

## vqsort (google/highway 1.4.0) versus fyx, 1M, best of 3

Reproduce with `tools/dev/vqsort.sh`.  Full table is written to
`build/vqsort_1000000.txt`.

| distribution | fyx par | vqsort | std::sort | ips4o | pdqsort |
|---|---|---|---|---|---|
| random int32 | 0.0103 | **0.0035** | 0.0702 | 0.0249 | 0.0284 |
| random int64 | **0.0073** | 0.0078 | 0.0704 | 0.0271 | 0.0301 |
| random double | 0.0106 | **0.0086** | 0.0784 | 0.0289 | 0.0327 |
| mod8 int32 | 0.0010 | **0.0005** | 0.0172 | 0.0028 | 0.0024 |
| lowcard16 int32 | 0.0009 | **0.0007** | 0.0263 | 0.0035 | 0.0045 |
| nearlysorted int32 | **0.00076** | 0.0034 | 0.0094 | 0.0156 | 0.0055 |
| zigzag int32 | **0.00052** | 0.0037 | 0.0436 | 0.0178 | 0.0018 |
| rotated int64 | **0.0027** | 0.0084 | 0.0103 | 0.0173 | 0.0123 |
| concat2 int64 | **0.0030** | 0.0083 | 0.0736 | 0.0196 | 0.0308 |
| string (16 chars) | **0.098** | unsupported | 0.356 | 0.211 | 0.322 |

**vqsort wins only on high-entropy, structureless numeric data** (worst case
2.9x, 1M int32).  fyx wins every structured shape, and every string shape by
default: vqsort has no string sort at all.  vqsort is also single threaded.

## Where the radix sort spends its time

1M int32, full-range random, serial radix.  `tools/dev/instrument.py` plus
`tools/dev/radix_timer.cpp`.

| stage | share |
|---|---|
| fused histogram | 7% |
| encode | 7% |
| **scatter (4 passes)** | **81%** |
| decode | 3% |

A scatter pass moves 8 MB (4 read + 4 write) in 0.0046 s = **1.45 GB/s**, which
is 8% of what the machine does, and works out to **11.5 cycles per element**.
vqsort spends **9 cycles per element on the whole sort**.  fyx pays
4 x 11.5 = 46 cycles/element for int32 and 8 x 11.5 = 92 for int64/double:
the cost is *passes x per-pass cost*, and the pass count is what hurts.

## Negative results — do not spend time on these again

- **Non-temporal stores are load bearing.** Forcing `can_stream = false` makes a
  scatter pass 1.5x slower for int32 and 3x slower for int64.
- **The number of open output streams is irrelevant.** 256 streams down to 8,
  same data size: 11.5 cycles/element in every case.  The DRAM-row-conflict
  theory is wrong (`tools/dev/streams.cpp`).
- **Deepening the write-combining buffer is worth 6-18%, not more.** One cache
  line per bucket -> 2 -> 4 -> 8 gives 0.00539 -> 0.00503 -> 0.00481 -> 0.00443
  s per pass for int64.  Real, small.
- **The per-element software prefetch is roughly neutral** (slightly better
  without it for int32, slightly worse for int64).
- **Cost is not linear in the number of passes.** 8-bit values (counting sort)
  0.0011 s, 11-bit (2 passes) 0.0066 s, 32-bit (4 passes) 0.0081 s: the marginal
  pass costs ~0.0005 s, and there is a ~0.006 s fixed cost to using the LSD
  radix at all.

## What follows from the above

Reducing traffic is not the lever; reducing *element-passes* is.  An MSD radix
that makes one or two partitioning passes and finishes each cache-sized bucket
with the existing SIMD sorting network (`parts/07_simd_net.hpp`) turns
46 cycles/element into ~16 for int32 and 92 into ~15 for int64/double, which is
the structure vqsort uses and the only measured route to closing the random-data
gap.
