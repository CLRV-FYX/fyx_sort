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

One scatter pass costs **~6 cycles per element** (see the three-way probe
below).  vqsort spends **~9 cycles per element on the entire sort**.  fyx pays
4 passes x 6 = 24 cycles/element for int32 and 8 x 6 = 48 for int64/double:
the cost is *passes x per-pass cost*, and it is the pass count that hurts.

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
- **A vectorised scatter is a dead end.** AVX-512CD's `vpconflictd` gives every
  lane its rank inside its bucket inside the block, so `vpscatterdd` can write
  straight to the final position with no write-combining buffer at all -- and it
  is still slower, because `vpgatherdd`/`vpscatterdd` cost more than the scalar
  loop they replace (`tools/dev/scatter_probe.cpp`):

  | scatter | cycles/element |
  |---|---|
  | naive `dst[off[b]++] = x` | 12.9 |
  | AVX-512CD conflict + scatter | 14.7 - 17.3 |
  | **write-combining buffer + NT flush (shipping)** | **5.9** |

  The shipping kernel is 2.2x faster than the naive loop it looks like it should
  be simplified to.  Do not "simplify" it.
- **Measure the variants interleaved, in one harness.** Timed back to back, the
  variant that runs first pays for cold pages and measures 2x slower than it is;
  two runs of the same code reported 8.9 and 4.6 cycles/element before the
  harness was fixed.
- **Cost is not linear in the number of passes.** 8-bit values (counting sort)
  0.0011 s, 11-bit (2 passes) 0.0066 s, 32-bit (4 passes) 0.0081 s: the marginal
  pass costs ~0.0005 s, and there is a ~0.006 s fixed cost to using the LSD
  radix at all.

## What follows from the above

Neither traffic nor the per-element kernel is the lever: the kernel is already
2.2x better than the obvious simplification, and it does not respond to cache
tuning.  The lever is **element-passes**.  An MSD radix that makes two
partitioning passes (8 bits, then 8 bits) and finishes each ~16-element bucket
with the existing SIMD sorting network (`parts/07_simd_net.hpp`,
`kNetworkMax = 64`) turns 24 cycles/element into ~12 for int32 and 48 into ~12
for int64/double.  That is the structure vqsort uses, and it is the only
measured route to the random-data gap -- note that it would also make every
numeric type cost the same, where today int64 costs twice int32 purely because
it needs twice the passes.
