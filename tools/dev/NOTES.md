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

## Random data: what is left, and what to build

The only family where fyx loses to vqsort is uniform random, and the size of
the loss is decided by one number: **the scatter costs 3.14 ns/elem and the
histogram 0.57, so three passes are ~11.1 ns/elem and the scatter is 85% of
the sort.**  Every pass moves the whole array twice (read + write) plus one
read for the histogram.  vqsort's partitions write sequentially, which is why
it gets more bytes per second out of the same machine.

### Three ways to cut the number of passes, all measured, all lost

`tools/dev/kernel.cpp` times `fyx::sort` next to each kernel in one process.
int32 random, parallel:

| variant                                  | 8M      | 16M     |
|------------------------------------------|---------|---------|
| wide sort, 10/11/11 (what runs today)    | 0.03726 | 0.08379 |
| high prefix 26/13, two passes + repair   | 0.05073 | -       |
| high prefix 24/12, two passes + repair   | 0.04419 | 0.11413 |
| high prefix, decline past 3M lifted      | 0.05073 | 0.11045 |

- Lifting the four-byte high-prefix decline (`n > 3<<20`): worse at 4M (2.2x),
  8M (1.2x) and 16M (1.3x).  The decline is correct.
- Split 10/11/11 -> 12/12/8: **does not compile.**  `radix_count_key_pass_
  banked_wide` refuses `Bits = 8`; the constraint is `static_assert(kPerLine *
  sizeof(Key) == kCacheLine)` in `09_radix.hpp:264`.  Deal with that first if
  an eight-bit last pass is wanted again.
- 24/12 is 13% better than 26/13 at 8M (fewer streams per pass) but still 19%
  behind the three-pass wide sort, and 36% behind at 16M.

The lesson is that a two-pass design does not pay above ~4M: the tie repair
eats what the missing pass saved.  Repair is only cheap while the groups are
tiny, and at 8M a 24-bit prefix already puts a fifth of the elements in a
group with someone else.

### What should work: cache-blocked radix

Passes after the first do not have to touch DRAM at all.

1. Pass 1 on **8 bits**: 256 coarse buckets, destination working set 256 x 64 B
   = 16 KB, so the scatter is L1-resident instead of spreading over
   1024-2048 streams.  One histogram + one scatter over the array.
2. Then walk the coarse buckets.  At 8M each holds ~32K elements = 128 KB,
   which is L2-resident, and gets finished with 8-bit digits **in cache**:
   three more histogram+scatter pairs, but on data that is already in L2.

Cost estimate: 1 x 3.7 ns/elem (DRAM) + 3 x ~1.0 ns/elem (L2) = ~6.7 ns/elem
against today's 11.1, i.e. about 1.7x, which turns 8M int32 random from 0.68x
into a win and does the same for the 1M cells that lose 0.48-0.71x.

Two things to be careful about.  The 8-bit histogram must compile (see the
static_assert above).  And this is a new kernel -- budget a full session for
it, verify with `tools/dev/correct.cpp` and `test/t_radix.cpp`, and measure
with paired binaries (this box moves a single cell by +-20% between runs).

### First blocking attempt: measured, and it is a lesson not a patch

`tools/dev/blocked.cpp` builds the blocked kernel against `fyx::detail::` and
races it with `fyx::sort` in one process (8M and 16M, random int32, best of 3):

```
8M    fyx::sort 0.04349   blocked(2 threads) 0.07231   blocked(1 thread) 0.11104
16M   fyx::sort 0.07969   blocked             0.14138   blocked1          0.19974
```

So the naive version is ~1.7x slower, and it is also not currently correct
(the tiny-bucket branch casts a radix key straight to a value, which ignores
the encode/decode transform).

**The reason it is slow is the interesting part, and it is not about locality
at all.** Blocking has a fixed cost per block per pass: the histogram has to
be cleared and the write-combining buffer has to be flushed, and that buffer
holds one line per bucket. With 11-bit digits that is 2048 lines = 128 KB of
flushing for a block that only holds 32 KB of data -- four times more traffic
than the data it is sorting. With 8-bit digits the same fixed cost is 256
lines = 16 KB, so blocks of about 128 KB put the ratio at 8:1 and the idea
works.

The corrected design is therefore narrower than the sketch above:

- pass 1 on **8 bits**, giving 256 coarse buckets (~32K elements, ~128 KB each
  at 8M), one histogram and one scatter over the array;
- each coarse bucket finished with **8-bit** digits, three passes, all inside
  its own 128 KB block plus one equally sized scratch block;
- per block per pass: 16 KB of WCB flush against 128 KB of data, so ~1.1 MB of
  L2 traffic per block instead of today's 96 MB of traffic per pass over the
  whole array.

Eight-bit passes now compile (see the commit before this one). Before porting
anything into `parts/`, fix the key/value handling in the experiment and
confirm it sorts correctly at 8M and 16M -- a wrong kernel that looks 1.7x
slower tells you nothing about the corrected one.
