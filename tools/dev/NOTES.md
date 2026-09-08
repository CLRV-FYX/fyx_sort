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

### Second blocking attempt: 8-bit digits, and the whole idea looks wrong here

Same experiment rebuilt with the corrected design -- one 8-bit pass over the
array (256 coarse buckets of ~128 KB at 8M), then three 8-bit passes inside
each block, all key/value handling through the library's encode/decode passes:

```
8M    fyx::sort 0.04362   blocked/2thr 0.06940   blocked/1thr 0.10872
16M   fyx::sort 0.07966   blocked/2thr 0.15117   blocked/1thr 0.22530
```

Still wrong (at one thread too, so not a race -- the per-bucket pass wiring
has a bug) and still far slower. But the timing is the useful part, because
the flush arithmetic said this version should have worked: 16 KB of WCB flush
per pass against a 128 KB block is only ~13% overhead, not 59%.

**The explanation is that blocking pays for memory traffic, and at these sizes
there is no memory traffic to save.** At 8M an int32 array is 32 MB and L3 is
54 MB, so today's three global passes already hit L3; moving them to L2 can
only help by maybe 1.5x on the memory component, which is a minority of the
cost. Meanwhile blocking costs one extra pass -- four passes over the data
against three -- which is +33% of the work before anything else. It is a bad
trade at 1M and 8M and might only turn positive well past L3, around 64M.

So the scatter is not slow because of distance to memory. It is slow because
its writes fan out over 1024-2048 streams, and that hurts even in L3: every
stream is a separate write-combining buffer line and a separate TLB entry.
Which means the next thing to attack is the number of live write streams per
pass, not the distance the writes travel.

### The scatter does not care how many streams it has

`tools/dev/fanout.cpp` times one histogram + one WCB scatter over the same
random keys, changing nothing but the digit width. 8M and 16M agree to within
2%, so these are solid:

```
Bits  8     256 streams   3.056 ns/elem      Bits 12   4096 streams  3.500
Bits  9     512 streams   2.821              Bits 13   8192 streams  4.016
Bits 10    1024 streams   2.961
Bits 11    2048 streams   3.214
```

**Flat from 256 to 2048 streams, and 256 is worse than 512.** The hypothesis
set down after the blocking attempt -- that the scatter is slow because its
writes fan out -- is wrong. The cost is intrinsic to moving one element
through a buffered scatter: roughly 2.3 ns/elem of scatter plus 0.6 of
histogram, and it does not come down until the digit width goes past 4096,
where it gets *worse*, not better.

Two things follow, and both close doors rather than open them:

- four passes of 8 bits would cost 4 x 3.06 = 12.2 ns/elem against the current
  three passes at 10/11/11 = 2.96 + 3.21 + 3.21 = 9.4. More passes with fewer
  streams is strictly worse, which is also why the byte-wise LSD loses.
- the current 10/11/11 split is already within 3% of the best three-pass split
  available (the flat region is wide and the minimum is shallow).

Reconciling with the timings: 9.4 ns/elem of CPU work over two threads is
4.7 ns/elem wall, and `fyx::sort` measures 5.4 at 8M, so overhead is ~0.7.
vqsort is at ~3.6 ns/elem wall there, i.e. ~7.2 ns/elem of CPU work. **The
whole remaining gap is that 9.4 against 7.2**, and since the per-pass cost
will not move, the only way to close it is to do less than three passes of
work -- which every two-pass design tried so far has failed to deliver, or to
find a per-element scatter that is cheaper than 2.3 ns.

### Our comparison sort is not the answer either

`tools/dev/cmpsort.cpp` races `fyx::sort` against `fyx::stable_sort` (the
parallel sample sort that is the `compare` column in the matrix), random int32:

```
8M   sort (radix) 0.04002    stable_sort 0.09856
1M   sort (radix) 0.00553    stable_sort 0.01204
```

Two and a half times slower. So swapping radix for a comparison sort on random
input would throw away more than it could ever gain, even though the fastest
comparison sort we know of is the one beating us there.

### Where that leaves the random-data gap

The accounting is now closed at every level:

- a pass costs 3.1 ns/elem and will not come down (fanout experiment);
- 32 bits over at most 13 bits per pass means three passes is the floor, so
  ~9.4 ns/elem of CPU work is the floor for LSD radix on int32 here;
- blocking cannot help because at 8M the array already lives in L3;
- our comparison sort is 2.5x slower than the radix it would replace.

vqsort sits at ~7.2 ns/elem of CPU work on the same input. Closing 9.4 to
under 7.2 therefore needs a vqsort-class vectorised sorter -- sorting networks
in AVX-512 plus a vectorised merge partition, which is a large piece of work
and not a tuning exercise. Everything cheaper has been tried and measured,
and the numbers are above.

---

## The accounting reopened: the vectorised quicksort got built (2026-09-07)

The section above ends by saying the only route left is "a vqsort-class
vectorised sorter -- sorting networks in AVX-512 plus a vectorised merge
partition". That is what `parts/10b_vsort.hpp` now is, and this section is the
measurement trail for it. Everything below was measured on the same machine
(2 vCPU Ice Lake-SP, g++ 12.2, `-O3 -march=native`), interleaved in one
process, best of N, warm pages -- the hazard from the top of this file bit
twice during this work and both times the numbers were nonsense until the
variants were put back in one harness (see "measurement hygiene" below).

### What the kernel is

* partition in place with `vpcompressd` / `vcompressps`: one compare per
  vector gives a mask, two masked compress-stores drop the low and high halves
  at the two ends of the free gap, nothing is written twice;
* **four vectors per iteration** and the side to read chosen **branchlessly**.
  This is the single biggest tuning result of the whole exercise: the choice
  depends on how the data splits, so at one vector per iteration the branch
  mispredicts about every other time and costs more than the partition does.
  1M random int32, serial, same code otherwise:

  | vectors/iteration, leaf | time |
  |---|---:|
  | 1 vector, branchy, leaf 64   | 0.00604 s |
  | 2 vectors, leaf 64           | 0.00518 s |
  | 4 vectors, leaf 64           | 0.00481 s |
  | 4 vectors, leaf 128          | 0.00429 s |
  | 4 vectors, leaf 256          | 0.00423 s |
  | 8 vectors, leaf 128          | 0.00438 s |

  Eight vectors is worse than four: 2x8 held plus 8 in flight spills.
* the leaf is the existing Batcher network from `parts/_network_body.inc`, but
  driven by policies over the **native** type instead of RadixTraits keys.
  Inside a quicksort the comparisons are already in the value domain, so the
  encode and decode passes the radix networks need are pure loss. Leaf = 16
  vectors (256 elements for 4-byte types, 128 for 8-byte), measured best of
  64/128/256 for both widths.
* pivot = median of two vectors' worth of strided samples, sorted in registers.
  One vector's worth measured the same within noise; 64 samples through
  `small_sort_numeric` (with its encode/decode) was measurably worse.

### Where it stands against the reference implementations

`tools/dev/vqs.cpp` races the prototype against Highway's vqsort and Intel's
x86-simd-sort in one process (build it with `-DHAVE_XSS -DHAVE_VQSORT`):

```
1M   int32   proto 0.00499   x86-simd-sort 0.00398   vqsort 0.00341
1M   float   proto 0.00854   x86-simd-sort 0.00277   vqsort 0.00285
8M   int32   proto 0.05171   x86-simd-sort 0.04050   vqsort 0.03629
8M   float   proto 0.06826   x86-simd-sort 0.03936   vqsort 0.03998
```

that was *before* the unrolling and the native-typed leaf; after them the
shipping kernel is at 0.0042 (1M int32) and 0.0384 (8M float), i.e. level with
x86-simd-sort and ~20% behind vqsort single-threaded. The second core makes up
the difference and more: 1M int32 parallel is 0.0025 against vqsort's 0.0034.
**Anyone continuing this should know the remaining single-thread 20% is real
and unexplained** -- both references use the same partition primitive, so it is
in the details (their leaf networks are hand-written per size, ours is a
generic bitonic; their pivot uses more samples at the top).

### What it did *not* beat, and why those cells kept their kernel

* **64-bit integers.** 8M random int64: quicksort 0.096 s, high-prefix radix
  0.089 s. The high-prefix kernel sorts a 24-26 bit prefix in two passes and
  repairs the ties, so it pays for fewer passes than the quicksort pays for
  levels. `vqsort_preferred()` therefore excludes them. Floating point has no
  such shortcut (the prefix carries the exponent, so its groups are large).
* **Low cardinality.** Counting still wins every shape the profile recognises:
  1M int32 mod8 counting 0.00051 s against the quicksort's 0.00183.
* **Very light disorder.** 4M int32 with 0.02% of positions swapped: patch
  merge 0.0077 s, quicksort 0.0178. See `patch_merge_dirty_budget` for the
  crossover table.

### Two dispatch pathologies this work uncovered

Both were pre-existing and both cost more than the kernel work saved:

* the **high-prefix radix on a near-constant prefix**: 4M int32 drawn from a
  2^22 range took 0.206 s in that kernel alone (plain `radix_sort` 0.035,
  quicksort 0.020) because the whole array lands in one tie group;
* **the range counters were unreachable for high-entropy input**: "narrow
  range" and "few values" were being treated as the same property. 4M int32
  over a 2^10 range: 0.049 s before, 0.0052 s after.

### Degenerate pivot (a real bug, now fixed)

The partition is two-way (`< pivot` | `>= pivot`), so a pivot equal to the
range minimum produces an empty low side and no progress: the recursion just
burned its depth budget and fell out to pdqsort. Ranges where one value owns
more than half the elements hit this. The fix is a second partition with the
strict test (`> pivot`), which puts the pivot-valued block -- at least one
element, already in its final place -- in front of everything else. 2M int32
that are 90% `INT32_MIN`: 0.0019 s.

### Measurement hygiene, again

Two full afternoons of numbers in this session were wrong for the reasons
already written at the top of this file, in new clothes:

* **a background test run.** A `fyx_test.py` left running in another process
  made every measurement 4-5x slow and mutually inconsistent -- including one
  that said the kernel was pathological on nearly-sorted data, which it is not
  (4.5-5.2 ns/elem on random, sorted, reverse and nearly-sorted alike). Check
  `uptime` before believing a surprising number.
* **cold pages.** A harness that allocates a fresh `std::vector` per variant
  measures the page faults, not the sort: the same partition measured 0.0196 s
  warm and 0.0923 s cold. Allocate the work buffer once, `std::copy` into it.
