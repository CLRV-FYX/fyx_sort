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

### What is still losing, and what the numbers say about why

The matrix after this work is 33 W / 9 L at 1M and 36 W / 6 L at 8M (arithmetic
cells only; `BENCHMARKS.md` has every number). The losses are no longer one
family, they are three unrelated things, and none of them is a tuning knob:

**1. Periodic, few-value input (`mod8`: 0.66x at 1M int32, 0.93x at 8M).**
The counting kernel itself is not the problem -- called directly it is 0.00061 s
against vqsort's 0.00053 for 1M int32. The gap is what runs *before* it:
`fyx::sort` measures 0.00081 on the same input, so ~0.0002 s goes on the
distribution weapons and the profile declining, one after another. On a sort
whose whole job costs half a millisecond that is 25-40% of the runtime. Fixing
it means making the weapon chain cheaper to decline (each weapon's early-exit
budget, and the order they are tried in), not making the counter faster.

**2. Block-swapped input (`blockswap` double: 0.71x at 8M, int32 0.92x).**
8M sorted doubles with twenty 128-element blocks swapped: we take 0.062 s
through the displacement patch merge, pdqsort takes 0.044. The patch merge is
already galloping (`merge_patch_back` uses exponential search plus bulk moves),
so the cost is the two characterisation passes plus the compaction -- four
sequential passes over 64 MB -- and all four are sequential while pdqsort's
comparisons are branch-predictable on this shape. The route is a *parallel*
prefix-max / suffix-min characterisation and a parallel compaction; both are
two-level scans, neither is hard, but they are real work.

**3. Ties (`allequal`, `lowcard16`: 0.91x-0.99x).** These are memcmp-bound and
within the run-to-run swing of this machine. Not worth chasing before the two
above.

Also open, and worth knowing:

* **8M double over a narrow range** (values 0..1e6): the quicksort takes it at
  0.064 s parallel where the radix path did 0.058. The array is 64 MB, past
  L3, and the quicksort's levels cost more there than radix's skipped passes.
  A size-and-span gate would fix it, but the difference is 10% and this machine
  swings 20%, so it was left alone rather than fitted to noise.
* **The single-thread 20% against vqsort** described above.
* `FYX_ENABLE_GPU=1 FYX_GPU_COMPUTE=1` compiles now but has never run: there is
  no GPU on this machine.

---

### 2026-09-09: canonical-matrix rerun; the "remaining losses" list above is mostly obsolete

The matrix numbers above (and the ones quoted in README/CHANGELOG before this
date) were taken from **filtered** bench_matrix runs (`--filter=blockswap` etc).
bench_matrix draws every dataset from ONE shared `g_rng` stream that advances
in matrix order, so a filtered run consumes a different stream prefix and
generates *different data* than the same row in a full run.  Two afternoons of
"blockswap losses" and the /tmp/ins replica analysis (first descent at 1536,
1.47M insertion shifts) were artefacts of that: on the real full-run datasets
the current dispatch already wins every blockswap cell (8M double ~1.5x vs
pdqsort, int32 ~3.7-4.3x).  bench_matrix now warns about this in its header.

Canonical full-matrix state (pdqsort + vqsort references, one clean run, 1M
reps 7 / 8M reps 5, machines swings 20-60% between runs as always):

* 1M: 37 W / 5 L -- losses are mod8 (int32/int64/double 0.67x-0.81x) and
  allequal (int32/int64 0.84x-0.88x) only.  lowcard16 now wins everywhere.
* 8M: 40 W / 2 L -- double rotated 0.98x vs vqsort, int64 allequal 0.97x; both
  inside run-to-run noise.  double zigzag, the one real 8M comparison loss
  (0.92x vs pdqsort through the organ-pipe detector), is fixed: the two
  full-range shape scans are fused into one pass, and the cell is now ~1.5x in
  paired runs (1M double zigzag ~1.9-2.0x).

Why the 1M few-value cells still lose and what that means:

* lc2-style anatomy (whole fyx sort vs every counting kernel vs vqsort, mod8 /
  lowcard16 / allequal, 1M and 8M): at 1M the dispatch+profile adds ~0.0002 s
  on top of a kernel that already trails vqsort's small-range partition by
  10-20% for int32/int64 mod8 -- so zeroing the dispatch cost would NOT flip
  those cells.  The double mod8 kernel is at parity with vqsort already
  (whole ~= kernel ~= vqsort).  At 8M the same kernels beat vqsort by
  10-90% and dispatch overhead is ~0.  Conclusion: the few-value 1M rows are
  a kernel problem at small n, not a dispatch-order problem, and are a few
  tenths of a millisecond on sorts 15-30x faster than std::sort.  Left alone.
* allequal rows are memcmp/scan-bound ties at both sizes (0.84x-1.05x).

---

### 2026-09-10: the matrix now has six opponents, and one of them was beating us on trivial input

The comparison matrix used to carry `std::sort`, `pdqsort` and `vqsort`.  It now
also carries **IPS4o serial and parallel (through oneTBB)** and
**intel/x86-simd-sort** (header-only build, `x86simdsortStatic::qsort`), because
"we beat pdqsort and vqsort on everything structural" is a weaker claim than it
sounds: IPS4o is the other multi-threaded library in this field and x86-simd-sort
is the other AVX-512 sorter.  `tools/dev/vqsort.sh` builds all six and skips
(rather than fakes) the IPS4o parallel column when oneTBB is missing.  One thing
worth knowing about this sandbox: `third_party/` is gitignored but *is*
persisted, while `/tmp` and `build/` are not, so keep anything expensive
downloaded there.

What the extra opponents changed:

* **IPS4o parallel was beating our whole sort on already-sorted 8M input,
  0.48x-0.60x, and on already-sorted 1M strings 0.55x.**  Not because its
  quicksort is fast on sorted data -- because its *sorted check is parallel* and
  ours was not.  On trivial input the check is the runtime.  Fixed by
  `try_parallel_fast_order_exit` (see CHANGELOG): sample gate, then the serial
  detector per chunk in parallel with early bail-out, then seam checks.  Same
  proof strength, exactly the serial decision.  8M int32 0.60x -> 1.09x, int64
  0.48x -> 0.61x, double 0.65x -> 0.91x-1.11x.  Threshold is 2M: at 1M the
  wake-up costs more than it saves (measured +8-14% on the 1M nearly-sorted
  row), and splitting the 1M *string* scan gained nothing measurable.
* **x86-simd-sort is the best of the six on small-value-domain input**, which is
  where our counting kernels still lose: 1M double mod8 0.68x, 8M double mod8
  0.72x, 1M/8M allequal 0.82x-0.99x, while 8M double lowcard16 is 1.09x (we
  win).  So the few-value losses are not a vqsort-specific artefact -- two
  independent AVX-512 implementations beat our counting dispatch on
  1M-and-under few-value ranges, which says the gap is the counting kernel at
  small n, not the dispatcher.  Consistent with the lc2 anatomy in the previous
  entry: at 8M our counters win big, at 1M they trail.
* **IPS4o parallel is the other thing that beats us on `string` sorted 1M**
  (0.55x; we win the string nearly-sorted row 5.1x), again a
  parallel-verification effect, and it is the best opponent on 8M int64
  allequal (0.76x).
* Everything else holds: we beat IPS4o serial on 53/56 cells at 1M and 39/42 at
  8M, IPS4o parallel 53/56 and 38/42, x86-simd-sort 35/42 and 35/42, vqsort
  36/42 and 37/42, and pdqsort / std::sort on every cell.  The per-opponent
  breakdown and the loss list are generated into `BENCHMARKS.md` by
  `tools/dev/mkbench_md.py`; the parser reads the opponent names out of the
  table header, so the document follows whatever the binary was built with.

Measurement trap that cost time again: **`bench_matrix --filter=` runs are not
comparable to full runs.**  Every dataset comes from one shared `g_rng` whose
stream advances in matrix order, so filtering changes the stream prefix and
therefore the data.  Sorted / reverse / all-equal / mod8 are the only shapes
whose data does not depend on the rng at all, which makes them the only rows
that can be spot-checked with a filtered run.

### 2026-09-10 (later): what the *shape* of the parallel proof costs

First version of the parallel orderedness exit classified each chunk with
`detect_fast_order_kind`, the function built to answer *either* direction.  It
worked on int32 but 8M int64 was still 0.0054 s against IPS4o parallel's
0.0033, and chasing that produced three findings worth keeping:

* **A proof that touches every element should touch it once, with one
  comparison per element.**  `detect_fast_order_kind` asks "up? down? equal?"
  per element (two comparisons), while a striped `std::is_sorted` asks one.  The
  sample gate already decides the direction, so the validation only has to
  confirm that direction: 8M int64 0.0054 -> 0.0034 s, int32 0.0037 -> 0.0029 s,
  and the cells went to 0.96x-1.07x against IPS4o parallel (from 0.48x-0.65x).
  Every chunk validates the pair straddling its seam, so the union of the chunks
  is the whole range -- this is not a sampling shortcut.
* **Encode once per element, not twice.**  For floating point the first
  direction-validating loop called `RadixTraits<double>::encode` on both ends of
  every pair and was 3x slower than the loop it replaced (8M double 0.0043 ->
  0.0164 s); caching the previous element's key fixed it.
* **The all-equal memcmp pass is a second full read.**  `detect_fast_order_kind`
  starts with a memcmp over the range for the all-equal answer.  When the gate
  has already witnessed a strictly ordered pair, that answer is impossible, and
  `detect_fast_order_kind` now takes an `all_equal_possible` argument so the
  pass can be skipped.  (It turned out not to matter much on this machine --
  chunks are 512K elements, so the comparison pass after the memcmp was reading
  L3, not DRAM -- but it is free and it removes the second read on other memory
  hierarchies.)

Also measured and rejected: lowering `kParallelOrderMinN` to 1M.  It makes the
1M sorted *numeric* rows marginally faster but taxes every 1M range that
declines (nearly-sorted 1M: +8-14%) because the pool wake-up is ~30 us, and the
1M sorted *string* scan (the worst remaining cell, 0.55x against IPS4o parallel)
gained nothing measurable: in that case the scan is 1.83 ms parallel against
2.94 ms serial, but the whole call is 5.3 ms either way, because the string
objects are re-read cold after the buffer copy the harness does -- the cost is
pointer chasing, not the proof.

Method note for future rounds: measure these with the harness's own
methodology.  A proof timed on a warm buffer (`try_parallel_fast_order_exit` in
a loop: 0.0016-0.0025 s at 8M) is *not* comparable to the same proof inside a
sort that starts right after a 32-64 MB buffer copy (0.003-0.005 s): the second
number is the honest one for "sort this buffer", and comparing the two nearly
sent me optimising the wrong thing.

## 2026-09-11 — 池辅助有序性证明（200 万以下）与"单线程扫描"缺陷

**缺陷本身**。1M 排序已排序的 string，IPS4o 并行 0.0037 s 对我们 0.0069 s。机制：
排序本体在该规模走串行（`kParallelOrderMinN = 2M`），有序性证明也跟着单线程——
而 IPS4o 的检查是条带化到线程池上的 `std::is_sorted`。平凡输入上检查就是全部运行时间，
单线程扫描正是它的知名缺陷：第二个核完全空转。串行侧没有比较次数可省
（`detect_fast_order_kind` 已经是"首对定方向 + 单向 1 比较/元素"），缺陷纯粹是"单线程"。

**第一版补丁直接复用 8M 的门——被测量否决**。`try_parallel_fast_order_exit` 的门是
4096 个等距（跨步）探针。8M 时这笔开销摊薄在 64 MB 扫描里无所谓；但刚被调用方写入过的
1M 数组是冷的，4096 个跨步探针 = 4096 次无空间局部性的硬未命中 ≈ 0.3 ms。A/B 实测：
1M double 已排序反而 +0.1 ms（0.45→0.55 ms），string 不赚不赔。**跨步探针的成本模型
随规模翻转**——这是"门比它把守的扫描还贵"的实例。

**最终形状：顺序前缀代替跨步门**。前 4096 个元素用串行检测器分类（顺序读、预取友好）：
乱序 → 微秒级拒绝，池根本不醒（近随机输入 ~2-3 个比较后就退出；nearlysorted 的首个
乱序对 98% 落在前缀内）；Sorted/Reverse → 剩余部分按单向验证铺到池上（每元素 1 比较、
chunk 起点自带跨缝对、浮点缓存前一个编码键）；AllEqual 前缀 → 逐 chunk 分类 + 块缝合并。
门槛：`kParallelProofMinN = 128K` 且 `n·sizeof(T) ≥ 3 MB`（int32 1M = 4 MB 恰好入圈），
再叠加既有的 `dynamic_parallel_allowed`（Auto 模式默认 ≥1M 才有池）。

**被测试抓住的 bug**。合并循环第一版把块缝边界错位了一格，最后一格在 `p[n]` 越界读了
垃圾对。语义上只会"错误拒绝"（垃圾对拆掉 can_sort/can_reverse），不会错误证明——
但「全等前缀 + 严格递减尾部」的等价性测试当场红掉。教训：接缝枚举要用
"entry c 与 entry c+1 的缝在 chunk c 的起点"这种一一对齐的写法，别复用大循环下标。

**测量纪律的又一次教训**。bench_matrix 里 `fyx`（parallel=Off）列在新二进制上系统性
显示 2x 慢——孤立探针（单进程只测串行 sort）显示两者逐位相同（0.000295 s），
是同进程内 TBB/布局干扰，不是代码。decliner A/B 里对手列（std、ips4o）自己也漂
20-150%。结论不变：只信全量跑与配对交替测量，孤证一律复核。

**结果**（同机背靠背全量，vs IPS4o 并行）：1M 已排序 int32 0.94x→1.52x、
int64 0.65-0.83x→1.26x、double 0.53x→1.08x、string 0.55x→0.94-1.03x；
8M int32 1.04→1.22x、double 0.93→1.16x、int64 0.91-1.10x（摆动）。
decliner 全部持平或更好（全等四类反而更好：并行分类路径）。全量 1M 48W/8L、8M 41W/1L。

**遗留**。8M int64 已排序在坏窗口下 0.91x：8M 路径的门仍是跨步探针。把 ≥2M 的门也换成
顺序前缀（让 8M 复用辅助实现）已试过并回退：3 轮 8M 已排序交替 A/B 无可测收益——
8M 数组 32-64 MB 大半还在 54 MB L3 里，跨步探针多数命中 L3 且乱序引擎能把独立探针
流水化，实际成本远低于冷内存的 ~0.3 ms 估计，信号被 int64/double 已排序 20-30% 的
逐轮摆动淹没。无收益不换已验证路径（备份分支 `backup/pre-gate-experiment` 保留了
实验代码）。8M sorted int64 的逐格数字建议按 0.91x–1.26x 区间引用。

## 2026-09-11（二） — 样本窗口贯通与低基数计数精简（普适性改造）

**动机**。目标从「对某个对手/某台机器调参」改为「绝对通用」。低基数数据（8-256 个
不同值：状态码、枚举、类别标签）是所有领域最常见的形态，而计数路径里存在与机器无关
的纯浪费：同一条 1024 点跨步抽样被 profile、`sample_distinct_keys`、各计数内核
重复读了 2-4 次；串行密集计数在「样本已给出值域窗口」的情况下仍做全量 minmax；
稀疏表固定 1024 槽（约 17 KB，与数据流互相驱逐）；并行内核对已值初始化的 chunk 表
再 fill(0) 一遍；计数表用 8 字节 size_t。

**改法**。`SampleWindow<T>`（encoded lo/hi + exact distinct）挂在 `InputProfile` 上，
四条计数内核（串行/并行 × 密集/稀疏）全部改为消费它：窗口即下界，键落在窗口外立即
逃逸回精确路径（与并行内核一直使用的设计对齐）；64 槽小表在 distinct ≤ 24 时常驻
L1，48 处饱和让位。逃逸在构造上只可能多花一趟、不可能错排序（计数按构造产出正确输出）。

**测量的诚实结论**。本机当日噪声 ±50-70%（同一二进制的 double/reverse 8M 格两次
全量跑 0.0098 与 0.0142，而该路径一行未改）。在此噪声下逐格宣称收益是不诚实的；
交替复核（mod8/concat2 对照）显示无回归、double mod8 略好。保留依据是：构造上
严格更少的内存流量（在任何缓存更紧张的机器上必然变现）+ 完整正确性验证。
规范战绩 1M 51W/5L、8M 39W/3L（含与 base 相同的噪声顺风与逆风格子）。

**教训**。当环境噪声大于改动效应时，唯一诚实的度量是「构造论证 + 无回归证明」，
并把噪声区间写进文档，而不是挑一轮顺风数字当收益。

## 2026-09-11（三） — 并行反转 + 编码键密集计数

**被纠正的误判**。矩阵的 mod8（`i % 8`）不是「已排序」——每 8 个元素一个 7→0 下降沿，
是 n/8 次逆序的锯齿。用调度轨迹（test_dispatch_trace）确认它走计数是正确路由，
差点按错误前提"修"一通。教训：先追踪，后动手。

**并行反转**。旧注释说任务切分在数值逆序上回归过；用今天的结构（证明趟已并行、
交换趟独立分段）重测：int32 8M 逆序稳定改善、int64/double 持平。本机 2 核且带宽封顶，
 striped swap 无法展现 DRAM 级收益，但也不会更慢——核多的机器上继续放大。保留。

**编码键密集计数**。密集计数器拒绝 double 的唯一理由是 `is_integral` 门，
而 RadixTraits 编码键本来就是保序整数。放宽后 double 小窗口列与整数共用
u32 计数器。全序（NaN/-0）由编码天然保持，窗口外逃逸保证样本漏极端时正确回退。

**噪声期的文档纪律**。lowcard16 三格与 int32 allequal 本轮摆进败格（0.89-0.97x），
上一轮它们在 0.96-1.05x。总数 51/5→49/7 是噪声，不是回归；结构性目标
（double 逆序、mod8 家族）在交错配对中全部改善。结论只按「交错配对 + 输格清单」写。

## 2026-09-11（四） — ISA 分发验证 + SIMD 窗口计数的否证与回退

**验证通过：运行时 ISA 分发**。`FYX_HAS_AVX512_CODE` 在 GCC/Clang 恒为 1，全部
SIMD 内核住在 `FYX_ISA_BEGIN` 目标区域里、由 `use_avx512()` 运行时门控。实测基线
编译（无 -march）与 native 编译逐项持平（1M：int32 随机 0.00253/0.00251、int64
已排序 0.000203/0.000206）——「要求用户会加编译参数」的通用性缺口**不存在**，
README 的承诺得到实测背书。MSVC（无 per-function target）仍是文档里写明的例外。

**否证并回退：AVX-512 one-hot 窗口计数**。假设：mod8 家族的瓶颈是标量计数链的
load/inc/store 依赖（~2.5-3 cyc/elem），one-hot（每窗口值一次 cmpeq 掩码 + popcount）
能把 ALU 降 ~5 倍。实测（同树单变量，FYX_NO_SIMD_WINDOW_COUNT 切换）：int32/int64
mod8 全部**慢约 2 倍**（1M int64：0.00147 vs 0.00067；8M int64：0.019 vs 0.0096）。
原因：小窗口的计数表只有 1-4 条 L1 缓存行，标量自增的依赖链被 mod 循环模式天然
错开（连续自增打在不同计数器上，存储转发把延迟摊掉）；而 one-hot 的操作数按窗口
宽度倍增（int64 每 8 元素要 8×(cmpeq+popcnt)，每元素 2 个 SIMD 操作 > 1 个标量自增）。
「向量化直方图」对**宽窗口**（256 槽 radix 直方图，parts/09 已有 vpconflict 版本）
成立，对**窄窗口密集计数**不成立——方向反了。已整体删除（内核、接线、宏），
标量 u32 计数 + 小表保留。若再攻此格子，唯一剩下的方向是 VPCONFLICTD 直方图
（本机已有现成区域可复用），预期收益 ≤30%，留作记录不再追逐。

**教训**。这次的流程是对的：先写内核（正确性探针含 SIMD 逃逸、FP 全序、奇数 n
全过）→ 单变量 A/B（kill-switch 宏，同一棵树）→ 立即否决回退。假设听起来再合理，
数字不支持就是不支持；回退要删干净，不给单头文件库留死代码。

## 2026-09-11（五） — int64 随机路由翻转：一台机器的测量记录绑架了默认内核

**背景**。int64 随机是连续多轮全量跑里唯一的大格稳定输家（0.44x–0.75x vs vqsort），
而 int32/double 随机对 vqsort 是 1.4x 的赢格。追踪调度决策（test_dispatch_trace）：
int32 random → VectorQuick，int64 random → Radix（high-prefix）。分歧在
`vqsort_preferred<int64>() == false`，其注释引用的依据是旧机器一次 8M 串行对决：
radix 0.089 vs vsort 0.096，7%。

**本机复赛**（同树 A/B，只切 vqsort_preferred 的 int64/uint64 位）：

  形状         规模   radix 路由            vsort 路由
  full64       1M     0.0140-0.0301        0.0061-0.0094   (2-3x)
  40-bit       1M     0.0204-0.0258        0.0092-0.0093   (2.5x)
  20-bit 重tie 1M     0.0169-0.0188        0.0095-0.0097   (2x)
  full64       8M     0.104-0.112          0.074-0.107
  40-bit       8M     0.712-0.782          0.079-0.107     (9x, 病态)
  20-bit 重tie 8M     0.172-0.182          0.091-0.110     (2x)

40-bit 8M 的 0.73 s（91 ns/elem）暴露了 high-prefix 内核的病态路径：top26 前缀
不近似唯一时，tie 修复趟吃掉一切。旧机器上 radix 赢的 7% 和本机 vsort 赢的
2-9 倍放在一起，结论不是「本机更快所以换」，而是**默认内核的最坏情况对比**：
一边最坏 -7%，另一边最坏 -90%。翻转向量快排对所有主机成立。

**结果**。规范矩阵：1M int64 随机 0.44-0.55x → **1.34x**，8M 0.75x → **1.57x**
（对 vqsort 本尊）；**8M 41 胜 1 负**，除 int32 已排序（坏窗口 0.87x，相邻轮 1.2x+）
外对全部六个对手全胜。int32 逆序/double 逆序等 1M 边缘格被重排的负载顺带改善。

**方法论**。这次的教训与上轮 SIMD 窗口计数相反：那次的否证靠单变量 kill-switch
A/B，这次的翻案靠追踪调度决策 + 复测被引用的旧测量。两件事是同一条纪律：
**路由表里的每个数字都是测量，测量会过期，过期就要复赛。**

## 2026-09-12 — 「纯比较模式」实测：包装比较器即库的纯比较栈

**问题**。用户问：纯比较排序以及标准模式是否碾压 IPS4o。标准模式的答案在规范矩阵里
（见 BENCHMARKS.md）；「纯比较排序」需要一个忠实的测法。

**测法**。库只认 `std::less`/`std::greater` 为原生序（`is_ascending_v`）；
任何包装比较器都会让 `radix_order` 为假，从而关闭全部键结构武器——基数、计数、
向量快排都不可达——只剩：单调扫描（比较）、并行样本排序（比较）、pdq 回退（比较）、
插入/网络（比较）。用 `Wrap{a<b}` 与 IPS4o（串行/并行，同一个比较器）对决，
就是「比较操作对比较操作」的干净测量，零库改动。

**结果**（best-of-7，本机空载，包装比较器）：

  形状          规模   纯比较 fyx     IPS4o 串行        IPS4o 并行
  i64 随机      1M     0.00505        0.0238 (4.7x)     0.0136 (2.7x)
  i32 随机      1M     0.00475        0.0242 (5.1x)     0.0130 (2.7x)
  dbl 随机      1M     0.00830        0.0277 (3.3x)     0.0156 (1.9x)
  i64 已排序    1M     0.000288       0.000449 (1.6x)   0.000386 (1.3x)
  i64 逆序      1M     0.000450       0.000843 (1.9x)   0.0100 (22x)
  i64 随机      8M     0.0627         0.2281 (3.6x)     0.1120 (1.8x)
  i32 随机      8M     0.0447         0.1878 (4.2x)     0.1056 (2.4x)
  dbl 随机      8M     0.0706         0.2487 (3.5x)     0.1266 (1.8x)
  i64 已排序    8M     0.00304        0.00583 (1.9x)    0.00337 (1.1x)
  i64 逆序      8M     0.00522        0.00978 (1.9x)    0.0898 (17x)

**结论**。纯比较栈在 IPS4o 的主场（随机数据）赢 1.8–5.1x（两个规模、串行并行通吃），
已排序 1.1–1.9x，逆序对串行 1.9x、对并行 17–22x（IPS4o 并行在逆序输入上有严重
病态，与其分区策略在预排序输入上的行为一致）。附带观察：包装比较器下 1M int64
随机 0.00505 比默认模式的同格数字还快——并行样本排序在与套件无争用时的真实实力，
但这是另一轮的单变量问题，此处只记录不下结论。

## 2026-09-12（二） — 更正：昨日纯比较数字含测量伪差；样本排序线索否证

**伪差的发现与复现**。昨日 NOTES 记录的纯比较 fyx 1M int64 随机 0.00505 s 无法复现：
用**完全相同的 harness**（同 rng、同 best-of-7、同包装比较器、独立进程）重测得
0.0141–0.0147；第三个同进程交错 harness（轮转变序、best-of-3×5 轮取中位）得 0.0152。
三个独立测量互相一致 → 昨日的 0.00505 是一次未解释的瞬时突发（主机短时睿态嫌疑），
不是该路径的稳定速度。**同格重测三次取一致值是纪律；单轮最佳值不是。**

**更正后的纯比较战绩**（今日，包装比较器，best-of-7，同进程三方对决）：

  形状        规模   纯比较 fyx   vs IPS4o 串行      vs IPS4o 并行
  i64 随机    1M     0.0165       1.58x              1.09x（平手）
  i32 随机    1M     0.0068       3.59x              2.15x
  dbl 随机    1M     0.0179       1.60x              **0.84x（输）**
  i64 已排序  1M     0.000260     1.42x              1.28x
  i64 逆序    1M     0.000483     1.43x              23.8x
  i64 随机    8M     0.0702       3.24x              1.64x
  i32 随机    8M     0.0448       4.67x              2.38x
  dbl 随机    8M     0.0797       3.04x              1.55x
  i64 已排序  8M     0.00379      1.78x              1.04x
  i64 逆序    8M     0.00945      1.25x              10.5x

**更正后的结论**：纯比较模式对 IPS4o **串行**全形状全胜（1.25–4.67x）——「碾压」成立；
对 IPS4o **并行**：8M 全胜（1.04–2.38x）、逆序大胜（10–24x，其逆序病态是真的），
但 1M 只有微弱优势，且 **1M double 随机是输格（0.84x）**。昨日的「1.8–5.1x」撤回。

**样本排序线索否证**。昨日附注「纯比较 1M 比 vsort 路由还快」同样源于那个伪差
（真正的 0.00505 疑似是并行样本排序的突发）。控制后：包装比较器栈 0.0143–0.0152、
直接调 parallel_sample_sort 0.0192、默认 vsort 路由 0.0057——**vsort 保持，
1M/8M 都是，无路由带可做**。int32 上样本排序更慢（2–7 倍），与既有认知一致。

## 2026-09-12（三） — 1M double 逆序：分段交换门槛实验否证，格子定性为冷上下文边界

**假设**。1M double 逆序（规范全量 0.72–0.82x vs IPS4o 串行）疑似 1M 规模的
striped swap 净亏（fork 开销 ~50-100µs 加在 0.25ms 的交换上）。把交换门槛
3MB→16MB（1M double 8MB 回到串行 reverse）。

**结果**。矩阵内三轮交错（filter=reverse，确定性数据）：全类型噪声内持平，
double 若有也是 +3%（更差）。假设证伪，按规则回退。

**顺带的诊断发现**（写下来防止未来重复挖这个格子）：
1. 融合的「验证+交换」单趟（try_fast_reverse_exit 的主体）比「验证一趟 +
   普通 std::reverse」**慢一倍**（1.04ms vs 0.55ms）——循环内邻居比较把读
   流量×2，这解释了为什么库选择两趟；
2. 全调用在 L3 温态下只要 **0.55ms**（8MB 数组，54MB L3 装得下），规范矩阵的
   1.14ms 是 14 种分布轮转后的冷态；IPS4o 同样 24MB 流量（验证 8MB 读 +
   交换 16MB），冷态差距是上下文效应，不是工作量差距；
3. 结论：**该格是冷上下文里的诚实边界**，交换策略无可挖——除非谁能把 24MB
   流量本身减掉（双方都不行，验证是正确性义务）。

## 2026-09-13（四） — 多轮中位数协议落地：1M 44-12 → 51-5，半数"输格"是单轮噪声

**动机**。上轮 1M 单轮矩阵 44 胜 12 负，输格清单可疑（lowcard16、全等、逆序这些
我们结构上该赢的形状也在输），与更早轮次的 53-3、56-0 记录漂移严重。结合
（二）的教训（单进程单轮 best 可被主机瞬态污染 3 倍），结论是**单轮矩阵对
sub-ms 格子不具备裁决力**。

**机制**。`tools/dev/merge_runs.py`：k 份独立 `bench_matrix` 输出（独立进程、
逐次 `uptime` 确认空载）逐格取时间中位数（component-wise，8 列各自取），再由
中位时间重算 best-other 与两列比值，保持原格式。协议：3 轮 × (--reps=7 @1M /
--reps=5 @8M)，跑前查负载。

**结果**（写进 README/BENCHMARKS.md，commit 见 git log）：

- **1M：44 胜 12 负 → 51 胜 5 负**。中位后转赢的 7 格：lowcard16 × 3、
  全等 × 2、int32/int64 逆序——全部是单轮噪声伪输。残余 5 格：
  mod8 三格 0.61–0.80x（xss 小值域专精，结构性的，下一步候选）、
  double sorted 0.90x（ips4opar）、int64 farswap 0.99x（平手级）。
- **8M：41 胜 1 负不变**（int32 sorted 0.88x ips4opar；单轮里 sorted 三格
  0.86–0.88 的"坏窗口"中位后恢复 1.0x+，唯 int32 仍差 12%）。
- 分对手 1M：std 56/0、ips4o **56/0**（原 53/3——逆序格是噪声）、
  ips4opar 55/1、pdqsort 55/1、vqsort/xss 39/3（原 36/6）。

**教训固化**：任何进文档的矩阵数字必须 ≥3 独立进程中位；比值在 0.95–1.05
的一律视为平手，不据此做路由决策。此协议对一切机器成立（它修的是测量，
不是这台机器）。

## 2026-09-13（五） — 纯比较 1M double random "0.84x 输格"否证：平手级

**复核**。（二）的教训同样适用于 7ce539b 的纯比较表：0.84x 来自单次会话。
新探针 `/tmp/purecmp4.cpp`：三方（fyx 包装比较器 / ips4o 串 / ips4o 并）**轮转交错**，
每方 3 轮 × best-of-3，方向每轮轮换；3 个独立进程（seed 23/77/991），编译需
`-pthread`（`_REENTRANT` 才暴露 parallel.hpp）+ `-latomic`（16 字节原子）。

**dblrand 1M fyx/ipsp 比值**：0.979 / 1.048 / 1.057 → **中位 1.048，平手级
（0.98–1.06）**。i64rand 1M 对照：0.990 / 0.959 / 0.967（微弱优势，与已发布
1.09x 一致方向）。dblrand 8M 对照：0.619–0.626（赢 1.6x，与已发布 1.55x 一致）。

**结论**：0.84x 三次独立复测均未出现，撤回。纯比较模式对 IPS4o 并行在 1M 的
诚实表述是「平手（随机）/微弱优势（整数随机）/逆序大胜」，不存在已知输格。
README 已同步更正。至此两个目标（①double 输格诊断 ②多轮中位机制）均闭环：
②顺带把 ①也解决了——同一病灶（单会话噪声）。

**下一步候选**（标准模式残余真输格，按差距排序）：
1. mod8 家族 1M（0.61–0.80x vs xss，三格同根）——xss 的小值域专精路径；
2. double sorted 1M 0.90x vs ips4opar（池辅助证明的固定开销）；
3. int64 farswap 0.99x（平手级，可不管）。

## 2026-09-13（六） — mod8 家族闭合 + 池竞态修复：1M 51-5 → 53-3，残负全在平手带

任务来自用户指令「全部完成」：清掉（四）留下的全部残余输格（mod8 三格 0.61–0.80x、
double sorted 0.90x、int64 farswap 0.99x）。

**探针旅程（先错后对）**。直调诊断的三次修正值得记录：
1. 直调 `try_integer_range_count_sort_parallel` 得到 0.56ms 的“漂亮数字”——验证发现
   返回 0 且数组未排序：拒绝路径不干活，耗时是假的。**直调内部核必须验证返回值**。
2. 计时含每轮 8MB 拷贝 → 拷贝挪出计时区后核本体只有 0.7–1.2ms。
3. L1 哈希表替换 L2 LUT **毫无加速**——瓶颈从来不是查表，是每元素
   “乘→移位→加载→依赖验证加载→计数”约 12–16 周期的链。

**块级漂移（本轮最大测量教训，比伪差更隐蔽）**。同一 HEAD 二进制：
canonical 块里 i64 mod8 = 1.57ms（0.61x 输 xss），两小时后的交错块里同一二进制
同一格 = 0.84ms（1.46x 赢）。sub-ms 格子跨块摆动可达 ±2x。结论：**比较必须同块交错、
新旧双二进制、3 进程中位**；跨块的“改进/退化”一律存疑。mod8 的“结构性输格”大半是
块漂移伪装的，真正的结构输格只有 i32 mod8（所有上下文一致输 0.5–0.8x）。

**改动清单**（`parts/13_api.hpp` + `parts/11_parallel.hpp`，全部同块 A/B 3×中位验证）：
1. `small_rank_count_fill_parallel`：d≤255 的 rank 计数核（8 位注入哈希，L1 表）；
2. AVX-512 比较式计数趟（d≤16，64/32 位键两变体）：每键每 8/16 元素块一次
   vpcmpeq + 一次掩码加，无依赖加载；未采样键 = 总和短缺 → 拒绝（正确性由
   `sum(counts)==n` 保障，不需逐元素验证）；运行时 `use_avx512()` 门控，无 AVX-512
   回落标量哈希路径；
3. `try_integer_range_count_sort_parallel` 路由：dhat≤16（有 AVX-512）或
   span>512 且 span>4d 时先试 rank 核——把 O(range×chunks) 的槽清零/求和/游走
   （mod8×4096 形状每次 1.8MB 计数器）换成 O(n+d)；拒绝后回落 dense，语义不变；
4. `parallel_fill_by_ranks`：按输出区间切分的并行 fill（按 rank 切分在 d=8、
   计数倾斜时退化为单核写 8MB）；dense_prefix/rank16/small_rank 采用；
   **dense 区间计数器保留按 rank 粒度**——输出区间切分在 i32 mod8 上实测慢 16%
   （二分查找簿记不划算），这一条是同块 A/B 抓出来后回退的；
5. **池竞态修复**（11_parallel.hpp）：`this_worker()` 原来对所有非池线程返回 0，
   N 个应用线程并发排序时全部假冒 worker 0 推/弹同一条 Chase-Lev deque——
   单所有者协议破坏，丢任务 → `wait_for` 永转（套件 8 线程压测 2/20 挂死，
   历史上从未观察到是因为时序未到；本头文件改动把窗口晃出来了）。
   修复：第一个外部线程认领 worker 0（粘性），其余外部线程走互斥 FIFO
   （`submit_foreign`/`foreign_pop`），空闲 worker 睡前抽干、外来等待者抽干；
   `try_get_task` 的热自旋路径**不**检查 FIFO（同块 A/B 显示无条件检查让
   reverse 家族慢 1.5–2.2x，门控后恢复并反超）。

**同块 A/B 终局**（fab2，old=HEAD vs new，各 3 进程中位，1M 全 56 格）：
8 优 / 0 劣。i32 mod8 0.53x（0.80x 输 → 1.50x 赢）、i32 sorted 0.56x、
i32 nearlysorted 0.64x、i64 sorted 0.65x、i64 reverse 0.80x、i32 reverse 0.82x、
i32 blockswap 0.84x、i64 lowcard256 0.84x。

**canonical 更新**（3 进程中位，与旧表同协议）：
- 1M：**53 胜 / 3 负**（残负：double sorted 0.95x、i64 lowcard16 0.98x、
  i32 全等 0.97x——全部 ≥0.95 平手带）；分对手：std 56/0、ips4o 56/0、
  ips4opar 55/1 [0.95]、pdqsort **56/0**、vqsort 41/1、xss 41/1。
- 8M：**41 胜 / 1 负**（唯余 double sorted 0.95x 平手带；int32 sorted 0.88x 消失）。

**farswap 0.99x**：维持平手判定，不动。**double sorted 1M/8M 0.95x**：池辅助证明的
固定开销，已从 0.90x 压到 0.95x 平手带，不追。

验证：套件 743/0；20 个对抗用例（隐藏键/同前缀隐藏键/极端倾斜/降序/±0/NaN 混合/
d 边界 16/17/255/256/300/负值/无符号）；8 线程压测 30/30 零挂死；TSan 无警告；
ASan+UBSan 套件通过；`-Werror` 干净。

## 2026-09-13/14（七） — 大轮收官：1M 56-0 / 8M 42-0 全胜，六对手无一格低于 1.00x

任务：用户令「开启一大轮，直到全场景完全碾压」。起点：1M 53-3 / 8M 41-1
（残负全在 0.95 平手带）。本轮平台重置一次（第 8 次，工作树幸存、提交历史重置，
恢复提交 4e394a3 重新落地全部内容）。

**改动一：向量化有序性证明**（radix_key_monotone_scan，两路径共用）。
琐碎输入的排序就是一次扫描，浮点每元素要重推编码键，标量 ~6 ops/elem。
AVX-512：每 8/16 元素一次 load + 3 条编码指令（srai/or/xor）+ alignr 移位自比较
+ 掩码测试，违例早退；位精确键保持 NaN/-0 总序；标量尾部/回落与探测器等价。
关键教训：**先确认目标路径**——第一次只改了 <2M 的池辅助路径，8M dbl sorted
纹丝不动才发现在 ≥2M 走的是 try_parallel_fast_order_exit（另一份相同标量循环）。
同块 A/B：1M 3 优 0 劣（dbl sorted 1.57→1.93x）；8M 2 优 0 劣
（dbl sorted 0.94x 输 → 1.30x 赢；dbl reverse 1.78→2.17x）。

**改动二：向量化全等扫掠**（range_all_equal_first_vec，两路径共用）。
全等判定 = "每元素与 p[0] 比较"，无需三向分类。每 64B 一次 cmpeq + 全掩码检查；
浮点比编码键（-0/+0 仍区分）；异议元素回落原分类路径。
实现教训：**全掩码位数必须按通道宽算**——第一版拿 64 位全 1 比较 16 位掩码，
扫掠永远误报不等而静默回落（时间毫无变化暴露了它）；dispatch 轨迹
（test_dispatch_trace）确认 assist 确实在跑后才去查掩码逻辑。
同块 A/B：1M i64 全等 0.95x→3.17x（247→77µs）、dbl 1.07x→3.10x、i32 0.91x→1.22x；
8M i64/dbl 全等 1.3x→3.8x。

**插曲**：中间一次 python 锚点替换吃掉了函数签名造成结构性损坏（build.sh 只拼装
不编译，坏头文件混过了两次"构建"），git checkout HEAD 恢复后改为"插入不动锚点"的
方式重做。教训：**每次 parts 编辑后立即编译探针验证，不能只看 build.sh 输出**。

**最终 canonical**（3 进程中位，bench_matrix 协议不变）：
- 1M：**56 胜 / 0 负**。分对手：std 56/0、ips4o 56/0、ips4opar 56/0、pdqsort 56/0、
  vqsort 42/0、xss 42/0。最弱格：i64 farswap 1.01x。
- 8M：**42 胜 / 0 负**。最弱格：i64 sorted 1.05x。
两个规模、四种类型、十四种分布，对六个对手无任何一格低于 1.00x。

**验证**：套件 743/0；ASan+UBSan 743/0；8 线程压测 5/5；证明等价性 30 试验
（零汤/NaN 胡椒/位垃圾/隐藏尖峰）全过；-Werror 干净。

## 2026-09-16（八） — 前沿 1：串行随机列 lean+max 分区，13–20% → 10–16%

任务：README「剩余的优化前沿」第 1 条——串行随机算术列对 hwy vqsort 本尊 0.80–0.90x。
工具：串行探针（`test_last_dispatch()` 轨迹 + fyx serial `Tri::Off` + hwy/xss 对手，
7 轮 best-of；链接 `-Lthird_party/highway/build -lhwy_contrib -lhwy` 库在源后）。
**dispatch 轨迹证实三种类型（含 int32）的串行随机 1M 全部路由 VectorQuick**——
README 早先「int32 串行走 wide-key radix」被证伪，已撤回订正。另做了 dispatch map
（复刻 bench_matrix 的 make_data）：1M 串行只有 random 三格走 vqsort，其余 13 种
分布全部分流（sorted=3/reverse=4/低基=5/结构化=6 PartialPdq/全等=2）。

**实验簿**（全部同块交错 A/B 中位；负结果照录防重试）：
1. 主循环软件预取 `prefetch_read<2>` + int32 采样 4×V：中性偏负（int32 掉点）——弃。
2. 纯 lean 分区（热循环去 min/max，`split==0` 检测退化 + strict 重分）：
   i64/dbl +4~5 点，但 int32 低于旧版（旧版免费早退被结构检测替代后重付整趟）。
3. 4 字节叶子 8 向量（128）：int32 更差（0.75–0.80x）——弃。
4. 分区 unroll=8：int32 8M 绝对时间 +7%，double 略差——弃。
5. 每槽 median-of-3 枢轴采样（48 裸采样）：int32 崩至 0.60x（192 次标量加载），
   i64/dbl 中性——弃。
6. **lean+max（落地版）**：热循环只跟 vmax（每向量 1 条向量指令；旧 min+max 是 2 条；
   最小值与 split==0 冗余），保留旧版两个免费早退（`!(pivot<hi)`：整段同值→返回；
   高侧全 pivot→丢弃不递归）。结果：i64 1M 0.82→0.85x、8M 0.84→0.86x；
   dbl 1M 0.87→0.89x、8M 0.85→0.86x；i32 持平偏正。

**本轮最大 bug（值代表作答）**：lean+max 首版把掩码尾向量直接喂给带 max 跟踪的
分区步（未掩码 gek），垃圾通道被计成右侧 → gc 多算 V-m → 空隙簿记崩坏 → 右侧
compress-store 越过左界、最终写出数组头（`free(): invalid pointer`）+ 乱序输出。
-O3 下堆损坏、-O1+ASan 下只见 UNSORTED（布局相关）。修复 = 数据移动仍走
`vpart_vec_lean_masked`，max 更新单独对 `blend(valid, cur, pv)` 的安全副本做。
教训三条：(a) 部分向量的垃圾通道不得进入任何计数或存储；(b) 探针的排序校验
（rc=1）先于 ASan 抓到问题，两个都要跑；(c) **用坏头编译的 A/B 二进制测出的
"回归/增益" 全部作废**——本轮第一次 bench A/B 就是这样测出 i64 random +7% 的
假回归，换修复版后真相是 -0.4%。

**测量环境教训**：本 boot 载入 1 小时内双态抖动（并行格 5.3ms↔8.6ms 两态，
未触碰的 farswap 格也 ±10%）+ 开局后台 `vqsort.sh` 与 README 会话重叠污染了
`build/vqsort_*.txt`（hwy i32 列 27ms = 8 倍偏离；i32 random par 4.18ms vs
canonical 2.26ms）。**canonical 记分牌文件待净窗口前台重跑**（BENCHMARKS.md 的
56-0/42-0 仍是上次净窗口的记录）。

**验证**：27 配置复现程序（3 类型 × 均匀/重复合/已排序 × 1K/64K/1M）；套件 743/0；
ASan+UBSan 743/0；-Werror 干净；并行 random 5 轮抽查 B 全面 ≥ A（dbl par +3.9%
在噪声带内）。剩余差距候选：hwy BaseCase 排序网络、PartitionRightmost 余数预处理、
GatherSample 枢轴（随机块 × MedianOf3）。

**净窗口 canonical 复测（lean+max 已落地后）**：3 独立进程 × 中位（merge_runs 协议）。
1M：56/0，最弱三格 int32 全等 1.01x（vqsort）、int64 farswap 1.02x（pdqsort）、
string sorted 1.03x（ips4opar）；8M：42/0，最弱 int64 sorted 1.07x、int32 sorted
1.08x、double farswap 1.13x。i32 random 1M par 2.32ms 与上一净窗口 2.26ms 一致
——被污染轮的 4.18ms 确认为测量事故。BENCHMARKS.md 与 README 记分牌摘要已按新
canonical 重生成/对齐。

## 2026-09-17（九） — 前沿 1 判定到顶；新理论线开工：单断点结构证明（rotate）

**前沿 1 收官判定**：本轮把剩余内核候选全部测尽——向量化 median-of-3 采样 + hwy
PivotRank 重复值调整（整数 +1.6~7.1% 负、double 中性）、noinline 尾处理抽取
（混合噪声；删除实验的 5–9% 是"少干活"假象，尾块真实成本 ≤2%）。七个杠杆累计
只捞回 min/max 消除的 +2~3 点。判定：0.82–0.87x 就是该算法族在本机的常数调优极限，
剩余差距分散在整个内核（codegen/调度/hwy 的叶子网络微调），属工程碾压非算法空间。

**新理论：单断点结构证明（try_one_break_rotate）**。「有序性证明」家族从 k=0 推广：
前缀与后缀各自单调 + 端点回绕 ⟺ 旋转有序数组，rotate 回来 O(n) 且稳定。扫描在第二
个违例处放弃（随机输入头几个元素就攒够两个断点，近零税）。挂点：sort_pointer_core_impl
的 try_order_exit_adaptive 之后 + sort_st 的 profile 快速出口之后；dispatch 记 ProfileSorted。

**实现教训**：断点后续扫必须先 `prev = encode(p[brk])` 再从 brk+1 起步——初版从
brk+1 起步还拿着 p[brk-1]（旋转数组的全数组最大值），第一个后缀元素永远像"第二断点"，
证明永远误否。单元对照（内联复刻 vs 头文件调用同数组一真一假）10 分钟定位。

**实测**（1M 串行，同块 A/B）：rotated i64 3.73→1.69ms（2.2x）、i32 3.90→1.04ms（3.8x）、
dbl 5.32→1.66ms（3.2x）；random 税 +2.5%（单轮，待复验）。验证：27 配置复现、
套件 743/0、ASan+UBSan 743/0、-Werror 干净。**下一步**：并行路（chunk 缝隙代数：
全 chunk 单调 + 恰一条坏缝 + 端点回绕 → 并行 rotate）、zigzag/concat2 的归并修复、
串行扫描向量化。平台第 10 次重置恢复 = ad1208a（备份链 +tar +backup/post-restore10）。

## 2026-09-17（十） — 单断点证明并行化：缝隙代数 + 双缓冲搬移

接（九）。并行版 = chunk 扫描（复用向量化单调扫描，seam 检查融入 chunk 首对）+
缝隙代数：恰一条坏缝（brk=缝位）或恰一个坏 chunk（radix_key_find_break 向量化
精化 + 后半段二次验证），回绕检查后修复。挂 sort_pointer_core_impl（rev_parallel_ok
门内、串行版之前）；dispatch 复用 ProfileSorted。

**本轮大坑：并行分块 memmove 的跨块竞态**。首版用「短侧 buf + 分块 memmove 长侧」，
1M/8M 的 rotated 11/36 错例、perm=0（不是排列=数据损坏）。机制：块长 g、偏移 s 时，
块 k 的写窗 [s+kg, s+(k+1)g) 与块 k+⌈s/g⌉ 的读窗重叠（s 非 g 整除时必然），
而 parallel_for_index 的叶子并发执行——源在读之前被别的块的写覆盖。串行顺序下
「dst 恒在 src 左侧」的安全论证在并发下不成立。修复 = 全量双缓冲（copy_out 全数组 +
每块两段直 memcpy 回，源索引 i+brk 每块至多回绕一次），4n 触达换无条件安全。
36 配置 × 5 重跑全过。

**触达账**（2 核 ~12GB/s 决定一切）：双缓冲 4n 触达 = 1M ~0.65ms、8M ~10.7ms。
结论：**1M/中型赢（vqsort 并行的调度固定开销占比大：dbl 1.52→0.93ms），
8M 平手（proof 14.7ms vs vqsort 并行 15.9ms）**——rotate 的带宽本质没有暴利，
真正的赢面在调度开销敏感的中小规模。串行 random 税复验 +0.2~0.9%（零税确认，
（九）的 +2.5% 是单轮噪声）。string 不走 proof（radix_ok=false，A/B -0.7% 噪声）。

**验证**：36 配置（4 类型 × 3 规模 × 多断点位 × 升降序 × 随机底）× 5 重跑；
套件 743/0；ASan+UBSan 743/0；-Werror 干净；random/farswap 抽查无回退。
第 11 次平台重置恢复 = e91654f（backup/post-restore11）。

## 2026-09-17（十一） — 新算法家族：结构证明排序（通用 k）

任务：用户明确「创造一种与向量化排序、冒泡、选择、快排、归并并列的全新算法，
有正提升且无任何负提升」。落地 **结构证明排序**（第三家族：基本操作 = 证明-重构，
区别于比较-交换与分发-收集）。

**算法定义**：一次封顶扫描（kProofStructMaxBreaks=7）收集目标方向的违例位，证明
输入 = r 个目标方向单调段的拼接；重构零元素比较：r=2 回绕 → rotate；r=2 不回绕 →
一次稳定归并；3≤r≤8 → 段界二叉归并树（⌈log₂r⌉ 趟，逐对归并+奇数段直拷）。
稳定性：rotate 由 stable_wrap 严格回绕门控（同 k=1）；所有归并左优先 → 稳定。
门控：`!dynamic_parallel_allowed`（调用方语境串行才触发）——PSS 只在并行引擎
不在场时工作，结构性保证「无负提升」。

**本坑两次**：(1) 内置并行自门（parallel_available && n≥1M → decline）不知道调用方
Tri::Off，串行 bench 里永远不触发，单测直接调用也 false——门必须用调用方语境
（dynamic_parallel_allowed），移到挂点；(2) 探针构造 bug：全排序切段 = 0 违例
（全局有序的拼接还是全局有序），r 段必须独立排序 + 值域重叠才能造出 r-1 个违例。

**实测**（串行 1M int64，3 轮同块交错中位）：runs=2 **-27.5%**、runs=3 **-18.0%**、
runs=8 **-17.2%**、concat2 -4.5%、random -1.4%（噪声带，零税）。r=5 被 try_bitonic
先截（速度相当，无回退）。验证：r=2..9 × 升降序 × 等值重型 × 全 uint64 值域探针
全过；套件 743/0；ASan+UBSan 743/0；-Werror 干净。第 12 次平台重置恢复 = d1f6766
（backup/post-restore12）。**下一步**：① 并行化（段界可 chunk 并行发现 + 归并树
天然并行）；② 吸收 zigzag/concat2/bitonic 专路（统一成一个证明）；③ 向量化扫描
（违例位收集 Currently 标量）。

## 2026-09-17（十二） — PSS 证明扫描向量化；揪出生产级整数键变换 bug

任务：① 用户要求详述新算法逻辑后再继续（正文见会话汇报：分解定理/旋转定理/归并树
不变式/稳定性/与 Timsort 四点本质区别/零负提升三道闸）；② 向量化封顶扫描。

**向量化封顶扫描**：与单调扫描同款（每 8/16 元素一次 load + encode + alignr 移位
自比较），违例掩码逐位抽取、封顶即中途退出。交错 A/B（NEW vs NOVEC 隔离变体，
同头仅开关扫描分支）：r=2 -0.2ms（噪声级）、r=3 **-1.2ms（-13%）**、r=8
**-2.5ms（-14%）**——收益远超扫描自身成本。机制假说（待深究）：标量 encode
链每迭代 ~4-5 cycle 串行依赖（1M ≈ 2.5ms，不是直觉的 0.3ms），且污染后续
merge 趟的前端/预取状态。差异随 r 增大（0.2→1.2→2.5ms）的模式未完全定位。

**生产级 bug：整数键向量变换用错**（v10.3 引入，潜伏至今）。radix_key_monotone_scan
的向量分支对所有键类型用 IEEE 浮点变换（srai 符号传播全翻转）；对有符号整数该
变换**在负数域内逆序**（k=-10 → encode 9 > k=-5 → encode 4），对高位 uint64 同样
逆序。实测后果：负 int64/高位 uint64 的单调数组被向量证明**误拒**（静默回落慢
路径，纯性能损失）；构造分析表明误收需要门控与缝检合作、现分发下不可达
（bughunt 形状全对）。修复 = 三处统一按类型分支：无符号=恒等、有符号=异或符号位、
浮点=保留 srai 变换（radix_key_monotone_scan / radix_key_find_break / PSS 扫描）。
**教训：向量化键变换必须从 RadixTraits<T> 按类型推导，严禁跨类型类目复制粘贴。**
直接单测入探针集（negative/high-region violation detected + sorted accepted）。

**off-by-one**：向量化 drain 循环"先写后查 cap"，第 7 个违例即误 cap（标量版第 8 个
才拒）→ r=8 形状被误拒。修复 = 先查后写。

**陈旧二进制又坑一次**：变换修复后 pss2 二进制未重编，"失败"十分钟后才用独立
复刻（pssdbg）对照定位到二进制过期。**规则重申：任何头文件改动后，全部探针必须重编。**

**验证**：pss2 36 配置（含 r=8）、bughunt（uint64 高位/负 int64 @2M/8M 串+并）、
direct 变换三语义、套件 743/0、ASan+UBSan 743/0、-Werror 干净。
第 13 次平台重置恢复 = c0dda15（backup/post-restore13）。

## 2026-09-19（十三） — 命名纠偏：PSS 不是全新算法

用户老师的判定，全文接受：

> 严格说不算「全新的算法」。更准确地说，它是自然归并排序 / Timsort 家族的一个
> 受限制特化版 + 工程优化组合。

对照表（老师原文，入册防再包装过头）：

| 内部叫法 | 实际对应 |
|---|---|
| 违例位 | run 边界 |
| 分解定理 | 自然归并的基本观察 |
| 证明阶段扫描 | Timsort / natural merge 的 run detection |
| 最多 7 个断点第 8 个放弃 | 有界 run 检测 + early abort |
| 3~8 段二叉归并树 | 标准 bottom-up merge |
| 两段回绕 rotate | 循环移位 / rotation |
| 稳定归并取左 | 标准稳定归并 |
| 太乱交给旧引擎 | 自适应 fallback 门 |

**可保留的工程新意（不是算法范式新意）**：封顶拒绝做成调度前门；r≤8 固定归并树
（实现简化）；严格回绕门 + rotate 接进 vqsort；实测零税。

学术上要叫「全新算法」需要新机制 / 新复杂度 / 新下界 / 新可处理结构 / 某分布上的
本质优势。PSS 目前没有这些。「O(1) 拒绝」只在随机期望下成立；「零负提升」是工程
闸门 + 实测，不是数学保证。

本轮起文档一律写：**vqsort 的封顶自然归并自适应前端**。价值仍在：结构化形状正提升、
随机不拖后腿。第 14 次平台重置恢复 = b2b3ead（tar `/home/user/backups/fyx_sort_post_restore14.tar.gz`）。

## 2026-09-22（十四） — 并行封顶自然归并前门接线 + 正确性/同块计时

**口径**：不是新范式。并行版 = 分块向量化 run 边界扫描（cap 7）+ r=2 wrap 走既有双缓冲 rotate + 其余 fork_join 成对稳定归并（深度 ≤3）。串行 PSS 仅在 `!rev_parallel_ok` 时触发，避免并行路径上再付一遍串行扫描。

**挂点**：`sort_pointer_core_impl` 在 one-break 之后、`try_bounded_insertion_repair` 之前。并行 PSS 定义早于 `merge_runs_moving` / `radix_key_find_break` / `try_one_break_rotate_parallel`，必须前向声明（模板非依赖名在定义点查找）。C ABI 尾被截成 `ODY(...)` 会重复定义包装函数，已清。

**正确性**：`tools/dev/pss_par_probe.cpp`（4K/256K/1M × random/concat2/runs3/runs8/rotated/neg_runs3）全过；`python3 tools/fyx_test.py` **743/0**。zigzag 仍走 organ-pipe 专路，PSS 不假装吸收反向 run。

**同块交错中位**（1M int64，5 轮，`tools/dev/pss_par_bench.cpp`，**不是** canonical 矩阵）：

| 形状 | ser ms | par ms | par/ser |
|---|---:|---:|---:|
| random | 16.44 | 9.10 | 0.55（并行 vqsort，PSS 封顶拒绝） |
| concat2 | 6.98 | 3.99 | 0.57（k=2 目标方向两段归并） |
| runs3 | 9.29 | 8.89 | 0.96 |
| runs8 | 15.87 | 11.96 | 0.75 |
| rotated | 1.51 | 0.94 | 0.62（one-break 并行 rotate 先截） |
| zigzag | 0.59 | 0.60 | 1.02（专路，不吸收） |

备份：`/home/user/backups/fyx_sort_pss_par_wip.tar.gz`。**BENCHMARKS.md 未改**：净窗口 3 进程中位尚未跑。

## 2026-09-25（十五） — 超级迭代收口：正确性闭环；1M 对 std/pdq 抽测；六对手 canonical 未跑

**目标对照**：正提升、零负提升、通用；并行封顶自然归并接线；k=2 只吸收目标方向两段；zigzag/bitonic 不假装吸收；不称新范式。

**本轮验证**
- `test/t_pss.cpp`：int32/int64/double × 4K/256K/1M × ser/par × random/concat2/runs8/rotated/zigzag/greater_concat2 → **96/0**
- `t_api` **847/0**；`t_adaptive` **456/0**；`fyx_test.py` 此前 **743/0**
- `t_counting` / `t_vsort` 本窗口 120s 内未完成，**不当绿**

**1M 矩阵（单进程 --reps=3，仅 std + pdqsort，boot+7min，load~1.0）**
- 不是六对手 canonical，不是 3 进程中位。
- PSS 目标形状（rotated/concat2/zigzag）对 std/pdq **全胜**。
- 唯一 <1.00x：int64 farswap fyxpar 0.78x vs pdqsort（patch/vqsort 路径，与 PSS 无关）。单轮噪声可能，未 3 中位。

**明确未达成**：六对手净窗口 3 进程中位（本环境无 highway/ips4o/xss 构建产物；仅 clone 了 pdqsort）。BENCHMARKS.md 不得用此表覆盖。

备份：`/home/user/backups/fyx_sort_superiter.tar.gz`