# Changelog

## Unreleased — v10.1 candidate

Date: 2026-08-27

### Added
- `fyx::sort` accepts containers whose storage is not one block.  `std::deque`
  and `std::list` used to be refused (`list` would not even compile): a
  container with a `.data()` keeps taking the contiguous kernels, one with its
  own `sort()` (the node-based containers) keeps splicing -- faster than moving,
  and nothing a buffer could beat -- and anything else is moved into a buffer,
  sorted by the same kernels a vector gets, and moved back.  A segmented range
  also used to have `Options` silently dropped, so asking for parallel cost
  exactly what not asking cost.  1M int32 in a `std::deque`: 0.051s before,
  0.026s now, against 0.086s for `std::sort`.
- `tools/fyx_test.py` -- one portable file holding the whole test suite.  It
  writes the C++ out, finds the header and a compiler, builds, and runs:
  `python3 tools/fyx_test.py [--header P] [--compiler clang++] [--bench]
  [--rlimit-mb N]`.  743 checks: 17 shapes x 6 types serial and parallel, sizes
  from 0 to either side of every dispatch threshold, containers, comparators,
  and stress -- inconsistent and non-transitive comparators, throwing
  comparators and throwing moves (checked for lost or duplicated elements),
  reentrant sorts from inside a comparator, eight threads sorting at once,
  repeated sorts of the same input, move-only payloads, NaN / -0 / +0 /
  denormals, and a memory-cap probe that runs under `ulimit -v` to prove the
  degraded paths work when scratch, buffers and worker threads cannot be had.

- Low-cardinality counting paths:
  - dense integer range counting;
  - sparse integer counting for up to 256 encoded keys;
  - compressed comparator-equivalence counting for arbitrary payload objects;
  - `std::string` value counting for default ascending/descending order;
  - guarded trivial-prefix field counting for common `struct { int key; ... }` custom comparator shapes.
- Parallel specializations for the low-cardinality paths:
  - parallel sparse integer/radix-key counting, including default-order `float`/`double` low-cardinality keys;
  - parallel trivial-prefix field counting/scatter;
  - parallel top-level `std::string` MSD byte-radix sorting.
- `std::string` MSD byte-radix sort for default string order, including descending support via reverse.
- Parallel sample sort over the existing Chase-Lev work-stealing pool.
- IPS4o comparison harness: `bench/bench_ips4o_compare.cpp` and documentation in `bench/README_ips4o_compare.md`.
- IPS4o regression matrix now includes a high-distinct `struct { int key; int payload; }` custom-comparator case.
- Adaptive-order weapons (`parts/12b_adaptive.hpp`), three general structure detectors that replace per-shape special cases:
  - natural-run merge — one scan finds the maximal monotone runs, descending runs are reversed in place and the runs are merged bottom-up with a buffer no larger than the smaller side, so rotated arrays, concatenated sorted blocks and block permutations cost one or two passes instead of 4-8 radix passes;
  - local dirty-patch merge — positions taking part in an adjacent inversion are pulled out, sorted, and merged back over the compacted clean run;
  - displacement-patch merge — element `i` may stay iff it is `>=` everything before it and `<=` everything after it; two scans compute that exactly (two bits per element, no iteration), so blocks and segments moved anywhere are found too, including shapes the adjacent-inversion test cannot see at all.
  All four are read-only until the structure is proven, only permute, and keep the radix total order for default-order floating point.  `FYX_ENABLE_ADAPTIVE_WEAPONS=0` compiles them out.
  - sorted-affix sort — longest non-decreasing prefix, longest non-decreasing suffix, sort the stretch between them, merge the three runs.  A sorted table with a batch of new records appended, a log with an unflushed tail, a file with one damaged region: 1M int32 whose last tenth is shuffled was 0.031s through eight radix passes and is 0.0039s here, against 0.0046s for pdqsort.  Both scans walk inwards and stop at the first inversion, so a range with no ordered head or tail costs two comparisons.
- `test/t_adaptive.cpp`: the three weapons over 13 shapes x 4 types, serial and parallel — result matches `std::sort`, a declining weapon leaves the array byte-identical, high-entropy input is rejected cheaply, NaN/`-0`/`+0` keep the radix total order, custom comparators and odd sizes included.
- `bench/bench_matrix.cpp`: local distribution matrix (int32/int64/double/string x 16 distributions) with optional ips4o/pdqsort/`std::sort` reference columns, copy time excluded.
- Counting/string/comparator-key/parallel-merge edge-case tests in `test/t_counting.cpp`.
- Unified top-level input profiling (`InputProfile` / `profile_input`) with dispatch test hooks for sorted, reverse, all-equal, low-cardinality, partially sorted and high-entropy cases.
- Contiguous iterator fast-path detection for vector/string-style normal iterators, so `fyx::sort(v.begin(), v.end())` reaches the same pointer dispatcher as `fyx::sort(v)`.

### Improved

- Adaptive repair for order that already nearly holds (`v10.1` weapon-eight work):
  - bounded insertion repair now decides from the disorder itself instead of from a sample: insertion sort's cost is the inversion count, and an array whose 128-element blocks were permuted twenty times has thirty-eight inversions in a million positions, none of them in the first four thousand.  The repair polices its own budget while it runs — long travel is refused (an element crossing an eighth of the range is a block move), the running rate is capped with a strike to spare (shifts arrive in bursts), and the absolute budget is two shifts per element.  1M int32 block swaps 0.011s -> 0.0027s.
  - it rehearses on a copy of the prefix first, because insertion sort permutes as it goes and a shape abandoned half-way through used to come out the other end no longer recognisable to the detectors downstream.
  - the density gate that used to refuse nearly-sorted input (it required 8 inversions per 100 ordered pairs) is gone; the shift budget does that job and measures the quantity insertion actually pays for.  1M nearly sorted: int32 0.0049s -> 0.00057s, int64 8.9x faster than before, string 0.031s -> 0.0076s.
  - merging a small patch into a large run now touches the run in blocks: compaction walks the dirty bitmap a word at a time, and the merge gallops back through the run to move in one go everything that belongs after the next patch element.  1M int32 far swaps: patch merge 0.0079s -> 0.0040s.
- The high-prefix radix pass picks its width from the data instead of from the type.  A pass costs two traversals per digit plus a tie repair inside every prefix group, so the width that pays depends on how many groups the range falls into, and that is a property of the data: doubles in a narrow range share their high bits and collapse into tens of thousands of groups at 24 bits, where the tie repair costs more than the extra pass a 36-bit prefix would have spent, while uniform 64-bit data splinters into millions of groups at 24 bits already and there the wider prefix buys nothing but two more traversals.  A 4096-element sample estimates the group count (S samples in G groups collide about `S*S/(2G)` times) and keeps the narrow prefix while the groups it leaves average two elements or fewer.  1M double, uniform bits: 0.0245s -> 0.0186s serial, 0.0147s -> 0.0118s parallel; 1M int64 in [0, 2^50): 0.046s -> 0.0152s parallel.
- Comparison sample sort now checks for already-ordered, reversed and single-value input itself, so callers that bypass the profile (recursive bucket sorts, a comparator the profile cannot see through) no longer pay for sampling, splitter search and 256-way classification.  The pass abandons itself as soon as the range can be none of the three, so random input leaves after two or three comparisons.  Calling the sample sort directly on 1M int32: sorted 0.0097s -> 0.0012s, reverse 0.0184s -> 0.0012s, all equal 0.0036s -> 0.0009s.
- Monotone distribution detection for sorted/reverse/all-equal inputs is now centralized in the weapon-seven profile layer and fronted by `FYX_ENABLE_FAST_PATHS`: arithmetic all-equal uses shifted `memcmp`, large string all-equal can validate in parallel, sorted inputs take a single proof scan, reverse inputs can verify while swapping, front-reversed organ-pipe zigzag (the bench_final shape) is detected before the reverse fast path and sorted by reversing only the first half, dense adjacent-swap zigzag now uses an in-place pair repair, local sawtooth zigzag uses a sample-budgeted bounded insertion repair, half-organ/bitonic zigzag verifies two consecutive monotone runs (with direct arithmetic fill for consecutive numeric domains and a half-buffer string interleave for the fixed-width organ shape), and broader interleaved zigzag inputs are recognized as two parity monotone runs; concatenable runs use a half-buffer deinterleave before falling back to the full linear merge.  The profile samples 1024 elements, then when worthwhile performs one linear validation scan that combines sorted/reverse/all-equal checks, capped distinct detection and adjacent inversion counting.  Low-cardinality samples are used as a dispatch hint and are still fully validated by the existing counting paths, avoiding an extra profile-only O(n) pass on the low-cardinality hot path.  Default-order floating point uses radix keys to preserve NaN and `-0/+0` semantics.
- Partially sorted inputs whose adjacent inversion count is at most `n/64` now first try adjacent/local repairs; numeric default-order long-distance nearly-sorted inputs first detect consecutive integer / integer-valued floating permutations and fill the ordered range directly, then fall straight into radix after local repair declines, avoiding the old profile+pdq path; non-numeric long-distance perturbations use an iterative dirty-patch merge before pdq fallback.  Floating-point default-order pattern repair/pdq uses radix-key comparison so NaN and `-0/+0` total-order guarantees are preserved.
- Auto parallel dispatch now has a dynamic minimum-size gate (default about one million elements, plus a 128 KiB/thread floor) that can be overridden with the `FYX_MIN_PARALLEL_SIZE` environment variable, reducing small-input task overhead without affecting explicit `Tri::On`.  1M-scale 32-bit wide-radix now uses 1 chunk/worker and disables NT stores before returning to the larger-input WCB/NT path, and guarded natural-order comparator recovery gets serial high-prefix radix for high-entropy 64-bit integers and `double`.
- Radix-key sparse counting now covers all radix-supported scalar keys, not just integral types; this gives default-order low-cardinality `float`/`double` inputs an O(n) count/fill path while preserving radix total-order semantics for `-0/+0` and NaNs. Large low-cardinality radix-key inputs also have a parallel count/fill variant, and the tiny distinct-table hash was shortened to reduce per-element overhead on low-cardinality numeric data.
- Sample sort now uses a lighter IPS4o-style oversampling policy and temp scatter for non-trivial object types to improve cache locality.
- Sample-sort permutation was replaced with block-level bucket reorder: per-source-block bucket counts/prefix bases scatter into a temporary buffer, then copy/move back by block.
- Comparator-key sorting for trivially copyable records now detects integer key fields (`int32/uint32/int64/uint64` prefixes/offsets), validates comparator semantics by sampling, uses low-cardinality count sort when appropriate, and otherwise uses payload-preserving field-key radix sort with a final comparator `is_sorted` guard.
- High-distinct comparator-key probes now skip expensive low-cardinality counting attempts before entering field-key radix sort.
- Parallel divide-and-conquer merge now prefers buffered recursive block merge and falls back to `std::inplace_merge` on allocation failure or unsupported shapes.
- High-entropy numeric default-order inputs now enter chunked parallel radix beyond the 64-bit-only case; 32-bit integers and `float` can use the same multi-chunk count/scatter schedule, the first radix pass reuses the initial local histograms, integer/`float` value-buffer radix avoids the extra encode/decode arrays when profitable, and the chunk scheduler uses more fine-grained chunks for better 4-core/VM load balance.
- High-entropy radix now has a safe all-pass fast plan: if an evenly-spaced sample proves every radix byte varies, the parallel path counts only byte 0 for the first scatter instead of building a full all-pass planning histogram.  If any byte looks degenerate in the sample, FYX falls back to the full histogram/skip-pass planner.
- Added an MSD-bucket hybrid for the remaining vqsort numeric gaps, but gate it to 2-worker/bandwidth-constrained runs after the 4H8G matrix showed normal chunked radix is better on four hardware threads.  The hybrid splits high-entropy `float` on the top radix byte and 64-bit integer/`double` on the top 16 radix bits, then sorts buckets with tiny-bucket PDQ or lower-pass radix.
- Profile-hinted sparse low-cardinality numeric inputs now use sample-gated parallel dense-range counting for 64..65536-wide integer domains before falling back to sparse/radix-key counting, avoiding a wasted full min/max scan on huge-span sparse data while speeding up 256-way integer lowcard.  Floating-point plus sparse 32/64-bit low-cardinality keys also get a collision-free rank16 direct-map counter before the hash-table sparse path, with dense integer ranges and the compact float/double prefix direct-map kept ahead of it; the hash-table radix-key fallback now also keeps the full 256-way cap for `float` instead of the old 64-way guard, so arbitrary 256-way float lowcard cannot fall through to radix just because the rank16 projection is unlucky. This turns arbitrary 256-way float and wrapped/sparse 32-bit integer lowcard into near pure count/fill passes while preserving exact radix-key validation.
- Chunked parallel radix now uses 32-bit per-chunk histograms/recount buffers when chunk sizes fit, and chooses coarser chunks for floating/wide-key numeric sorts.  Recount passes use banked 4-way local counters; 64-bit integer value-buffer radix is restored to the measured 3 chunks/worker setting after the 2 chunks/worker experiment regressed the 4H2G vqsort matrix, and the `double` key-buffer radix path decodes during the final scatter into the user array without entering the integer value-buffer instantiations.  This reduces store-forwarding pressure, cache footprint and the extra output-copy cost without changing the public API or low-cardinality dispatch.
- High-entropy 32-bit integer random inputs now try a three-pass 10/11/11 wide-key radix before the byte-wise fallback, cutting one full count/scatter round and using 1M-aware chunk/NT-store gating to reduce task and WCB pressure.  High-entropy 64-bit default-order random inputs try a guarded high-prefix radix before the full 8-pass fallback: `int64`/`uint64` use a two-pass top24 prefix at 1M scale and top26 beyond that, while `double` uses top36 at 1M scale and keeps top39 for larger ranges.  The high-prefix tie repair scans decoded output with one cached prefix per element instead of re-encoding boundary elements twice.  A prefix-distinct sample gate keeps narrow-range/low-prefix data on the existing full radix path, preserving low-cardinality wins and correctness.
- Custom comparators that sample as natural ascending/descending order now safely recover the radix/counting fast paths for arithmetic types and the high-prefix radix path for high-entropy 64-bit numerics; partially-sorted natural-comparator numerics also probe this guarded radix route before pdq/patch fallback.  Low-cardinality `std::string` comparator inputs now use unordered value-count/fill regardless of comparator type, and large inputs use sampled exact keys plus parallel count/fill while sorting only the distinct strings; guarded MSD remains available for natural-order string random data.
- Sample-sort classification now uses an unrolled fixed-256 Eytzinger descent for cheap/trivial payloads while keeping the compact looped classifier, previous 128K recursion handoff, and block scatter for `std::string` fallback paths. Non-string serial sample-sort scatter uses a single prefix-position pass. Parallel arithmetic comparator fallback now uses a 64-way top partition with the old 256-way sampling budget, cutting random numeric classification from eight to six comparisons per element while preserving the 256-way low-cardinality signal. Low-distinct arithmetic samples use a tiny exact counter before falling back to pdqsort; the floating-point duplicate gate now also tries a collision-free rank16 exact counter (then the sparse hash counter as fallback) up to the 256-way sample band instead of sending 256-way float/double comparator inputs straight to pdqsort. The comparison recursion threshold remains lower for non-string data to keep high-distinct buckets in sample-sort longer when that is cheaper than large pdqsort leaves.
- Degenerate sample-sort splitter cases now fall back to pdqsort instead of recursing without progress.

### Changed

- The high-prefix partition — two passes over the top bits, ties finished while
  the buckets are still in cache — used to be reachable only for 8-byte keys,
  so random `int32` fell through to the 32-wide sort, which was 1.6x slower than
  even the plain LSD sort it was meant to replace.  The helpers now take their
  shifts from `sizeof(Key)`, 4-byte keys get their own `radix_choose_prefix_bits`
  branch ((24,12) up to 2^21 elements, (26,13) above), and both dispatchers try
  the high-prefix sort first.  Parallel `int32` routes through the parallel
  high-prefix kernel as well; the two parallel kernels cross over between 2M and
  4M, so the 4-byte branch declines past 3M.  Random data, 2 vCPU: int32 1M
  parallel 0.0098 -> 0.0061, int32 8M serial 0.154 -> 0.083, 16M parallel 0.080.
- Tie repair no longer pays a full sweep to find the groups it is about to
  repair.  AVX-512 compares sixteen elements (eight for 64-bit keys) against the
  vector rotated by one lane and produces a mask with a bit wherever an element
  continues its predecessor's group, so a vector with no ties is skipped whole
  and one with ties is walked a group at a time.  Tie scan on 1M int32 without
  ties: 2.23 -> 0.21 ns/elem; on 8M: 1.36 -> 0.53.
- Scatters are now chosen per size instead of per kernel.  The write-combining
  scatter — one cache line per bucket, flushed with a non-temporal store —
  keeps a large sort out of the read-for-ownership business, but its bookkeeping
  costs about 3.3 ns/element, and an array that fits in cache has nothing to
  save.  Below roughly 8 MB of keys the wide high-prefix kernel uses a
  conflict-detection scatter instead: `vpconflictd` for the lanes that want the
  same bucket, `vpopcntd` for each lane's rank among them, a gather for the
  bucket's running position, and two scatters to write the keys and the advanced
  positions back.  `tools/dev/scat.cpp`, 12-bit digits, 1M: 2.74 ns/elem against
  3.26 for write-combining; at 8M the two switch places (4.57 against 3.14),
  which is where the threshold sits.  End to end, 1M random: int32 parallel
  0.0061 -> 0.0049, double parallel 0.0118 -> 0.0077; 8M unchanged.

### Added

- `tools/dev/{scat,hist,correct,timer}.cpp` — the measurements the last two
  changes rest on, so the next round does not have to rebuild them: scatter
  kernels against each other across sizes, histogram variants, a radix
  correctness harness that straddles every crossover (sortedness and equality
  with `std::sort`, serial and parallel, 6 int32 shapes, every key width), and a
  serial-vs-parallel timer.  `hist.cpp` is why the AVX-512 conflict histogram in
  `09_radix` stays unwired: 0.72 ns/elem against 0.57 for the shipping four-bank
  scalar one.

### Fixed

- Move-only payloads (`std::unique_ptr`) now sort.  The parallel merge assigned
  through a const lvalue, and the sample sorts keep copies of the elements they
  sample: both refused to compile for a type that cannot be copied.  The merge
  moves now, and a payload that cannot be copied is routed to the comparison
  sort or the task-parallel divide, which only move.
- Running out of memory no longer propagates out of `fyx::sort`.  The fast paths
  allocate -- scratch, buffers, worker threads -- and `std::sort` never does, so
  a sorter whose fast paths do inherit that failure is strictly less robust than
  the sort it replaces.  `bad_alloc` and a worker that cannot be started now fall
  back to the in-place comparison sort, which needs neither.  Verified by sorting
  2M int32 under a 60 MB address-space cap.

### Benchmark snapshot

Google Highway 1.4.0 is in the tree now, so the matrix has a third competitor
next to `std::sort` and `pdqsort`: `vqsort`.  `bash tools/dev/vqsort.sh 1000000`
(or `8000000`) clones Highway, builds the matrix and writes
`build/vqsort_<n>.txt`; every cell is scored against the best of the three
(`vqsort` is arithmetic-only, so `std::string` rows degrade to a two-way race).

    random, parallel, 1M / 8M          fyx      vqsort
      int32     0.00459 / 0.04943   0.00371 / 0.03919   0.81x / 0.79x
      int64     0.00552 / 0.06703   0.00771 / 0.08547   1.40x / 1.28x
      double    0.00632 / 0.08242   0.00683 / 0.08159   1.08x / 0.99x

    whole matrix, arithmetic cells     1M: 35 wins / 7 losses
                                       8M: 37 wins / 5 losses

Structured input is where the lead is: at 8M, reverse 3.1-4.0x, nearly sorted
5.6-7.3x, concatenated sorted halves 1.9-2.4x, rotated 1.7-2.6x, 256 distinct
values 2.0-2.7x.  The losses are concentrated in two families -- uniform random
`int32` (0.79-0.81x, the scatter pass is dominated by random writes) and inputs
with a handful of distinct values (0.50-0.83x, the counting pass) -- plus
all-equal `int32`/`int64` at 8M, where detection is already within 25% of the
`memcpy` roof.  `BENCHMARKS.md` lists every losing cell with its number.
