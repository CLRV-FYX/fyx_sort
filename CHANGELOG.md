# Changelog

## Unreleased — v10.1 candidate

Date: 2026-08-27

### Added

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
- Counting/string/comparator-key/parallel-merge edge-case tests in `test/t_counting.cpp`.
- Unified top-level input profiling (`InputProfile` / `profile_input`) with dispatch test hooks for sorted, reverse, all-equal, low-cardinality, partially sorted and high-entropy cases.
- Contiguous iterator fast-path detection for vector/string-style normal iterators, so `fyx::sort(v.begin(), v.end())` reaches the same pointer dispatcher as `fyx::sort(v)`.

### Improved

- Monotone distribution detection for sorted/reverse/all-equal inputs is now centralized in the weapon-seven profile layer.  The profile samples 1024 elements, then when worthwhile performs one linear validation scan that combines sorted/reverse/all-equal checks, capped distinct detection and adjacent inversion counting.  Low-cardinality samples are used as a dispatch hint and are still fully validated by the existing counting paths, avoiding an extra profile-only O(n) pass on the low-cardinality hot path.  Default-order floating point uses radix keys to preserve NaN and `-0/+0` semantics.
- Partially sorted inputs whose adjacent inversion count is at most `n/64` now short-circuit to pdqsort before sample/radix fallback when that is the safer dispatch choice.
- Radix-key sparse counting now covers all radix-supported scalar keys, not just integral types; this gives default-order low-cardinality `float`/`double` inputs an O(n) count/fill path while preserving radix total-order semantics for `-0/+0` and NaNs. Large low-cardinality radix-key inputs also have a parallel count/fill variant, and the tiny distinct-table hash was shortened to reduce per-element overhead on low-cardinality numeric data.
- Sample sort now uses a lighter IPS4o-style oversampling policy and temp scatter for non-trivial object types to improve cache locality.
- Sample-sort permutation was replaced with block-level bucket reorder: per-source-block bucket counts/prefix bases scatter into a temporary buffer, then copy/move back by block.
- Comparator-key sorting for trivially copyable records now detects integer key fields (`int32/uint32/int64/uint64` prefixes/offsets), validates comparator semantics by sampling, uses low-cardinality count sort when appropriate, and otherwise uses payload-preserving field-key radix sort with a final comparator `is_sorted` guard.
- High-distinct comparator-key probes now skip expensive low-cardinality counting attempts before entering field-key radix sort.
- Parallel divide-and-conquer merge now prefers buffered recursive block merge and falls back to `std::inplace_merge` on allocation failure or unsupported shapes.
- High-entropy numeric default-order inputs now enter chunked parallel radix beyond the 64-bit-only case; 32-bit integers and `float` can use the same multi-chunk count/scatter schedule, the first radix pass reuses the initial local histograms, integer/`float` value-buffer radix avoids the extra encode/decode arrays when profitable, and the chunk scheduler uses more fine-grained chunks for better 4-core/VM load balance.
- High-entropy radix now has a safe all-pass fast plan: if an evenly-spaced sample proves every radix byte varies, the parallel path counts only byte 0 for the first scatter instead of building a full all-pass planning histogram.  If any byte looks degenerate in the sample, FYX falls back to the full histogram/skip-pass planner.
- Added an MSD-bucket hybrid for the remaining vqsort numeric gaps, but gate it to 2-worker/bandwidth-constrained runs after the 4H8G matrix showed normal chunked radix is better on four hardware threads.  The hybrid splits high-entropy `float` on the top radix byte and 64-bit integer/`double` on the top 16 radix bits, then sorts buckets with tiny-bucket PDQ or lower-pass radix.
- Profile-hinted sparse low-cardinality numeric inputs now try sparse/radix-key counting before dense min/max range counting, avoiding a wasted full range scan on random lowcard16/lowcard256 data with large value spans.
- Chunked parallel radix now uses 32-bit per-chunk histograms/recount buffers when chunk sizes fit, and chooses coarser chunks for floating/wide-key numeric sorts.  Recount passes use banked 4-way local counters; 64-bit integer value-buffer radix is restored to the measured 3 chunks/worker setting after the 2 chunks/worker experiment regressed the 4H2G vqsort matrix, and the `double` key-buffer radix path decodes during the final scatter into the user array without entering the integer value-buffer instantiations.  This reduces store-forwarding pressure, cache footprint and the extra output-copy cost without changing the public API or low-cardinality dispatch.
- Custom comparators that sample as natural ascending/descending order now safely recover the radix/counting fast paths for arithmetic types and the MSD radix path for `std::string`; every guarded fast path is verified with `std::is_sorted(comp)` before returning, otherwise the comparison sorter continues from the resulting permutation.
- Sample-sort classification now uses an unrolled fixed-256 Eytzinger descent for cheap/trivial payloads while keeping the compact looped classifier, previous 128K recursion handoff, and block scatter for `std::string` fallback paths. Non-string serial sample-sort scatter uses a single prefix-position pass, and low-distinct arithmetic samples use a tiny sparse value counter before falling back to pdqsort, avoiding the bucket-scatter cost on compare-only lowcard16/lowcard256 data. The comparison recursion threshold remains lower for non-string data to keep high-distinct buckets in sample-sort longer when that is cheaper than large pdqsort leaves.
- Degenerate sample-sort splitter cases now fall back to pdqsort instead of recursing without progress.

### Benchmark snapshot

On the local 2-vCPU Xeon sandbox, GCC 12.2, `-O3 -march=native`, FYX is faster than IPS4o on the tracked sequential and real oneTBB parallel matrices: i32 random, i32 low distinct, i64 sparse 256 distinct, random strings, low-distinct strings, low-distinct struct-key payloads, and high-distinct struct-key payloads. Latest snapshot after weapon seven: high-distinct struct-key 1M is 0.0138s vs IPS4o sequential 0.0278s, and 0.0127s vs real IPS4o parallel 0.0142s.

The parallel matrix was verified against real oneTBB and linked with `-ltbb -latomic`.
