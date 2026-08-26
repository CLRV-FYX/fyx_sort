# Changelog

## Unreleased — v10.1 candidate

Date: 2026-08-26

### Added

- Low-cardinality counting paths:
  - dense integer range counting;
  - sparse integer counting for up to 256 encoded keys;
  - compressed comparator-equivalence counting for arbitrary payload objects;
  - `std::string` value counting for default ascending/descending order;
  - guarded trivial-prefix field counting for common `struct { int key; ... }` custom comparator shapes.
- Parallel specializations for the low-cardinality paths:
  - parallel sparse integer counting;
  - parallel trivial-prefix field counting/scatter;
  - parallel top-level `std::string` MSD byte-radix sorting.
- `std::string` MSD byte-radix sort for default string order, including descending support via reverse.
- Parallel sample sort over the existing Chase-Lev work-stealing pool.
- IPS4o comparison harness: `bench/bench_ips4o_compare.cpp` and documentation in `bench/README_ips4o_compare.md`.
- IPS4o regression matrix now includes a high-distinct `struct { int key; int payload; }` custom-comparator case.
- Counting/string/comparator-key/parallel-merge edge-case tests in `test/t_counting.cpp`.
- Unified top-level input profiling (`InputProfile` / `profile_input`) with dispatch test hooks for sorted, reverse, all-equal, low-cardinality, partially sorted and high-entropy cases.

### Improved

- Monotone distribution detection for sorted/reverse/all-equal inputs is now centralized in the weapon-seven profile layer.  The profile samples 1024 elements, then when worthwhile performs one linear validation scan that combines sorted/reverse/all-equal checks, capped distinct detection and adjacent inversion counting.  Low-cardinality samples are used as a dispatch hint and are still fully validated by the existing counting paths, avoiding an extra profile-only O(n) pass on the low-cardinality hot path.  Default-order floating point uses radix keys to preserve NaN and `-0/+0` semantics.
- Partially sorted inputs whose adjacent inversion count is at most `n/64` now short-circuit to pdqsort before sample/radix fallback when that is the safer dispatch choice.
- Sample sort now uses a lighter IPS4o-style oversampling policy and temp scatter for non-trivial object types to improve cache locality.
- Sample-sort permutation was replaced with block-level bucket reorder: per-source-block bucket counts/prefix bases scatter into a temporary buffer, then copy/move back by block.
- Comparator-key sorting for trivially copyable records now detects integer key fields (`int32/uint32/int64/uint64` prefixes/offsets), validates comparator semantics by sampling, uses low-cardinality count sort when appropriate, and otherwise uses payload-preserving field-key radix sort with a final comparator `is_sorted` guard.
- High-distinct comparator-key probes now skip expensive low-cardinality counting attempts before entering field-key radix sort.
- Parallel divide-and-conquer merge now prefers buffered recursive block merge and falls back to `std::inplace_merge` on allocation failure or unsupported shapes.
- Degenerate sample-sort splitter cases now fall back to pdqsort instead of recursing without progress.

### Benchmark snapshot

On the local 2-vCPU Xeon sandbox, GCC 12.2, `-O3 -march=native`, FYX is faster than IPS4o on the tracked sequential and real oneTBB parallel matrices: i32 random, i32 low distinct, i64 sparse 256 distinct, random strings, low-distinct strings, low-distinct struct-key payloads, and high-distinct struct-key payloads. Latest snapshot after weapon seven: high-distinct struct-key 1M is 0.0138s vs IPS4o sequential 0.0278s, and 0.0127s vs real IPS4o parallel 0.0142s.

The parallel matrix was verified against real oneTBB and linked with `-ltbb -latomic`.
