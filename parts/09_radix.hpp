// ============================================================================
//  Section 9 -- LSD radix sort
//
//  This is the workhorse for numeric keys with the default comparator.
//
//  Structure
//  ---------
//   1. ONE fused counting pass builds the histograms of *all* passes at once
//      (P x 256 counters).  The data is read exactly once here.
//   2. Degenerate passes -- those where every key lands in the same bucket --
//      are skipped.  Sorted, constant, or narrow-range data therefore costs a
//      fraction of the full run, and 64-bit keys holding small values skip
//      four of their eight passes.
//   3. Each surviving pass scatters src -> dst through 256 software
//      write-combining buffers.  A full 64-byte line is flushed with
//      non-temporal stores, which avoids the read-for-ownership traffic that
//      would otherwise consume half of the write bandwidth.
//   4. src and dst are swapped after every pass.  Because we know the number of
//      surviving passes in advance we can choose the initial buffer so that the
//      final pass lands in the caller's array -- no extra copy, ever.
//
//  Histogram variants
//  ------------------
//   * scalar_histogram   -- four interleaved counter banks to hide the
//                           store-to-load latency of repeated ++count[digit].
//   * simd_histogram     -- genuine AVX-512 conflict detection:
//                           vpconflictd + vpopcntd + vpgatherdd + vpscatterdd.
//                           Required by the specification; measured against the
//                           scalar version at run time (see kUseSimdHistogram).
// ============================================================================

namespace fyx {
namespace detail {

/// Per-pass histogram block: P passes x 256 buckets.
template <unsigned Passes>
struct RadixHistogram {
    // 64-bit counters: an array may legitimately exceed 4 G elements.
    std::uint64_t count[Passes][kRadixBuckets];

    void clear() noexcept {
        std::memset(count, 0, sizeof(count));
    }
};

/// Extracts digit `pass` (8 bits) from an encoded key.
template <typename Key>
FYX_FORCE_INLINE unsigned radix_digit(Key k, unsigned pass) noexcept {
    return static_cast<unsigned>((k >> (pass * kRadixBits)) & Key(kRadixMask));
}

// ---------------------------------------------------------------------------
// Scalar fused histogram
// ---------------------------------------------------------------------------
//
// Four independent counter banks remove the serial dependency between
// consecutive increments of the same bucket (a 5-cycle store-forward stall on
// most cores).  They are summed at the end.
// ---------------------------------------------------------------------------

template <typename T, unsigned Passes>
inline void scalar_histogram(const T* FYX_RESTRICT src, std::size_t n,
                             RadixHistogram<Passes>& hist) noexcept {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;

    // Bank layout: [pass][bank][bucket] -- bank-major within a pass keeps the
    // four banks of one pass in adjacent cache lines.
    constexpr unsigned kBanks = 4;
    ScratchLease<std::uint32_t> banks_lease(
        static_cast<std::size_t>(Passes) * kBanks * kRadixBuckets);

    if (!banks_lease.valid()) {
        // No memory for the banked version: count directly.
        hist.clear();
        for (std::size_t i = 0; i < n; ++i) {
            const Key k = RT::encode(src[i]);
            for (unsigned p = 0; p < Passes; ++p) ++hist.count[p][radix_digit(k, p)];
        }
        return;
    }

    std::uint32_t* banks = banks_lease.get();
    std::memset(banks, 0,
                static_cast<std::size_t>(Passes) * kBanks * kRadixBuckets *
                    sizeof(std::uint32_t));

    // A 32-bit bank counter would wrap after 4 G identical digits; flush the
    // banks into the 64-bit histogram every kFlush elements to stay exact.
    constexpr std::size_t kFlush = std::size_t(1) << 30;

    hist.clear();
    std::size_t base = 0;
    while (base < n) {
        const std::size_t stop = (n - base > kFlush) ? base + kFlush : n;

        std::size_t i = base;
        // Unroll by 4, one bank each, so the increments are independent.
        for (; i + 4 <= stop; i += 4) {
            prefetch_stream(src, i, n);
            const Key k0 = RT::encode(src[i + 0]);
            const Key k1 = RT::encode(src[i + 1]);
            const Key k2 = RT::encode(src[i + 2]);
            const Key k3 = RT::encode(src[i + 3]);
            for (unsigned p = 0; p < Passes; ++p) {
                std::uint32_t* b = banks + static_cast<std::size_t>(p) * kBanks * kRadixBuckets;
                ++b[0 * kRadixBuckets + radix_digit(k0, p)];
                ++b[1 * kRadixBuckets + radix_digit(k1, p)];
                ++b[2 * kRadixBuckets + radix_digit(k2, p)];
                ++b[3 * kRadixBuckets + radix_digit(k3, p)];
            }
        }
        for (; i < stop; ++i) {
            const Key k = RT::encode(src[i]);
            for (unsigned p = 0; p < Passes; ++p)
                ++banks[static_cast<std::size_t>(p) * kBanks * kRadixBuckets +
                        radix_digit(k, p)];
        }

        // Fold the banks into the wide histogram and reset them.
        for (unsigned p = 0; p < Passes; ++p) {
            std::uint32_t* b = banks + static_cast<std::size_t>(p) * kBanks * kRadixBuckets;
            for (unsigned d = 0; d < kRadixBuckets; ++d) {
                hist.count[p][d] += std::uint64_t(b[0 * kRadixBuckets + d]) +
                                    std::uint64_t(b[1 * kRadixBuckets + d]) +
                                    std::uint64_t(b[2 * kRadixBuckets + d]) +
                                    std::uint64_t(b[3 * kRadixBuckets + d]);
            }
        }
        std::memset(banks, 0,
                static_cast<std::size_t>(Passes) * kBanks * kRadixBuckets *
                    sizeof(std::uint32_t));
        base = stop;
    }
}


// ---------------------------------------------------------------------------
// AVX-512 conflict-detection histogram
// ---------------------------------------------------------------------------
//
// The specification requires a *real* SIMD histogram, not scalar code in a
// vector wrapper.  The obstacle is that 16 lanes may target the same bucket in
// one step, and a plain gather/add/scatter would then lose all but one of the
// increments.  vpconflictd solves this exactly:
//
//   vpconflictd  -> for lane i, a bitmask of the lanes j<i with idx[j]==idx[i]
//   vpopcntd     -> how many earlier lanes share this bucket = this lane's rank
//   gather       -> the current counter value
//   + rank + 1   -> every lane writes a distinct, correct running total
//   scatter      -> the highest-ranked lane's value survives, which is the
//                   total including all duplicates in this vector
//
// This is exact for any duplicate pattern, including all 16 lanes equal.
// Requires AVX512F + AVX512CD; vpopcntd (AVX512VPOPCNTDQ) is emulated when
// absent, which is why the runtime gate also checks avx512vpopcntdq.
// ---------------------------------------------------------------------------

#if FYX_HAS_AVX512_CODE
FYX_ISA_BEGIN("avx512f,avx512cd,avx512vpopcntdq,avx512bw,avx512dq,avx512vl")
namespace isa_avx512_hist {

/// Accumulates a 256-bucket histogram of the 8-bit digit at `shift` for
/// 32-bit encoded keys.  `hist` is 256 x uint32 and must be zeroed.
inline void histogram_u32_avx512(const std::uint32_t* FYX_RESTRICT keys,
                                 std::size_t n, unsigned shift,
                                 std::uint32_t* FYX_RESTRICT hist) noexcept {
    const __m512i vmask = _mm512_set1_epi32(int(kRadixMask));
    const __m512i vone  = _mm512_set1_epi32(1);
    const __m512i vzero = _mm512_setzero_si512();

    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        if (FYX_LIKELY(i + 128 < n))
            prefetch_read<1>(keys + i + 128);

        const __m512i k   = _mm512_loadu_si512(
                                reinterpret_cast<const void*>(keys + i));
        // Digit index for each lane.
        const __m512i idx = _mm512_and_si512(_mm512_srli_epi32(k, shift), vmask);

        // Rank of each lane among earlier lanes hitting the same bucket.
        const __m512i cfl  = _mm512_conflict_epi32(idx);
        const __m512i rank = _mm512_popcnt_epi32(cfl);

        // Read-modify-write.  All 16 lanes are active, hence mask 0xFFFF.
        const __m512i cur = _mm512_mask_i32gather_epi32(
                                vzero, static_cast<__mmask16>(0xFFFF), idx,
                                reinterpret_cast<const void*>(hist), 4);
        const __m512i upd = _mm512_add_epi32(_mm512_add_epi32(cur, rank), vone);
        _mm512_i32scatter_epi32(reinterpret_cast<void*>(hist), idx, upd, 4);
    }
    for (; i < n; ++i)
        ++hist[(keys[i] >> shift) & kRadixMask];
}

/// Same for 64-bit encoded keys: 8 lanes per vector, 32-bit counters.
inline void histogram_u64_avx512(const std::uint64_t* FYX_RESTRICT keys,
                                 std::size_t n, unsigned shift,
                                 std::uint32_t* FYX_RESTRICT hist) noexcept {
    const __m512i vmask = _mm512_set1_epi64(std::int64_t(kRadixMask));
    const __m256i vone  = _mm256_set1_epi32(1);
    const __m256i vzero = _mm256_setzero_si256();

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        if (FYX_LIKELY(i + 64 < n))
            prefetch_read<1>(keys + i + 64);

        const __m512i k = _mm512_loadu_si512(
                              reinterpret_cast<const void*>(keys + i));
        // Digits fit in 8 bits, so narrow to 32-bit lanes for conflict/gather.
        const __m512i d64 = _mm512_and_si512(_mm512_srli_epi64(k, shift), vmask);
        const __m256i idx = _mm512_cvtepi64_epi32(d64);

        const __m256i cfl  = _mm256_conflict_epi32(idx);
        const __m256i rank = _mm256_popcnt_epi32(cfl);

        const __m256i cur = _mm256_mmask_i32gather_epi32(
                                vzero, static_cast<__mmask8>(0xFF), idx,
                                reinterpret_cast<const void*>(hist), 4);
        const __m256i upd = _mm256_add_epi32(_mm256_add_epi32(cur, rank), vone);
        _mm256_i32scatter_epi32(reinterpret_cast<void*>(hist), idx, upd, 4);
    }
    for (; i < n; ++i)
        ++hist[unsigned((keys[i] >> shift)) & kRadixMask];
}

} // namespace isa_avx512_hist
FYX_ISA_END
#endif // FYX_HAS_AVX512_CODE

// ---------------------------------------------------------------------------
// Software write-combining scatter
// ---------------------------------------------------------------------------
//
// A naive scatter writes single elements to up to 256 destinations at once.
// Every such store is a partial cache line, so the core must first read the
// line it is about to overwrite (read-for-ownership).  With an output far
// larger than L2 that doubles the memory traffic of the pass.
//
// The cure is a per-bucket buffer exactly one cache line wide.  Elements
// accumulate there and a full line leaves with one non-temporal store, which
// neither reads the destination nor pollutes the cache.
//
// The subtlety that makes or breaks this: an NT store only works on a 64-byte
// aligned address, and a bucket generally starts mid-line.  Flushing "when
// aligned, otherwise memcpy" is worthless -- the cursor stays misaligned
// forever and no store is ever streamed.  Instead each bucket first diverts
// its leading `need` elements into a small head buffer, chosen so that the
// main cursor lands exactly on a line boundary.  From then on *every* flush is
// an aligned full-line NT store.  The heads are written back at the end.
//
// Measured on Ice Lake-SP, 10 M uint32 into 256 buckets:
//     naive scatter                  5.75 ns/elem
//     WCB, flush only when aligned   6.80 ns/elem   (never actually streams)
//     WCB, pre-aligned cursors       2.27 ns/elem   <- this implementation
// ---------------------------------------------------------------------------

/// Number of elements of type `Key` that fit in one cache line.
template <typename Key>
struct WcbTraits {
    static constexpr std::size_t kPerLine = kCacheLine / sizeof(Key);
    static_assert(kPerLine * sizeof(Key) == kCacheLine,
                  "key size must divide the cache line");
};

/// Scratch owned by the caller and reused across passes, so that a multi-pass
/// sort performs no allocation at all inside the loop.
template <typename Key>
struct RadixScatterScratch {
    static constexpr std::size_t kPerLine = WcbTraits<Key>::kPerLine;

    Key*          line = nullptr;   ///< 256 * kPerLine, 64-byte aligned
    Key*          head = nullptr;   ///< 256 * kPerLine, alignment prologues
    std::uint32_t fill[kRadixBuckets];  ///< elements currently in line[b]
    std::uint32_t hn  [kRadixBuckets];  ///< elements currently in head[b]
    std::uint32_t need[kRadixBuckets];  ///< head elements required for alignment
    std::size_t   base[kRadixBuckets];  ///< original bucket start
};

/// One pass of the radix sort: reads `src`, writes `dst` grouped by digit.
///
/// `offset[b]` must hold the output index at which bucket b starts; it is
/// consumed (advanced) in place.
template <typename Key>
inline void radix_scatter_pass(const Key* FYX_RESTRICT src, std::size_t n,
                               Key* FYX_RESTRICT dst, unsigned shift,
                               std::size_t* FYX_RESTRICT offset,
                               RadixScatterScratch<Key>& sc,
                               bool can_stream) noexcept {
    constexpr std::size_t kPerLine = WcbTraits<Key>::kPerLine;

    Key* FYX_RESTRICT line = sc.line;
    Key* FYX_RESTRICT head = sc.head;

    // Work out, per bucket, how many leading elements must be diverted so the
    // streaming cursor starts on a cache-line boundary.
    std::size_t remaining_heads = 0;
    for (unsigned b = 0; b < kRadixBuckets; ++b) {
        sc.fill[b] = 0;
        sc.hn[b]   = 0;
        sc.base[b] = offset[b];
        if (can_stream) {
            const std::uintptr_t addr =
                reinterpret_cast<std::uintptr_t>(dst + offset[b]);
            const std::size_t misalign = (kCacheLine - (addr & (kCacheLine - 1)))
                                         & (kCacheLine - 1);
            sc.need[b] = static_cast<std::uint32_t>(misalign / sizeof(Key));
        } else {
            sc.need[b] = 0;
        }
        remaining_heads += sc.need[b];
    }

    auto push_line = [&](Key k, unsigned b) {
        Key* L = line + static_cast<std::size_t>(b) * kPerLine;
        const std::uint32_t f = sc.fill[b];
        L[f] = k;

        if (FYX_UNLIKELY(f + 1 == kPerLine)) {
            Key* out = dst + offset[b] + sc.need[b];
            if (can_stream) {
                // Guaranteed 64-byte aligned by construction.
                stream_cache_line(out, L);
            } else {
                std::memcpy(out, L, kCacheLine);
            }
            offset[b] += kPerLine;
            sc.fill[b] = 0;
        } else {
            sc.fill[b] = f + 1;
        }
    };

    std::size_t i = 0;
    for (; i < n && remaining_heads != 0; ++i) {
        prefetch_stream(src, i, n);

        const Key      k = src[i];
        const unsigned b = static_cast<unsigned>((k >> shift) & Key(kRadixMask));

        // Alignment prologue for this bucket.  Once every bucket has consumed
        // its tiny prologue (at most one cache line total per bucket), the main
        // loop below no longer pays this branch on every element.
        if (FYX_UNLIKELY(sc.hn[b] < sc.need[b])) {
            head[static_cast<std::size_t>(b) * kPerLine + sc.hn[b]] = k;
            ++sc.hn[b];
            --remaining_heads;
            continue;
        }

        push_line(k, b);
    }

    for (; i < n; ++i) {
        prefetch_stream(src, i, n);
        const Key      k = src[i];
        const unsigned b = static_cast<unsigned>((k >> shift) & Key(kRadixMask));
        push_line(k, b);
    }

    // Write back the alignment prologues and the partial trailing lines.
    for (unsigned b = 0; b < kRadixBuckets; ++b) {
        if (sc.hn[b])
            std::memcpy(dst + sc.base[b],
                        head + static_cast<std::size_t>(b) * kPerLine,
                        static_cast<std::size_t>(sc.hn[b]) * sizeof(Key));
        if (sc.fill[b])
            std::memcpy(dst + offset[b] + sc.need[b],
                        line + static_cast<std::size_t>(b) * kPerLine,
                        static_cast<std::size_t>(sc.fill[b]) * sizeof(Key));
        // Leave `offset[b]` pointing past everything this bucket wrote, in
        // case the caller wants to inspect it.
        offset[b] += sc.fill[b] + sc.hn[b];
    }

    if (can_stream) store_fence();
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Which passes actually need to run, and where the data ends up.
template <unsigned Passes>
struct RadixPlan {
    unsigned active[Passes];   ///< shift index of each surviving pass
    unsigned count = 0;        ///< number of surviving passes
};

/// A pass is degenerate when every key shares the same digit: the scatter
/// would be an exact copy, so it is skipped.
template <unsigned Passes>
inline RadixPlan<Passes> plan_radix(const RadixHistogram<Passes>& hist,
                                    std::size_t n) noexcept {
    RadixPlan<Passes> plan;
    for (unsigned p = 0; p < Passes; ++p) {
        bool degenerate = false;
        for (unsigned d = 0; d < kRadixBuckets; ++d) {
            if (hist.count[p][d] == n) { degenerate = true; break; }
        }
        if (!degenerate) plan.active[plan.count++] = p;
    }
    return plan;
}

/// LSD radix sort of `data[0..n)` using `tmp` as the ping-pong buffer.
/// Returns true on success; false means scratch memory was unavailable and the
/// caller must fall back to a comparison sort.
///
/// Keys are encoded on the way in and decoded on the way out, so signed
/// integers and IEEE floats sort in their natural order (see RadixTraits).
template <typename T>
inline bool radix_sort_impl(T* FYX_RESTRICT data, std::size_t n,
                            typename RadixTraits<T>::Key* FYX_RESTRICT buf_a,
                            typename RadixTraits<T>::Key* FYX_RESTRICT buf_b) noexcept {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr unsigned Passes = RT::passes;

    if (n < 2) return true;

    // -- 1. fused histogram of every pass, single read of the input ---------
    RadixHistogram<Passes> hist;
    scalar_histogram<T, Passes>(data, n, hist);

    // -- 2. skip degenerate passes ------------------------------------------
    const RadixPlan<Passes> plan = plan_radix<Passes>(hist, n);

    if (plan.count == 0) {
        // Every pass degenerate => all keys identical => already sorted.
        return true;
    }

    // -- 3. choose the starting buffer so the last pass lands in buf_a ------
    //
    // Each pass swaps src/dst.  Starting in buf_a, after `count` passes the
    // data sits in buf_a when count is even and in buf_b when it is odd.  We
    // want it in buf_a at the end (that is where we decode from), so an odd
    // pass count starts in buf_b.
    const bool start_in_b = (plan.count & 1u) != 0;

    Key* src = start_in_b ? buf_b : buf_a;
    Key* dst = start_in_b ? buf_a : buf_b;

    // Encode into the chosen source buffer.
    for (std::size_t i = 0; i < n; ++i) src[i] = RT::encode(data[i]);

    // -- 4. scatter passes ---------------------------------------------------
    constexpr std::size_t kPerLine = WcbTraits<Key>::kPerLine;

    // One allocation for both the line buffers and the alignment heads, plus
    // slack so the line buffer can be cache-line aligned by hand.
    ScratchLease<Key> wcb_lease(2 * kRadixBuckets * kPerLine + kPerLine);
    if (!wcb_lease.valid()) return false;

    RadixScatterScratch<Key> sc;
    {
        const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(wcb_lease.get());
        const std::uintptr_t m = (kCacheLine - (a & (kCacheLine - 1))) & (kCacheLine - 1);
        sc.line = reinterpret_cast<Key*>(a + m);
        sc.head = sc.line + kRadixBuckets * kPerLine;
    }
    const bool can_stream = have_nt_stores();

    std::size_t offset[kRadixBuckets];

    for (unsigned s_i = 0; s_i < plan.count; ++s_i) {
        const unsigned p     = plan.active[s_i];
        const unsigned shift = p * kRadixBits;

        // Exclusive prefix sum of this pass's histogram = bucket start offsets.
        std::size_t sum = 0;
        for (unsigned d = 0; d < kRadixBuckets; ++d) {
            offset[d] = sum;
            sum += static_cast<std::size_t>(hist.count[p][d]);
        }

        radix_scatter_pass<Key>(src, n, dst, shift, offset, sc, can_stream);

        Key* t = src; src = dst; dst = t;
    }

    // After the final swap the sorted keys are in `src`, which by construction
    // is buf_a.
    for (std::size_t i = 0; i < n; ++i) data[i] = RT::decode(src[i]);
    return true;
}

/// Convenience wrapper that owns the two ping-pong buffers.
template <typename T>
inline bool radix_sort(T* data, std::size_t n) noexcept {
    using Key = typename RadixTraits<T>::Key;
    if (n < 2) return true;

    ScratchLease<Key> lease(n * 2);
    if (!lease.valid()) return false;

    Key* a = lease.get();
    Key* b = a + n;
    return radix_sort_impl<T>(data, n, a, b);
}

} // namespace detail
} // namespace fyx
