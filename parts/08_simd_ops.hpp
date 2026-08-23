// ============================================================================
//  Section 8 -- Per-ISA "Ops" policies for the sorting network
//
//  Each policy sorts *unsigned keys* of a fixed width.  Signed and floating
//  point values reach here already encoded by RadixTraits, so only unsigned
//  min/max is ever needed.
// ============================================================================

namespace fyx {
namespace detail {

// ---------------------------------------------------------------------------
// Scalar policy -- 1 lane.  Always available; also the reference semantics.
// ---------------------------------------------------------------------------
template <typename KeyT>
struct ScalarOps {
    using Key  = KeyT;
    using Vec  = KeyT;
    using Mask = std::uint64_t;
    static constexpr unsigned kLanes = 1;

    FYX_FORCE_INLINE static Vec  load(const Key* p)            { return *p; }
    FYX_FORCE_INLINE static void store(Key* p, Vec v)          { *p = v; }
    FYX_FORCE_INLINE static Vec  splat(Key k)                  { return k; }
    FYX_FORCE_INLINE static Vec  min(Vec a, Vec b)             { return a < b ? a : b; }
    FYX_FORCE_INLINE static Vec  max(Vec a, Vec b)             { return a < b ? b : a; }
    FYX_FORCE_INLINE static Vec  load_partial(const Key* p, unsigned n, Key fill) {
        return n ? *p : fill;
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        if (n) *p = v;
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) { return v; }  // single lane
    FYX_FORCE_INLINE static Vec blend(Mask, Vec mn, Vec) { return mn; }
};

#if FYX_HAS_SSE42_CODE
FYX_ISA_BEGIN("sse4.2")
namespace isa_sse42 {
using namespace ::fyx::detail;
//<<<NETWORK_BODY>>>
// ---------------------------------------------------------------------------
// SSE4.2 -- 4 x uint32 or 2 x uint64
// ---------------------------------------------------------------------------

struct Sse42Ops32 {
    using Key  = std::uint32_t;
    using Vec  = __m128i;
    using Mask = std::uint32_t;
    static constexpr unsigned kLanes = 4;

    FYX_FORCE_INLINE static Vec load(const Key* p) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
    }
    FYX_FORCE_INLINE static Vec splat(Key k) {
        return _mm_set1_epi32(static_cast<int>(k));
    }
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) { return _mm_min_epu32(a, b); }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) { return _mm_max_epu32(a, b); }

    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        Key tmp[4] = {fill, fill, fill, fill};
        for (unsigned i = 0; i < n; ++i) tmp[i] = p[i];
        return load(tmp);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        Key tmp[4];
        store(tmp, v);
        for (unsigned i = 0; i < n; ++i) p[i] = tmp[i];
    }

    // Lane i <- lane i^J.  J is 1 or 2 for a 4-lane vector.
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1 || J == 2, "J must be 1 or 2 for 4 lanes");
        // _MM_SHUFFLE takes lanes in (3,2,1,0) order.
        if constexpr (J == 1) return _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
        else                  return _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
    }

    // No mask registers before AVX-512: build a 128-bit selector constant.
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        const __m128i sel = _mm_set_epi32(
            (keepmin & 8u) ? -1 : 0, (keepmin & 4u) ? -1 : 0,
            (keepmin & 2u) ? -1 : 0, (keepmin & 1u) ? -1 : 0);
        return _mm_blendv_epi8(mx, mn, sel);
    }
};

struct Sse42Ops64 {
    using Key  = std::uint64_t;
    using Vec  = __m128i;
    using Mask = std::uint32_t;
    static constexpr unsigned kLanes = 2;

    FYX_FORCE_INLINE static Vec load(const Key* p) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
    }
    FYX_FORCE_INLINE static Vec splat(Key k) {
        return _mm_set1_epi64x(static_cast<long long>(k));
    }
    // SSE has no unsigned 64-bit compare; bias by 2^63 and use the signed one.
    FYX_FORCE_INLINE static __m128i cmpgt_u64(Vec a, Vec b) {
        const __m128i bias = _mm_set1_epi64x(static_cast<long long>(0x8000000000000000ULL));
        return _mm_cmpgt_epi64(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias));
    }
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) {
        return _mm_blendv_epi8(a, b, cmpgt_u64(a, b));
    }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) {
        return _mm_blendv_epi8(b, a, cmpgt_u64(a, b));
    }
    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        Key tmp[2] = {fill, fill};
        for (unsigned i = 0; i < n; ++i) tmp[i] = p[i];
        return load(tmp);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        Key tmp[2];
        store(tmp, v);
        for (unsigned i = 0; i < n; ++i) p[i] = tmp[i];
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1, "J must be 1 for 2 lanes");
        return _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));   // swap the two 64-bit halves
    }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        const __m128i sel = _mm_set_epi64x((keepmin & 2u) ? -1LL : 0LL,
                                           (keepmin & 1u) ? -1LL : 0LL);
        return _mm_blendv_epi8(mx, mn, sel);
    }
};

} // namespace isa_sse42
FYX_ISA_END
#endif // FYX_HAS_SSE42_CODE

#if FYX_HAS_AVX2_CODE
FYX_ISA_BEGIN("avx2,fma")
namespace isa_avx2 {
using namespace ::fyx::detail;
//<<<NETWORK_BODY>>>
// ---------------------------------------------------------------------------
// AVX2 -- 8 x uint32 or 4 x uint64
// ---------------------------------------------------------------------------

struct Avx2Ops32 {
    using Key  = std::uint32_t;
    using Vec  = __m256i;
    using Mask = std::uint32_t;
    static constexpr unsigned kLanes = 8;

    FYX_FORCE_INLINE static Vec load(const Key* p) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    FYX_FORCE_INLINE static Vec splat(Key k) {
        return _mm256_set1_epi32(static_cast<int>(k));
    }
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) { return _mm256_min_epu32(a, b); }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) { return _mm256_max_epu32(a, b); }

    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        Key tmp[8];
        for (unsigned i = 0; i < 8; ++i) tmp[i] = (i < n) ? p[i] : fill;
        return load(tmp);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        Key tmp[8];
        store(tmp, v);
        for (unsigned i = 0; i < n; ++i) p[i] = tmp[i];
    }

    // J in {1,2,4}.  1 and 2 stay inside each 128-bit half (vpshufd); 4 swaps
    // the halves (vpermq on the 64-bit granularity).
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1 || J == 2 || J == 4, "J must be 1, 2 or 4 for 8 lanes");
        if constexpr (J == 1)      return _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
        else if constexpr (J == 2) return _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
        else                       return _mm256_permute2x128_si256(v, v, 0x01);
    }

    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        const __m256i sel = _mm256_set_epi32(
            (keepmin & 0x80u) ? -1 : 0, (keepmin & 0x40u) ? -1 : 0,
            (keepmin & 0x20u) ? -1 : 0, (keepmin & 0x10u) ? -1 : 0,
            (keepmin & 0x08u) ? -1 : 0, (keepmin & 0x04u) ? -1 : 0,
            (keepmin & 0x02u) ? -1 : 0, (keepmin & 0x01u) ? -1 : 0);
        return _mm256_blendv_epi8(mx, mn, sel);
    }
};

struct Avx2Ops64 {
    using Key  = std::uint64_t;
    using Vec  = __m256i;
    using Mask = std::uint32_t;
    static constexpr unsigned kLanes = 4;

    FYX_FORCE_INLINE static Vec load(const Key* p) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    FYX_FORCE_INLINE static Vec splat(Key k) {
        return _mm256_set1_epi64x(static_cast<long long>(k));
    }
    FYX_FORCE_INLINE static __m256i cmpgt_u64(Vec a, Vec b) {
        const __m256i bias = _mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ULL));
        return _mm256_cmpgt_epi64(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias));
    }
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) {
        return _mm256_blendv_epi8(a, b, cmpgt_u64(a, b));
    }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) {
        return _mm256_blendv_epi8(b, a, cmpgt_u64(a, b));
    }
    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        Key tmp[4];
        for (unsigned i = 0; i < 4; ++i) tmp[i] = (i < n) ? p[i] : fill;
        return load(tmp);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        Key tmp[4];
        store(tmp, v);
        for (unsigned i = 0; i < n; ++i) p[i] = tmp[i];
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1 || J == 2, "J must be 1 or 2 for 4 lanes");
        // vpermq selects 64-bit lanes: index i takes source lane (i^J).
        if constexpr (J == 1) return _mm256_permute4x64_epi64(v, _MM_SHUFFLE(2, 3, 0, 1));
        else                  return _mm256_permute4x64_epi64(v, _MM_SHUFFLE(1, 0, 3, 2));
    }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        const __m256i sel = _mm256_set_epi64x(
            (keepmin & 8u) ? -1LL : 0LL, (keepmin & 4u) ? -1LL : 0LL,
            (keepmin & 2u) ? -1LL : 0LL, (keepmin & 1u) ? -1LL : 0LL);
        return _mm256_blendv_epi8(mx, mn, sel);
    }
};

} // namespace isa_avx2
FYX_ISA_END
#endif // FYX_HAS_AVX2_CODE

#if FYX_HAS_AVX512_CODE
FYX_ISA_BEGIN("avx512f,avx512bw,avx512dq,avx512vl,avx512cd")
namespace isa_avx512 {
using namespace ::fyx::detail;
//<<<NETWORK_BODY>>>
// ---------------------------------------------------------------------------
// AVX-512 -- 16 x uint32 or 8 x uint64.  Native mask registers make the blend
// a single instruction and the permutation a single vpermd/vpermq.
// ---------------------------------------------------------------------------

struct Avx512Ops32 {
    using Key  = std::uint32_t;
    using Vec  = __m512i;
    using Mask = __mmask16;
    static constexpr unsigned kLanes = 16;

    FYX_FORCE_INLINE static Vec load(const Key* p) {
        return _mm512_loadu_si512(reinterpret_cast<const void*>(p));
    }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) {
        _mm512_storeu_si512(reinterpret_cast<void*>(p), v);
    }
    FYX_FORCE_INLINE static Vec splat(Key k) {
        return _mm512_set1_epi32(static_cast<int>(k));
    }
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) { return _mm512_min_epu32(a, b); }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) { return _mm512_max_epu32(a, b); }

    // Masked load/store: no scratch array, no branch.
    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        const __mmask16 m = static_cast<__mmask16>((1u << n) - 1u);
        return _mm512_mask_loadu_epi32(_mm512_set1_epi32(static_cast<int>(fill)), m, p);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        const __mmask16 m = static_cast<__mmask16>((1u << n) - 1u);
        _mm512_mask_storeu_epi32(p, m, v);
    }

    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1 || J == 2 || J == 4 || J == 8, "J must be 1,2,4,8 for 16 lanes");
        // A single vpermd with a compile-time index vector handles every J.
        const __m512i idx = _mm512_set_epi32(
            int(15u ^ J), int(14u ^ J), int(13u ^ J), int(12u ^ J),
            int(11u ^ J), int(10u ^ J), int( 9u ^ J), int( 8u ^ J),
            int( 7u ^ J), int( 6u ^ J), int( 5u ^ J), int( 4u ^ J),
            int( 3u ^ J), int( 2u ^ J), int( 1u ^ J), int( 0u ^ J));
        return _mm512_permutexvar_epi32(idx, v);
    }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        return _mm512_mask_blend_epi32(keepmin, mx, mn);
    }
};

struct Avx512Ops64 {
    using Key  = std::uint64_t;
    using Vec  = __m512i;
    using Mask = __mmask8;
    static constexpr unsigned kLanes = 8;

    FYX_FORCE_INLINE static Vec load(const Key* p) {
        return _mm512_loadu_si512(reinterpret_cast<const void*>(p));
    }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) {
        _mm512_storeu_si512(reinterpret_cast<void*>(p), v);
    }
    FYX_FORCE_INLINE static Vec splat(Key k) {
        return _mm512_set1_epi64(static_cast<long long>(k));
    }
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) { return _mm512_min_epu64(a, b); }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) { return _mm512_max_epu64(a, b); }

    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        const __mmask8 m = static_cast<__mmask8>((1u << n) - 1u);
        return _mm512_mask_loadu_epi64(_mm512_set1_epi64(static_cast<long long>(fill)), m, p);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        const __mmask8 m = static_cast<__mmask8>((1u << n) - 1u);
        _mm512_mask_storeu_epi64(p, m, v);
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1 || J == 2 || J == 4, "J must be 1,2,4 for 8 lanes");
        const __m512i idx = _mm512_set_epi64(
            static_cast<long long>(7u ^ J), static_cast<long long>(6u ^ J),
            static_cast<long long>(5u ^ J), static_cast<long long>(4u ^ J),
            static_cast<long long>(3u ^ J), static_cast<long long>(2u ^ J),
            static_cast<long long>(1u ^ J), static_cast<long long>(0u ^ J));
        return _mm512_permutexvar_epi64(idx, v);
    }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        return _mm512_mask_blend_epi64(keepmin, mx, mn);
    }
};

} // namespace isa_avx512
FYX_ISA_END
#endif // FYX_HAS_AVX512_CODE

#if FYX_HAS_NEON_CODE
namespace isa_neon {
using namespace ::fyx::detail;
//<<<NETWORK_BODY>>>
// ---------------------------------------------------------------------------
// ARM NEON -- 4 x uint32 or 2 x uint64
// ---------------------------------------------------------------------------

struct NeonOps32 {
    using Key  = std::uint32_t;
    using Vec  = uint32x4_t;
    using Mask = std::uint32_t;
    static constexpr unsigned kLanes = 4;

    FYX_FORCE_INLINE static Vec  load(const Key* p)   { return vld1q_u32(p); }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) { vst1q_u32(p, v); }
    FYX_FORCE_INLINE static Vec  splat(Key k)         { return vdupq_n_u32(k); }
    FYX_FORCE_INLINE static Vec  min(Vec a, Vec b)    { return vminq_u32(a, b); }
    FYX_FORCE_INLINE static Vec  max(Vec a, Vec b)    { return vmaxq_u32(a, b); }

    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        Key tmp[4];
        for (unsigned i = 0; i < 4; ++i) tmp[i] = (i < n) ? p[i] : fill;
        return load(tmp);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        Key tmp[4];
        store(tmp, v);
        for (unsigned i = 0; i < n; ++i) p[i] = tmp[i];
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1 || J == 2, "J must be 1 or 2 for 4 lanes");
        if constexpr (J == 1) {
            // Swap neighbours: [1,0,3,2]
            return vrev64q_u32(v);
        } else {
            // Swap 64-bit halves: [2,3,0,1]
            return vextq_u32(v, v, 2);
        }
    }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        const uint32_t s[4] = {
            (keepmin & 1u) ? ~0u : 0u, (keepmin & 2u) ? ~0u : 0u,
            (keepmin & 4u) ? ~0u : 0u, (keepmin & 8u) ? ~0u : 0u};
        return vbslq_u32(vld1q_u32(s), mn, mx);
    }
};

struct NeonOps64 {
    using Key  = std::uint64_t;
    using Vec  = uint64x2_t;
    using Mask = std::uint32_t;
    static constexpr unsigned kLanes = 2;

    FYX_FORCE_INLINE static Vec  load(const Key* p)   { return vld1q_u64(p); }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) { vst1q_u64(p, v); }
    FYX_FORCE_INLINE static Vec  splat(Key k)         { return vdupq_n_u64(k); }
    // NEON has no 64-bit integer min/max; synthesise from the compare mask.
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) {
#if FYX_ARCH_ARM64
        return vbslq_u64(vcgtq_u64(a, b), b, a);
#else
        // 32-bit NEON lacks vcgtq_u64; fall back to lane extraction.
        std::uint64_t x[2], y[2];
        vst1q_u64(x, a); vst1q_u64(y, b);
        std::uint64_t r[2] = {x[0] < y[0] ? x[0] : y[0], x[1] < y[1] ? x[1] : y[1]};
        return vld1q_u64(r);
#endif
    }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) {
#if FYX_ARCH_ARM64
        return vbslq_u64(vcgtq_u64(a, b), a, b);
#else
        std::uint64_t x[2], y[2];
        vst1q_u64(x, a); vst1q_u64(y, b);
        std::uint64_t r[2] = {x[0] < y[0] ? y[0] : x[0], x[1] < y[1] ? y[1] : x[1]};
        return vld1q_u64(r);
#endif
    }
    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        Key tmp[2] = {fill, fill};
        for (unsigned i = 0; i < n; ++i) tmp[i] = p[i];
        return load(tmp);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        Key tmp[2];
        store(tmp, v);
        for (unsigned i = 0; i < n; ++i) p[i] = tmp[i];
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1, "J must be 1 for 2 lanes");
        return vextq_u64(v, v, 1);
    }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        const std::uint64_t s[2] = {(keepmin & 1u) ? ~0ULL : 0ULL,
                                    (keepmin & 2u) ? ~0ULL : 0ULL};
        return vbslq_u64(vld1q_u64(s), mn, mx);
    }
};

} // namespace isa_neon
#endif // FYX_HAS_NEON_CODE

//<<<NETWORK_BODY>>>

// ---------------------------------------------------------------------------
// Runtime selection of the best network for a key width
// ---------------------------------------------------------------------------

/// Sorts n <= 64 unsigned 32-bit keys with the widest available ISA.
inline void network_sort_u32(std::uint32_t* keys, std::size_t n) {
    if (n < 2) return;
#if FYX_HAS_AVX512_CODE
    if (use_avx512()) { isa_avx512::network_sort_keys<isa_avx512::Avx512Ops32>(keys, n); return; }
#endif
#if FYX_HAS_AVX2_CODE
    if (use_avx2())   { isa_avx2::network_sort_keys<isa_avx2::Avx2Ops32>(keys, n);       return; }
#endif
#if FYX_HAS_SSE42_CODE
    if (use_sse42())  { isa_sse42::network_sort_keys<isa_sse42::Sse42Ops32>(keys, n);    return; }
#endif
#if FYX_HAS_NEON_CODE
    if (use_neon())   { isa_neon::network_sort_keys<isa_neon::NeonOps32>(keys, n);       return; }
#endif
    bitonic_pad_scalar<std::uint32_t>(keys, n);
}

/// Sorts n <= 64 unsigned 64-bit keys with the widest available ISA.
inline void network_sort_u64(std::uint64_t* keys, std::size_t n) {
    if (n < 2) return;
#if FYX_HAS_AVX512_CODE
    if (use_avx512()) { isa_avx512::network_sort_keys<isa_avx512::Avx512Ops64>(keys, n); return; }
#endif
#if FYX_HAS_AVX2_CODE
    if (use_avx2())   { isa_avx2::network_sort_keys<isa_avx2::Avx2Ops64>(keys, n);       return; }
#endif
#if FYX_HAS_SSE42_CODE
    if (use_sse42())  { isa_sse42::network_sort_keys<isa_sse42::Sse42Ops64>(keys, n);    return; }
#endif
#if FYX_HAS_NEON_CODE
    if (use_neon())   { isa_neon::network_sort_keys<isa_neon::NeonOps64>(keys, n);       return; }
#endif
    bitonic_pad_scalar<std::uint64_t>(keys, n);
}

/// Small-array entry point for any radix-encodable numeric type.
/// Encodes into a stack buffer, runs the network, decodes back.
template <typename T>
inline void small_sort_numeric(T* data, std::size_t n) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    static_assert(RT::supported, "small_sort_numeric requires a radix-encodable type");
    FYX_ASSERT_IMPL(n <= kNetworkMax);
    if (n < 2) return;

    Key buf[kNetworkMax];
    for (std::size_t i = 0; i < n; ++i) buf[i] = RT::encode(data[i]);

    if constexpr (sizeof(Key) == 4)      network_sort_u32(reinterpret_cast<std::uint32_t*>(buf), n);
    else if constexpr (sizeof(Key) == 8) network_sort_u64(reinterpret_cast<std::uint64_t*>(buf), n);
    else                                 bitonic_pad_scalar<Key>(buf, n);

    for (std::size_t i = 0; i < n; ++i) data[i] = RT::decode(buf[i]);
}

} // namespace detail
} // namespace fyx
