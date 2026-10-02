// ============================================================================
//  Section 10b -- AVX-512 vectorised quicksort
//
//  Why this exists
//  ---------------
//  `tools/dev/NOTES.md` closes the accounting on the LSD radix for
//  high-entropy numeric input: one count+scatter pass costs ~3.1 ns/elem and
//  does not respond to blocking, wider digits or a deeper write-combining
//  buffer, and 32 bits at <=13 bits per pass means three passes is the floor.
//  ~9.4 ns/elem of CPU work is therefore the floor for LSD radix on int32
//  here, against ~7.2 for a vectorised quicksort of the vqsort class.  Every
//  cheaper idea was measured and lost.  This is that vectorised quicksort.
//
//  Structure (the same one vqsort and x86-simd-sort use)
//  -----------------------------------------------------
//    * partition with `vpcompressd`/`vcompressps`: one compare gives a mask,
//      two masked compress-stores place the low and high halves at the two
//      ends of the free gap, and no element is ever written twice;
//    * four vectors are consumed per iteration and the side to read from is
//      chosen *branchlessly*.  That choice depends on how the data splits, so
//      a branch there mispredicts on roughly every other iteration -- at one
//      vector per iteration the misprediction alone costs more than the
//      partition (measured: 1 vector/branchy 0.0062 s vs 4 vectors/branchless
//      0.0042 s for 1M int32);
//    * the leaf (<=16 vectors) is a Batcher bitonic network run entirely in
//      registers, reusing the generic template from `parts/_network_body.inc`
//      with policies over the *native* type -- inside a quicksort the
//      comparisons are already in the value domain, so the encode and decode
//      passes the radix networks need are pure loss here.
//
//  Ordering semantics
//  ------------------
//  Integers compare natively.  Floating point uses `_CMP_GE_OQ`, which orders
//  everything except NaN and cannot separate -0 from +0, while the rest of the
//  library promises the radix total order (-0 before +0, NaN at one end).  The
//  entry point therefore screens the range first (a read-only SIMD pass) and
//  declines when it holds a NaN or a -0, leaving those inputs to the radix
//  path that already handles them.
// ============================================================================

#if FYX_HAS_AVX512_CODE

FYX_ISA_BEGIN("avx512f,avx512bw,avx512dq,avx512vl,avx512cd")
namespace fyx {
namespace detail {
namespace isa_avx512 {

// ---------------------------------------------------------------------------
// Per-type policies.  Everything is expressed in the native type: no encode.
// ---------------------------------------------------------------------------
// FYX_VQ_COMPRESS_TO_MEMORY=0 selects register-form compress + masked store:
// the memory form is microcoded on AMD Zen 4, but on Ice Lake-SP the register
// form measured ~15% slower, so the memory form stays the default.
FYX_FORCE_INLINE std::uint64_t bzhi_mask(unsigned c) {
    return c >= 64 ? ~0ull : ((1ull << c) - 1ull);
}


template <class T>
struct VOps;

template <>
struct VOps<std::int32_t> {
    using T    = std::int32_t;
    using reg  = __m512i;
    using mask = __mmask16;
    static constexpr int V = 16;
    FYX_FORCE_INLINE static reg  loadu(const T* p) { return _mm512_loadu_si512(reinterpret_cast<const void*>(p)); }
    FYX_FORCE_INLINE static void storeu(T* p, reg v) { _mm512_storeu_si512(reinterpret_cast<void*>(p), v); }
    FYX_FORCE_INLINE static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_epi32(k, p); }
    FYX_FORCE_INLINE static reg  mask_loadu(reg s, mask k, const T* p) { return _mm512_mask_loadu_epi32(s, k, p); }
    FYX_FORCE_INLINE static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_epi32(p, k, v); }
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) {
#if FYX_VQ_COMPRESS_TO_MEMORY
        _mm512_mask_compressstoreu_epi32(p, k, v);
#else
        _mm512_mask_storeu_epi32(p, static_cast<__mmask16>(bzhi_mask(popcount64(static_cast<std::uint64_t>(k)))), _mm512_maskz_compress_epi32(k, v));
#endif
    }
    FYX_FORCE_INLINE static mask ge(reg a, reg b) { return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLT); }
    FYX_FORCE_INLINE static mask gt(reg a, reg b) { return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLE); }
    FYX_FORCE_INLINE static reg  min(reg a, reg b) { return _mm512_min_epi32(a, b); }
    FYX_FORCE_INLINE static reg  max(reg a, reg b) { return _mm512_max_epi32(a, b); }
    FYX_FORCE_INLINE static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_epi32(s, k, a, b); }
    FYX_FORCE_INLINE static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_epi32(s, k, a, b); }
    FYX_FORCE_INLINE static reg  set1(T x) { return _mm512_set1_epi32(x); }
    FYX_FORCE_INLINE static T    reduce_min(reg v) { return _mm512_reduce_min_epi32(v); }
    FYX_FORCE_INLINE static T    reduce_max(reg v) { return _mm512_reduce_max_epi32(v); }
    FYX_FORCE_INLINE static T    hi() { return (std::numeric_limits<T>::max)(); }
    FYX_FORCE_INLINE static T    lo() { return (std::numeric_limits<T>::lowest)(); }
    template <unsigned J>
    FYX_FORCE_INLINE static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi32(
            int(15u ^ J), int(14u ^ J), int(13u ^ J), int(12u ^ J),
            int(11u ^ J), int(10u ^ J), int( 9u ^ J), int( 8u ^ J),
            int( 7u ^ J), int( 6u ^ J), int( 5u ^ J), int( 4u ^ J),
            int( 3u ^ J), int( 2u ^ J), int( 1u ^ J), int( 0u ^ J));
        return _mm512_permutexvar_epi32(idx, v);
    }
    FYX_FORCE_INLINE static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_epi32(keepmin, mx, mn); }
};

template <>
struct VOps<std::uint32_t> {
    using T    = std::uint32_t;
    using reg  = __m512i;
    using mask = __mmask16;
    static constexpr int V = 16;
    FYX_FORCE_INLINE static reg  loadu(const T* p) { return _mm512_loadu_si512(reinterpret_cast<const void*>(p)); }
    FYX_FORCE_INLINE static void storeu(T* p, reg v) { _mm512_storeu_si512(reinterpret_cast<void*>(p), v); }
    FYX_FORCE_INLINE static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_epi32(k, p); }
    FYX_FORCE_INLINE static reg  mask_loadu(reg s, mask k, const T* p) { return _mm512_mask_loadu_epi32(s, k, p); }
    FYX_FORCE_INLINE static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_epi32(p, k, v); }
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) {
#if FYX_VQ_COMPRESS_TO_MEMORY
        _mm512_mask_compressstoreu_epi32(p, k, v);
#else
        _mm512_mask_storeu_epi32(p, static_cast<__mmask16>(bzhi_mask(popcount64(static_cast<std::uint64_t>(k)))), _mm512_maskz_compress_epi32(k, v));
#endif
    }
    FYX_FORCE_INLINE static mask ge(reg a, reg b) { return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_NLT); }
    FYX_FORCE_INLINE static mask gt(reg a, reg b) { return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_NLE); }
    FYX_FORCE_INLINE static reg  min(reg a, reg b) { return _mm512_min_epu32(a, b); }
    FYX_FORCE_INLINE static reg  max(reg a, reg b) { return _mm512_max_epu32(a, b); }
    FYX_FORCE_INLINE static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_epu32(s, k, a, b); }
    FYX_FORCE_INLINE static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_epu32(s, k, a, b); }
    FYX_FORCE_INLINE static reg  set1(T x) { return _mm512_set1_epi32(static_cast<int>(x)); }
    FYX_FORCE_INLINE static T    reduce_min(reg v) { return static_cast<T>(_mm512_reduce_min_epu32(v)); }
    FYX_FORCE_INLINE static T    reduce_max(reg v) { return static_cast<T>(_mm512_reduce_max_epu32(v)); }
    FYX_FORCE_INLINE static T    hi() { return (std::numeric_limits<T>::max)(); }
    FYX_FORCE_INLINE static T    lo() { return (std::numeric_limits<T>::lowest)(); }
    template <unsigned J>
    FYX_FORCE_INLINE static reg permute_xor(reg v) { return VOps<std::int32_t>::permute_xor<J>(v); }
    FYX_FORCE_INLINE static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_epi32(keepmin, mx, mn); }
};

template <>
struct VOps<std::int64_t> {
    using T    = std::int64_t;
    using reg  = __m512i;
    using mask = __mmask8;
    static constexpr int V = 8;
    FYX_FORCE_INLINE static reg  loadu(const T* p) { return _mm512_loadu_si512(reinterpret_cast<const void*>(p)); }
    FYX_FORCE_INLINE static void storeu(T* p, reg v) { _mm512_storeu_si512(reinterpret_cast<void*>(p), v); }
    FYX_FORCE_INLINE static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_epi64(k, p); }
    FYX_FORCE_INLINE static reg  mask_loadu(reg s, mask k, const T* p) { return _mm512_mask_loadu_epi64(s, k, p); }
    FYX_FORCE_INLINE static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_epi64(p, k, v); }
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) {
#if FYX_VQ_COMPRESS_TO_MEMORY
        _mm512_mask_compressstoreu_epi64(p, k, v);
#else
        _mm512_mask_storeu_epi64(p, static_cast<__mmask8>(bzhi_mask(popcount64(static_cast<std::uint64_t>(k)))), _mm512_maskz_compress_epi64(k, v));
#endif
    }
    FYX_FORCE_INLINE static mask ge(reg a, reg b) { return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_NLT); }
    FYX_FORCE_INLINE static mask gt(reg a, reg b) { return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_NLE); }
    FYX_FORCE_INLINE static reg  min(reg a, reg b) { return _mm512_min_epi64(a, b); }
    FYX_FORCE_INLINE static reg  max(reg a, reg b) { return _mm512_max_epi64(a, b); }
    FYX_FORCE_INLINE static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_epi64(s, k, a, b); }
    FYX_FORCE_INLINE static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_epi64(s, k, a, b); }
    FYX_FORCE_INLINE static reg  set1(T x) { return _mm512_set1_epi64(x); }
    FYX_FORCE_INLINE static T    reduce_min(reg v) { return _mm512_reduce_min_epi64(v); }
    FYX_FORCE_INLINE static T    reduce_max(reg v) { return _mm512_reduce_max_epi64(v); }
    FYX_FORCE_INLINE static T    hi() { return (std::numeric_limits<T>::max)(); }
    FYX_FORCE_INLINE static T    lo() { return (std::numeric_limits<T>::lowest)(); }
    template <unsigned J>
    FYX_FORCE_INLINE static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi64(
            static_cast<long long>(7u ^ J), static_cast<long long>(6u ^ J),
            static_cast<long long>(5u ^ J), static_cast<long long>(4u ^ J),
            static_cast<long long>(3u ^ J), static_cast<long long>(2u ^ J),
            static_cast<long long>(1u ^ J), static_cast<long long>(0u ^ J));
        return _mm512_permutexvar_epi64(idx, v);
    }
    FYX_FORCE_INLINE static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_epi64(keepmin, mx, mn); }
};

template <>
struct VOps<std::uint64_t> {
    using T    = std::uint64_t;
    using reg  = __m512i;
    using mask = __mmask8;
    static constexpr int V = 8;
    FYX_FORCE_INLINE static reg  loadu(const T* p) { return _mm512_loadu_si512(reinterpret_cast<const void*>(p)); }
    FYX_FORCE_INLINE static void storeu(T* p, reg v) { _mm512_storeu_si512(reinterpret_cast<void*>(p), v); }
    FYX_FORCE_INLINE static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_epi64(k, p); }
    FYX_FORCE_INLINE static reg  mask_loadu(reg s, mask k, const T* p) { return _mm512_mask_loadu_epi64(s, k, p); }
    FYX_FORCE_INLINE static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_epi64(p, k, v); }
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) {
#if FYX_VQ_COMPRESS_TO_MEMORY
        _mm512_mask_compressstoreu_epi64(p, k, v);
#else
        _mm512_mask_storeu_epi64(p, static_cast<__mmask8>(bzhi_mask(popcount64(static_cast<std::uint64_t>(k)))), _mm512_maskz_compress_epi64(k, v));
#endif
    }
    FYX_FORCE_INLINE static mask ge(reg a, reg b) { return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_NLT); }
    FYX_FORCE_INLINE static mask gt(reg a, reg b) { return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_NLE); }
    FYX_FORCE_INLINE static reg  min(reg a, reg b) { return _mm512_min_epu64(a, b); }
    FYX_FORCE_INLINE static reg  max(reg a, reg b) { return _mm512_max_epu64(a, b); }
    FYX_FORCE_INLINE static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_epu64(s, k, a, b); }
    FYX_FORCE_INLINE static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_epu64(s, k, a, b); }
    FYX_FORCE_INLINE static reg  set1(T x) { return _mm512_set1_epi64(static_cast<long long>(x)); }
    FYX_FORCE_INLINE static T    reduce_min(reg v) { return _mm512_reduce_min_epu64(v); }
    FYX_FORCE_INLINE static T    reduce_max(reg v) { return _mm512_reduce_max_epu64(v); }
    FYX_FORCE_INLINE static T    hi() { return (std::numeric_limits<T>::max)(); }
    FYX_FORCE_INLINE static T    lo() { return (std::numeric_limits<T>::lowest)(); }
    template <unsigned J>
    FYX_FORCE_INLINE static reg permute_xor(reg v) { return VOps<std::int64_t>::permute_xor<J>(v); }
    FYX_FORCE_INLINE static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_epi64(keepmin, mx, mn); }
};

template <>
struct VOps<float> {
    using T    = float;
    using reg  = __m512;
    using mask = __mmask16;
    static constexpr int V = 16;
    FYX_FORCE_INLINE static reg  loadu(const T* p) { return _mm512_loadu_ps(p); }
    FYX_FORCE_INLINE static void storeu(T* p, reg v) { _mm512_storeu_ps(p, v); }
    FYX_FORCE_INLINE static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_ps(k, p); }
    FYX_FORCE_INLINE static reg  mask_loadu(reg s, mask k, const T* p) { return _mm512_mask_loadu_ps(s, k, p); }
    FYX_FORCE_INLINE static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_ps(p, k, v); }
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) {
#if FYX_VQ_COMPRESS_TO_MEMORY
        _mm512_mask_compressstoreu_ps(p, k, v);
#else
        _mm512_mask_storeu_ps(p, static_cast<__mmask16>(bzhi_mask(popcount64(static_cast<std::uint64_t>(k)))), _mm512_maskz_compress_ps(k, v));
#endif
    }
    FYX_FORCE_INLINE static mask ge(reg a, reg b) { return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ); }
    FYX_FORCE_INLINE static mask gt(reg a, reg b) { return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ); }
    FYX_FORCE_INLINE static reg  min(reg a, reg b) { return _mm512_min_ps(a, b); }
    FYX_FORCE_INLINE static reg  max(reg a, reg b) { return _mm512_max_ps(a, b); }
    FYX_FORCE_INLINE static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_ps(s, k, a, b); }
    FYX_FORCE_INLINE static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_ps(s, k, a, b); }
    FYX_FORCE_INLINE static reg  set1(T x) { return _mm512_set1_ps(x); }
    FYX_FORCE_INLINE static T    reduce_min(reg v) { return _mm512_reduce_min_ps(v); }
    FYX_FORCE_INLINE static T    reduce_max(reg v) { return _mm512_reduce_max_ps(v); }
    FYX_FORCE_INLINE static T    hi() { return (std::numeric_limits<T>::infinity)(); }
    FYX_FORCE_INLINE static T    lo() { return -(std::numeric_limits<T>::infinity)(); }
    template <unsigned J>
    FYX_FORCE_INLINE static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi32(
            int(15u ^ J), int(14u ^ J), int(13u ^ J), int(12u ^ J),
            int(11u ^ J), int(10u ^ J), int( 9u ^ J), int( 8u ^ J),
            int( 7u ^ J), int( 6u ^ J), int( 5u ^ J), int( 4u ^ J),
            int( 3u ^ J), int( 2u ^ J), int( 1u ^ J), int( 0u ^ J));
        return _mm512_permutexvar_ps(idx, v);
    }
    FYX_FORCE_INLINE static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_ps(keepmin, mx, mn); }
};

template <>
struct VOps<double> {
    using T    = double;
    using reg  = __m512d;
    using mask = __mmask8;
    static constexpr int V = 8;
    FYX_FORCE_INLINE static reg  loadu(const T* p) { return _mm512_loadu_pd(p); }
    FYX_FORCE_INLINE static void storeu(T* p, reg v) { _mm512_storeu_pd(p, v); }
    FYX_FORCE_INLINE static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_pd(k, p); }
    FYX_FORCE_INLINE static reg  mask_loadu(reg s, mask k, const T* p) { return _mm512_mask_loadu_pd(s, k, p); }
    FYX_FORCE_INLINE static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_pd(p, k, v); }
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) {
#if FYX_VQ_COMPRESS_TO_MEMORY
        _mm512_mask_compressstoreu_pd(p, k, v);
#else
        _mm512_mask_storeu_pd(p, static_cast<__mmask8>(bzhi_mask(popcount64(static_cast<std::uint64_t>(k)))), _mm512_maskz_compress_pd(k, v));
#endif
    }
    FYX_FORCE_INLINE static mask ge(reg a, reg b) { return _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ); }
    FYX_FORCE_INLINE static mask gt(reg a, reg b) { return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ); }
    FYX_FORCE_INLINE static reg  min(reg a, reg b) { return _mm512_min_pd(a, b); }
    FYX_FORCE_INLINE static reg  max(reg a, reg b) { return _mm512_max_pd(a, b); }
    FYX_FORCE_INLINE static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_pd(s, k, a, b); }
    FYX_FORCE_INLINE static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_pd(s, k, a, b); }
    FYX_FORCE_INLINE static reg  set1(T x) { return _mm512_set1_pd(x); }
    FYX_FORCE_INLINE static T    reduce_min(reg v) { return _mm512_reduce_min_pd(v); }
    FYX_FORCE_INLINE static T    reduce_max(reg v) { return _mm512_reduce_max_pd(v); }
    FYX_FORCE_INLINE static T    hi() { return (std::numeric_limits<T>::infinity)(); }
    FYX_FORCE_INLINE static T    lo() { return -(std::numeric_limits<T>::infinity)(); }
    template <unsigned J>
    FYX_FORCE_INLINE static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi64(
            static_cast<long long>(7u ^ J), static_cast<long long>(6u ^ J),
            static_cast<long long>(5u ^ J), static_cast<long long>(4u ^ J),
            static_cast<long long>(3u ^ J), static_cast<long long>(2u ^ J),
            static_cast<long long>(1u ^ J), static_cast<long long>(0u ^ J));
        return _mm512_permutexvar_pd(idx, v);
    }
    FYX_FORCE_INLINE static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_pd(keepmin, mx, mn); }
};

// ---------------------------------------------------------------------------
// Leaf: the generic bitonic network over native values.
// ---------------------------------------------------------------------------
template <class T>
struct VNetOps {
    using P    = VOps<T>;
    using Key  = T;
    using Vec  = typename P::reg;
    using Mask = typename P::mask;
    static constexpr unsigned kLanes = static_cast<unsigned>(P::V);

    FYX_FORCE_INLINE static Vec  load(const Key* p)   { return P::loadu(p); }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) { P::storeu(p, v); }
    FYX_FORCE_INLINE static Vec  splat(Key k)         { return P::set1(k); }
    FYX_FORCE_INLINE static Vec  min(Vec a, Vec b)    { return P::min(a, b); }
    FYX_FORCE_INLINE static Vec  max(Vec a, Vec b)    { return P::max(a, b); }
    FYX_FORCE_INLINE static Vec  load_partial(const Key* p, unsigned n, Key fill) {
        const Mask m = static_cast<Mask>((1ull << n) - 1ull);
        return P::mask_loadu(P::set1(fill), m, p);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        P::mask_storeu(p, static_cast<Mask>((1ull << n) - 1ull), v);
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) { return P::template permute_xor<J>(v); }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) { return P::blend(keepmin, mn, mx); }
};

/// Like `network_sort_v`, but the padding sentinel comes from the policy (a
/// float network must pad with +inf, not FLT_MAX) and the leaf may hold up to
/// 16 vectors rather than 64 elements.
template <class T, unsigned NV>
FYX_FORCE_INLINE void vnet_sort_v(T* keys, std::size_t n) {
    using Ops = VNetOps<T>;
    using Vec = typename Ops::Vec;
    constexpr unsigned L = Ops::kLanes;
    const T sentinel = VOps<T>::hi();
    Vec v[NV];
    for (unsigned i = 0; i < NV; ++i) {
        const std::size_t off = static_cast<std::size_t>(i) * L;
        if (off + L <= n)  v[i] = Ops::load(keys + off);
        else if (off < n)  v[i] = Ops::load_partial(keys + off, static_cast<unsigned>(n - off), sentinel);
        else               v[i] = Ops::splat(sentinel);
    }
    BitonicVec<Ops, NV>::run(v);
    for (unsigned i = 0; i < NV; ++i) {
        const std::size_t off = static_cast<std::size_t>(i) * L;
        if (off + L <= n)  Ops::store(keys + off, v[i]);
        else if (off < n)  Ops::store_partial(keys + off, v[i], static_cast<unsigned>(n - off));
    }
}

// ---------------------------------------------------------------------------
// Column-network leaf (the layout vqsort uses): V registers x L lanes.
//
//   1. a size-optimal sorting network across the V registers sorts every
//      lane ("column") with min/max only -- no shuffles at all;
//   2. log2(L) merge levels join lane groups pairwise.  Each level is one
//      reversed compare (two lane permutes per register pair), lane
//      half-cleaners (one permute per register) and register half-cleaners
//      (plain min/max);
//   3. one in-register transpose restores memory order.
//
// For 16 x int32 (256 keys) that is ~1000 vector ops against ~1800 for the
// row-wise Batcher network above, whose every intra-vector stage pays a
// permute, a min, a max and a blend.
// ---------------------------------------------------------------------------
template <unsigned V> struct ColNet;
template <> struct ColNet<4> {
    static constexpr unsigned n = 5;
    static constexpr unsigned char a[n] = {0, 2, 0, 1, 1};
    static constexpr unsigned char b[n] = {1, 3, 2, 3, 2};
};
template <> struct ColNet<8> {
    static constexpr unsigned n = 19;
    static constexpr unsigned char a[n] = {0, 1, 4, 5, 0, 1, 2, 3, 0, 2, 4, 6, 2, 3, 1, 3, 1, 3, 5};
    static constexpr unsigned char b[n] = {2, 3, 6, 7, 4, 5, 6, 7, 1, 3, 5, 7, 4, 5, 4, 6, 2, 4, 6};
};
template <> struct ColNet<16> {   // 60 comparators, depth 10
    static constexpr unsigned n = 60;
    static constexpr unsigned char a[n] = {
        0, 1, 2, 3, 4, 5, 7, 9,   0, 1, 2, 3, 6, 8, 10, 11,   0, 2, 4, 6, 7, 10, 12, 14,
        0, 1, 4, 5, 6, 8, 12, 13,  1, 3, 4, 5, 8, 9, 13,      1, 2, 5, 7, 9, 11,
        2, 3, 9, 11,               3, 6, 7, 10,               3, 5, 7, 9, 11,   6, 8};
    static constexpr unsigned char b[n] = {
        13, 12, 15, 14, 8, 6, 11, 10,   5, 7, 9, 4, 13, 14, 15, 12,   1, 3, 5, 8, 9, 11, 13, 15,
        2, 3, 10, 11, 7, 9, 14, 15,      2, 12, 6, 7, 10, 11, 14,      4, 6, 8, 10, 13, 14,
        4, 6, 12, 13,                    5, 8, 9, 12,                  4, 6, 8, 10, 12,  7, 9};
};

template <> struct ColNet<32> {   // Batcher odd-even merge sort, 191 comparators
    static constexpr unsigned n = 191;
    static constexpr unsigned char a[n] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 1, 5, 9, 13, 17, 21, 25, 29, 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 2, 3, 10, 11, 18, 19, 26, 27, 1, 3, 5, 9, 11, 13, 17, 19, 21, 25, 27, 29, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 4, 5, 6, 7, 20, 21, 22, 23, 2, 3, 6, 7, 10, 11, 18, 19, 22, 23, 26, 27, 1, 3, 5, 7, 9, 11, 13, 17, 19, 21, 23, 25, 27, 29, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29};
    static constexpr unsigned char b[n] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 2, 6, 10, 14, 18, 22, 26, 30, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 4, 5, 12, 13, 20, 21, 28, 29, 2, 4, 6, 10, 12, 14, 18, 20, 22, 26, 28, 30, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 8, 9, 10, 11, 24, 25, 26, 27, 4, 5, 8, 9, 12, 13, 20, 21, 24, 25, 28, 29, 2, 4, 6, 8, 10, 12, 14, 18, 20, 22, 24, 26, 28, 30, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
};

template <class T>
FYX_FORCE_INLINE typename VOps<T>::reg vp2v(typename VOps<T>::reg a, __m512i idx,
                                            typename VOps<T>::reg b) {
    if constexpr (std::is_same<T, float>::value)       return _mm512_permutex2var_ps(a, idx, b);
    else if constexpr (std::is_same<T, double>::value) return _mm512_permutex2var_pd(a, idx, b);
    else if constexpr (sizeof(T) == 4)                 return _mm512_permutex2var_epi32(a, idx, b);
    else                                               return _mm512_permutex2var_epi64(a, idx, b);
}

/// Index vectors for one transpose round at block size B over L lanes.
template <unsigned L, unsigned B, bool Hi>
struct VTransIdx {
    using I = typename std::conditional<L == 16, std::int32_t, std::int64_t>::type;
    struct Tab { alignas(64) I v[L]; };
    static constexpr Tab make() {
        Tab t{};
        for (unsigned l = 0; l < L; ++l) {
            const bool up = (l & B) != 0;
            t.v[l] = static_cast<I>(Hi ? (up ? L + l : l + B) : (up ? L + (l - B) : l));
        }
        return t;
    }
    static constexpr Tab tab = make();
    FYX_FORCE_INLINE static __m512i load() {
        return _mm512_load_si512(reinterpret_cast<const void*>(tab.v));
    }
};

template <class T>
struct VColLeaf {
    using P   = VOps<T>;
    using reg = typename P::reg;
    using M   = typename P::mask;
    static constexpr unsigned L = static_cast<unsigned>(P::V);

    FYX_FORCE_INLINE static void ce(reg& x, reg& y) {
        const reg mn = P::min(x, y);
        y = P::max(x, y);
        x = mn;
    }
    static constexpr std::uint64_t low_lanes(unsigned bit) {
        std::uint64_t m = 0;
        for (unsigned l = 0; l < L; ++l) if (((l >> bit) & 1u) == 0) m |= 1ull << l;
        return m;
    }
    template <unsigned V, std::size_t... I>
    FYX_FORCE_INLINE static void columns(reg* v, std::index_sequence<I...>) {
        (ce(v[ColNet<V>::a[I]], v[ColNet<V>::b[I]]), ...);
    }
    template <unsigned V, unsigned K>
    FYX_FORCE_INLINE static void reversed_compare(reg* v) {
        constexpr unsigned Mx = (2u << K) - 1u;
        const M low = static_cast<M>(low_lanes(K));
        rc_each<V, Mx>(v, low, std::make_index_sequence<V / 2>{});
    }
    template <unsigned V, unsigned Mx, unsigned R>
    FYX_FORCE_INLINE static void rc_one(reg* v, M low) {
        const reg x = v[R];
        const reg s = P::template permute_xor<Mx>(v[V - 1 - R]);
        const reg mn = P::min(x, s);
        const reg mx = P::max(x, s);
        v[R]         = P::blend(low, mn, mx);
        v[V - 1 - R] = P::template permute_xor<Mx>(P::blend(low, mx, mn));
    }
    template <unsigned V, unsigned Mx, std::size_t... R>
    FYX_FORCE_INLINE static void rc_each(reg* v, M low, std::index_sequence<R...>) {
        (rc_one<V, Mx, static_cast<unsigned>(R)>(v, low), ...);
    }
    template <unsigned V, unsigned Bit>
    FYX_FORCE_INLINE static void lane_clean(reg* v) {
        const M low = static_cast<M>(low_lanes(Bit));
        lc_each<Bit>(v, low, std::make_index_sequence<V>{});
    }
    template <unsigned Bit, std::size_t... R>
    FYX_FORCE_INLINE static void lc_each(reg* v, M low, std::index_sequence<R...>) {
        ((v[R] = lc_one<Bit>(v[R], low)), ...);
    }
    template <unsigned Bit>
    FYX_FORCE_INLINE static reg lc_one(reg x, M low) {
        const reg q = P::template permute_xor<(1u << Bit)>(x);
        return P::blend(low, P::min(x, q), P::max(x, q));
    }
    template <unsigned V, unsigned K, unsigned Bit>
    FYX_FORCE_INLINE static void lane_cleans(reg* v) {
        if constexpr (Bit < K) {
            lane_cleans<V, K, Bit + 1>(v);       // larger bits first
            lane_clean<V, Bit>(v);
        }
    }
    template <unsigned V, unsigned J>
    FYX_FORCE_INLINE static void reg_cleans(reg* v) {
        if constexpr (J > 0) {
            rcl_each<J>(v, std::make_index_sequence<V>{});
            reg_cleans<V, J / 2>(v);
        }
    }
    template <unsigned J, std::size_t... R>
    FYX_FORCE_INLINE static void rcl_each(reg* v, std::index_sequence<R...>) {
        ((((R & J) == 0) ? ce(v[R], v[R | J]) : void()), ...);
    }
    template <unsigned V, unsigned K>
    FYX_FORCE_INLINE static void levels(reg* v) {
        if constexpr ((1u << K) < L) {
            reversed_compare<V, K>(v);
            lane_cleans<V, K, 0>(v);
            reg_cleans<V, V / 2>(v);
            levels<V, K + 1>(v);
        }
    }
    template <unsigned B, unsigned N = L>
    FYX_FORCE_INLINE static void transpose_round(reg* v) {
        if constexpr (B > 0) {
            const __m512i lo = VTransIdx<L, B, false>::load();
            const __m512i hi = VTransIdx<L, B, true>::load();
            tr_each<B>(v, lo, hi, std::make_index_sequence<N>{});
            transpose_round<B / 2, N>(v);
        }
    }

    template <unsigned B, std::size_t... R>
    FYX_FORCE_INLINE static void tr_each(reg* v, __m512i lo, __m512i hi, std::index_sequence<R...>) {
        ((((R & B) == 0) ? tr_one<B, static_cast<unsigned>(R)>(v, lo, hi) : void()), ...);
    }
    template <unsigned B, unsigned R>
    FYX_FORCE_INLINE static void tr_one(reg* v, __m512i lo, __m512i hi) {
        {
            const reg x = v[R], y = v[R + B];
            v[R]     = vp2v<T>(x, lo, y);
            v[R + B] = vp2v<T>(x, hi, y);
        }
    }

    FYX_FORCE_INLINE static reg load_one(const T* keys, std::size_t n, std::size_t off) {
        if (off + L <= n) return P::loadu(keys + off);
        const reg s = P::set1(P::hi());
        if (off < n) return P::mask_loadu(s, static_cast<M>((1ull << (n - off)) - 1ull), keys + off);
        return s;
    }
    FYX_FORCE_INLINE static void store_one(T* keys, std::size_t n, std::size_t off, reg x) {
        if (off + L <= n)  P::storeu(keys + off, x);
        else if (off < n)  P::mask_storeu(keys + off, static_cast<M>((1ull << (n - off)) - 1ull), x);
    }
    template <std::size_t... I>
    FYX_FORCE_INLINE static void load_all(reg* v, const T* keys, std::size_t n, std::index_sequence<I...>) {
        ((v[I] = load_one(keys, n, I * L)), ...);
    }
    // Logical order is lane-major: key (lane l, register r) sits at l*V + r.
    // After transposing each L x L register block c, its register l holds
    // output vector l*C + c.
    template <unsigned C, std::size_t... I>
    FYX_FORCE_INLINE static void store_all(const reg* v, T* keys, std::size_t n, std::index_sequence<I...>) {
        (store_one(keys, n, ((I % L) * C + I / L) * L, v[I]), ...);
    }

    // V == L/2: each register's halves are transposed as two independent
    // V x V blocks, then registers (2i, 2i+1) are recombined into output
    // vectors i and i + V/2.
    template <unsigned V, std::size_t... I>
    FYX_FORCE_INLINE static void store_half(const reg* v, T* keys, std::size_t n, std::index_sequence<I...>) {
        (store_one(keys, n, ((I % 2 == 0) ? I / 2 : I / 2 + V / 2) * L, v[I]), ...);
    }
    template <unsigned R, std::size_t... I>
    FYX_FORCE_INLINE static void pair_each(reg* v, __m512i lo, __m512i hi, std::index_sequence<I...>) {
        ((tr_pair(v[2 * I], v[2 * I + 1], lo, hi)), ...);
    }
    FYX_FORCE_INLINE static void tr_pair(reg& a, reg& b, __m512i lo, __m512i hi) {
        const reg x = a, y = b;
        a = vp2v<T>(x, lo, y);
        b = vp2v<T>(x, hi, y);
    }

    /// Sorts n <= V*L keys; V is a multiple of L, or exactly L/2.
    template <unsigned V>
    FYX_FORCE_INLINE static void sort(T* keys, std::size_t n) {
        static_assert(V % L == 0 || 2 * V == L || (4 * V == L && V == 4), "unsupported column leaf shape");
        reg v[V];
        load_all(v, keys, n, std::make_index_sequence<V>{});
        columns<V>(v, std::make_index_sequence<ColNet<V>::n>{});
        levels<V, 0>(v);
        if constexpr (4 * V == L) {
            // 4 registers x 16 lanes: transpose the 4x4 blocks inside each
            // lane quarter, then the 4x4 matrix of quarters.
            transpose_round<V / 2, V>(v);
            tr_each<2>(v, VTransIdx<L, 8, false>::load(), VTransIdx<L, 8, true>::load(),
                       std::make_index_sequence<V>{});
            tr_each<1>(v, VTransIdx<L, 4, false>::load(), VTransIdx<L, 4, true>::load(),
                       std::make_index_sequence<V>{});
            store_all<1>(v, keys, n, std::make_index_sequence<V>{});
            return;
        }
        if constexpr (2 * V == L) {
            transpose_round<V / 2, V>(v);
            pair_each<0>(v, VTransIdx<L, L / 2, false>::load(), VTransIdx<L, L / 2, true>::load(),
                         std::make_index_sequence<V / 2>{});
            store_half<V>(v, keys, n, std::make_index_sequence<V>{});
            return;
        }
        transpose_round<L / 2>(v);
        if constexpr (V > L)     transpose_round<L / 2>(v + L);
        if constexpr (V > 2 * L) transpose_round<L / 2>(v + 2 * L);
        if constexpr (V > 3 * L) transpose_round<L / 2>(v + 3 * L);
        static_assert(V <= 4 * L, "column leaf supports at most four blocks");
        store_all<V / L>(v, keys, n, std::make_index_sequence<V>{});
    }
};

#ifndef FYX_VQ_COLUMN_LEAF
#  define FYX_VQ_COLUMN_LEAF 1
#endif

/// Leaf sort for n <= 16 vectors' worth of elements.
template <class T>
inline void vnet_sort(T* a, std::size_t n) {
    constexpr unsigned L = static_cast<unsigned>(VOps<T>::V);
    if (n < 2) return;
    const std::size_t vecs = (static_cast<std::size_t>(next_pow2(n)) + L - 1) / L;
    switch (vecs) {
        case 1:  vnet_sort_v<T, 1>(a, n);  return;
        case 2:  vnet_sort_v<T, 2>(a, n);  return;
        case 4:
#if FYX_VQ_COLUMN_LEAF
            VColLeaf<T>::template sort<4>(a, n); return;
#endif
            vnet_sort_v<T, 4>(a, n);  return;
        case 8:
#if FYX_VQ_COLUMN_LEAF
            VColLeaf<T>::template sort<8>(a, n); return;
#endif
            vnet_sort_v<T, 8>(a, n);  return;
        case 16:
#if FYX_VQ_COLUMN_LEAF
            VColLeaf<T>::template sort<16>(a, n); return;
#endif
            vnet_sort_v<T, 16>(a, n); return;
#if FYX_VQ_COLUMN_LEAF
        case 32: VColLeaf<T>::template sort<32>(a, n); return;
#endif
        default: break;
    }
    pdqsort(a, a + n, std::less<T>());
}

// ---------------------------------------------------------------------------
// Partition
// ---------------------------------------------------------------------------
/// The high side takes `>= pivot` normally and `> pivot` in the Strict form,
/// which is what separates a pivot-valued block from the rest.
template <class T, bool Strict>
FYX_FORCE_INLINE typename VOps<T>::mask vhigh_mask(typename VOps<T>::reg curr,
                                                   typename VOps<T>::reg pivot) {
    if constexpr (Strict) return VOps<T>::gt(curr, pivot);
    else                  return VOps<T>::ge(curr, pivot);
}

template <class T, bool Strict>
FYX_FORCE_INLINE void vpart_vec(T* a, std::size_t& ls, std::size_t& rs,
                                typename VOps<T>::reg curr, typename VOps<T>::reg pivot,
                                typename VOps<T>::reg& vmin, typename VOps<T>::reg& vmax) {
    using P = VOps<T>;
    const typename P::mask gek = vhigh_mask<T, Strict>(curr, pivot);
    const int gc = static_cast<int>(popcount64(static_cast<std::uint64_t>(gek)));
    P::compressstore(a + ls, static_cast<typename P::mask>(~gek), curr);
    P::compressstore(a + rs - static_cast<std::size_t>(gc), gek, curr);
    ls += static_cast<std::size_t>(P::V - gc);
    rs -= static_cast<std::size_t>(gc);
    vmin = P::min(vmin, curr);
    vmax = P::max(vmax, curr);
}

template <class T, bool Strict>
FYX_FORCE_INLINE void vpart_vec_masked(T* a, std::size_t& ls, std::size_t& rs,
                                       typename VOps<T>::reg curr, typename VOps<T>::reg pivot,
                                       typename VOps<T>::mask valid,
                                       typename VOps<T>::reg& vmin, typename VOps<T>::reg& vmax) {
    using P = VOps<T>;
    const typename P::mask gek =
        static_cast<typename P::mask>(vhigh_mask<T, Strict>(curr, pivot) & valid);
    const typename P::mask ltk = static_cast<typename P::mask>((~gek) & valid);
    const int gc = static_cast<int>(popcount64(static_cast<std::uint64_t>(gek)));
    const int lc = static_cast<int>(popcount64(static_cast<std::uint64_t>(ltk)));
    P::compressstore(a + ls, ltk, curr);
    P::compressstore(a + rs - static_cast<std::size_t>(gc), gek, curr);
    ls += static_cast<std::size_t>(lc);
    rs -= static_cast<std::size_t>(gc);
    vmin = P::mask_min(vmin, valid, vmin, curr);
    vmax = P::mask_max(vmax, valid, vmax, curr);
}

/// Scalar partition for ranges too small to hold the two boundary vectors.
template <class T, bool Strict>
inline std::size_t vpartition_scalar(T* a, std::size_t n, T pivot, T& out_min, T& out_max) {
    T mn = VOps<T>::hi();
    T mx = VOps<T>::lo();
    for (std::size_t i = 0; i < n; ++i) {
        const T x = a[i];
        mn = x < mn ? x : mn;
        mx = mx < x ? x : mx;
    }
    std::size_t i = 0, j = n;
    while (i < j) {
        const bool low = Strict ? !(pivot < a[i]) : (a[i] < pivot);
        if (low) { ++i; continue; }
        --j;
        const T t = a[i]; a[i] = a[j]; a[j] = t;
    }
    out_min = mn;
    out_max = mx;
    return i;
}

/// In-place partition of `a[0,n)` around `pivot`: everything `< pivot` moves
/// to the front, everything `>= pivot` to the back, and the count of the
/// former is returned.  `out_min`/`out_max` are the extremes of the whole
/// range, which is what lets the caller prune a side that is all pivot.
///
/// `U` vectors are consumed per iteration; the side read from is selected
/// without a branch (see the file header).
template <class T, bool Strict = false, int U = 4>
inline std::size_t vpartition(T* a, std::size_t n, T pivot, T& out_min, T& out_max) {
    using P   = VOps<T>;
    using reg = typename P::reg;
    constexpr int V = P::V;
    constexpr std::size_t CH = static_cast<std::size_t>(U) * V;

    if (n < 2 * CH) {
        if constexpr (U > 1) return vpartition<T, Strict, U / 2>(a, n, pivot, out_min, out_max);
        else                 return vpartition_scalar<T, Strict>(a, n, pivot, out_min, out_max);
    }

    const reg pv = P::set1(pivot);
    reg vmin = P::set1(P::hi());
    reg vmax = P::set1(P::lo());

    std::size_t l = 0, r = n;      // unread region [l, r)
    std::size_t ls = 0, rs = n;    // free gap: [ls, l) on the left, [r, rs) on the right

    // Lift the U boundary vectors on each side into registers to make room.
    reg vl[U], vr[U];
    for (int i = 0; i < U; ++i) vl[i] = P::loadu(a + static_cast<std::size_t>(i) * V);
    for (int i = 0; i < U; ++i) vr[i] = P::loadu(a + n - static_cast<std::size_t>(i + 1) * V);
    l += CH;
    r -= CH;

    // Reading from the side with less free space keeps at least CH free on
    // both sides, which is what a worst-case split of one iteration needs.
    while (r - l >= CH) {
        const bool from_right = (rs - r) < (l - ls);
        const std::size_t src = from_right ? (r - CH) : l;
        reg cur[U];
        for (int i = 0; i < U; ++i) cur[i] = P::loadu(a + src + static_cast<std::size_t>(i) * V);
        r -= from_right ? CH : 0;
        l += from_right ? 0 : CH;
        for (int i = 0; i < U; ++i) vpart_vec<T, Strict>(a, ls, rs, cur[i], pv, vmin, vmax);
    }
    while (r - l >= static_cast<std::size_t>(V)) {
        const bool from_right = (rs - r) < (l - ls);
        const std::size_t src = from_right ? (r - V) : l;
        const reg cur = P::loadu(a + src);
        r -= from_right ? V : 0;
        l += from_right ? 0 : V;
        vpart_vec<T, Strict>(a, ls, rs, cur, pv, vmin, vmax);
    }
    if (r != l) {  // 0 < r - l < V; once these are read the gap is contiguous
        const unsigned m = static_cast<unsigned>(r - l);
        const typename P::mask valid = static_cast<typename P::mask>((1ull << m) - 1ull);
        const reg cur = P::maskz_loadu(valid, a + l);
        vpart_vec_masked<T, Strict>(a, ls, rs, cur, pv, valid, vmin, vmax);
    }
    for (int i = 0; i < U; ++i) vpart_vec<T, Strict>(a, ls, rs, vl[i], pv, vmin, vmax);
    for (int i = 0; i < U; ++i) vpart_vec<T, Strict>(a, ls, rs, vr[i], pv, vmin, vmax);

    out_min = P::reduce_min(vmin);
    out_max = P::reduce_max(vmax);
    return ls;
}

template <class T, bool Strict>
FYX_FORCE_INLINE void vpart_vec_lean(T* a, std::size_t& ls, std::size_t& rs,
                                     typename VOps<T>::reg curr, typename VOps<T>::reg pivot) {
    using P = VOps<T>;
    const typename P::mask gek = vhigh_mask<T, Strict>(curr, pivot);
    const int gc = static_cast<int>(popcount64(static_cast<std::uint64_t>(gek)));
    P::compressstore(a + ls, static_cast<typename P::mask>(~gek), curr);
    P::compressstore(a + rs - static_cast<std::size_t>(gc), gek, curr);
    ls += static_cast<std::size_t>(P::V - gc);
    rs -= static_cast<std::size_t>(gc);
}

template <class T, bool Strict>
FYX_FORCE_INLINE void vpart_vec_lean_masked(T* a, std::size_t& ls, std::size_t& rs,
                                            typename VOps<T>::reg curr, typename VOps<T>::reg pivot,
                                            typename VOps<T>::mask valid) {
    using P = VOps<T>;
    const typename P::mask gek =
        static_cast<typename P::mask>(vhigh_mask<T, Strict>(curr, pivot) & valid);
    const typename P::mask ltk = static_cast<typename P::mask>((~gek) & valid);
    const int gc = static_cast<int>(popcount64(static_cast<std::uint64_t>(gek)));
    const int lc = static_cast<int>(popcount64(static_cast<std::uint64_t>(ltk)));
    P::compressstore(a + ls, ltk, curr);
    P::compressstore(a + rs - static_cast<std::size_t>(gc), gek, curr);
    ls += static_cast<std::size_t>(lc);
    rs -= static_cast<std::size_t>(gc);
}

template <class T, bool Strict>
FYX_FORCE_INLINE void vpart_vec_lean_max(T* a, std::size_t& ls, std::size_t& rs,
                                         typename VOps<T>::reg curr, typename VOps<T>::reg pivot,
                                         typename VOps<T>::reg& vmax) {
    using P = VOps<T>;
    const typename P::mask gek = vhigh_mask<T, Strict>(curr, pivot);
    const int gc = static_cast<int>(popcount64(static_cast<std::uint64_t>(gek)));
    P::compressstore(a + ls, static_cast<typename P::mask>(~gek), curr);
    P::compressstore(a + rs - static_cast<std::size_t>(gc), gek, curr);
    ls += static_cast<std::size_t>(P::V - gc);
    rs -= static_cast<std::size_t>(gc);
    vmax = P::max(vmax, curr);
}

/// Lean partition for the serial recursion: like `vpartition`, but it tracks
/// only the range maximum, not the minimum (one vector op per partitioned
/// vector instead of two).  The minimum is redundant: `split == 0` already
/// proves the pivot is the range minimum, for free.  The maximum pays for
/// itself on duplicate-heavy inputs: `pivot == max` marks the whole range one
/// value (return, no extra pass) and marks the high side all-pivot (its
/// elements are in their final place; the recursion drops it without sorting
/// it), which costs two full passes to discover structurally.
template <class T, bool Strict = false, int U = 4, bool TrackMax = false>
inline std::size_t vpartition_lean(T* a, std::size_t n, T pivot, T& out_max) {
    using P   = VOps<T>;
    using reg = typename P::reg;
    constexpr int V = P::V;
    constexpr std::size_t CH = static_cast<std::size_t>(U) * V;

    if (n < 2 * CH) {
        if constexpr (U > 1) return vpartition_lean<T, Strict, U / 2, TrackMax>(a, n, pivot, out_max);
        T mn, mx;
        if constexpr (U == 1) {
            const std::size_t s = vpartition_scalar<T, Strict>(a, n, pivot, mn, mx);
            if constexpr (TrackMax) out_max = mx;
            return s;
        }
    }

    const reg pv = P::set1(pivot);

    std::size_t l = 0, r = n;
    std::size_t ls = 0, rs = n;

    reg vmax;
    if constexpr (TrackMax) vmax = P::set1(pivot);

    reg vl[U], vr[U];
    FYX_VQ_UNROLL for (int i = 0; i < U; ++i) vl[i] = P::loadu(a + static_cast<std::size_t>(i) * V);
    FYX_VQ_UNROLL for (int i = 0; i < U; ++i) vr[i] = P::loadu(a + n - static_cast<std::size_t>(i + 1) * V);
    l += CH;
    r -= CH;

    while (r - l >= CH) {
        const bool from_right = (rs - r) < (l - ls);
        const std::size_t src = from_right ? (r - CH) : l;
        reg cur[U];
        FYX_VQ_UNROLL for (int i = 0; i < U; ++i) cur[i] = P::loadu(a + src + static_cast<std::size_t>(i) * V);
        r -= from_right ? CH : 0;
        l += from_right ? 0 : CH;
        if constexpr (TrackMax) {
            FYX_VQ_UNROLL for (int i = 0; i < U; ++i) vpart_vec_lean_max<T, Strict>(a, ls, rs, cur[i], pv, vmax);
        } else {
            FYX_VQ_UNROLL for (int i = 0; i < U; ++i) vpart_vec_lean<T, Strict>(a, ls, rs, cur[i], pv);
        }
    }
    while (r - l >= static_cast<std::size_t>(V)) {
        const bool from_right = (rs - r) < (l - ls);
        const std::size_t src = from_right ? (r - V) : l;
        const reg cur = P::loadu(a + src);
        r -= from_right ? V : 0;
        l += from_right ? 0 : V;
        if constexpr (TrackMax) vpart_vec_lean_max<T, Strict>(a, ls, rs, cur, pv, vmax);
        else                    vpart_vec_lean<T, Strict>(a, ls, rs, cur, pv);
    }
    if (r != l) {
        const unsigned m = static_cast<unsigned>(r - l);
        const typename P::mask valid = static_cast<typename P::mask>((1ull << m) - 1ull);
        const reg cur = P::maskz_loadu(valid, a + l);
        // The masked variant keeps the junk lanes out of the counts and the
        // stores; the max update runs separately over a copy whose junk lanes
        // hold the pivot (<= every right-side value, and the pivot itself is
        // in the range, so this cannot exceed the true maximum).
        if constexpr (TrackMax) {
            vpart_vec_lean_masked<T, Strict>(a, ls, rs, cur, pv, valid);
            vmax = P::max(vmax, P::blend(valid, cur, pv));
        } else {
            vpart_vec_lean_masked<T, Strict>(a, ls, rs, cur, pv, valid);
        }
    }
    FYX_VQ_UNROLL for (int i = 0; i < U; ++i) {
        if constexpr (TrackMax) vpart_vec_lean_max<T, Strict>(a, ls, rs, vl[i], pv, vmax);
        else                    vpart_vec_lean<T, Strict>(a, ls, rs, vl[i], pv);
    }
    FYX_VQ_UNROLL for (int i = 0; i < U; ++i) {
        if constexpr (TrackMax) vpart_vec_lean_max<T, Strict>(a, ls, rs, vr[i], pv, vmax);
        else                    vpart_vec_lean<T, Strict>(a, ls, rs, vr[i], pv);
    }
    if constexpr (TrackMax) out_max = P::reduce_max(vmax);
    return ls;
}

/// Pivot: two vectors' worth of strided samples, sorted in registers.
template <class T>
inline T vpick_pivot(const T* a, std::size_t n) {
    constexpr int S = 2 * VOps<T>::V;
    T s[S];
    const std::size_t step = n / S;
    for (int i = 0; i < S; ++i) s[i] = a[static_cast<std::size_t>(i) * step + (step >> 1)];
    vnet_sort<T>(s, static_cast<std::size_t>(S));
    return s[S / 2];
}

/// Leaf size: 32 vectors, i.e. 512 elements for 4-byte types and 256 for
/// 8-byte ones.  With the column-network leaf (VColLeaf) a 32-register leaf
/// beat 16 registers for every width in the standalone kernel benchmark; the
/// row-wise Batcher leaf it replaced was best at 16.
#ifndef FYX_VQ_LEAF_VECS
#  define FYX_VQ_LEAF_VECS 32
#endif
template <class T>
struct VqLeaf {
    static constexpr std::size_t value = static_cast<std::size_t>(FYX_VQ_LEAF_VECS) * static_cast<std::size_t>(VOps<T>::V);
};

template <class T>
inline void vqsort_rec(T* a, std::size_t n, int budget) {
    while (n > VqLeaf<T>::value) {
        if (budget <= 0) {           // pathological splits: hand over to pdqsort
            pdqsort(a, a + n, std::less<T>());
            return;
        }
        --budget;
        const T pivot = vpick_pivot<T>(a, n);
        // Lean partition: the hot loop tracks only the range maximum (one
        // vector op per partitioned vector, not the two that min+max cost).
        // The minimum is redundant -- `split == 0` proves the pivot is the
        // range minimum for free.  The maximum keeps the two cheap exits the
        // tracked partition had: a range whose every element equals the pivot
        // is finished, and a high side made entirely of pivot values is in
        // its final place and is dropped without being recursed into.  On
        // duplicate-heavy inputs those two cases are common, and discovering
        // them structurally costs two extra passes each.
        T hi;
        std::size_t split = vpartition_lean<T, false, 4, true>(a, n, pivot, hi);
        if (split == 0) {
            // The pivot is the range minimum, so the low side came out empty
            // and splitting there again would not move.
            if (!(pivot < hi)) return;            // the whole range is one value
            // Re-partition with the strict test instead: that puts the
            // pivot-valued elements -- at least one, and they are already in
            // their final place -- in front of everything else, so the range
            // always shrinks.
            T hi2;
            const std::size_t eq = vpartition_lean<T, true>(a, n, pivot, hi2);
            a += eq;
            n -= eq;
            continue;
        }
        if (!(pivot < hi)) {          // the high side is all pivot: already done
            n = split;
            continue;
        }
        if (split < n - split) {      // recurse into the smaller side
            vqsort_rec<T>(a, split, budget);
            a += split;
            n -= split;
        } else {
            vqsort_rec<T>(a + split, n - split, budget);
            n = split;
        }
    }
    vnet_sort<T>(a, n);
}

/// True when the range holds no NaN and no negative zero, i.e. when the
/// hardware's floating-point order is the same total order the radix path
/// would have produced.  Integers are always clean.
template <class T>
inline bool vrange_clean(const T* a, std::size_t n) {
    if constexpr (!std::is_floating_point<T>::value) {
        (void)a; (void)n;
        return true;
    } else if constexpr (sizeof(T) == 4) {
        const std::size_t V = 16;
        __mmask16 bad = 0;
        const __m512i negzero = _mm512_set1_epi32(static_cast<int>(0x80000000u));
        std::size_t i = 0;
        for (; i + V <= n; i += V) {
            const __m512 v = _mm512_loadu_ps(a + i);
            bad = static_cast<__mmask16>(bad | _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q));
            bad = static_cast<__mmask16>(bad | _mm512_cmpeq_epi32_mask(_mm512_castps_si512(v), negzero));
            if (bad) return false;
        }
        if (i < n) {
            const __mmask16 k = static_cast<__mmask16>((1u << (n - i)) - 1u);
            const __m512 v = _mm512_maskz_loadu_ps(k, a + i);
            bad = static_cast<__mmask16>(bad | (k & _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q)));
            bad = static_cast<__mmask16>(bad | (k & _mm512_cmpeq_epi32_mask(_mm512_castps_si512(v), negzero)));
        }
        return bad == 0;
    } else {
        const std::size_t V = 8;
        __mmask8 bad = 0;
        const __m512i negzero = _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ull));
        std::size_t i = 0;
        for (; i + V <= n; i += V) {
            const __m512d v = _mm512_loadu_pd(a + i);
            bad = static_cast<__mmask8>(bad | _mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q));
            bad = static_cast<__mmask8>(bad | _mm512_cmpeq_epi64_mask(_mm512_castpd_si512(v), negzero));
            if (bad) return false;
        }
        if (i < n) {
            const __mmask8 k = static_cast<__mmask8>((1u << (n - i)) - 1u);
            const __m512d v = _mm512_maskz_loadu_pd(k, a + i);
            bad = static_cast<__mmask8>(bad | (k & _mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q)));
            bad = static_cast<__mmask8>(bad | (k & _mm512_cmpeq_epi64_mask(_mm512_castpd_si512(v), negzero)));
        }
        return bad == 0;
    }
}

template <class T>
FYX_FORCE_INLINE typename VOps<T>::mask vunclean_lanes(typename VOps<T>::reg v) {
    using M = typename VOps<T>::mask;
    if constexpr (std::is_same<T, float>::value) {
        return static_cast<M>(_mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q) |
                              _mm512_cmpeq_epi32_mask(_mm512_castps_si512(v),
                                                      _mm512_set1_epi32(static_cast<int>(0x80000000u))));
    } else if constexpr (std::is_same<T, double>::value) {
        return static_cast<M>(_mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q) |
                              _mm512_cmpeq_epi64_mask(_mm512_castpd_si512(v),
                                                      _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ull))));
    } else {
        (void)v;
        return 0;
    }
}

template <class T>
inline unsigned vsmall_prescan(const T* a, std::size_t n) {
    using P = VOps<T>;
    using M = typename P::mask;
    constexpr std::size_t L = static_cast<std::size_t>(P::V);
    constexpr bool fp = std::is_floating_point<T>::value;
    M up = 0, dn = 0, bad = 0;
    std::size_t i = 0;
    if (n >= 2) {
        for (; i + L + 1 <= n; i += L) {
            const auto v = P::loadu(a + i);
            const auto w = P::loadu(a + i + 1);
            up  = static_cast<M>(up | P::gt(w, v));
            dn  = static_cast<M>(dn | P::gt(v, w));
            bad = static_cast<M>(bad | vunclean_lanes<T>(v));
            if (up && dn) {
                // Order is settled; what is left is the cleanliness scan.
                // Floating point: the rest of the cleanliness scan is left to
                // the caller (bit 3), which needs it only if the structural
                // probes decline.
                if constexpr (!fp) return 3u;
                else return bad ? 7u : 11u;
            }
        }
        // Remaining pairs (j, j+1) for j in [i, n-1): fewer than L + 1 keys.
        const std::size_t k = n - 1 - i;            // pairs left, <= L
        const M m = static_cast<M>(k >= L ? ~0ull : ((1ull << k) - 1ull));
        const auto v = P::maskz_loadu(m, a + i);
        const auto w = P::maskz_loadu(m, a + i + 1);
        up  = static_cast<M>(up | (P::gt(w, v) & m));
        dn  = static_cast<M>(dn | (P::gt(v, w) & m));
        bad = static_cast<M>(bad | (vunclean_lanes<T>(v) & m));
        i += k;                                      // a[n-1] not scanned yet
    }
    if constexpr (fp) {
        if (n > 0 && i < n) {
            const std::size_t r = n - i;             // 1 key (the last)
            const M m = static_cast<M>((1ull << r) - 1ull);
            bad = static_cast<M>(bad | (vunclean_lanes<T>(P::maskz_loadu(m, a + i)) & m));
        }
    }
    return (up ? 1u : 0u) | (dn ? 2u : 0u) | (bad ? 4u : 0u);
}



// ---------------------------------------------------------------------------
// Two-run merge on radix keys: a 2V-element bitonic merge network per output
// vector (one vector of carry against the next vector from whichever run has
// the smaller head).  ~0.4 ns/key for int32 on Ice Lake-SP against ~1.5 for
// the best scalar merge.  Works on encoded keys, so NaN / -0 / descending
// follow the radix total order exactly.
// ---------------------------------------------------------------------------
template <class T>
struct VMergeKeys {
    static constexpr bool k64 = sizeof(T) == 8;
    static constexpr int  V   = k64 ? 8 : 16;
    using Key = typename std::conditional<k64, std::uint64_t, std::uint32_t>::type;
    FYX_FORCE_INLINE static __m512i sign() {
        return k64 ? _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL))
                   : _mm512_set1_epi32(static_cast<int>(0x80000000u));
    }
    FYX_FORCE_INLINE static __m512i srai_top(__m512i x) {
        if constexpr (k64) return _mm512_srai_epi64(x, 63); else return _mm512_srai_epi32(x, 31);
    }
    FYX_FORCE_INLINE static __m512i enc(__m512i x, __m512i flip) {
        if constexpr (std::is_floating_point<T>::value)
            x = _mm512_xor_si512(x, _mm512_or_si512(srai_top(x), sign()));
        else if constexpr (std::is_signed<T>::value)
            x = _mm512_xor_si512(x, sign());
        return _mm512_xor_si512(x, flip);
    }
    FYX_FORCE_INLINE static __m512i dec(__m512i k, __m512i flip) {
        k = _mm512_xor_si512(k, flip);
        if constexpr (std::is_floating_point<T>::value)
            k = _mm512_xor_si512(k, _mm512_or_si512(_mm512_andnot_si512(srai_top(k), _mm512_set1_epi32(-1)), sign()));
        else if constexpr (std::is_signed<T>::value)
            k = _mm512_xor_si512(k, sign());
        return k;
    }
    FYX_FORCE_INLINE static __m512i vmin(__m512i a, __m512i b) {
        if constexpr (k64) return _mm512_min_epu64(a, b); else return _mm512_min_epu32(a, b);
    }
    FYX_FORCE_INLINE static __m512i vmax(__m512i a, __m512i b) {
        if constexpr (k64) return _mm512_max_epu64(a, b); else return _mm512_max_epu32(a, b);
    }
    FYX_FORCE_INLINE static __m512i stage(__m512i v, __m512i q, unsigned m) {
        if constexpr (k64) return _mm512_mask_blend_epi64(static_cast<__mmask8>(m), vmin(v, q), vmax(v, q));
        else return _mm512_mask_blend_epi32(static_cast<__mmask16>(m), vmin(v, q), vmax(v, q));
    }
    // sort one bitonic vector
    FYX_FORCE_INLINE static __m512i bitonic(__m512i v) {
        if constexpr (k64) {
            v = stage(v, _mm512_shuffle_i64x2(v, v, 0x4E), 0xF0u);
            v = stage(v, _mm512_permutex_epi64(v, 0x4E), 0xCCu);
            v = stage(v, _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(0x4E)), 0xAAu);
        } else {
            v = stage(v, _mm512_shuffle_i32x4(v, v, 0x4E), 0xFF00u);
            v = stage(v, _mm512_shuffle_i32x4(v, v, 0xB1), 0xF0F0u);
            v = stage(v, _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(0x4E)), 0xCCCCu);
            v = stage(v, _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(0xB1)), 0xAAAAu);
        }
        return v;
    }
    FYX_FORCE_INLINE static void merge2(__m512i a, __m512i b, __m512i& lo, __m512i& hi) {
        const __m512i rev = k64 ? _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7)
                                : _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        b  = k64 ? _mm512_permutexvar_epi64(rev, b) : _mm512_permutexvar_epi32(rev, b);
        lo = bitonic(vmin(a, b));
        hi = bitonic(vmax(a, b));
    }
};

template <class T>
struct VMergeChain {
    const T* A; std::size_t na, ia;
    const T* B; std::size_t nb, ib;
    T* out; std::size_t o;
    __m512i hi;
};

template <class T>
FYX_FORCE_INLINE void vmerge_chain_init(VMergeChain<T>& c, __m512i flip) {
    using M = VMergeKeys<T>;
    constexpr std::size_t V = M::V;
    __m512i lo;
    M::merge2(M::enc(_mm512_loadu_si512(reinterpret_cast<const void*>(c.A)), flip),
              M::enc(_mm512_loadu_si512(reinterpret_cast<const void*>(c.B)), flip), lo, c.hi);
    _mm512_storeu_si512(reinterpret_cast<void*>(c.out), M::dec(lo, flip));
    c.ia = V; c.ib = V; c.o = V;
}

template <class T, class KeyF>
FYX_FORCE_INLINE bool vmerge_chain_live(const VMergeChain<T>& c, KeyF) {
    constexpr std::size_t V = VMergeKeys<T>::V;
    return c.ia + V <= c.na && c.ib + V <= c.nb;
}

template <class T, class KeyF>
FYX_FORCE_INLINE void vmerge_chain_step(VMergeChain<T>& c, __m512i flip, KeyF key) {
    using M = VMergeKeys<T>;
    constexpr std::size_t V = M::V;
    // Branch-free source select: which run feeds the next vector is a coin
    // flip on interleaved data, so a branch here mispredicts every other step.
    const bool fa = key(c.A[c.ia]) < key(c.B[c.ib]);
    const std::uintptr_t msk = std::uintptr_t(0) - static_cast<std::uintptr_t>(fa);
    const std::uintptr_t src = (reinterpret_cast<std::uintptr_t>(c.A + c.ia) & msk) |
                               (reinterpret_cast<std::uintptr_t>(c.B + c.ib) & ~msk);
    const __m512i nx = _mm512_loadu_si512(reinterpret_cast<const void*>(src));
    c.ia += V & msk;
    c.ib += V & ~msk;
    __m512i lo;
    M::merge2(c.hi, M::enc(nx, flip), lo, c.hi);
    _mm512_storeu_si512(reinterpret_cast<void*>(c.out + c.o), M::dec(lo, flip));
    c.o += V;
}

template <class T, class KeyF>
inline void vmerge_chain_finish(VMergeChain<T>& c, __m512i flip, KeyF key) {
    using M = VMergeKeys<T>;
    constexpr std::size_t V = M::V;
    while (vmerge_chain_live(c, key)) vmerge_chain_step(c, flip, key);
    // carry (V keys) + the short rest -> S (< 2V keys); then S against the
    // long rest by binary search and block copies.
    alignas(64) T carry[V];
    _mm512_store_si512(reinterpret_cast<void*>(carry), M::dec(c.hi, flip));
    const bool a_short = c.ia + V > c.na;
    const T* sh = a_short ? c.A + c.ia : c.B + c.ib;
    const std::size_t nsh = a_short ? c.na - c.ia : c.nb - c.ib;
    const T* lg = a_short ? c.B + c.ib : c.A + c.ia;
    const std::size_t nlg = a_short ? c.nb - c.ib : c.na - c.ia;
    T S[2 * V];
    std::size_t ns = 0, ic = 0, is = 0;
    while (ic < V && is < nsh) S[ns++] = key(sh[is]) < key(carry[ic]) ? sh[is++] : carry[ic++];
    while (ic < V) S[ns++] = carry[ic++];
    while (is < nsh) S[ns++] = sh[is++];
    T* out = c.out;
    std::size_t o = c.o, il = 0;
    for (std::size_t j = 0; j < ns; ++j) {
        const auto ks = key(S[j]);
        std::size_t lo2 = il, hi2 = nlg;                // first long key > S[j]
        while (lo2 < hi2) {
            const std::size_t mid = lo2 + (hi2 - lo2) / 2;
            if (ks < key(lg[mid])) hi2 = mid; else lo2 = mid + 1;
        }
        std::memcpy(static_cast<void*>(out + o), static_cast<const void*>(lg + il), (lo2 - il) * sizeof(T));
        o += lo2 - il;
        il = lo2;
        out[o++] = S[j];
    }
    std::memcpy(static_cast<void*>(out + o), static_cast<const void*>(lg + il), (nlg - il) * sizeof(T));
}

// The network of one step is a ~20-cycle dependency chain on the carry, so
// the output is cut at co-ranks into independent chains that are stepped
// round-robin (the out-of-order core overlaps them).
template <class T, class KeyF>
inline void vmerge_runs_impl(const T* A, std::size_t na, const T* B, std::size_t nb, T* out,
                             bool descending, KeyF key) {
    using M = VMergeKeys<T>;
    constexpr std::size_t V = M::V;
    constexpr int C = FYX_VMERGE_CHAINS;
    const __m512i flip = descending ? _mm512_set1_epi32(-1) : _mm512_setzero_si512();
    const std::size_t N = na + nb;
    VMergeChain<T> ch[C];
    std::size_t sa_prev = 0, sb_prev = 0;
    bool split_ok = true;
    for (int c = 0; c < C; ++c) {
        std::size_t sa = na, sb = nb;
        if (c + 1 < C) {
            const std::size_t sidx = N / C * static_cast<std::size_t>(c + 1);
            std::size_t lo = sidx > nb ? sidx - nb : 0, hi = sidx < na ? sidx : na;
            while (lo < hi) {                       // smallest i with B[s-i-1] < A[i]
                const std::size_t i = lo + (hi - lo) / 2;
                const std::size_t j = sidx - i;
                if (j > 0 && !(key(B[j - 1]) < key(A[i]))) lo = i + 1; else hi = i;
            }
            sa = lo; sb = sidx - lo;
        }
        ch[c].A = A + sa_prev; ch[c].na = sa - sa_prev;
        ch[c].B = B + sb_prev; ch[c].nb = sb - sb_prev;
        ch[c].out = out + sa_prev + sb_prev;
        if (ch[c].na < V || ch[c].nb < V) split_ok = false;
        sa_prev = sa; sb_prev = sb;
    }
    if (!split_ok) {
        VMergeChain<T> one{A, na, 0, B, nb, 0, out, 0, _mm512_setzero_si512()};
        vmerge_chain_init(one, flip);
        vmerge_chain_finish(one, flip, key);
        return;
    }
    for (int c = 0; c < C; ++c) vmerge_chain_init(ch[c], flip);
    for (;;) {
        bool live = true;
        for (int c = 0; c < C; ++c) live = live && vmerge_chain_live(ch[c], key);
        if (!live) break;
        FYX_VQ_UNROLL for (int c = 0; c < C; ++c) vmerge_chain_step(ch[c], flip, key);
    }
    for (int c = 0; c < C; ++c) vmerge_chain_finish(ch[c], flip, key);
}

} // namespace isa_avx512
} // namespace detail
} // namespace fyx
FYX_ISA_END

#endif // FYX_HAS_AVX512_CODE

namespace fyx {
namespace detail {

/// SIMD merge of two sorted runs (radix-key order, `descending` flips it)
/// into out[0..na+nb).  Returns false when no vector kernel applies; the
/// caller then merges itself.  Requires na, nb >= one vector.
template <class T>
inline bool vmerge_runs(const T* A, std::size_t na, const T* B, std::size_t nb, T* out,
                        bool descending) {
#if FYX_HAS_AVX512_CODE
    if constexpr (radix_supported_v<T> && (sizeof(T) == 4 || sizeof(T) == 8)) {
        if (!use_avx512() || na < 16 || nb < 16) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
        isa_avx512::vmerge_runs_impl(A, na, B, nb, out, descending,
            [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); });
        return true;
    } else {
        (void)A; (void)na; (void)B; (void)nb; (void)out; (void)descending;
        return false;
    }
#else
    (void)A; (void)na; (void)B; (void)nb; (void)out; (void)descending;
    return false;
#endif
}

/// Types the vectorised quicksort has a kernel for.
template <class T>
struct vqsort_kernel_supported : std::integral_constant<bool,
#if FYX_HAS_AVX512_CODE
    std::is_same<T, std::int32_t>::value  || std::is_same<T, std::uint32_t>::value ||
    std::is_same<T, std::int64_t>::value  || std::is_same<T, std::uint64_t>::value ||
    std::is_same<T, float>::value         || std::is_same<T, double>::value
#else
    false
#endif
> {};

template <class T>
inline constexpr bool vqsort_kernel_supported_v = vqsort_kernel_supported<T>::value;

/// One partition step, for callers that want to drive the recursion
/// themselves (the parallel driver).  `left_done`/`right_done` mark a side
/// that holds one value only and therefore needs no sort.  `split` is always
/// a real reduction: see the strict re-partition in vqsort_rec.
struct VqStep {
    std::size_t split      = 0;
    bool        left_done  = false;
    bool        right_done = false;
};

#if FYX_HAS_AVX512_CODE

template <class T>
inline bool vqsort_range_clean(const T* p, std::size_t n) {
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)p; (void)n; return false; }
    else return isa_avx512::vrange_clean<T>(p, n);
}

/// One fused pass for the small-n front door: neighbour order flags plus the
/// NaN / -0 scan.  Bit 0 of the result: some a[j] < a[j+1]; bit 1: some
/// a[j+1] < a[j]; bit 2: the range is NOT clean (then the order bits are
/// meaningless -- hardware order differs from the library's total order);
/// bit 3: cleanliness not established -- the scan stopped once both
/// directions were seen (always with bits 0 and 1 set, bit 2 clear).
template <class T>
inline unsigned vqsort_small_prescan(const T* p, std::size_t n) {
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)p; (void)n; return 4u; }
    else return isa_avx512::vsmall_prescan<T>(p, n);
}

/// Depth budget: 2 log2(n) partitions is what a correct pivot stream needs;
/// past that the range goes to pdqsort, which has its own introsort guard.
inline int vqsort_budget(std::size_t n) {
    int budget = 2;
    for (std::size_t m = n; m > 1; m >>= 1) budget += 2;
    return budget;
}

template <class T>
inline void vqsort_serial(T* p, std::size_t n) {
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)p; (void)n; }
    else isa_avx512::vqsort_rec<T>(p, n, vqsort_budget(n));
}

template <class T>
inline void vqsort_serial_budget(T* p, std::size_t n, int budget) {
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)p; (void)n; (void)budget; }
    else isa_avx512::vqsort_rec<T>(p, n, budget);
}

template <class T>
inline VqStep vqsort_partition_step(T* p, std::size_t n) {
    VqStep out;
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)p; (void)n; return out; }
    else {
        const T pivot = isa_avx512::vpick_pivot<T>(p, n);
        T lo, hi;
        out.split = isa_avx512::vpartition<T>(p, n, pivot, lo, hi);
        if (!(lo < pivot)) {                    // pivot is the range minimum
            if (!(pivot < hi)) {                // ... and its maximum
                out.split      = n;
                out.left_done  = true;
                out.right_done = true;
                return out;
            }
            T lo2, hi2;
            out.split     = isa_avx512::vpartition<T, true>(p, n, pivot, lo2, hi2);
            out.left_done = true;               // the prefix is all pivot
            return out;
        }
        out.right_done = !(pivot < hi);
        return out;
    }
}

/// Leaf size of the kernel, exported so drivers can stop splitting in time.
template <class T>
inline constexpr std::size_t vqsort_leaf() {
#if FYX_HAS_AVX512_CODE
    if constexpr (vqsort_kernel_supported_v<T>) return isa_avx512::VqLeaf<T>::value;
    else return 0;
#else
    return 0;
#endif
}

#else  // no AVX-512 kernel compiled in

template <class T> inline unsigned vqsort_small_prescan(const T*, std::size_t) { return 4u; }
template <class T> inline bool   vqsort_range_clean(const T*, std::size_t) { return false; }
inline int                       vqsort_budget(std::size_t) { return 0; }
template <class T> inline void   vqsort_serial(T*, std::size_t) {}
template <class T> inline void   vqsort_serial_budget(T*, std::size_t, int) {}
template <class T> inline VqStep vqsort_partition_step(T*, std::size_t) { return VqStep{}; }
template <class T> inline constexpr std::size_t vqsort_leaf() { return 0; }

#endif // FYX_HAS_AVX512_CODE

/// Which types the vector quicksort is actually *faster* on than the radix
/// family, decided by head-to-head kernel races on whole ranges (numbers in
/// tools/dev/NOTES.md, tables in BENCHMARKS.md):
///
///   int32   radix 0.0070 -> vq 0.0044 s (8M serial, old host)   take it
///   float   radix 0.147  -> vq 0.038  s (8M serial, old host)   take it
///   double  radix 0.114  -> vq 0.088  s (8M serial, old host)   take it
///   int64   radix 0.089  -> vq 0.096  s (8M serial, old host)   was radix
///
/// int64 was the lone "keep radix" row -- by 7% on the old host, serially at
/// 8M.  On the current host the same race runs the other way by 2-3x (1M
/// random full-range: 0.0061-0.0094 against 0.014-0.030), and the high-prefix
/// kernel owns a pathological tie case: 8M values over a 40-bit span sort in
/// 0.73 s against the vectorised quicksort's 0.083 -- nine times -- on the
/// tie-repair pass it pays whenever the top prefix bits are not near-unique.
/// A default whose worst measured case is -7% cannot stand against one whose
/// worst measured case is -90%, so 64-bit integers now take the vectorised
/// quicksort on every host; the radix family keeps the shapes only it can do
/// (permutation ranges, prefix-guarded count sorts).
template <class T>
inline constexpr bool vqsort_preferred() {
    return std::is_same<T, std::int32_t>::value || std::is_same<T, std::uint32_t>::value ||
           std::is_same<T, std::int64_t>::value || std::is_same<T, std::uint64_t>::value ||
           std::is_same<T, float>::value        || std::is_same<T, double>::value;
}

/// Runtime gate: the kernel exists for this type, the CPU has AVX-512, and
/// the range is big enough for the vector partition to pay for itself.
template <class T>
inline bool vqsort_usable(std::size_t n) {
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)n; return false; }
    else return use_avx512() && n > vqsort_leaf<T>();
}

} // namespace detail
} // namespace fyx
