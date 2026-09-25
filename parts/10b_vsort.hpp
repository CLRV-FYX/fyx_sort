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
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) { _mm512_mask_compressstoreu_epi32(p, k, v); }
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
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) { _mm512_mask_compressstoreu_epi32(p, k, v); }
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
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) { _mm512_mask_compressstoreu_epi64(p, k, v); }
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
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) { _mm512_mask_compressstoreu_epi64(p, k, v); }
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
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) { _mm512_mask_compressstoreu_ps(p, k, v); }
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
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) { _mm512_mask_compressstoreu_pd(p, k, v); }
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

/// Leaf sort for n <= 16 vectors' worth of elements.
template <class T>
inline void vnet_sort(T* a, std::size_t n) {
    constexpr unsigned L = static_cast<unsigned>(VOps<T>::V);
    if (n < 2) return;
    const std::size_t vecs = (static_cast<std::size_t>(next_pow2(n)) + L - 1) / L;
    switch (vecs) {
        case 1:  vnet_sort_v<T, 1>(a, n);  return;
        case 2:  vnet_sort_v<T, 2>(a, n);  return;
        case 4:  vnet_sort_v<T, 4>(a, n);  return;
        case 8:  vnet_sort_v<T, 8>(a, n);  return;
        case 16: vnet_sort_v<T, 16>(a, n); return;
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
    for (int i = 0; i < U; ++i) vl[i] = P::loadu(a + static_cast<std::size_t>(i) * V);
    for (int i = 0; i < U; ++i) vr[i] = P::loadu(a + n - static_cast<std::size_t>(i + 1) * V);
    l += CH;
    r -= CH;

    while (r - l >= CH) {
        const bool from_right = (rs - r) < (l - ls);
        const std::size_t src = from_right ? (r - CH) : l;
        reg cur[U];
        for (int i = 0; i < U; ++i) cur[i] = P::loadu(a + src + static_cast<std::size_t>(i) * V);
        r -= from_right ? CH : 0;
        l += from_right ? 0 : CH;
        if constexpr (TrackMax) {
            for (int i = 0; i < U; ++i) vpart_vec_lean_max<T, Strict>(a, ls, rs, cur[i], pv, vmax);
        } else {
            for (int i = 0; i < U; ++i) vpart_vec_lean<T, Strict>(a, ls, rs, cur[i], pv);
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
    for (int i = 0; i < U; ++i) {
        if constexpr (TrackMax) vpart_vec_lean_max<T, Strict>(a, ls, rs, vl[i], pv, vmax);
        else                    vpart_vec_lean<T, Strict>(a, ls, rs, vl[i], pv);
    }
    for (int i = 0; i < U; ++i) {
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

/// Leaf size: 16 vectors, i.e. 256 elements for 4-byte types and 128 for
/// 8-byte ones (measured best of 64/128/256 for both widths).
template <class T>
struct VqLeaf {
    static constexpr std::size_t value = 16u * static_cast<std::size_t>(VOps<T>::V);
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

} // namespace isa_avx512
} // namespace detail
} // namespace fyx
FYX_ISA_END

#endif // FYX_HAS_AVX512_CODE

namespace fyx {
namespace detail {

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
