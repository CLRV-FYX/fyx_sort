// ---------------------------------------------------------------------------
// Prototype: AVX-512 vectorised quicksort (compress-store partition), raced
// against fyx::sort and std::sort in one process.
//
// tools/dev/NOTES.md closes the accounting on the LSD radix: a pass costs
// ~3.1 ns/elem and three passes is the floor for int32, i.e. ~9.4 ns/elem of
// CPU work, while vqsort does the whole sort in ~7.2.  The only route left is
// a vqsort-class vectorised sorter.  This file measures whether one is worth
// building inside fyx.
//
//   g++ -std=c++17 -O3 -march=native -pthread -DNDEBUG tools/dev/vqs.cpp -o /tmp/vqs
// ---------------------------------------------------------------------------
#include "../../fyx_sort.hpp"

#include <immintrin.h>

#ifdef HAVE_XSS
#  include "x86simdsort-static-incl.h"      // intel/x86-simd-sort, reference only
#endif
#ifdef HAVE_VQSORT
#  include "hwy/contrib/sort/vqsort.h"      // google/highway
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace proto {

// ---------------------------------------------------------------------------
// Per-type vector policies.  Everything is expressed in the *native* type, so
// no encode/decode pass is needed (unlike the radix path).
// ---------------------------------------------------------------------------
struct I32 {
    using T    = std::int32_t;
    using reg  = __m512i;
    using mask = __mmask16;
    static constexpr int V = 16;
    static reg  loadu(const T* p)            { return _mm512_loadu_si512(p); }
    static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_epi32(k, p); }
    static void storeu(T* p, reg v)          { _mm512_storeu_si512(p, v); }
    static void compressstore(T* p, mask k, reg v) { _mm512_mask_compressstoreu_epi32(p, k, v); }
    static mask ge(reg a, reg b)             { return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLT); }
    static reg  min(reg a, reg b)            { return _mm512_min_epi32(a, b); }
    static reg  max(reg a, reg b)            { return _mm512_max_epi32(a, b); }
    static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_epi32(s, k, a, b); }
    static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_epi32(s, k, a, b); }
    static reg  set1(T x)                    { return _mm512_set1_epi32(x); }
    static T    reduce_min(reg v)            { return _mm512_reduce_min_epi32(v); }
    static T    reduce_max(reg v)            { return _mm512_reduce_max_epi32(v); }
    static T    hi()                         { return INT32_MAX; }
    static T    lo()                         { return INT32_MIN; }
    static reg  mask_loadu(reg src, mask k, const T* p) { return _mm512_mask_loadu_epi32(src, k, p); }
    static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_epi32(p, k, v); }
    template <unsigned J> static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi32(
            int(15u ^ J), int(14u ^ J), int(13u ^ J), int(12u ^ J),
            int(11u ^ J), int(10u ^ J), int( 9u ^ J), int( 8u ^ J),
            int( 7u ^ J), int( 6u ^ J), int( 5u ^ J), int( 4u ^ J),
            int( 3u ^ J), int( 2u ^ J), int( 1u ^ J), int( 0u ^ J));
        return _mm512_permutexvar_epi32(idx, v);
    }
    static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_epi32(keepmin, mx, mn); }
};

struct U32 {
    using T    = std::uint32_t;
    using reg  = __m512i;
    using mask = __mmask16;
    static constexpr int V = 16;
    static reg  loadu(const T* p)            { return _mm512_loadu_si512(p); }
    static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_epi32(k, p); }
    static void storeu(T* p, reg v)          { _mm512_storeu_si512(p, v); }
    static void compressstore(T* p, mask k, reg v) { _mm512_mask_compressstoreu_epi32(p, k, v); }
    static mask ge(reg a, reg b)             { return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_NLT); }
    static reg  min(reg a, reg b)            { return _mm512_min_epu32(a, b); }
    static reg  max(reg a, reg b)            { return _mm512_max_epu32(a, b); }
    static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_epu32(s, k, a, b); }
    static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_epu32(s, k, a, b); }
    static reg  set1(T x)                    { return _mm512_set1_epi32(static_cast<int>(x)); }
    static T    reduce_min(reg v)            { return static_cast<T>(_mm512_reduce_min_epu32(v)); }
    static T    reduce_max(reg v)            { return static_cast<T>(_mm512_reduce_max_epu32(v)); }
    static T    hi()                         { return UINT32_MAX; }
    static T    lo()                         { return 0; }
    static reg  mask_loadu(reg src, mask k, const T* p) { return _mm512_mask_loadu_epi32(src, k, p); }
    static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_epi32(p, k, v); }
    template <unsigned J> static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi32(
            int(15u ^ J), int(14u ^ J), int(13u ^ J), int(12u ^ J),
            int(11u ^ J), int(10u ^ J), int( 9u ^ J), int( 8u ^ J),
            int( 7u ^ J), int( 6u ^ J), int( 5u ^ J), int( 4u ^ J),
            int( 3u ^ J), int( 2u ^ J), int( 1u ^ J), int( 0u ^ J));
        return _mm512_permutexvar_epi32(idx, v);
    }
    static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_epi32(keepmin, mx, mn); }
};

struct I64 {
    using T    = std::int64_t;
    using reg  = __m512i;
    using mask = __mmask8;
    static constexpr int V = 8;
    static reg  loadu(const T* p)            { return _mm512_loadu_si512(p); }
    static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_epi64(k, p); }
    static void storeu(T* p, reg v)          { _mm512_storeu_si512(p, v); }
    static void compressstore(T* p, mask k, reg v) { _mm512_mask_compressstoreu_epi64(p, k, v); }
    static mask ge(reg a, reg b)             { return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_NLT); }
    static reg  min(reg a, reg b)            { return _mm512_min_epi64(a, b); }
    static reg  max(reg a, reg b)            { return _mm512_max_epi64(a, b); }
    static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_epi64(s, k, a, b); }
    static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_epi64(s, k, a, b); }
    static reg  set1(T x)                    { return _mm512_set1_epi64(x); }
    static T    reduce_min(reg v)            { return _mm512_reduce_min_epi64(v); }
    static T    reduce_max(reg v)            { return _mm512_reduce_max_epi64(v); }
    static T    hi()                         { return INT64_MAX; }
    static T    lo()                         { return INT64_MIN; }
    static reg  mask_loadu(reg src, mask k, const T* p) { return _mm512_mask_loadu_epi64(src, k, p); }
    static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_epi64(p, k, v); }
    template <unsigned J> static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi64(
            (7u ^ J), (6u ^ J), (5u ^ J), (4u ^ J), (3u ^ J), (2u ^ J), (1u ^ J), (0u ^ J));
        return _mm512_permutexvar_epi64(idx, v);
    }
    static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_epi64(keepmin, mx, mn); }
};

struct U64 {
    using T    = std::uint64_t;
    using reg  = __m512i;
    using mask = __mmask8;
    static constexpr int V = 8;
    static reg  loadu(const T* p)            { return _mm512_loadu_si512(p); }
    static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_epi64(k, p); }
    static void storeu(T* p, reg v)          { _mm512_storeu_si512(p, v); }
    static void compressstore(T* p, mask k, reg v) { _mm512_mask_compressstoreu_epi64(p, k, v); }
    static mask ge(reg a, reg b)             { return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_NLT); }
    static reg  min(reg a, reg b)            { return _mm512_min_epu64(a, b); }
    static reg  max(reg a, reg b)            { return _mm512_max_epu64(a, b); }
    static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_epu64(s, k, a, b); }
    static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_epu64(s, k, a, b); }
    static reg  set1(T x)                    { return _mm512_set1_epi64(static_cast<long long>(x)); }
    static T    reduce_min(reg v)            { return _mm512_reduce_min_epu64(v); }
    static T    reduce_max(reg v)            { return _mm512_reduce_max_epu64(v); }
    static T    hi()                         { return UINT64_MAX; }
    static T    lo()                         { return 0; }
    static reg  mask_loadu(reg src, mask k, const T* p) { return _mm512_mask_loadu_epi64(src, k, p); }
    static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_epi64(p, k, v); }
    template <unsigned J> static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi64(
            (7u ^ J), (6u ^ J), (5u ^ J), (4u ^ J), (3u ^ J), (2u ^ J), (1u ^ J), (0u ^ J));
        return _mm512_permutexvar_epi64(idx, v);
    }
    static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_epi64(keepmin, mx, mn); }
};

struct F32 {
    using T    = float;
    using reg  = __m512;
    using mask = __mmask16;
    static constexpr int V = 16;
    static reg  loadu(const T* p)            { return _mm512_loadu_ps(p); }
    static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_ps(k, p); }
    static void storeu(T* p, reg v)          { _mm512_storeu_ps(p, v); }
    static void compressstore(T* p, mask k, reg v) { _mm512_mask_compressstoreu_ps(p, k, v); }
    static mask ge(reg a, reg b)             { return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ); }
    static reg  min(reg a, reg b)            { return _mm512_min_ps(a, b); }
    static reg  max(reg a, reg b)            { return _mm512_max_ps(a, b); }
    static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_ps(s, k, a, b); }
    static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_ps(s, k, a, b); }
    static reg  set1(T x)                    { return _mm512_set1_ps(x); }
    static T    reduce_min(reg v)            { return _mm512_reduce_min_ps(v); }
    static T    reduce_max(reg v)            { return _mm512_reduce_max_ps(v); }
    static T    hi()                         { return __builtin_inff(); }
    static T    lo()                         { return -__builtin_inff(); }
    static reg  mask_loadu(reg src, mask k, const T* p) { return _mm512_mask_loadu_ps(src, k, p); }
    static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_ps(p, k, v); }
    template <unsigned J> static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi32(
            int(15u ^ J), int(14u ^ J), int(13u ^ J), int(12u ^ J),
            int(11u ^ J), int(10u ^ J), int( 9u ^ J), int( 8u ^ J),
            int( 7u ^ J), int( 6u ^ J), int( 5u ^ J), int( 4u ^ J),
            int( 3u ^ J), int( 2u ^ J), int( 1u ^ J), int( 0u ^ J));
        return _mm512_permutexvar_ps(idx, v);
    }
    static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_ps(keepmin, mx, mn); }
};

struct F64 {
    using T    = double;
    using reg  = __m512d;
    using mask = __mmask8;
    static constexpr int V = 8;
    static reg  loadu(const T* p)            { return _mm512_loadu_pd(p); }
    static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_pd(k, p); }
    static void storeu(T* p, reg v)          { _mm512_storeu_pd(p, v); }
    static void compressstore(T* p, mask k, reg v) { _mm512_mask_compressstoreu_pd(p, k, v); }
    static mask ge(reg a, reg b)             { return _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ); }
    static reg  min(reg a, reg b)            { return _mm512_min_pd(a, b); }
    static reg  max(reg a, reg b)            { return _mm512_max_pd(a, b); }
    static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_pd(s, k, a, b); }
    static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_pd(s, k, a, b); }
    static reg  set1(T x)                    { return _mm512_set1_pd(x); }
    static T    reduce_min(reg v)            { return _mm512_reduce_min_pd(v); }
    static T    reduce_max(reg v)            { return _mm512_reduce_max_pd(v); }
    static T    hi()                         { return __builtin_inf(); }
    static T    lo()                         { return -__builtin_inf(); }
    static reg  mask_loadu(reg src, mask k, const T* p) { return _mm512_mask_loadu_pd(src, k, p); }
    static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_pd(p, k, v); }
    template <unsigned J> static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi64(
            (7u ^ J), (6u ^ J), (5u ^ J), (4u ^ J), (3u ^ J), (2u ^ J), (1u ^ J), (0u ^ J));
        return _mm512_permutexvar_pd(idx, v);
    }
    static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_pd(keepmin, mx, mn); }
};


// ---------------------------------------------------------------------------
// Native-typed sorting networks.  fyx's networks run on RadixTraits-encoded
// unsigned keys, which costs an encode and a decode pass at every leaf; inside
// a quicksort the comparisons are already in the native domain (the recursion
// never sees a NaN once the caller has screened for them), so the leaf can use
// the type's own min/max and skip both passes.  The generic bitonic template
// from parts/_network_body.inc is reused unchanged.
// ---------------------------------------------------------------------------
template <class P>
struct NetOps {
    using Key  = typename P::T;
    using Vec  = typename P::reg;
    using Mask = typename P::mask;
    static constexpr unsigned kLanes = static_cast<unsigned>(P::V);

    static Key sentinel() { return P::hi(); }

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
        const Mask m = static_cast<Mask>((1ull << n) - 1ull);
        P::mask_storeu(p, m, v);
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) { return P::template permute_xor<J>(v); }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        return P::blend(keepmin, mn, mx);
    }
};

/// Same shape as fyx::detail::network_sort_v, but takes the padding sentinel
/// from the policy (a float network must pad with +inf, not FLT_MAX) and
/// allows more than 64 elements per leaf.
template <class Ops, unsigned V>
FYX_FORCE_INLINE void net_sort_v(typename Ops::Key* keys, std::size_t n) {
    using Key = typename Ops::Key;
    using Vec = typename Ops::Vec;
    constexpr unsigned L = Ops::kLanes;
    const Key sentinel = Ops::sentinel();
    Vec v[V];
    for (unsigned i = 0; i < V; ++i) {
        const std::size_t off = std::size_t(i) * L;
        if (off + L <= n)      v[i] = Ops::load(keys + off);
        else if (off < n)      v[i] = Ops::load_partial(keys + off, unsigned(n - off), sentinel);
        else                   v[i] = Ops::splat(sentinel);
    }
    fyx::detail::isa_avx512::BitonicVec<Ops, V>::run(v);
    for (unsigned i = 0; i < V; ++i) {
        const std::size_t off = std::size_t(i) * L;
        if (off + L <= n)      Ops::store(keys + off, v[i]);
        else if (off < n)      Ops::store_partial(keys + off, v[i], unsigned(n - off));
    }
}

template <class P>
FYX_FORCE_INLINE void net_sort(typename P::T* a, std::size_t n) {
    using Ops = NetOps<P>;
    constexpr unsigned L = Ops::kLanes;
    if (n < 2) return;
    const std::size_t vecs = (static_cast<std::size_t>(fyx::detail::next_pow2(n)) + L - 1) / L;
    switch (vecs) {
        case 1:  net_sort_v<Ops, 1>(a, n);  return;
        case 2:  net_sort_v<Ops, 2>(a, n);  return;
        case 4:  net_sort_v<Ops, 4>(a, n);  return;
        case 8:  net_sort_v<Ops, 8>(a, n);  return;
        case 16: net_sort_v<Ops, 16>(a, n); return;
        case 32: net_sort_v<Ops, 32>(a, n); return;
        default: break;
    }
    fyx::detail::pdqsort(a, a + n, std::less<typename P::T>());
}


// ---------------------------------------------------------------------------
// One vector through the partition.  `ls` is the next write position of the
// "< pivot" region growing from the left, `rs` the end (exclusive) of the
// ">= pivot" region growing down from the right.
// ---------------------------------------------------------------------------
template <class P>
static inline void part_vec(typename P::T* a, std::size_t& ls, std::size_t& rs,
                            typename P::reg curr, typename P::reg pivot,
                            typename P::reg& vmin, typename P::reg& vmax) {
    const typename P::mask gek = P::ge(curr, pivot);
    const int gc = static_cast<int>(__builtin_popcountll(static_cast<unsigned long long>(gek)));
    P::compressstore(a + ls, static_cast<typename P::mask>(~gek), curr);
    P::compressstore(a + rs - gc, gek, curr);
    ls += static_cast<std::size_t>(P::V - gc);
    rs -= static_cast<std::size_t>(gc);
    vmin = P::min(vmin, curr);
    vmax = P::max(vmax, curr);
}

template <class P>
static inline void part_vec_masked(typename P::T* a, std::size_t& ls, std::size_t& rs,
                                   typename P::reg curr, typename P::reg pivot,
                                   typename P::mask valid,
                                   typename P::reg& vmin, typename P::reg& vmax) {
    const typename P::mask gek = static_cast<typename P::mask>(P::ge(curr, pivot) & valid);
    const typename P::mask ltk = static_cast<typename P::mask>((~gek) & valid);
    const int gc = static_cast<int>(__builtin_popcountll(static_cast<unsigned long long>(gek)));
    const int lc = static_cast<int>(__builtin_popcountll(static_cast<unsigned long long>(ltk)));
    P::compressstore(a + ls, ltk, curr);
    P::compressstore(a + rs - gc, gek, curr);
    ls += static_cast<std::size_t>(lc);
    rs -= static_cast<std::size_t>(gc);
    vmin = P::mask_min(vmin, valid, vmin, curr);
    vmax = P::mask_max(vmax, valid, vmax, curr);
}

/// In-place partition of a[0,n) around `pivot`.  Returns the number of
/// elements strictly below the pivot; also reports the range min/max so the
/// caller can prune all-equal partitions.
///
/// `U` vectors are consumed per iteration and the side to read from is chosen
/// branchlessly: the choice depends on how the data splits, so a branch there
/// mispredicts on roughly every other iteration, which at one vector per
/// iteration costs more than the partition itself.
template <class P, int U = 4>
static std::size_t partition(typename P::T* a, std::size_t n, typename P::T pivot,
                             typename P::T& out_min, typename P::T& out_max) {
    using T   = typename P::T;
    using reg = typename P::reg;
    constexpr int V   = P::V;
    constexpr std::size_t CH = static_cast<std::size_t>(U) * V;  // elements per iteration
    const reg pv = P::set1(pivot);
    reg vmin = P::set1(P::hi());
    reg vmax = P::set1(P::lo());

    if (n < 2 * CH + 1) {   // too small to unroll: fall back to one vector at a time
        if constexpr (U > 1) {
            return partition<P, 1>(a, n, pivot, out_min, out_max);
        }
    }

    std::size_t l = 0, r = n;          // unread region [l, r)
    std::size_t ls = 0, rs = n;        // free gap is [ls, l) and [r, rs)

    // Lift the U boundary vectors on each side into registers to make room.
    reg vl[U], vr[U];
    for (int i = 0; i < U; ++i) vl[i] = P::loadu(a + static_cast<std::size_t>(i) * V);
    for (int i = 0; i < U; ++i) vr[i] = P::loadu(a + n - static_cast<std::size_t>(i + 1) * V);
    l += CH;
    r -= CH;

    while (r - l >= CH) {
        // Read from whichever side has less free space; that leaves at least
        // CH free on both sides, which is what a worst-case split needs.
        const bool from_right = (rs - r) < (l - ls);
        const std::size_t src = from_right ? (r - CH) : l;
        reg cur[U];
        for (int i = 0; i < U; ++i) cur[i] = P::loadu(a + src + static_cast<std::size_t>(i) * V);
        r -= from_right ? CH : 0;
        l += from_right ? 0 : CH;
        for (int i = 0; i < U; ++i) part_vec<P>(a, ls, rs, cur[i], pv, vmin, vmax);
    }
    while (r - l >= static_cast<std::size_t>(V)) {
        const bool from_right = (rs - r) < (l - ls);
        const std::size_t src = from_right ? (r - V) : l;
        const reg cur = P::loadu(a + src);
        r -= from_right ? V : 0;
        l += from_right ? 0 : V;
        part_vec<P>(a, ls, rs, cur, pv, vmin, vmax);
    }
    if (r != l) {  // 0 < r - l < V; the gap is contiguous after this read
        const int m = static_cast<int>(r - l);
        const typename P::mask valid = static_cast<typename P::mask>((1ull << m) - 1ull);
        const reg cur = P::maskz_loadu(valid, a + l);
        part_vec_masked<P>(a, ls, rs, cur, pv, valid, vmin, vmax);
    }
    for (int i = 0; i < U; ++i) part_vec<P>(a, ls, rs, vl[i], pv, vmin, vmax);
    for (int i = 0; i < U; ++i) part_vec<P>(a, ls, rs, vr[i], pv, vmin, vmax);

    out_min = P::reduce_min(vmin);
    out_max = P::reduce_max(vmax);
    return ls;
}

// Pivot: SAMP vectors' worth of strided samples, sorted in registers.
template <class P, int SAMP>
static typename P::T pick_pivot(const typename P::T* a, std::size_t n) {
    using T = typename P::T;
    constexpr int S = SAMP * P::V;
    T s[S];
    const std::size_t step = n / S;
    for (int i = 0; i < S; ++i) s[i] = a[static_cast<std::size_t>(i) * step + (step >> 1)];
    net_sort<P>(s, static_cast<std::size_t>(S));
    return s[S / 2];
}

template <class P, int BASE, int SAMP, int UNROLL>
static void qsort_(typename P::T* a, std::size_t n, int budget) {
    using T = typename P::T;
    while (n > static_cast<std::size_t>(BASE)) {
        if (budget <= 0) {  // pathological splits: hand over to the safe sort
            fyx::detail::pdqsort(a, a + n, std::less<T>());
            return;
        }
        --budget;
        const T pivot = pick_pivot<P, SAMP>(a, n);
        T lo, hi;
        const std::size_t split = partition<P, UNROLL>(a, n, pivot, lo, hi);
        if (pivot == hi) {          // right side is all equal to the pivot
            if (pivot == lo) return;
            n = split;
            continue;
        }
        if (pivot == lo) {          // nothing below the pivot
            a += split;
            n -= split;
            continue;
        }
        if (split < n - split) {
            qsort_<P, BASE, SAMP, UNROLL>(a, split, budget);
            a += split;
            n -= split;
        } else {
            qsort_<P, BASE, SAMP, UNROLL>(a + split, n - split, budget);
            n = split;
        }
    }
    if (n > 1) net_sort<P>(a, n);
}

template <class P, int BASE = 128, int SAMP = 2, int UNROLL = 4>
void sort(typename P::T* a, std::size_t n) {
    if (n < 2) return;
    int budget = 2;
    for (std::size_t m = n; m > 1; m >>= 1) budget += 2;
    qsort_<P, BASE, SAMP, UNROLL>(a, n, budget);
}

}  // namespace proto

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;
static double now() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

template <class T, class F>
static double timeit(const std::vector<T>& src, std::vector<T>& work, F&& f, int reps) {
    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
        work = src;
        const double t0 = now();
        f(work.data(), work.size());
        const double t1 = now();
        best = std::min(best, t1 - t0);
        if (!std::is_sorted(work.begin(), work.end())) {
            std::printf("  !!! NOT SORTED\n");
            break;
        }
    }
    return best;
}

template <class T>
static std::vector<T> make_random(std::size_t n, unsigned seed) {
    std::vector<T> src(n);
    std::mt19937_64 rng(seed);
    for (auto& x : src) {
        if constexpr (std::is_floating_point<T>::value) {
            x = static_cast<T>(static_cast<double>(rng()) / static_cast<double>(UINT64_MAX));
        } else {
            x = static_cast<T>(rng());
        }
    }
    return src;
}

/// Variant grid: base-case size x pivot sample count, measured interleaved.
template <class P>
static void grid(const char* name, std::size_t n, int reps, unsigned seed) {
    using T = typename P::T;
    const std::vector<T> src = make_random<T>(n, seed);
    std::vector<T> work(n);
    struct Row { const char* tag; double t; };
    Row rows[] = {
        {"u1/b64",   timeit(src, work, [](T* p, std::size_t m) { proto::sort<P, 64, 2, 1>(p, m); }, reps)},
        {"u2/b64",   timeit(src, work, [](T* p, std::size_t m) { proto::sort<P, 64, 2, 2>(p, m); }, reps)},
        {"u4/b64",   timeit(src, work, [](T* p, std::size_t m) { proto::sort<P, 64, 2, 4>(p, m); }, reps)},
        {"u8/b64",   timeit(src, work, [](T* p, std::size_t m) { proto::sort<P, 64, 2, 8>(p, m); }, reps)},
        {"u4/b128",  timeit(src, work, [](T* p, std::size_t m) { proto::sort<P, 128, 2, 4>(p, m); }, reps)},
        {"u4/b256",  timeit(src, work, [](T* p, std::size_t m) { proto::sort<P, 256, 2, 4>(p, m); }, reps)},
        {"u8/b128",  timeit(src, work, [](T* p, std::size_t m) { proto::sort<P, 128, 2, 8>(p, m); }, reps)},
        {"u4/b128/s1", timeit(src, work, [](T* p, std::size_t m) { proto::sort<P, 128, 1, 4>(p, m); }, reps)},
    };
    const double t_fyx = timeit(src, work, [](T* p, std::size_t m) {
        fyx::Options o; o.parallel = fyx::Tri::Off; fyx::sort(p, m, o);
    }, reps);
    std::printf("%-4s n=%-9zu fyx-seq %.5f  ", name, n, t_fyx);
    for (const Row& r : rows) std::printf("%s %.5f (%.2fx)  ", r.tag, r.t, t_fyx / r.t);
    std::printf("\n");
}

template <class P>
static void run(const char* name, std::size_t n, int reps, unsigned seed) {
    using T = typename P::T;
    std::vector<T> src(n);
    std::mt19937_64 rng(seed);
    for (auto& x : src) {
        if constexpr (std::is_floating_point<T>::value) {
            x = static_cast<T>(static_cast<double>(rng()) / static_cast<double>(UINT64_MAX));
        } else {
            x = static_cast<T>(rng());
        }
    }
    std::vector<T> work(n);

    const double t_proto = timeit(src, work, [](T* p, std::size_t m) { proto::sort<P>(p, m); }, reps);
    const double t_fyx   = timeit(src, work, [](T* p, std::size_t m) {
        fyx::Options o; o.parallel = fyx::Tri::Off; fyx::sort(p, m, o);
    }, reps);
    const double t_fyxp  = timeit(src, work, [](T* p, std::size_t m) { fyx::sort(p, m); }, reps);
    const double t_std   = timeit(src, work, [](T* p, std::size_t m) { std::sort(p, p + m); }, reps);
#ifdef HAVE_XSS
    const double t_xss = timeit(src, work, [](T* p, std::size_t m) {
        x86simdsortStatic::qsort(p, m, false, false);
    }, reps);
#else
    const double t_xss = 0.0;
#endif
#ifdef HAVE_VQSORT
    const double t_vq = timeit(src, work, [](T* p, std::size_t m) {
        hwy::VQSort(p, m, hwy::SortAscending());
    }, reps);
#else
    const double t_vq = 0.0;
#endif
    std::printf("%-8s n=%-9zu  proto %.5f   xss %.5f   vqsort %.5f\n", name, n, t_proto, t_xss, t_vq);

    std::printf("%-8s n=%-9zu  proto %.5f   fyx-seq %.5f   fyx-par %.5f   std %.5f   |  proto/fyxseq %.2fx  proto/fyxpar %.2fx\n",
                name, n, t_proto, t_fyx, t_fyxp, t_std, t_fyx / t_proto, t_fyxp / t_proto);
}

// correctness over adversarial shapes
template <class P>
static bool verify(const char* name) {
    using T = typename P::T;
    std::mt19937_64 rng(12345);
    static const std::size_t sizes[] = {0, 1, 2, 3, 15, 16, 17, 31, 33, 64, 65, 66, 100, 127,
                                        128, 129, 255, 257, 1000, 4096, 10007, 65537};
    for (std::size_t n : sizes) {
        for (int shape = 0; shape < 7; ++shape) {
            std::vector<T> v(n);
            for (std::size_t i = 0; i < n; ++i) {
                switch (shape) {
                    case 0: v[i] = static_cast<T>(rng()); break;
                    case 1: v[i] = static_cast<T>(i); break;
                    case 2: v[i] = static_cast<T>(n - i); break;
                    case 3: v[i] = static_cast<T>(7); break;
                    case 4: v[i] = static_cast<T>(rng() % 3); break;
                    case 5: v[i] = static_cast<T>((i % 2) ? 1 : 1000000); break;
                    default: v[i] = static_cast<T>((i * 2654435761u) % 1024); break;
                }
            }
            std::vector<T> ref = v;
            std::sort(ref.begin(), ref.end());
            proto::sort<P>(v.data(), v.size());
            if (v != ref) {
                std::printf("VERIFY FAIL %s n=%zu shape=%d\n", name, n, shape);
                return false;
            }
        }
    }
    std::printf("verify %-8s ok\n", name);
    return true;
}

int main(int argc, char** argv) {
    std::size_t n = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 1000000;
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 3;

    bool ok = true;
    ok &= verify<proto::I32>("i32");
    ok &= verify<proto::U32>("u32");
    ok &= verify<proto::I64>("i64");
    ok &= verify<proto::U64>("u64");
    ok &= verify<proto::F32>("f32");
    ok &= verify<proto::F64>("f64");
    if (!ok) return 1;

    if (argc > 3 && std::string(argv[3]) == "grid") {
        for (std::size_t sz : {n, n * 8}) {
            grid<proto::I32>("i32", sz, reps, 1);
            grid<proto::I64>("i64", sz, reps, 2);
            grid<proto::F64>("f64", sz, reps, 3);
            grid<proto::F32>("f32", sz, reps, 4);
            std::printf("\n");
        }
        return 0;
    }
    for (std::size_t sz : {n, n * 8}) {
        run<proto::I32>("i32", sz, reps, 1);
        run<proto::I64>("i64", sz, reps, 2);
        run<proto::F64>("f64", sz, reps, 3);
        run<proto::F32>("f32", sz, reps, 4);
        std::printf("\n");
    }
    return 0;
}
