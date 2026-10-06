// ===========================================================================
//  Section 12/13 -- Public API + adaptive dispatcher
//
//  This is the layer the rest of the library was missing: the callable,
//  documented surface.  Everything below dispatches into the already-tested
//  kernels in fyx::detail:
//
//      * numeric keys, default "<" comparator, n <= 64  -> SIMD bitonic net
//      * numeric keys, default "<" comparator, n  > 64  -> LSD radix (stable)
//      * numeric keys, default ">" comparator           -> radix + reverse
//      * everything else (custom comparator / non-numeric)-> pdqsort
//      * numeric, AVX-512, n < kVqsortMinN              -> order scan + vector quicksort
//      * stable_sort of integers, "<" or ">"            -> unstable engine (equal ints
//                                                          are indistinguishable)
//      * stable_sort of numeric ascending               -> radix (stable)
//      * stable_sort otherwise                          -> bottom-up merge sort
//
//  The dispatcher is intentionally honest: it only takes fast paths it can
//  *prove* are equivalent to the requested order.  No distribution guessing,
//  no "already sorted" shortcuts that could mis-classify and return garbage.
// ===========================================================================

namespace fyx {

// ---------------------------------------------------------------------------
// Tunables the caller can set
// ---------------------------------------------------------------------------

/// Three-state switch for the parallel path.
enum class Tri : unsigned char {
    Auto = 0,  ///< parallel when a pool exists and the problem is large enough
    Off  = 1,  ///< force single-threaded
    On   = 2   ///< force parallel (falls back to serial if no pool)
};

/// Runtime options for a single sort call.
struct Options {
    Tri      parallel = Tri::Auto;  ///< serial / parallel policy
    unsigned threads  = 0;          ///< advisory worker count (0 = pool default)
    bool     gpu      = false;      ///< reserved; only meaningful with FYX_ENABLE_GPU
    constexpr Options() noexcept = default;
};

namespace detail {

// ---- SFINAE helpers used by the public overload set -----------------------

template <class T>
struct is_fyx_options : std::is_same<std::remove_cv_t<std::remove_reference_t<T>>, Options> {};
template <class T>
inline constexpr bool is_fyx_options_v = is_fyx_options<T>::value;

template <class C, class = void>
struct has_std_data : std::false_type {};
template <class C>
struct has_std_data<C, std::void_t<
    decltype(std::data(std::declval<C&>())),
    decltype(std::size(std::declval<C&>()))>> : std::true_type {};
template <class C>
inline constexpr bool has_std_data_v = has_std_data<C>::value;

template <class It>
struct is_std_reverse_iterator : std::false_type {};
template <class It>
struct is_std_reverse_iterator<std::reverse_iterator<It>> : std::true_type {};

template <class It, class = void>
struct iterator_base_pointer {
    static constexpr bool value = false;
};

template <class It>
struct iterator_base_pointer<It, std::void_t<decltype(std::declval<It>().base()), decltype(*std::declval<It>())>> {
    using raw_base = std::remove_reference_t<decltype(std::declval<It>().base())>;
    using pointee  = std::remove_pointer_t<raw_base>;
    static constexpr bool value = std::is_pointer<raw_base>::value &&
        !std::is_const<pointee>::value &&
        std::is_lvalue_reference<decltype(*std::declval<It>())>::value &&
        !is_std_reverse_iterator<typename std::decay<It>::type>::value;
    static raw_base get(It it) noexcept { return it.base(); }
};

template <class It>
inline constexpr bool has_mutable_base_pointer_v = iterator_base_pointer<It>::value;

/// Containers whose elements live in nodes (std::list, std::forward_list).
/// Their own sort splices nodes instead of moving elements, which is both
/// cheaper than any move and the only thing that can be done through a
/// bidirectional iterator.
template <class C, class = void>
struct has_member_sort : std::false_type {};
template <class C>
struct has_member_sort<C, std::void_t<decltype(std::declval<C&>().sort())>> : std::true_type {};
template <class C>
inline constexpr bool has_member_sort_v = has_member_sort<C>::value;

template <class C, class Comp, class = void>
struct has_member_sort_with : std::false_type {};
template <class C, class Comp>
struct has_member_sort_with<C, Comp,
    std::void_t<decltype(std::declval<C&>().sort(std::declval<Comp&>()))>> : std::true_type {};
template <class C, class Comp>
inline constexpr bool has_member_sort_with_v = has_member_sort_with<C, Comp>::value;

// Forward declaration of the (optionally compiled) GPU dispatch.  Defined in
// parts/15_gpu.hpp under #ifdef FYX_ENABLE_GPU; when that switch is off the
// function does not exist and the guarded call sites below compile away.
#if FYX_ENABLE_GPU
template <class T, class Comp>
inline bool gpu_sort_dispatch(T* p, std::size_t n, Comp comp, const Options& o);
#endif

// ---------------------------------------------------------------------------
// Monotone-run and counting-sort fast paths (武器二).
//
//  * sorted / reverse-sorted detection gives the best case an O(n) exit for
//    every comparator and every type;
//  * integer range counting handles dense tiny domains (u8/i8/enums-by-value
//    shapes) without paying radix passes;
//  * compressed low-cardinality counting handles arbitrary sparse values and
//    arbitrary payload-carrying objects by sorting equivalence classes under
//    the user comparator, then moving the original objects into bucket order.
//
// The compressed path is deliberately conservative: it first probes an evenly
// spaced sample and only performs the full O(n log 256) classification when the
// sample did not already prove high cardinality.  Floating-point default-order
// data stays on the radix path so NaN / -0 / +0 keep the library's documented
// total-order behaviour.
// ---------------------------------------------------------------------------

inline constexpr std::size_t kCountingClassLimit = 256;
inline constexpr std::size_t kCountingProbeLimit = 4096;
inline constexpr std::size_t kCountingMinN       = kRadixThreshold;
inline constexpr std::size_t kCountingRangeLimit = 1u << 20;

template <class It, class Comp>
inline bool try_monotonic_sort(It first, It last, Comp comp, bool allow_reverse) {
    const std::size_t n = static_cast<std::size_t>(last - first);
    if (n < 2) return true;

    std::size_t i = 1;
    for (; i < n; ++i) {
        if (comp(first[i], first[i - 1])) {          // descending under comp
            for (++i; i < n; ++i)
                if (comp(first[i - 1], first[i])) return false;
            if (allow_reverse) std::reverse(first, last);
            return allow_reverse;
        }
        if (comp(first[i - 1], first[i])) {          // ascending under comp
            for (++i; i < n; ++i)
                if (comp(first[i], first[i - 1])) return false;
            return true;
        }
    }
    // All elements equivalent under comp.
    return true;
}

// Stable sorting may reverse only a *strictly* descending range: if equivalent
// adjacent elements exist, reversing them would violate stability. Kept out of
// line so the common already-sorted monotonic exit remains compact.
template <class It, class Comp>
FYX_NOINLINE inline bool try_stable_reverse_sort(It first, It last, Comp comp) {
    const std::size_t n = static_cast<std::size_t>(last - first);
    if (n < 2) return false;
    for (std::size_t i = 1; i < n; ++i)
        if (!comp(first[i], first[i - 1])) return false;
    std::reverse(first, last);
    return true;
}

template <class T>
inline bool try_radix_monotonic_sort(T* p, std::size_t n,
                                     bool descending, bool allow_reverse) {
    if constexpr (!radix_supported_v<T>) {
        (void)p; (void)n; (void)descending; (void)allow_reverse;
        return false;
    } else {
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        if (n < 2) return true;

        // Branch-free blocked pre-check for the common "already in target
        // order" case.  Each block reduces a violation flag without an early
        // exit, so compilers vectorise it and sorted input is verified at
        // close to memory bandwidth; the first violating block falls through
        // to the exact scalar classifier below (which also handles reverse
        // and one-break rotations).  Random input fails in the first block.
        {
            constexpr std::size_t kBlock = 256;
            bool in_order = true;
            for (std::size_t b = 1; b < n && in_order; b += kBlock) {
                const std::size_t e = std::min(n, b + kBlock);
                unsigned bad = 0;
                if (descending) {
                    for (std::size_t j = b; j < e; ++j)
                        bad |= static_cast<unsigned>(RT::encode(p[j - 1]) < RT::encode(p[j]));
                } else {
                    for (std::size_t j = b; j < e; ++j)
                        bad |= static_cast<unsigned>(RT::encode(p[j]) < RT::encode(p[j - 1]));
                }
                in_order = bad == 0;
            }
            if (in_order) return true;
        }

        Key prev = RT::encode(p[0]);
        for (std::size_t i = 1; i < n; ++i) {
            const Key cur = RT::encode(p[i]);
            if (cur == prev) continue;

            const bool target_order = descending ? (cur < prev) : (prev < cur);
            if (target_order) {
                prev = cur;
                for (++i; i < n; ++i) {
                    const Key k = RT::encode(p[i]);
                    if (descending ? (prev < k) : (k < prev)) {
                        // One-break structural proof: the prefix [0, brk) is
                        // verified in target order and the pair at brk is the
                        // single violation so far.  If the suffix from brk on
                        // is also monotone in target order and the endpoints
                        // wrap (last <= first), the range is a rotation of a
                        // sorted array; rotating it back is O(n) and stable.
                        // Random input piles up a second violation within the
                        // first few elements after the break and declines, so
                        // the extra scan costs almost nothing elsewhere.
                        const std::size_t brk = i;
                        for (++i; i < n; ++i) {
                            const Key k2 = RT::encode(p[i]);
                            if (descending ? (prev < k2) : (k2 < prev)) return false;
                            prev = k2;
                        }
                        const Key front = RT::encode(p[0]);
                        const Key back  = RT::encode(p[n - 1]);
                        const bool cyclic = descending ? (back >= front) : (back <= front);
                        if (!cyclic) return false;
                        std::rotate(p, p + brk, p + n);
                        return true;
                    }
                    prev = k;
                }
                return true;
            }

            // The range is monotone in the opposite direction.  For unstable
            // sort we may reverse it; stable_sort asks us not to because that
            // would reverse equal-key groups.
            prev = cur;
            for (++i; i < n; ++i) {
                const Key k = RT::encode(p[i]);
                if (descending ? (k < prev) : (prev < k)) return false;
                prev = k;
            }
            if (allow_reverse) std::reverse(p, p + n);
            return allow_reverse;
        }
        return true;
    }
}


template <class T, class = void>
struct has_equal_operator : std::false_type {};
template <class T>
struct has_equal_operator<T, std::void_t<decltype(std::declval<const T&>() == std::declval<const T&>())>>
    : std::true_type {};

enum class FastOrderKind : unsigned char {
    None,
    Sorted,
    Reverse,
    AllEqual
};

// Exact order classification over radix keys with branch-free 256-element
// blocks: both direction flags are reduced per block (vectorisable), and the
// scan stops at the first block where both directions have been violated.
// Sorted input therefore runs at close to memory bandwidth instead of paying
// a data-dependent branch per element; random input stops after one block.
template <class T>
inline FastOrderKind blocked_radix_order_kind(const T* p, std::size_t n, bool descending) {
    using RT = RadixTraits<T>;
    if (n < 2) return FastOrderKind::AllEqual;
    // The first block is short: random input shows both directions within
    // a handful of pairs, and on small ranges a full 256-pair block would
    // cost a sizeable fraction of the whole sort.
    constexpr std::size_t kBlock = 256;
    unsigned up = 0, down = 0;   // saw k[j] > k[j-1] / k[j] < k[j-1]
    for (std::size_t b = 1, blk = 16; b < n && !(up && down); b += blk, blk = kBlock) {
        const std::size_t e = std::min(n, b + blk);
        unsigned bu = 0, bd = 0;
        for (std::size_t j = b; j < e; ++j) {
            const auto k0 = RT::encode(p[j - 1]);
            const auto k1 = RT::encode(p[j]);
            bu |= static_cast<unsigned>(k0 < k1);
            bd |= static_cast<unsigned>(k1 < k0);
        }
        up |= bu;
        down |= bd;
    }
    if (!up && !down) return FastOrderKind::AllEqual;
    if (up && down) return FastOrderKind::None;
    const bool ascending_run = up != 0;
    return ascending_run != descending ? FastOrderKind::Sorted : FastOrderKind::Reverse;
}

// Merge two sorted runs A[0..na) and B[0..nb) into out[0..na+nb) as four
// independent branch-free merges over co-ranked output quarters.  A single
// branch-free merge is bound by its load -> compare -> index dependency
// (~3 ns/key on Ice Lake-SP); four interleaved chains run at ~1.5 ns/key.
// Reads A[na] and B[nb] (never uses them): both must be addressable.
template <class T, class KeyF>
inline void merge_runs_4way(const T* A, std::size_t na, const T* B, std::size_t nb, T* out, KeyF key) {
    constexpr int C = 4;
    using Key  = decltype(key(*A));
    using Bits = typename std::conditional<sizeof(T) == 8, std::uint64_t,
                 typename std::conditional<sizeof(T) == 4, std::uint32_t,
                 typename std::conditional<sizeof(T) == 2, std::uint16_t, std::uint8_t>::type>::type>::type;
    static_assert(sizeof(Bits) == sizeof(T) && std::is_trivially_copyable<T>::value, "raw-bit merge");
    const std::size_t N = na + nb;
    std::size_t sa[C + 1], sb[C + 1];
    for (int c = 0; c <= C; ++c) {
        const std::size_t s = N / C * static_cast<std::size_t>(c) + (c == C ? N % C : 0);
        std::size_t lo = s > nb ? s - nb : 0, hi = s < na ? s : na;
        while (lo < hi) {                  // smallest i with B[s-i-1] < A[i]
            const std::size_t i = lo + (hi - lo) / 2;
            const std::size_t j = s - i;
            if (j > 0 && !(key(B[j - 1]) < key(A[i]))) lo = i + 1; else hi = i;
        }
        sa[c] = lo;
        sb[c] = s - lo;
    }
    std::size_t a[C], ae[C], b[C], be[C], o[C];
    std::size_t steps = N;
    for (int c = 0; c < C; ++c) {
        a[c] = sa[c]; ae[c] = sa[c + 1];
        b[c] = sb[c]; be[c] = sb[c + 1];
        o[c] = sa[c] + sb[c];
        steps = std::min(steps, (sa[c + 1] + sb[c + 1]) - o[c]);
    }
    for (std::size_t t = 0; t < steps; ++t) {
        FYX_VQ_UNROLL for (int c = 0; c < C; ++c) {
            const Key ka = key(A[a[c]]);
            const Key kb = key(B[b[c]]);
            const bool ta = (a[c] < ae[c]) & ((b[c] >= be[c]) | !(kb < ka));
            // Mask select on raw bits: GCC turns value selects on FP into
            // branches, which mispredict on every interleaving.
            if constexpr (std::is_integral<T>::value) {
                out[o[c]++] = ta ? A[a[c]] : B[b[c]];
            } else {
            Bits va, vb;
            std::memcpy(&va, A + a[c], sizeof(T));
            std::memcpy(&vb, B + b[c], sizeof(T));
            const Bits mk = static_cast<Bits>(Bits(0) - static_cast<Bits>(ta));
            const Bits v = static_cast<Bits>((va & mk) | (vb & static_cast<Bits>(~mk)));
            std::memcpy(out + o[c]++, &v, sizeof(T));
            }
            a[c] += static_cast<std::size_t>(ta);
            b[c] += static_cast<std::size_t>(!ta);
        }
    }
    for (int c = 0; c < C; ++c) {
        while (a[c] < ae[c] && b[c] < be[c]) {
            const bool ta = !(key(B[b[c]]) < key(A[a[c]]));
            out[o[c]++] = ta ? A[a[c]] : B[b[c]];
            a[c] += static_cast<std::size_t>(ta);
            b[c] += static_cast<std::size_t>(!ta);
        }
        while (a[c] < ae[c]) out[o[c]++] = A[a[c]++];
        while (b[c] < be[c]) out[o[c]++] = B[b[c]++];
    }
}

template <class T>
inline std::size_t radix_key_find_break(const T* p, std::size_t start, std::size_t hi,
                                        typename RadixTraits<T>::Key prev_key,
                                        const bool want_up, const bool descending);

// Neighbour descents (in target order) over the whole range, counted in
// branch-free 256-pair blocks; stops once the count exceeds `cap` and then
// returns cap + 1.  `first` gets the start of the 256-pair block holding the
// first descent (exact position: radix_key_find_break from there).  With
// `blk`, the starts of the first blk_cap blocks holding a descent go there
// (*nblk of them): every such block when the result is <= blk_cap, so a
// caller can extract the positions from those blocks alone.
template <class T>
inline std::size_t count_descents_capped(const T* p, std::size_t n, bool descending,
                                         std::size_t cap, std::size_t& first, std::size_t from = 1,
                                         std::uint32_t* blk = nullptr, std::size_t blk_cap = 0,
                                         std::size_t* nblk = nullptr) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    std::size_t d = 0, nb = 0;
    first = n;
    for (std::size_t b = from < 1 ? 1 : from; b < n; b += 256) {
        const std::size_t e = std::min(n, b + 256);
        Key bd = 0;                        // same lane width as the keys: vectorizes
        for (std::size_t j = b; j < e; ++j)
            bd = static_cast<Key>(bd + static_cast<Key>(static_cast<Key>(RT::encode(p[j]) ^ flip) <
                                                        static_cast<Key>(RT::encode(p[j - 1]) ^ flip)));
        if (bd != 0) {
            if (first == n) first = b;     // block of the first descent
            if (nb < blk_cap) blk[nb++] = static_cast<std::uint32_t>(b);
        }
        d += bd;
        if (d > cap) break;
    }
    if (nblk != nullptr) *nblk = nb;
    return d > cap ? cap + 1 : d;
}

// Positions of all D descents from the nblk block starts that
// count_descents_capped left in io[] (overwritten with the positions).  Out
// of line: inlined, the extra call site shifts GCC's inlining of the
// numeric dispatcher (1M uint32 nearly-sorted 1.7x slower).
template <class T>
FYX_NOINLINE bool descent_positions_in_blocks(const T* p, std::size_t n, bool descending,
                                              std::uint32_t* io, std::size_t nblk, std::size_t D) {
    std::uint32_t blk[32];
    if (nblk > 32) return false;
    std::memcpy(blk, io, nblk * sizeof(std::uint32_t));
    std::size_t m = 0;
    for (std::size_t t = 0; t < nblk && m < D; ++t)
        m += descent_positions(p, blk[t], std::min<std::size_t>(n, blk[t] + 256u), descending, io + m, D - m);
    return m == D;
}

// Cheap stand-in for "more than n/16 descents" on random-like input: three
// 512-pair windows (head, middle, tail) each over a quarter descents, and
// at least 8 of 64 strided neighbour pairs descending.  A sorted range with
// a few disordered windows fails the strided test and gets the exact count.
// (The exact capped count scans n/8 pairs of a random range: ~26k cycles at
// n = 200000, 1% of its sort.)
template <class T>
inline bool descents_look_dense(const T* p, std::size_t n, bool descending) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    if (n < 8192) return false;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    std::size_t sd = 0;
    for (std::size_t j = 0; j < 64; ++j) {
        const std::size_t i = (j * (n - 1)) >> 6;
        sd += key(p[i + 1]) < key(p[i]);
    }
    if (sd < 8) return false;
    const std::size_t starts[3] = {1, n / 2, n - 512};
    for (std::size_t w = 0; w < 3; ++w) {
        Key bd = 0;
        for (std::size_t j = starts[w]; j < starts[w] + 512; ++j)
            bd = static_cast<Key>(bd + static_cast<Key>(key(p[j]) < key(p[j - 1])));
        if (bd <= 128) return false;
    }
    return true;
}

// True when the first few descents (from `from`) are local: the key after
// each drop still sits at or above the key `reach` places back.  Near-sorted
// input (short displacements) passes; sorted runs, far swaps and random
// tails, whose drops fall far, fail on the first descent.
template <class T>
inline bool descents_look_local(const T* p, std::size_t n, std::size_t from, bool descending,
                                std::size_t reach, int samples) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    std::size_t i = from < 1 ? 1 : from;
    for (int s = 0; s < samples && i < n; ++s) {
        i = radix_key_find_break(p, i, n, RT::encode(p[i - 1]), true, descending);
        if (i >= n) break;
        const std::size_t back = i > reach ? i - reach : 0;
        if (key(p[i]) < key(p[back])) return false;
        ++i;
    }
    return true;
}

// True when the first `samples` descents (from `from`) look like boundaries
// between sorted runs: spaced at least `gap` apart and not explained by one
// misplaced key (dropping neither neighbour restores the order).  Far swaps,
// sparse outliers and dense random stretches all fail it.
template <class T>
inline bool descents_look_like_runs(const T* p, std::size_t n, std::size_t from, bool descending,
                                    std::size_t gap, int samples) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    std::size_t i = from < 1 ? 1 : from, last = 0;
    for (int s = 0; s < samples; ++s) {
        if (i >= n) return false;
        i = radix_key_find_break(p, i, n, RT::encode(p[i - 1]), true, descending);
        if (i + 1 >= n || i < 2) return false;
        if (s != 0 && i - last < gap) return false;
        if (!(key(p[i]) < key(p[i - 2])) || !(key(p[i + 1]) < key(p[i - 1]))) return false;
        last = i;
        ++i;
    }
    return true;
}

// Exactly two ascending runs p[0..brk) and p[brk..n): rotate when the second
// run lies entirely below the first, otherwise one four-way merge.
template <class T>
inline void merge_two_runs(T* p, std::size_t n, std::size_t brk, bool descending) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    if (!(key(p[0]) < key(p[n - 1]))) {          // back <= front: a rotation
        std::rotate(p, p + brk, p + n);
        return;
    }
    const std::size_t nb = n - brk;
    {
        // The thread arena keeps the scratch mapped across calls (a fresh
        // allocation of this size page-faults on every call).
        ScratchLease<T> out(n);
        if (out.valid() && vmerge_runs(p, brk, p + brk, nb, out.get(), descending)) {
            std::memcpy(static_cast<void*>(p), static_cast<const void*>(out.get()), n * sizeof(T));
            return;
        }
    }
    std::unique_ptr<T[]> tmp(new T[nb + 1]);
    std::memcpy(static_cast<void*>(tmp.get()), static_cast<const void*>(p + brk), nb * sizeof(T));
    std::unique_ptr<T[]> out(new T[n]);
    merge_runs_4way(p, brk, tmp.get(), nb, out.get(), key);   // reads p[brk], tmp[nb]
    std::memcpy(static_cast<void*>(p), static_cast<const void*>(out.get()), n * sizeof(T));
}

// p[0..cnt) -> p[s..s+cnt), s >= 1: backward 64-byte chunks through
// registers.  Short overlapping shifts are where a libc memmove call per
// outlier dominated (~100 cycles each on 1000-key inputs).
template <class T>
FYX_FORCE_INLINE void shift_up_small(T* p, std::size_t cnt, std::size_t s) {
    constexpr std::size_t C = 64 / sizeof(T) ? 64 / sizeof(T) : 1;
    std::size_t e = cnt;
    while (e >= C) {
        unsigned char tmp[C * sizeof(T)];
        std::memcpy(tmp, static_cast<const void*>(p + e - C), sizeof(tmp));
        std::memcpy(static_cast<void*>(p + e - C + s), tmp, sizeof(tmp));
        e -= C;
    }
    while (e > 0) { --e; p[e + s] = p[e]; }
}

// Merge of two sorted runs that first tries block moves: while the runs
// interleave in long stretches (block swaps, rotations, concatenations of
// disjoint ranges) each stretch is found by a galloping search and copied
// whole; after a few short stretches the rest goes through the SIMD merge
// (or std::merge).  Radix-key order, so NaN / -0 / descending agree with
// the other run kernels.
template <class T>
inline void merge_runs_galloping(const T* A, std::size_t na, const T* B, std::size_t nb, T* out,
                                 bool descending) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    // Count of leading x in R[0..m) with key(x) < kv (strict) or <= kv.
    auto gallop = [&](const T* R, std::size_t m, Key kv, bool strict) -> std::size_t {
        auto before = [&](std::size_t t) { const Key kr = key(R[t]); return strict ? kr < kv : !(kv < kr); };
        std::size_t hi = 1;
        while (hi < m && before(hi)) hi = 2 * hi + 1;
        std::size_t lo = (hi - 1) / 2 + (hi > 1 ? 1 : 0);
        if (hi > m) hi = m;
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (before(mid)) lo = mid + 1; else hi = mid;
        }
        return lo;
    };
    std::size_t i = 0, j = 0, o = 0;
    int shorts = 0;
    while (i < na && j < nb) {
        std::size_t len;
        if (key(B[j]) < key(A[i])) {
            len = gallop(B + j, nb - j, key(A[i]), true);
            std::memcpy(static_cast<void*>(out + o), static_cast<const void*>(B + j), len * sizeof(T));
            j += len;
        } else {
            len = gallop(A + i, na - i, key(B[j]), false);
            std::memcpy(static_cast<void*>(out + o), static_cast<const void*>(A + i), len * sizeof(T));
            i += len;
        }
        o += len;
        if (len < 32 && ++shorts > 3) break;
    }
    if (i < na && j < nb) {
        auto less = [&](const T& a, const T& b) { return key(a) < key(b); };
        if (!vmerge_runs(A + i, na - i, B + j, nb - j, out + o, descending))
            std::merge(A + i, A + na, B + j, B + nb, out + o, less);
        return;
    }
    if (i < na) std::memcpy(static_cast<void*>(out + o), static_cast<const void*>(A + i), (na - i) * sizeof(T));
    if (j < nb) std::memcpy(static_cast<void*>(out + o), static_cast<const void*>(B + j), (nb - j) * sizeof(T));
}

// Sample gate of try_hash_count_sort (kept out of line: inlined, it
// perturbed the counting loop's code by ~15%).  256 keys in 16 blocks of 16
// consecutive keys (on cold data ~2 lines per block instead of one miss per
// key; block-local clustering can only inflate repeats, i.e. pass more
// inputs on to the prefix check, which decides for itself).  The Chao1
// estimate D + f1^2 / (2 f2) of the total key count (f1 / f2: keys seen
// once / twice) is returned (all-ones: nearly all distinct); the caller
// declines long-tailed inputs on it before any table is touched.  The keys are sorted in registers and their runs counted
// without branches: a hash-probe gate cost ~7000-16000 cycles per call on
// fresh inputs (mispredicted probes; a heap table also arrives cold).
template <class U, std::size_t S>
FYX_NOINLINE std::size_t hash_count_sample_gate(const unsigned char* base, std::size_t n) {
    constexpr std::size_t B = 16;                       // keys per block (64 measured slower)
    static_assert(S % B == 0, "whole blocks");
    alignas(64) U smp[S];
    for (std::size_t j = 0; j < S; ++j)
        std::memcpy(&smp[j], base + (((j / B) * (n - B)) / (S / B) + (j % B)) * sizeof(U), sizeof(U));
    // Random-like input leaves on the first 128 keys holding no repeat
    // (1536 uniform keys: P(no repeat) ~ 0.5%); one leaf-sized sort.
    auto sort_n = [&](std::size_t m) {
#if FYX_HAS_AVX512_CODE
        if (use_avx512()) { vqsort_serial<U>(smp, m); return; }
#endif
        std::sort(smp, smp + m);
    };
    sort_n(128);
    {
        unsigned rep = 0;
        for (std::size_t j = 1; j < 128; ++j) rep |= static_cast<unsigned>(smp[j] == smp[j - 1]);
        if (!rep) return ~std::size_t(0);
    }
    sort_n(S);
    // A run of length L ends at j when smp[j] != smp[j + 1]; it adds one to
    // D, and to f1 / f2 when L is 1 / 2 -- i.e. when smp[j - 1] / smp[j - 2]
    // also differ from smp[j].
    std::size_t sd = 0, f1 = 0, f2 = 0;
    for (std::size_t j = 0; j < S; ++j) {
        const bool end = j + 1 == S || smp[j] != smp[j + 1];
        const bool s1  = j < 1 || smp[j - 1] != smp[j];
        const bool s2  = !s1 && (j < 2 || smp[j - 2] != smp[j]);
        sd += end;
        f1 += end & s1;
        f2 += end & s2;
    }
    if (sd > S - S / 16) return ~std::size_t(0);
    return sd + f1 * f1 / (2 * (f2 ? f2 : 1));
}

// Moderate-cardinality sort (a few hundred to ~2000 distinct keys, beyond
// the <= 32-key vector counter): one read-only pass counts bit patterns in
// an open-addressing table, then the distinct keys are ordered by radix key
// and written back as runs.  ~1.5 ns/key against ~3 ns/key for a vectorised
// quicksort on 64-bit keys, and no NaN / -0 special cases (bit patterns are
// counted, radix keys give the library's total order).  A 256-key strided
// sample must show duplicates first; the pass itself declines -- range
// untouched -- once the table holds more than kCap keys.
template <class T>
inline bool try_hash_count_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T> || !(sizeof(T) == 4 || sizeof(T) == 8)) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        using U   = typename std::conditional<sizeof(T) == 8, std::uint64_t, std::uint32_t>::type;
        if (n < 4096 || n > 0xFFFFFFFFull) return false;
        auto bits = [](const T& x) { U u; std::memcpy(&u, &x, sizeof(U)); return u; };
        auto mix = [](U k, unsigned b) {
            return static_cast<std::size_t>((static_cast<std::uint64_t>(k) * 0x9E3779B97F4A7C15ull) >> (64 - b));
        };
        constexpr unsigned kBits = 13;
        constexpr std::size_t kSlots = std::size_t(1) << kBits, kMask = kSlots - 1, kCap = 2048;
        // Estimated key count (Chao1 over a sample; larger inputs afford 512
        // keys, which separate a long tail (zipf) from ~1500 uniform keys
        // better than 256 do) must fit the table (3/4 of kCap) and leave
        // enough keys per distinct value: below n/24 (8-byte) or n/32
        // (4-byte) the cold table and the distinct-key sort cost more than
        // the quicksort they replace.  4-byte keys also need K > n/640: the
        // 16-lane quicksort finishes cheaply once partitions turn
        // single-valued above its 512-key leaf (n = 200000, 256 keys: 1.4
        // ns/key against 2.0 for the table; 447 keys -- leaf-sized values --
        // 3.3 against 2.2).
        const unsigned char* const raw = reinterpret_cast<const unsigned char*>(p);
        const std::size_t est = n >= 65536 ? hash_count_sample_gate<U, 512>(raw, n)
                                           : hash_count_sample_gate<U, 256>(raw, n);
        const std::size_t limit = std::min<std::size_t>(kCap * 3 / 4, n / (sizeof(T) == 8 ? 24 : 32));
        if (est > limit) return false;
        if (sizeof(T) == 4 && est <= n / 640) return false;
        // Tables from the thread arena (a fresh 96 KB new/delete pair per call
        // can be trimmed back to the OS and page-faulted in again).
        // Layout: keys | cnt | ord (Key) | vals (T) | runs | slot list.
        constexpr std::size_t oCnt  = kSlots * sizeof(U);
        constexpr std::size_t oOrd  = oCnt + kSlots * sizeof(std::uint32_t);
        constexpr std::size_t oVals = oOrd + kCap * sizeof(Key);
        constexpr std::size_t oRuns = oVals + kCap * sizeof(T);
        constexpr std::size_t oLst  = oRuns + kCap * sizeof(std::uint32_t);
        ScratchLease<unsigned char> lease(oLst + (kCap + 1) * sizeof(std::uint16_t));
        if (!lease.valid()) return false;
        unsigned char* const mem = lease.get();
        U* const keys = reinterpret_cast<U*>(mem);
        std::uint32_t* const cnt = reinterpret_cast<std::uint32_t*>(mem + oCnt);
        Key* const ord = reinterpret_cast<Key*>(mem + oOrd);
        T* const vals = reinterpret_cast<T*>(mem + oVals);
        std::uint32_t* const runs = reinterpret_cast<std::uint32_t*>(mem + oRuns);
        std::uint16_t* const lst = reinterpret_cast<std::uint16_t*>(mem + oLst);
        std::memset(cnt, 0, kSlots * sizeof(std::uint32_t));
        std::size_t D = 0;
        // A fixed key set stops adding keys early; a long-tailed one (zipf)
        // keeps producing singletons.  At checkpoints (a prefix of
        // max(n/64, 512) keys, then each doubling up to n/4) the Chao1
        // estimate D + f1^2 / (2 f2) of the total key count (f1 / f2: keys
        // seen once / twice) must stay within the table, so a doomed pass
        // stops early.  A single prefix check let borderline zipf inputs run
        // on to the cap (~20 us at n = 20000).
        // lst: slots in insertion order, so extraction never scans the table.
        std::size_t f1 = 0, f2 = 0;
        std::size_t i = 0;
        std::size_t chk = std::min(n, std::max<std::size_t>(n / 64, 512));
        for (;;) {
            for (; i < chk; ++i) {
                const U k = bits(p[i]);
                std::size_t h = mix(k, kBits);
                std::uint32_t c = cnt[h];
                // Short-circuit on purpose: an empty slot's key line is
                // never read, so first touches of the (usually cold) key
                // table are stores, not load misses.
                while (c != 0 && keys[h] != k) {
                    h = (h + 1) & kMask;
                    c = cnt[h];
                }
                lst[D] = static_cast<std::uint16_t>(h);
                D += c == 0;
                if (D > kCap) return false;            // table stays <= 1/4 full
                keys[h] = k;
                f1 += static_cast<std::size_t>(c == 0) - static_cast<std::size_t>(c == 1);
                f2 += static_cast<std::size_t>(c == 1) - static_cast<std::size_t>(c == 2);
                cnt[h] = c + 1;
            }
            if (D + f1 * f1 / (2 * (f2 ? f2 : 1)) > limit) return false;
            if (chk >= n / 4) break;
            chk = std::min(n / 4, chk * 2);
        }
        for (; i < n; ++i) {
            const U k = bits(p[i]);
            std::size_t h = mix(k, kBits);
            while (cnt[h] != 0 && keys[h] != k) h = (h + 1) & kMask;
            if (cnt[h] == 0) {
                if (D >= kCap) return false;
                lst[D++] = static_cast<std::uint16_t>(h);
                keys[h] = k;
            }
            ++cnt[h];
        }
        // Distinct keys in output order: radix-encode, sort the D keys (one
        // SIMD sort when available), then decode and re-probe for the count.
        // (A table scan plus std::sort of key/count records cost ~40 us at
        // n = 20000, 600 keys.)
        const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
        for (std::size_t t = 0; t < D; ++t) {
            T x;
            std::memcpy(&x, &keys[lst[t]], sizeof(U));
            ord[t] = static_cast<Key>(RT::encode(x) ^ flip);
        }
#if FYX_HAS_AVX512_CODE
        if (use_avx512()) vqsort_serial<Key>(ord, D);
        else
#endif
            std::sort(ord, ord + D);
        // Decode and re-probe for counts; the probe chain from mix(k) holds
        // only occupied slots up to k's.
        for (std::size_t t = 0; t < D; ++t) {
            const T x = RT::decode(static_cast<Key>(ord[t] ^ flip));
            const U k = bits(x);
            std::size_t h = mix(k, kBits);
            while (keys[h] != k) h = (h + 1) & kMask;
            vals[t] = x;
            runs[t] = cnt[h];
        }
#if FYX_HAS_AVX512_CODE
        if (use_avx512()) {
            vfill_runs_dispatch<T, std::uint32_t>(p, n, vals, runs, D);
            return true;
        }
#endif
        T* q = p;
        for (std::size_t t = 0; t < D; ++t) {
            std::fill(q, q + runs[t], vals[t]);
            q += runs[t];
        }
        return true;
    }
}

#ifndef FYX_FUSED_REVERSE_MIN_BYTES
#define FYX_FUSED_REVERSE_MIN_BYTES (std::size_t(64) << 10)
#endif
inline constexpr std::size_t kFusedReverseMinBytes = FYX_FUSED_REVERSE_MIN_BYTES;

/// First key strictly above the last in sort order and 16 strided samples
/// non-increasing: worth a fused verify+reverse attempt.
template <class T>
inline bool looks_reversed(const T* p, std::size_t n, bool descending) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    if (n < 64 || !(key(p[n - 1]) < key(p[0]))) return false;
    constexpr std::size_t S = 16;
    Key prev = key(p[0]);
    for (std::size_t s = 1; s <= S; ++s) {
        const Key k = key(p[(n - 1) * s / S]);
        if (prev < k) return false;
        prev = k;
    }
    return true;
}

/// Front probe of the small-vq dispatcher: 1 bitwise all-equal; 2 reversed
/// in place (looks_reversed, then the checked two-ended reverse without
/// undo -- a break leaves a permutation); 0 neither.  Out of line, one call
/// in place of the inlined all-equal test: an extra call site in the
/// dispatcher shifted GCC's inlining of the paths after it (i32 two-run / V
/// shapes 15-20% slower with the reverse attempt never taken).
template <class T>
FYX_NOINLINE int front_order_probe(T* p, std::size_t n, bool descending) {
    if (range_bitwise_all_equal(p, n)) return 1;
    if (n * sizeof(T) >= kFusedReverseMinBytes && looks_reversed(p, n, descending) &&
        reverse_tail_checked<T, false>(p, 0, n, descending) == 0)
        return 2;
    return 0;
}

// A handful of monotone runs (organ pipe, block swaps, concatenations, a
// rotation, mixed ascending / descending stretches): split into maximal runs
// (descending ones reversed), then pairwise merges ping-ponging through one
// leased buffer, each through the SIMD two-run merge when it applies.
// Detection is read-only and gives up past `max_runs` runs, so a declined
// range is untouched; random data declines within a few dozen keys.
template <class T>
inline bool try_few_runs_merge(T* p, std::size_t n, bool descending, std::size_t mono = 0) {
    constexpr std::size_t kMaxRuns = 8;
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    std::size_t bnd[kMaxRuns + 1];
    bool rev[kMaxRuns];
    const std::size_t min_run = std::max<std::size_t>(32, n / 64);
    // [lo, n) non-increasing?  Then reverse it in the same pass: blocks from
    // both ends are checked (OR-reduced, vectorizes) and swapped.  A break
    // swaps the done blocks back.  A strided sample gates the attempt.
    auto fused_reverse_tail = [&](std::size_t lo) -> bool {
        constexpr std::size_t S = 16, B = 64;
        if (n - lo < 4 * B) return false;
        Key prev = key(p[lo]);
        for (std::size_t s = 1; s <= S; ++s) {
            const Key kq = key(p[lo + (n - 1 - lo) * s / S]);
            if (prev < kq) return false;
            prev = kq;
        }
        // Vector kernel: hardware order on clean keys; a NaN / -0 hands
        // back to the key-order loop below (input restored).
        const int vk = reverse_tail_checked(p, lo, n, descending);
        if (vk != 2) return vk == 0;
        std::size_t l = lo, r = n;
        auto undo = [&]() {
            for (std::size_t k = 0; k < l - lo; ++k) std::swap(p[lo + k], p[n - 1 - k]);
        };
        while (r - l >= 2 * B) {
            Key bad = 0;
            for (std::size_t k = 0; k < B; ++k)
                bad = static_cast<Key>(bad | static_cast<Key>(key(p[l + k]) < key(p[l + k + 1])));
            for (std::size_t k = 0; k < B; ++k)
                bad = static_cast<Key>(bad | static_cast<Key>(key(p[r - B + k - 1]) < key(p[r - B + k])));
            if (bad) { undo(); return false; }
            for (std::size_t k = 0; k < B; ++k) std::swap(p[l + k], p[r - 1 - k]);
            l += B;
            r -= B;
        }
        for (std::size_t k = l; k + 1 < r; ++k)
            if (key(p[k]) < key(p[k + 1])) { undo(); return false; }
        std::reverse(p + l, p + r);
        return true;
    };
    std::size_t R = 0, i = 0;
    while (i < n) {
        if (R == kMaxRuns) return false;
        std::size_t start = i;
        std::size_t j = i + 1;
        const Key k0 = key(p[i]);
        while (j < n && key(p[j]) == k0) ++j;
        const bool down = j < n && key(p[j]) < k0;
        // A peak (ascending run, then a descending one) joins the
        // descending run: p[start - 1] >= p[start], so that run stays
        // non-increasing, and once reversed it starts at the peak -- the
        // classic organ pipe (sort, reverse the back half) then coalesces
        // with no data movement.
        if (down && R > 0 && !rev[R - 1] && start - bnd[R - 1] > min_run) --start;
        bool done = false;
        if (down && j < n && fused_reverse_tail(start)) { j = n; done = true; }
        // mono: no descent among the first `mono` keys (caller's scan).
        if (i == 0 && !down && j < mono) j = mono;
        // vectorised run end (non-strict in the run's own direction)
        if (j < n) j = radix_key_find_break(p, j, n, RT::encode(p[j - 1]), !down, descending);
        // Short runs mean local disorder (insertion's job) or noise.
        if (j - start < min_run) {
            if (done) std::reverse(p + start, p + n);   // leave a permutation of the input
            return false;
        }
        bnd[R] = start;
        rev[R] = down && !done;
        ++R;
        i = j;
    }
    bnd[R] = n;
    if (R < 2) return false;
    for (std::size_t r = 0; r < R; ++r)
        if (rev[r]) std::reverse(p + bnd[r], p + bnd[r + 1]);
    // Neighbours already in order (value-disjoint runs: organ pipe built by
    // reversing a sorted half, concatenations of ascending blocks) coalesce
    // for free; a pair in swapped order is one rotation.  Saves the merge
    // pass and the copy-back.
    {
        std::size_t w = 1;
        for (std::size_t r = 1; r < R; ++r) {
            if (!(key(p[bnd[r]]) < key(p[bnd[r] - 1]))) continue;
            bnd[w++] = bnd[r];
        }
        bnd[w] = n;
        R = w;
        if (R == 1) return true;
        if (R == 2 && !(key(p[0]) < key(p[n - 1]))) {
            // max(second) <= min(first): swap the blocks.
            std::rotate(p, p + bnd[1], p + n);
            return true;
        }
    }
    if (R == 2) {
        // Two runs: keys of the first run up to min(second) and of the
        // second from max(first) on are already final (binary searches);
        // merge only the overlap in place, buffering its smaller side.  An
        // organ pipe whose peak landed in the first run overlaps in one
        // key: one block move instead of merge-to-buffer plus copy-back.
        const std::size_t mid = bnd[1];
        const Key kb0 = key(p[mid]), ka1 = key(p[mid - 1]);
        std::size_t lo = 0, hi = mid;                      // first key(p[i]) > kb0 in [0, mid)
        while (lo < hi) { const std::size_t m = lo + (hi - lo) / 2; if (kb0 < key(p[m])) hi = m; else lo = m + 1; }
        const std::size_t s = lo;
        lo = mid; hi = n;                                  // first key(p[i]) >= ka1 in [mid, n)
        while (lo < hi) { const std::size_t m = lo + (hi - lo) / 2; if (key(p[m]) < ka1) lo = m + 1; else hi = m; }
        const std::size_t t = lo;
        const std::size_t a = mid - s, b = t - mid;        // both >= 1 (runs did not coalesce)
        const std::size_t m = std::min(a, b);
        ScratchLease<T> lease(m);
        std::unique_ptr<T[]> priv;
        if (!lease.valid()) priv.reset(new T[m]);
        T* tmp = lease.valid() ? lease.get() : priv.get();
        auto lower = [&](std::size_t l, std::size_t h, Key k) {   // first key >= k
            while (l < h) { const std::size_t q = l + (h - l) / 2; if (key(p[q]) < k) l = q + 1; else h = q; }
            return l;
        };
        auto upper = [&](std::size_t l, std::size_t h, Key k) {   // first key > k
            while (l < h) { const std::size_t q = l + (h - l) / 2; if (k < key(p[q])) h = q; else l = q + 1; }
            return l;
        };
        const bool gallop = m * 16 < std::max(a, b);
        if (a <= b) {
            // Forward: write index o never passes the next unread B slot.
            std::memcpy(static_cast<void*>(tmp), static_cast<const void*>(p + s), a * sizeof(T));
            std::size_t o = s, i = 0, j = mid;
            if (gallop) {
                for (; i < a; ++i) {
                    const std::size_t e = lower(j, t, key(tmp[i]));
                    std::memmove(static_cast<void*>(p + o), static_cast<const void*>(p + j), (e - j) * sizeof(T));
                    o += e - j; j = e;
                    p[o++] = tmp[i];
                }
            } else {
                while (i < a && j < t) {
                    const bool tb = key(p[j]) < key(tmp[i]);
                    p[o++] = tb ? p[j] : tmp[i];
                    j += tb ? 1u : 0u;
                    i += tb ? 0u : 1u;
                }
                if (i < a) std::memcpy(static_cast<void*>(p + o), static_cast<const void*>(tmp + i), (a - i) * sizeof(T));
            }
        } else {
            // Backward: write index o never passes below the next unread A slot.
            std::memcpy(static_cast<void*>(tmp), static_cast<const void*>(p + mid), b * sizeof(T));
            std::size_t o = t, i = b, j = mid;              // counts / ends
            if (gallop) {
                for (; i > 0; --i) {
                    const std::size_t e = upper(s, j, key(tmp[i - 1]));
                    std::memmove(static_cast<void*>(p + o - (j - e)), static_cast<const void*>(p + e), (j - e) * sizeof(T));
                    o -= j - e; j = e;
                    p[--o] = tmp[i - 1];
                }
            } else {
                while (i > 0 && j > s) {
                    const bool ta = key(tmp[i - 1]) < key(p[j - 1]);
                    p[--o] = ta ? p[j - 1] : tmp[i - 1];
                    j -= ta ? 1u : 0u;
                    i -= ta ? 0u : 1u;
                }
                if (i > 0) std::memcpy(static_cast<void*>(p + s), static_cast<const void*>(tmp), i * sizeof(T));
            }
        }
        return true;
    }
    ScratchLease<T> lease(n);
    std::unique_ptr<T[]> priv;
    if (!lease.valid()) priv.reset(new T[n]);
    T* buf = lease.valid() ? lease.get() : priv.get();
    T* src = p;
    T* dst = buf;
    while (R > 1) {
        std::size_t w = 0;
        for (std::size_t r = 0; r < R; r += 2) {
            const std::size_t lo = bnd[r];
            if (r + 1 == R) {
                std::memcpy(static_cast<void*>(dst + lo), static_cast<const void*>(src + lo),
                            (bnd[r + 1] - lo) * sizeof(T));
            } else {
                const std::size_t mid = bnd[r + 1], hi = bnd[r + 2];
                merge_runs_galloping(src + lo, mid - lo, src + mid, hi - mid, dst + lo, descending);
            }
            bnd[w++] = lo;
        }
        bnd[w] = n;
        R = w;
        std::swap(src, dst);
    }
    if (src != p) std::memcpy(static_cast<void*>(p), static_cast<const void*>(src), n * sizeof(T));
    return true;
}

// Extract-and-merge repair for "sorted with a minority out of place": a
// sorted range with a few far displacements, a sorted prefix followed by an
// unsorted tail, two (or a few) concatenated sorted runs.  One pass keeps a
// non-decreasing subsequence in place (compacted to the front) and moves the
// rest to a side buffer: on a descent either the new key or the previous kept
// key -- whichever breaks the order -- is set aside.  The side buffer is then
// sorted (if it is not already) and merged back from the right, branch-free.
// Cost on success: one scan + sort(side) + one merge, against a full sort.
//
// Random-like input is rejected after a few hundred keys: the abort tests are
// (a) the kept subsequence falls below half of what was scanned, (b) the side
// buffer turns disordered (descents inside it beyond i/8 + 64 or n/16 + 64),
// (c) the side buffer exceeds n/2.  On abort the side keys are written back
// behind the kept ones, so the range is still a permutation of the input.
template <class T, class SideSort>
inline bool try_extract_merge_repair(T* p, std::size_t n, bool descending, SideSort&& side_sort) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    if (n < 256) return false;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };

    const std::size_t cap = n / 2 + 1;
    // One leased block holds the side buffer and the merge output; the side
    // sort below nests and so allocates privately if it needs scratch.
    ScratchLease<T> lease(cap + n);
    std::unique_ptr<T[]> priv;
    if (!lease.valid()) priv.reset(new T[cap + n]);
    T* const sd  = lease.valid() ? lease.get() : priv.get();
    T* const obuf = sd + cap;
    std::size_t m = 0;
    std::size_t k = 1, side_desc = 0;
    Key klast = key(p[0]);
    Key sidelast = 0;
    bool ok = true;
    bool kept_last = true;                 // retry the block skip only after a keep
    constexpr std::size_t kBlk = 16;
    std::size_t i = 1;
    for (; i < n; ++i) {
        // Ordered stretches move 16 keys at a time (branch-free test, then a
        // block copy -- nothing to copy at all before the first extraction).
        while (kept_last && i + kBlk <= n) {
            // Same-width lane sum: vectorizes (a shifted OR mask does not).
            Key bd = static_cast<Key>(key(p[i]) < klast);
            for (std::size_t j = 1; j < kBlk; ++j)
                bd = static_cast<Key>(bd + static_cast<Key>(key(p[i + j]) < key(p[i + j - 1])));
            std::size_t f = kBlk;
            if (bd != 0) {
                // Keep the ordered prefix in one go and hand the first
                // descent to the scalar step (re-testing the same block
                // key by key cost ~16x near every outlier).
                f = 0;
                if (!(key(p[i]) < klast)) {
                    f = 1;
                    while (!(key(p[i + f]) < key(p[i + f - 1]))) ++f;
                }
            }
            if (f != 0) {
                if (k != i) {
                    // k < i (may overlap): forward element copy, no libc call.
                    for (std::size_t j = 0; j < f; ++j) p[k + j] = p[i + j];
                }
                k += f;
                i += f;
                klast = key(p[k - 1]);
            }
            if (f != kBlk) break;
        }
        if (i >= n) break;
        const T x = p[i];
        const Key kx = key(x);
        if (!(kx < klast)) {               // in order: keep
            p[k++] = x;
            klast = kx;
            kept_last = true;
            continue;
        }
        kept_last = false;
        // Descent.  If at most 8 kept keys exceed x (and at least one kept
        // key does not), those are the outliers: move them aside and keep x.
        // Otherwise x is the outlier.
        std::size_t j = 1;
        while (j < 8 && j < k && kx < key(p[k - 1 - j])) ++j;
        if (j < k && !(kx < key(p[k - 1 - j]))) {
            for (std::size_t q = k - j; q < k; ++q) {
                const Key kq = key(p[q]);
                if (m != 0 && kq < sidelast) ++side_desc;
                sidelast = kq;
                sd[m++] = p[q];
            }
            k -= j;
            p[k++] = x;
            klast = kx;
        } else {
            if (m != 0 && kx < sidelast) ++side_desc;
            sidelast = kx;
            sd[m++] = x;
        }
        if ((i & 255u) == 0 || m + 8 >= cap) {
            if (2 * k + 64 < i || m + 8 >= cap ||
                side_desc > (i >> 3) + 64 || side_desc > (n >> 4) + 64) {
                ok = false;
                ++i;
                break;
            }
        }
    }
    if (ok && side_desc > (n >> 4) + 64) ok = false;
    if (!ok) {
        // p[0..k) kept, side holds the rest of [0, i), p[i..n) untouched.
        std::copy(sd, sd + m, p + k);
        return false;
    }
    if (m == 0) return true;
    if (side_desc != 0) side_sort(sd, m);
    // Backward merge: kept run in p[0..k), side run in side[0..m), k + m == n.
    if (m * 32 < k) {
        // Few side keys: binary-search each one's slot and shift whole kept
        // blocks (memmove bandwidth instead of a compare per key).
        std::size_t hi = k;                // kept keys [0, hi) not yet placed
        for (std::size_t b = m; b-- > 0;) {
            const Key kb = key(sd[b]);
            std::size_t lo = 0, h = hi;    // first kept index with key > kb
            while (lo < h) {
                const std::size_t mid = lo + (h - lo) / 2;
                if (kb < key(p[mid])) h = mid; else lo = mid + 1;
            }
            const std::size_t cnt = hi - lo;
            if (cnt) std::memmove(static_cast<void*>(p + lo + b + 1), static_cast<const void*>(p + lo), cnt * sizeof(T));
            p[lo + b] = sd[b];
            hi = lo;
        }
        return true;
    }
    // Large side: four-way merge into a scratch buffer, then copy back.
    // (A[k] = p[k] is addressable since m >= 1; sd has cap >= m + 1.)
    if (!vmerge_runs(p, k, sd, m, obuf, descending))
        merge_runs_4way(p, k, sd, m, obuf, key);
    std::memcpy(static_cast<void*>(p), static_cast<const void*>(obuf), n * sizeof(T));
    return true;
}


// Few isolated outliers in an otherwise sorted range (far swaps, a handful
// of misplaced keys): locate every descent with a vectorized block test,
// blame one key per descent, verify the remainder is ordered, then compact,
// sort the (<= 2 * kMaxD) outliers and insert them.  Only a constant number of
// data-dependent branches per descent, so it stays fast on cold predictors
// where the streaming extractor pays several mispredictions per outlier.
// Returns false (range untouched) when the blame/verify step does not hold.
template <class T>
inline bool sparse_outlier_repair_at(T* p, std::size_t n, const std::uint32_t* pos, std::size_t D,
                                     bool descending) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr std::size_t kMaxD = 32;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    if (n < 64 || D > kMaxD) return false;
    if (D == 0) return true;
    // Blame: p[i-1] when dropping it leaves p[i-2] <= p[i], else p[i] when
    // dropping it leaves p[i-1] <= p[i+1].  rem[] stays ascending.
    std::size_t rem[kMaxD];
    std::size_t r = 0;
    for (std::size_t d = 0; d < D; ++d) {
        const std::size_t i = pos[d];
        const bool prev_removed = r != 0 && rem[r - 1] == i - 1;
        if (!prev_removed && (i < 2 || !(key(p[i]) < key(p[i - 2])))) {
            rem[r++] = i - 1;
        } else if (i + 1 >= n || !(key(p[i + 1]) < key(p[i - 1]))) {
            rem[r++] = i;
        } else {
            return false;
        }
    }
    // Verify: every kept neighbour pair spanning a removed index is ordered
    // (pairs not touching one were not descents).
    for (std::size_t q = 0; q < r;) {
        std::size_t q2 = q;
        while (q2 + 1 < r && rem[q2 + 1] == rem[q2] + 1) ++q2;    // consecutive
        const std::size_t lo = rem[q], hi = rem[q2];
        if (lo > 0 && hi + 1 < n && key(p[hi + 1]) < key(p[lo - 1])) return false;
        q = q2 + 1;
    }
    // Outliers to a small buffer, insertion-sorted by key.
    T out[kMaxD];
    for (std::size_t q = 0; q < r; ++q) out[q] = p[rem[q]];
    for (std::size_t a = 1; a < r; ++a) {
        const T x = out[a];
        const Key kx = key(x);
        std::size_t b = a;
        while (b > 0 && kx < key(out[b - 1])) { out[b] = out[b - 1]; --b; }
        out[b] = x;
    }
    // Kept keys are addressed in place: kept index k lives at k plus the
    // number of removed indices at or before it (rem[] ascending, few).
    auto orig = [&](std::size_t k) {
        for (std::size_t q = 0; q < r && rem[q] <= k; ++q) ++k;
        return k;
    };
    // Insertion slots (first kept index with key > outlier), all searches
    // in lockstep so their load latencies overlap.
    const std::size_t kept = n - r;
    std::size_t slot[kMaxD];
    Key okey[kMaxD];
    for (std::size_t b = 0; b < r; ++b) { slot[b] = 0; okey[b] = key(out[b]); }
    std::size_t len = kept;
    while (len > 1) {
        const std::size_t half = len / 2;
        for (std::size_t b = 0; b < r; ++b)
            slot[b] += (std::size_t(0) - static_cast<std::size_t>(!(okey[b] < key(p[orig(slot[b] + half)])))) & half;
        len -= half;
    }
    if (len == 1)
        for (std::size_t b = 0; b < r; ++b) slot[b] += static_cast<std::size_t>(!(okey[b] < key(p[orig(slot[b])])));
    // Kept key k moves from k + R(k) (removed before it) to k + C(k)
    // (outliers placed before it): piecewise constant between the
    // breakpoints rem[q] - q and slot[b].  Segments with no net shift --
    // everything between the two ends of a far swap -- stay where they
    // are; left-moving segments go left to right, then right-moving ones
    // right to left (C, R non-decreasing: no move overwrites a source that
    // is still to be read).
    std::size_t bp[2 * kMaxD + 2];
    std::size_t nb = 0;
    for (std::size_t q = 0; q < r; ++q) {
        const std::size_t x[2] = {rem[q] - q, slot[q]};
        for (std::size_t x1 : x) {
            if (x1 == 0 || x1 >= kept) continue;
            std::size_t j = nb++;
            while (j > 0 && bp[j - 1] > x1) { bp[j] = bp[j - 1]; --j; }
            bp[j] = x1;
        }
    }
    {
        std::size_t u = 0;
        for (std::size_t j = 0; j < nb; ++j) if (u == 0 || bp[u - 1] != bp[j]) bp[u++] = bp[j];
        nb = u;
    }
    struct Seg { std::size_t k0, len, src, dst; };
    Seg seg[2 * kMaxD + 2];
    std::size_t ns = 0;
    {
        std::size_t k0 = 0, R = 0, C = 0;
        for (std::size_t t = 0; t <= nb; ++t) {
            const std::size_t k1 = t < nb ? bp[t] : kept;
            while (R < r && rem[R] - R <= k0) ++R;
            while (C < r && slot[C] <= k0) ++C;
            if (k1 > k0) seg[ns++] = Seg{k0, k1 - k0, k0 + R, k0 + C};
            k0 = k1;
        }
    }
    for (std::size_t t = 0; t < ns; ++t)
        if (seg[t].dst < seg[t].src)
            std::memmove(static_cast<void*>(p + seg[t].dst), static_cast<const void*>(p + seg[t].src), seg[t].len * sizeof(T));
    for (std::size_t t = ns; t-- > 0;)
        if (seg[t].dst > seg[t].src)
            std::memmove(static_cast<void*>(p + seg[t].dst), static_cast<const void*>(p + seg[t].src), seg[t].len * sizeof(T));
    for (std::size_t b = 0; b < r; ++b) p[slot[b] + b] = out[b];
    return true;
}

template <class T>
inline bool try_sparse_outlier_repair(T* p, std::size_t n, std::size_t from, bool descending) {
    if (n > 0xFFFFFFFFull) return false;
    std::uint32_t pos[32];
    const std::size_t D = descent_positions(p, from, n, descending, pos, 32);
    return D <= 32 && sparse_outlier_repair_at(p, n, pos, D, descending);
}

// Insertion repair over radix keys (the library's total order, so NaN / -0
// are placed exactly as the radix paths place them) with a hard budget on
// element moves.  Near-sorted input -- a few local displacements -- finishes
// in one predictable pass; anything else exhausts the budget early and
// returns false, leaving a permutation of the input for the caller's sort.
template <class T>
inline bool budgeted_insertion_repair(T* p, std::size_t n, bool descending,
                                      std::size_t budget) {
    using RT = RadixTraits<T>;
    constexpr std::size_t kSkip = 16;
    std::size_t moves = 0;
    std::size_t i = 1;
    while (i < n) {
        // Branch-free skip over ordered stretches, 16 neighbours at a time.
        while (i + kSkip <= n) {
            unsigned bad = 0;
            for (std::size_t k = 0; k < kSkip; ++k) {
                const auto a = RT::encode(p[i + k - 1]);
                const auto b = RT::encode(p[i + k]);
                bad |= static_cast<unsigned>(descending ? (a < b) : (b < a));
            }
            if (bad) break;
            i += kSkip;
        }
        const std::size_t e = std::min(n, i + kSkip);
        for (; i < e; ++i) {
        const auto ki = RT::encode(p[i]);
        const auto kp = RT::encode(p[i - 1]);
        if (descending ? !(kp < ki) : !(ki < kp)) continue;
        const T x = p[i];
        std::size_t j = i;
        do {
            p[j] = p[j - 1];
            --j;
        } while (j > 0 && (descending ? (RT::encode(p[j - 1]) < ki) : (ki < RT::encode(p[j - 1]))));
        p[j] = x;
        moves += i - j;
        if (moves > budget) return false;
        }
    }
    return true;
}

// Same hint as a predicate: true when the first `limit` neighbour pairs hold
// at most `max_inv` inversions.  Counts in blocks of 32 pairs and stops as
// soon as the allowance is exceeded, so random input -- half its pairs
// inverted -- is rejected after the first block instead of after `limit`.
template <class T>
inline bool head_inversions_within(const T* p, std::size_t n, bool descending,
                                   std::size_t limit, std::size_t max_inv) {
    using RT = RadixTraits<T>;
    const std::size_t m = std::min(n == 0 ? std::size_t(0) : n - 1, limit);
    std::size_t c = 0;
    for (std::size_t b = 0; b < m; b += 32) {
        const std::size_t e = std::min(m, b + 32);
        unsigned bc = 0;
        if (descending) {
            for (std::size_t i = b; i < e; ++i)
                bc += static_cast<unsigned>(RT::encode(p[i]) < RT::encode(p[i + 1]));
        } else {
            for (std::size_t i = b; i < e; ++i)
                bc += static_cast<unsigned>(RT::encode(p[i + 1]) < RT::encode(p[i]));
        }
        c += bc;
        if (c > max_inv) return false;
    }
    return true;
}

template <class T>
FYX_FORCE_INLINE bool fast_string_equal_value(const T* p, std::size_t n) {
    (void)p; (void)n;
    return false;
}

template <>
FYX_FORCE_INLINE bool fast_string_equal_value<std::string>(const std::string* p, std::size_t n) {
#if FYX_USE_STRING_VIEW
    if (n < 2) return true;
    const std::string& first = p[0];
    const char* first_data = first.data();
    const std::size_t first_size = first.size();
    for (std::size_t i = 1; i < n; ++i) {
        const std::string& cur = p[i];
        if (cur.size() != first_size) return false;
        if (first_size != 0 && std::char_traits<char>::compare(cur.data(), first_data, first_size) != 0)
            return false;
    }
    return true;
#else
    (void)p; (void)n;
    return false;
#endif
}

// First descent found by the serial order exit's ascending scan, so the
// affix weapon tried right after it (nothing in between moves elements)
// need not rescan the ordered head.  Reset with the vq memo per top-level
// call; consumed (cleared) by its reader; checked against (p, n).
struct OrderBreakHint {
    const void* p = nullptr;
    std::size_t n = 0;
    std::size_t head = 0;
};
inline OrderBreakHint& order_break_hint() noexcept {
    static thread_local OrderBreakHint h;
    return h;
}

// any_adjacent_pair: does pred(p[j], p[j-1]) (asc) / pred(p[j-1], p[j])
// (!asc) hold for some j in [lo, hi)?  OR-reduced with no early exit so an
// inlined key comparator vectorizes; the ISA-targeted copies let a portable
// build reach 64-bit vector compares (SSE2 has none).
#define FYX_ANY_ADJ_BODY                                                                       \
    unsigned char bad = 0;                                                                    \
    if (asc) for (std::size_t j = lo; j < hi; ++j) bad |= static_cast<unsigned char>(pred(p[j], p[j - 1])); \
    else     for (std::size_t j = lo; j < hi; ++j) bad |= static_cast<unsigned char>(pred(p[j - 1], p[j])); \
    return bad != 0;
template <class T, class Pred>
inline bool any_adjacent_pair_default(const T* p, std::size_t lo, std::size_t hi, Pred& pred, bool asc) {
    FYX_ANY_ADJ_BODY
}
#if FYX_ARCH_X86 && FYX_GNUC_LIKE && !defined(__AVX512BW__)
template <class T, class Pred>
FYX_TARGET_AVX512 bool any_adjacent_pair_avx512(const T* p, std::size_t lo, std::size_t hi, Pred& pred, bool asc) {
    FYX_ANY_ADJ_BODY
}
template <class T, class Pred>
FYX_TARGET_AVX2 bool any_adjacent_pair_avx2(const T* p, std::size_t lo, std::size_t hi, Pred& pred, bool asc) {
    FYX_ANY_ADJ_BODY
}
#endif
#undef FYX_ANY_ADJ_BODY
template <class T, class Pred>
inline bool any_adjacent_pair(const T* p, std::size_t lo, std::size_t hi, Pred& pred, bool asc) {
#if FYX_ARCH_X86 && FYX_GNUC_LIKE && !defined(__AVX512BW__)
    if (use_avx512()) return any_adjacent_pair_avx512(p, lo, hi, pred, asc);
    if (use_avx2())   return any_adjacent_pair_avx2(p, lo, hi, pred, asc);
#endif
    return any_adjacent_pair_default(p, lo, hi, pred, asc);
}

template <class T, class Comp>
// `all_equal_possible` is a promise from the caller: pass false when something
// outside this range already witnessed two strictly ordered elements, which
// makes the all-equal answer impossible and lets the SIMD all-equal memcmp --
// a full extra read of the range -- be skipped.  An all-equal range still
// classifies as AllEqual without it (every adjacent pair compares equal, so the
// monotonicity loop below falls through), so the flag only removes work.
inline FastOrderKind detect_fast_order_kind(T* p, std::size_t n, Comp comp,
                                           bool all_equal_possible = true) {
    if (n < 2) return FastOrderKind::AllEqual;
#if !FYX_ENABLE_FAST_PATHS
    (void)p; (void)comp; (void)all_equal_possible;
    return FastOrderKind::None;
#else
    if (all_equal_possible) {
        if constexpr (std::is_arithmetic<T>::value && !std::is_same<T, bool>::value &&
                      std::is_trivially_copyable<T>::value) {
            if (std::memcmp(p, p + 1, (n - 1) * sizeof(T)) == 0)
                return FastOrderKind::AllEqual;
        }
    }
    constexpr bool radix_order = radix_supported_v<T> && std::is_floating_point<T>::value &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    constexpr bool blocked_order = radix_supported_v<T> &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    if constexpr (blocked_order) {
        return blocked_radix_order_kind(p, n, is_descending_v<Comp, T>);
    } else if constexpr (radix_order) {
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        const bool descending = is_descending_v<Comp, T>;
        Key prev = RT::encode(p[0]);
        for (std::size_t i = 1; i < n; ++i) {
            Key cur = RT::encode(p[i]);
            if (cur == prev) continue;

            const bool already_in_target_order = descending ? (cur < prev) : (prev < cur);
            prev = cur;
            if (already_in_target_order) {
                for (++i; i < n; ++i) {
                    cur = RT::encode(p[i]);
                    if (descending ? (prev < cur) : (cur < prev)) return FastOrderKind::None;
                    prev = cur;
                }
                return FastOrderKind::Sorted;
            }

            for (++i; i < n; ++i) {
                cur = RT::encode(p[i]);
                if (descending ? (cur < prev) : (prev < cur)) return FastOrderKind::None;
                prev = cur;
            }
            return FastOrderKind::Reverse;
        }
        return FastOrderKind::AllEqual;
    } else {
        // Exact-value equality is the cheapest all-equal proof for strings and
        // trivial payloads.  It is also safe for custom comparators because a
        // strict weak ordering cannot order an object before itself.
        if constexpr (std::is_same<T, std::string>::value) {
            if (fast_string_equal_value(p, n)) return FastOrderKind::AllEqual;
        } else if constexpr (has_equal_operator<T>::value) {
            const T& first = p[0];
            std::size_t i = 1;
            for (; i < n; ++i) {
                if (!(p[i] == first)) break;
            }
            if (i == n) return FastOrderKind::AllEqual;
        }

        // Once the direction is known the rest is a block-wise OR-reduced
        // scan (vectorizes for plain key comparators, ISA-dispatched), with
        // an exit per block.
        // Returns the block holding the first break (n if none).
        auto any_break = [&](std::size_t from, bool asc) -> std::size_t {
            constexpr std::size_t B = 1024;
            for (std::size_t b = from; b < n; b += B)
                if (any_adjacent_pair(p, b, std::min(n, b + B), comp, asc)) return b;
            return n;
        };
        for (std::size_t i = 1; i < n; ++i) {
            if (comp(p[i], p[i - 1]))            // reverse of comp order
                return any_break(i + 1, false) != n ? FastOrderKind::None : FastOrderKind::Reverse;
            if (comp(p[i - 1], p[i])) {          // already in comp order
                std::size_t b = any_break(i + 1, true);
                if (b == n) return FastOrderKind::Sorted;
                while (!comp(p[b], p[b - 1])) ++b;
                OrderBreakHint& h = order_break_hint();
                h.p = p; h.n = n; h.head = b;
                return FastOrderKind::None;
            }
        }
        return FastOrderKind::AllEqual;
    }
#endif
}

template <class T, class Comp>
inline bool pdq_preferred_order_sample(T* p, std::size_t n, Comp comp) {
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n; (void)comp;
    return false;
#else
    if (n < std::size_t(1024)) return false;
    constexpr bool radix_order = radix_supported_v<T> &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    const std::size_t contiguous = std::min<std::size_t>(n, 2048);
    if (contiguous < 8) return false;

    if constexpr (std::is_arithmetic<T>::value && !std::is_same<T, bool>::value) {
        constexpr std::size_t Cap = 512;
        constexpr std::size_t Mask = Cap - 1;
        std::array<std::uint64_t, Cap> seen{};
        std::array<unsigned char, Cap> used{};
        std::size_t distinct = 0;
        for (std::size_t i = 0; i < contiguous; ++i) {
            std::uint64_t key = 0;
            if constexpr (radix_order) {
                key = static_cast<std::uint64_t>(RadixTraits<T>::encode(p[i]));
            } else {
                std::memcpy(&key, &p[i], sizeof(T));
            }
            std::uint64_t hbits = key;
            hbits ^= hbits >> 33;
            hbits *= 0xff51afd7ed558ccdULL;
            hbits ^= hbits >> 33;
            std::size_t h = static_cast<std::size_t>(hbits) & Mask;
            for (;;) {
                if (!used[h]) {
                    used[h] = 1;
                    seen[h] = key;
                    if (++distinct > kCountingClassLimit) goto high_distinct_sample;
                    break;
                }
                if (seen[h] == key) break;
                h = (h + 1) & Mask;
            }
        }
        return false;
    high_distinct_sample: ;
    }

    auto before = [&](const T& a, const T& b) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            using Key = typename RT::Key;
            const Key ka = RT::encode(a);
            const Key kb = RT::encode(b);
            if constexpr (is_descending_v<Comp, T>) return kb < ka;
            else return ka < kb;
        } else {
            return comp(a, b);
        }
    };

    std::size_t inv = 0;
    std::size_t ordered = 0;
    std::size_t turns = 0;
    int prev_dir = 0;
    for (std::size_t i = 1; i < contiguous; ++i) {
        const bool down = before(p[i], p[i - 1]);
        const bool up = before(p[i - 1], p[i]);
        if (down) ++inv;
        if (up || down) ++ordered;
        const int dir = up ? 1 : (down ? -1 : 0);
        if (dir != 0) {
            if (prev_dir != 0 && dir != prev_dir) ++turns;
            prev_dir = dir;
        }
    }
    if (ordered == 0) return false;

    // Nearly sorted inputs are pdqsort's best case; random inputs have about
    // 50% local inversions, so this does not steal high-entropy radix/sample
    // wins.  Zigzag/organ-pipe style inputs flip direction almost every step;
    // pdqsort handles those structured partitions better than a full radix or
    // sample-sort permutation at 1M-scale.
    if (inv * 20 <= ordered) return true; // <= 5% inversions
    if (turns * 10 >= ordered * 9 && inv * 100 >= ordered * 35 && inv * 100 <= ordered * 65)
        return true;
    return false;
#endif
}


template <class T, class Comp>
inline void pdqsort_for_profile_pattern(T* p, std::size_t n, Comp comp);

template <class T, class Comp>
inline bool try_nearly_sorted_insertion_repair(T* p, std::size_t n, Comp comp) {
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n; (void)comp;
    return false;
#else
    if (n < std::size_t(1024)) return false;
    constexpr bool radix_order = radix_supported_v<T> && std::is_floating_point<T>::value &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    auto before = [&](const T& a, const T& b) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            using Key = typename RT::Key;
            const Key ka = RT::encode(a);
            const Key kb = RT::encode(b);
            if constexpr (is_descending_v<Comp, T>) return kb < ka;
            else return ka < kb;
        } else {
            return comp(a, b);
        }
    };
    if constexpr (!std::is_move_constructible<T>::value || !std::is_move_assignable<T>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        // Bounded insertion repair is pdqsort's killer case for genuinely
        // local disorder.  Long-distance random swaps look "nearly sorted" by
        // adjacent-inversion count, but insertion would slowly bubble a remote
        // element across a huge clean run.  Stop early and let the patch/merge
        // repair below handle that shape; callers fall back to pdqsort if the
        // patch proof rejects the partially repaired permutation.
        const std::size_t max_shifts = std::max<std::size_t>(4096, n / 1024);
        std::size_t shifts = 0;
        for (std::size_t i = 1; i < n; ++i) {
            if (!before(p[i], p[i - 1])) continue;
            T v = std::move(p[i]);
            std::size_t j = i;
            while (j > 0 && before(v, p[j - 1])) {
                p[j] = std::move(p[j - 1]);
                --j;
                if (++shifts > max_shifts) {
                    p[j] = std::move(v);
                    return false;
                }
            }
            p[j] = std::move(v);
        }
        return true;
    }
#endif
}



template <class T, class Comp>
inline bool try_bounded_insertion_repair(T* p, std::size_t n, Comp comp,
                                          bool thorough = false) {
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n; (void)comp; (void)thorough;
    return false;
#else
    if (n < std::size_t(1024)) return false;
    constexpr bool radix_order = radix_supported_v<T> && std::is_floating_point<T>::value &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    auto before = [&](const T& a, const T& b) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            using Key = typename RT::Key;
            const Key ka = RT::encode(a);
            const Key kb = RT::encode(b);
            if constexpr (is_descending_v<Comp, T>) return kb < ka;
            else return ka < kb;
        } else {
            return comp(a, b);
        }
    };
    if constexpr (!std::is_move_constructible<T>::value || !std::is_move_assignable<T>::value ||
                  !std::is_copy_constructible<T>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        // Two very different callers share this probe, and they can afford
        // different things.  At the top of a sort, before the profile runs, it
        // only pays for a sample: it is there for the zigzag / sawtooth
        // family, whose disorder is dense but travels one position, and which
        // the profile classifies as high entropy and will not repair.  Sparse
        // disorder is left alone there -- the profile claims it and the repair
        // chain below serves it better than this probe can.  Called from that
        // chain the range is already known to be nearly sorted, a scan is
        // already paid for, and sparse disorder is exactly what to look for:
        // an array whose blocks were permuted has a few dozen inversions in a
        // million positions and the first four thousand of them may be
        // perfectly ordered, which no density test can see.
        const std::size_t sample_n = std::min<std::size_t>(n, std::size_t(4096));
        std::size_t inv = 0, ordered = 0;
        for (std::size_t i = 1; i < sample_n; ++i) {
            const bool down = before(p[i], p[i - 1]);
            const bool up   = before(p[i - 1], p[i]);
            if (down) ++inv;
            if (up || down) ++ordered;
        }
        if (ordered == 0) return false;
        if (!thorough && inv * 100u < ordered * 8u) return false;

        // Low-cardinality random data has plenty of local inversions but no
        // local structure to exploit: counting sort owns it, and insertion
        // would drag values across the whole range.
        if constexpr (std::is_arithmetic<T>::value && !std::is_same<T, bool>::value) {
            constexpr std::size_t Cap = 512;
            constexpr std::size_t Mask = Cap - 1;
            std::array<std::uint64_t, Cap> seen{};
            std::array<unsigned char, Cap> used{};
            std::size_t distinct = 0;
            const std::size_t dn = std::min<std::size_t>(sample_n, std::size_t(512));
            for (std::size_t i = 0; i < dn; ++i) {
                std::uint64_t key = 0;
                if constexpr (radix_order) {
                    key = static_cast<std::uint64_t>(RadixTraits<T>::encode(p[i]));
                } else {
                    std::memcpy(&key, &p[i], sizeof(T));
                }
                std::uint64_t hbits = key;
                hbits ^= hbits >> 33;
                hbits *= 0xff51afd7ed558ccdULL;
                hbits ^= hbits >> 33;
                std::size_t h = static_cast<std::size_t>(hbits) & Mask;
                for (;;) {
                    if (!used[h]) {
                        used[h] = 1;
                        seen[h] = key;
                        ++distinct;
                        break;
                    }
                    if (seen[h] == key) break;
                    h = (h + 1u) & Mask;
                }
            }
            if (distinct <= 64u) return false;
        } else if constexpr (std::is_same<T, std::string>::value) {
            std::vector<std::string> distinct;
            distinct.reserve(129);
            const std::size_t dn = std::min<std::size_t>(sample_n, std::size_t(512));
            for (std::size_t i = 0; i < dn; ++i) {
                bool found = false;
                for (const auto& s : distinct) {
                    if (s == p[i]) { found = true; break; }
                }
                if (!found) {
                    distinct.push_back(p[i]);
                    if (distinct.size() > 128u) break;
                }
            }
            if (distinct.size() <= 64u) return false;
        }

        // Insertion sort permutes the range as it goes, so before touching a
        // single element it rehearses on a copy of the prefix: the same
        // insertion sort over the first four thousand positions, abandoned as
        // soon as they cost more shifts than they contain.  Disorder that is
        // dense but expensive -- interleaved runs, random data -- is refused
        // here with the range still exactly as it was, which is what leaves
        // the detectors downstream able to recognise it.  Sparse disorder
        // costs nothing to rehearse and is let through.
        {
            std::vector<T> probe(p, p + sample_n);
            std::size_t cost = 0;
            for (std::size_t i = 1; i < sample_n; ++i) {
                if (!before(probe[i], probe[i - 1])) continue;
                T v = std::move(probe[i]);
                std::size_t j = i;
                while (j > 0 && before(v, probe[j - 1])) {
                    probe[j] = std::move(probe[j - 1]);
                    --j;
                    if (++cost > sample_n) return false;
                }
                probe[j] = std::move(v);
            }
        }

        // Insertion costs one pass plus the inversion count, so the repair may
        // spend a small multiple of n shifts and still beat a radix pass.
        // Three guards keep it honest, and none of them is a sample: a sample
        // cannot see disorder this sparse.  Long travel is refused outright --
        // an element crossing an eighth of the range is a block move, and the
        // run merge and the patch merges own those.  The absolute budget is
        // two shifts per element.  And the running rate is capped, with a
        // strike to spare, because the shifts arrive in bursts: one permuted
        // block is tens of thousands of them at a single position, while
        // random input breaks the cap at every checkpoint from the first.
        const std::size_t reach = n / 8u + 1u;
        std::size_t shifts = 0, strikes = 0, next_check = 64, walk_until = 0;
        for (std::size_t i = 1; i < n; ++i) {
            if (!before(p[i], p[i - 1])) {
                // Skip an ordered stretch with one vectorized block test
                // (only positions ahead of i are read; none were moved yet).
                // A block that holds a descent is walked element-wise.
                if (i >= walk_until) {
                    const std::size_t e = std::min(n, i + 128u);
                    if (i + 16u < e && !any_adjacent_pair(p, i + 1, e, before, true)) i = e - 1;
                    else walk_until = e;
                }
                continue;
            }
            // Refuse long travel before paying for it: p[i] below the element
            // `reach` back must cross more than that, and a lone intruder at
            // i-1 above the element `reach` ahead would be bubbled forward one
            // shift per element all the way (far swaps: ~2 ms of shifts per
            // 1M records before the old in-loop check fired).
            if (i >= reach && before(p[i], p[i - reach])) return false;
            // Two neighbours that both travel more than 4096 back are the
            // front of a moved block (each one would pay the whole distance).
            if (i >= 4096 && i + 1 < n && before(p[i + 1], p[i - 4096]) &&
                before(p[i], p[i - 4096])) return false;
            if (i >= 2 && i + reach < n && !before(p[i], p[i - 2]) &&
                before(p[i + reach], p[i - 1])) return false;
            T v = std::move(p[i]);
            std::size_t j = i;
            while (j > 0 && before(v, p[j - 1])) {
                p[j] = std::move(p[j - 1]);
                --j;
                if (((i - j) & 63u) == 0u && i - j > reach) {
                    p[j] = std::move(v);
                    return false;
                }
            }
            p[j] = std::move(v);
            shifts += i - j;
            if (shifts > n * 2u) return false;
            if (i >= next_check) {
                strikes = (shifts > i * 8u) ? strikes + 1u : 0u;
                if (strikes >= 2u) return false;
                next_check <<= 1;
            }
        }
        return true;
    }
#endif
}

template <class T, class Comp>
inline bool try_adjacent_swap_repair(T* p, std::size_t n, Comp comp) {
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n; (void)comp;
    return false;
#else
    if (n < std::size_t(1024)) return false;
    constexpr bool radix_order = radix_supported_v<T> && std::is_floating_point<T>::value &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    auto before = [&](const T& a, const T& b) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            using Key = typename RT::Key;
            const Key ka = RT::encode(a);
            const Key kb = RT::encode(b);
            if constexpr (is_descending_v<Comp, T>) return kb < ka;
            else return ka < kb;
        } else {
            return comp(a, b);
        }
    };
    if constexpr (!std::is_move_constructible<T>::value || !std::is_move_assignable<T>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        // Pair-swapped/adjacent-zigzag inputs are common in adversarial suites:
        // [1,0,3,2,...] or the same shape with strings.  The existing
        // interleaved-run merge sorted them correctly, but paid for a full
        // temporary array and (for strings) millions of extra moves.  Prove in
        // one logical scan that swapping only disjoint inverted neighbours would
        // make the whole range ordered, then apply exactly those swaps.  Random
        // long-distance nearly-sorted inputs fail the proof before mutation and
        // can continue to the patch/merge repair.
        std::vector<std::size_t> swaps;
        bool dense_pairs = true;
        std::size_t first_pair = n;
        std::size_t expected_pair = n;
        std::size_t pair_count = 0;
        auto remember_pair = [&](std::size_t idx) {
            if (pair_count == 0) {
                first_pair = idx;
                expected_pair = idx;
            } else {
                expected_pair += 2;
                if (dense_pairs && idx != expected_pair) {
                    dense_pairs = false;
                    swaps.reserve(64);
                    for (std::size_t j = first_pair; j < expected_pair; j += 2)
                        swaps.push_back(j);
                }
            }
            if (!dense_pairs) swaps.push_back(idx);
            ++pair_count;
        };

        const T* prev = nullptr;
        std::size_t i = 0;
        while (i < n) {
            if (i + 1 < n && before(p[i + 1], p[i])) {
                const T& first = p[i + 1];
                const T& second = p[i];
                if (prev && before(first, *prev)) return false;
                remember_pair(i);
                prev = &second;
                i += 2;
            } else {
                if (prev && before(p[i], *prev)) return false;
                prev = p + i;
                ++i;
            }
        }
        if (pair_count == 0) return false;
        if (dense_pairs) {
            std::size_t idx = first_pair;
            for (std::size_t c = 0; c < pair_count; ++c, idx += 2)
                std::iter_swap(p + idx, p + idx + 1);
        } else {
            for (std::size_t idx : swaps)
                std::iter_swap(p + idx, p + idx + 1);
        }
        return true;
    }
#endif
}


template <class T, class Comp>
inline bool try_nearly_sorted_repair(T* p, std::size_t n, Comp comp) {
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n; (void)comp;
    return false;
#else
    if (n < std::size_t(1024)) return false;
    constexpr bool radix_order = radix_supported_v<T> && std::is_floating_point<T>::value &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    auto before = [&](const T& a, const T& b) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            using Key = typename RT::Key;
            const Key ka = RT::encode(a);
            const Key kb = RT::encode(b);
            if constexpr (is_descending_v<Comp, T>) return kb < ka;
            else return ka < kb;
        } else {
            return comp(a, b);
        }
    };
    if constexpr (!std::is_move_constructible<T>::value || !std::is_move_assignable<T>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        const std::size_t max_dirty = std::max<std::size_t>(64, n / 8);
        std::vector<unsigned char> dirty(n, 0);
        std::size_t dirty_count = 0;
        auto mark_dirty = [&](std::size_t idx) {
            if (!dirty[idx]) {
                dirty[idx] = 1;
                ++dirty_count;
            }
        };
        for (std::size_t i = 1; i < n; ++i) {
            if (before(p[i], p[i - 1])) {
                mark_dirty(i - 1);
                mark_dirty(i);
                if (dirty_count > max_dirty) return false;
            }
        }
        if (dirty_count == 0) return true;

        bool clean_ordered = false;
        for (unsigned pass = 0; pass < 4; ++pass) {
            bool changed = false;
            std::size_t last_clean = n;
            for (std::size_t i = 0; i < n; ++i) {
                if (dirty[i]) continue;
                if (last_clean != n && before(p[i], p[last_clean])) {
                    mark_dirty(last_clean);
                    mark_dirty(i);
                    if (dirty_count > max_dirty) return false;
                    changed = true;
                    last_clean = n;
                    continue;
                }
                last_clean = i;
            }
            if (!changed) { clean_ordered = true; break; }
        }
        if (!clean_ordered) return false;

        std::vector<T> clean;
        std::vector<T> patch;
        clean.reserve(n - dirty_count);
        patch.reserve(dirty_count);
        for (std::size_t i = 0; i < n; ++i) {
            if (dirty[i]) patch.push_back(std::move(p[i]));
            else          clean.push_back(std::move(p[i]));
        }
        std::sort(patch.begin(), patch.end(), before);
        std::size_t ci = 0, pi = 0, out = 0;
        while (ci < clean.size() && pi < patch.size()) {
            if (before(patch[pi], clean[ci])) p[out++] = std::move(patch[pi++]);
            else                              p[out++] = std::move(clean[ci++]);
        }
        while (ci < clean.size())  p[out++] = std::move(clean[ci++]);
        while (pi < patch.size())  p[out++] = std::move(patch[pi++]);
        return true;
    }
#endif
}


/// `patch_dirty_max` caps how much disorder the patch merges may take on:
/// kPatchDirtyDefault leaves their own budget of n/8, zero skips them.
/// Callers that hold a kernel which scales better than three sequential
/// passes lower it -- see patch_merge_dirty_budget.
template <class T, class Comp>
inline bool try_partially_sorted_local_repair(T* p, std::size_t n, Comp comp,
                                              std::size_t patch_dirty_max = kPatchDirtyDefault) {
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n; (void)comp; (void)patch_dirty_max;
    return false;
#else
    const std::size_t patch_max =
        (patch_dirty_max == kPatchDirtyDefault) ? n / 8 : patch_dirty_max;
    if (try_adjacent_swap_repair(p, n, comp)) return true;
    // Insertion costs one pass plus the distance the displaced elements
    // actually travel, so it is the cheapest repair that exists for shapes
    // whose disorder is short-range even when it is spread over the whole
    // range (permuted blocks, scattered local edits): every position is
    // looked at once and only the displaced elements move.  It polices its
    // own budget as it goes, so a shape it cannot finish costs a fraction of a
    // pass.  The patch merges below move every element at least twice even
    // when they succeed.
    if (try_bounded_insertion_repair(p, n, comp, true)) return true;
    // Sparse disorder is the common shape: a few percent of the positions take
    // part in an inversion while the clean subsequence is already ordered.
    // Pulling those positions out and merging them back costs three sequential
    // passes instead of 4-8 radix passes, and it is width-independent, so
    // int64/double gain the most.  Densely disordered input is rejected after
    // touching about an eighth of the range, before anything is moved.
    if (patch_max && try_dirty_patch_merge_adaptive(p, n, comp, patch_max)) return true;
    // Adjacent inversions cannot see a block that was moved wholesale: every
    // element inside it is still in order.  The prefix-max / suffix-min
    // characterisation finds those, so spliced / block-moved inputs also cost
    // a couple of linear passes instead of a full sort.
    if (patch_max && try_displacement_patch_merge_adaptive(p, n, comp, patch_max)) return true;
    if (try_nearly_sorted_insertion_repair(p, n, comp)) return true;
    return false;
#endif
}

template <class T, class Comp>
inline bool try_partially_sorted_repair(T* p, std::size_t n, Comp comp) {
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n; (void)comp;
    return false;
#else
    if (try_adjacent_swap_repair(p, n, comp)) return true;
    // Bounded insertion comes first here too, for the same reason as in
    // try_partially_sorted_local_repair and not despite the expensive
    // comparisons: it is the cheapest repair when it fires and the cheapest
    // one to decline.  1M 16-character strings whose 128-element blocks were
    // permuted twenty times sort in 0.012s this way, against 0.037s through
    // the patch merge that used to run first -- and on shapes it refuses it
    // costs 0.0003s, where refusing a patch merge costs two full passes of
    // comparisons, 0.021s.  A comparison per shift is not the expensive part;
    // two passes over the whole range are.
    if (try_bounded_insertion_repair(p, n, comp, true)) return true;
    // For string/object payloads the patch merge replaces an O(n log n)
    // comparison recursion with three sequential move passes plus a sort of
    // the (tiny) dirty patch.
    if (try_dirty_patch_merge_adaptive(p, n, comp)) return true;
    // See above: moved blocks and long-distance splices.
    if (try_displacement_patch_merge_adaptive(p, n, comp)) return true;
    if (try_nearly_sorted_repair(p, n, comp)) return true;
    if (try_nearly_sorted_insertion_repair(p, n, comp)) return true;
    return false;
#endif
}

template <class T, class Comp>
inline void pdqsort_for_profile_pattern(T* p, std::size_t n, Comp comp) {
    constexpr bool radix_order = radix_supported_v<T> &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    if constexpr (radix_order && std::is_floating_point<T>::value) {
        using RT = RadixTraits<T>;
        if constexpr (is_descending_v<Comp, T>) {
            pdqsort(p, p + n, [](const T& a, const T& b) {
                return RT::encode(b) < RT::encode(a);
            });
        } else {
            pdqsort(p, p + n, [](const T& a, const T& b) {
                return RT::encode(a) < RT::encode(b);
            });
        }
    } else {
        pdqsort(p, p + n, comp);
    }
}




template <class T, class Comp>
inline bool try_zigzag_organ_pipe_sort(T* p, std::size_t n, Comp comp) {
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n; (void)comp;
    return false;
#else
    if (n < std::size_t(1024)) return false;
    const std::size_t mid = n / 2u;
    if (mid < 2 || mid >= n) return false;
    constexpr bool radix_order = radix_supported_v<T> && std::is_floating_point<T>::value &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    auto before = [&](const T& a, const T& b) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            using Key = typename RT::Key;
            const Key ka = RT::encode(a);
            const Key kb = RT::encode(b);
            if constexpr (is_descending_v<Comp, T>) return kb < ka;
            else return ka < kb;
        } else {
            return comp(a, b);
        }
    };
    if constexpr (!std::is_move_constructible<T>::value || !std::is_move_assignable<T>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        // bench_final.py's zigzag generator sorts random data, reverses only
        // data[0:mid], then leaves data[mid:n] ascending.  This is cheaper than
        // a full bitonic merge: once we prove the prefix is non-increasing, the
        // suffix is non-decreasing, and max(prefix) <= min(suffix), reversing
        // the prefix alone sorts the whole range.  Put this before the generic
        // reverse detector, which would otherwise classify the descending head
        // as a failed reverse run and fall back to pdqsort.
        const std::size_t probe = std::min<std::size_t>(mid, std::size_t(64));
        bool saw_prefix_down = false;
        bool saw_suffix_up = false;
        for (std::size_t i = 1; i < probe; ++i) {
            const bool prefix_up = before(p[i - 1], p[i]);
            if (prefix_up) return false;
            saw_prefix_down = saw_prefix_down || before(p[i], p[i - 1]);

            const std::size_t si = mid + i;
            if (si < n) {
                const bool suffix_down = before(p[si], p[si - 1]);
                if (suffix_down) return false;
                saw_suffix_up = saw_suffix_up || before(p[si - 1], p[si]);
            }
        }
        if (!saw_prefix_down || !saw_suffix_up) return false;
        if (before(p[mid], p[0])) return false;

        // Full shape check: the prefix must stay non-increasing over
        // [probe-1, mid) and the suffix non-decreasing over [mid+probe-1, n).
        // On an even n those two ranges have the same length, so read them in
        // one fused pass -- two sequential 64 MB scans measured ~15% of an 8M
        // double zigzag sort, and fusing them halves the scan loop overhead
        // while touching every element exactly once per range either way.
        const std::size_t plen = mid - probe;            // prefix edges left
        const std::size_t slen = n - (mid + probe);      // suffix edges left
        const std::size_t fused = plen < slen ? plen : slen;
        for (std::size_t i = 0; i < fused; ++i) {
            if (before(p[probe - 1 + i], p[probe + i])) return false;              // prefix went up
            if (before(p[mid + probe + i], p[mid + probe - 1 + i])) return false;  // suffix went down
        }
        // odd-n tail (at most one edge more in the suffix than the prefix)
        for (std::size_t i = probe + fused; i < mid; ++i) {
            if (before(p[i - 1], p[i])) return false;
        }
        for (std::size_t i = mid + probe + fused; i < n; ++i) {
            if (before(p[i], p[i - 1])) return false;
        }
        std::reverse(p, p + mid);
        return true;
    }
#endif
}


// Minimum size for proof-only pool assists and striped swaps (see the
// orderedness-exit section); declared here because the reverse exit uses it.
inline constexpr std::size_t kParallelProofMinN      = 128u << 10;
inline constexpr std::size_t kParallelProofMinBytes  = 3u << 20;

template <class T, class Comp>
inline bool try_fast_reverse_exit(T* p, std::size_t n, Comp comp, bool allow_parallel = false) {
#if !FYX_ENABLE_FAST_PATHS
    (void)p; (void)n; (void)comp; (void)allow_parallel;
    return false;
#else
    if (n < 2) return false;
    // The fused verify+swap loop below is one serial bandwidth-bound pass and
    // leaves every other core idle.  When the caller may use threads and the
    // array is at least a few megabytes, decline: the parallel orderedness
    // proof verifies striped and reverse_range_adaptive swaps striped -- two
    // parallel passes beat one serial one, by more on machines with more cores.
    if (allow_parallel && n * sizeof(T) >= kParallelProofMinBytes) return false;
    constexpr bool radix_order = radix_supported_v<T> && std::is_floating_point<T>::value &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    auto before = [&](const T& a, const T& b) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            using Key = typename RT::Key;
            const Key ka = RT::encode(a);
            const Key kb = RT::encode(b);
            if constexpr (is_descending_v<Comp, T>) return kb < ka;
            else return ka < kb;
        } else {
            return comp(a, b);
        }
    };
    if constexpr ((std::is_arithmetic<T>::value && sizeof(T) < 4) ||
                  !std::is_move_constructible<T>::value || !std::is_move_assignable<T>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        // Require an immediately descending first pair so random/sorted inputs
        // pay only one or two comparisons before falling back to the normal
        // detector.  Reverse inputs then verify while swapping, avoiding the
        // old full proof scan followed by a second reversal pass.
        if (!before(p[1], p[0]) || before(p[0], p[1])) return false;
        const std::size_t probe = std::min<std::size_t>(n, 64);
        for (std::size_t i = 2; i < probe; ++i) {
            if (before(p[i - 1], p[i])) return false;
        }

        std::size_t l = 0;
        std::size_t r = n - 1;
        while (l < r) {
            if (l + 1 < n && before(p[l], p[l + 1])) {
                pdqsort_for_profile_pattern(p, n, comp);
                return true;
            }
            if (r > l + 1 && before(p[r - 1], p[r])) {
                pdqsort_for_profile_pattern(p, n, comp);
                return true;
            }
            std::iter_swap(p + l, p + r);
            ++l;
            --r;
        }
        return true;
    }
#endif
}


template <class T, class Comp>
inline bool try_interleaved_runs_sort(T* p, std::size_t n, Comp comp) {
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n; (void)comp;
    return false;
#else
    if (n < std::size_t(1024)) return false;
    constexpr bool radix_order = radix_supported_v<T> && std::is_floating_point<T>::value &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    auto before = [&](const T& a, const T& b) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            using Key = typename RT::Key;
            const Key ka = RT::encode(a);
            const Key kb = RT::encode(b);
            if constexpr (is_descending_v<Comp, T>) return kb < ka;
            else return ka < kb;
        } else {
            return comp(a, b);
        }
    };
    if constexpr (!std::is_move_constructible<T>::value || !std::is_move_assignable<T>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        // Gate this path on a genuinely alternating adjacent pattern.  Nearly
        // sorted inputs often have monotone even/odd subsequences too, but they
        // should stay on the pdq/repair path instead of paying a full merge.
        const std::size_t probe = std::min<std::size_t>(n, std::size_t(257));
        std::size_t ordered = 0, inv = 0, turns = 0;
        int prev_dir = 0;
        for (std::size_t i = 1; i < probe; ++i) {
            const bool down = before(p[i], p[i - 1]);
            const bool up = before(p[i - 1], p[i]);
            if (down) ++inv;
            if (up || down) ++ordered;
            const int dir = up ? 1 : (down ? -1 : 0);
            if (dir != 0) {
                if (prev_dir != 0 && dir != prev_dir) ++turns;
                prev_dir = dir;
            }
        }
        if (ordered == 0) return false;
        if (turns * 10 < ordered * 8) return false;
        if (inv * 100 < ordered * 20 || inv * 100 > ordered * 80) return false;

        bool even_asc = true, even_desc = true, odd_asc = true, odd_desc = true;
        const std::size_t stride_probe = std::min<std::size_t>(n, std::size_t(513));
        for (std::size_t i = 2; i < stride_probe && (even_asc || even_desc); i += 2) {
            if (before(p[i], p[i - 2])) even_asc = false;
            if (before(p[i - 2], p[i])) even_desc = false;
        }
        for (std::size_t i = 3; i < stride_probe && (odd_asc || odd_desc); i += 2) {
            if (before(p[i], p[i - 2])) odd_asc = false;
            if (before(p[i - 2], p[i])) odd_desc = false;
        }
        if ((!even_asc && !even_desc) || (!odd_asc && !odd_desc)) return false;

        struct Cursor {
            std::size_t cur;
            bool have;
            bool forward;
        };
        auto make_cursor = [&](std::size_t parity, bool asc) -> Cursor {
            if (parity >= n) return Cursor{0, false, true};
            if (asc) return Cursor{parity, true, true};
            std::size_t last = ((n - 1) & ~std::size_t(1)) | parity;
            if (last >= n) last -= 2;
            return Cursor{last, true, false};
        };
        auto advance = [&](Cursor& c) {
            if (c.forward) {
                c.cur += 2;
                if (c.cur >= n) c.have = false;
            } else {
                if (c.cur >= 2) c.cur -= 2;
                else { c.have = false; return; }
            }
        };

        auto run_one = [&](bool even_forward, bool odd_forward) -> bool {
            auto run_first = [&](std::size_t parity, bool forward) -> std::size_t {
                return make_cursor(parity, forward).cur;
            };
            auto run_last = [&](std::size_t parity, bool forward) -> std::size_t {
                return make_cursor(parity, !forward).cur;
            };
            const bool have_even = n >= 1;
            const bool have_odd = n >= 2;
            int concat = 0; // 0 = merge, 1 = even then odd, 2 = odd then even
            if (have_even && have_odd) {
                const std::size_t ef = run_first(0, even_forward);
                const std::size_t el = run_last(0, even_forward);
                const std::size_t of = run_first(1, odd_forward);
                const std::size_t ol = run_last(1, odd_forward);
                if (!before(p[of], p[el])) concat = 1;
                else if (!before(p[ef], p[ol])) concat = 2;
            } else {
                concat = 1;
            }

            auto run_length = [&](std::size_t parity) -> std::size_t {
                return parity >= n ? std::size_t(0) : ((n - 1 - parity) / 2 + 1);
            };
            auto try_concat_reorder = [&](std::size_t first_parity, bool first_forward,
                                          std::size_t second_parity, bool second_forward) -> bool {
                const std::size_t first_len = run_length(first_parity);
                const std::size_t second_len = run_length(second_parity);
                if (first_len + second_len != n) return false;

                // The common high/low zigzag is "ascending even run" followed by
                // the odd run read backwards.  Verify that ordered output while
                // moving it, so strings do not pay an additional full
                // std::is_sorted pass after millions of moves.  The rarer
                // backwards-first cases keep the old post-check because their
                // overwrite-avoiding copy order is not the final output order.
                bool verified_output_order = false;
                bool output_ordered = true;

                if constexpr (std::is_trivially_copyable<T>::value) {
                    if (first_forward) {
                        ScratchLease<T> tmp_lease(second_len);
                        if (!tmp_lease.valid() && second_len != 0) return false;
                        T* tmp = tmp_lease.get();
                        Cursor s = make_cursor(second_parity, second_forward);
                        for (std::size_t i = 0; i < second_len; ++i) {
                            tmp[i] = p[s.cur];
                            advance(s);
                        }
                        Cursor f = make_cursor(first_parity, first_forward);
                        for (std::size_t out = 0; out < first_len; ++out) {
                            const std::size_t src = f.cur;
                            if (out != 0 && before(p[src], p[out - 1])) output_ordered = false;
                            p[out] = p[src];
                            advance(f);
                        }
                        for (std::size_t i = 0; i < second_len; ++i) {
                            if (first_len + i != 0 && before(tmp[i], p[first_len + i - 1]))
                                output_ordered = false;
                            p[first_len + i] = tmp[i];
                        }
                        verified_output_order = true;
                    } else {
                        ScratchLease<T> tmp_lease(first_len);
                        if (!tmp_lease.valid() && first_len != 0) return false;
                        T* tmp = tmp_lease.get();
                        Cursor f = make_cursor(first_parity, first_forward);
                        for (std::size_t i = 0; i < first_len; ++i) {
                            tmp[i] = p[f.cur];
                            advance(f);
                        }
                        if (second_forward) {
                            Cursor s = make_cursor(second_parity, !second_forward);
                            for (std::size_t off = second_len; off-- > 0;) {
                                p[first_len + off] = p[s.cur];
                                advance(s);
                            }
                        } else {
                            Cursor s = make_cursor(second_parity, second_forward);
                            for (std::size_t out = 0; out < second_len; ++out) {
                                p[first_len + out] = p[s.cur];
                                advance(s);
                            }
                        }
                        if (first_len != 0)
                            std::memcpy(p, tmp, first_len * sizeof(T));
                    }
                } else {
                    if (first_forward) {
                        std::vector<T> tmp;
                        tmp.reserve(second_len);
                        Cursor s = make_cursor(second_parity, second_forward);
                        for (std::size_t i = 0; i < second_len; ++i) {
                            tmp.push_back(std::move(p[s.cur]));
                            advance(s);
                        }
                        Cursor f = make_cursor(first_parity, first_forward);
                        for (std::size_t out = 0; out < first_len; ++out) {
                            const std::size_t src = f.cur;
                            if (out != 0 && before(p[src], p[out - 1])) output_ordered = false;
                            if (out != src) p[out] = std::move(p[src]);
                            advance(f);
                        }
                        for (std::size_t i = 0; i < second_len; ++i) {
                            if (first_len + i != 0 && before(tmp[i], p[first_len + i - 1]))
                                output_ordered = false;
                            p[first_len + i] = std::move(tmp[i]);
                        }
                        verified_output_order = true;
                    } else {
                        std::vector<T> tmp;
                        tmp.reserve(first_len);
                        Cursor f = make_cursor(first_parity, first_forward);
                        for (std::size_t i = 0; i < first_len; ++i) {
                            tmp.push_back(std::move(p[f.cur]));
                            advance(f);
                        }
                        if (second_forward) {
                            Cursor s = make_cursor(second_parity, !second_forward);
                            for (std::size_t off = second_len; off-- > 0;) {
                                p[first_len + off] = std::move(p[s.cur]);
                                advance(s);
                            }
                        } else {
                            Cursor s = make_cursor(second_parity, second_forward);
                            for (std::size_t out = 0; out < second_len; ++out) {
                                p[first_len + out] = std::move(p[s.cur]);
                                advance(s);
                            }
                        }
                        for (std::size_t i = 0; i < first_len; ++i)
                            p[i] = std::move(tmp[i]);
                    }
                }
                if (verified_output_order) {
                    if (!output_ordered) pdqsort_for_profile_pattern(p, n, comp);
                } else if (!std::is_sorted(p, p + n, before)) {
                    pdqsort_for_profile_pattern(p, n, comp);
                }
                return true;
            };

            if (concat == 1 && try_concat_reorder(0, even_forward, 1, odd_forward)) return true;
            if (concat == 2 && try_concat_reorder(1, odd_forward, 0, even_forward)) return true;

            Cursor e = make_cursor(0, even_forward);
            Cursor o = make_cursor(1, odd_forward);
            bool sorted = true;

            if constexpr (std::is_trivially_copyable<T>::value) {
                ScratchLease<T> tmp_lease(n);
                if (!tmp_lease.valid()) return false;
                T* tmp = tmp_lease.get();
                std::size_t out = 0;
                T prev{};
                bool have_prev = false;
                auto emit = [&](std::size_t idx) {
                    const T v = p[idx];
                    if (have_prev && before(v, prev)) sorted = false;
                    tmp[out++] = v;
                    prev = v;
                    have_prev = true;
                };
                auto drain = [&](Cursor& c) { while (c.have) { emit(c.cur); advance(c); } };
                if (concat == 1) {
                    drain(e); drain(o);
                } else if (concat == 2) {
                    drain(o); drain(e);
                } else {
                    while (e.have && o.have) {
                        if (before(p[o.cur], p[e.cur])) { emit(o.cur); advance(o); }
                        else                            { emit(e.cur); advance(e); }
                    }
                    drain(e); drain(o);
                }
                if (out != n || !sorted) return false;
                std::memcpy(p, tmp, n * sizeof(T));
                return true;
            } else {
                std::vector<T> tmp;
                tmp.reserve(n);
                auto emit = [&](std::size_t idx) {
                    if (!tmp.empty() && before(p[idx], tmp.back())) sorted = false;
                    tmp.push_back(std::move(p[idx]));
                };
                auto drain = [&](Cursor& c) { while (c.have) { emit(c.cur); advance(c); } };
                if (concat == 1) {
                    drain(e); drain(o);
                } else if (concat == 2) {
                    drain(o); drain(e);
                } else {
                    while (e.have && o.have) {
                        if (before(p[o.cur], p[e.cur])) { emit(o.cur); advance(o); }
                        else                            { emit(e.cur); advance(e); }
                    }
                    drain(e); drain(o);
                }
                if (tmp.size() != n) return false;
                if (!sorted) {
                    pdqsort_for_profile_pattern(tmp.data(), n, comp);
                }
                for (std::size_t i = 0; i < n; ++i) p[i] = std::move(tmp[i]);
                return true;
            }
        };

        if (even_asc && odd_asc)   return run_one(true,  true);
        if (even_asc && odd_desc)  return run_one(true,  false);
        if (even_desc && odd_asc)  return run_one(false, true);
        if (even_desc && odd_desc) return run_one(false, false);
        return false;
    }
#endif
}


template <class T, class Comp>
inline bool try_numeric_half_organ_fill(T* p, std::size_t n, Comp comp) {
    (void)comp;
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n;
    return false;
#else
    if constexpr (!(is_ascending_v<Comp, T> && radix_supported_v<T> &&
                    !std::is_same<T, bool>::value && std::is_arithmetic<T>::value)) {
        (void)p; (void)n;
        return false;
    } else {
        if (n < std::size_t(1024)) return false;
        auto try_split = [&](std::size_t sp) -> bool {
            if (sp < n / 16u || n - sp < n / 16u || sp == 0 || sp >= n) return false;
            if constexpr (std::is_integral<T>::value) {
                using RT = RadixTraits<T>;
                using Key = typename RT::Key;
                auto key = [&](const T& v) -> Key { return RT::encode(v); };
                const Key k0 = key(p[0]);
                const Key k1 = key(p[1]);
                int dir = 0;
                if (static_cast<Key>(k1 - k0) == Key(2)) dir = 1;
                else if (static_cast<Key>(k0 - k1) == Key(2)) dir = -1;
                else return false;

                Key lo = k0;
                Key hi = k0;
                auto note = [&](Key k) {
                    if (k < lo) lo = k;
                    if (hi < k) hi = k;
                };
                note(key(p[sp - 1]));
                note(key(p[sp]));
                note(key(p[n - 1]));
                if (static_cast<unsigned long long>(hi - lo) + 1ull !=
                    static_cast<unsigned long long>(n)) return false;

                bool ok = true;
                for (std::size_t i = 1; i < sp && ok; ++i) {
                    const Key a = key(p[i - 1]);
                    const Key b = key(p[i]);
                    ok = dir > 0 ? (static_cast<Key>(b - a) == Key(2))
                                 : (static_cast<Key>(a - b) == Key(2));
                }
                for (std::size_t i = sp + 1; i < n && ok; ++i) {
                    const Key a = key(p[i - 1]);
                    const Key b = key(p[i]);
                    ok = dir > 0 ? (static_cast<Key>(a - b) == Key(2))
                                 : (static_cast<Key>(b - a) == Key(2));
                }
                if (!ok) return false;
                for (std::size_t out = 0; out < n; ++out)
                    p[out] = RT::decode(static_cast<Key>(lo + static_cast<Key>(out)));
                return true;
            } else if constexpr (std::is_floating_point<T>::value) {
                const double x0 = static_cast<double>(p[0]);
                const double x1 = static_cast<double>(p[1]);
                if (!std::isfinite(x0) || !std::isfinite(x1)) return false;
                int dir = 0;
                if (x1 - x0 == 2.0) dir = 1;
                else if (x0 - x1 == 2.0) dir = -1;
                else return false;

                auto finite_value = [&](std::size_t idx, double& out) -> bool {
                    out = static_cast<double>(p[idx]);
                    return std::isfinite(out);
                };
                double a = 0.0, b = 0.0, c = 0.0, d = 0.0;
                if (!finite_value(0, a) || !finite_value(sp - 1, b) ||
                    !finite_value(sp, c) || !finite_value(n - 1, d)) return false;
                double lo = std::min(std::min(a, b), std::min(c, d));
                double hi = std::max(std::max(a, b), std::max(c, d));
                if (hi - lo + 1.0 != static_cast<double>(n)) return false;

                bool ok = true;
                double prev = x0;
                for (std::size_t i = 1; i < sp && ok; ++i) {
                    const double cur = static_cast<double>(p[i]);
                    ok = dir > 0 ? (cur - prev == 2.0) : (prev - cur == 2.0);
                    prev = cur;
                }
                prev = static_cast<double>(p[sp]);
                for (std::size_t i = sp + 1; i < n && ok; ++i) {
                    const double cur = static_cast<double>(p[i]);
                    ok = dir > 0 ? (prev - cur == 2.0) : (cur - prev == 2.0);
                    prev = cur;
                }
                if (!ok) return false;
                for (std::size_t out = 0; out < n; ++out)
                    p[out] = static_cast<T>(lo + static_cast<double>(out));
                return true;
            } else {
                return false;
            }
        };
        const std::size_t mid = n / 2u;
        return try_split(mid) || ((n & 1u) && try_split(mid + 1u));
    }
#endif
}

template <class T>
inline bool try_integer_permutation_range_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!(std::is_integral<T>::value && !std::is_same<T, bool>::value && radix_supported_v<T>)) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        using RT = RadixTraits<T>;
        using Key = typename RT::Key;
        {
            const std::size_t sample_n = std::min<std::size_t>(n, std::size_t(257));
            Key smn = RT::encode(p[0]);
            Key smx = smn;
            for (std::size_t j = 1; j < sample_n; ++j) {
                const std::size_t idx = (j * (n - 1)) / (sample_n - 1);
                const Key k = RT::encode(p[idx]);
                if (k < smn) smn = k;
                if (smx < k) smx = k;
            }
            const Key sample_span = static_cast<Key>(smx - smn);
            const unsigned long long limit = static_cast<unsigned long long>(
                std::min<std::size_t>(n, std::numeric_limits<std::size_t>::max() / 4u)) * 4ull;
            if (static_cast<unsigned long long>(sample_span) + 1ull > limit)
                return false;
        }
        T mn = p[0], mx = p[0];
        for (std::size_t i = 1; i < n; ++i) {
            if (p[i] < mn) mn = p[i];
            if (mx < p[i]) mx = p[i];
        }
        const Key lo = RT::encode(mn);
        const Key hi = RT::encode(mx);
        const Key span = static_cast<Key>(hi - lo);
        if (span == std::numeric_limits<Key>::max()) return false;
        if (static_cast<unsigned long long>(span) + 1ull != static_cast<unsigned long long>(n))
            return false;

        const std::size_t words = (n + 63u) / 64u;
        ScratchLease<std::uint64_t> seen_lease(words);
        if (!seen_lease.valid()) return false;
        std::uint64_t* seen = seen_lease.get();
        std::memset(seen, 0, words * sizeof(std::uint64_t));
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t idx = static_cast<std::size_t>(RT::encode(p[i]) - lo);
            const std::uint64_t bit = std::uint64_t(1) << (idx & 63u);
            std::uint64_t& w = seen[idx >> 6];
            if (w & bit) return false;
            w |= bit;
        }
        if (!descending) {
            for (std::size_t i = 0; i < n; ++i)
                p[i] = RT::decode(static_cast<Key>(lo + static_cast<Key>(i)));
        } else {
            for (std::size_t i = 0; i < n; ++i)
                p[i] = RT::decode(static_cast<Key>(hi - static_cast<Key>(i)));
        }
        return true;
    }
}


template <class T>
inline bool try_floating_integer_permutation_range_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!(std::is_floating_point<T>::value && radix_supported_v<T>)) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        {
            const std::size_t sample_n = std::min<std::size_t>(n, std::size_t(257));
            double smn = static_cast<double>(p[0]);
            double smx = smn;
            if (!std::isfinite(smn) || std::floor(smn) != smn) return false;
            for (std::size_t j = 1; j < sample_n; ++j) {
                const std::size_t idx = (j * (n - 1)) / (sample_n - 1);
                const double x = static_cast<double>(p[idx]);
                if (!std::isfinite(x) || std::floor(x) != x) return false;
                if (x < smn) smn = x;
                if (smx < x) smx = x;
            }
            if (smx - smn + 1.0 > static_cast<double>(n) * 4.0) return false;
        }
        double mn = static_cast<double>(p[0]);
        double mx = mn;
        if (!std::isfinite(mn)) return false;
        for (std::size_t i = 1; i < n; ++i) {
            const double x = static_cast<double>(p[i]);
            if (!std::isfinite(x)) return false;
            if (x < mn) mn = x;
            if (mx < x) mx = x;
        }
        if (std::floor(mn) != mn || std::floor(mx) != mx) return false;
        const double span_d = mx - mn;
        if (!(span_d >= 0.0) || span_d > static_cast<double>(std::numeric_limits<std::size_t>::max()))
            return false;
        const std::size_t span = static_cast<std::size_t>(span_d);
        if (static_cast<double>(span) != span_d || span != n - 1u) return false;

        const std::size_t words = (n + 63u) / 64u;
        ScratchLease<std::uint64_t> seen_lease(words);
        if (!seen_lease.valid()) return false;
        std::uint64_t* seen = seen_lease.get();
        std::memset(seen, 0, words * sizeof(std::uint64_t));
        for (std::size_t i = 0; i < n; ++i) {
            const double off_d = static_cast<double>(p[i]) - mn;
            if (!(off_d >= 0.0) || off_d > span_d) return false;
            const std::size_t idx = static_cast<std::size_t>(off_d);
            if (static_cast<double>(idx) != off_d) return false;
            const std::uint64_t bit = std::uint64_t(1) << (idx & 63u);
            std::uint64_t& w = seen[idx >> 6];
            if (w & bit) return false;
            w |= bit;
        }
        if (!descending) {
            for (std::size_t i = 0; i < n; ++i)
                p[i] = static_cast<T>(mn + static_cast<double>(i));
        } else {
            for (std::size_t i = 0; i < n; ++i)
                p[i] = static_cast<T>(mx - static_cast<double>(i));
        }
        return true;
    }
}


template <class T>
inline bool try_radix_permutation_range_sort(T* p, std::size_t n, bool descending) {
    return try_integer_permutation_range_sort(p, n, descending) ||
           try_floating_integer_permutation_range_sort(p, n, descending);
}



template <class T, class Comp>
inline bool try_string_half_organ_reorder(T* p, std::size_t n, Comp comp) {
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n; (void)comp;
    return false;
#else
    if constexpr (!std::is_same<T, std::string>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        if (n < std::size_t(1024) || (n & 1u)) return false;
        const std::size_t half = n / 2u;
        if (!comp(p[0], p[1]) || !comp(p[half + 1], p[half]) ||
            !comp(p[n - 1], p[n - 2])) return false;

        const std::size_t probe = std::min<std::size_t>(half, 64u);
        for (std::size_t i = 1; i < probe; ++i) {
            if (comp(p[i], p[i - 1])) return false;
            if (comp(p[half + i - 1], p[half + i])) return false;
        }
        for (std::size_t i = 0; i + 1 < probe; ++i) {
            const std::string& a = p[i];
            const std::string& b = p[n - 1u - i];
            if (comp(b, a) || comp(p[i + 1], b)) return false;
        }

        std::vector<std::string> tmp;
        tmp.reserve(half);
        for (std::size_t i = 0; i < half; ++i)
            tmp.push_back(std::move(p[n - 1u - i]));
        for (std::size_t i = half; i-- > 0;) {
            const std::size_t dst = i * 2u;
            if (dst != i) p[dst] = std::move(p[i]);
        }
        for (std::size_t i = 0; i < half; ++i)
            p[i * 2u + 1u] = std::move(tmp[i]);
        if (!std::is_sorted(p, p + n, comp))
            pdqsort_for_profile_pattern(p, n, comp);
        return true;
    }
#endif
}

template <class T, class Comp>
inline bool likely_mid_bitonic_runs(T* p, std::size_t n, Comp comp) {
    if (n < std::size_t(1024)) return false;
    const std::size_t mid = n / 2u;
    if (mid + 1 >= n) return false;
    const bool head_up = comp(p[0], p[1]);
    const bool head_down = comp(p[1], p[0]);
    if (!head_up && !head_down) return false;
    const bool mid_up = comp(p[mid], p[mid + 1]);
    const bool mid_down = comp(p[mid + 1], p[mid]);
    const bool tail_up = comp(p[n - 2], p[n - 1]);
    const bool tail_down = comp(p[n - 1], p[n - 2]);
    return (head_up && mid_down && tail_down) ||
           (head_down && mid_up && tail_up);
}

// try_bitonic_runs_sort: above this footprint merge in place from a buffer
// holding only the descending run.
inline constexpr std::size_t kBitonicHalfBufferBytes = std::size_t(4) << 20;

template <class T, class Comp>
inline bool try_bitonic_runs_sort(T* p, std::size_t n, Comp comp) {
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n; (void)comp;
    return false;
#else
    if (n < std::size_t(1024)) return false;
    constexpr bool radix_order = radix_supported_v<T> && std::is_floating_point<T>::value &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    auto before = [&](const T& a, const T& b) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            using Key = typename RT::Key;
            const Key ka = RT::encode(a);
            const Key kb = RT::encode(b);
            if constexpr (is_descending_v<Comp, T>) return kb < ka;
            else return ka < kb;
        } else {
            return comp(a, b);
        }
    };
    if constexpr (!std::is_move_constructible<T>::value || !std::is_move_assignable<T>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        // Organ-pipe / bitonic benchmark inputs are two consecutive monotone
        // runs, e.g. 0,2,4,...,999999,999997,...,1.  The generic numeric path
        // used to treat them as high-entropy and pay radix passes; strings paid
        // a sample-sort path.  Prove that there is exactly one direction change
        // and then merge the two runs linearly.  Random and adjacent-swap inputs
        // see a second direction change almost immediately and decline.
        std::size_t i = 1;
        while (i < n && !before(p[i], p[i - 1]) && !before(p[i - 1], p[i])) ++i;
        if (i == n) return false;
        const bool first_asc = before(p[i - 1], p[i]);
        ++i;

        // Block OR-reductions without early exits (they vectorize for plain
        // key comparators); a hit is located inside its block afterwards.
        constexpr std::size_t kScanBlock = 512;
        auto any_break = [&](std::size_t lo, std::size_t hi, bool asc) -> bool {
            return any_adjacent_pair(p, lo, hi, before, asc);
        };
        std::size_t split = n;
        for (std::size_t b = i; b < n; b += kScanBlock) {
            const std::size_t e = std::min(n, b + kScanBlock);
            if (!any_break(b, e, first_asc)) continue;
            for (std::size_t j = b; j < e; ++j)
                if (first_asc ? before(p[j], p[j - 1]) : before(p[j - 1], p[j])) { split = j; break; }
            break;
        }
        if (split == n) return false;
        if (split < n / 16u || n - split < n / 16u) return false;
        for (std::size_t b = split + 1; b < n; b += kScanBlock)
            if (any_break(b, std::min(n, b + kScanBlock), !first_asc)) return false;

        auto try_arithmetic_organ_fill = [&]() -> bool {
            if constexpr (!(is_ascending_v<Comp, T> && radix_supported_v<T> &&
                            !std::is_same<T, bool>::value && std::is_arithmetic<T>::value)) {
                return false;
            } else if constexpr (std::is_integral<T>::value) {
                using RT = RadixTraits<T>;
                using Key = typename RT::Key;
                auto key = [&](const T& v) -> Key { return RT::encode(v); };
                auto check_and_fill = [&](std::size_t sp) -> bool {
                    if (sp == 0 || sp >= n) return false;
                    if (sp < n / 16u || n - sp < n / 16u) return false;
                    Key lo = key(p[0]);
                    Key hi = lo;
                    auto note = [&](Key k) {
                        if (k < lo) lo = k;
                        if (hi < k) hi = k;
                    };
                    note(key(p[sp - 1]));
                    note(key(p[sp]));
                    note(key(p[n - 1]));
                    if (static_cast<unsigned long long>(hi - lo) + 1ull !=
                        static_cast<unsigned long long>(n)) return false;

                    bool ok = true;
                    for (std::size_t j = 1; j < sp && ok; ++j) {
                        const Key a = key(p[j - 1]);
                        const Key b = key(p[j]);
                        ok = first_asc ? (static_cast<Key>(b - a) == Key(2))
                                       : (static_cast<Key>(a - b) == Key(2));
                    }
                    for (std::size_t j = sp + 1; j < n && ok; ++j) {
                        const Key a = key(p[j - 1]);
                        const Key b = key(p[j]);
                        ok = first_asc ? (static_cast<Key>(a - b) == Key(2))
                                       : (static_cast<Key>(b - a) == Key(2));
                    }
                    if (!ok) return false;
                    for (std::size_t out = 0; out < n; ++out)
                        p[out] = RT::decode(static_cast<Key>(lo + static_cast<Key>(out)));
                    return true;
                };
                return check_and_fill(split) || (split > 0 && check_and_fill(split - 1));
            } else if constexpr (std::is_floating_point<T>::value) {
                const double exact_limit = std::is_same<T, float>::value ? 16777216.0 : 9007199254740992.0;
                auto val = [&](const T& v, double& out) -> bool {
                    out = static_cast<double>(v);
                    return std::isfinite(out) && std::floor(out) == out && std::fabs(out) <= exact_limit;
                };
                auto check_and_fill = [&](std::size_t sp) -> bool {
                    if (sp == 0 || sp >= n) return false;
                    if (sp < n / 16u || n - sp < n / 16u) return false;
                    double x0 = 0.0, x1 = 0.0, x2 = 0.0, x3 = 0.0;
                    if (!val(p[0], x0) || !val(p[sp - 1], x1) ||
                        !val(p[sp], x2) || !val(p[n - 1], x3)) return false;
                    double lo = std::min(std::min(x0, x1), std::min(x2, x3));
                    double hi = std::max(std::max(x0, x1), std::max(x2, x3));
                    if (hi - lo + 1.0 != static_cast<double>(n)) return false;
                    if (std::fabs(lo) + static_cast<double>(n) > exact_limit) return false;

                    bool ok = true;
                    double prev = x0;
                    for (std::size_t j = 1; j < sp && ok; ++j) {
                        double cur = 0.0;
                        ok = val(p[j], cur) && (first_asc ? (cur - prev == 2.0)
                                                           : (prev - cur == 2.0));
                        prev = cur;
                    }
                    if (!ok) return false;
                    if (!val(p[sp], prev)) return false;
                    for (std::size_t j = sp + 1; j < n && ok; ++j) {
                        double cur = 0.0;
                        ok = val(p[j], cur) && (first_asc ? (prev - cur == 2.0)
                                                           : (cur - prev == 2.0));
                        prev = cur;
                    }
                    if (!ok) return false;
                    for (std::size_t out = 0; out < n; ++out)
                        p[out] = static_cast<T>(lo + static_cast<double>(out));
                    return true;
                };
                return check_and_fill(split) || (split > 0 && check_and_fill(split - 1));
            } else {
                return false;
            }
        };
        // Value-disjoint runs (organ pipe built as sort-then-reverse-half,
        // and its mirror): after the descending run is reversed in place the
        // range is sorted, or one rotation from sorted -- no merge pass.
        // X = ascending run, Y = descending run; the test reads their ends.
        {
            const std::size_t xl = first_asc ? 0 : split, xh = first_asc ? split - 1 : n - 1;
            const std::size_t yl = first_asc ? split : 0, yh = first_asc ? n - 1 : split - 1;
            // Y's minimum is p[yh], its maximum p[yl].
            const bool x_then_y = !before(p[yh], p[xh]);   // max X <= min Y
            const bool y_then_x = !before(p[xl], p[yl]);   // max Y <= min X
            if (x_then_y || y_then_x) {
                std::reverse(p + yl, p + yh + 1);
                if (first_asc ? y_then_x && !x_then_y : x_then_y && !y_then_x)
                    std::rotate(p, p + split, p + n);
                return true;
            }
        }

        if (try_arithmetic_organ_fill()) return true;

        if constexpr (std::is_trivially_copyable<T>::value) {
          if (n * sizeof(T) > kBitonicHalfBufferBytes) {
            // Memory-bound size: save only the descending run (reversed, so
            // ascending) and merge in place toward it -- about 40% less
            // traffic than merge-to-buffer plus copy back.  first_asc: the
            // back chain fills p from n-1 down (write index = X tail + buffer
            // left, never past an unread X slot).  Otherwise the front chain
            // fills from 0 up (write index <= next unread Y slot).
            const std::size_t m = first_asc ? n - split : split;
            ScratchLease<T> tmp_lease(m);
            if (!tmp_lease.valid()) return false;
            T* tmp = tmp_lease.get();
            const T* src = first_asc ? p + split : p;
            for (std::size_t k = 0; k < m; ++k) tmp[k] = src[m - 1 - k];
            if (first_asc) {
                std::size_t xr = split, tr = m, o = n;   // counts left
                while (xr != 0 && tr != 0) {
                    const bool tx = before(tmp[tr - 1], p[xr - 1]);
                    p[--o] = *(tx ? p + (xr - 1) : tmp + (tr - 1));
                    xr -= tx ? 1u : 0u;
                    tr -= tx ? 0u : 1u;
                }
                if (tr != 0) std::memcpy(p, tmp, tr * sizeof(T));
            } else {
                std::size_t t = 0, y = split, o = 0;
                while (t != m && y != n) {
                    const bool ty = before(p[y], tmp[t]);
                    p[o++] = *(ty ? p + y : tmp + t);
                    y += ty ? 1u : 0u;
                    t += ty ? 0u : 1u;
                }
                if (t != m) std::memcpy(p + o, tmp + t, (m - t) * sizeof(T));
            }
            return true;
          }
            ScratchLease<T> tmp_lease(n);
            if (!tmp_lease.valid()) return false;
            T* tmp = tmp_lease.get();
            // X ascends over [x, xe], Y descends over [ye, y] (so p[y] is its
            // smallest).  Two branchless chains: the front emits the smallest
            // (ties -> X), the back the largest (ties -> Y), which is one
            // consistent total order.  Each round runs h steps per chain with
            // h = min(remaining X, remaining Y) / 2, so neither chain can
            // drain a run inside the round: no bounds checks in the loop.
            std::size_t x  = first_asc ? 0 : split;
            std::size_t xe = first_asc ? split - 1 : n - 1;
            std::size_t ye = first_asc ? split : 0;
            std::size_t y  = first_asc ? n - 1 : split - 1;
            std::size_t of = 0, ob = n - 1;
            while (true) {
                const std::size_t xr = xe - x + 1, yr = y - ye + 1;
                const std::size_t h = std::min(xr, yr) / 2;
                if (h < 4) break;
                for (std::size_t k = 0; k < h; ++k) {
                    const bool ty = before(p[y], p[x]);
                    tmp[of + k] = *(ty ? p + y : p + x);
                    x += ty ? 0u : 1u;
                    y -= ty ? 1u : 0u;
                    const bool tx = before(p[ye], p[xe]);
                    tmp[ob - k] = *(tx ? p + xe : p + ye);
                    xe -= tx ? 1u : 0u;
                    ye += tx ? 0u : 1u;
                }
                of += h;
                ob -= h;
            }
            // Tail, front only.  Runs stay non-empty here (each round left
            // at least half of the smaller one); x > xe marks X drained.
            while (x <= xe) {
                const bool ty = before(p[y], p[x]);
                tmp[of++] = *(ty ? p + y : p + x);
                if (!ty) { ++x; continue; }
                if (y == ye) { ye = y + 1; break; }
                --y;
            }
            while (x <= xe) tmp[of++] = p[x++];
            if (ye <= y) for (std::size_t k = y + 1; k-- > ye;) tmp[of++] = p[k];
            if (of != ob + 1) return false;
            std::memcpy(p, tmp, n * sizeof(T));
        } else {
            if (first_asc) std::reverse(p + split, p + n);
            else           std::reverse(p, p + split);
            std::inplace_merge(p, p + split, p + n, before);
        }
        return true;
    }
#endif
}

// The profiling sample's value window, carried so the counting kernels do not
// have to re-read their own strided samples.  `lo`/`hi` are *encoded* radix
// keys observed in the profile's evenly spaced sample -- a lower bound of the
// true range (the sample can miss extremes, never invent them) -- and
// `distinct` is the exact sample distinct count up to kCountingClassLimit
// (kCountingClassLimit + 1 once it overflows, 0 when unknown).  Every consumer
// still validates before committing: a window miss escapes to the exact pass.
template <class T>
struct SampleWindow {
    bool valid = false;
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;
    std::size_t distinct = 0;
};

template <class T>
inline bool try_radix_key_sparse_count_sort(T* p, std::size_t n, bool descending,
                                            const SampleWindow<T>* win = nullptr);
template <class Key>
FYX_FORCE_INLINE std::size_t low_card_hash_key(Key k) noexcept;

/// Distinct radix keys in an evenly spaced sample, counted exactly up to
/// `cap` (the return is `cap + 1` when there are more).  Used to tell "a
/// handful of values spread over a wide range" from "a value at every point of
/// a narrow range": the two want different counting kernels, and the range
/// alone cannot distinguish them.
template <class T>
inline std::size_t sample_distinct_keys(const T* p, std::size_t n, std::size_t cap) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr std::size_t kSlots = 1024;              // >= 2x the 256+1 cap
    constexpr std::size_t kMask  = kSlots - 1;
    Key          slot[kSlots];
    std::uint8_t used[kSlots];
    for (std::size_t i = 0; i < kSlots; ++i) used[i] = 0;

    const std::size_t sample_n = std::min<std::size_t>(n, kProfileSampleLimit);
    std::size_t distinct = 0;
    for (std::size_t j = 0; j < sample_n; ++j) {
        const std::size_t idx = (j * n) / sample_n;
        const Key k = RT::encode(p[idx]);
        std::size_t h = low_card_hash_key(k) & kMask;
        while (used[h] && slot[h] != k) h = (h + 1) & kMask;
        if (!used[h]) {
            used[h] = 1;
            slot[h] = k;
            if (++distinct > cap) return cap + 1;
        }
    }
    return distinct;
}

// The exact fallback: one full min/max sweep, then count and fill.  Only the
// window-fused path above calls this -- the sample window missed an extreme
// and the whole range has to be measured before counting.
template <class T>
inline bool try_integer_range_count_exact(T* p, std::size_t n, bool descending) {
    if constexpr (!(radix_supported_v<T> && !std::is_same<T, bool>::value)) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;

        // Same crossover as the parallel kernel: a handful of values spread
        // over a wide range belongs to the sparse counter, which finds them by
        // value instead of paying for every slot between them.  Checked on the
        // sample, before the min/max scan, so declining costs one pass less.
        {
            const std::size_t dhat = sample_distinct_keys(p, n, kCountingClassLimit);
            if (dhat <= kCountingClassLimit) {
                Key smn = RT::encode(p[0]), smx = smn;
                const std::size_t sample_n = std::min<std::size_t>(n, kProfileSampleLimit);
                for (std::size_t j = 1; j < sample_n; ++j) {
                    const Key k = RT::encode(p[(j * n) / sample_n]);
                    if (k < smn) smn = k;
                    if (smx < k) smx = k;
                }
                const unsigned long long srange =
                    static_cast<unsigned long long>(static_cast<Key>(smx - smn)) + 1ull;
                const unsigned long long spread = sizeof(T) <= 4 ? 64ull : 256ull;
                if (srange >= 32768ull &&
                    srange > spread * static_cast<unsigned long long>(dhat)) {
                    if (try_radix_key_sparse_count_sort(p, n, descending)) return true;
                }
            }
        }

        T mn = p[0], mx = p[0];
        for (std::size_t i = 1; i < n; ++i) {
            if (p[i] < mn) mn = p[i];
            if (mx < p[i]) mx = p[i];
        }
        const Key lo   = RT::encode(mn);
        const Key hi   = RT::encode(mx);
        const Key span = static_cast<Key>(hi - lo);
        if (span == std::numeric_limits<Key>::max()) return false;

        // Dense counting is O(n + range): it wins while the range is small
        // beside n and loses badly once they are comparable.

        const std::size_t adaptive =
            (vqsort_preferred<T>() && vqsort_usable<T>(n))
                ? std::max<std::size_t>(std::size_t(4096), n / 8)
                : std::max<std::size_t>(std::size_t(4096), n);
        const std::size_t limit    = std::min<std::size_t>(kCountingRangeLimit, adaptive);
        const unsigned long long range64 = static_cast<unsigned long long>(span) + 1ull;
        if (range64 > static_cast<unsigned long long>(limit)) return false;
        const std::size_t range = static_cast<std::size_t>(range64);

        ScratchLease<std::size_t> counts_lease(range);
        if (!counts_lease.valid()) return false;
        std::size_t* counts = counts_lease.get();
        for (std::size_t i = 0; i < range; ++i) counts[i] = 0;
        for (std::size_t i = 0; i < n; ++i)
            ++counts[static_cast<std::size_t>(RT::encode(p[i]) - lo)];

        std::size_t out = 0;
        if (!descending) {
            for (std::size_t r = 0; r < range; ++r) {
                const T v = RT::decode(static_cast<Key>(lo + static_cast<Key>(r)));
                for (std::size_t c = counts[r]; c != 0; --c) p[out++] = v;
            }
        } else {
            for (std::size_t rr = range; rr-- > 0;) {
                const T v = RT::decode(static_cast<Key>(lo + static_cast<Key>(rr)));
                for (std::size_t c = counts[rr]; c != 0; --c) p[out++] = v;
            }
        }
        return true;
    }
}

// Despite the historical name this counts *encoded radix keys*, so it serves
// every radix type: floating-point keys are order-preserving integers, and a
// float column whose encoded values land in a narrow window (ratings, scores,
// money in fixed-point) gets the same dense O(n + range) counter as ints
// instead of the hash-probed sparse one.

// The vector quicksort's NaN / -0 screen is a property of the multiset, so it
// survives every permutation the declining probes may leave behind.  One
// top-level sort call screens a given (pointer, length) at most once: the
// memo is cleared on entry to and exit from sort_pointer_core_impl (a full
// extra read of a 200K double range is ~55us, and up to three gates asked).
struct VqCleanMemo {
    const void* p = nullptr;
    std::size_t n = 0;
    bool clean = false;
};
inline VqCleanMemo& vq_clean_memo() noexcept {
    static thread_local VqCleanMemo m;
    return m;
}
struct VqCleanMemoScope {
    VqCleanMemoScope() noexcept { vq_clean_memo().p = nullptr; order_break_hint().p = nullptr; }
    ~VqCleanMemoScope() { vq_clean_memo().p = nullptr; order_break_hint().p = nullptr; }
    VqCleanMemoScope(const VqCleanMemoScope&) = delete;
    VqCleanMemoScope& operator=(const VqCleanMemoScope&) = delete;
};
template <class T>
inline bool vq_range_clean_memo(const T* p, std::size_t n) {
    VqCleanMemo& m = vq_clean_memo();
    if (m.p == static_cast<const void*>(p) && m.n == n) return m.clean;
    const bool c = vqsort_range_clean(p, n);
    m.p = p;
    m.n = n;
    m.clean = c;
    return c;
}

// True when the AVX-512 vector quicksort will accept this range.  Its
// equal-key handling beats the hash-probed sparse counters on every
// few-distinct shape measured (Ice Lake-SP, 200K serial: int32 16 values
// 133us vs 523us, 256 values 247us vs 517us; int64 16 values 298us vs
// 363us), so sparse counting is skipped where the quicksort is available.
// Dense O(n + range) counting is unaffected.
template <class T>
inline bool vq_beats_sparse_count(const T* p, std::size_t n) {
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)p; (void)n; return false; }
    else {
        return vqsort_preferred<T>() && vqsort_usable<T>(n) && n >= kVqsortMinN &&
               vq_range_clean_memo(p, n);
    }
}

template <class T>
inline bool try_integer_range_count_sort(T* p, std::size_t n, bool descending,
                                         const SampleWindow<T>* win = nullptr) {
    if constexpr (!(radix_supported_v<T> && !std::is_same<T, bool>::value)) {
        (void)p; (void)n; (void)descending; (void)win;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;

        // Same crossover as the parallel kernel: a handful of values spread
        // over a wide range belongs to the sparse counter, which finds them by
        // value instead of paying for every slot between them.  Checked on the
        // profile's carried sample (a fresh strided sample when absent),
        // before any window work, so declining costs one pass less.
        const std::size_t dhat = (win && win->distinct != 0)
            ? win->distinct
            : sample_distinct_keys(p, n, kCountingClassLimit);
        if (dhat <= kCountingClassLimit) {
            Key smn, smx;
            const bool have_window = win && win->valid;
            if (have_window) {
                smn = static_cast<Key>(win->lo);
                smx = static_cast<Key>(win->hi);
            } else {
                smn = RT::encode(p[0]);
                smx = smn;
                const std::size_t sample_n = std::min<std::size_t>(n, kProfileSampleLimit);
                for (std::size_t j = 1; j < sample_n; ++j) {
                    const Key k = RT::encode(p[(j * n) / sample_n]);
                    if (k < smn) smn = k;
                    if (smx < k) smx = k;
                }
            }
            const unsigned long long srange =
                static_cast<unsigned long long>(static_cast<Key>(smx - smn)) + 1ull;
            const unsigned long long spread = sizeof(T) <= 4 ? 64ull : 256ull;
            if (srange >= 32768ull &&
                srange > spread * static_cast<unsigned long long>(dhat)) {
                if (vq_beats_sparse_count(p, n)) return false;
                if (try_radix_key_sparse_count_sort(p, n, descending, win)) return true;
            }
        }

        // The sample window is a guess, not a measurement (it lower-bounds the
        // true range).  Spending a whole sweep on the exact minimum and
        // maximum costs one more read of the array than the sort itself
        // needs, so count straight into the guessed window and abandon the
        // pass the moment a key lands outside it -- the exact pass below then
        // takes over.  On in-window data -- every genuinely narrow-domain
        // input the profiler lets through -- this saves the full min/max
        // sweep entirely.
        Key glo, ghi;
        if (win && win->valid) {
            glo = static_cast<Key>(win->lo);
            ghi = static_cast<Key>(win->hi);
        } else {
            const std::size_t sample_n = std::min<std::size_t>(n, std::size_t(257));
            glo = RT::encode(p[0]);
            ghi = glo;
            for (std::size_t j = 1; j < sample_n; ++j) {
                const Key k = RT::encode(p[(j * n) / sample_n]);
                if (k < glo) glo = k;
                if (ghi < k) ghi = k;
            }
        }
        const Key span = static_cast<Key>(ghi - glo);
        if (span == std::numeric_limits<Key>::max())
            return try_integer_range_count_exact(p, n, descending);

        // Dense counting is O(n + range): it wins while the range is small
        // beside n and loses badly once they are comparable (4M int32 over a
        // 2^22 range: 0.028 s counting, 0.016 s vectorised quicksort; over a
        // 2^18 range: 0.013 s counting, 0.019 s quicksort).  Where a quicksort
        // kernel exists for the type, hand the wide half of that trade to it;
        // where it does not, counting is still better than the alternatives.
        const std::size_t adaptive =
            (vqsort_preferred<T>() && vqsort_usable<T>(n))
                ? std::max<std::size_t>(std::size_t(4096), n / 8)
                : std::max<std::size_t>(std::size_t(4096), n);
        const std::size_t limit    = std::min<std::size_t>(kCountingRangeLimit, adaptive);
        const unsigned long long range64 = static_cast<unsigned long long>(span) + 1ull;
        if (range64 > static_cast<unsigned long long>(limit)) return false;
        if (n > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
            return try_integer_range_count_exact(p, n, descending);
        const std::size_t range = static_cast<std::size_t>(range64);

        ScratchLease<std::uint32_t> counts_lease(range);
        if (!counts_lease.valid())
            return try_integer_range_count_exact(p, n, descending);
        std::uint32_t* counts = counts_lease.get();
        for (std::size_t i = 0; i < range; ++i) counts[i] = 0;
        bool bail = false;
        for (std::size_t i = 0; i < n; ++i) {
            const Key d = static_cast<Key>(RT::encode(p[i]) - glo);
            if (d <= span) ++counts[static_cast<std::size_t>(d)];
            else { bail = true; break; }
        }
        if (bail) return try_integer_range_count_exact(p, n, descending);

        std::size_t out = 0;
        if (!descending) {
            for (std::size_t r = 0; r < range; ++r) {
                const T v = RT::decode(static_cast<Key>(glo + static_cast<Key>(r)));
                for (std::uint32_t c = counts[r]; c != 0; --c) p[out++] = v;
            }
        } else {
            for (std::size_t rr = range; rr-- > 0;) {
                const T v = RT::decode(static_cast<Key>(glo + static_cast<Key>(rr)));
                for (std::uint32_t c = counts[rr]; c != 0; --c) p[out++] = v;
            }
        }
        return true;
    }
}


template <class Key>
FYX_FORCE_INLINE std::size_t low_card_hash_key(Key k) noexcept {
    // Low-cardinality tables never hold more than 256 keys in 512/1024 slots;
    // a full Murmur finalizer was measurable overhead on float/double lowcard
    // inputs.  Fibonacci-style mixing is enough for these tiny open-addressed
    // tables and costs one multiply instead of two expensive 64-bit finalizer
    // rounds.
    if constexpr (sizeof(Key) <= 4) {
        const std::uint32_t x = static_cast<std::uint32_t>(k);
        return static_cast<std::size_t>(x * 2654435761u);
    } else {
        std::uint64_t x = static_cast<std::uint64_t>(k);
        x ^= x >> 32;
        x *= 0x9E3779B97F4A7C15ULL;
        x ^= x >> 32;
        return static_cast<std::size_t>(x);
    }
}

// ---------------------------------------------------------------------------
// Unified top-level input profiling (武器七).
//
// A small evenly-spaced sample filters out obvious high-entropy inputs without
// paying an O(n) detector pass.  When the sample suggests a proven shortcut
// (sorted/reverse/all-equal/near-sorted), one full linear scan validates the
// order profile at once: monotonicity, all-equal, capped distinct count and
// adjacent inversion count.  Low-cardinality samples are treated as candidates;
// the existing counting paths still validate the full range before committing,
// avoiding an extra profile-only O(n) pass on the low-cardinality hot path.
// For default-order radix types the monotone/equality tests use RadixTraits keys
// instead of raw comparator calls, preserving the documented float NaN and
// -0/+0 ordering.
// ---------------------------------------------------------------------------

inline constexpr std::size_t kProfileMinN            = 1024;
// try_kv16_early_sort: below kKv16EarlyMinN the probe stack's small-n paths
// win; from kKv16EarlyParallelMinN up parallel-eligible calls fork the
// quicksort's top levels.
inline constexpr std::size_t kKv16EarlyMinN          = 2048;
inline constexpr std::size_t kKv16EarlyParallelMinN  = std::size_t(1) << 17;
inline constexpr std::size_t kProfilePartialDivisor  = 64;
inline constexpr std::size_t kProfilePartialPdqMax   = 64u << 20;
// Below this many elements the thread-pool wake-up costs more than the extra
// scan it saves, so the orderedness proof stays on one thread.  Measured on
// 8M: parallel proof 0.0016-0.0025 against serial 0.0032-0.0042; at 1M the
// wake-up (~30 us on a 300 us sort) eats the gain.
inline constexpr std::size_t kParallelOrderMinN      = 2u << 20;
// Proof-only pool assist below the full-parallel floor.  The orderedness
// proof is a read-only scan -- no allocation, no writes unless it succeeds --
// so it can borrow the pool at sizes where handing the whole sort to the
// pool is still a net loss.  The scan *is* the whole runtime on trivial
// input, and a single thread leaves the second core idle exactly there;
// that is the shape IPS4o's striped is_sorted exploits on sub-2M sorted
// ranges.  Assist only when the range is at least a few megabytes: below
// that the whole scan is comparable to the pool wake-up itself.

enum class DispatchDecision : unsigned char {
    None = 0,
    Network,
    ProfileAllEqual,
    ProfileSorted,
    ProfileReverse,
    LowCardinality,
    PartialPdq,
    Radix,
    VectorQuick,
    Sample,
    ParallelSample,
    Pdq
};

#if FYX_ENABLE_TEST_HOOKS
inline DispatchDecision& test_dispatch_slot() noexcept {
    static thread_local DispatchDecision d = DispatchDecision::None;
    return d;
}
inline void test_reset_dispatch() noexcept { test_dispatch_slot() = DispatchDecision::None; }
inline DispatchDecision test_last_dispatch() noexcept { return test_dispatch_slot(); }
struct DispatchTraceEntry {
    DispatchDecision d;
    double t;         // seconds since first record on this thread
    std::size_t n;
};
inline std::vector<DispatchTraceEntry>& test_dispatch_trace() noexcept {
    static thread_local std::vector<DispatchTraceEntry> v;
    return v;
}
inline void test_reset_dispatch_trace() noexcept { test_dispatch_trace().clear(); }
inline void record_dispatch(DispatchDecision d) noexcept {
    test_dispatch_slot() = d;
    using C = std::chrono::steady_clock;
    const double now = std::chrono::duration<double>(C::now().time_since_epoch()).count();
    test_dispatch_trace().push_back({d, now, 0});
}
#else
inline void record_dispatch(DispatchDecision) noexcept {}
#endif


inline std::size_t configured_min_parallel_size() noexcept;

template <class T>
inline void reverse_range_adaptive(T* p, std::size_t n, bool parallel_swap = false) {
    // `std::reverse` is a tight bidirectional swap loop; as one serial pass it
    // is bandwidth-bound and leaves every other core idle.  The mirror pairs
    // are mutually independent, so when the caller is already pool-authorised
    // (the orderedness proof ran striped) and the array is at least a few
    // megabytes, splitting the pair space scales with real memory bandwidth on
    // any multicore machine.  The old task-splitting regression predates the
    // proof-then-swap split: verification happens in the proof pass, so this
    // pass only swaps.
#if FYX_ENABLE_PARALLEL
    if (parallel_swap && n >= 2 && parallel_available() &&
        n * sizeof(T) >= kParallelProofMinBytes) {
        const std::size_t m = n / 2;
        std::size_t chunks =
            std::max<std::size_t>(2, static_cast<std::size_t>(global_pool().nworkers()) * 4);
        if (chunks > m) chunks = m;
        auto job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * m) / chunks;
                const std::size_t hi = ((c + 1) * m) / chunks;
                std::size_t l = lo, r = n - 1 - lo;
                for (; l < hi; ++l, --r) std::swap(p[l], p[r]);
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), job);
        return;
    }
#else
    (void)parallel_swap;
#endif
    std::reverse(p, p + n);
}

template <class T, class Comp>
inline bool try_fast_order_exit(T* p, std::size_t n, Comp comp, bool allow_reverse) {
#if !FYX_ENABLE_FAST_PATHS
    (void)p; (void)n; (void)comp; (void)allow_reverse;
    return false;
#else
    const FastOrderKind k = detect_fast_order_kind(p, n, comp);
    if (k == FastOrderKind::AllEqual) {
        record_dispatch(DispatchDecision::ProfileAllEqual);
        return true;
    }
    if (k == FastOrderKind::Sorted) {
        record_dispatch(DispatchDecision::ProfileSorted);
        return true;
    }
    if (k == FastOrderKind::Reverse && allow_reverse) {
        reverse_range_adaptive(p, n);
        record_dispatch(DispatchDecision::ProfileReverse);
        return true;
    }
    return false;
#endif
}



inline std::size_t parse_parallel_size_env(const char* s) noexcept {
    if (!s || !*s) return 0;
    std::size_t v = 0;
    for (; *s; ++s) {
        if (*s < '0' || *s > '9') return 0;
        const std::size_t digit = static_cast<std::size_t>(*s - '0');
        if (v > (std::numeric_limits<std::size_t>::max() - digit) / 10) return 0;
        v = v * 10 + digit;
    }
    return v;
}

inline std::size_t configured_min_parallel_size() noexcept {
#if FYX_ENABLE_PARALLEL
    static const std::size_t env_min = parse_parallel_size_env(std::getenv("FYX_MIN_PARALLEL_SIZE"));
    // Keep the old kernels available at 1M+ (the public benchmark sizes) but
    // avoid launching the pool for small Auto-mode calls where a serial path or
    // a fast monotone exit is cheaper than task scheduling.
    return env_min != 0 ? env_min : std::size_t(1000000);
#else
    return std::numeric_limits<std::size_t>::max();
#endif
}

template <class T>
inline bool dynamic_parallel_allowed(std::size_t n, const Options& o) {
#if FYX_ENABLE_PARALLEL
    if (o.parallel == Tri::Off || o.threads == 1) return false;
    if (!parallel_available()) return false;
    if (o.parallel != Tri::On && n < configured_min_parallel_size()) return false;

    ThreadPool& pool = global_pool();
    std::size_t workers = static_cast<std::size_t>(pool.nworkers());
    if (o.threads != 0) workers = std::min<std::size_t>(workers, o.threads);
    workers = std::max<std::size_t>(workers, 1);
    constexpr std::size_t min_bytes_per_worker = 128u * 1024u;
    const std::size_t bytes_per_elem = std::max<std::size_t>(std::size_t(1), sizeof(T));
    const std::size_t elems_per_worker = std::max<std::size_t>(1, min_bytes_per_worker / bytes_per_elem);
    if (o.parallel != Tri::On && n / workers < elems_per_worker) return false;
    return true;
#else
    (void)n; (void)o;
    return false;
#endif
}


// ---------------------------------------------------------------------------
// Parallel orderedness exit.
//
// The serial detector below is a single-threaded O(n) scan (with an encode per
// element for floating point).  On 8M already-sorted doubles that scan is the
// whole runtime, and IPS4o's parallel sorted-check beats it by ~2x -- the one
// shape family where a mature parallel quicksort outsorts us on trivial input.
//
// Same proof, split across workers:
//   1. a strided sample decides whether the range is even a candidate; a shape
//      that is neither non-decreasing nor non-increasing declines in a few
//      microseconds, which is what keeps this from taxing everything else;
//   2. otherwise every chunk is classified with the existing serial detector,
//      in parallel, with the chunks that lose the race skipped as soon as any
//      chunk reports disorder;
//   3. the chunk kinds are combined with the usual boundary comparisons, so the
//      answer is exactly the serial one: AllEqual only if every chunk is, and
//      Sorted/Reverse only if every chunk is monotone the same way and the
//      chunk seams agree.
// It never returns true unless the whole range was verified.
// Vectorised monotonicity scan over radix keys: elements [start, hi) must be
// in non-decreasing target order given the key of element start-1.  One load,
// three encode ops, one shifted self-compare per 8/16 elements replaces the
// ~6-op-per-element scalar loop -- on trivial input this scan IS the sort.
// Bit-exact keys keep the floating total order (NaN/-0 gates preserved); the
// scalar tail and the scalar fallback are the same comparison the assist path
// equivalence-tests in test/t_counting.cpp.
#if FYX_HAS_AVX512_CODE
FYX_DIAG_PUSH_SIMD
template <class T>
FYX_TARGET_AVX512 inline bool radix_key_monotone_scan_avx512(
        const T* p, std::size_t start, std::size_t hi,
        typename RadixTraits<T>::Key prev_key,
        const bool want_up, const bool descending) {
    using RT = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr unsigned lanes = (sizeof(Key) == 8) ? 8u : 16u;
    __m512i prev_vec = (sizeof(Key) == 8)
        ? _mm512_set1_epi64(static_cast<long long>(prev_key))
        : _mm512_set1_epi32(static_cast<int>(prev_key));
    const __m512i vsign = (sizeof(Key) == 8)
        ? _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL))
        : _mm512_set1_epi32(static_cast<int>(0x80000000u));
    std::size_t i = start;
    for (; i + lanes <= hi; i += lanes) {
        const __m512i k = _mm512_loadu_si512(reinterpret_cast<const void*>(p + i));
        __m512i e;
        if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_unsigned_v<T>) e = k;
            else e = _mm512_xor_si512(k, vsign);
        } else if constexpr (sizeof(Key) == 8) {
            const __m512i t = _mm512_srai_epi64(k, 63);
            e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        } else {
            const __m512i t = _mm512_srai_epi32(k, 31);
            e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        }
        const __m512i shifted = (sizeof(Key) == 8)
            ? _mm512_alignr_epi64(e, prev_vec, 7)
            : _mm512_alignr_epi32(e, prev_vec, 15);
        const bool viol = (want_up != descending)
            ? ((sizeof(Key) == 8 ? _mm512_cmpgt_epu64_mask(shifted, e)
                                 : _mm512_cmpgt_epu32_mask(shifted, e)) != 0)
            : ((sizeof(Key) == 8 ? _mm512_cmplt_epu64_mask(shifted, e)
                                 : _mm512_cmplt_epu32_mask(shifted, e)) != 0);
        if (viol) return false;
        prev_vec = e;
    }
    if (i > start) prev_key = RT::encode(p[i - 1]);
    for (; i < hi; ++i) {
        const Key cur = RT::encode(p[i]);
        const bool bad = want_up ? (descending ? (prev_key < cur) : (cur < prev_key))
                                 : (descending ? (cur < prev_key) : (prev_key < cur));
        if (bad) return false;
        prev_key = cur;
    }
    return true;
}
FYX_DIAG_POP_SIMD
#endif

template <class T>
inline bool radix_key_monotone_scan(const T* p, std::size_t start, std::size_t hi,
                                    typename RadixTraits<T>::Key prev_key,
                                    const bool want_up, const bool descending) {
#if FYX_HAS_AVX512_CODE
    if (use_avx512())
        return radix_key_monotone_scan_avx512(p, start, hi, prev_key, want_up, descending);
#endif
    using Key = typename RadixTraits<T>::Key;
    Key prev = prev_key;
    for (std::size_t i = start; i < hi; ++i) {
        const Key cur = RadixTraits<T>::encode(p[i]);
        const bool bad = want_up ? (descending ? (prev < cur) : (cur < prev))
                                 : (descending ? (cur < prev) : (prev < cur));
        if (bad) return false;
        prev = cur;
    }
    return true;
}

// Vectorised all-equal sweep. The AVX-512 body is isolated in a target-specific
// helper; its baseline wrapper dispatches only after the runtime feature check.
#if FYX_HAS_AVX512_CODE
FYX_DIAG_PUSH_SIMD
template <class T>
FYX_TARGET_AVX512 inline bool range_all_equal_first_avx512(
        const T* p, std::size_t lo, std::size_t hi) {
    constexpr bool use_key = radix_supported_v<T> && std::is_floating_point<T>::value;
    using RT = RadixTraits<T>;
    using Key = typename RT::Key;
    typename std::conditional<use_key, Key, T>::type first;
    if constexpr (use_key) first = RT::encode(p[0]);
    else                   first = p[0];
    constexpr std::size_t width = sizeof(first);
    const __m512i vfirst = (width == 8)
        ? _mm512_set1_epi64(static_cast<long long>(
              static_cast<typename std::conditional<use_key, std::uint64_t, std::int64_t>::type>(first)))
        : (width == 4)
            ? _mm512_set1_epi32(static_cast<int>(
                  static_cast<typename std::conditional<use_key, std::uint32_t, std::int32_t>::type>(first)))
            : (width == 2)
                ? _mm512_set1_epi16(static_cast<short>(static_cast<std::int16_t>(first)))
                : _mm512_set1_epi8(static_cast<char>(static_cast<std::int8_t>(first)));
    const __mmask64 full = (width == 8) ? __mmask64(0xFF)
                         : (width == 4) ? __mmask64(0xFFFF)
                         : (width == 2) ? __mmask64(0xFFFFFFFFull)
                                        : ~__mmask64(0);
    std::size_t i = lo;
    for (; i + width * 8 <= hi; i += width * 8) {
        const __m512i k = _mm512_loadu_si512(reinterpret_cast<const void*>(p + i));
        __m512i e = k;
        if constexpr (use_key) {
            if constexpr (sizeof(Key) == 8) {
                const __m512i t = _mm512_srai_epi64(k, 63);
                e = _mm512_xor_si512(k, _mm512_or_si512(t, _mm512_set1_epi64(
                    static_cast<long long>(0x8000000000000000ULL))));
            } else {
                const __m512i t = _mm512_srai_epi32(k, 31);
                e = _mm512_xor_si512(k, _mm512_or_si512(t, _mm512_set1_epi32(
                    static_cast<int>(0x80000000u))));
            }
        }
        const __mmask64 m = (width == 8) ? _mm512_cmpeq_epi64_mask(e, vfirst)
                          : (width == 4) ? static_cast<__mmask64>(_mm512_cmpeq_epi32_mask(e, vfirst))
                          : (width == 2) ? static_cast<__mmask64>(_mm512_cmpeq_epi16_mask(e, vfirst))
                                         : static_cast<__mmask64>(_mm512_cmpeq_epi8_mask(e, vfirst));
        if (m != full) return false;
    }
    for (; i < hi; ++i) {
        if constexpr (use_key) {
            if (RT::encode(p[i]) != static_cast<Key>(first)) return false;
        } else if (p[i] != first) return false;
    }
    return true;
}
FYX_DIAG_POP_SIMD
#endif

template <class T>
inline bool range_all_equal_first_vec(const T* p, std::size_t lo, std::size_t hi) {
    if (lo >= hi) return true;
    if constexpr (!std::is_arithmetic<T>::value) {
        (void)p;
        return false;
    } else {
#if FYX_HAS_AVX512_CODE
        if (use_avx512()) return range_all_equal_first_avx512(p, lo, hi);
#endif
        if constexpr (radix_supported_v<T> && std::is_floating_point<T>::value) {
            using RT = RadixTraits<T>;
            const auto k0 = RT::encode(p[0]);
            for (std::size_t i = lo; i < hi; ++i)
                if (RT::encode(p[i]) != k0) return false;
            return true;
        } else {
            const T& f = p[0];
            for (std::size_t i = lo; i < hi; ++i)
                if (!(p[i] == f)) return false;
            return true;
        }
    }
}

// helper-insert-point
template <class T, class Comp>
inline bool try_parallel_fast_order_exit(T* p, std::size_t n, Comp comp, bool allow_reverse) {
#if !FYX_ENABLE_FAST_PATHS || !FYX_ENABLE_PARALLEL
    return try_fast_order_exit(p, n, comp, allow_reverse);
#else
    if (n < 4) return try_fast_order_exit(p, n, comp, allow_reverse);
    constexpr bool radix_order = radix_supported_v<T> && std::is_floating_point<T>::value &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    // only read by the radix-order branch of order_before below
    [[maybe_unused]] constexpr bool descending = is_descending_v<Comp, T>;

    // "x comes strictly before y in the target order"
    auto order_before = [&](const T& x, const T& y) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            using Key = typename RT::Key;
            const Key kx = RT::encode(x);
            const Key ky = RT::encode(y);
            return descending ? (ky < kx) : (kx < ky);
        } else {
            return comp(x, y);
        }
    };

    // 1. sample gate -- 4096 strided probes, no allocation, no threads
    const std::size_t probe = std::min<std::size_t>(n, std::size_t(4096));
    bool sample_up = true, sample_down = true, sample_all_equal = true;
    std::size_t prev = 0;
    for (std::size_t j = 1; j < probe; ++j) {
        const std::size_t idx = (j * (n - 1)) / (probe - 1);
        const bool down = order_before(p[idx], p[prev]);
        const bool upvp = order_before(p[prev], p[idx]);
        if (down) sample_up = false;
        if (upvp) sample_down = false;
        if (down || upvp) sample_all_equal = false;
        if (!sample_up && !sample_down) return false;
        prev = idx;
    }
    if (probe < std::size_t(2)) return try_fast_order_exit(p, n, comp, allow_reverse);

    ThreadPool& pool = global_pool();
    std::size_t chunks = std::max<std::size_t>(2, static_cast<std::size_t>(pool.nworkers()) * 8);
    if (chunks > n) chunks = n;

    // Fast path: the sample found a direction and a strictly ordered pair, so
    // the range is monotone-or-nothing and the answer, if true, is the sampled
    // direction.  Validate exactly that, one comparison per element -- every
    // chunk covers [lo, hi) and c > 0 also re-checks the pair straddling the
    // seam, so touching all elements is the whole proof.  Doing it this way
    // instead of classifying each chunk costs one comparison per element rather
    // than two, which is what a striped std::is_sorted does (and what IPS4o
    // uses); on 8M that is the difference between a bandwidth-bound scan and a
    // scan that cannot keep the prefetcher fed.
    if (!sample_all_equal) {
        const bool want_up = sample_up;
        // [start, hi) is monotone in the sampled direction.  The radix branch
        // keeps the previous *key*, so floating point costs one encode per
        // element instead of two (two encodes per element measured 3x slower
        // than the comparison loop it replaced).
        auto segment_ok = [&](std::size_t start, std::size_t hi) -> bool {
            if constexpr (radix_order) {
                using RT = RadixTraits<T>;
                return radix_key_monotone_scan(p, start, hi, RT::encode(p[start - 1]),
                                               want_up, descending);
            } else {
                if (want_up) {
                    for (std::size_t i = start; i < hi; ++i)
                        if (comp(p[i], p[i - 1])) return false;
                } else {
                    for (std::size_t i = start; i < hi; ++i)
                        if (comp(p[i - 1], p[i])) return false;
                }
                return true;
            }
        };
        std::atomic<bool> ok{true};
        auto job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                if (!ok.load(std::memory_order_relaxed)) return;
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                const std::size_t start = (c == 0) ? lo + 1 : lo;
                if (!segment_ok(start, hi)) { ok.store(false, std::memory_order_relaxed); return; }
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), job);
        if (!ok.load(std::memory_order_relaxed)) return false;
        if (want_up) {
            record_dispatch(DispatchDecision::ProfileSorted);
            return true;
        }
        if (allow_reverse) {
            reverse_range_adaptive(p, n, /*parallel_swap=*/true);
            record_dispatch(DispatchDecision::ProfileReverse);
            return true;
        }
        return false;
    }

    // Fast all-equal sweep before classification: the sample said every
    // probed element matched, so try one vectorised compare-to-first pass
    // per chunk.  It needs no seam bookkeeping (every element is checked
    // against p[0]); on success the verdict is AllEqual outright, and a
    // dissenting element falls through to the classification, which still
    // owns sorted/reverse/all-equal verdicts for mixed shapes.
    {
        std::atomic<bool> eq{true};
        auto eqjob = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                if (!eq.load(std::memory_order_relaxed)) return;
                const std::size_t elo = (c * n) / chunks;
                const std::size_t ehi = ((c + 1) * n) / chunks;
                if (!range_all_equal_first_vec(p, elo, ehi)) {
                    eq.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), eqjob);
        if (eq.load(std::memory_order_relaxed)) {
            record_dispatch(DispatchDecision::ProfileAllEqual);
            return true;
        }
    }

    // 2. chunked classification, for the ranges the sample saw as all-equal
    enum : unsigned char { KNone = 0, KEqual = 1, KSorted = 2, KReverse = 3 };
    std::vector<unsigned char> kind(chunks, KNone);
    std::atomic<bool> dead{false};
    auto job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            if (dead.load(std::memory_order_relaxed)) return;
            const std::size_t lo = (c * n) / chunks;
            const std::size_t hi = ((c + 1) * n) / chunks;
            switch (detect_fast_order_kind(p + lo, hi - lo, comp, true)) {
                case FastOrderKind::AllEqual: kind[c] = KEqual;  break;
                case FastOrderKind::Sorted:   kind[c] = KSorted; break;
                case FastOrderKind::Reverse:  kind[c] = KReverse; break;
                default:                      kind[c] = KNone; dead.store(true, std::memory_order_relaxed); return;
            }
        }
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), job);

    // 3. combine
    bool all_equal = true, any = false;
    for (unsigned char k : kind) {
        if (k == KNone) return false;
        any = true;
        if (k != KEqual) all_equal = false;
    }
    if (!any) return false;
    if (all_equal) {
        record_dispatch(DispatchDecision::ProfileAllEqual);
        return true;
    }
    bool can_sort = sample_up, can_reverse = sample_down;
    for (std::size_t c = 0; c + 1 < chunks; ++c) {
        const unsigned char a = kind[c], b = kind[c + 1];
        if (a != KSorted && a != KEqual) can_sort = false;
        if (b != KSorted && b != KEqual) can_sort = false;
        if (a != KReverse && a != KEqual) can_reverse = false;
        if (b != KReverse && b != KEqual) can_reverse = false;
        if (!can_sort && !can_reverse) return false;
        const std::size_t hi = ((c + 1) * n) / chunks - 1;
        const std::size_t lo = ((c + 1) * n) / chunks;
        if (can_sort && order_before(p[lo], p[hi])) can_sort = false;
        if (can_reverse && order_before(p[hi], p[lo])) can_reverse = false;
        if (!can_sort && !can_reverse) return false;
    }
    if (can_sort) {
        record_dispatch(DispatchDecision::ProfileSorted);
        return true;
    }
    if (can_reverse && allow_reverse) {
        reverse_range_adaptive(p, n, /*parallel_swap=*/true);
        record_dispatch(DispatchDecision::ProfileReverse);
        return true;
    }
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Pool-assisted orderedness proof for ranges below kParallelOrderMinN.
//
// A single-threaded scan is the known weakness on trivial input: at 1M the
// orderedness proof IS the sort, and one thread leaves the second core idle
// exactly there -- which is how IPS4o's striped is_sorted outsorts a full
// parallel quicksort on sorted ranges.  The proof must not get slower for
// everything else, and the >= kParallelOrderMinN exit's opening move, a
// 4096-point *strided* sample, is exactly what an assist band cannot afford:
// on a 1-8 MB range freshly written by the caller the strided probes are cold
// misses with no spatial locality (~0.3 ms measured), more than the scan they
// gate.  So this path opens with a *sequential* prefix instead:
//
//   1. classify the first kAssistOrderPrefix elements with the serial
//      detector.  Sequential reads keep the prefetcher fed, and disordered
//      input -- random, nearly-sorted, rotated, ... -- declines here without
//      ever waking the pool;
//   2. a prefix verdict of Sorted/Reverse validates exactly that direction
//      across the pool, one comparison per element, each chunk re-checking
//      the pair across its seam.  The prefix plus the chunks is the whole
//      range, so this is a proof, not a sample;
//   3. an all-equal prefix falls back to per-chunk classification plus the
//      seam combine, which is also a full proof.
//
// The verdicts and dispatch records are identical to the serial detector's
// (equivalence-tested in test/t_counting.cpp).
// ---------------------------------------------------------------------------
inline constexpr std::size_t kAssistOrderPrefix = 4096;

template <class T, class Comp>
inline bool try_pool_assist_order_exit(T* p, std::size_t n, Comp comp, bool allow_reverse) {
#if !FYX_ENABLE_FAST_PATHS || !FYX_ENABLE_PARALLEL
    (void)p; (void)n; (void)comp; (void)allow_reverse;
    return false;
#else
    if (n < 4) return false;
    constexpr bool radix_order = radix_supported_v<T> && std::is_floating_point<T>::value &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    [[maybe_unused]] constexpr bool descending = is_descending_v<Comp, T>;

    // "x comes strictly before y in the target order"
    auto order_before = [&](const T& x, const T& y) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            const typename RT::Key kx = RT::encode(x);
            const typename RT::Key ky = RT::encode(y);
            return descending ? (ky < kx) : (kx < ky);
        } else {
            return comp(x, y);
        }
    };

    const std::size_t prefix = std::min(n, kAssistOrderPrefix);
    const FastOrderKind head = detect_fast_order_kind(p, prefix, comp);
    if (head == FastOrderKind::None) return false;
    if (prefix == n) {                      // tiny range: the prefix was the proof
        if (head == FastOrderKind::AllEqual) { record_dispatch(DispatchDecision::ProfileAllEqual); return true; }
        if (head == FastOrderKind::Sorted)  { record_dispatch(DispatchDecision::ProfileSorted);    return true; }
        if (allow_reverse) {
            reverse_range_adaptive(p, n);
            record_dispatch(DispatchDecision::ProfileReverse);
            return true;
        }
        return false;
    }

    ThreadPool& pool = global_pool();
    const std::size_t rest = n - prefix;
    std::size_t chunks = std::max<std::size_t>(2, static_cast<std::size_t>(pool.nworkers()) * 8);
    if (chunks > rest) chunks = rest;

    // One-direction validation of [prefix, n): element i is compared with its
    // predecessor exactly once, and chunk c starts at its own lo, so the seam
    // pair (p[lo-1], p[lo]) is inside the chunk.  The radix branch keeps the
    // previous *key*: one encode per element (two measured 3x slower).
    auto validate_direction = [&](bool want_up) -> bool {
        auto segment_ok = [&](std::size_t start, std::size_t hi, const T* prev_elem) -> bool {
            if constexpr (radix_order) {
                using RT = RadixTraits<T>;
                return radix_key_monotone_scan(p, start, hi, RT::encode(*prev_elem),
                                               want_up, descending);
                        } else {
                if (want_up) {
                    for (std::size_t i = start; i < hi; ++i)
                        if (comp(p[i], p[i - 1])) return false;
                } else {
                    for (std::size_t i = start; i < hi; ++i)
                        if (comp(p[i - 1], p[i])) return false;
                }
                return true;
            }
        };
        std::atomic<bool> ok{true};
        auto job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                if (!ok.load(std::memory_order_relaxed)) return;
                const std::size_t lo = prefix + (c * rest) / chunks;
                const std::size_t hi = prefix + ((c + 1) * rest) / chunks;
                if (!segment_ok(lo, hi, p + lo - 1)) { ok.store(false, std::memory_order_relaxed); return; }
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), job);
        return ok.load(std::memory_order_relaxed);
    };

    if (head == FastOrderKind::Sorted || head == FastOrderKind::Reverse) {
        const bool want_up = (head == FastOrderKind::Sorted);
        if (!validate_direction(want_up)) return false;
        if (want_up) {
            record_dispatch(DispatchDecision::ProfileSorted);
            return true;
        }
        if (allow_reverse) {
            reverse_range_adaptive(p, n, /*parallel_swap=*/true);
            record_dispatch(DispatchDecision::ProfileReverse);
            return true;
        }
        return false;
    }

    // Fast all-equal sweep: one vectorised compare-to-first per chunk.  Every
    // element is checked against p[0] directly, so the verdict needs no seam
    // bookkeeping; any dissenting element falls through to the classify path
    // below, which still owns the sorted/reverse verdicts.
    {
        std::atomic<bool> eq{true};
        auto eqjob = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                if (!eq.load(std::memory_order_relaxed)) return;
                const std::size_t elo = prefix + (c * rest) / chunks;
                const std::size_t ehi = prefix + ((c + 1) * rest) / chunks;
                if (!range_all_equal_first_vec(p, elo, ehi)) {
                    eq.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), eqjob);
        if (eq.load(std::memory_order_relaxed)) {
            record_dispatch(DispatchDecision::ProfileAllEqual);
            return true;
        }
    }

    // All-equal prefix: classify the rest per chunk and combine with the seam
    // comparisons.  Same combine shape the >= kParallelOrderMinN path uses for
    // its all-equal sample; the answer is exactly the serial detector's.
    enum : unsigned char { KEqual = 1, KSorted = 2, KReverse = 3 };
    std::vector<unsigned char> kind(chunks + 1, KEqual);
    std::atomic<bool> dead{false};
    auto cjob = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            if (dead.load(std::memory_order_relaxed)) return;
            const std::size_t lo = prefix + (c * rest) / chunks;
            const std::size_t hi = prefix + ((c + 1) * rest) / chunks;
            switch (detect_fast_order_kind(p + lo, hi - lo, comp, true)) {
                case FastOrderKind::AllEqual: kind[c + 1] = KEqual;  break;
                case FastOrderKind::Sorted:   kind[c + 1] = KSorted; break;
                case FastOrderKind::Reverse:  kind[c + 1] = KReverse; break;
                default: kind[c + 1] = 0; dead.store(true, std::memory_order_relaxed); return;
            }
        }
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), cjob);
    for (unsigned char k : kind)
        if (k == 0) return false;

    // Combine: entry 0 is the prefix, entry c >= 1 is chunk c-1.  The seam
    // between the prefix and chunk 0 sits at `prefix`; the seam between
    // chunk c-1 and chunk c sits at prefix + c*rest/chunks.  Together with
    // the per-chunk classifications these constraints cover every adjacent
    // pair of the range exactly once, so the verdict is the serial one.
    bool can_sort = true, can_reverse = true, all_equal = true;
    auto apply_pair = [&](unsigned char a, unsigned char b) {
        if (a != KSorted && a != KEqual) can_sort = false;
        if (b != KSorted && b != KEqual) can_sort = false;
        if (a != KReverse && a != KEqual) can_reverse = false;
        if (b != KReverse && b != KEqual) can_reverse = false;
        if (a != KEqual || b != KEqual) all_equal = false;
    };
    apply_pair(kind[0], kind[1]);
    auto seam_breaks = [&](std::size_t boundary) {              // first index of the right side
        if (order_before(p[boundary], p[boundary - 1])) can_sort = false;
        if (order_before(p[boundary - 1], p[boundary])) can_reverse = false;
    };
    seam_breaks(prefix);
    for (std::size_t c = 1; c < chunks; ++c) {
        if (!can_sort && !can_reverse) return false;
        seam_breaks(prefix + (c * rest) / chunks);
        apply_pair(kind[c], kind[c + 1]);
    }
    if (!can_sort && !can_reverse) return false;
    if (all_equal) {
        record_dispatch(DispatchDecision::ProfileAllEqual);
        return true;
    }
    if (can_sort) {
        record_dispatch(DispatchDecision::ProfileSorted);
        return true;
    }
    if (can_reverse && allow_reverse) {
        reverse_range_adaptive(p, n, /*parallel_swap=*/true);
        record_dispatch(DispatchDecision::ProfileReverse);
        return true;
    }
    return false;
#endif
}

// Pick the parallel proof for large ranges when the caller allowed threads,
// otherwise the serial one.  The proof is identical either way, so a decline
// here only means the caller falls through to the rest of the dispatcher.
// Between the proof-only floor and kParallelOrderMinN the range borrows the
// pool for the proof but keeps a serial sort when the proof declines: the
// scan is read-only, so this cannot perturb anything downstream, and on
// trivial input the scan is the whole runtime -- exactly where a single
// thread is the known weakness.
template <class T, class Comp>
inline bool try_order_exit_adaptive(T* p, std::size_t n, Comp comp, bool allow_reverse,
                                    bool parallel_ok) {
#if FYX_ENABLE_FAST_PATHS && FYX_ENABLE_PARALLEL
    if (parallel_ok && n >= kParallelOrderMinN)
        return try_parallel_fast_order_exit(p, n, comp, allow_reverse);
    if (parallel_ok && n >= kParallelProofMinN &&
        n * sizeof(T) >= kParallelProofMinBytes)
        return try_pool_assist_order_exit(p, n, comp, allow_reverse);
#else
    (void)parallel_ok;
#endif
    return try_fast_order_exit(p, n, comp, allow_reverse);
}

template <class T, class Comp>
inline bool try_parallel_all_equal_exit(T* p, std::size_t n, Comp comp) {
#if !FYX_ENABLE_FAST_PATHS || !FYX_ENABLE_PARALLEL
    (void)p; (void)n; (void)comp;
    return false;
#else
    if constexpr (std::is_arithmetic<T>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    }
    if (n < configured_min_parallel_size() || !parallel_available()) return false;
    constexpr bool radix_order = radix_supported_v<T> &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);

    auto equal_first = [&](const T& x) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            return RT::encode(x) == RT::encode(p[0]);
        } else if constexpr (std::is_same<T, std::string>::value) {
#if FYX_USE_STRING_VIEW
            const std::string& first = p[0];
            return x.size() == first.size() &&
                (x.size() == 0 || std::char_traits<char>::compare(x.data(), first.data(), x.size()) == 0);
#else
            return x == p[0];
#endif
        } else if constexpr (has_equal_operator<T>::value) {
            return x == p[0];
        } else {
            return false;
        }
    };

    if constexpr (!radix_order && !std::is_same<T, std::string>::value && !has_equal_operator<T>::value) {
        (void)equal_first;
        return false;
    } else {
        const std::size_t probe = std::min<std::size_t>(n, 64);
        for (std::size_t j = 1; j < probe; ++j) {
            const std::size_t idx = (j * (n - 1)) / (probe - 1);
            if (!equal_first(p[idx])) return false;
        }

        ThreadPool& pool = global_pool();
        std::size_t chunks = std::max<std::size_t>(2, static_cast<std::size_t>(pool.nworkers()) * 8);
        if (chunks > n) chunks = n;
        std::vector<unsigned char> miss(chunks, 0);
        auto job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                for (std::size_t i = lo; i < hi; ++i) {
                    if (!equal_first(p[i])) { miss[c] = 1; break; }
                }
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), job);
        for (unsigned char v : miss) if (v) return false;
        record_dispatch(DispatchDecision::ProfileAllEqual);
        return true;
    }
#endif
}

template <class T, class Comp>
struct InputProfile {
    bool is_sorted             = false;  // sorted according to Comp
    bool is_reverse            = false;  // reverse of Comp's order
    bool is_all_equal          = false;  // all equivalent under the proven order
    bool is_low_cardinality    = false;  // full-scan-proven distinct/equivalence classes <= 256
    bool is_low_cardinality_candidate = false; // sample hint; counting path validates before commit
    bool is_high_entropy       = false;  // sampled high-cardinality, not near-sorted
    bool is_partially_sorted   = false;  // adjacent inversions <= n / 64
    std::size_t distinct_count = 0;      // 0 means not detected; 257 means >256
    SampleWindow<T> sample_window;       // radix types: the profile sample's key window
};

struct ProfileAdjacentRelation {
    bool cur_before_prev;
    bool prev_before_cur;
    bool equivalent;
};

template <class T, class Comp>
inline ProfileAdjacentRelation profile_relation(const T& prev, const T& cur, Comp comp) {
    (void)comp;
    constexpr bool use_radix_order = radix_supported_v<T> &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    if constexpr (use_radix_order) {
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        const Key a = RT::encode(prev);
        const Key b = RT::encode(cur);
        if constexpr (is_descending_v<Comp, T>) {
            return ProfileAdjacentRelation{a < b, b < a, a == b};
        } else {
            return ProfileAdjacentRelation{b < a, a < b, a == b};
        }
    } else {
        const bool cbp = comp(cur, prev);
        const bool pbc = comp(prev, cur);
        return ProfileAdjacentRelation{cbp, pbc, !cbp && !pbc};
    }
}

template <class T, class Comp, bool UseRadixKey>
class ProfileDistinctTracker;

template <class T, class Comp>
class ProfileDistinctTracker<T, Comp, true> {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    static constexpr std::size_t Cap  = 512;
    static constexpr std::size_t Mask = Cap - 1;

public:
    static constexpr bool supported = true;
    explicit ProfileDistinctTracker(Comp) {}

    bool add(const T& x) noexcept {
        if (overflow_) return false;
        const Key k = RT::encode(x);
        std::size_t h = low_card_hash_key(k) & Mask;
        for (;;) {
            if (!used_[h]) {
                if (distinct_ >= kCountingClassLimit) {
                    overflow_ = true;
                    return false;
                }
                used_[h] = 1;
                keys_[h] = k;
                ++distinct_;
                return true;
            }
            if (keys_[h] == k) return true;
            h = (h + 1) & Mask;
        }
    }

    bool overflow() const noexcept { return overflow_; }
    std::size_t distinct() const noexcept { return distinct_; }

private:
    std::array<Key, Cap> keys_{};
    std::array<unsigned char, Cap> used_{};
    std::size_t distinct_ = 0;
    bool overflow_ = false;
};

template <class T, class Comp>
class ProfileDistinctTracker<T, Comp, false> {
public:
    static constexpr bool supported = std::is_copy_constructible<T>::value;
    explicit ProfileDistinctTracker(Comp comp) : comp_(comp) {
        if constexpr (supported) reps_.reserve(kCountingClassLimit + 1);
    }

    bool add(const T& x) {
        if constexpr (!supported) {
            (void)x;
            return false;
        } else {
            if (overflow_) return false;
            auto it = std::lower_bound(reps_.begin(), reps_.end(), x,
                [&](const T& a, const T& b) { return comp_(a, b); });
            if (it != reps_.end() && !comp_(x, *it)) return true;
            if (reps_.size() >= kCountingClassLimit) {
                overflow_ = true;
                return false;
            }
            reps_.insert(it, x);
            return true;
        }
    }

    bool overflow() const noexcept { return overflow_; }
    std::size_t distinct() const noexcept {
        if constexpr (!supported) return 0;
        else return reps_.size();
    }

private:
    Comp comp_;
    std::vector<T> reps_;
    bool overflow_ = !supported;
};

template <class T, class Comp>
InputProfile<T, Comp> profile_input(const T* data, std::size_t n, Comp comp) {
    InputProfile<T, Comp> prof{};
    if (n == 0) {
        prof.is_sorted = prof.is_reverse = prof.is_all_equal = true;
        return prof;
    }
    if (n == 1) {
        prof.is_sorted = prof.is_reverse = prof.is_all_equal = true;
        prof.is_low_cardinality = true;
        prof.distinct_count = 1;
        return prof;
    }
    if (n < kProfileMinN) return prof;

    constexpr bool use_radix_order = radix_supported_v<T> &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    using Tracker = ProfileDistinctTracker<T, Comp, use_radix_order>;

    const std::size_t s = std::min<std::size_t>(n, kProfileSampleLimit);
    Tracker sample_distinct(comp);
    bool sample_tracks_distinct = Tracker::supported;
    if (sample_tracks_distinct) sample_distinct.add(data[0]);

    bool sample_sorted = true;
    bool sample_reverse = true;
    bool sample_all_equal = true;
    std::size_t sample_inv = 0;
    std::size_t prev_idx = 0;
    // Window capture for the counting kernels (radix types only; zero extra
    // reads -- the loop below already touches these exact elements).
    std::uint64_t wlo = 0, whi = 0;
    if constexpr (use_radix_order) {
        using RTw = RadixTraits<T>;
        const std::uint64_t k0 = static_cast<std::uint64_t>(RTw::encode(data[0]));
        wlo = whi = k0;
    }
    for (std::size_t j = 1; j < s; ++j) {
        const std::size_t idx = (j * (n - 1)) / (s - 1);
        const ProfileAdjacentRelation r = profile_relation(data[prev_idx], data[idx], comp);
        if constexpr (use_radix_order) {
            using RTw = RadixTraits<T>;
            const std::uint64_t k = static_cast<std::uint64_t>(RTw::encode(data[idx]));
            if (k < wlo) wlo = k;
            if (whi < k) whi = k;
        }
        if (r.cur_before_prev) {
            sample_sorted = false;
            ++sample_inv;
        }
        if (r.prev_before_cur) sample_reverse = false;
        if (!r.equivalent) sample_all_equal = false;
        if (sample_tracks_distinct) {
            sample_distinct.add(data[idx]);
            if (sample_distinct.overflow()) sample_tracks_distinct = false;
        }
        prev_idx = idx;
    }

    const bool sample_distinct_overflow = Tracker::supported && sample_distinct.overflow();
    const std::size_t sample_inv_limit = std::max<std::size_t>(1, s / kProfilePartialDivisor);
    const bool need_full_order = sample_sorted || sample_reverse || sample_all_equal ||
                                 sample_inv <= sample_inv_limit;

    if constexpr (use_radix_order) {
        prof.sample_window.valid = true;
        prof.sample_window.lo = wlo;
        prof.sample_window.hi = whi;
        prof.sample_window.distinct = sample_distinct_overflow
            ? (kCountingClassLimit + 1)
            : (sample_tracks_distinct ? sample_distinct.distinct() : std::size_t(0));
    }

    if (!need_full_order) {
        if (sample_distinct_overflow) {
            prof.is_high_entropy = true;
            prof.distinct_count = kCountingClassLimit + 1;
        } else if (Tracker::supported && sample_distinct.distinct() != 0) {
            // Candidate low-cardinality: the actual counting path still
            // validates the complete range before it commits.  Avoiding a full
            // profile-only distinct scan keeps low-cardinality dispatch a net
            // win rather than an extra O(n) tax.
            prof.is_low_cardinality_candidate = true;
            prof.distinct_count = 0;
        }
        return prof;
    }

    const bool need_full_distinct = Tracker::supported && !sample_distinct_overflow;
    Tracker distinct(comp);
    bool track_distinct = need_full_distinct;
    if (track_distinct) distinct.add(data[0]);

    bool sorted = true;
    bool reverse = true;
    bool all_equal = true;
    std::size_t inv = 0;
    const std::size_t inv_limit = n / kProfilePartialDivisor;

    for (std::size_t i = 1; i < n; ++i) {
        const ProfileAdjacentRelation r = profile_relation(data[i - 1], data[i], comp);
        if (r.cur_before_prev) {
            sorted = false;
            ++inv;
        }
        if (r.prev_before_cur) reverse = false;
        if (!r.equivalent) all_equal = false;

        if (track_distinct) {
            distinct.add(data[i]);
            if (distinct.overflow()) track_distinct = false;
        }

        if (!sorted && !reverse && !all_equal && inv > inv_limit && !track_distinct)
            break;
    }

    prof.is_sorted = sorted;
    prof.is_reverse = reverse;
    prof.is_all_equal = all_equal;
    prof.is_partially_sorted = !sorted && !reverse && !all_equal && inv <= inv_limit;

    if (need_full_distinct) {
        if (distinct.overflow()) {
            prof.distinct_count = kCountingClassLimit + 1;
            prof.is_low_cardinality = false;
        } else {
            prof.distinct_count = distinct.distinct();
            prof.is_low_cardinality = prof.distinct_count != 0 && prof.distinct_count <= kCountingClassLimit;
        }
    } else if (sample_distinct_overflow) {
        prof.distinct_count = kCountingClassLimit + 1;
    }

    prof.is_high_entropy = !prof.is_sorted && !prof.is_reverse && !prof.is_all_equal &&
                           !prof.is_partially_sorted && !prof.is_low_cardinality &&
                           (sample_distinct_overflow || prof.distinct_count > kCountingClassLimit);
    return prof;
}

/// One-break structural proof: a range whose prefix and suffix are each
/// monotone in the target order and whose endpoints wrap (last <= first for
/// ascending, last >= first for descending) is a rotation of a sorted range;
/// rotating it back is O(n), stable, and exact.  The scan declines on the
/// second violation, so random input pays only the first few elements.  The
/// sampled profile cannot see this shape -- both runs are individually
/// sorted, which is exactly what the sorted proof checks per chunk.
template <class T>
inline bool try_one_break_rotate(T* p, std::size_t n, bool descending,
                                 bool stable_wrap = false) {
    if constexpr (!radix_supported_v<T>) {
        (void)p; (void)n; (void)descending; (void)stable_wrap;
        return false;
    } else {
        if (n < 3) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        Key prev = RT::encode(p[0]);
        std::size_t brk = n;
        for (std::size_t i = 1; i < n; ++i) {
            const Key cur = RT::encode(p[i]);
            if (descending ? (prev < cur) : (cur < prev)) { brk = i; break; }
            prev = cur;
        }
        if (brk == n) return false;        // 0 breaks: the sorted proof owns it
        prev = RT::encode(p[brk]);         // resume from the break element itself
        for (std::size_t i = brk + 1; i < n; ++i) {
            const Key cur = RT::encode(p[i]);
            if (descending ? (prev < cur) : (cur < prev)) return false;
            prev = cur;
        }
        const Key front = RT::encode(p[0]);
        const Key back  = RT::encode(p[n - 1]);
        const bool cyclic = stable_wrap
            ? (descending ? (back > front) : (back < front))
            : (descending ? (back >= front) : (back <= front));
        if (!cyclic) return false;
        std::rotate(p, p + brk, p + n);
        return true;
    }
}

inline constexpr unsigned kProofStructMaxBreaks = 7;   // up to 8 monotone runs

/// Capped natural-merge front door (engineering name: PSS).
///
/// Not a new sorting paradigm: this is natural mergesort / Timsort with a
/// hard abort after 7 breaks (8 monotone runs).  A single capped pass over
/// radix-encoded keys finds run boundaries; reconstruction is rotate (r=2
/// wrap) or left-preferring stable merge (otherwise).  Random input exceeds
/// the cap within its first few elements and declines.
///   r == 2: rotate back when the endpoints wrap -- the one-break proof --
///           otherwise one stable merge of the two runs;
///   3 <= r <= 8: a binary merge tree over the run boundaries, ceil(log2 r)
///           stable passes of straight merges.
/// Only the rotate can reorder equal keys, and it is gated by `stable_wrap`
/// exactly like the one-break proof; every merge prefers the left run on
/// ties, so the whole family is stable except where explicitly gated.
/// The serial entry is gated by the caller to `!dynamic_parallel_allowed`
/// so it does not run a second full scan on a parallel call.  The parallel
/// counterpart (`try_proof_structured_sort_parallel`) owns the same shapes
/// when the pool is live.
#if FYX_HAS_AVX512_CODE
FYX_DIAG_PUSH_SIMD
template <class T>
FYX_TARGET_AVX512 inline bool proof_structured_scan_avx512(
        const T* p, std::size_t n, bool descending,
        std::size_t* brk, unsigned& nb) {
    using RT = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr unsigned lanes = (sizeof(Key) == 8) ? 8u : 16u;
    const __m512i vsign = (sizeof(Key) == 8)
        ? _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL))
        : _mm512_set1_epi32(static_cast<int>(0x80000000u));
    __m512i prev_vec = (sizeof(Key) == 8)
        ? _mm512_set1_epi64(static_cast<long long>(RT::encode(p[0])))
        : _mm512_set1_epi32(static_cast<int>(RT::encode(p[0])));
    std::size_t i = 1;
    for (; i + lanes <= n; i += lanes) {
        const __m512i k = _mm512_loadu_si512(reinterpret_cast<const void*>(p + i));
        __m512i e;
        if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_unsigned_v<T>) e = k;
            else e = _mm512_xor_si512(k, vsign);
        } else if constexpr (sizeof(Key) == 8) {
            const __m512i t = _mm512_srai_epi64(k, 63);
            e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        } else {
            const __m512i t = _mm512_srai_epi32(k, 31);
            e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        }
        const __m512i shifted = (sizeof(Key) == 8)
            ? _mm512_alignr_epi64(e, prev_vec, 7)
            : _mm512_alignr_epi32(e, prev_vec, 15);
        unsigned bad = (sizeof(Key) == 8)
            ? (descending ? _mm512_cmplt_epu64_mask(shifted, e)
                          : _mm512_cmplt_epu64_mask(e, shifted))
            : (descending ? _mm512_cmplt_epu32_mask(shifted, e)
                          : _mm512_cmplt_epu32_mask(e, shifted));
        while (bad) {
            if (nb == kProofStructMaxBreaks) return false;
            const unsigned bit = static_cast<unsigned>(__builtin_ctz(bad));
            brk[nb++] = i + bit;
            bad &= bad - 1;
        }
        prev_vec = e;
    }
    Key prev = RT::encode(p[i - 1]);
    for (; i < n; ++i) {
        const Key cur = RT::encode(p[i]);
        if (descending ? (prev < cur) : (cur < prev)) {
            if (nb == kProofStructMaxBreaks) return false;
            brk[nb++] = i;
        }
        prev = cur;
    }
    return true;
}
FYX_DIAG_POP_SIMD
#endif

template <class T>
inline bool try_proof_structured_sort(T* p, std::size_t n, bool descending,
                                      bool stable_wrap = false) {
    if constexpr (!radix_supported_v<T>) {
        (void)p; (void)n; (void)descending; (void)stable_wrap;
        return false;
    } else {
        if (n < 4096) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;

        // 1) the capped proof: positions where the target order is violated.
        //    Vectorised exactly like the monotone scan (one load, three encode
        //    ops, one shifted self-compare per 16/8 elements); the violation
        //    mask is drained bit by bit and the cap aborts mid-vector, so a
        //    random input declines within the first 1-2 vector iterations.
        std::size_t brk[kProofStructMaxBreaks];
        unsigned nb = 0;
#if FYX_HAS_AVX512_CODE
        if (use_avx512()) {
            if (!proof_structured_scan_avx512(p, n, descending, brk, nb)) return false;
        } else
#endif
        {
            Key prev = RT::encode(p[0]);
            for (std::size_t i = 1; i < n; ++i) {
                const Key cur = RT::encode(p[i]);
                if (descending ? (prev < cur) : (cur < prev)) {
                    if (nb == kProofStructMaxBreaks) return false;
                    brk[nb++] = i;
                }
                prev = cur;
            }
        }
        if (nb == 0) return false;            // 0 breaks: the monotone exits own it

        const auto key_before = [&](Key a, Key b) {
            return descending ? (b < a) : (a < b);
        };

        if (nb == 1) {
            const std::size_t b = brk[0];
            const Key front = RT::encode(p[0]);
            const Key back  = RT::encode(p[n - 1]);
            const bool wrap = descending ? (back >= front) : (back <= front);
            if (wrap) {
                const bool strict = descending ? (back > front) : (back < front);
                if (stable_wrap && !strict) return false;
                std::rotate(p, p + b, p + n);
                return true;
            }
            // Two runs, no wrap: one stable merge (the concat case and every
            // other two-run shape).
            ScratchLease<T> lease(b);
            if (!lease.valid()) return false;
            T* buf = lease.get();
            std::memcpy(buf, p, b * sizeof(T));
            std::size_t i = 0, j = b, out = 0;
            while (i < b && j < n) {
                const Key ki = RT::encode(buf[i]);
                const Key kj = RT::encode(p[j]);
                if (!key_before(kj, ki)) p[out++] = buf[i++];
                else                     p[out++] = p[j++];
            }
            if (i < b) std::memcpy(p + out, buf + i, (b - i) * sizeof(T));
            return true;
        }

        // 3..8 runs: binary merge tree over the boundaries.
        std::size_t bounds[kProofStructMaxBreaks + 2];
        bounds[0] = 0;
        for (unsigned k = 0; k < nb; ++k) bounds[k + 1] = brk[k];
        bounds[nb + 1] = n;
        unsigned r = nb + 1;

        ScratchLease<T> lease(n);
        if (!lease.valid()) return false;
        T* buf = lease.get();
        T* src = p;
        T* dst = buf;
        while (r > 1) {
            unsigned w = 0;
            for (unsigned k = 0; k + 1 < r; k += 2) {
                const std::size_t lo = bounds[k], mid = bounds[k + 1], hi = bounds[k + 2];
                std::size_t i = lo, j = mid, out = lo;
                while (i < mid && j < hi) {
                    const Key ki = RT::encode(src[i]);
                    const Key kj = RT::encode(src[j]);
                    if (!key_before(kj, ki)) dst[out++] = src[i++];
                    else                     dst[out++] = src[j++];
                }
                if (i < mid) std::memcpy(dst + out, src + i, (mid - i) * sizeof(T));
                if (j < hi)  std::memcpy(dst + out, src + j, (hi - j) * sizeof(T));
                bounds[w + 1] = hi;
                ++w;
            }
            if (r & 1) {
                const std::size_t lo = bounds[r - 1], hi = bounds[r];
                std::memcpy(dst + lo, src + lo, (hi - lo) * sizeof(T));
                bounds[w + 1] = hi;
                ++w;
            }
            bounds[0] = 0;
            r = w;
            std::swap(src, dst);
        }
        if (src != p) std::memcpy(p, buf, n * sizeof(T));
        return true;
    }
}

#if FYX_ENABLE_PARALLEL
template <class T, class Comp>
inline void merge_runs_moving(T* a, std::size_t n1, T* b, std::size_t n2, T* dst, Comp comp);
template <class T>
inline std::size_t radix_key_find_break(const T* p, std::size_t start, std::size_t hi,
                                        typename RadixTraits<T>::Key prev_key,
                                        const bool want_up, const bool descending);
template <class T>
inline bool try_one_break_rotate_parallel(T* p, std::size_t n, bool descending);
template <class T, class Comp>
inline void parallel_merge_to_buffer_rec(T* src,
                                         std::size_t a0, std::size_t a1,
                                         std::size_t b0, std::size_t b1,
                                         T* dst, std::size_t out,
                                         Comp comp);

/// Parallel reconstruction for the capped natural-merge front door.
/// Proof: chunked vectorised break scan (cap 7, abort on the 8th). Repair:
/// r=2 wrap -> the existing double-buffered rotate; otherwise independent
/// pair-merges of the known runs via fork_join (depth ceil(log2 r) <= 3).
/// Declines without mutation if the cap is hit or the arena cannot host a copy.
template <class T>
inline bool try_proof_structured_sort_parallel(T* p, std::size_t n, bool descending,
                                               bool stable_wrap = false) {
    if constexpr (!radix_supported_v<T>) {
        (void)p; (void)n; (void)descending; (void)stable_wrap;
        return false;
    } else {
        constexpr std::size_t kMin = std::size_t(1) << 18;
        if (n < kMin || !parallel_available()) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        const bool want_up = !descending;

        ThreadPool& pool = global_pool();
        std::size_t chunks = std::max<std::size_t>(2, static_cast<std::size_t>(pool.nworkers()) * 4);
        if (chunks * 4096 > n) chunks = std::max<std::size_t>(2, n / 4096);

        std::vector<std::size_t> local_brks(chunks * 8, 0);
        std::vector<unsigned> local_nb(chunks, 0);
        std::vector<unsigned char> seam_bad(chunks, 0);
        std::atomic<bool> give_up{false};

        auto job = [&](std::size_t jc_lo, std::size_t jc_hi) {
            for (std::size_t c = jc_lo; c < jc_hi; ++c) {
                if (give_up.load(std::memory_order_relaxed)) return;
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                if (hi <= lo) continue;
                if (c > 0) {
                    const Key a = RT::encode(p[lo - 1]);
                    const Key b = RT::encode(p[lo]);
                    if (want_up ? (b < a) : (a < b)) seam_bad[c] = 1;
                }
                std::size_t start = lo + 1;
                unsigned nb = 0;
                while (start < hi) {
                    const std::size_t b = radix_key_find_break(
                        p, start, hi, RT::encode(p[start - 1]), want_up, descending);
                    if (b == hi) break;
                    if (nb == 8) { give_up.store(true, std::memory_order_relaxed); return; }
                    local_brks[c * 8 + nb] = b;
                    ++nb;
                    start = b + 1;
                }
                local_nb[c] = nb;
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), job);
        if (give_up.load(std::memory_order_relaxed)) return false;

        std::size_t brk[kProofStructMaxBreaks];
        unsigned nb = 0;
        for (std::size_t c = 0; c < chunks; ++c) {
            if (seam_bad[c]) {
                if (nb == kProofStructMaxBreaks) return false;
                brk[nb++] = (c * n) / chunks;
            }
            for (unsigned k = 0; k < local_nb[c]; ++k) {
                if (nb == kProofStructMaxBreaks) return false;
                brk[nb++] = local_brks[c * 8 + k];
            }
        }
        if (nb == 0) return false;

        auto key_comp = [&](const T& a, const T& b) -> bool {
            const Key ka = RT::encode(a);
            const Key kb = RT::encode(b);
            return descending ? (kb < ka) : (ka < kb);
        };

        if (nb == 1) {
            const std::size_t b = brk[0];
            const Key front = RT::encode(p[0]);
            const Key back  = RT::encode(p[n - 1]);
            const bool wrap = descending ? (back >= front) : (back <= front);
            if (wrap) {
                const bool strict = descending ? (back > front) : (back < front);
                if (stable_wrap && !strict) return false;
                return try_one_break_rotate_parallel(p, n, descending)
                    || (std::rotate(p, p + b, p + n), true);
            }
            ScratchLease<T> lease(n);
            if (!lease.valid()) return false;
            T* buf = lease.get();
            parallel_merge_to_buffer_rec(p, std::size_t(0), b, b, n, buf, std::size_t(0), key_comp);
            auto copy_job = [&](std::size_t lo, std::size_t hi) {
                std::memcpy(p + lo, buf + lo, (hi - lo) * sizeof(T));
            };
            const std::size_t grain = std::max<std::size_t>(std::size_t(1) << 16, n / 8);
            parallel_for_index(std::size_t(0), n, grain, copy_job);
            return true;
        }

        std::size_t bounds[kProofStructMaxBreaks + 2];
        bounds[0] = 0;
        for (unsigned k = 0; k < nb; ++k) bounds[k + 1] = brk[k];
        bounds[nb + 1] = n;
        unsigned r = nb + 1;

        ScratchLease<T> lease(n);
        if (!lease.valid()) return false;
        T* buf = lease.get();
        T* src = p;
        T* dst = buf;
        while (r > 1) {
            const unsigned n_pairs = r / 2;
            auto merge_pair = [&](unsigned pk) {
                const unsigned i = pk * 2u;
                const std::size_t lo = bounds[i], mid = bounds[i + 1], hi = bounds[i + 2];
                merge_runs_moving(src + lo, mid - lo, src + mid, hi - mid, dst + lo, key_comp);
            };
            if (n_pairs == 1) {
                merge_pair(0);
            } else if (n_pairs == 2) {
                fork_join([&] { merge_pair(0); }, [&] { merge_pair(1); });
            } else {
                fork_join([&] { merge_pair(0); if (n_pairs > 2) merge_pair(2); },
                          [&] { merge_pair(1); if (n_pairs > 3) merge_pair(3); });
            }
            if (r & 1) {
                const std::size_t lo = bounds[r - 1], hi = bounds[r];
                std::memcpy(dst + lo, src + lo, (hi - lo) * sizeof(T));
            }
            std::size_t newb[kProofStructMaxBreaks + 2];
            newb[0] = 0;
            unsigned w = 0;
            for (unsigned k = 0; k + 1 < r; k += 2) newb[++w] = bounds[k + 2];
            if (r & 1) newb[++w] = bounds[r];
            for (unsigned i = 0; i <= w; ++i) bounds[i] = newb[i];
            r = w;
            std::swap(src, dst);
        }
        if (src != p) std::memcpy(p, buf, n * sizeof(T));
        return true;
    }
}
#endif

template <class T, class Comp>
inline bool apply_profile_fast_exit(T* p, std::size_t n,
                                    const InputProfile<T, Comp>& prof,
                                    bool allow_reverse) {
    if (prof.is_all_equal) { record_dispatch(DispatchDecision::ProfileAllEqual); return true; }
    if (prof.is_sorted)    { record_dispatch(DispatchDecision::ProfileSorted); return true; }
    if (allow_reverse && prof.is_reverse) {
        reverse_range_adaptive(p, n);
        record_dispatch(DispatchDecision::ProfileReverse);
        return true;
    }
    return false;
}

template <class T>
inline bool try_integer_sparse_count_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!(std::is_integral<T>::value && !std::is_same<T, bool>::value && radix_supported_v<T>)) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr std::size_t Cap  = 1024;
        constexpr std::size_t Mask = Cap - 1;

        std::array<Key, Cap> keys{};
        std::array<std::size_t, Cap> counts{};
        std::array<unsigned char, Cap> used{};
        std::vector<Key> distinct;
        distinct.reserve(kCountingClassLimit);

        for (std::size_t i = 0; i < n; ++i) {
            const Key k = RT::encode(p[i]);
            std::size_t h = low_card_hash_key(k) & Mask;
            for (;;) {
                if (!used[h]) {
                    if (distinct.size() >= kCountingClassLimit) return false;
                    used[h] = 1;
                    keys[h] = k;
                    counts[h] = 1;
                    distinct.push_back(k);
                    break;
                }
                if (keys[h] == k) { ++counts[h]; break; }
                h = (h + 1) & Mask;
            }
        }

        if (distinct.size() <= 1) return true;
        std::sort(distinct.begin(), distinct.end());

        auto lookup_count = [&](Key k) noexcept -> std::size_t {
            std::size_t h = low_card_hash_key(k) & Mask;
            while (used[h]) {
                if (keys[h] == k) return counts[h];
                h = (h + 1) & Mask;
            }
            return 0;
        };

        std::size_t out = 0;
        if (!descending) {
            for (Key k : distinct) {
                const T v = RT::decode(k);
                for (std::size_t c = lookup_count(k); c != 0; --c) p[out++] = v;
            }
        } else {
            for (std::size_t i = distinct.size(); i-- > 0;) {
                const Key k = distinct[i];
                const T v = RT::decode(k);
                for (std::size_t c = lookup_count(k); c != 0; --c) p[out++] = v;
            }
        }
        return true;
    }
}

template <class T>
inline bool try_radix_key_sparse_count_sort(T* p, std::size_t n, bool descending,
                                            const SampleWindow<T>* win) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value) {
        (void)p; (void)n; (void)descending; (void)win;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        if (n > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr std::size_t Cap  = 1024;
        constexpr std::size_t Limit = kCountingClassLimit;

        // When the carried sample already says "a couple dozen values or
        // fewer", a 64-slot table holds them with room to spare and is small
        // enough to stay resident in L1 beside the streaming reads; the
        // 1024-slot table only pays off when the sample hinted a wide
        // distinct set (or nothing at all).  A table that saturates would
        // walk forever, so the small table bails out at 48 distinct -- the
        // 1024-slot caller then answers with its 4x margin.
        const std::size_t guess = (win && win->distinct != 0 && win->distinct <= 24) ? win->distinct : 0;
        const std::size_t cap2  = guess ? 64 : Cap;
        const std::size_t mask2 = cap2 - 1;
        const std::size_t tlimit = guess ? 48 : Limit;

        std::array<Key, Cap> keys{};
        std::array<std::uint32_t, Cap> counts{};
        std::array<unsigned char, Cap> used{};
        std::vector<Key> distinct;
        distinct.reserve(tlimit);

        for (std::size_t i = 0; i < n; ++i) {
            const Key k = RT::encode(p[i]);
            std::size_t h = low_card_hash_key(k) & mask2;
            for (;;) {
                if (!used[h]) {
                    if (distinct.size() >= tlimit) return false;
                    used[h] = 1;
                    keys[h] = k;
                    counts[h] = 1;
                    distinct.push_back(k);
                    break;
                }
                if (keys[h] == k) { ++counts[h]; break; }
                h = (h + 1) & mask2;
            }
        }

        if (distinct.size() <= 1) return true;
        std::sort(distinct.begin(), distinct.end());

        auto lookup_count = [&](Key k) noexcept -> std::uint32_t {
            std::size_t h = low_card_hash_key(k) & mask2;
            while (used[h]) {
                if (keys[h] == k) return counts[h];
                h = (h + 1) & mask2;
            }
            return 0;
        };

        std::size_t out = 0;
        if (!descending) {
            for (Key k : distinct) {
                const T v = RT::decode(k);
                for (std::uint32_t c = lookup_count(k); c != 0; --c) p[out++] = v;
            }
        } else {
            for (std::size_t i = distinct.size(); i-- > 0;) {
                const Key k = distinct[i];
                const T v = RT::decode(k);
                for (std::uint32_t c = lookup_count(k); c != 0; --c) p[out++] = v;
            }
        }
        return true;
    }
}

template <class T, class Comp>
inline bool try_string_value_count_sort(T* p, std::size_t n, Comp comp, bool descending) {
    if constexpr (!std::is_same<T, std::string>::value) {
        (void)p; (void)n; (void)comp; (void)descending;
        return false;
    } else {
        if (n < kCountingMinN) return false;

        std::unordered_map<std::string, std::size_t> counts;
        counts.reserve(kCountingClassLimit * 2);
        std::vector<std::string> distinct;
        distinct.reserve(kCountingClassLimit);

        for (std::size_t i = 0; i < n; ++i) {
            auto it = counts.find(p[i]);
            if (it == counts.end()) {
                if (distinct.size() >= kCountingClassLimit) return false;
                distinct.push_back(p[i]);
                counts.emplace(distinct.back(), std::size_t(1));
            } else {
                ++it->second;
            }
        }
        if (distinct.size() <= 1) return true;
        std::sort(distinct.begin(), distinct.end(), comp);

        std::size_t out = 0;
        for (const std::string& s : distinct) {
            const std::size_t c = counts.find(s)->second;
            std::fill_n(p + out, c, s);
            out += c;
        }
        (void)descending;
        return true;
    }
}

#if FYX_ENABLE_PARALLEL
inline std::size_t adaptive_parallel_chunks(std::size_t n);

template <class T, class Comp>
inline bool try_string_value_count_sort_parallel(T* p, std::size_t n, Comp comp, bool descending) {
    if constexpr (!std::is_same<T, std::string>::value) {
        (void)p; (void)n; (void)comp; (void)descending;
        return false;
    } else {
        if (n < kParallelThreshold || !parallel_available()) return false;
        (void)descending;

        std::unordered_map<std::string, unsigned short> seen;
        seen.reserve(kCountingClassLimit * 2);
        std::vector<std::string> distinct;
        distinct.reserve(kCountingClassLimit);
        const std::size_t s = std::min<std::size_t>(n, kCountingProbeLimit);
        for (std::size_t j = 0; j < s; ++j) {
            const std::size_t idx = (j * n) / s;
            if (seen.find(p[idx]) == seen.end()) {
                if (distinct.size() >= kCountingClassLimit) return false;
                const unsigned short id = static_cast<unsigned short>(distinct.size());
                seen.emplace(p[idx], id);
                distinct.push_back(p[idx]);
            }
        }
        if (distinct.empty()) return false;
        std::sort(distinct.begin(), distinct.end(), comp);

        std::unordered_map<std::string, unsigned short> rank;
        rank.reserve(distinct.size() * 2);
        for (std::size_t r = 0; r < distinct.size(); ++r)
            rank.emplace(distinct[r], static_cast<unsigned short>(r));
        const auto& rank_ref = rank;

        const std::size_t d = distinct.size();
        const std::size_t chunks = adaptive_parallel_chunks(n);
        if ((n + chunks - 1) / chunks >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return false;
        std::vector<std::uint32_t> local(chunks * d, 0);
        std::vector<unsigned char> miss(chunks, 0);
        auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                std::uint32_t* lc = local.data() + c * d;
                for (std::size_t i = lo; i < hi; ++i) {
                    const auto it = rank_ref.find(p[i]);
                    if (it == rank_ref.end()) { miss[c] = 1; break; }
                    ++lc[it->second];
                }
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);
        for (unsigned char v : miss) if (v) return false;

        std::vector<std::size_t> offset(d + 1, 0);
        for (std::size_t out_rank = 0; out_rank < d; ++out_rank) {
            std::size_t total = 0;
            for (std::size_t c = 0; c < chunks; ++c)
                total += local[c * d + out_rank];
            offset[out_rank + 1] = offset[out_rank] + total;
        }

        auto fill_job = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t out_rank = lo; out_rank < hi; ++out_rank) {
                std::fill_n(p + offset[out_rank], offset[out_rank + 1] - offset[out_rank], distinct[out_rank]);
            }
        };
        parallel_for_index(std::size_t(0), d, std::size_t(8), fill_job);
        return true;
    }
}
#endif


template <class Field, class T>
FYX_FORCE_INLINE typename RadixTraits<Field>::Key
load_trivial_field_key(const T& x, std::size_t offset) noexcept {
    Field v;
    std::memcpy(&v, reinterpret_cast<const unsigned char*>(&x) + offset, sizeof(Field));
    return RadixTraits<Field>::encode(v);
}

template <class Field, class T, class Comp>
inline bool trivial_field_candidate_order(T* p, std::size_t n, Comp comp,
                                          std::size_t offset, bool& descending,
                                          std::size_t samples = 96) {
    using Key = typename RadixTraits<Field>::Key;
    if (offset + sizeof(Field) > sizeof(T)) return false;

    const std::size_t s = std::min<std::size_t>(n, std::min<std::size_t>(samples, 96));
    std::array<std::size_t, 96> idx{};
    for (std::size_t i = 0; i < s; ++i) idx[i] = (i * n) / s;

    bool have_order = false;
    bool desc = false;
    for (std::size_t a = 0; a < s; ++a) {
        const T& xa = p[idx[a]];
        const Key ka = load_trivial_field_key<Field>(xa, offset);
        for (std::size_t b = a + 1; b < s; ++b) {
            const T& xb = p[idx[b]];
            const Key kb = load_trivial_field_key<Field>(xb, offset);
            const bool ab = comp(xa, xb);
            const bool ba = comp(xb, xa);
            if (!ab && !ba) {
                if (ka != kb) return false;
                continue;
            }
            if (ka == kb) return false;
            const bool field_ab = ka < kb;
            const bool this_desc = ab ? !field_ab : field_ab;
            if (!have_order) { have_order = true; desc = this_desc; }
            else if (desc != this_desc) return false;

            if (!desc) {
                if (ab != (ka < kb) || ba != (kb < ka)) return false;
            } else {
                if (ab != (kb < ka) || ba != (ka < kb)) return false;
            }
        }
    }
    if (!have_order) return false;
    descending = desc;
    return true;
}

template <class Field, class T>
inline bool trivial_field_low_cardinality_probe(T* p, std::size_t n,
                                                std::size_t offset,
                                                std::size_t* distinct_out = nullptr) {
    using Key = typename RadixTraits<Field>::Key;
    constexpr std::size_t Probe = 512;
    const std::size_t s = std::min<std::size_t>(n, Probe);
    std::array<Key, kCountingClassLimit> seen{};
    std::size_t distinct = 0;
    for (std::size_t j = 0; j < s; ++j) {
        const std::size_t idx = (j * n) / s;
        const Key k = load_trivial_field_key<Field>(p[idx], offset);
        bool found = false;
        for (std::size_t i = 0; i < distinct; ++i) {
            if (seen[i] == k) { found = true; break; }
        }
        if (!found) {
            if (distinct >= kCountingClassLimit) return false;
            seen[distinct++] = k;
        }
    }
    if (distinct_out) *distinct_out = distinct;
    return true;
}

// In-place counting for records with <= 256 distinct sampled-equivalent
// keys.  The out-of-place version below reads and writes the records three
// times (scatter, copy back, check); here a counting pass also records each
// record's class in a byte array, an American-flag cycle permutation moves
// every record once, and a read-only pass proves the order with `comp`.
// Returns 0 = declined untouched, 1 = sorted, 2 = permuted but the proof
// failed (the caller must sort it some other way).
// Class bytes for p[lo, hi): ids[i] = class of p[i] (first-seen order),
// id_key[c] = key of class c, cnt[c] = its count.  Returns the class count,
// or SIZE_MAX past kCountingClassLimit.  Nothing in p moves.
template <class Field, class T>
inline std::size_t flag_classify_range(const T* p, std::size_t lo, std::size_t hi,
                                       std::size_t offset, unsigned char* ids,
                                       typename RadixTraits<Field>::Key* id_key,
                                       std::size_t* cnt) {
    using Key = typename RadixTraits<Field>::Key;
    static_assert(kCountingClassLimit <= 256, "class ids are bytes");
    constexpr std::size_t Cap = 1024, Mask = Cap - 1;
    std::array<Key, Cap> keys{};
    std::array<unsigned short, Cap> slot_id{};      // 0 = empty, else id + 1
    std::size_t distinct = 0;
    if (lo >= hi) return 0;
    Key last_k = load_trivial_field_key<Field>(p[lo], offset);
    unsigned last_id = 256;
    for (std::size_t i = lo; i < hi; ++i) {
        const Key k = load_trivial_field_key<Field>(p[i], offset);
        if (k == last_k && last_id < 256) { ids[i] = static_cast<unsigned char>(last_id); ++cnt[last_id]; continue; }
        std::size_t h = low_card_hash_key(k) & Mask;
        unsigned id;
        for (;;) {
            if (slot_id[h] == 0) {
                if (distinct >= kCountingClassLimit) return static_cast<std::size_t>(-1);
                id = static_cast<unsigned>(distinct);
                slot_id[h] = static_cast<unsigned short>(id + 1u);
                keys[h] = k;
                id_key[distinct++] = k;
                break;
            }
            if (keys[h] == k) { id = slot_id[h] - 1u; break; }
            h = (h + 1) & Mask;
        }
        ids[i] = static_cast<unsigned char>(id);
        ++cnt[id];
        last_k = k; last_id = id;
    }
    return distinct;
}

// Scatter p -> out by class cursor (nx indexed by class byte), with a write
// prefetch per stream: past ~16-32 destination streams the hardware
// prefetchers lose track and every fresh line costs an exposed
// read-for-ownership (few256 kv16 1M: 19.8 -> ~11 ms).  The iterations are
// independent, so this runs at store throughput; an in-place cycle
// permutation is a chain of dependent loads and measured 3x slower.
template <class T>
FYX_FORCE_INLINE void flag_scatter_range(const T* p, std::size_t lo, std::size_t hi,
                                         const unsigned char* ids, std::size_t* nx, T* out) {
    constexpr std::size_t kAheadElems = (256 + sizeof(T) - 1) / sizeof(T);
    for (std::size_t i = lo; i < hi; ++i) {
        const std::size_t d = nx[ids[i]]++;
        prefetch_write<3>(out + d + kAheadElems);
        out[d] = p[i];
    }
}

template <class Field, class T, class Comp>
inline int trivial_field_flag_sort(T* p, std::size_t n, Comp comp, std::size_t offset,
                                   bool descending) {
    using Key = typename RadixTraits<Field>::Key;
    std::array<Key, kCountingClassLimit> id_key{};
    // One lease for both the scatter buffer and the class bytes: a second
    // concurrent lease misses the thread arena and pays fresh page faults on
    // every call (16 MiB of them for 1M 16-byte records).
    const std::size_t out_bytes = (n * sizeof(T) + 63u) & ~std::size_t(63);
    ScratchLease<unsigned char> lease(out_bytes + n);
    if (!lease.valid()) return 0;
    T* out = reinterpret_cast<T*>(lease.get());
    unsigned char* ids = lease.get() + out_bytes;
    std::array<std::size_t, 256> cnt{};
    const std::size_t distinct = flag_classify_range<Field>(p, 0, n, offset, ids, id_key.data(), cnt.data());
    if (distinct == static_cast<std::size_t>(-1)) return 0;
    if (distinct <= 1) return std::is_sorted(p, p + n, comp) ? 1 : 2;
    std::array<unsigned char, kCountingClassLimit> order{};
    for (std::size_t i = 0; i < distinct; ++i) order[i] = static_cast<unsigned char>(i);
    std::sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(distinct),
              [&](unsigned char a, unsigned char b) {
                  return descending ? id_key[b] < id_key[a] : id_key[a] < id_key[b];
              });
    std::array<std::size_t, 256> nx{};
    std::size_t sum = 0;
    for (std::size_t r = 0; r < distinct; ++r) {
        const unsigned id = order[r];
        nx[id] = sum;
        sum += cnt[id];
    }
    flag_scatter_range(p, 0, n, ids, nx.data(), out);
    // Copy back while proving the order.
    bool ok = true;
    p[0] = out[0];
    for (std::size_t i = 1; i < n; ++i) {
        p[i] = out[i];
        ok &= !comp(out[i], out[i - 1]);
    }
    return ok ? 1 : 2;
}

#if FYX_ENABLE_PARALLEL
// Parallel flag sort: chunks classify independently (local class tables),
// the tables are merged into one key-ordered class list, and each chunk
// scatters through its local->global cursor map.  Same return contract.
template <class Field, class T, class Comp>
inline int trivial_field_flag_sort_parallel(T* p, std::size_t n, Comp comp, std::size_t offset,
                                            bool descending) {
    using Key = typename RadixTraits<Field>::Key;
    const std::size_t chunks = adaptive_parallel_chunks(n);
    if (chunks < 2) return trivial_field_flag_sort<Field>(p, n, comp, offset, descending);
    const std::size_t out_bytes = (n * sizeof(T) + 63u) & ~std::size_t(63);
    ScratchLease<unsigned char> lease(out_bytes + n);
    if (!lease.valid()) return 0;
    T* out = reinterpret_cast<T*>(lease.get());
    unsigned char* ids = lease.get() + out_bytes;
    std::vector<Key> lkey(chunks * 256);
    std::vector<std::size_t> lcnt(chunks * 256, 0), ldist(chunks, 0);
    auto classify_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c)
            ldist[c] = flag_classify_range<Field>(p, (c * n) / chunks, ((c + 1) * n) / chunks, offset,
                                                  ids, lkey.data() + c * 256, lcnt.data() + c * 256);
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), classify_job);
    std::vector<Key> gkey;
    gkey.reserve(256);
    for (std::size_t c = 0; c < chunks; ++c) {
        if (ldist[c] == static_cast<std::size_t>(-1)) return 0;
        for (std::size_t j = 0; j < ldist[c]; ++j) gkey.push_back(lkey[c * 256 + j]);
    }
    if (descending) std::sort(gkey.begin(), gkey.end(), [](Key a, Key b) { return b < a; });
    else std::sort(gkey.begin(), gkey.end());
    gkey.erase(std::unique(gkey.begin(), gkey.end()), gkey.end());
    if (gkey.size() > kCountingClassLimit) return 0;
    const std::size_t G = gkey.size();
    // Global rank of every local class, then per-chunk cursors in rank order.
    std::vector<unsigned short> rank(chunks * 256, 0);
    std::vector<std::size_t> gcnt(chunks * G, 0);
    for (std::size_t c = 0; c < chunks; ++c)
        for (std::size_t j = 0; j < ldist[c]; ++j) {
            const Key k = lkey[c * 256 + j];
            const auto it = descending
                ? std::lower_bound(gkey.begin(), gkey.end(), k, [](Key a, Key b) { return b < a; })
                : std::lower_bound(gkey.begin(), gkey.end(), k);
            const std::size_t g = static_cast<std::size_t>(it - gkey.begin());
            rank[c * 256 + j] = static_cast<unsigned short>(g);
            gcnt[c * G + g] = lcnt[c * 256 + j];
        }
    std::vector<std::size_t> gstart(chunks * G, 0);
    std::size_t sum = 0;
    for (std::size_t g = 0; g < G; ++g)
        for (std::size_t c = 0; c < chunks; ++c) { gstart[c * G + g] = sum; sum += gcnt[c * G + g]; }
    auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            std::array<std::size_t, 256> nx{};
            for (std::size_t j = 0; j < ldist[c]; ++j) nx[j] = gstart[c * G + rank[c * 256 + j]];
            flag_scatter_range(p, (c * n) / chunks, ((c + 1) * n) / chunks, ids, nx.data(), out);
        }
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);
    std::atomic<bool> bad{false};
    auto back_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            const std::size_t lo = (c * n) / chunks, hi = ((c + 1) * n) / chunks;
            bool ok = true;
            for (std::size_t i = lo; i < hi; ++i) {
                p[i] = out[i];
                if (i > 0) ok &= !comp(out[i], out[i - 1]);
            }
            if (!ok) bad.store(true, std::memory_order_relaxed);
        }
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), back_job);
    return bad.load() ? 2 : 1;
}
#endif

// 16-byte records keyed by a 64-bit field: in-place AVX-512 record quicksort
// (no scratch, no copy back; equal-key ranges end without leaves).  Returns
// 1 sorted, 2 permuted but not sorted under comp (caller re-sorts), 0 not
// applicable (untouched).
template <class T, class Comp>
struct Kv16Proof {
    // Seams per task slot (task tree ids < kSlots): no locking, each slot is
    // written by one task at a time.
    static constexpr unsigned kSlots = 64;
    T* p;
    Comp* comp;
    bool descending;
    std::vector<std::size_t> seams[kSlots];

    bool verify(std::size_t lo, std::size_t cnt, bool with_prev) const {
        // OR-reduction with no early exit: the compiler vectorizes it for
        // plain key comparators (strided key loads).
        if (cnt == 0) return true;
        const T* q = p + (with_prev ? lo - 1 : lo);
        const std::size_t m = cnt - (with_prev ? 0 : 1);
        const Comp& c = *comp;
        unsigned char bad = 0;
        if (descending) for (std::size_t i = 0; i < m; ++i) bad |= static_cast<unsigned char>(c(q[i], q[i + 1]));
        else            for (std::size_t i = 0; i < m; ++i) bad |= static_cast<unsigned char>(c(q[i + 1], q[i]));
        return bad == 0;
    }
    bool equiv(std::size_t ref, std::size_t lo, std::size_t cnt) const {
        const T r = p[ref];
        const T* q = p + lo;
        const Comp& c = *comp;
        unsigned char bad = 0;
        for (std::size_t i = 0; i < cnt; ++i)
            bad |= static_cast<unsigned char>(c(q[i], r) | c(r, q[i]));
        return bad == 0;
    }
    bool seams_ok() const {
        for (unsigned s = 0; s < kSlots; ++s)
            for (const std::size_t i : seams[s])
                if (descending ? (*comp)(p[i - 1], p[i]) : (*comp)(p[i], p[i - 1])) return false;
        return true;
    }
};

template <class T, class Comp>
struct Kv16Fin {
    Kv16Proof<T, Comp>* pr;
    unsigned slot;
    bool verify(std::size_t lo, std::size_t cnt, bool with_prev) { return pr->verify(lo, cnt, with_prev); }
    bool equiv(std::size_t ref, std::size_t lo, std::size_t cnt) { return pr->equiv(ref, lo, cnt); }
    void seam(std::size_t lo) { if (lo != 0) pr->seams[slot].push_back(lo); }
    bool operator()(std::size_t lo, std::size_t cnt) {
        seam(lo);
        return pr->verify(lo, cnt, false);
    }
};

#if FYX_ENABLE_PARALLEL
template <class T, class Comp>
inline bool kv16_vqsort_parallel_rec(T* p, std::size_t lo, std::size_t nr, int budget, unsigned depth,
                                     bool key_hi, bool is_signed, Kv16Proof<T, Comp>& pr, unsigned slot) {
    constexpr std::size_t kMinParTask = std::size_t(1) << 15;
    Kv16Fin<T, Comp> fin{&pr, slot};
    while (true) {
        if (depth == 0 || budget <= 0 || nr < kMinParTask)
            return kv16_vqsort_range(static_cast<void*>(p), lo, nr, budget, key_hi, is_signed, fin);
        std::size_t split = 0;
        const int r = kv16_vqsort_step(static_cast<void*>(p), lo, nr, split, key_hi, is_signed, fin);
        --budget;
        if (r == 0) return true;
        if (r < 0) return false;
        if (r == 1) continue;
        bool okl = false, okr = false;
        const std::size_t l0 = lo, ln = split, r0 = lo + split, rn = nr - split;
        const int nb = budget;
        const unsigned nd = depth - 1;
        fork_join([&] { okl = kv16_vqsort_parallel_rec(p, l0, ln, nb, nd, key_hi, is_signed, pr, 2 * slot); },
                  [&] { okr = kv16_vqsort_parallel_rec(p, r0, rn, nb, nd, key_hi, is_signed, pr, 2 * slot + 1); });
        return okl && okr;
    }
}
#endif

// 16-byte records keyed by a 64-bit field: in-place AVX-512 record quicksort
// (no scratch, no copy back; equal-key ranges end without leaves).  Every
// final range is proven under comp while hot and the seams between ranges
// afterwards (ascending, so a descending comp is checked mirrored before the
// reverse).  par: fork the top levels over the pool.  Returns 1 sorted, 2
// permuted but not sorted under comp (caller re-sorts), 0 not applicable
// (untouched).
template <class Field, class T, class Comp>
FYX_NOINLINE int trivial_field_kv16_vqsort(T* p, std::size_t n, Comp comp, std::size_t offset,
                                           bool descending, bool par = false) {
    if constexpr (sizeof(T) != 16 || sizeof(Field) != 8 || !std::is_trivially_copyable<T>::value) {
        (void)p; (void)n; (void)comp; (void)offset; (void)descending; (void)par;
        return 0;
    } else {
        if ((offset != 0 && offset != 8) || !use_avx512()) return 0;
        const bool key_hi = offset == 8;
        constexpr bool is_signed = std::is_signed<Field>::value;
        std::unique_ptr<Kv16Proof<T, Comp>> pr(new Kv16Proof<T, Comp>{p, &comp, descending, {}});
        bool ok;
#if FYX_ENABLE_PARALLEL
        if (par && parallel_available()) {
            unsigned depth = 1;
            for (unsigned t = 2; t < global_pool().nworkers() * 2u && depth < 5; t *= 2) ++depth;
            ok = kv16_vqsort_parallel_rec(p, 0, n, vqsort_budget(n), depth, key_hi, is_signed, *pr, 1u);
        } else
#else
        (void)par;
#endif
        {
            Kv16Fin<T, Comp> fin{pr.get(), 0};
            ok = kv16_vqsort(static_cast<void*>(p), n, key_hi, is_signed, fin);
        }
        if (!ok || !pr->seams_ok()) return 2;
        if (descending) std::reverse(p, p + n);
        return 1;
    }
}

// Early route for unordered 16-byte records keyed by a 64-bit field (AVX-512
// hosts): the in-place record quicksort beats the profile -> counting / MSD
// chain on every unordered shape measured (random, few-unique, sqrt, zipf),
// so it runs before the probe stack -- after a 64-sample order gate (both
// directions, frequent turns) that leaves presorted, few-run and pattern
// shapes (organ pipe, rotations, sorted prefixes) to the adaptive paths.
// Many short sorted runs pass the gate on purpose: at sqrt(n) runs the
// quicksort beats run merging (1M: 18.1 vs 20.5 ms; 10k: 0.10 vs 0.49 ms).
// true: sorted.  false: p holds a permutation of its input (maybe unchanged).
// Equivalence of every element to p[0] (a strict weak order makes that
// transitive): one load stream, OR-reduced per block.  AVX-512 target so a
// portable build still vectorizes the inlined comparator 512 bits wide (only
// called after use_avx512()).
template <class T, class Comp>
FYX_TARGET_AVX512 bool kv16_all_equiv_avx512(const T* p, std::size_t n, Comp& comp) {
    // Blocks run back to front: a producer that just wrote the range left
    // its tail in L1/L2, and a forward scan of a range larger than L2 would
    // evict exactly the lines it reads next (LRU), so tail-first takes the
    // cache hits (2-4% at 100k / 1M right after a copy).
    constexpr std::size_t B = 2048;
    const T ref = p[0];
    std::size_t e = n;
    while (e > 1) {
        const std::size_t b = e > B + 1 ? e - B : 1;
        const T* q = p + b;
        const std::size_t m = e - b;
        unsigned char bad = 0;
        for (std::size_t i = 0; i < m; ++i)
            bad |= static_cast<unsigned char>(comp(q[i], ref) | comp(ref, q[i]));
        if (bad) return false;
        e = b;
    }
    return true;
}

template <class Field, class T, class Comp>
inline int kv16_early_field(T* p, std::size_t n, Comp comp, std::size_t off, bool par) {
    bool desc = false;
    if (!trivial_field_candidate_order<Field>(p, n, comp, off, desc, 48)) return 0;
    return trivial_field_kv16_vqsort<Field>(p, n, comp, off, desc, par);
}

template <class T, class Comp>
inline bool try_kv16_early_sort(T* p, std::size_t n, Comp comp, bool par = false) {
    if constexpr (sizeof(T) != 16 || !std::is_trivially_copyable<T>::value || std::is_arithmetic<T>::value) {
        (void)p; (void)n; (void)comp; (void)par;
        return false;
    } else {
        if (n < kKv16EarlyMinN || !use_avx512()) return false;
        constexpr std::size_t S = 64;
        std::size_t asc = 0, dsc = 0, turns = 0;
        int last = 0;
        const T* prev = p;
        for (std::size_t j = 1; j < S; ++j) {
            const T* cur = p + (j * (n - 1)) / (S - 1);
            const bool up = comp(*prev, *cur), down = comp(*cur, *prev);
            asc += up ? 1u : 0u;
            dsc += down ? 1u : 0u;
            const int dir = up ? 1 : (down ? -1 : 0);
            turns += (dir != 0 && last != 0 && dir != last) ? 1u : 0u;
            last = dir != 0 ? dir : last;
            prev = cur;
        }
        if (asc == 0 && dsc == 0) {
            // Every sample equivalent: prove the range all-equivalent (hence
            // sorted) in one vectorizable pass, OR-reduced per block -- the
            // profile's tracker scan costs ~3x more.
            return kv16_all_equiv_avx512(p, n, comp);
        }
        // Both directions often, and alternating (organ pipes / bitonic
        // shapes have both but turn once or twice).
        if (asc < S / 5 || dsc < S / 5 || turns < S / 5) return false;
        for (std::size_t off = 0; off <= 8; off += 8) {
            int r = kv16_early_field<std::int64_t>(p, n, comp, off, par);
            if (r == 0) r = kv16_early_field<std::uint64_t>(p, n, comp, off, par);
            if (r != 0) return r == 1;
        }
        return false;
    }
}

template <class Field, class T, class Comp>
inline bool try_trivial_field_count_sort(T* p, std::size_t n, Comp comp,
                                         std::size_t offset, bool par = false) {
    using Key = typename RadixTraits<Field>::Key;
    bool descending = false;
    if (!trivial_field_candidate_order<Field>(p, n, comp, offset, descending)) return false;
    std::size_t sample_distinct = 0;
    if (!trivial_field_low_cardinality_probe<Field>(p, n, offset, &sample_distinct)) return false;

    // Class-byte flag sort for sparse key domains with up to 255 sampled
    // classes (the scatter write-prefetches each stream, so 256 streams no
    // longer fall off the prefetchers' cliff).
    if (sample_distinct <= 255) {
        const int r = trivial_field_kv16_vqsort<Field>(p, n, comp, offset, descending,
                                                       par && n >= kKv16EarlyParallelMinN);
        if (r != 0) return r == 1;
    }
    if constexpr (std::is_trivially_copyable<T>::value) if (sample_distinct <= 255) {
        // Sample span first: a sparse key domain (hashes, random 64-bit
        // palettes) can skip the dense-range min/max pass altogether.
        Key smn = load_trivial_field_key<Field>(p[0], offset), smx = smn;
        for (std::size_t j = 0; j < 256; ++j) {
            const Key k = load_trivial_field_key<Field>(p[(j * n) / 256], offset);
            smn = k < smn ? k : smn;
            smx = smx < k ? k : smx;
        }
        const unsigned long long sspan = static_cast<unsigned long long>(static_cast<Key>(smx - smn));
        if (sspan >= static_cast<unsigned long long>(std::min<std::size_t>(kCountingRangeLimit, std::max<std::size_t>(n, 4096)))) {
#if FYX_ENABLE_PARALLEL
            const int r = par && n >= kParallelThreshold && parallel_available()
                ? trivial_field_flag_sort_parallel<Field>(p, n, comp, offset, descending)
                : trivial_field_flag_sort<Field>(p, n, comp, offset, descending);
#else
            (void)par;
            const int r = trivial_field_flag_sort<Field>(p, n, comp, offset, descending);
#endif
            if (r != 0) return r == 1;
        }
    }

    // Fast subpath for the overwhelmingly common struct-key case: the key field
    // is a dense integer domain (e.g. `struct { int key; ... }` with 64 keys).
    // This avoids hash probes in the scatter loop; correctness is still guarded
    // by the final comparator-based sorted check.
    {
        Key mn = load_trivial_field_key<Field>(p[0], offset);
        Key mx = mn;
        for (std::size_t i = 1; i < n; ++i) {
            const Key k = load_trivial_field_key<Field>(p[i], offset);
            if (k < mn) mn = k;
            if (mx < k) mx = k;
        }
        const Key span = static_cast<Key>(mx - mn);
        if (span != std::numeric_limits<Key>::max()) {
            const unsigned long long range64 = static_cast<unsigned long long>(span) + 1ull;
            const std::size_t limit = std::min<std::size_t>(kCountingRangeLimit,
                                                            std::max<std::size_t>(n, 4096));
            if (range64 <= static_cast<unsigned long long>(limit)) {
                const std::size_t range = static_cast<std::size_t>(range64);
                // Counters and scatter buffer share one lease (a second,
                // concurrent lease would miss the thread arena and fault in
                // fresh pages on every call).
                const std::size_t out_bytes = (n * sizeof(T) + 63u) & ~std::size_t(63);
                ScratchLease<unsigned char> both_lease(out_bytes + range * sizeof(std::size_t));
                if (both_lease.valid()) {
                    std::size_t* counts = reinterpret_cast<std::size_t*>(both_lease.get() + out_bytes);
                    for (std::size_t i = 0; i < range; ++i) counts[i] = 0;
                    for (std::size_t i = 0; i < n; ++i)
                        ++counts[static_cast<std::size_t>(load_trivial_field_key<Field>(p[i], offset) - mn)];

                    std::size_t distinct = 0;
                    for (std::size_t i = 0; i < range; ++i) distinct += counts[i] != 0;
                    if (distinct <= kCountingClassLimit) {
                        std::size_t sum = 0;
                        if (!descending) {
                            for (std::size_t i = 0; i < range; ++i) {
                                const std::size_t c = counts[i];
                                counts[i] = sum;
                                sum += c;
                            }
                        } else {
                            for (std::size_t i = range; i-- > 0;) {
                                const std::size_t c = counts[i];
                                counts[i] = sum;
                                sum += c;
                            }
                        }

                        {
                            T* out = reinterpret_cast<T*>(both_lease.get());
                            for (std::size_t i = 0; i < n; ++i) {
                                const std::size_t r = static_cast<std::size_t>(
                                    load_trivial_field_key<Field>(p[i], offset) - mn);
                                out[counts[r]++] = p[i];
                            }
                            for (std::size_t i = 0; i < n; ++i) p[i] = out[i];
                            return std::is_sorted(p, p + n, comp);
                        }
                    }
                }
            }
        }
    }

    constexpr std::size_t Cap  = 1024;
    constexpr std::size_t Mask = Cap - 1;
    std::array<Key, Cap> keys{};
    std::array<std::size_t, Cap> counts{};
    std::array<unsigned char, Cap> used{};
    std::vector<Key> distinct;
    distinct.reserve(kCountingClassLimit);

    for (std::size_t i = 0; i < n; ++i) {
        const Key k = load_trivial_field_key<Field>(p[i], offset);
        std::size_t h = low_card_hash_key(k) & Mask;
        for (;;) {
            if (!used[h]) {
                if (distinct.size() >= kCountingClassLimit) return false;
                used[h] = 1;
                keys[h] = k;
                counts[h] = 1;
                distinct.push_back(k);
                break;
            }
            if (keys[h] == k) { ++counts[h]; break; }
            h = (h + 1) & Mask;
        }
    }
    if (distinct.size() <= 1) return true;
    std::sort(distinct.begin(), distinct.end());

    auto lookup_slot = [&](Key k) noexcept -> std::size_t {
        std::size_t h = low_card_hash_key(k) & Mask;
        while (used[h]) {
            if (keys[h] == k) return h;
            h = (h + 1) & Mask;
        }
        return Cap;
    };
    auto lookup_count = [&](Key k) noexcept -> std::size_t {
        const std::size_t h = lookup_slot(k);
        return h == Cap ? 0 : counts[h];
    };

    std::array<std::size_t, kCountingClassLimit> pos{};
    if (!descending) {
        std::size_t sum = 0;
        for (std::size_t i = 0; i < distinct.size(); ++i) {
            pos[i] = sum;
            sum += lookup_count(distinct[i]);
        }
    } else {
        std::size_t sum = 0;
        for (std::size_t rr = distinct.size(); rr-- > 0;) {
            pos[rr] = sum;
            sum += lookup_count(distinct[rr]);
        }
    }

    for (std::size_t r = 0; r < distinct.size(); ++r) {
        const std::size_t h = lookup_slot(distinct[r]);
        if (h != Cap) counts[h] = r;   // counts[] becomes key -> rank
    }
    auto lookup_rank = [&](Key k) noexcept -> std::size_t {
        const std::size_t h = lookup_slot(k);
        return h == Cap ? 0 : counts[h];
    };

    ScratchLease<T> out_lease(n);
    if (!out_lease.valid()) return false;
    T* out = out_lease.get();

    for (std::size_t i = 0; i < n; ++i) {
        const Key k = load_trivial_field_key<Field>(p[i], offset);
        const std::size_t r = lookup_rank(k);
        out[pos[r]++] = p[i];
    }
    for (std::size_t i = 0; i < n; ++i) p[i] = out[i];

    return std::is_sorted(p, p + n, comp);
}


// ---------------------------------------------------------------------------
// MSD radix for trivially copyable records keyed by one integer field.
// The LSD record radix above pays a full read+write of every record per active
// digit -- eight passes for a random 64-bit key, which left 1M 16-byte records
// 3-4x behind ips2ra / vqsort's key-value mode.  This version spends one
// streaming pass (min/max), one histogram pass and one scatter into scratch;
// everything after that works on buckets that fit in L1/L2: a range-adaptive
// second digit, insertion-sorted leaves, and per-bucket comparator checks
// while the bucket is still cached (the field/comparator equivalence is only
// sampled, so the final order is always proven with `comp`).
// ---------------------------------------------------------------------------
template <class Key>
FYX_FORCE_INLINE unsigned record_msd_key_bits(Key d) noexcept {
    unsigned b = 0;
    while (d != 0) { ++b; d = static_cast<Key>(d >> 1); }
    return b;
}

template <class Field, class T>
FYX_FORCE_INLINE typename RadixTraits<Field>::Key
record_msd_key(const T& x, std::size_t offset, bool descending) noexcept {
    using Key = typename RadixTraits<Field>::Key;
    const Key k = load_trivial_field_key<Field>(x, offset);
    return descending ? static_cast<Key>(~k) : k;
}

template <class Field, class T>
inline void record_msd_insertion(T* p, std::size_t m, std::size_t offset, bool descending) {
    using Key = typename RadixTraits<Field>::Key;
    for (std::size_t i = 1; i < m; ++i) {
        const Key k = record_msd_key<Field>(p[i], offset, descending);
        if (!(k < record_msd_key<Field>(p[i - 1], offset, descending))) continue;
        T tmp = p[i];
        std::size_t j = i;
        do { p[j] = p[j - 1]; --j; }
        while (j > 0 && k < record_msd_key<Field>(p[j - 1], offset, descending));
        p[j] = tmp;
    }
}

#ifndef FYX_RMSD_TOP_BITS
#  define FYX_RMSD_TOP_BITS 6
#endif
#ifndef FYX_RMSD_LEAF
#  define FYX_RMSD_LEAF 24
#endif
#ifndef FYX_RMSD_SUB_SHIFT
#  define FYX_RMSD_SUB_SHIFT 3
#endif
#ifndef FYX_RMSD_MAX_BITS
#  define FYX_RMSD_MAX_BITS 11
#endif
inline constexpr unsigned kRecordMsdMaxBits  = FYX_RMSD_MAX_BITS;
inline constexpr unsigned kRecordMsdMaxDepth = 10;
#ifndef FYX_RMSD_RANK_LEAF
#  define FYX_RMSD_RANK_LEAF 1
#endif

// Branch-free leaf: each record's final slot is its rank (keys below it plus
// equal keys earlier in the bucket), so a bucket of s records costs s*s
// predictable compares instead of insertion sort's data-dependent branches.
// `scratch` (the bucket's slot in the other buffer) is free at this point.
template <class Field, class T>
inline void record_msd_rank_leaf(T* p, T* scratch, std::size_t s, std::size_t offset,
                                 bool descending) {
    using Key = typename RadixTraits<Field>::Key;
    Key ks[FYX_RMSD_LEAF];
    for (std::size_t i = 0; i < s; ++i) ks[i] = record_msd_key<Field>(p[i], offset, descending);
    std::memcpy(static_cast<void*>(scratch), static_cast<const void*>(p), s * sizeof(T));
    for (std::size_t i = 0; i < s; ++i) {
        const Key k = ks[i];
        std::size_t r = 0;
        for (std::size_t j = 0; j < i; ++j) r += ks[j] <= k;
        for (std::size_t j = i + 1; j < s; ++j) r += ks[j] < k;
        p[r] = scratch[i];
    }
}
inline constexpr std::size_t kRecordMsdFrame = (std::size_t(2) << kRecordMsdMaxBits) + 1;

// Sorts src[0,m) into dst[0,m); src is scratch afterwards.  `work` holds
// kRecordMsdFrame counters per remaining recursion level.
template <class Field, class T>
inline void record_msd_rec(T* src, T* dst, std::size_t m, std::size_t offset,
                           bool descending, unsigned depth, std::size_t* work) {
    using Key = typename RadixTraits<Field>::Key;
    constexpr std::size_t kLeaf = FYX_RMSD_LEAF;
    if (m <= kLeaf || depth >= kRecordMsdMaxDepth) {
        std::memcpy(static_cast<void*>(dst), static_cast<const void*>(src), m * sizeof(T));
        if (m <= kLeaf) record_msd_insertion<Field>(dst, m, offset, descending);
        else {
            std::sort(dst, dst + m, [&](const T& a, const T& b) {
                return record_msd_key<Field>(a, offset, descending) <
                       record_msd_key<Field>(b, offset, descending);
            });
        }
        return;
    }
    Key mn = record_msd_key<Field>(src[0], offset, descending), mx = mn;
    for (std::size_t i = 1; i < m; ++i) {
        const Key k = record_msd_key<Field>(src[i], offset, descending);
        mn = k < mn ? k : mn;
        mx = mx < k ? k : mx;
    }
    if (mn == mx) {
        std::memcpy(static_cast<void*>(dst), static_cast<const void*>(src), m * sizeof(T));
        return;
    }
    const unsigned bits = record_msd_key_bits(static_cast<Key>(mx - mn));
    unsigned lg = 0;
    while ((std::size_t(1) << (lg + 1)) <= m) ++lg;
    unsigned B = lg > FYX_RMSD_SUB_SHIFT ? lg - FYX_RMSD_SUB_SHIFT : 1;
    if (B > kRecordMsdMaxBits) B = kRecordMsdMaxBits;
    if (B > bits) B = bits;
    const unsigned shift = bits - B;
    const std::size_t nb = std::size_t(1) << B;

    std::size_t* cnt = work;
    std::size_t* pos = work + nb + 1;
    std::memset(cnt, 0, (nb + 1) * sizeof(std::size_t));
    for (std::size_t i = 0; i < m; ++i)
        ++cnt[static_cast<std::size_t>((record_msd_key<Field>(src[i], offset, descending) - mn) >> shift) + 1];
    for (std::size_t b = 1; b <= nb; ++b) cnt[b] += cnt[b - 1];
    std::memcpy(pos, cnt, nb * sizeof(std::size_t));
    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t b = static_cast<std::size_t>(
            (record_msd_key<Field>(src[i], offset, descending) - mn) >> shift);
        dst[pos[b]++] = src[i];
    }
    // Large sub-buckets recurse; the small ones are finished by one
    // insertion pass over the whole range (keys never cross a sub-bucket
    // boundary, so it only moves records inside their own sub-bucket).
    for (std::size_t b = 0; b < nb; ++b) {
        const std::size_t lo = cnt[b], s = cnt[b + 1] - lo;
        if (s <= 1) continue;
        if (s <= kLeaf) {
#if FYX_RMSD_RANK_LEAF
            record_msd_rank_leaf<Field>(dst + lo, src + lo, s, offset, descending);
#endif
            continue;
        }
        record_msd_rec<Field>(dst + lo, src + lo, s, offset, descending, depth + 1,
                              work + kRecordMsdFrame);
        std::memcpy(static_cast<void*>(dst + lo), static_cast<const void*>(src + lo), s * sizeof(T));
    }
#if !FYX_RMSD_RANK_LEAF
    record_msd_insertion<Field>(dst, m, offset, descending);
#endif
}

template <class Field, class T, class Comp>
inline bool record_msd_sort(T* p, std::size_t n, Comp comp, std::size_t offset,
                            bool descending, bool par = false) {
    using Key = typename RadixTraits<Field>::Key;
    // The top digit is planned from a sample (out-of-sample keys clamp to the
    // edge buckets, which only makes those buckets larger) and kept narrow:
    // a full-array scatter into hundreds of streams is several times slower
    // than into a few dozen once the destination spans more pages than the
    // TLB covers, while the second level runs on L2-resident buckets.
    constexpr std::size_t S = 256;
    Key mn = record_msd_key<Field>(p[0], offset, descending), mx = mn;
    for (std::size_t j = 0; j < S; ++j) {
        const Key k = record_msd_key<Field>(p[(j * n) / S], offset, descending);
        mn = k < mn ? k : mn;
        mx = mx < k ? k : mx;
    }
    if (mn == mx) {
        mn = record_msd_key<Field>(p[0], offset, descending); mx = mn;
        for (std::size_t i = 1; i < n; ++i) {
            const Key k = record_msd_key<Field>(p[i], offset, descending);
            mn = k < mn ? k : mn;
            mx = mx < k ? k : mx;
        }
        if (mn == mx) return true;   // every key equal under a sampled-equivalent comparator
    }
    const unsigned bits = record_msd_key_bits(static_cast<Key>(mx - mn));
    unsigned lg = 0;
    while ((std::size_t(1) << (lg + 1)) <= n) ++lg;
    unsigned B = FYX_RMSD_TOP_BITS;
    if (lg < 14) B = lg > 8 ? lg - 8 : 1;
    if (B > bits) B = bits;
    const unsigned shift = bits - B;
    const std::size_t nb = std::size_t(1) << B;
    auto bucket_of = [&](Key k) noexcept -> std::size_t {
        if (k < mn) return 0;
        const std::size_t b = static_cast<std::size_t>((k - mn) >> shift);
        return b < nb ? b : nb - 1;
    };

    ScratchLease<T> tmp_lease(n);
    if (!tmp_lease.valid()) return false;
    T* tmp = tmp_lease.get();
#if FYX_ENABLE_PARALLEL
    // Parallel form: per-chunk histograms and scatter for the top digit, then
    // the buckets (independent, L2-sized after the first sub-digit) are
    // finished and proven by whichever worker picks them up.
    if (par && n >= kParallelThreshold && parallel_available()) {
        const std::size_t chunks = adaptive_parallel_chunks(n);
        std::vector<std::size_t> local(chunks * nb, 0), start(nb + 1, 0);
        auto rmsd_count_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                std::size_t* lc = local.data() + c * nb;
                const std::size_t lo = (c * n) / chunks, hi = ((c + 1) * n) / chunks;
                for (std::size_t i = lo; i < hi; ++i)
                    ++lc[bucket_of(record_msd_key<Field>(p[i], offset, descending))];
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), rmsd_count_job);
        std::size_t sum = 0;
        for (std::size_t b = 0; b < nb; ++b) {
            start[b] = sum;
            for (std::size_t c = 0; c < chunks; ++c) {
                const std::size_t v = local[c * nb + b];
                local[c * nb + b] = sum;
                sum += v;
            }
        }
        start[nb] = sum;
        auto rmsd_scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                std::size_t* pos = local.data() + c * nb;
                const std::size_t lo = (c * n) / chunks, hi = ((c + 1) * n) / chunks;
                for (std::size_t i = lo; i < hi; ++i)
                    tmp[pos[bucket_of(record_msd_key<Field>(p[i], offset, descending))]++] = p[i];
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), rmsd_scatter_job);
        std::atomic<bool> bad{false};
        auto rmsd_bucket_job = [&](std::size_t b_lo, std::size_t b_hi) {
            thread_local std::vector<std::size_t> tl_work;
            if (tl_work.size() < kRecordMsdFrame * (kRecordMsdMaxDepth + 1))
                tl_work.resize(kRecordMsdFrame * (kRecordMsdMaxDepth + 1));
            for (std::size_t b = b_lo; b < b_hi; ++b) {
                const std::size_t lo = start[b], s = start[b + 1] - lo;
                if (s == 0) continue;
                if (bad.load(std::memory_order_relaxed)) {
                    // Declining still has to hand back a permutation of the input.
                    std::memcpy(static_cast<void*>(p + lo), static_cast<const void*>(tmp + lo), s * sizeof(T));
                    continue;
                }
                record_msd_rec<Field>(tmp + lo, p + lo, s, offset, descending, 1, tl_work.data());
                for (std::size_t i = lo + 1; i < lo + s; ++i)
                    if (comp(p[i], p[i - 1])) { bad.store(true, std::memory_order_relaxed); break; }
            }
        };
        parallel_for_index(std::size_t(0), nb, std::size_t(1), rmsd_bucket_job);
        if (bad.load()) return false;
        const T* prev = nullptr;
        for (std::size_t b = 0; b < nb; ++b) {
            const std::size_t lo = start[b], s = start[b + 1] - lo;
            if (s == 0) continue;
            if (prev && comp(p[lo], *prev)) return false;
            prev = p + lo + s - 1;
        }
        return true;
    }
#else
    (void)par;
#endif
    std::vector<std::size_t> work(kRecordMsdFrame * (kRecordMsdMaxDepth + 1) + 2 * nb + 2, 0);
    std::size_t* cnt = work.data() + kRecordMsdFrame * (kRecordMsdMaxDepth + 1);
    std::size_t* pos = cnt + nb + 1;
    for (std::size_t i = 0; i < n; ++i)
        ++cnt[bucket_of(record_msd_key<Field>(p[i], offset, descending)) + 1];
    for (std::size_t b = 1; b <= nb; ++b) cnt[b] += cnt[b - 1];
    std::memcpy(pos, cnt, nb * sizeof(std::size_t));
    for (std::size_t i = 0; i < n; ++i)
        tmp[pos[bucket_of(record_msd_key<Field>(p[i], offset, descending))]++] = p[i];
    const T* prev = nullptr;
    for (std::size_t b = 0; b < nb; ++b) {
        const std::size_t lo = cnt[b], s = cnt[b + 1] - lo;
        if (s == 0) continue;
        record_msd_rec<Field>(tmp + lo, p + lo, s, offset, descending, 1, work.data());
        bool ok = !(prev && comp(p[lo], *prev));
        for (std::size_t i = lo + 1; ok && i < lo + s; ++i) ok = !comp(p[i], p[i - 1]);
        if (!ok) {
            // The comparator is not the field order.  The later buckets still
            // live in scratch: put them back so the caller gets a permutation
            // of its input (it falls back to a comparison sort).
            const std::size_t rest = cnt[b + 1];
            std::memcpy(static_cast<void*>(p + rest), static_cast<const void*>(tmp + rest), (n - rest) * sizeof(T));
            return false;
        }
        prev = p + lo + s - 1;
    }
    return true;
}

/// 512 strided key samples hold >= 32 repeats (distinct-key data: ~0).
template <class Field, class T>
inline bool trivial_field_dup_heavy(const T* p, std::size_t n, std::size_t offset) {
    using Key = typename RadixTraits<Field>::Key;
    if (!use_avx512() || n < 4096) return false;
    constexpr std::size_t S = 512;
    std::array<Key, S> k;
    for (std::size_t j = 0; j < S; ++j) k[j] = load_trivial_field_key<Field>(p[(j * n) / S], offset);
    std::sort(k.begin(), k.end());
    std::size_t rep = 0;
    for (std::size_t j = 1; j < S; ++j) rep += k[j] == k[j - 1] ? 1u : 0u;
    return rep >= 32;
}

template <class Field, class T, class Comp>
inline bool try_trivial_field_radix_sort(T* p, std::size_t n, Comp comp,
                                         std::size_t offset, bool par = false) {
    if (n < kSampleThreshold) return false;
    if constexpr (!std::is_trivially_copyable<T>::value) {
        (void)p; (void)n; (void)comp; (void)offset;
        return false;
    } else {
        using Key = typename RadixTraits<Field>::Key;
        constexpr unsigned Passes = (sizeof(Key) * CHAR_BIT + kRadixBits - 1) / kRadixBits;

        bool descending = false;
        if (!trivial_field_candidate_order<Field>(p, n, comp, offset, descending)) return false;
        // Duplicate-heavy 16-byte records (sqrt-n / zipf-like key counts,
        // past the counting paths' class limit): the in-place record
        // quicksort finishes equal-key ranges without leaves and beats the
        // MSD passes; distinct-key data stays on the radix path.
        if (sizeof(T) == 16 && sizeof(Field) == 8 && trivial_field_dup_heavy<Field>(p, n, offset)) {
            const int r = trivial_field_kv16_vqsort<Field>(p, n, comp, offset, descending,
                                                           par && n >= kKv16EarlyParallelMinN);
            if (r == 1) return true;
        }
        if (FYX_RECORD_MSD) return record_msd_sort<Field>(p, n, comp, offset, descending, par);

        RadixHistogram<Passes> hist;
        hist.clear();
        for (std::size_t i = 0; i < n; ++i) {
            Key k = load_trivial_field_key<Field>(p[i], offset);
            if (descending) k = static_cast<Key>(~k);
            for (unsigned pass = 0; pass < Passes; ++pass)
                ++hist.count[pass][radix_digit(k, pass)];
        }

        const RadixPlan<Passes> plan = plan_radix<Passes>(hist, n);
        if (plan.count == 0) return true;

        ScratchLease<T> tmp_lease(n);
        if (!tmp_lease.valid()) return false;
        T* tmp = tmp_lease.get();

        T* src = p;
        T* dst = tmp;
        if ((plan.count & 1u) != 0) {
            std::memcpy(tmp, p, n * sizeof(T));
            src = tmp;
            dst = p;
        }

        std::size_t pos[kRadixBuckets];
        for (unsigned pi = 0; pi < plan.count; ++pi) {
            const unsigned pass = plan.active[pi];
            std::size_t sum = 0;
            for (unsigned d = 0; d < kRadixBuckets; ++d) {
                pos[d] = sum;
                sum += static_cast<std::size_t>(hist.count[pass][d]);
            }
            const unsigned shift = pass * kRadixBits;
            for (std::size_t i = 0; i < n; ++i) {
                Key k = load_trivial_field_key<Field>(src[i], offset);
                if (descending) k = static_cast<Key>(~k);
                const unsigned d = static_cast<unsigned>((k >> shift) & Key(kRadixMask));
                dst[pos[d]++] = src[i];
            }
            T* t = src; src = dst; dst = t;
        }

        return std::is_sorted(p, p + n, comp);
    }
}

template <class T, class Comp>
inline bool try_trivial_prefix_key_radix_sort(T* p, std::size_t n, Comp comp, bool par = false) {
    if constexpr (!std::is_trivially_copyable<T>::value || std::is_arithmetic<T>::value ||
                  std::is_same<T, std::string>::value) {
        (void)p; (void)n; (void)comp; (void)par;
        return false;
    } else {
        // 8-byte fields first: the upper half of a 64-bit key is itself a
        // 32-bit field that orders a duplicate-free sample exactly like the
        // key, so probing 4-byte fields first used to pick it, radix-sort on
        // half the key and then fail the final check on the first tie.
        constexpr std::size_t max_probe = sizeof(T) < 32 ? sizeof(T) : 32;
        for (std::size_t off = 0; off + 8 <= max_probe; off += 8) {
            if (try_trivial_field_radix_sort<std::int64_t>(p, n, comp, off, par)) return true;
            if (try_trivial_field_radix_sort<std::uint64_t>(p, n, comp, off, par)) return true;
        }
        for (std::size_t off = 0; off + 4 <= max_probe; off += 4) {
            if (try_trivial_field_radix_sort<std::int32_t>(p, n, comp, off, par)) return true;
            if (try_trivial_field_radix_sort<std::uint32_t>(p, n, comp, off, par)) return true;
        }
        return false;
    }
}

template <class T, class Comp>
inline bool try_trivial_prefix_key_count_sort(T* p, std::size_t n, Comp comp, bool par = false) {
    if constexpr (!std::is_trivially_copyable<T>::value || std::is_arithmetic<T>::value ||
                  std::is_same<T, std::string>::value) {
        (void)p; (void)n; (void)comp; (void)par;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        constexpr std::size_t max_probe = sizeof(T) < 32 ? sizeof(T) : 32;
        // 8-byte fields first, as in the radix probe (a 64-bit key's upper
        // half samples like a 32-bit key).
        for (std::size_t off = 0; off + 8 <= max_probe; off += 8) {
            if (try_trivial_field_count_sort<std::int64_t>(p, n, comp, off, par)) return true;
            if (try_trivial_field_count_sort<std::uint64_t>(p, n, comp, off, par)) return true;
        }
        for (std::size_t off = 0; off + 4 <= max_probe; off += 4) {
            if (try_trivial_field_count_sort<std::int32_t>(p, n, comp, off, par)) return true;
            if (try_trivial_field_count_sort<std::uint32_t>(p, n, comp, off, par)) return true;
        }
        return false;
    }
}

template <class T>
struct LowCardRep {
    T             value;
    unsigned char id;
};

template <class T, class Comp>
inline typename std::vector<LowCardRep<T>>::iterator
low_card_lower_bound(std::vector<LowCardRep<T>>& reps, const T& x, Comp comp) {
    return std::lower_bound(reps.begin(), reps.end(), x,
        [&](const LowCardRep<T>& r, const T& v) { return comp(r.value, v); });
}

template <class It, class Comp>
inline bool low_cardinality_probe_ok(It first, std::size_t n, Comp comp) {
    using T = iter_value_t<It>;
    const std::size_t s = std::min<std::size_t>(n, kCountingProbeLimit);
    if (s == 0) return false;

    std::vector<LowCardRep<T>> reps;
    reps.reserve(kCountingClassLimit + 1);
    for (std::size_t j = 0; j < s; ++j) {
        const std::size_t idx = (j * n) / s;
        const T& x = first[idx];
        auto it = low_card_lower_bound(reps, x, comp);
        if (it != reps.end() && !comp(x, it->value)) continue;  // equivalent
        if (reps.size() >= kCountingClassLimit) return false;
        const unsigned char id = static_cast<unsigned char>(reps.size());
        reps.insert(it, LowCardRep<T>{x, id});
    }
    return true;
}

template <class It, class Comp>
inline bool try_low_cardinality_count_sort(It first, It last, Comp comp) {
    using T = iter_value_t<It>;
    const std::size_t n = static_cast<std::size_t>(last - first);

    if constexpr (!std::is_copy_constructible<T>::value ||
                  !std::is_move_constructible<T>::value ||
                  !std::is_move_assignable<T>::value ||
                  std::is_floating_point<T>::value) {
        (void)first; (void)last; (void)comp; (void)n;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        if (!low_cardinality_probe_ok(first, n, comp)) return false;

        ScratchLease<unsigned char> ids_lease(n);
        if (!ids_lease.valid()) return false;
        unsigned char* ids = ids_lease.get();

        std::vector<LowCardRep<T>> reps;
        reps.reserve(kCountingClassLimit);
        std::array<std::size_t, kCountingClassLimit> counts{};

        for (std::size_t i = 0; i < n; ++i) {
            const T& x = first[i];
            auto it = low_card_lower_bound(reps, x, comp);
            unsigned char id;
            if (it != reps.end() && !comp(x, it->value)) {
                id = it->id;
            } else {
                if (reps.size() >= kCountingClassLimit) return false;
                id = static_cast<unsigned char>(reps.size());
                it = reps.insert(it, LowCardRep<T>{x, id});
            }
            ids[i] = id;
            ++counts[id];
        }

        const std::size_t d = reps.size();
        if (d <= 1) return true;

        std::array<unsigned char, kCountingClassLimit> rank_by_id{};
        for (std::size_t rank = 0; rank < d; ++rank)
            rank_by_id[reps[rank].id] = static_cast<unsigned char>(rank);

        if constexpr (std::is_pointer<It>::value && std::is_trivially_copyable<T>::value) {
            std::array<std::size_t, kCountingClassLimit> pos{};
            std::size_t sum = 0;
            for (std::size_t rank = 0; rank < d; ++rank) {
                pos[rank] = sum;
                sum += counts[reps[rank].id];
            }
            ScratchLease<T> out_lease(n);
            if (!out_lease.valid()) return false;
            T* out = out_lease.get();
            for (std::size_t i = 0; i < n; ++i) {
                const unsigned char rank = rank_by_id[ids[i]];
                out[pos[rank]++] = first[i];
            }
            for (std::size_t i = 0; i < n; ++i) first[i] = out[i];
        } else {
            std::vector<std::vector<T>> buckets(d);
            for (std::size_t rank = 0; rank < d; ++rank)
                buckets[rank].reserve(counts[reps[rank].id]);
            for (std::size_t i = 0; i < n; ++i) {
                const unsigned char rank = rank_by_id[ids[i]];
                buckets[rank].push_back(std::move(first[i]));
            }
            It out = first;
            for (std::size_t rank = 0; rank < d; ++rank) {
                for (T& x : buckets[rank]) {
                    *out = std::move(x);
                    ++out;
                }
            }
        }
        return true;
    }
}


// ---------------------------------------------------------------------------
// MSD radix sort for default-ordered std::string.
// Random strings are the one case where comparison sample sort still pays too
// many full lexicographic comparisons.  Byte-wise MSD radix reads only the
// distinguishing prefix (usually 2-4 bytes for random text) and then finishes
// tiny buckets with std::sort.  It is exact for std::string's lexicographic
// unsigned-byte order used by char_traits::compare on mainstream libstdc++.
// ---------------------------------------------------------------------------
inline unsigned string_msd_bucket(const std::string& s, std::size_t depth) noexcept {
    return depth < s.size()
        ? static_cast<unsigned>(static_cast<unsigned char>(s[depth])) + 1u
        : 0u;
}

inline void string_msd_sort_rec(std::string* p, std::string* tmp,
                                std::size_t n, std::size_t depth) {
    constexpr std::size_t kSmall = 96;
    if (n <= kSmall) { std::sort(p, p + n); return; }

    std::array<std::size_t, 258> off{};
    for (std::size_t i = 0; i < n; ++i) ++off[string_msd_bucket(p[i], depth) + 1u];

    unsigned nonzero = 0, only = 0;
    for (unsigned b = 0; b < 257; ++b) {
        if (off[b + 1] != 0) { ++nonzero; only = b; }
    }
    if (nonzero <= 1) {
        if (only != 0) string_msd_sort_rec(p, tmp, n, depth + 1);
        return;
    }

    for (unsigned b = 1; b <= 257; ++b) off[b] += off[b - 1];
    std::array<std::size_t, 257> pos{};
    for (unsigned b = 0; b < 257; ++b) pos[b] = off[b];

    for (std::size_t i = 0; i < n; ++i) {
        const unsigned b = string_msd_bucket(p[i], depth);
        tmp[pos[b]++] = std::move(p[i]);
    }
    for (std::size_t i = 0; i < n; ++i) p[i] = std::move(tmp[i]);

    for (unsigned b = 1; b < 257; ++b) {
        const std::size_t lo = off[b], hi = off[b + 1];
        if (hi - lo > 1) string_msd_sort_rec(p + lo, tmp + lo, hi - lo, depth + 1);
    }
}

// ---------------------------------------------------------------------------
// Prefix-key string sort.  The byte-wise MSD below moves every std::string
// object twice per level (about three levels for random text); here each
// string contributes one 8-byte big-endian chunk of its bytes plus its index,
// the 16-byte (index, key) records go through the record MSD radix, equal
// chunks are refined with the next 8 bytes (strings that end inside a chunk
// go first, ordered by length -- they are prefixes of the longer ones), and
// the strings themselves move exactly twice at the end.  Exact for
// std::string's unsigned-byte lexicographic order.
// ---------------------------------------------------------------------------
#ifndef FYX_STRING_PREFIX_SORT
#  define FYX_STRING_PREFIX_SORT 1
#endif
struct StrKeyIdx {
    std::uint32_t idx;
    std::uint32_t len;     // the string's length (inputs with longer strings decline)
    std::uint64_t key;
};
struct StrKeyIdxLess {
    bool operator()(const StrKeyIdx& a, const StrKeyIdx& b) const noexcept { return a.key < b.key; }
};

inline std::uint64_t string_chunk_be(const std::string& s, std::size_t d) noexcept {
    const std::size_t len = s.size();
    if (d >= len) return 0;
    const unsigned char* q = reinterpret_cast<const unsigned char*>(s.data()) + d;
    const std::size_t m = len - d;
    std::uint64_t v = 0;
    if (m >= 8) {
        for (std::size_t i = 0; i < 8; ++i) v = (v << 8) | q[i];
        return v;
    }
    for (std::size_t i = 0; i < m; ++i) v |= static_cast<std::uint64_t>(q[i]) << (56 - 8 * i);
    return v;
}

inline void string_prefix_sort_keys(StrKeyIdx* a, std::size_t m) {
    if (m <= 48) {
        for (std::size_t i = 1; i < m; ++i) {
            const StrKeyIdx x = a[i];
            std::size_t j = i;
            while (j > 0 && x.key < a[j - 1].key) { a[j] = a[j - 1]; --j; }
            a[j] = x;
        }
        return;
    }
    if (!record_msd_sort<std::uint64_t>(a, m, StrKeyIdxLess{}, 8, false))
        std::sort(a, a + m, StrKeyIdxLess{});
}

// a[0, m) agree on bytes [0, depth) (zero-padded).  Orders them completely.
inline void string_prefix_refine(StrKeyIdx* a, std::size_t m, const std::string* p,
                                 std::size_t depth) {
    if (depth > 512) {
        std::sort(a, a + m, [p](const StrKeyIdx& x, const StrKeyIdx& y) {
            return p[x.idx] < p[y.idx];
        });
        return;
    }
    std::size_t e = 0;
    for (std::size_t i = 0; i < m; ++i)
        if (a[i].len <= depth) std::swap(a[i], a[e++]);
    if (e > 1) {
        std::sort(a, a + e, [](const StrKeyIdx& x, const StrKeyIdx& y) { return x.len < y.len; });
    }
    StrKeyIdx* b = a + e;
    const std::size_t k = m - e;
    if (k <= 1) return;
    // Duplicate-heavy text: a group is usually one string repeated.  One
    // memcmp per member settles that, instead of a cache-missing chunk load
    // per member per 8 bytes until every copy has ended.
    {
        const std::uint32_t l0 = b[0].len;
        const char* f0 = p[b[0].idx].data();
        bool same = true;
        for (std::size_t i = 1; i < k && same; ++i)
            same = b[i].len == l0 &&
                   std::memcmp(p[b[i].idx].data() + depth, f0 + depth, l0 - depth) == 0;
        if (same) return;
    }
    for (std::size_t i = 0; i < k; ++i) b[i].key = string_chunk_be(p[b[i].idx], depth);
    string_prefix_sort_keys(b, k);
    for (std::size_t lo = 0; lo < k;) {
        std::size_t hi = lo + 1;
        while (hi < k && b[hi].key == b[lo].key) ++hi;
        if (hi - lo > 1) string_prefix_refine(b + lo, hi - lo, p, depth + 8);
        lo = hi;
    }
}

inline bool string_prefix_sort(std::string* p, std::size_t n, bool descending) {
    ScratchLease<StrKeyIdx> lease(n);
    if (!lease.valid()) return false;
    StrKeyIdx* a = lease.get();
    for (std::size_t i = 0; i < n; ++i) {
        if (i + 16 < n) prefetch_read<3>(p[i + 16].data());
        const std::size_t len = p[i].size();
        if (len >= 0xffffffffu) return false;      // nothing moved yet
        a[i].idx = static_cast<std::uint32_t>(i);
        a[i].len = static_cast<std::uint32_t>(len);
        a[i].key = string_chunk_be(p[i], 0);
    }
    string_prefix_sort_keys(a, n);
    for (std::size_t lo = 0; lo < n;) {
        std::size_t hi = lo + 1;
        while (hi < n && a[hi].key == a[lo].key) ++hi;
        if (hi - lo > 1) string_prefix_refine(a + lo, hi - lo, p, 8);
        lo = hi;
    }
    std::vector<std::string> tmp;
    tmp.reserve(n);
    // The gather is a dependent-free random read per string: prefetch it a
    // few dozen iterations ahead so the misses overlap.
    constexpr std::size_t kAhead = 24;
    if (!descending) {
        for (std::size_t k = 0; k < n; ++k) {
            if (k + kAhead < n) prefetch_read<3>(p + a[k + kAhead].idx);
            tmp.emplace_back(std::move(p[a[k].idx]));
        }
    } else {
        for (std::size_t k = n; k-- > 0;) {
            if (k >= kAhead) prefetch_read<3>(p + a[k - kAhead].idx);
            tmp.emplace_back(std::move(p[a[k].idx]));
        }
    }
    for (std::size_t k = 0; k < n; ++k) p[k] = std::move(tmp[k]);
    return true;
}

inline bool string_msd_sort_default(std::string* p, std::size_t n, bool descending) {
#if FYX_STRING_PREFIX_SORT
    if (n >= 4096 && n < (std::size_t(1) << 32) && string_prefix_sort(p, n, descending)) return true;
#endif
    if (n < 4096) return false;
    std::vector<std::string> tmp(n);
    string_msd_sort_rec(p, tmp.data(), n, 0);
    if (descending) std::reverse(p, p + n);
    return true;
}

template <class T, class Comp>
inline bool try_string_msd_sort(T* p, std::size_t n, Comp comp, bool descending) {
    if constexpr (!std::is_same<T, std::string>::value) {
        (void)p; (void)n; (void)comp; (void)descending;
        return false;
    } else {
        if (!(is_ascending_v<Comp, T> || is_descending_v<Comp, T>)) return false;
        return string_msd_sort_default(p, n, descending);
    }
}


#if FYX_ENABLE_PARALLEL

inline std::size_t adaptive_parallel_chunks(std::size_t n) {
    ThreadPool& pool = global_pool();
    std::size_t chunks = (n + kParallelThreshold - 1) / kParallelThreshold;
    const std::size_t max_chunks = std::max<std::size_t>(2, static_cast<std::size_t>(pool.nworkers()) * 8);
    if (chunks < 2) chunks = 2;
    if (chunks > max_chunks) chunks = max_chunks;
    return chunks;
}

template <class T>
inline std::size_t adaptive_parallel_radix_chunks(std::size_t n) {
    ThreadPool& pool = global_pool();
    constexpr bool wide_key = radix_supported_v<T> && (sizeof(typename RadixTraits<T>::Key) >= 8);
    constexpr bool int64_value_key = wide_key && std::is_integral<T>::value && !std::is_same<T, bool>::value;
    constexpr bool heavier_digit = wide_key || std::is_floating_point<T>::value;
    const std::size_t per_worker = (int64_value_key && pool.nworkers() >= 3)
        ? std::size_t(3)
        : (heavier_digit ? std::size_t(4) : std::size_t(8));
    std::size_t chunks = (n + kParallelThreshold - 1) / kParallelThreshold;
    const std::size_t max_chunks = std::max<std::size_t>(2, static_cast<std::size_t>(pool.nworkers()) * per_worker);
    if (chunks < 2) chunks = 2;
    if (chunks > max_chunks) chunks = max_chunks;
    return chunks;
}

template <unsigned Passes>
struct RadixLocalHistogram {
    // Parallel radix chunks are kept far below 4G elements; 32-bit local
    // counters halve the per-chunk hot histogram/recount footprint, while the
    // global folded histogram remains 64-bit for correctness on huge arrays.
    std::uint32_t count[Passes][kRadixBuckets];
    void clear() noexcept { std::memset(count, 0, sizeof(count)); }
};

template <class T>
inline void radix_scatter_value_pass(const T* FYX_RESTRICT src, std::size_t n,
                                     T* FYX_RESTRICT dst, unsigned shift,
                                     std::size_t* FYX_RESTRICT offset,
                                     RadixScatterScratch<T>& sc,
                                     bool can_stream) noexcept {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr std::size_t kPerLine = WcbTraits<T>::kPerLine;

    T* FYX_RESTRICT line = sc.line;
    T* FYX_RESTRICT head = sc.head;

    std::size_t remaining_heads = 0;
    for (unsigned b = 0; b < kRadixBuckets; ++b) {
        sc.fill[b] = 0;
        sc.hn[b]   = 0;
        sc.base[b] = offset[b];
        if (can_stream) {
            const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(dst + offset[b]);
            const std::size_t misalign = (kCacheLine - (addr & (kCacheLine - 1))) & (kCacheLine - 1);
            sc.need[b] = static_cast<std::uint32_t>(misalign / sizeof(T));
        } else {
            sc.need[b] = 0;
        }
        remaining_heads += sc.need[b];
    }

    auto push_line = [&](T v, unsigned b) {
        T* L = line + static_cast<std::size_t>(b) * kPerLine;
        const std::uint32_t f = sc.fill[b];
        L[f] = v;

        if (FYX_UNLIKELY(f + 1 == kPerLine)) {
            T* out = dst + offset[b] + sc.need[b];
            if (can_stream) stream_cache_line(out, L);
            else std::memcpy(out, L, kCacheLine);
            offset[b] += kPerLine;
            sc.fill[b] = 0;
        } else {
            sc.fill[b] = f + 1;
        }
    };

    std::size_t i = 0;
    for (; i < n && remaining_heads != 0; ++i) {
        prefetch_stream(src, i, n);
        const T v = src[i];
        const Key k = RT::encode(v);
        const unsigned b = static_cast<unsigned>((k >> shift) & Key(kRadixMask));

        if (FYX_UNLIKELY(sc.hn[b] < sc.need[b])) {
            head[static_cast<std::size_t>(b) * kPerLine + sc.hn[b]] = v;
            ++sc.hn[b];
            --remaining_heads;
            continue;
        }

        push_line(v, b);
    }

    for (; i < n; ++i) {
        prefetch_stream(src, i, n);
        const T v = src[i];
        const Key k = RT::encode(v);
        const unsigned b = static_cast<unsigned>((k >> shift) & Key(kRadixMask));
        push_line(v, b);
    }

    for (unsigned b = 0; b < kRadixBuckets; ++b) {
        if (sc.hn[b])
            std::memcpy(dst + sc.base[b],
                        head + static_cast<std::size_t>(b) * kPerLine,
                        static_cast<std::size_t>(sc.hn[b]) * sizeof(T));
        if (sc.fill[b])
            std::memcpy(dst + offset[b] + sc.need[b],
                        line + static_cast<std::size_t>(b) * kPerLine,
                        static_cast<std::size_t>(sc.fill[b]) * sizeof(T));
        offset[b] += sc.fill[b] + sc.hn[b];
    }
}

template <class T>
inline void radix_scatter_decode_pass(const typename RadixTraits<T>::Key* FYX_RESTRICT src,
                                      std::size_t n,
                                      T* FYX_RESTRICT dst, unsigned shift,
                                      std::size_t* FYX_RESTRICT offset,
                                      RadixScatterScratch<T>& sc,
                                      bool can_stream) noexcept {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr std::size_t kPerLine = WcbTraits<T>::kPerLine;

    T* FYX_RESTRICT line = sc.line;
    T* FYX_RESTRICT head = sc.head;

    std::size_t remaining_heads = 0;
    for (unsigned b = 0; b < kRadixBuckets; ++b) {
        sc.fill[b] = 0;
        sc.hn[b]   = 0;
        sc.base[b] = offset[b];
        if (can_stream) {
            const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(dst + offset[b]);
            const std::size_t misalign = (kCacheLine - (addr & (kCacheLine - 1))) & (kCacheLine - 1);
            sc.need[b] = static_cast<std::uint32_t>(misalign / sizeof(T));
        } else {
            sc.need[b] = 0;
        }
        remaining_heads += sc.need[b];
    }

    auto push_line = [&](T v, unsigned b) {
        T* L = line + static_cast<std::size_t>(b) * kPerLine;
        const std::uint32_t f = sc.fill[b];
        L[f] = v;

        if (FYX_UNLIKELY(f + 1 == kPerLine)) {
            T* out = dst + offset[b] + sc.need[b];
            if (can_stream) stream_cache_line(out, L);
            else std::memcpy(out, L, kCacheLine);
            offset[b] += kPerLine;
            sc.fill[b] = 0;
        } else {
            sc.fill[b] = f + 1;
        }
    };

    std::size_t i = 0;
    for (; i < n && remaining_heads != 0; ++i) {
        prefetch_stream(src, i, n);
        const Key k = src[i];
        const unsigned b = static_cast<unsigned>((k >> shift) & Key(kRadixMask));
        const T v = RT::decode(k);

        if (FYX_UNLIKELY(sc.hn[b] < sc.need[b])) {
            head[static_cast<std::size_t>(b) * kPerLine + sc.hn[b]] = v;
            ++sc.hn[b];
            --remaining_heads;
            continue;
        }

        push_line(v, b);
    }

    for (; i < n; ++i) {
        prefetch_stream(src, i, n);
        const Key k = src[i];
        const unsigned b = static_cast<unsigned>((k >> shift) & Key(kRadixMask));
        const T v = RT::decode(k);
        push_line(v, b);
    }

    for (unsigned b = 0; b < kRadixBuckets; ++b) {
        if (sc.hn[b])
            std::memcpy(dst + sc.base[b],
                        head + static_cast<std::size_t>(b) * kPerLine,
                        static_cast<std::size_t>(sc.hn[b]) * sizeof(T));
        if (sc.fill[b])
            std::memcpy(dst + offset[b] + sc.need[b],
                        line + static_cast<std::size_t>(b) * kPerLine,
                        static_cast<std::size_t>(sc.fill[b]) * sizeof(T));
        offset[b] += sc.fill[b] + sc.hn[b];
    }
}

template <class Key>
inline void radix_count_key_pass_banked(const Key* FYX_RESTRICT src,
                                        std::size_t n, unsigned shift,
                                        std::uint32_t* FYX_RESTRICT out) noexcept {
    std::uint32_t bank[4][kRadixBuckets];
    std::memset(bank, 0, sizeof(bank));
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        prefetch_stream(src, i, n);
        ++bank[0][static_cast<unsigned>((src[i + 0] >> shift) & Key(kRadixMask))];
        ++bank[1][static_cast<unsigned>((src[i + 1] >> shift) & Key(kRadixMask))];
        ++bank[2][static_cast<unsigned>((src[i + 2] >> shift) & Key(kRadixMask))];
        ++bank[3][static_cast<unsigned>((src[i + 3] >> shift) & Key(kRadixMask))];
    }
    for (; i < n; ++i)
        ++bank[0][static_cast<unsigned>((src[i] >> shift) & Key(kRadixMask))];
    for (unsigned d = 0; d < kRadixBuckets; ++d)
        out[d] = bank[0][d] + bank[1][d] + bank[2][d] + bank[3][d];
}

template <class T>
inline void radix_count_value_pass_banked(const T* FYX_RESTRICT src,
                                          std::size_t n, unsigned shift,
                                          std::uint32_t* FYX_RESTRICT out) noexcept {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    std::uint32_t bank[4][kRadixBuckets];
    std::memset(bank, 0, sizeof(bank));
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        prefetch_stream(src, i, n);
        const Key k0 = RT::encode(src[i + 0]);
        const Key k1 = RT::encode(src[i + 1]);
        const Key k2 = RT::encode(src[i + 2]);
        const Key k3 = RT::encode(src[i + 3]);
        ++bank[0][static_cast<unsigned>((k0 >> shift) & Key(kRadixMask))];
        ++bank[1][static_cast<unsigned>((k1 >> shift) & Key(kRadixMask))];
        ++bank[2][static_cast<unsigned>((k2 >> shift) & Key(kRadixMask))];
        ++bank[3][static_cast<unsigned>((k3 >> shift) & Key(kRadixMask))];
    }
    for (; i < n; ++i) {
        const Key k = RT::encode(src[i]);
        ++bank[0][static_cast<unsigned>((k >> shift) & Key(kRadixMask))];
    }
    for (unsigned d = 0; d < kRadixBuckets; ++d)
        out[d] = bank[0][d] + bank[1][d] + bank[2][d] + bank[3][d];
}


template <class T, unsigned PrefixBits>
inline bool radix_high_prefix_probe(const T* p, std::size_t n) noexcept {
    if constexpr (!radix_supported_v<T> || PrefixBits == 0 || PrefixBits > 64) {
        (void)p; (void)n;
        return false;
    } else {
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        if constexpr (sizeof(Key) != 8 && sizeof(Key) != 4) {
            (void)p; (void)n;
            return false;
        } else {
            constexpr unsigned Shift = unsigned(sizeof(Key) * 8) - PrefixBits;
            constexpr std::size_t Cap = 2048;
            constexpr std::size_t Mask = Cap - 1;
            std::array<Key, Cap> keys{};
            std::array<unsigned char, Cap> used{};
            std::size_t distinct = 0;
            const std::size_t sample_n = std::min<std::size_t>(n, kProfileSampleLimit);
            if (sample_n < 64) return false;
            for (std::size_t j = 0; j < sample_n; ++j) {
                const std::size_t idx = (j * (n - 1)) / (sample_n - 1);
                const Key prefix = RT::encode(p[idx]) >> Shift;
                std::size_t h = low_card_hash_key(prefix) & Mask;
                for (;;) {
                    if (!used[h]) {
                        used[h] = 1;
                        keys[h] = prefix;
                        ++distinct;
                        break;
                    }
                    if (keys[h] == prefix) break;
                    h = (h + 1) & Mask;
                }
            }
            return distinct * 16 >= sample_n * 15;
        }
    }
}

template <class T, unsigned PrefixBits>
inline void radix_sort_high_prefix_value_ties(T* p, std::size_t n) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr unsigned Shift = unsigned(sizeof(Key) * 8) - PrefixBits;
    if (n < 2) return;
    std::size_t group_lo = 0;
    Key prev_prefix = RT::encode(p[0]) >> Shift;
    auto repair_group = [&](std::size_t lo, std::size_t hi) {
        const std::size_t sz = hi - lo;
        if (sz <= 1) return;
        if (sz <= kNetworkMax) {
            small_sort_numeric(p + lo, sz);
        } else {
            std::sort(p + lo, p + hi, [](const T& a, const T& b) {
                return RT::encode(a) < RT::encode(b);
            });
        }
    };
    for (std::size_t i = 1; i < n; ++i) {
        const Key cur_prefix = RT::encode(p[i]) >> Shift;
        if (cur_prefix != prev_prefix) {
            repair_group(group_lo, i);
            group_lo = i;
            prev_prefix = cur_prefix;
        }
    }
    repair_group(group_lo, n);
}

template <class Key, unsigned PrefixBits>
inline void radix_sort_high_prefix_key_ties(Key* p, std::size_t n) {
    constexpr unsigned Shift = unsigned(sizeof(Key) * 8) - PrefixBits;
    std::size_t i = 0;
    while (i < n) {
        const Key prefix = p[i] >> Shift;
        std::size_t j = i + 1;
        while (j < n && (p[j] >> Shift) == prefix) ++j;
        const std::size_t sz = j - i;
        if (sz > 1) {
            if (sz <= kNetworkMax) small_sort_numeric(p + i, sz);
            else std::sort(p + i, p + j);
        }
        i = j;
    }
}

template <class T, unsigned PrefixBits>
inline bool try_parallel_radix_high_prefix_value_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T> || PrefixBits == 0 || PrefixBits > 64 || (PrefixBits % 8) != 0) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        if constexpr (sizeof(Key) != 8) {
            (void)p; (void)n; (void)descending;
            return false;
        } else {
            constexpr unsigned PrefixBytes = PrefixBits / 8;
            constexpr unsigned FirstShift = unsigned(sizeof(Key) * 8) - PrefixBits;
            if (n < (std::size_t(1) << 19) || !parallel_available()) return false;
            if (!radix_high_prefix_probe<T, PrefixBits>(p, n)) return false;

            ScratchLease<T> tmp_lease(n);
            if (!tmp_lease.valid()) return false;
            T* tmp = tmp_lease.get();

            const std::size_t chunks = adaptive_parallel_radix_chunks<T>(n);
            if ((n + chunks - 1) / chunks > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return false;
            std::vector<std::array<std::uint32_t, kRadixBuckets>> local(chunks);
            std::vector<std::array<std::size_t, kRadixBuckets>> base(chunks);
            T* src = p;
            T* dst = tmp;
            const bool can_stream = have_nt_stores();

            for (unsigned pi = 0; pi < PrefixBytes; ++pi) {
                const unsigned shift = FirstShift + pi * kRadixBits;
                auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
                    for (std::size_t c = c_lo; c < c_hi; ++c) {
                        const std::size_t lo = (c * n) / chunks;
                        const std::size_t hi = ((c + 1) * n) / chunks;
                        radix_count_value_pass_banked<T>(src + lo, hi - lo, shift, local[c].data());
                    }
                };
                parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);

                std::size_t run = 0;
                for (unsigned d = 0; d < kRadixBuckets; ++d) {
                    for (std::size_t c = 0; c < chunks; ++c) {
                        base[c][d] = run;
                        run += local[c][d];
                    }
                }

                auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
                    constexpr std::size_t kPerLine = WcbTraits<T>::kPerLine;
                    ScratchLease<T> wcb_lease(2 * kRadixBuckets * kPerLine + kPerLine);
                    RadixScatterScratch<T> sc;
                    bool local_stream = false;
                    if (wcb_lease.valid()) {
                        const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(wcb_lease.get());
                        const std::uintptr_t mis = (kCacheLine - (addr & (kCacheLine - 1))) & (kCacheLine - 1);
                        sc.line = reinterpret_cast<T*>(addr + mis);
                        sc.head = sc.line + kRadixBuckets * kPerLine;
                        local_stream = can_stream;
                    }
                    for (std::size_t c = c_lo; c < c_hi; ++c) {
                        const std::size_t lo = (c * n) / chunks;
                        const std::size_t hi = ((c + 1) * n) / chunks;
                        auto pos = base[c];
                        if (wcb_lease.valid()) {
                            radix_scatter_value_pass<T>(src + lo, hi - lo, dst, shift, pos.data(), sc, local_stream);
                        } else {
                            for (std::size_t i = lo; i < hi; ++i) {
                                const T v = src[i];
                                const Key k = RT::encode(v);
                                const unsigned d = static_cast<unsigned>((k >> shift) & Key(kRadixMask));
                                dst[pos[d]++] = v;
                            }
                        }
                    }
                };
                parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);
                T* t = src; src = dst; dst = t;
            }

            if (src != p) {
                auto copy_job = [&](std::size_t lo, std::size_t hi) {
                    for (std::size_t i = lo; i < hi; ++i) p[i] = src[i];
                };
                parallel_for_index(std::size_t(0), n, kParallelThreshold, copy_job);
            }
            radix_sort_high_prefix_value_ties<T, PrefixBits>(p, n);
            if (descending) std::reverse(p, p + n);
            return true;
        }
    }
}

template <class Key, unsigned Bits>
inline void radix_count_key_pass_banked_wide(const Key* FYX_RESTRICT src,
                                             std::size_t n, unsigned shift,
                                             std::uint32_t* FYX_RESTRICT out) noexcept {
    static_assert(Bits >= 8 && Bits <= 13, "wide radix count is tuned for 8..13 bits");
    constexpr std::size_t Buckets = std::size_t(1) << Bits;
    constexpr Key Mask = Key(Buckets - 1);
    std::array<std::uint32_t, 4 * Buckets> bank{};
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        prefetch_stream(src, i, n);
        const Key k0 = src[i + 0];
        const Key k1 = src[i + 1];
        const Key k2 = src[i + 2];
        const Key k3 = src[i + 3];
        ++bank[0 * Buckets + static_cast<std::size_t>((k0 >> shift) & Mask)];
        ++bank[1 * Buckets + static_cast<std::size_t>((k1 >> shift) & Mask)];
        ++bank[2 * Buckets + static_cast<std::size_t>((k2 >> shift) & Mask)];
        ++bank[3 * Buckets + static_cast<std::size_t>((k3 >> shift) & Mask)];
    }
    for (; i < n; ++i)
        ++bank[static_cast<std::size_t>((src[i] >> shift) & Mask)];
    for (std::size_t d = 0; d < Buckets; ++d)
        out[d] = bank[d] + bank[Buckets + d] + bank[2 * Buckets + d] + bank[3 * Buckets + d];
}

template <class T, unsigned Bits>
inline void radix_count_value_pass_banked_wide(const T* FYX_RESTRICT src,
                                               std::size_t n, unsigned shift,
                                               std::uint32_t* FYX_RESTRICT out) noexcept {
    static_assert(Bits >= 8 && Bits <= 13, "wide radix count is tuned for 8..13 bits");
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr std::size_t Buckets = std::size_t(1) << Bits;
    constexpr Key Mask = Key(Buckets - 1);
    std::array<std::uint32_t, 4 * Buckets> bank{};
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        prefetch_stream(src, i, n);
        const Key k0 = RT::encode(src[i + 0]);
        const Key k1 = RT::encode(src[i + 1]);
        const Key k2 = RT::encode(src[i + 2]);
        const Key k3 = RT::encode(src[i + 3]);
        ++bank[0 * Buckets + static_cast<std::size_t>((k0 >> shift) & Mask)];
        ++bank[1 * Buckets + static_cast<std::size_t>((k1 >> shift) & Mask)];
        ++bank[2 * Buckets + static_cast<std::size_t>((k2 >> shift) & Mask)];
        ++bank[3 * Buckets + static_cast<std::size_t>((k3 >> shift) & Mask)];
    }
    for (; i < n; ++i) {
        const Key k = RT::encode(src[i]);
        ++bank[static_cast<std::size_t>((k >> shift) & Mask)];
    }
    for (std::size_t d = 0; d < Buckets; ++d)
        out[d] = bank[d] + bank[Buckets + d] + bank[2 * Buckets + d] + bank[3 * Buckets + d];
}

/// One sweep over `src` fills the digit histogram of every radix pass at once.
///
/// Counting pass by pass re-reads the whole array once per pass: at 8M int32
/// that is 0.0134s of the 0.0863s the two passes spend.  The digits of all the
/// passes are shifts of one and the same encoded key, so a single read of the
/// array -- and a single encode per element -- can feed all of them.  Four
/// banks per pass keep consecutive increments off the same cache line, which
/// is what keeps the read-modify-write chain from stalling.
///
/// `out` receives `Passes` consecutive histograms of `1 << Bits` counters.
template <class T, unsigned Bits, unsigned Passes>
inline void radix_count_all_value_passes_wide(const T* FYX_RESTRICT src, std::size_t n,
                                              unsigned first_shift,
                                              std::uint32_t* FYX_RESTRICT out) noexcept {
    static_assert(Bits >= 8 && Bits <= 13, "wide radix count is tuned for 8..13 bits");
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr std::size_t Buckets = std::size_t(1) << Bits;
    constexpr std::size_t kBanks  = 4;
    constexpr Key Mask = Key(Buckets - 1);

    std::array<std::uint32_t, kBanks * Passes * Buckets> bank{};
    std::size_t i = 0;
    for (; i + kBanks <= n; i += kBanks) {
        prefetch_stream(src, i, n);
        Key k[kBanks];
        for (std::size_t j = 0; j < kBanks; ++j) k[j] = RT::encode(src[i + j]);
        for (unsigned pi = 0; pi < Passes; ++pi) {
            const unsigned shift = first_shift + pi * Bits;
            std::uint32_t* b = bank.data() + std::size_t(pi) * kBanks * Buckets;
            for (std::size_t j = 0; j < kBanks; ++j)
                ++b[j * Buckets + static_cast<std::size_t>((k[j] >> shift) & Mask)];
        }
    }
    for (; i < n; ++i) {
        const Key k = RT::encode(src[i]);
        for (unsigned pi = 0; pi < Passes; ++pi)
            ++bank[std::size_t(pi) * kBanks * Buckets +
                   static_cast<std::size_t>((k >> (first_shift + pi * Bits)) & Mask)];
    }
    for (unsigned pi = 0; pi < Passes; ++pi) {
        const std::uint32_t* b = bank.data() + std::size_t(pi) * kBanks * Buckets;
        std::uint32_t* o = out + std::size_t(pi) * Buckets;
        for (std::size_t d = 0; d < Buckets; ++d)
            o[d] = b[d] + b[Buckets + d] + b[2 * Buckets + d] + b[3 * Buckets + d];
    }
}

template <class Key, unsigned Bits>
inline void radix_scatter_key_pass_wide(const Key* FYX_RESTRICT src, std::size_t n,
                                        Key* FYX_RESTRICT dst, unsigned shift,
                                        std::size_t* FYX_RESTRICT offset,
                                        bool can_stream) {
    static_assert(Bits >= 8 && Bits <= 13, "wide radix scatter is tuned for 8..13 bits");
    constexpr std::size_t Buckets = std::size_t(1) << Bits;
    constexpr std::size_t kPerLine = WcbTraits<Key>::kPerLine;
    constexpr Key Mask = Key(Buckets - 1);

    ScratchLease<Key> wcb_lease(2 * Buckets * kPerLine + kPerLine);
    if (!wcb_lease.valid()) {
        for (std::size_t i = 0; i < n; ++i) {
            const Key k = src[i];
            const std::size_t b = static_cast<std::size_t>((k >> shift) & Mask);
            dst[offset[b]++] = k;
        }
        return;
    }

    const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(wcb_lease.get());
    const std::uintptr_t mis = (kCacheLine - (raw & (kCacheLine - 1))) & (kCacheLine - 1);
    Key* FYX_RESTRICT line = reinterpret_cast<Key*>(raw + mis);
    Key* FYX_RESTRICT head = line + Buckets * kPerLine;

    std::array<std::uint32_t, Buckets> fill{};
    std::array<std::uint32_t, Buckets> hn{};
    std::array<std::uint32_t, Buckets> need{};
    std::array<std::size_t, Buckets> base{};
    std::size_t remaining_heads = 0;
    for (std::size_t b = 0; b < Buckets; ++b) {
        base[b] = offset[b];
        if (can_stream) {
            const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(dst + offset[b]);
            const std::size_t align = (kCacheLine - (addr & (kCacheLine - 1))) & (kCacheLine - 1);
            need[b] = static_cast<std::uint32_t>(align / sizeof(Key));
            remaining_heads += need[b];
        }
    }

    auto push_line = [&](Key k, std::size_t b) {
        Key* L = line + b * kPerLine;
        const std::uint32_t f = fill[b];
        L[f] = k;
        if (FYX_UNLIKELY(f + 1 == kPerLine)) {
            Key* out = dst + offset[b] + need[b];
            if (can_stream) stream_cache_line(out, L);
            else std::memcpy(out, L, kCacheLine);
            offset[b] += kPerLine;
            fill[b] = 0;
        } else {
            fill[b] = f + 1;
        }
    };

    std::size_t i = 0;
    for (; i < n && remaining_heads != 0; ++i) {
        prefetch_stream(src, i, n);
        const Key k = src[i];
        const std::size_t b = static_cast<std::size_t>((k >> shift) & Mask);
        if (FYX_UNLIKELY(hn[b] < need[b])) {
            head[b * kPerLine + hn[b]] = k;
            ++hn[b];
            --remaining_heads;
            continue;
        }
        push_line(k, b);
    }
    for (; i < n; ++i) {
        prefetch_stream(src, i, n);
        const Key k = src[i];
        const std::size_t b = static_cast<std::size_t>((k >> shift) & Mask);
        push_line(k, b);
    }

    for (std::size_t b = 0; b < Buckets; ++b) {
        if (hn[b])
            std::memcpy(dst + base[b], head + b * kPerLine,
                        static_cast<std::size_t>(hn[b]) * sizeof(Key));
        if (fill[b])
            std::memcpy(dst + offset[b] + need[b], line + b * kPerLine,
                        static_cast<std::size_t>(fill[b]) * sizeof(Key));
        offset[b] += fill[b] + hn[b];
    }
    if (can_stream) store_fence();
}

template <class T, class Key, unsigned Bits>
inline void radix_scatter_encode_pass_wide(const T* FYX_RESTRICT src, std::size_t n,
                                           Key* FYX_RESTRICT dst, unsigned shift,
                                           std::size_t* FYX_RESTRICT offset,
                                           bool can_stream) {
    static_assert(Bits >= 8 && Bits <= 13, "wide radix scatter is tuned for 8..13 bits");
    using RT = RadixTraits<T>;
    constexpr std::size_t Buckets = std::size_t(1) << Bits;
    constexpr std::size_t kPerLine = WcbTraits<Key>::kPerLine;
    constexpr Key Mask = Key(Buckets - 1);

    ScratchLease<Key> wcb_lease(2 * Buckets * kPerLine + kPerLine);
    if (!wcb_lease.valid()) {
        for (std::size_t i = 0; i < n; ++i) {
            const Key k = RT::encode(src[i]);
            const std::size_t b = static_cast<std::size_t>((k >> shift) & Mask);
            dst[offset[b]++] = k;
        }
        return;
    }

    const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(wcb_lease.get());
    const std::uintptr_t mis = (kCacheLine - (raw & (kCacheLine - 1))) & (kCacheLine - 1);
    Key* FYX_RESTRICT line = reinterpret_cast<Key*>(raw + mis);
    Key* FYX_RESTRICT head = line + Buckets * kPerLine;

    std::array<std::uint32_t, Buckets> fill{};
    std::array<std::uint32_t, Buckets> hn{};
    std::array<std::uint32_t, Buckets> need{};
    std::array<std::size_t, Buckets> base{};
    std::size_t remaining_heads = 0;
    for (std::size_t b = 0; b < Buckets; ++b) {
        base[b] = offset[b];
        if (can_stream) {
            const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(dst + offset[b]);
            const std::size_t align = (kCacheLine - (addr & (kCacheLine - 1))) & (kCacheLine - 1);
            need[b] = static_cast<std::uint32_t>(align / sizeof(Key));
            remaining_heads += need[b];
        }
    }

    auto push_line = [&](Key k, std::size_t b) {
        Key* L = line + b * kPerLine;
        const std::uint32_t f = fill[b];
        L[f] = k;
        if (FYX_UNLIKELY(f + 1 == kPerLine)) {
            Key* out = dst + offset[b] + need[b];
            if (can_stream) stream_cache_line(out, L);
            else std::memcpy(out, L, kCacheLine);
            offset[b] += kPerLine;
            fill[b] = 0;
        } else {
            fill[b] = f + 1;
        }
    };

    std::size_t i = 0;
    for (; i < n && remaining_heads != 0; ++i) {
        prefetch_stream(src, i, n);
        const Key k = RT::encode(src[i]);
        const std::size_t b = static_cast<std::size_t>((k >> shift) & Mask);
        if (FYX_UNLIKELY(hn[b] < need[b])) {
            head[b * kPerLine + hn[b]] = k;
            ++hn[b];
            --remaining_heads;
            continue;
        }
        push_line(k, b);
    }
    for (; i < n; ++i) {
        prefetch_stream(src, i, n);
        const Key k = RT::encode(src[i]);
        const std::size_t b = static_cast<std::size_t>((k >> shift) & Mask);
        push_line(k, b);
    }

    for (std::size_t b = 0; b < Buckets; ++b) {
        if (hn[b])
            std::memcpy(dst + base[b], head + b * kPerLine,
                        static_cast<std::size_t>(hn[b]) * sizeof(Key));
        if (fill[b])
            std::memcpy(dst + offset[b] + need[b], line + b * kPerLine,
                        static_cast<std::size_t>(fill[b]) * sizeof(Key));
        offset[b] += fill[b] + hn[b];
    }
    if (can_stream) store_fence();
}

template <class T, class Key, unsigned Bits>
inline void radix_scatter_key_decode_pass_wide(const Key* FYX_RESTRICT src, std::size_t n,
                                               T* FYX_RESTRICT dst, unsigned shift,
                                               std::size_t* FYX_RESTRICT offset,
                                               bool can_stream) {
    static_assert(Bits >= 8 && Bits <= 13, "wide radix scatter is tuned for 8..13 bits");
    static_assert(sizeof(T) == sizeof(Key), "decoded scatter expects word-sized values");
    using RT = RadixTraits<T>;
    constexpr std::size_t Buckets = std::size_t(1) << Bits;
    constexpr std::size_t kPerLine = WcbTraits<T>::kPerLine;
    constexpr Key Mask = Key(Buckets - 1);

    ScratchLease<T> wcb_lease(2 * Buckets * kPerLine + kPerLine);
    if (!wcb_lease.valid()) {
        for (std::size_t i = 0; i < n; ++i) {
            const Key k = src[i];
            const std::size_t b = static_cast<std::size_t>((k >> shift) & Mask);
            dst[offset[b]++] = RT::decode(k);
        }
        return;
    }

    const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(wcb_lease.get());
    const std::uintptr_t mis = (kCacheLine - (raw & (kCacheLine - 1))) & (kCacheLine - 1);
    T* FYX_RESTRICT line = reinterpret_cast<T*>(raw + mis);
    T* FYX_RESTRICT head = line + Buckets * kPerLine;

    std::array<std::uint32_t, Buckets> fill{};
    std::array<std::uint32_t, Buckets> hn{};
    std::array<std::uint32_t, Buckets> need{};
    std::array<std::size_t, Buckets> base{};
    std::size_t remaining_heads = 0;
    for (std::size_t b = 0; b < Buckets; ++b) {
        base[b] = offset[b];
        if (can_stream) {
            const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(dst + offset[b]);
            const std::size_t align = (kCacheLine - (addr & (kCacheLine - 1))) & (kCacheLine - 1);
            need[b] = static_cast<std::uint32_t>(align / sizeof(T));
            remaining_heads += need[b];
        }
    }

    auto push_line = [&](T v, std::size_t b) {
        T* L = line + b * kPerLine;
        const std::uint32_t f = fill[b];
        L[f] = v;
        if (FYX_UNLIKELY(f + 1 == kPerLine)) {
            T* out = dst + offset[b] + need[b];
            if (can_stream) stream_cache_line(out, L);
            else std::memcpy(out, L, kCacheLine);
            offset[b] += kPerLine;
            fill[b] = 0;
        } else {
            fill[b] = f + 1;
        }
    };

    std::size_t i = 0;
    for (; i < n && remaining_heads != 0; ++i) {
        prefetch_stream(src, i, n);
        const Key k = src[i];
        const std::size_t b = static_cast<std::size_t>((k >> shift) & Mask);
        const T v = RT::decode(k);
        if (FYX_UNLIKELY(hn[b] < need[b])) {
            head[b * kPerLine + hn[b]] = v;
            ++hn[b];
            --remaining_heads;
            continue;
        }
        push_line(v, b);
    }
    for (; i < n; ++i) {
        prefetch_stream(src, i, n);
        const Key k = src[i];
        const std::size_t b = static_cast<std::size_t>((k >> shift) & Mask);
        push_line(RT::decode(k), b);
    }

    for (std::size_t b = 0; b < Buckets; ++b) {
        if (hn[b])
            std::memcpy(dst + base[b], head + b * kPerLine,
                        static_cast<std::size_t>(hn[b]) * sizeof(T));
        if (fill[b])
            std::memcpy(dst + offset[b] + need[b], line + b * kPerLine,
                        static_cast<std::size_t>(fill[b]) * sizeof(T));
        offset[b] += fill[b] + hn[b];
    }
    if (can_stream) store_fence();
}

// ---------------------------------------------------------------------------
// Vectorised tie scan
// ---------------------------------------------------------------------------
// After the two radix passes every key is ordered by its top PrefixBits bits
// and only the groups that share a prefix still need sorting.  Finding those
// groups means comparing every element's prefix with its predecessor's -- a
// full pass whose cost has nothing to do with how little work it turns up.
// On uniform data almost every group holds a single element, so the scan is
// the most expensive part of the repair and repairs almost nothing.
//
// AVX-512 answers for sixteen elements (eight for 64-bit keys) at a time.  One
// load, one encode, one shift, and one compare against the vector rotated by
// a single lane with the previous prefix shifted into lane 0: the resulting
// mask has a bit for every lane whose prefix equals its predecessor's, which
// is exactly "this vector starts a tie group".  An empty mask -- the common
// case -- skips the whole vector.  Only a vector that actually holds a tie
// falls back to the element-at-a-time path, and only for its own lanes.
//
// The vector encoder reproduces RadixTraits<T>::encode bit for bit: unsigned
// keys pass through, signed keys flip the sign bit, and IEC-559 floats flip
// with (arithmetic-shift(sign) | sign_bit), all ones for a negative.
// ---------------------------------------------------------------------------

#if FYX_HAS_AVX512_CODE
FYX_ISA_BEGIN("avx512f")
namespace isa_avx512_keys {

/// 1 << 63 as a signed long long, spelled once so the intrinsics macro sees a
/// plain identifier instead of a cast it cannot parse.
static constexpr long long kSign64 = -9223372036854775807LL - 1;

template <unsigned Size, bool Signed, bool Float>
struct TieKeyImpl {
    static constexpr bool     defined = false;
    static constexpr unsigned lanes   = 1;
    static inline __m512i enc(__m512i) { return _mm512_setzero_si512(); }
    static inline __m512i dec(__m512i) { return _mm512_setzero_si512(); }
};

template <> struct TieKeyImpl<4, false, false> {          // uint32
    static constexpr bool     defined = true;
    static constexpr unsigned lanes   = 16;
    static inline __m512i enc(__m512i v) { return v; }
    static inline __m512i dec(__m512i k) { return k; }
};
template <> struct TieKeyImpl<4, true, false> {           // int32
    static constexpr bool     defined = true;
    static constexpr unsigned lanes   = 16;
    static inline __m512i enc(__m512i v) {
        return _mm512_xor_si512(v, _mm512_set1_epi32(int(0x80000000u)));
    }
    static inline __m512i dec(__m512i k) { return enc(k); }
};
template <> struct TieKeyImpl<4, true, true> {            // float
    static constexpr bool     defined = true;
    static constexpr unsigned lanes   = 16;
    static inline __m512i enc(__m512i u) {
        const __m512i m = _mm512_or_si512(_mm512_srai_epi32(u, 31),
                                          _mm512_set1_epi32(int(0x80000000u)));
        return _mm512_xor_si512(u, m);
    }
    // Inverse: the encoded top bit is set when the original was positive.
    static inline __m512i dec(__m512i k) {
        const __m512i s = _mm512_srai_epi32(k, 31);           // all ones if positive
        const __m512i m = _mm512_or_si512(
            _mm512_and_si512(s, _mm512_set1_epi32(int(0x80000000u))),
            _mm512_andnot_si512(s, _mm512_set1_epi32(-1)));
        return _mm512_xor_si512(k, m);
    }
};
template <> struct TieKeyImpl<8, false, false> {          // uint64
    static constexpr bool     defined = true;
    static constexpr unsigned lanes   = 8;
    static inline __m512i enc(__m512i v) { return v; }
    static inline __m512i dec(__m512i k) { return k; }
};
template <> struct TieKeyImpl<8, true, false> {           // int64
    static constexpr bool     defined = true;
    static constexpr unsigned lanes   = 8;
    static constexpr long long kSign  = kSign64;
    static inline __m512i enc(__m512i v) {
        return _mm512_xor_si512(v, _mm512_set1_epi64(kSign));
    }
    static inline __m512i dec(__m512i k) { return enc(k); }
};
template <> struct TieKeyImpl<8, true, true> {            // double
    static constexpr bool     defined = true;
    static constexpr unsigned lanes   = 8;
    static constexpr long long kSign  = kSign64;
    static inline __m512i enc(__m512i u) {
        const __m512i m = _mm512_or_si512(_mm512_srai_epi64(u, 63),
                                          _mm512_set1_epi64(kSign));
        return _mm512_xor_si512(u, m);
    }
    static inline __m512i dec(__m512i k) {
        const __m512i s = _mm512_srai_epi64(k, 63);
        const __m512i m = _mm512_or_si512(_mm512_and_si512(s, _mm512_set1_epi64(kSign)),
                                          _mm512_andnot_si512(s, _mm512_set1_epi64(-1)));
        return _mm512_xor_si512(k, m);
    }
};

template <class T>
using TieKey = TieKeyImpl<sizeof(T),
                          std::is_signed<T>::value || std::is_floating_point<T>::value,
                          std::is_floating_point<T>::value>;

/// Sorts every run of equal top-PrefixBits prefixes in `p` in place.
template <class T, unsigned PrefixBits>
inline void scan_ties(T* FYX_RESTRICT p, std::size_t n) noexcept {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    using TK  = TieKey<T>;
    constexpr unsigned Lanes = TK::lanes;
    constexpr unsigned Shift = unsigned(sizeof(Key) * 8) - PrefixBits;
    if (n < 2) return;

    auto repair = [&](std::size_t lo, std::size_t hi) {
        for (std::size_t a = lo + 1; a < hi; ++a) {
            T v = p[a];
            const Key kv = RT::encode(v);
            std::size_t b = a;
            while (b > lo && RT::encode(p[b - 1]) > kv) { p[b] = p[b - 1]; --b; }
            p[b] = v;
        }
    };

    std::size_t i = 1, group_lo = 0;
    Key prev = RT::encode(p[0]) >> Shift;

    // Element-at-a-time fallback, used for the rare vector that holds a tie
    // and for the tail that does not fill one.
    auto step = [&](std::size_t stop) {
        for (; i < stop; ++i) {
            const Key cur = RT::encode(p[i]) >> Shift;
            if (cur != prev) {
                if (i - group_lo > 1) repair(group_lo, i);
                group_lo = i;
                prev = cur;
            }
        }
    };

    while (i + Lanes <= n) {
        const __m512i pref = Lanes == 16
            ? _mm512_srli_epi32(TK::enc(_mm512_loadu_si512(
                  reinterpret_cast<const void*>(p + i))), Shift)
            : _mm512_srli_epi64(TK::enc(_mm512_loadu_si512(
                  reinterpret_cast<const void*>(p + i))), Shift);
        // Rotate by one lane, shifting the previous vector's last prefix in.
        // The set1/intrinsic macros want plain identifiers, not casts.
        const int       pv32 = int(std::uint32_t(prev));
        const long long pv64 = static_cast<long long>(std::uint64_t(prev));
        __m512i carry;
        __mmask16 ties;
        if constexpr (Lanes == 16) {
            carry = _mm512_mask_set1_epi32(pref, __mmask16(1u << 15), pv32);
            ties  = _mm512_cmpeq_epi32_mask(pref, _mm512_alignr_epi32(pref, carry, 15));
        } else {
            carry = _mm512_mask_set1_epi64(pref, __mmask8(1u << 7), pv64);
            ties  = __mmask16(_mm512_cmpeq_epi64_mask(pref, _mm512_alignr_epi64(pref, carry, 7)));
        }
        if (ties == 0) {
            // Every element here starts its own group: close the one that was
            // open and skip the whole vector.
            if (i - group_lo > 1) repair(group_lo, i);
            i += Lanes;
            group_lo = i - 1;
        } else {
            // A tie poisons only its own run of lanes.  Walk the set bits --
            // one iteration per group, not per element -- and repair just
            // those; every other element here is a group of one.  A run of
            // set bits [a..b] means elements i+a .. i+b+1 share a prefix, so
            // the group is [i+a, i+b+2), or [group_lo, i+b+2) when the run
            // reaches lane 0 and continues the group that was already open.
            if ((ties & 1u) == 0 && i - group_lo > 1) repair(group_lo, i);
            std::uint32_t m = static_cast<std::uint32_t>(ties);
            while (m) {
                const unsigned a = unsigned(__builtin_ctz(m));
                unsigned b = a;
                while (b + 1 < Lanes && (m & (1u << (b + 1)))) ++b;
                // Bits a..b set means elements i+a-1 .. i+b share a prefix.
                const std::size_t lo = (a == 0) ? group_lo : i + a - 1;
                const std::size_t hi = i + b + 1;
                if (b == Lanes - 1) group_lo = lo;       // runs off the end
                else if (hi - lo > 1) repair(lo, hi);
                m &= ~((1u << (b + 1)) - 1);
            }
            if ((ties & (1u << (Lanes - 1))) == 0) group_lo = i + Lanes - 1;
            i += Lanes;
        }
        prev = RT::encode(p[i - 1]) >> Shift;   // i has advanced past the vector
    }
    step(n);
    if (n - group_lo > 1) repair(group_lo, n);
}

}   // namespace isa_avx512_keys
FYX_ISA_END
#endif   // FYX_HAS_AVX512_CODE

#if FYX_HAS_AVX512_CODE
FYX_ISA_BEGIN("avx512f,avx512cd,avx512vpopcntdq")
namespace isa_avx512_scat {

// ---------------------------------------------------------------------------
// Conflict-detection scatter
// ---------------------------------------------------------------------------
// Sixteen keys per step (eight for 64-bit), where the write-combining scatter
// moves one element at a time:
//
//   vpconflictd -> which earlier lanes of this vector want the same bucket
//   vpopcntd    -> this lane's rank among them
//   gather      -> the bucket's running destination index
//   + rank      -> a distinct destination for every lane, in bucket order
//   scatter     -> the keys land in order inside their bucket
//   scatter     -> every lane writes index+1 back; where lanes collide the
//                  highest one wins, and that lane carries the largest index
//                  of the group -- which is the next free position
//
// Needs AVX512CD and AVX512VPOPCNTDQ, hence its own ISA block.
//
// It beats the write-combining scatter while the array stays in cache and
// loses to it once the array leaves: this writes single elements, so every
// line it touches is read back first, whereas the write-combining scatter
// flushes whole lines with non-temporal stores that read nothing.
// Measured with tools/dev/scat.cpp, 12-bit digits, uniformly random keys:
//
//       this        write-combining
//   1M  2.74 ns/elem   3.26
//   2M  2.83           3.18
//   4M  4.78           3.38
//   8M  4.57           3.14
//
// so the kernel picks per size, not per type.
// ---------------------------------------------------------------------------

template <class T, class S, class D, unsigned Bits>
inline void scatter32(const S* FYX_RESTRICT src, std::size_t n, D* FYX_RESTRICT dst,
                      unsigned shift, std::uint32_t* FYX_RESTRICT off) noexcept {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    using KV  = isa_avx512_keys::TieKey<T>;
    constexpr std::uint32_t Mask = (std::uint32_t(1) << Bits) - 1;
    const __m512i vmask = _mm512_set1_epi32(int(Mask));
    const __m512i vone  = _mm512_set1_epi32(1);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(src + i));
        __m512i k, val;
        if constexpr (std::is_same<S, Key>::value) k = raw; else k = KV::enc(raw);
        if constexpr (std::is_same<D, Key>::value) val = k;  else val = KV::dec(k);
        const __m512i idx  = _mm512_and_si512(_mm512_srli_epi32(k, shift), vmask);
        const __m512i rank = _mm512_popcnt_epi32(_mm512_conflict_epi32(idx));
        const __m512i pos  = _mm512_add_epi32(_mm512_i32gather_epi32(idx, off, 4), rank);
        _mm512_i32scatter_epi32(dst, pos, val, 4);
        _mm512_i32scatter_epi32(off, idx, _mm512_add_epi32(pos, vone), 4);
    }
    for (; i < n; ++i) {
        Key k;
        if constexpr (std::is_same<S, Key>::value) k = Key(src[i]); else k = RT::encode(src[i]);
        const std::size_t b = std::size_t((k >> shift) & Key(Mask));
        if constexpr (std::is_same<D, Key>::value) dst[off[b]++] = D(k);
        else                                       dst[off[b]++] = RT::decode(k);
    }
}

template <class T, class S, class D, unsigned Bits>
inline void scatter64(const S* FYX_RESTRICT src, std::size_t n, D* FYX_RESTRICT dst,
                      unsigned shift, std::size_t* FYX_RESTRICT off) noexcept {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    using KV  = isa_avx512_keys::TieKey<T>;
    constexpr std::uint64_t Mask = (std::uint64_t(1) << Bits) - 1;
    const __m512i vmask = _mm512_set1_epi64(static_cast<long long>(Mask));
    const __m512i vone  = _mm512_set1_epi64(1);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(src + i));
        __m512i k, val;
        if constexpr (std::is_same<S, Key>::value) k = raw; else k = KV::enc(raw);
        if constexpr (std::is_same<D, Key>::value) val = k;  else val = KV::dec(k);
        const __m512i idx64 = _mm512_and_si512(_mm512_srli_epi64(k, shift), vmask);
        // Truncate to eight 32-bit indices.  A bitcast will not do: the low
        // half of the register holds four 64-bit lanes, which read as eight
        // 32-bit ones would be the halves of k0..k3, not k0..k7.
        const __m256i idx   = _mm512_cvtepi64_epi32(idx64);
        const __m512i rank  = _mm512_popcnt_epi64(_mm512_conflict_epi64(idx64));
        const __m512i pos   = _mm512_add_epi64(_mm512_i32gather_epi64(idx, off, 8), rank);
        _mm512_i64scatter_epi64(dst, pos, val, 8);
        _mm512_i32scatter_epi64(off, idx, _mm512_add_epi64(pos, vone), 8);
    }
    for (; i < n; ++i) {
        Key k;
        if constexpr (std::is_same<S, Key>::value) k = Key(src[i]); else k = RT::encode(src[i]);
        const std::size_t b = std::size_t((k >> shift) & Key(Mask));
        if constexpr (std::is_same<D, Key>::value) dst[off[b]++] = D(k);
        else                                       dst[off[b]++] = RT::decode(k);
    }
}

}   // namespace isa_avx512_scat
FYX_ISA_END
#endif   // FYX_HAS_AVX512_CODE

/// One scatter pass, on whichever of the two engines the size calls for.
///
/// `off32` is scratch for the 32-bit AVX-512 path, which counts destinations
/// in 32 bits and only runs when n fits in them.
template <class T, class S, class D, unsigned Bits>
inline void radix_scatter_wide_pass(const S* FYX_RESTRICT src, std::size_t n,
                                    D* FYX_RESTRICT dst, unsigned shift,
                                    std::size_t* FYX_RESTRICT pos,
                                    std::uint32_t* FYX_RESTRICT off32,
                                    bool avx512_ok, bool can_stream) noexcept {
    using Key = typename RadixTraits<T>::Key;
#if FYX_HAS_AVX512_CODE
    if (avx512_ok) {
        if constexpr (sizeof(Key) == 4) {
            constexpr std::size_t B = std::size_t(1) << Bits;
            for (std::size_t d = 0; d < B; ++d) off32[d] = std::uint32_t(pos[d]);
            isa_avx512_scat::scatter32<T, S, D, Bits>(src, n, dst, shift, off32);
        } else {
            isa_avx512_scat::scatter64<T, S, D, Bits>(src, n, dst, shift, pos);
        }
        return;
    }
#else
    (void)off32; (void)avx512_ok;
#endif
    if constexpr (std::is_same<S, Key>::value && std::is_same<D, Key>::value)
        radix_scatter_key_pass_wide<Key, Bits>(src, n, dst, shift, pos, can_stream);
    else if constexpr (std::is_same<D, Key>::value)
        radix_scatter_encode_pass_wide<T, Key, Bits>(src, n, dst, shift, pos, can_stream);
    else
        radix_scatter_key_decode_pass_wide<T, Key, Bits>(src, n, dst, shift, pos, can_stream);
}


template <class T, unsigned PrefixBits>
inline void radix_sort_high_prefix_decoded_ties(T* p, std::size_t n) {
#if FYX_HAS_AVX512_CODE
    if constexpr (isa_avx512_keys::TieKey<T>::defined) {
        if (use_avx512()) { isa_avx512_keys::scan_ties<T, PrefixBits>(p, n); return; }
    }
#endif

    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr unsigned Shift = unsigned(sizeof(Key) * 8) - PrefixBits;
    if (n < 2) return;
    std::size_t group_lo = 0;
    Key prev_prefix = RT::encode(p[0]) >> Shift;
    auto repair_group = [&](std::size_t lo, std::size_t hi) {
        for (std::size_t a = lo + 1; a < hi; ++a) {
            T v = p[a];
            const Key kv = RT::encode(v);
            std::size_t b = a;
            while (b > lo && RT::encode(p[b - 1]) > kv) {
                p[b] = p[b - 1];
                --b;
            }
            p[b] = v;
        }
    };
    for (std::size_t i = 1; i < n; ++i) {
        const Key cur_prefix = RT::encode(p[i]) >> Shift;
        if (cur_prefix != prev_prefix) {
            repair_group(group_lo, i);
            group_lo = i;
            prev_prefix = cur_prefix;
        }
    }
    repair_group(group_lo, n);
}

template <class T, unsigned PrefixBits, unsigned Bits>
inline bool try_parallel_radix_high_prefix_key_sort_wide(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T> || PrefixBits == 0 || PrefixBits > 64 ||
                  Bits <= 8 || Bits > 13 || (PrefixBits % Bits) != 0) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        if constexpr (sizeof(Key) != 8 && sizeof(Key) != 4) {
            (void)p; (void)n; (void)descending;
            return false;
        } else {
            constexpr unsigned Passes = PrefixBits / Bits;
            constexpr unsigned FirstShift = unsigned(sizeof(Key) * 8) - PrefixBits;
            constexpr std::size_t Buckets = std::size_t(1) << Bits;
            if (n < (std::size_t(1) << 19) || !parallel_available()) return false;
            if (!radix_high_prefix_probe<T, PrefixBits>(p, n)) return false;

            ScratchLease<Key> lease(n * 2);
            if (!lease.valid()) return false;
            Key* a = lease.get();
            Key* b = a + n;

            const std::size_t chunks = std::min<std::size_t>(
                adaptive_parallel_radix_chunks<T>(n),
                std::max<std::size_t>(2, global_pool().nworkers()));
            if ((n + chunks - 1) / chunks > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return false;
            std::vector<std::array<std::uint32_t, Buckets>> local(chunks);
            std::vector<std::array<std::size_t, Buckets>> base(chunks);
            Key* src = a;
            Key* dst = b;
            const bool can_stream = have_nt_stores();

            for (unsigned pi = 0; pi < Passes; ++pi) {
                const unsigned shift = FirstShift + pi * Bits;
                auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
                    for (std::size_t c = c_lo; c < c_hi; ++c) {
                        const std::size_t lo = (c * n) / chunks;
                        const std::size_t hi = ((c + 1) * n) / chunks;
                        if (pi == 0)
                            radix_count_value_pass_banked_wide<T, Bits>(p + lo, hi - lo, shift, local[c].data());
                        else
                            radix_count_key_pass_banked_wide<Key, Bits>(src + lo, hi - lo, shift, local[c].data());
                    }
                };
                parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);

                std::size_t run = 0;
                for (std::size_t d = 0; d < Buckets; ++d) {
                    for (std::size_t c = 0; c < chunks; ++c) {
                        base[c][d] = run;
                        run += local[c][d];
                    }
                }

                // Chunks, not the whole array, decide the scatter engine: a
                // chunk is what one thread walks, and it is the working set of
                // one thread that has to stay in cache.  See the table above
                // isa_avx512_scat::scatter32.
                const bool avx512_ok =
                    use_avx512_conflict() && n <= std::size_t(0xffffffffu) &&
                    ((n + chunks - 1) / chunks) * sizeof(Key) <= (std::size_t(1) << 23);

                if (pi == 0) {
                    auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
                        std::vector<std::uint32_t> off32(
                            (avx512_ok && sizeof(Key) == 4) ? Buckets : std::size_t(0));
                        for (std::size_t c = c_lo; c < c_hi; ++c) {
                            const std::size_t lo = (c * n) / chunks;
                            const std::size_t hi = ((c + 1) * n) / chunks;
                            auto pos = base[c];
                            radix_scatter_wide_pass<T, T, Key, Bits>(
                                p + lo, hi - lo, src, shift, pos.data(), off32.data(),
                                avx512_ok, can_stream);
                        }
                    };
                    parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);
                } else if (pi + 1 == Passes) {
                    auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
                        std::vector<std::uint32_t> off32(
                            (avx512_ok && sizeof(Key) == 4) ? Buckets : std::size_t(0));
                        for (std::size_t c = c_lo; c < c_hi; ++c) {
                            const std::size_t lo = (c * n) / chunks;
                            const std::size_t hi = ((c + 1) * n) / chunks;
                            auto pos = base[c];
                            radix_scatter_wide_pass<T, Key, T, Bits>(
                                src + lo, hi - lo, p, shift, pos.data(), off32.data(),
                                avx512_ok, can_stream);
                        }
                    };
                    parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);
                } else {
                    auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
                        std::vector<std::uint32_t> off32(
                            (avx512_ok && sizeof(Key) == 4) ? Buckets : std::size_t(0));
                        for (std::size_t c = c_lo; c < c_hi; ++c) {
                            const std::size_t lo = (c * n) / chunks;
                            const std::size_t hi = ((c + 1) * n) / chunks;
                            auto pos = base[c];
                            radix_scatter_wide_pass<T, Key, Key, Bits>(
                                src + lo, hi - lo, dst, shift, pos.data(), off32.data(),
                                avx512_ok, can_stream);
                        }
                    };
                    parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);
                    Key* t = src; src = dst; dst = t;
                }
            }

            radix_sort_high_prefix_decoded_ties<T, PrefixBits>(p, n);
            if (descending) std::reverse(p, p + n);
            return true;
        }
    }
}

// ---------------------------------------------------------------------------
// How wide the high prefix should be
//
// A high-prefix radix pass costs two traversals of the range per digit plus a
// tie repair inside every prefix group, so the width that pays depends on how
// many groups the data falls into -- which is a property of the data, not of
// the type it happens to be stored in.  Values in [0, 2^30) share their high
// bits and collapse into tens of thousands of groups at 24 bits, where the tie
// repair costs more than the extra pass a 36-bit prefix would have spent
// (0.046s against 0.026s for 1M doubles); uniform 64-bit data splinters into
// millions of groups at 24 bits already, and there the wider prefix buys
// nothing but two more traversals (0.011s against 0.015s).
//
// Sampling counts the groups: S samples landing in G groups collide about
// S*S/(2G) times, so the collisions the sample sees estimate G, and the only
// question left is whether the average group is small enough for the tie
// repair to be free (two elements or fewer).
// ---------------------------------------------------------------------------
template <class T>
inline unsigned radix_choose_prefix_bits(const T* p, std::size_t n,
                                         unsigned narrow, unsigned wide) noexcept {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr std::size_t S = 4096;
    if (n < S * 8) return wide;
    std::array<Key, S> sample;
    const unsigned shift = unsigned(sizeof(Key) * 8) - narrow;
    for (std::size_t j = 0; j < S; ++j)
        sample[j] = RT::encode(p[(j * (n - 1)) / (S - 1)]) >> shift;
    std::sort(sample.begin(), sample.end());
    std::size_t collisions = 0;
    for (std::size_t j = 1; j < S; ++j)
        if (sample[j] == sample[j - 1]) ++collisions;
    // groups ~ S*S/(2*collisions), so the average group holds about
    // 2*n*collisions/(S*S) elements.
    return (n * collisions <= S * S) ? narrow : wide;
}

template <class T>
inline bool try_parallel_radix_high_prefix_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        using Key = typename RadixTraits<T>::Key;
        if constexpr (sizeof(Key) != 8 && sizeof(Key) != 4) {
            (void)p; (void)n; (void)descending;
            return false;
        } else if constexpr (std::is_same<T, double>::value ||
                             (std::is_integral<T>::value && sizeof(T) == 8)) {
            // Narrow when the data splinters, wide when it clumps: see
            // radix_choose_prefix_bits.  The width used to be picked by type,
            // which is right for whichever kind of data it was tuned on and
            // wrong for the other.
            if (n <= (std::size_t(1) << 21)) {
                const unsigned w = radix_choose_prefix_bits<T>(p, n, 24, 36);
                return w == 24
                    ? try_parallel_radix_high_prefix_key_sort_wide<T, 24, 12>(p, n, descending)
                    : try_parallel_radix_high_prefix_key_sort_wide<T, 36, 12>(p, n, descending);
            }
            const unsigned w = radix_choose_prefix_bits<T>(p, n, 26, 39);
            return w == 26
                ? try_parallel_radix_high_prefix_key_sort_wide<T, 26, 13>(p, n, descending)
                : try_parallel_radix_high_prefix_key_sort_wide<T, 39, 13>(p, n, descending);
        } else if constexpr (sizeof(Key) == 4) {
            // Two threads make this the faster of the two parallel sorts up to
            // about three million elements, and the slower one above that:
            // 2M 0.0125 vs 0.0176, 4M 0.0253 vs 0.0201, 8M 0.0516 vs 0.0397,
            // 16M 0.1137 vs 0.0787.  Decline past the crossover and let the
            // dispatcher fall back to try_parallel_radix32_wide_sort.
            if (n > (std::size_t(3) << 20)) return false;
            // A 32-bit key gets the same two-pass treatment as a 64-bit one.
            // Routing it here instead of through try_parallel_radix32_wide_sort
            // is what makes random int32 scale with the core count: at 1M the
            // wide sort gained 1.13x from two threads and this one gains 1.7x.
            if (n <= (std::size_t(1) << 21)) {
                const unsigned w = radix_choose_prefix_bits<T>(p, n, 24, 26);
                return w == 24
                    ? try_parallel_radix_high_prefix_key_sort_wide<T, 24, 12>(p, n, descending)
                    : try_parallel_radix_high_prefix_key_sort_wide<T, 26, 13>(p, n, descending);
            }
            return try_parallel_radix_high_prefix_key_sort_wide<T, 26, 13>(p, n, descending);
        } else {
            (void)p; (void)n; (void)descending;
            return false;
        }
    }
}

template <class T>
inline bool radix_sample_all_passes_vary(const T* p, std::size_t n) noexcept {
    if constexpr (!radix_supported_v<T>) {
        (void)p; (void)n;
        return false;
    } else {
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr unsigned Passes = RT::passes;
        if (n < 2) return false;
        const Key first = RT::encode(p[0]);
        unsigned varied = 0;
        const std::size_t s = std::min<std::size_t>(n, kProfileSampleLimit);
        for (std::size_t j = 1; j < s; ++j) {
            const std::size_t idx = (j * (n - 1)) / (s - 1);
            const Key k = RT::encode(p[idx]);
            for (unsigned pass = 0; pass < Passes; ++pass) {
                if (radix_digit(k, pass) != radix_digit(first, pass))
                    varied |= (1u << pass);
            }
            if (varied == ((1u << Passes) - 1u)) return true;
        }
        return false;
    }
}

template <unsigned Bits, class CountOne, class ScatterOne>
inline void parallel_radix_wide_count_scatter(std::size_t chunks,
                                              CountOne count_one,
                                              ScatterOne scatter_one) {
    constexpr std::size_t Buckets = std::size_t(1) << Bits;
    std::vector<std::array<std::uint32_t, Buckets>> local(chunks);
    std::vector<std::array<std::size_t, Buckets>> base(chunks);

    auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c)
            count_one(c, local[c].data());
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);

    std::size_t run = 0;
    for (std::size_t d = 0; d < Buckets; ++d) {
        for (std::size_t c = 0; c < chunks; ++c) {
            base[c][d] = run;
            run += local[c][d];
        }
    }

    auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            auto pos = base[c];
            scatter_one(c, pos.data());
        }
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);
}

template <class T>
inline std::size_t adaptive_parallel_radix32_wide_chunks(std::size_t n) {
    ThreadPool& pool = global_pool();
    std::size_t chunks = (n + kParallelThreshold - 1) / kParallelThreshold;
    const std::size_t per_worker = (n <= (std::size_t(1) << 22)) ? std::size_t(1) : std::size_t(2);
    const std::size_t max_chunks = std::max<std::size_t>(2, static_cast<std::size_t>(pool.nworkers()) * per_worker);
    if (chunks < 2) chunks = 2;
    if (chunks > max_chunks) chunks = max_chunks;
    (void)sizeof(T);
    return chunks;
}

template <class T>
inline bool try_parallel_radix32_wide_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!(std::is_integral<T>::value && !std::is_same<T, bool>::value &&
                    radix_supported_v<T> && sizeof(T) == 4)) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        if (n < (std::size_t(1) << 18) || !parallel_available()) return false;

        ScratchLease<Key> lease(n * 2);
        if (!lease.valid()) return false;
        Key* a = lease.get();
        Key* b = a + n;

        const std::size_t chunks = adaptive_parallel_radix32_wide_chunks<T>(n);
        if ((n + chunks - 1) / chunks >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return false;
        const bool can_stream = have_nt_stores() && n > (std::size_t(1) << 21);

        parallel_radix_wide_count_scatter<10>(chunks,
            [&](std::size_t c, std::uint32_t* out) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                radix_count_value_pass_banked_wide<T, 10>(p + lo, hi - lo, 0, out);
            },
            [&](std::size_t c, std::size_t* pos) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                radix_scatter_encode_pass_wide<T, Key, 10>(p + lo, hi - lo, a, 0, pos, can_stream);
            });

        parallel_radix_wide_count_scatter<11>(chunks,
            [&](std::size_t c, std::uint32_t* out) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                radix_count_key_pass_banked_wide<Key, 11>(a + lo, hi - lo, 10, out);
            },
            [&](std::size_t c, std::size_t* pos) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                radix_scatter_key_pass_wide<Key, 11>(a + lo, hi - lo, b, 10, pos, can_stream);
            });

        parallel_radix_wide_count_scatter<11>(chunks,
            [&](std::size_t c, std::uint32_t* out) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                radix_count_key_pass_banked_wide<Key, 11>(b + lo, hi - lo, 21, out);
            },
            [&](std::size_t c, std::size_t* pos) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                radix_scatter_key_decode_pass_wide<T, Key, 11>(b + lo, hi - lo, p, 21, pos, can_stream);
            });

        if (descending) std::reverse(p, p + n);
        return true;
    }
}

template <class T>
inline bool try_parallel_radix_sort(T* p, std::size_t n, bool descending, bool assume_all_passes = false) {
    if constexpr (!radix_supported_v<T>) {
        (void)p; (void)n; (void)descending; (void)assume_all_passes;
        return false;
    } else {
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr unsigned Passes = RT::passes;
        // Chunked parallel radix removes the old sort-halves-and-compare-merge
        // fallback for large high-entropy numeric inputs.
        {
            if (n < (std::size_t(1) << 19) || !parallel_available()) return false;
            assume_all_passes = assume_all_passes && radix_sample_all_passes_vary<T>(p, n);

            if constexpr ((std::is_integral<T>::value && !std::is_same<T, bool>::value) ||
                          std::is_same<T, float>::value) {
                ScratchLease<T> tmp_lease(n);
                if (!tmp_lease.valid()) return false;
                T* tmp = tmp_lease.get();

                const std::size_t chunks = adaptive_parallel_radix_chunks<T>(n);
                if ((n + chunks - 1) / chunks >
                    static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return false;
                std::vector<RadixLocalHistogram<Passes>> local(chunks);
                for (std::size_t c = 0; c < chunks; ++c) {
                    if (assume_all_passes) std::memset(local[c].count[0], 0, sizeof(local[c].count[0]));
                    else local[c].clear();
                }
                if (assume_all_passes) {
                    auto hist_job = [&](std::size_t c_lo, std::size_t c_hi) {
                        for (std::size_t c = c_lo; c < c_hi; ++c) {
                            const std::size_t lo = (c * n) / chunks;
                            const std::size_t hi = ((c + 1) * n) / chunks;
                            auto& h = local[c];
                            radix_count_value_pass_banked<T>(p + lo, hi - lo, 0, h.count[0]);
                        }
                    };
                    parallel_for_index(std::size_t(0), chunks, std::size_t(1), hist_job);
                } else {
                    auto hist_job = [&](std::size_t c_lo, std::size_t c_hi) {
                        for (std::size_t c = c_lo; c < c_hi; ++c) {
                            const std::size_t lo = (c * n) / chunks;
                            const std::size_t hi = ((c + 1) * n) / chunks;
                            auto& h = local[c];
                            for (std::size_t i = lo; i < hi; ++i) {
                                const Key k = RT::encode(p[i]);
                                for (unsigned pass = 0; pass < Passes; ++pass)
                                    ++h.count[pass][radix_digit(k, pass)];
                            }
                        }
                    };
                    parallel_for_index(std::size_t(0), chunks, std::size_t(1), hist_job);
                }

                RadixPlan<Passes> plan;
                if (assume_all_passes) {
                    for (unsigned pass = 0; pass < Passes; ++pass) plan.active[plan.count++] = pass;
                } else {
                    RadixHistogram<Passes> hist;
                    hist.clear();
                    for (std::size_t c = 0; c < chunks; ++c) {
                        for (unsigned pass = 0; pass < Passes; ++pass)
                            for (unsigned d = 0; d < kRadixBuckets; ++d)
                                hist.count[pass][d] += local[c].count[pass][d];
                    }
                    plan = plan_radix<Passes>(hist, n);
                    if (plan.count == 0) { if (descending) std::reverse(p, p + n); return true; }
                }

                T* src = p;
                T* dst = tmp;
                std::vector<std::array<std::size_t, kRadixBuckets>> base(chunks);
                std::vector<std::array<std::uint32_t, kRadixBuckets>> pass_local(chunks);
                const bool can_stream = have_nt_stores();

                for (unsigned pi = 0; pi < plan.count; ++pi) {
                    const unsigned pass = plan.active[pi];
                    const unsigned shift = pass * kRadixBits;
                    if (pi != 0) {
                        auto pass_count_job = [&](std::size_t c_lo, std::size_t c_hi) {
                            for (std::size_t c = c_lo; c < c_hi; ++c) {
                                auto& pc = pass_local[c];
                                const std::size_t lo = (c * n) / chunks;
                                const std::size_t hi = ((c + 1) * n) / chunks;
                                radix_count_value_pass_banked<T>(src + lo, hi - lo, shift, pc.data());
                            }
                        };
                        parallel_for_index(std::size_t(0), chunks, std::size_t(1), pass_count_job);
                    }
                    std::size_t run = 0;
                    for (unsigned d = 0; d < kRadixBuckets; ++d) {
                        for (std::size_t c = 0; c < chunks; ++c) {
                            base[c][d] = run;
                            run += (pi == 0) ? static_cast<std::size_t>(local[c].count[pass][d])
                                             : pass_local[c][d];
                        }
                    }
                    auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
                        constexpr std::size_t kPerLine = WcbTraits<T>::kPerLine;
                        ScratchLease<T> wcb_lease(2 * kRadixBuckets * kPerLine + kPerLine);
                        RadixScatterScratch<T> sc;
                        bool local_stream = false;
                        if (wcb_lease.valid()) {
                            const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(wcb_lease.get());
                            const std::uintptr_t mis = (kCacheLine - (addr & (kCacheLine - 1))) & (kCacheLine - 1);
                            sc.line = reinterpret_cast<T*>(addr + mis);
                            sc.head = sc.line + kRadixBuckets * kPerLine;
                            local_stream = can_stream;
                        }
                        for (std::size_t c = c_lo; c < c_hi; ++c) {
                            const std::size_t lo = (c * n) / chunks;
                            const std::size_t hi = ((c + 1) * n) / chunks;
                            auto pos = base[c];
                            if (wcb_lease.valid()) {
                                radix_scatter_value_pass<T>(src + lo, hi - lo, dst, shift, pos.data(), sc, local_stream);
                            } else {
                                for (std::size_t i = lo; i < hi; ++i) {
                                    const T v = src[i];
                                    const Key k = RT::encode(v);
                                    const unsigned d = static_cast<unsigned>((k >> shift) & Key(kRadixMask));
                                    dst[pos[d]++] = v;
                                }
                            }
                        }
                    };
                    parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);
                    T* t = src; src = dst; dst = t;
                }
                if (src != p) {
                    auto copy_job = [&](std::size_t lo, std::size_t hi) {
                        for (std::size_t i = lo; i < hi; ++i) p[i] = src[i];
                    };
                    parallel_for_index(std::size_t(0), n, kParallelThreshold, copy_job);
                }
                if (descending) std::reverse(p, p + n);
                return true;
            }

            ScratchLease<Key> lease(n * 2);
            if (!lease.valid()) return false;
            Key* a = lease.get();
            Key* b = a + n;

            const std::size_t chunks = adaptive_parallel_radix_chunks<T>(n);
            if ((n + chunks - 1) / chunks >
                static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return false;
            std::vector<RadixLocalHistogram<Passes>> local(chunks);
            for (std::size_t c = 0; c < chunks; ++c) {
                if (assume_all_passes) std::memset(local[c].count[0], 0, sizeof(local[c].count[0]));
                else local[c].clear();
            }

            if (assume_all_passes) {
                auto hist_job = [&](std::size_t c_lo, std::size_t c_hi) {
                    for (std::size_t c = c_lo; c < c_hi; ++c) {
                        const std::size_t lo = (c * n) / chunks;
                        const std::size_t hi = ((c + 1) * n) / chunks;
                        auto& h = local[c];
                        for (std::size_t i = lo; i < hi; ++i) {
                            const Key k = RT::encode(p[i]);
                            a[i] = k;
                            ++h.count[0][radix_digit(k, 0)];
                        }
                    }
                };
                parallel_for_index(std::size_t(0), chunks, std::size_t(1), hist_job);
            } else {
                auto hist_job = [&](std::size_t c_lo, std::size_t c_hi) {
                    for (std::size_t c = c_lo; c < c_hi; ++c) {
                        const std::size_t lo = (c * n) / chunks;
                        const std::size_t hi = ((c + 1) * n) / chunks;
                        auto& h = local[c];
                        for (std::size_t i = lo; i < hi; ++i) {
                            const Key k = RT::encode(p[i]);
                            a[i] = k;
                            for (unsigned pass = 0; pass < Passes; ++pass)
                                ++h.count[pass][radix_digit(k, pass)];
                        }
                    }
                };
                parallel_for_index(std::size_t(0), chunks, std::size_t(1), hist_job);
            }

            RadixPlan<Passes> plan;
            if (assume_all_passes) {
                for (unsigned pass = 0; pass < Passes; ++pass) plan.active[plan.count++] = pass;
            } else {
                RadixHistogram<Passes> hist;
                hist.clear();
                for (std::size_t c = 0; c < chunks; ++c) {
                    for (unsigned pass = 0; pass < Passes; ++pass)
                        for (unsigned d = 0; d < kRadixBuckets; ++d)
                            hist.count[pass][d] += local[c].count[pass][d];
                }

                plan = plan_radix<Passes>(hist, n);
                if (plan.count == 0) return true;
            }

            Key* src = a;
            Key* dst = b;
            std::vector<std::array<std::size_t, kRadixBuckets>> base(chunks);
            std::vector<std::array<std::uint32_t, kRadixBuckets>> pass_local(chunks);
            const bool can_stream = have_nt_stores();

            for (unsigned pi = 0; pi < plan.count; ++pi) {
                const unsigned pass = plan.active[pi];
                const unsigned shift = pass * kRadixBits;

                if (pi != 0) {
                    auto pass_count_job = [&](std::size_t c_lo, std::size_t c_hi) {
                        for (std::size_t c = c_lo; c < c_hi; ++c) {
                            auto& pc = pass_local[c];
                            const std::size_t lo = (c * n) / chunks;
                            const std::size_t hi = ((c + 1) * n) / chunks;
                            radix_count_key_pass_banked<Key>(src + lo, hi - lo, shift, pc.data());
                        }
                    };
                    parallel_for_index(std::size_t(0), chunks, std::size_t(1), pass_count_job);
                }

                std::size_t run = 0;
                for (unsigned d = 0; d < kRadixBuckets; ++d) {
                    for (std::size_t c = 0; c < chunks; ++c) {
                        base[c][d] = run;
                        run += (pi == 0) ? static_cast<std::size_t>(local[c].count[pass][d])
                                         : pass_local[c][d];
                    }
                }

                if constexpr (std::is_same<T, double>::value) {
                if (pi + 1 == plan.count) {
                    auto scatter_decode_job = [&](std::size_t c_lo, std::size_t c_hi) {
                        constexpr std::size_t kPerLine = WcbTraits<T>::kPerLine;
                        ScratchLease<T> wcb_lease(2 * kRadixBuckets * kPerLine + kPerLine);
                        RadixScatterScratch<T> sc;
                        bool local_stream = false;
                        if (wcb_lease.valid()) {
                            const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(wcb_lease.get());
                            const std::uintptr_t mis = (kCacheLine - (addr & (kCacheLine - 1))) & (kCacheLine - 1);
                            sc.line = reinterpret_cast<T*>(addr + mis);
                            sc.head = sc.line + kRadixBuckets * kPerLine;
                            local_stream = can_stream;
                        }
                        for (std::size_t c = c_lo; c < c_hi; ++c) {
                            const std::size_t lo = (c * n) / chunks;
                            const std::size_t hi = ((c + 1) * n) / chunks;
                            auto pos = base[c];
                            if (wcb_lease.valid()) {
                                radix_scatter_decode_pass<T>(src + lo, hi - lo, p, shift, pos.data(), sc, local_stream);
                            } else {
                                for (std::size_t i = lo; i < hi; ++i) {
                                    const Key k = src[i];
                                    const unsigned d = static_cast<unsigned>((k >> shift) & Key(kRadixMask));
                                    p[pos[d]++] = RT::decode(k);
                                }
                            }
                        }
                    };
                    parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_decode_job);
                    if (descending) std::reverse(p, p + n);
                    return true;
                }
                }

                auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
                    constexpr std::size_t kPerLine = WcbTraits<Key>::kPerLine;
                    ScratchLease<Key> wcb_lease(2 * kRadixBuckets * kPerLine + kPerLine);
                    RadixScatterScratch<Key> sc;
                    bool local_stream = false;
                    if (wcb_lease.valid()) {
                        const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(wcb_lease.get());
                        const std::uintptr_t mis = (kCacheLine - (addr & (kCacheLine - 1))) & (kCacheLine - 1);
                        sc.line = reinterpret_cast<Key*>(addr + mis);
                        sc.head = sc.line + kRadixBuckets * kPerLine;
                        local_stream = can_stream;
                    }
                    for (std::size_t c = c_lo; c < c_hi; ++c) {
                        const std::size_t lo = (c * n) / chunks;
                        const std::size_t hi = ((c + 1) * n) / chunks;
                        auto pos = base[c];
                        if (wcb_lease.valid()) {
                            radix_scatter_pass<Key>(src + lo, hi - lo, dst, shift, pos.data(), sc, local_stream);
                        } else {
                            for (std::size_t i = lo; i < hi; ++i) {
                                const Key k = src[i];
                                const unsigned d = static_cast<unsigned>((k >> shift) & Key(kRadixMask));
                                dst[pos[d]++] = k;
                            }
                        }
                    }
                };
                parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);
                Key* t = src; src = dst; dst = t;
            }

            auto decode_job = [&](std::size_t lo, std::size_t hi) {
                for (std::size_t i = lo; i < hi; ++i) p[i] = RT::decode(src[i]);
            };
            parallel_for_index(std::size_t(0), n, kParallelThreshold, decode_job);
            if (descending) std::reverse(p, p + n);
            return true;
        }
    }
}

// Fill the output of a rank-counting kernel: `offset[0..d]` are the output
// boundaries of each rank (in output order) and `rank_value[r]` is the value
// rank r emits.  Splitting by output range instead of by rank keeps the work
// balanced when one rank dominates the counts -- with d=8 the per-rank split
// launched a single job and filled 8 MB on one thread while the second core
// idled.
template <class T>
inline void parallel_fill_by_ranks(T* p, const std::size_t* offset, const T* rank_value,
                                   std::size_t d) {
    const std::size_t total = offset[d];
    std::size_t ntasks = d;
    {
        ThreadPool& pool        = global_pool();
        const std::size_t hotmax = std::max<std::size_t>(2, pool.nworkers()) * 4;
        if (ntasks > hotmax) ntasks = hotmax;
        while (ntasks > 1 && total / ntasks < std::size_t(4096)) --ntasks;
    }
    if (ntasks <= 1) {
        for (std::size_t r = 0; r < d; ++r)
            std::fill(p + offset[r], p + offset[r + 1], rank_value[r]);
        return;
    }
    auto job = [&](std::size_t t_lo, std::size_t t_hi) {
        for (std::size_t t = t_lo; t < t_hi; ++t) {
            const std::size_t lo = (t * total) / ntasks;
            const std::size_t hi = ((t + 1) * total) / ntasks;
            if (lo >= hi) continue;
            std::size_t r = (std::upper_bound(offset, offset + d + 1, lo) - offset) - 1;
            std::size_t out = lo;
            while (out < hi && r < d) {
                const std::size_t rend = std::min(hi, offset[r + 1]);
                if (rend > out) { std::fill(p + out, p + rend, rank_value[r]); out = rend; }
                ++r;
            }
        }
    };
    parallel_for_index(std::size_t(0), ntasks, std::size_t(1), job);
}

#if FYX_HAS_AVX512_CODE
FYX_DIAG_PUSH_SIMD
// AVX-512 count pass of the small-distinct kernel: one register holds eight
// encoded 64-bit keys, and each of the <= 16 distinct keys costs one compare
// plus one masked add per block -- no LUT reference, no dependent verify
// load.  Exactness comes from the total: a key the sample missed is counted
// by nothing, so sum(counts) != n declines the kernel instead of corrupting
// the output.  Counts are full-key compares, so the floating total order is
// the radix one (bit-exact keys, bijective decode).
template <class T>
FYX_TARGET_AVX512
inline bool simd_small_rank_count64(const T* p, std::size_t n,
                                    const typename RadixTraits<T>::Key* distinct,
                                    std::size_t d, std::uint32_t* counts) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    static_assert(sizeof(Key) == 8, "64-bit keys only");
    if (d == 0 || d > 16) return false;
    __m512i vkey[16];
    for (std::size_t r = 0; r < d; ++r)
        vkey[r] = _mm512_set1_epi64(static_cast<long long>(distinct[r]));
    const __m512i vsign = _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL));
    const __m512i vone  = _mm512_set1_epi32(1);
    __m512i acc[16];
    for (std::size_t r = 0; r < d; ++r) acc[r] = _mm512_setzero_si512();
    constexpr bool flip_all = std::is_floating_point<T>::value;
    constexpr bool flip_one = std::is_integral<T>::value && std::is_signed<T>::value;
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i k = _mm512_loadu_si512(reinterpret_cast<const void*>(p + i));
        if constexpr (flip_all) {
            const __m512i t = _mm512_srai_epi64(k, 63);
            k = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        } else if constexpr (flip_one) {
            k = _mm512_xor_si512(k, vsign);
        }
        for (std::size_t r = 0; r < d; ++r) {
            const __mmask8 m = _mm512_cmpeq_epi64_mask(k, vkey[r]);
            acc[r] = _mm512_mask_add_epi32(acc[r], m, acc[r], vone);
        }
    }
    for (; i < n; ++i) {  // tail: exact scalar match or decline
        const Key k = RT::encode(p[i]);
        std::size_t r = 0;
        while (r < d && distinct[r] != k) ++r;
        if (r == d) return false;
        ++counts[r];
    }
    for (std::size_t r = 0; r < d; ++r)
        counts[r] += static_cast<std::uint32_t>(_mm512_reduce_add_epi32(acc[r]));
    return true;
}

template <class T>
FYX_TARGET_AVX512
inline bool simd_small_rank_count32(const T* p, std::size_t n,
                                    const typename RadixTraits<T>::Key* distinct,
                                    std::size_t d, std::uint32_t* counts) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    static_assert(sizeof(Key) == 4, "32-bit keys only");
    if (d == 0 || d > 16) return false;
    __m512i vkey[16];
    for (std::size_t r = 0; r < d; ++r)
        vkey[r] = _mm512_set1_epi32(static_cast<int>(distinct[r]));
    const __m512i vsign = _mm512_set1_epi32(static_cast<int>(0x80000000u));
    const __m512i vone  = _mm512_set1_epi32(1);
    __m512i acc[16];
    for (std::size_t r = 0; r < d; ++r) acc[r] = _mm512_setzero_si512();
    constexpr bool flip_all = std::is_floating_point<T>::value;
    constexpr bool flip_one = std::is_integral<T>::value && std::is_signed<T>::value;
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512i k = _mm512_loadu_si512(reinterpret_cast<const void*>(p + i));
        if constexpr (flip_all) {
            const __m512i t = _mm512_srai_epi32(k, 31);
            k = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        } else if constexpr (flip_one) {
            k = _mm512_xor_si512(k, vsign);
        }
        for (std::size_t r = 0; r < d; ++r) {
            const __mmask16 m = _mm512_cmpeq_epi32_mask(k, vkey[r]);
            acc[r] = _mm512_mask_add_epi32(acc[r], m, acc[r], vone);
        }
    }
    for (; i < n; ++i) {  // tail: exact scalar match or decline
        const Key k = RT::encode(p[i]);
        std::size_t r = 0;
        while (r < d && distinct[r] != k) ++r;
        if (r == d) return false;
        ++counts[r];
    }
    for (std::size_t r = 0; r < d; ++r)
        counts[r] += static_cast<std::uint32_t>(_mm512_reduce_add_epi32(acc[r]));
    return true;
}
FYX_DIAG_POP_SIMD
#endif  // FYX_HAS_AVX512_CODE

// ---------------------------------------------------------------------------
// Small-distinct dense counter (shared fast path of the two rank kernels).
//
// When the sample's distinct encoded keys fit in one byte of rank (<= 255),
// every element's rank is one multiply and one *L1* load away: an 8-bit
// multiplicative hash over 256 slots plus a full-key verify load.  The
// 16-bit projection tables above are 32-128 KB and sit in L2, and on 1M
// double mod8 that difference is the whole gap to the SIMD sorters -- the
// counting pass was latency-bound on L2 references, not bandwidth-bound.
// The verify load stays: an unseen key that hashes into a claimed slot must
// abort the kernel (the caller's wider, exact paths take over), never be
// counted into a neighbour.
// ---------------------------------------------------------------------------
template <class T>
inline bool small_rank_count_fill_parallel(
    T* p, std::size_t n, bool descending,
    const typename RadixTraits<T>::Key* distinct, std::size_t d) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value) {
        (void)p; (void)n; (void)descending; (void)distinct; (void)d;
        return false;
    } else {
        if (d == 0 || d > 255 || n < kParallelThreshold || !parallel_available()) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;

        const std::size_t chunks = adaptive_parallel_chunks(n);
        if ((n + chunks - 1) / chunks >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return false;

        // Shared tail: cross-chunk offsets, one decode per rank, output-range
        // parallel fill.
        std::vector<std::uint32_t> local(chunks * d, 0);
        std::vector<unsigned char> miss(chunks, 0);
        auto finish_fill = [&]() {
            std::vector<std::size_t> offset(d + 1, 0);
            for (std::size_t out_rank = 0; out_rank < d; ++out_rank) {
                const std::size_t src_rank = descending ? (d - 1 - out_rank) : out_rank;
                std::size_t total = 0;
                for (std::size_t c = 0; c < chunks; ++c)
                    total += local[c * d + src_rank];
                offset[out_rank + 1] = offset[out_rank] + total;
            }
            std::vector<T> values(d);
            for (std::size_t r = 0; r < d; ++r)
                values[r] = RT::decode(distinct[descending ? (d - 1 - r) : r]);
            parallel_fill_by_ranks(p, offset.data(), values.data(), d);
        };

#if FYX_HAS_AVX512_CODE
        // One compare + one masked add per distinct key per vector block: no
        // LUT load, no dependent verify load, and no injective-hash search.
        // A key the sample missed is counted by nothing, so the total below
        // comes out short and the kernel declines.
        if (use_avx512() && d <= 16 && (sizeof(Key) == 8 || sizeof(Key) == 4)) {
            auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
                for (std::size_t c = c_lo; c < c_hi; ++c) {
                    const std::size_t lo = (c * n) / chunks;
                    const std::size_t hi = ((c + 1) * n) / chunks;
                    std::uint32_t* lc = local.data() + c * d;
                    bool ok = false;
                    if constexpr (sizeof(Key) == 8)
                        ok = simd_small_rank_count64(p + lo, hi - lo, distinct, d, lc);
                    else if constexpr (sizeof(Key) == 4)
                        ok = simd_small_rank_count32(p + lo, hi - lo, distinct, d, lc);
                    if (!ok) miss[c] = 1;
                }
            };
            parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);
            for (unsigned char v : miss) if (v) return false;
            std::size_t accounted = 0;
            for (std::size_t c = 0; c < chunks * d; ++c) accounted += local[c];
            if (accounted != n) return false;
            finish_fill();
            return true;
        }
#endif

        // Pick a hash that is injective on the sample.  d <= 255 leaves rank
        // 255 free as the "no rank" sentinel.
        constexpr std::uint64_t kMults[8] = {
            0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL,
            0x27D4EB2F165667C5ULL, 0x85EBCA77C2B2AE63ULL, 0x94D049BB133111EBULL,
            0xD6E8FEB86659FD93ULL, 0xBF58476D1CE4E5B9ULL,
        };
        unsigned char rank_of[256];
        std::uint64_t mult = 0;
        bool found = false;
        for (int t = 0; t < 8 && !found; ++t) {
            for (unsigned i = 0; i < 256; ++i) rank_of[i] = 255;
            bool ok = true;
            for (std::size_t r = 0; r < d; ++r) {
                const unsigned slot =
                    (unsigned)(((static_cast<std::uint64_t>(distinct[r])) * kMults[t]) >> 56);
                if (rank_of[slot] != 255) { ok = false; break; }
                rank_of[slot] = static_cast<unsigned char>(r);
            }
            if (ok) { mult = kMults[t]; found = true; }
        }
        if (!found) return false;

        auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                std::uint32_t* lc = local.data() + c * d;
                for (std::size_t i = lo; i < hi; ++i) {
                    const Key k = RT::encode(p[i]);
                    const unsigned char r =
                        rank_of[(unsigned)((static_cast<std::uint64_t>(k) * mult) >> 56)];
                    if (r == 255 || distinct[r] != k) { miss[c] = 1; break; }
                    ++lc[r];
                }
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);
        for (unsigned char v : miss) if (v) return false;

        // Every element must be accounted for exactly once.  The scalar count
        // miss-flags before it can be short, but the check costs d adds and
        // guards the invariant for both paths.
        std::size_t accounted = 0;
        for (std::size_t c = 0; c < chunks * d; ++c) accounted += local[c];
        if (accounted != n) return false;
        finish_fill();
        return true;
    }
}

template <class T>
inline bool try_radix_key_dense_prefix_count_sort_parallel(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T> || !std::is_floating_point<T>::value || std::is_same<T, bool>::value) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        if (n < kParallelThreshold || !parallel_available()) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr std::size_t Limit = kCountingClassLimit;
        constexpr std::size_t HashCap = 1024;
        constexpr std::size_t HashMask = HashCap - 1;
        constexpr unsigned short Sentinel = std::numeric_limits<unsigned short>::max();
        constexpr std::size_t MaxRange = std::size_t(1) << 20;

        std::array<Key, HashCap> sample_keys{};
        std::array<unsigned char, HashCap> sample_used{};
        std::vector<Key> distinct;
        distinct.reserve(Limit);
        const std::size_t s = std::min<std::size_t>(n, kCountingProbeLimit);
        for (std::size_t j = 0; j < s; ++j) {
            const std::size_t idx = (j * n) / s;
            const Key k = RT::encode(p[idx]);
            std::size_t h = low_card_hash_key(k) & HashMask;
            for (;;) {
                if (!sample_used[h]) {
                    if (distinct.size() >= Limit) return false;
                    sample_used[h] = 1;
                    sample_keys[h] = k;
                    distinct.push_back(k);
                    break;
                }
                if (sample_keys[h] == k) break;
                h = (h + 1) & HashMask;
            }
        }
        if (distinct.empty()) return false;
        std::sort(distinct.begin(), distinct.end());
        if (small_rank_count_fill_parallel(p, n, descending, distinct.data(), distinct.size()))
            return true;

        unsigned chosen_shift = 0;
        Key prefix_base = 0;
        std::size_t prefix_range = 0;
        auto try_shift = [&](unsigned sh) -> bool {
            Key mn = distinct.front() >> sh;
            Key mx = mn;
            Key prev = mn;
            for (std::size_t i = 1; i < distinct.size(); ++i) {
                const Key q = distinct[i] >> sh;
                if (q == prev) return false;
                prev = q;
                mx = q;
            }
            const unsigned long long range64 = static_cast<unsigned long long>(mx - mn) + 1ull;
            if (range64 == 0 || range64 > static_cast<unsigned long long>(MaxRange)) return false;
            chosen_shift = sh;
            prefix_base = mn;
            prefix_range = static_cast<std::size_t>(range64);
            return true;
        };

        bool mapped = false;
        if constexpr (sizeof(Key) == 4) {
            mapped = try_shift(16) || try_shift(12) || try_shift(8) || try_shift(20) || try_shift(0);
        } else {
            mapped = try_shift(48) || try_shift(44) || try_shift(40) || try_shift(52) || try_shift(36) || try_shift(32);
        }
        if (!mapped) return false;

        const std::size_t d = distinct.size();
        std::vector<unsigned short> rank_of(prefix_range, Sentinel);
        for (std::size_t r = 0; r < d; ++r) {
            const std::size_t idx = static_cast<std::size_t>((distinct[r] >> chosen_shift) - prefix_base);
            rank_of[idx] = static_cast<unsigned short>(r);
        }

        const std::size_t chunks = adaptive_parallel_chunks(n);
        if ((n + chunks - 1) / chunks >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return false;
        std::vector<std::uint32_t> local(chunks * d, 0);
        std::vector<unsigned char> miss(chunks, 0);
        auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                std::uint32_t* lc = local.data() + c * d;
                for (std::size_t i = lo; i < hi; ++i) {
                    const Key k = RT::encode(p[i]);
                    const Key q = k >> chosen_shift;
                    if (q < prefix_base) { miss[c] = 1; break; }
                    const std::size_t map_idx = static_cast<std::size_t>(q - prefix_base);
                    if (map_idx >= prefix_range) { miss[c] = 1; break; }
                    const unsigned short r = rank_of[map_idx];
                    if (r == Sentinel || distinct[r] != k) { miss[c] = 1; break; }
                    ++lc[r];
                }
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);
        for (unsigned char v : miss) if (v) return false;

        std::vector<std::size_t> offset(d + 1, 0);
        for (std::size_t out_rank = 0; out_rank < d; ++out_rank) {
            const std::size_t src_rank = descending ? (d - 1 - out_rank) : out_rank;
            std::size_t total = 0;
            for (std::size_t c = 0; c < chunks; ++c)
                total += local[c * d + src_rank];
            offset[out_rank + 1] = offset[out_rank] + total;
        }

        std::vector<T> values(d);
        for (std::size_t r = 0; r < d; ++r)
            values[r] = RT::decode(distinct[descending ? (d - 1 - r) : r]);
        parallel_fill_by_ranks(p, offset.data(), values.data(), d);
        return true;
    }
}

template <class T>
inline bool try_radix_key_rank16_count_sort_parallel(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        if (n < kParallelThreshold || !parallel_available()) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr std::size_t Limit = kCountingClassLimit;
        constexpr std::size_t HashCap = 1024;
        constexpr std::size_t HashMask = HashCap - 1;
        constexpr unsigned short Sentinel = std::numeric_limits<unsigned short>::max();

        std::array<Key, HashCap> sample_keys{};
        std::array<unsigned char, HashCap> sample_used{};
        std::vector<Key> distinct;
        distinct.reserve(Limit);
        const std::size_t s = std::min<std::size_t>(n, kCountingProbeLimit);
        for (std::size_t j = 0; j < s; ++j) {
            const std::size_t idx = (j * n) / s;
            const Key k = RT::encode(p[idx]);
            std::size_t h = low_card_hash_key(k) & HashMask;
            for (;;) {
                if (!sample_used[h]) {
                    if (distinct.size() >= Limit) return false;
                    sample_used[h] = 1;
                    sample_keys[h] = k;
                    distinct.push_back(k);
                    break;
                }
                if (sample_keys[h] == k) break;
                h = (h + 1) & HashMask;
            }
        }
        if (distinct.empty()) return false;
        if constexpr (std::is_integral<T>::value) {
            Key smn = distinct[0], smx = distinct[0];
            for (Key k : distinct) { if (k < smn) smn = k; if (smx < k) smx = k; }
            const Key sample_span = static_cast<Key>(smx - smn);
            if (sample_span != std::numeric_limits<Key>::max() &&
                static_cast<unsigned long long>(sample_span) + 1ull <= 65536ull)
                return false;
            if constexpr (sizeof(T) > 4) {
                if (distinct.size() <= 32) return false;
            }
        }
        std::sort(distinct.begin(), distinct.end());
        if (small_rank_count_fill_parallel(p, n, descending, distinct.data(), distinct.size()))
            return true;

        auto project = [](Key k, unsigned kind) noexcept -> unsigned short {
            if constexpr (sizeof(Key) <= 4) {
                const std::uint32_t x = static_cast<std::uint32_t>(k);
                switch (kind) {
                    case 0: return static_cast<unsigned short>(x >> 16);
                    case 1: return static_cast<unsigned short>(x);
                    case 2: return static_cast<unsigned short>(x >> 8);
                    case 3: return static_cast<unsigned short>((x >> 16) ^ x);
                    case 4: return static_cast<unsigned short>((x * 2654435761u) >> 16);
                    case 5: return static_cast<unsigned short>((x * 2246822519u) >> 16);
                    case 6: return static_cast<unsigned short>(((x ^ 0x9E3779B9u) * 3266489917u) >> 16);
                    case 7: return static_cast<unsigned short>(((x ^ 0x85EBCA6Bu) * 668265263u) >> 16);
                    case 8: return static_cast<unsigned short>(x >> 12);
                    case 9: return static_cast<unsigned short>(x >> 4);
                    default:return static_cast<unsigned short>((x >> 8) ^ x);
                }
            } else {
                const std::uint64_t x = static_cast<std::uint64_t>(k);
                switch (kind) {
                    case 0: return static_cast<unsigned short>(x >> 48);
                    case 1: return static_cast<unsigned short>(x >> 32);
                    case 2: return static_cast<unsigned short>(x >> 16);
                    case 3: return static_cast<unsigned short>(x);
                    case 4: return static_cast<unsigned short>(x >> 40);
                    case 5: return static_cast<unsigned short>(x >> 24);
                    case 6: return static_cast<unsigned short>(x >> 8);
                    case 7: return static_cast<unsigned short>((x >> 48) ^ (x >> 32) ^ (x >> 16) ^ x);
                    case 8: return static_cast<unsigned short>((x * 0x9E3779B97F4A7C15ULL) >> 48);
                    case 9: return static_cast<unsigned short>((x * 0xC2B2AE3D27D4EB4FULL) >> 48);
                    case 10:return static_cast<unsigned short>(((x ^ 0xD6E8FEB86659FD93ULL) * 0x94D049BB133111EBULL) >> 48);
                    case 11:return static_cast<unsigned short>(((x ^ 0x9E3779B97F4A7C15ULL) * 0xBF58476D1CE4E5B9ULL) >> 48);
                    default:return static_cast<unsigned short>((x >> 36) ^ (x >> 20) ^ (x >> 4));
                }
            }
        };

        std::vector<unsigned short> rank_of(65536, Sentinel);
        unsigned chosen = std::numeric_limits<unsigned>::max();
        constexpr unsigned Projections = (sizeof(Key) <= 4) ? 11u : 13u;
        for (unsigned kind = 0; kind < Projections; ++kind) {
            std::fill(rank_of.begin(), rank_of.end(), Sentinel);
            bool ok = true;
            for (std::size_t r = 0; r < distinct.size(); ++r) {
                const unsigned short q = project(distinct[r], kind);
                if (rank_of[q] != Sentinel) { ok = false; break; }
                rank_of[q] = static_cast<unsigned short>(r);
            }
            if (ok) { chosen = kind; break; }
        }
        if (chosen == std::numeric_limits<unsigned>::max()) return false;

        const std::size_t d = distinct.size();
        const std::size_t chunks = adaptive_parallel_chunks(n);
        if ((n + chunks - 1) / chunks >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return false;
        std::vector<std::uint32_t> local(chunks * d, 0);
        std::vector<unsigned char> miss(chunks, 0);
        auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                std::uint32_t* lc = local.data() + c * d;
                for (std::size_t i = lo; i < hi; ++i) {
                    const Key k = RT::encode(p[i]);
                    const unsigned short r = rank_of[project(k, chosen)];
                    if (r == Sentinel || distinct[r] != k) { miss[c] = 1; break; }
                    ++lc[r];
                }
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);
        for (unsigned char v : miss) if (v) return false;

        std::vector<std::size_t> offset(d + 1, 0);
        for (std::size_t out_rank = 0; out_rank < d; ++out_rank) {
            const std::size_t src_rank = descending ? (d - 1 - out_rank) : out_rank;
            std::size_t total = 0;
            for (std::size_t c = 0; c < chunks; ++c)
                total += local[c * d + src_rank];
            offset[out_rank + 1] = offset[out_rank] + total;
        }

        std::vector<T> values(d);
        for (std::size_t r = 0; r < d; ++r)
            values[r] = RT::decode(distinct[descending ? (d - 1 - r) : r]);
        parallel_fill_by_ranks(p, offset.data(), values.data(), d);
        return true;
    }
}

template <class T>
inline bool radix_key_sparse_probe_ok(T* p, std::size_t n) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value) {
        (void)p; (void)n;
        return false;
    } else {
        using RT = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr std::size_t Cap = 512;
        constexpr std::size_t Mask = Cap - 1;
        constexpr std::size_t Limit = kCountingClassLimit;
        std::array<Key, Cap> keys{};
        std::array<unsigned char, Cap> used{};
        std::size_t distinct = 0;
        const std::size_t s = std::min<std::size_t>(n, kCountingProbeLimit);
        for (std::size_t j = 0; j < s; ++j) {
            const std::size_t idx = (j * n) / s;
            const Key k = RT::encode(p[idx]);
            std::size_t h = low_card_hash_key(k) & Mask;
            for (;;) {
                if (!used[h]) {
                    if (distinct++ >= Limit) return false;
                    used[h] = 1;
                    keys[h] = k;
                    break;
                }
                if (keys[h] == k) break;
                h = (h + 1) & Mask;
            }
        }
        return true;
    }
}

template <class T>
inline bool try_radix_key_sparse_count_sort_parallel(T* p, std::size_t n, bool descending,
                                                     std::size_t distinct_guess = 0) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value) {
        (void)p; (void)n; (void)descending; (void)distinct_guess;
        return false;
    } else {
        if (n < kParallelThreshold || !parallel_available()) return false;
        if (n > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return false;
        // The carried sample already bounded the distinct count for us; only
        // probe when that evidence is missing.
        if (!(distinct_guess != 0 && distinct_guess <= kCountingClassLimit) &&
            !radix_key_sparse_probe_ok(p, n))
            return false;

        using RT = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr std::size_t Cap = 1024;
        constexpr std::size_t Limit = kCountingClassLimit;

        // Same small-table rule as the serial kernel: a couple dozen values
        // live in a 64-slot table that stays L1-resident beside the stream;
        // bail out at 48 distinct and let the 1024-slot kernel answer.
        const std::size_t cap2   = (distinct_guess != 0 && distinct_guess <= 24) ? 64 : Cap;
        const std::size_t mask2  = cap2 - 1;
        const std::size_t tlimit = (distinct_guess != 0 && distinct_guess <= 24) ? 48 : Limit;

        const std::size_t chunks = adaptive_parallel_chunks(n);
        std::vector<std::array<Key, Cap>> local_keys(chunks);
        std::vector<std::array<std::uint32_t, Cap>> local_counts(chunks);
        std::vector<std::array<unsigned char, Cap>> local_used(chunks);
        std::vector<std::vector<Key>> local_distinct(chunks);
        std::vector<unsigned char> overflow(chunks, 0);
        // vector value-initialisation already zeroed keys/counts/used; the
        // per-chunk fill(0) loops that used to run here were a full extra
        // 17 KB of writes per chunk for nothing.
        for (std::size_t c = 0; c < chunks; ++c) {
            local_distinct[c].reserve(tlimit);
        }

        auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                auto& keys = local_keys[c];
                auto& counts = local_counts[c];
                auto& used = local_used[c];
                auto& distinct = local_distinct[c];
                for (std::size_t i = lo; i < hi; ++i) {
                    const Key k = RT::encode(p[i]);
                    std::size_t h = low_card_hash_key(k) & mask2;
                    for (;;) {
                        if (!used[h]) {
                            if (distinct.size() >= tlimit) { overflow[c] = 1; goto done_chunk; }
                            used[h] = 1;
                            keys[h] = k;
                            counts[h] = 1;
                            distinct.push_back(k);
                            break;
                        }
                        if (keys[h] == k) { ++counts[h]; break; }
                        h = (h + 1) & mask2;
                    }
                }
            done_chunk: ;
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);
        for (unsigned char v : overflow) if (v) return false;

        std::array<Key, Cap> keys{};
        std::array<std::uint32_t, Cap> counts{};
        std::array<unsigned char, Cap> used{};
        std::vector<Key> distinct;
        distinct.reserve(tlimit);
        auto global_add = [&](Key k, std::uint32_t add) -> bool {
            std::size_t h = low_card_hash_key(k) & mask2;
            for (;;) {
                if (!used[h]) {
                    if (distinct.size() >= tlimit) return false;
                    used[h] = 1; keys[h] = k; counts[h] = add; distinct.push_back(k); return true;
                }
                if (keys[h] == k) { counts[h] += add; return true; }
                h = (h + 1) & mask2;
            }
        };
        auto local_lookup = [&](std::size_t c, Key k) noexcept -> std::uint32_t {
            std::size_t h = low_card_hash_key(k) & mask2;
            while (local_used[c][h]) {
                if (local_keys[c][h] == k) return local_counts[c][h];
                h = (h + 1) & mask2;
            }
            return 0;
        };
        for (std::size_t c = 0; c < chunks; ++c) {
            for (Key k : local_distinct[c])
                if (!global_add(k, local_lookup(c, k))) return false;
        }
        if (distinct.size() <= 1) return true;
        std::sort(distinct.begin(), distinct.end());

        auto global_lookup = [&](Key k) noexcept -> std::uint32_t {
            std::size_t h = low_card_hash_key(k) & mask2;
            while (used[h]) {
                if (keys[h] == k) return counts[h];
                h = (h + 1) & mask2;
            }
            return 0;
        };
        const std::size_t d = distinct.size();
        std::vector<Key> order(d);
        std::vector<std::size_t> offset(d + 1, 0);
        for (std::size_t r = 0; r < d; ++r) {
            const std::size_t src = descending ? (d - 1 - r) : r;
            order[r] = distinct[src];
            offset[r + 1] = offset[r] + global_lookup(order[r]);
        }

        auto fill_job = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t r = lo; r < hi; ++r) {
                const T v = RT::decode(order[r]);
                std::fill(p + offset[r], p + offset[r + 1], v);
            }
        };
        parallel_for_index(std::size_t(0), d, std::size_t(8), fill_job);
        return true;
    }
}

template <class T>
inline bool integer_sparse_probe_ok(T* p, std::size_t n) {
    if constexpr (!(std::is_integral<T>::value && !std::is_same<T, bool>::value && radix_supported_v<T>)) {
        (void)p; (void)n;
        return false;
    } else {
        using RT = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr std::size_t Cap = 512;
        constexpr std::size_t Mask = Cap - 1;
        std::array<Key, Cap> keys{};
        std::array<unsigned char, Cap> used{};
        std::size_t distinct = 0;
        const std::size_t s = std::min<std::size_t>(n, kCountingProbeLimit);
        for (std::size_t j = 0; j < s; ++j) {
            const std::size_t idx = (j * n) / s;
            const Key k = RT::encode(p[idx]);
            std::size_t h = low_card_hash_key(k) & Mask;
            for (;;) {
                if (!used[h]) {
                    if (distinct++ >= kCountingClassLimit) return false;
                    used[h] = 1;
                    keys[h] = k;
                    break;
                }
                if (keys[h] == k) break;
                h = (h + 1) & Mask;
            }
        }
        return true;
    }
}

template <class T>
inline bool try_integer_sparse_count_sort_parallel(T* p, std::size_t n, bool descending) {
    if constexpr (!(std::is_integral<T>::value && !std::is_same<T, bool>::value && radix_supported_v<T>)) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        if (n < kParallelThreshold || !parallel_available()) return false;
        if (!integer_sparse_probe_ok(p, n)) return false;

        using RT = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr std::size_t Cap = 1024;
        constexpr std::size_t Mask = Cap - 1;

        const std::size_t chunks = adaptive_parallel_chunks(n);
        std::vector<std::array<Key, Cap>> local_keys(chunks);
        std::vector<std::array<std::size_t, Cap>> local_counts(chunks);
        std::vector<std::array<unsigned char, Cap>> local_used(chunks);
        std::vector<std::vector<Key>> local_distinct(chunks);
        std::vector<unsigned char> overflow(chunks, 0);
        for (std::size_t c = 0; c < chunks; ++c) {
            local_used[c].fill(0);
            local_counts[c].fill(0);
            local_distinct[c].reserve(kCountingClassLimit);
        }

        auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                auto& keys = local_keys[c];
                auto& counts = local_counts[c];
                auto& used = local_used[c];
                auto& distinct = local_distinct[c];
                for (std::size_t i = lo; i < hi; ++i) {
                    const Key k = RT::encode(p[i]);
                    std::size_t h = low_card_hash_key(k) & Mask;
                    for (;;) {
                        if (!used[h]) {
                            if (distinct.size() >= kCountingClassLimit) { overflow[c] = 1; goto done_chunk; }
                            used[h] = 1;
                            keys[h] = k;
                            counts[h] = 1;
                            distinct.push_back(k);
                            break;
                        }
                        if (keys[h] == k) { ++counts[h]; break; }
                        h = (h + 1) & Mask;
                    }
                }
            done_chunk: ;
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);
        for (unsigned char v : overflow) if (v) return false;

        std::array<Key, Cap> keys{};
        std::array<std::size_t, Cap> counts{};
        std::array<unsigned char, Cap> used{};
        std::vector<Key> distinct;
        distinct.reserve(kCountingClassLimit);
        auto global_add = [&](Key k, std::size_t add) -> bool {
            std::size_t h = low_card_hash_key(k) & Mask;
            for (;;) {
                if (!used[h]) {
                    if (distinct.size() >= kCountingClassLimit) return false;
                    used[h] = 1; keys[h] = k; counts[h] = add; distinct.push_back(k); return true;
                }
                if (keys[h] == k) { counts[h] += add; return true; }
                h = (h + 1) & Mask;
            }
        };
        auto local_lookup = [&](std::size_t c, Key k) noexcept -> std::size_t {
            std::size_t h = low_card_hash_key(k) & Mask;
            while (local_used[c][h]) {
                if (local_keys[c][h] == k) return local_counts[c][h];
                h = (h + 1) & Mask;
            }
            return 0;
        };
        for (std::size_t c = 0; c < chunks; ++c) {
            for (Key k : local_distinct[c])
                if (!global_add(k, local_lookup(c, k))) return false;
        }
        if (distinct.size() <= 1) return true;
        std::sort(distinct.begin(), distinct.end());

        auto global_lookup = [&](Key k) noexcept -> std::size_t {
            std::size_t h = low_card_hash_key(k) & Mask;
            while (used[h]) {
                if (keys[h] == k) return counts[h];
                h = (h + 1) & Mask;
            }
            return 0;
        };
        const std::size_t d = distinct.size();
        std::vector<Key> order(d);
        std::vector<std::size_t> offset(d + 1, 0);
        for (std::size_t r = 0; r < d; ++r) {
            const std::size_t src = descending ? (d - 1 - r) : r;
            order[r] = distinct[src];
            offset[r + 1] = offset[r] + global_lookup(order[r]);
        }

        auto fill_job = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t r = lo; r < hi; ++r) {
                const T v = RT::decode(order[r]);
                std::fill(p + offset[r], p + offset[r + 1], v);
            }
        };
        parallel_for_index(std::size_t(0), d, std::size_t(8), fill_job);
        return true;
    }
}

template <class T>
FYX_NOINLINE inline bool try_integer_range_count_sort_parallel(T* p, std::size_t n, bool descending,
                                                               const SampleWindow<T>* win = nullptr) {
    if constexpr (!(radix_supported_v<T> && !std::is_same<T, bool>::value)) {
        (void)p; (void)n; (void)descending; (void)win;
        return false;
    } else {
        if (n < kParallelThreshold || !parallel_available()) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr std::size_t MaxParallelRange = 65536;

        // The profile already read a strided sample for us when it is here;
        // re-reading one inside the kernel was a second cold sweep of
        // cache-line-strided reads before any useful work.
        Key smn, smx;
        const bool have_window = win && win->valid;
        if (have_window) {
            smn = static_cast<Key>(win->lo);
            smx = static_cast<Key>(win->hi);
        } else {
            const std::size_t sample_n = std::min<std::size_t>(n, kProfileSampleLimit);
            smn = RT::encode(p[0]);
            smx = smn;
            for (std::size_t j = 1; j < sample_n; ++j) {
                const std::size_t idx = (j * n) / sample_n;
                const Key k = RT::encode(p[idx]);
                if (k < smn) smn = k;
                if (smx < k) smx = k;
            }
        }
        const Key sample_span = static_cast<Key>(smx - smn);
        if (sample_span == std::numeric_limits<Key>::max()) return false;
        const unsigned long long sample_range64 = static_cast<unsigned long long>(sample_span) + 1ull;
        const unsigned long long range_cap =
            (vqsort_preferred<T>() && vqsort_usable<T>(n))
                ? std::min<unsigned long long>(MaxParallelRange,
                      std::max<unsigned long long>(4096ull, n / 8))
                : MaxParallelRange;
        if (sample_range64 < 2ull || sample_range64 > range_cap) return false;

        // A wide range with few values in it is the sparse counter's shape,
        // not this one's.  Dense counting pays for every slot of the range in
        // every chunk -- 16 values spread over 61440 (1M int64, the matrix's
        // lowcard16 shape) means 61440 counters per chunk cleared, summed and
        // walked, and it measures 0.0021 s against the sparse counter's
        // 0.0013.  Below the crossover the dense kernel is the faster one and
        // keeps the work, and if the sample underestimated the value count the
        // sparse kernel declines after validating and control returns here.
        const std::size_t dhat = (win && win->distinct != 0)
            ? win->distinct
            : sample_distinct_keys(p, n, kCountingClassLimit);
        const unsigned long long spread = sizeof(T) <= 4 ? 64ull : 256ull;
        if (sample_range64 >= 32768ull && dhat <= kCountingClassLimit &&
            sample_range64 > spread * static_cast<unsigned long long>(dhat)) {
            if (try_radix_key_sparse_count_sort_parallel(p, n, descending, dhat)) return true;
        }

        const bool simd_rank_shape =
            dhat <= 16
#if FYX_HAS_AVX512_CODE
            && use_avx512()
#endif
            ;
        if (dhat <= 255 &&
            (simd_rank_shape ||
             (sample_range64 > 512ull &&
              sample_range64 > 4ull * static_cast<unsigned long long>(dhat)))) {
            constexpr std::size_t ProbeCap = 1024;
            std::array<Key, ProbeCap> probe_keys{};
            std::array<unsigned char, ProbeCap> probe_used{};
            Key dist_keys[256];
            std::size_t dcount = 0;
            bool overflow = false;
            const std::size_t ps = std::min<std::size_t>(n, kCountingProbeLimit);
            for (std::size_t j = 0; j < ps && !overflow; ++j) {
                const std::size_t idx = (j * n) / ps;
                const Key k = RT::encode(p[idx]);
                std::size_t h = low_card_hash_key(k) & (ProbeCap - 1);
                for (;;) {
                    if (!probe_used[h]) {
                        if (dcount >= 256) { overflow = true; break; }
                        probe_used[h] = 1;
                        probe_keys[h] = k;
                        dist_keys[dcount++] = k;
                        break;
                    }
                    if (probe_keys[h] == k) break;
                    h = (h + 1) & (ProbeCap - 1);
                }
            }
            if (!overflow && dcount > 0) {
                std::sort(dist_keys, dist_keys + dcount);
                if (small_rank_count_fill_parallel(p, n, descending, dist_keys, dcount))
                    return true;
            }
        }

        const std::size_t chunks = adaptive_parallel_chunks(n);
        if ((n + chunks - 1) / chunks >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return false;

        // The sample's range is a guess, not a measurement.  Spending a whole
        // sweep on the exact minimum and maximum costs one more read of the
        // array than the sort itself needs, so the count pass below checks
        // every key against the guess instead and abandons the sort if the
        // sample missed an extreme -- the serial range sort then takes it.
        const Key mn    = smn;
        const std::size_t range = static_cast<std::size_t>(sample_range64);
        std::atomic<unsigned> bail(0);

        std::vector<std::uint32_t> local(chunks * range, 0);
        auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                std::uint32_t* lc = local.data() + c * range;
                for (std::size_t i = lo; i < hi; ++i) {
                    const std::size_t d = static_cast<std::size_t>(RT::encode(p[i]) - mn);
                    if (d >= range) { bail.store(1, std::memory_order_relaxed); return; }
                    ++lc[d];
                }
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);
        if (bail.load(std::memory_order_relaxed)) return false;

        std::vector<std::size_t> offset(range + 1, 0);
        if (!descending) {
            for (std::size_t r = 0; r < range; ++r) {
                std::size_t total = 0;
                for (std::size_t c = 0; c < chunks; ++c)
                    total += local[c * range + r];
                offset[r + 1] = offset[r] + total;
            }
        } else {
            std::size_t sum = 0;
            for (std::size_t rr = range; rr-- > 0;) {
                std::size_t total = 0;
                for (std::size_t c = 0; c < chunks; ++c)
                    total += local[c * range + rr];
                offset[range - rr] = sum + total;
                sum += total;
            }
        }

        // Per-rank grain here, not the shared output-range split: with a
        // handful of ranks the per-rank split is one fill task per rank (the
        // counts are near-uniform for a dense window), and the split's
        // binary-search bookkeeping measured 16% slower on int32 mod8.
        auto fill_job = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t out_rank = lo; out_rank < hi; ++out_rank) {
                const std::size_t src_rank = descending ? (range - 1 - out_rank) : out_rank;
                const T v = RT::decode(static_cast<Key>(mn + static_cast<Key>(src_rank)));
                std::fill(p + offset[out_rank], p + offset[out_rank + 1], v);
            }
        };
        // Eight tasks per worker: with a range of sixteen the old grain of 64
        // made the whole fill one task, i.e. one core writing the array.
        const std::size_t fill_grain = std::max<std::size_t>(1, range / (chunks * 4));
        parallel_for_index(std::size_t(0), range, fill_grain, fill_job);
        return true;
    }
}

template <class Field, class T, class Comp>
inline bool try_trivial_field_count_sort_parallel(T* p, std::size_t n, Comp comp,
                                                  std::size_t offset_bytes) {
    using Key = typename RadixTraits<Field>::Key;
    if (n < kParallelThreshold || !parallel_available()) return false;
    bool descending = false;
    if (!trivial_field_candidate_order<Field>(p, n, comp, offset_bytes, descending)) return false;
    if (!trivial_field_low_cardinality_probe<Field>(p, n, offset_bytes)) return false;

    const std::size_t chunks = adaptive_parallel_chunks(n);
    std::vector<Key> local_mn(chunks), local_mx(chunks);
    auto minmax_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            const std::size_t lo = (c * n) / chunks;
            const std::size_t hi = ((c + 1) * n) / chunks;
            Key mn = load_trivial_field_key<Field>(p[lo], offset_bytes);
            Key mx = mn;
            for (std::size_t i = lo + 1; i < hi; ++i) {
                const Key k = load_trivial_field_key<Field>(p[i], offset_bytes);
                if (k < mn) mn = k;
                if (mx < k) mx = k;
            }
            local_mn[c] = mn;
            local_mx[c] = mx;
        }
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), minmax_job);
    Key mn = local_mn[0], mx = local_mx[0];
    for (std::size_t c = 1; c < chunks; ++c) {
        if (local_mn[c] < mn) mn = local_mn[c];
        if (mx < local_mx[c]) mx = local_mx[c];
    }
    const Key span = static_cast<Key>(mx - mn);
    if (span == std::numeric_limits<Key>::max()) return false;
    const unsigned long long range64 = static_cast<unsigned long long>(span) + 1ull;
    if (range64 > 65536ull) return false;
    const std::size_t range = static_cast<std::size_t>(range64);

    std::vector<std::size_t> local(chunks * range, 0), total(range, 0);
    auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            const std::size_t lo = (c * n) / chunks;
            const std::size_t hi = ((c + 1) * n) / chunks;
            std::size_t* lc = local.data() + c * range;
            for (std::size_t i = lo; i < hi; ++i)
                ++lc[static_cast<std::size_t>(load_trivial_field_key<Field>(p[i], offset_bytes) - mn)];
        }
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);

    std::size_t distinct = 0;
    for (std::size_t r = 0; r < range; ++r) {
        for (std::size_t c = 0; c < chunks; ++c) total[r] += local[c * range + r];
        distinct += total[r] != 0;
    }
    if (distinct == 0) return true;
    if (distinct > kCountingClassLimit) return false;

    std::vector<std::size_t> base(chunks * range, 0);
    std::size_t sum = 0;
    if (!descending) {
        for (std::size_t r = 0; r < range; ++r) {
            std::size_t run = sum;
            for (std::size_t c = 0; c < chunks; ++c) {
                base[c * range + r] = run;
                run += local[c * range + r];
            }
            sum += total[r];
        }
    } else {
        for (std::size_t rr = range; rr-- > 0;) {
            std::size_t run = sum;
            for (std::size_t c = 0; c < chunks; ++c) {
                base[c * range + rr] = run;
                run += local[c * range + rr];
            }
            sum += total[rr];
        }
    }

    ScratchLease<T> out_lease(n);
    if (!out_lease.valid()) return false;
    T* out = out_lease.get();
    auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            const std::size_t lo = (c * n) / chunks;
            const std::size_t hi = ((c + 1) * n) / chunks;
            std::size_t* pos = base.data() + c * range;
            for (std::size_t i = lo; i < hi; ++i) {
                const std::size_t r = static_cast<std::size_t>(load_trivial_field_key<Field>(p[i], offset_bytes) - mn);
                out[pos[r]++] = p[i];
            }
        }
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);
    auto copy_job = [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) p[i] = out[i];
    };
    parallel_for_index(std::size_t(0), n, kParallelThreshold, copy_job);
    return std::is_sorted(p, p + n, comp);
}

template <class T, class Comp>
inline bool try_trivial_prefix_key_count_sort_parallel(T* p, std::size_t n, Comp comp) {
    if constexpr (!std::is_trivially_copyable<T>::value || std::is_arithmetic<T>::value ||
                  std::is_same<T, std::string>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        constexpr std::size_t max_probe = sizeof(T) < 32 ? sizeof(T) : 32;
        for (std::size_t off = 0; off + 4 <= max_probe; off += 4) {
            if (try_trivial_field_count_sort_parallel<std::int32_t>(p, n, comp, off)) return true;
            if (try_trivial_field_count_sort_parallel<std::uint32_t>(p, n, comp, off)) return true;
        }
        for (std::size_t off = 0; off + 8 <= max_probe; off += 8) {
            if (try_trivial_field_count_sort_parallel<std::int64_t>(p, n, comp, off)) return true;
            if (try_trivial_field_count_sort_parallel<std::uint64_t>(p, n, comp, off)) return true;
        }
        return false;
    }
}

inline void string_msd_sort_bucket_range_parallel(std::string* data, std::string* aux,
                                                   const std::vector<std::size_t>& off,
                                                   unsigned lo_b, unsigned hi_b,
                                                   std::size_t depth) {
    if (hi_b <= lo_b) return;
    if (hi_b - lo_b <= 4 || !parallel_available()) {
        for (unsigned b = lo_b; b < hi_b; ++b) {
            const std::size_t lo = off[b], hi = off[b + 1];
            if (b != 0 && hi - lo > 1) string_msd_sort_rec(data + lo, aux + lo, hi - lo, depth);
        }
        return;
    }
    const unsigned mid = lo_b + (hi_b - lo_b) / 2;
    fork_join([&] { string_msd_sort_bucket_range_parallel(data, aux, off, lo_b, mid, depth); },
              [&] { string_msd_sort_bucket_range_parallel(data, aux, off, mid, hi_b, depth); });
}

inline bool string_msd_sort_parallel_default(std::string* p, std::size_t n, bool descending) {
        if (n < kParallelThreshold || !parallel_available()) return false;
        const std::size_t chunks = adaptive_parallel_chunks(n);
        std::vector<std::array<std::size_t, 257>> local(chunks);
        for (auto& a : local) a.fill(0);
        auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                auto& lc = local[c];
                for (std::size_t i = lo; i < hi; ++i) ++lc[string_msd_bucket(p[i], 0)];
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);

        std::vector<std::size_t> off(258, 0);
        unsigned nonzero = 0, only = 0;
        for (unsigned b = 0; b < 257; ++b) {
            for (std::size_t c = 0; c < chunks; ++c) off[b + 1] += local[c][b];
            if (off[b + 1] != 0) { ++nonzero; only = b; }
        }
        if (nonzero <= 1) {
            std::vector<std::string> tmp(n);
            if (only != 0) string_msd_sort_rec(p, tmp.data(), n, 1);
            if (descending) std::reverse(p, p + n);
            return true;
        }
        for (unsigned b = 1; b <= 257; ++b) off[b] += off[b - 1];

        std::vector<std::array<std::size_t, 257>> base(chunks);
        for (unsigned b = 0; b < 257; ++b) {
            std::size_t run = off[b];
            for (std::size_t c = 0; c < chunks; ++c) {
                base[c][b] = run;
                run += local[c][b];
            }
        }

        std::vector<std::string> tmp(n);
        auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                auto pos = base[c];
                for (std::size_t i = lo; i < hi; ++i) {
                    const unsigned b = string_msd_bucket(p[i], 0);
                    tmp[pos[b]++] = std::move(p[i]);
                }
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);

        string_msd_sort_bucket_range_parallel(tmp.data(), p, off, 0u, 257u, 1);
        auto copy_job = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t i = lo; i < hi; ++i) p[i] = std::move(tmp[i]);
        };
        parallel_for_index(std::size_t(0), n, kParallelThreshold, copy_job);
        if (descending) std::reverse(p, p + n);
        return true;
}

template <class T, class Comp>
inline bool try_string_msd_sort_parallel(T* p, std::size_t n, Comp comp, bool descending) {
    if constexpr (!std::is_same<T, std::string>::value) {
        (void)p; (void)n; (void)comp; (void)descending;
        return false;
    } else {
        if (!(is_ascending_v<Comp, T> || is_descending_v<Comp, T>)) return false;
        return string_msd_sort_parallel_default(p, n, descending);
    }
}


template <class T, unsigned PrefixBits, unsigned Bits>
inline bool try_serial_radix_high_prefix_key_sort_wide(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T> || PrefixBits == 0 || PrefixBits > 64 ||
                  Bits <= 8 || Bits > 13 || (PrefixBits % Bits) != 0) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        if constexpr (sizeof(Key) != 8 && sizeof(Key) != 4) {
            (void)p; (void)n; (void)descending;
            return false;
        } else {
            if (n < std::size_t(262144)) return false;
            if (!radix_high_prefix_probe<T, PrefixBits>(p, n)) return false;
            constexpr unsigned Passes = PrefixBits / Bits;
            constexpr unsigned FirstShift = unsigned(sizeof(Key) * 8) - PrefixBits;
            constexpr std::size_t Buckets = std::size_t(1) << Bits;

            ScratchLease<Key> lease(n * 2);
            if (!lease.valid()) return false;
            Key* a = lease.get();
            Key* b = a + n;
            Key* src = a;
            Key* dst = b;
            std::vector<std::uint32_t> count(std::size_t(Passes) * Buckets);
            std::vector<std::size_t> pos(Buckets);
            const bool can_stream = have_nt_stores();
            // Conflict/rank scatter while the keys stay in cache, write-
            // combining scatter once they leave it: see the table above
            // isa_avx512_scat::scatter32.  The AVX-512 path counts
            // destinations in 32 bits, so it also needs n to fit in them.
            const bool avx512_ok = use_avx512_conflict() &&
                                   n <= std::size_t(0xffffffffu) &&
                                   n * sizeof(Key) <= (std::size_t(1) << 23);
            std::vector<std::uint32_t> off32(
                (avx512_ok && sizeof(Key) == 4) ? Buckets : std::size_t(0));
            // Only worth fusing from three passes up.  Passes after the first
            // would otherwise re-read the scratch array, which the previous
            // scatter wrote and which has already left the cache, so one sweep
            // over the original keys beats several: double 1M, three passes,
            // 0.0042s -> 0.0018s.  With two passes the extra banks cost more
            // than the saved read (int32 8M: 0.0118s -> 0.0124s), so those keep
            // counting pass by pass.
            if constexpr (Passes >= 3)
                radix_count_all_value_passes_wide<T, Bits, Passes>(p, n, FirstShift, count.data());

            for (unsigned pi = 0; pi < Passes; ++pi) {
                const unsigned shift = FirstShift + pi * Bits;
                if constexpr (Passes < 3) {
                    if (pi == 0)
                        radix_count_value_pass_banked_wide<T, Bits>(p, n, shift, count.data());
                    else
                        radix_count_key_pass_banked_wide<Key, Bits>(
                            src, n, shift, count.data() + Buckets);
                }
                const std::uint32_t* pc = count.data() + std::size_t(pi) * Buckets;
                std::size_t run = 0;
                for (std::size_t d = 0; d < Buckets; ++d) {
                    pos[d] = run;
                    run += pc[d];
                }
                if (pi == 0) {
                    radix_scatter_wide_pass<T, T, Key, Bits>(
                        p, n, src, shift, pos.data(), off32.data(), avx512_ok, can_stream);
                } else if (pi + 1 == Passes) {
                    radix_scatter_wide_pass<T, Key, T, Bits>(
                        src, n, p, shift, pos.data(), off32.data(), avx512_ok, can_stream);
                } else {
                    radix_scatter_wide_pass<T, Key, Key, Bits>(
                        src, n, dst, shift, pos.data(), off32.data(), avx512_ok, can_stream);
                    Key* t = src; src = dst; dst = t;
                }
            }
            radix_sort_high_prefix_decoded_ties<T, PrefixBits>(p, n);
            if (descending) std::reverse(p, p + n);
            return true;
        }
    }
}

template <class T>
inline bool try_serial_radix_high_prefix_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        using Key = typename RadixTraits<T>::Key;
        if constexpr (sizeof(Key) != 8 && sizeof(Key) != 4) {
            (void)p; (void)n; (void)descending;
            return false;
        } else if constexpr (std::is_same<T, double>::value ||
                             (std::is_integral<T>::value && sizeof(T) == 8)) {
            // Same choice as the parallel path: see radix_choose_prefix_bits.
            if (n <= (std::size_t(1) << 21)) {
                const unsigned w = radix_choose_prefix_bits<T>(p, n, 24, 36);
                return w == 24
                    ? try_serial_radix_high_prefix_key_sort_wide<T, 24, 12>(p, n, descending)
                    : try_serial_radix_high_prefix_key_sort_wide<T, 36, 12>(p, n, descending);
            }
            const unsigned w = radix_choose_prefix_bits<T>(p, n, 26, 39);
            return w == 26
                ? try_serial_radix_high_prefix_key_sort_wide<T, 26, 13>(p, n, descending)
                : try_serial_radix_high_prefix_key_sort_wide<T, 39, 13>(p, n, descending);
        } else if constexpr (sizeof(Key) == 4) {
            // The helpers take their shifts from the key width now, so a 32-bit
            // key gets the same treatment a 64-bit one does: two passes over
            // the high bits, then the ties are finished bucket by bucket while
            // they are still in cache.  1M random int32: 0.0120s through the
            // 32-wide sort, 0.0090s here.  8M: 0.154s against 0.080s.
            if (n <= (std::size_t(1) << 21)) {
                const unsigned w = radix_choose_prefix_bits<T>(p, n, 24, 26);
                return w == 24
                    ? try_serial_radix_high_prefix_key_sort_wide<T, 24, 12>(p, n, descending)
                    : try_serial_radix_high_prefix_key_sort_wide<T, 26, 13>(p, n, descending);
            }
            return try_serial_radix_high_prefix_key_sort_wide<T, 26, 13>(p, n, descending);
        } else {
            (void)p; (void)n; (void)descending;
            return false;
        }
    }
}

template <class T>
inline bool try_serial_radix32_wide_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!(std::is_integral<T>::value && !std::is_same<T, bool>::value &&
                    radix_supported_v<T> && sizeof(T) == 4)) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        using RT = RadixTraits<T>;
        using Key = typename RT::Key;
        if (n < std::size_t(262144)) return false;
        ScratchLease<Key> lease(n * 2);
        if (!lease.valid()) return false;
        Key* a = lease.get();
        Key* b = a + n;
        auto pass_value = [&](unsigned bits, unsigned shift, Key* out) {
            const std::size_t buckets = std::size_t(1) << bits;
            const Key mask = Key(buckets - 1);
            std::vector<std::uint32_t> count(buckets, 0);
            std::vector<std::size_t> pos(buckets);
            for (std::size_t i = 0; i < n; ++i) {
                const Key k = RT::encode(p[i]);
                ++count[static_cast<std::size_t>((k >> shift) & mask)];
            }
            std::size_t run = 0;
            for (std::size_t d = 0; d < buckets; ++d) { pos[d] = run; run += count[d]; }
            for (std::size_t i = 0; i < n; ++i) {
                const Key k = RT::encode(p[i]);
                out[pos[static_cast<std::size_t>((k >> shift) & mask)]++] = k;
            }
        };
        auto pass_key = [&](const Key* in, unsigned bits, unsigned shift, auto emit) {
            const std::size_t buckets = std::size_t(1) << bits;
            const Key mask = Key(buckets - 1);
            std::vector<std::uint32_t> count(buckets, 0);
            std::vector<std::size_t> pos(buckets);
            for (std::size_t i = 0; i < n; ++i)
                ++count[static_cast<std::size_t>((in[i] >> shift) & mask)];
            std::size_t run = 0;
            for (std::size_t d = 0; d < buckets; ++d) { pos[d] = run; run += count[d]; }
            for (std::size_t i = 0; i < n; ++i) {
                const Key k = in[i];
                emit(pos[static_cast<std::size_t>((k >> shift) & mask)]++, k);
            }
        };
        pass_value(11, 0, a);
        pass_key(a, 11, 11, [&](std::size_t out, Key k) { b[out] = k; });
        pass_key(b, 10, 22, [&](std::size_t out, Key k) { p[out] = RT::decode(k); });
        if (descending) std::reverse(p, p + n);
        return true;
    }
}

#endif // FYX_ENABLE_PARALLEL

// ---------------------------------------------------------------------------
// Guarded recovery of default-order fast paths for custom comparators.
// A caller may spell the natural order as a lambda (`[](auto a, auto b){return
// a < b;}`), which intentionally does not match the compile-time std::less /
// fyx::less traits above.  We only use radix/count/MSD after a cheap sample
// proves the comparator is compatible with natural ascending/descending order,
// and we always finish with std::is_sorted(comp).  If the sample was fooled the
// array is still a permutation, so the caller can safely continue into the
// comparison sorter.
// ---------------------------------------------------------------------------
template <class T, class Comp, class Less>
inline int probe_guarded_default_order(T* p, std::size_t n, Comp comp, Less less) {
    if (n < 2) return 0;
    const std::size_t s = std::min<std::size_t>(n, kProfileSampleLimit);
    bool asc_ok = true, desc_ok = true, saw_order = false;

    auto check_pair = [&](const T& a, const T& b) {
        const bool ab = comp(a, b);
        const bool ba = comp(b, a);
        if (ab || ba) saw_order = true;
        const bool alb = less(a, b);
        const bool bla = less(b, a);
        if ((ab && !alb) || (ba && !bla)) asc_ok = false;
        if ((ab && !bla) || (ba && !alb)) desc_ok = false;
    };

    std::size_t prev = 0;
    for (std::size_t j = 1; j < s && (asc_ok || desc_ok); ++j) {
        const std::size_t idx = (j * (n - 1)) / (s - 1);
        check_pair(p[prev], p[idx]);
        check_pair(p[0], p[idx]);
        prev = idx;
    }

    if (!saw_order) return 0;
    if (asc_ok) return 1;
    if (desc_ok) return -1;
    return 0;
}

template <class T, class Comp>
inline int probe_guarded_radix_order(T* p, std::size_t n, Comp comp) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value ||
                  is_ascending_v<Comp, T> || is_descending_v<Comp, T>) {
        (void)p; (void)n; (void)comp;
        return 0;
    } else {
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        return probe_guarded_default_order(p, n, comp, [](const T& a, const T& b) {
            const Key ka = RT::encode(a);
            const Key kb = RT::encode(b);
            return ka < kb;
        });
    }
}

template <class T, class Comp>
inline bool try_guarded_string_value_count_sort(T* p, std::size_t n, Comp comp) {
    if constexpr (!std::is_same<T, std::string>::value ||
                  is_ascending_v<Comp, T> || is_descending_v<Comp, T>) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        const int dir = probe_guarded_default_order(p, n, comp, std::less<std::string>{});
        if (dir == 0) return false;

        std::unordered_map<std::string, std::size_t> counts;
        counts.reserve(kCountingClassLimit * 2);
        std::vector<std::string> distinct;
        distinct.reserve(kCountingClassLimit);
        for (std::size_t i = 0; i < n; ++i) {
            auto it = counts.find(p[i]);
            if (it == counts.end()) {
                if (distinct.size() >= kCountingClassLimit) return false;
                distinct.push_back(p[i]);
                counts.emplace(distinct.back(), std::size_t(1));
            } else {
                ++it->second;
            }
        }
        if (distinct.size() <= 1) return true;
        std::sort(distinct.begin(), distinct.end(), comp);
        std::size_t out = 0;
        for (const std::string& s : distinct) {
            const std::size_t c = counts.find(s)->second;
            std::fill_n(p + out, c, s);
            out += c;
        }
        return std::is_sorted(p, p + n, comp);
    }
}

template <class T, class Comp>
inline bool try_guarded_string_order_sort(T* p, std::size_t n, Comp comp,
                                          bool prefer_parallel) {
    if constexpr (!std::is_same<T, std::string>::value ||
                  is_ascending_v<Comp, T> || is_descending_v<Comp, T>) {
        (void)p; (void)n; (void)comp; (void)prefer_parallel;
        return false;
    } else {
        const int dir = probe_guarded_default_order(p, n, comp, std::less<std::string>{});
        if (dir == 0) return false;
        const bool descending = dir < 0;
        bool done = false;
#if FYX_ENABLE_PARALLEL
        if (prefer_parallel) done = string_msd_sort_parallel_default(p, n, descending);
#else
        (void)prefer_parallel;
#endif
        if (!done) done = string_msd_sort_default(p, n, descending);
        return done && std::is_sorted(p, p + n, comp);
    }
}

template <class T, class Comp>
inline bool try_guarded_radix_order_sort(T* p, std::size_t n, Comp comp,
                                         bool prefer_parallel, bool high_entropy) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value ||
                  is_ascending_v<Comp, T> || is_descending_v<Comp, T>) {
        (void)p; (void)n; (void)comp; (void)prefer_parallel; (void)high_entropy;
        return false;
    } else {
        const int dir = probe_guarded_radix_order(p, n, comp);
        if (dir == 0) return false;
        const bool descending = dir < 0;

        bool done = false;
        if (!high_entropy) {
#if FYX_ENABLE_PARALLEL
            if (prefer_parallel) {
                done = try_integer_range_count_sort_parallel(p, n, descending) ||
                       try_radix_key_dense_prefix_count_sort_parallel(p, n, descending) ||
                       try_radix_key_rank16_count_sort_parallel(p, n, descending) ||
                       try_integer_sparse_count_sort_parallel(p, n, descending) ||
                       try_radix_key_sparse_count_sort_parallel(p, n, descending);
            }
#endif
            if (!done) {
                done = try_radix_key_sparse_count_sort(p, n, descending) ||
                       try_integer_sparse_count_sort(p, n, descending) ||
                       try_integer_range_count_sort(p, n, descending);
            }
        }

        if (!done)
            done = try_radix_permutation_range_sort(p, n, descending);
#if FYX_ENABLE_PARALLEL
        if (!done && prefer_parallel && high_entropy)
            done = try_parallel_radix_high_prefix_sort(p, n, descending) ||
                   try_parallel_radix32_wide_sort(p, n, descending);
        if (!done && prefer_parallel)
            done = try_parallel_radix_sort(p, n, descending, high_entropy);
#else
        (void)prefer_parallel;
#endif

#if FYX_ENABLE_PARALLEL
        if (!done && high_entropy)
            done = try_serial_radix32_wide_sort(p, n, descending) ||
                   try_serial_radix_high_prefix_sort(p, n, descending);
#endif
        if (!done) {
            done = radix_sort(p, n);
            if (done && descending) std::reverse(p, p + n);
        }
        return done && std::is_sorted(p, p + n, comp);
    }
}

// ---------------------------------------------------------------------------
// Single-threaded best-kernel selection for a contiguous pointer range.
// `descending` is only consulted for radix-encodable types (where we may have
// sorted ascending and must reverse to honour a ">" comparator).  For the
// generic comparison path it is ignored and `comp` is used directly.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Ordered except for one stretch in the middle
// ---------------------------------------------------------------------------

// Partially sorted input that the patch repairs declined (many short runs,
// e.g. sqrt(n)-length sorted runs) used to fall to comparison pdqsort.  For a
// default-ordered std::string or a record whose comparator samples as an
// integer key field, the key radix sorts are several times cheaper than that.
// The patch repairs decline a "many independent sorted runs" input only after
// several full comparison passes (1M sqrt-length runs: ~20 ms for 16-byte
// records, ~60 ms for strings).  Probe 16 windows instead: if they hold many
// descents and almost every descent is *deep and wide* -- the element 8 past
// the break is still below the one 2 before it, which a lone displaced
// element never produces -- the input is run-structured and the key radix
// should go first.
template <class T, class Comp>
inline bool partial_runs_look_independent(const T* p, std::size_t n, Comp comp) {
    constexpr std::size_t kWindows = 16, kWin = 2048, kW = 8;
    if constexpr (!std::is_same<T, std::string>::value &&
                  !(std::is_trivially_copyable<T>::value && !std::is_arithmetic<T>::value)) {
        (void)p; (void)n; (void)comp;
        return false;
    }
    if (n < kWindows * kWin * 2) return false;
    std::size_t desc = 0, deep = 0;
    for (std::size_t w = 0; w < kWindows; ++w) {
        const std::size_t lo = 2 + (w * (n - kWin - kW - 4)) / kWindows;
        for (std::size_t i = lo; i < lo + kWin; ++i) {
            if (!comp(p[i], p[i - 1])) continue;
            ++desc;
            deep += comp(p[i + kW], p[i - 2]) ? 1u : 0u;
        }
    }
    return desc >= 24 && deep * 8 >= desc * 7;
}

template <class T, class Comp>
inline bool try_key_radix_for_partial(T* p, std::size_t n, Comp comp, bool par = false) {
    if constexpr (std::is_same<T, std::string>::value) {
        (void)par;
        return try_string_msd_sort(p, n, comp, is_descending_v<Comp, T>);
    } else if constexpr (std::is_trivially_copyable<T>::value && !std::is_arithmetic<T>::value) {
        return try_trivial_prefix_key_radix_sort(p, n, comp, par);
    } else {
        (void)p; (void)n; (void)comp; (void)par;
        return false;
    }
}

template <class T, class Comp>
inline void sort_st(T* p, std::size_t n, Comp comp, bool descending,
                    const InputProfile<T, Comp>* known_profile = nullptr);

/// Sorts a range that is ordered except for one contiguous stretch: a table
/// with a batch of new records appended, a log with an unflushed tail, a file
/// with one damaged region.  Those cost sort(middle) plus one merge pass, and
/// paying for the whole range instead is the difference between 0.005s and
/// 0.031s for a million int32 whose last tenth is shuffled -- which is what
/// radix spends, because radix cannot see order that stops part way.
///
/// Finding the stretch is free.  Both scans walk inwards from an end and stop
/// at the first inversion, so a range with no ordered head or tail costs two
/// comparisons, and the cost of a range that has one is the length of it.
#ifndef FYX_RUNS_KCHAIN
#define FYX_RUNS_KCHAIN 8
#endif

// Co-rank: how many of the first k merged outputs come from A (ties: A
// first, i.e. a stable merge of A then B).
template <class T, class Less>
inline std::size_t merge_corank(const T* A, std::size_t na, const T* B, std::size_t nb, std::size_t k,
                                Less& before) {
    std::size_t lo = k > nb ? k - nb : 0, hi = std::min(k, na);
    while (lo < hi) {
        const std::size_t i = lo + (hi - lo) / 2;
        if (before(B[k - i - 1], A[i])) hi = i;
        else lo = i + 1;
    }
    return lo;
}

// Stable merge of sorted A and B into out with K independent branchless
// chains (output split by co-rank).  A two-way merge is latency-bound --
// index -> load -> compare -> index, ~10 cycles a record -- so K chains in
// one loop overlap K of those (measured 2x at 100k 16-byte records).  The
// select is an address mask: GCC turns a value ternary on a record back into
// a branch, which mispredicts half the time on interleaved runs.
template <unsigned K, class T, class Less>
inline void merge_runs_kchain(const T* A, std::size_t na, const T* B, std::size_t nb, T* out, Less& before) {
    const std::size_t n = na + nb;
    std::size_t i[K], j[K], ie[K], je[K], o[K];
    std::size_t prev_i = 0, prev_k = 0;
    for (unsigned c = 0; c < K; ++c) {
        const std::size_t k1 = (n * (c + 1)) / K;
        const std::size_t i1 = c + 1 == K ? na : merge_corank(A, na, B, nb, k1, before);
        i[c] = prev_i; j[c] = prev_k - prev_i; o[c] = prev_k;
        ie[c] = i1; je[c] = k1 - i1;
        prev_i = i1; prev_k = k1;
    }
    while (true) {
        std::size_t h = ~std::size_t(0);
        for (unsigned c = 0; c < K; ++c) h = std::min(h, std::min(ie[c] - i[c], je[c] - j[c]));
        if (h == 0) break;
        for (std::size_t s = 0; s < h; ++s) {
            for (unsigned c = 0; c < K; ++c) {
                const T* pa = A + i[c];
                const T* pb = B + j[c];
                const std::size_t tb = before(*pb, *pa) ? 1u : 0u;
                const std::uintptr_t mk = std::uintptr_t(0) - static_cast<std::uintptr_t>(tb);
                const std::uintptr_t ua = reinterpret_cast<std::uintptr_t>(pa);
                const std::uintptr_t ub = reinterpret_cast<std::uintptr_t>(pb);
                out[o[c]++] = *reinterpret_cast<const T*>(ua ^ ((ua ^ ub) & mk));
                j[c] += tb;
                i[c] += 1 - tb;
            }
        }
    }
    for (unsigned c = 0; c < K; ++c) {
        std::size_t x = i[c], y = j[c], w = o[c];
        while (x < ie[c] && y < je[c]) {
            if (before(B[y], A[x])) out[w++] = B[y++];
            else                    out[w++] = A[x++];
        }
        while (x < ie[c]) out[w++] = A[x++];
        while (y < je[c]) out[w++] = B[y++];
    }
}

/// Records (trivially copyable, comparator-ordered) made of at most nine
/// ascending runs -- rotated sorted ranges, concatenated sorted batches.
///  * one break and the second run wholly <= the first's head: rotated back
///    through a buffer the size of the smaller run;
///  * one break otherwise: trimmed (elements already final stay) and merged;
///  * up to eight breaks: merged pairwise, ping-ponging with one buffer.
/// Merges are co-ranked parallel merges when `par`, else K-chain merges.
/// The break scan is a vectorized block reduction.  Gates, all before
/// anything moves: 64 sampled adjacent pairs ordered, at most eight
/// descents in the 64-point coarse sample, and every break a real run
/// boundary -- an isolated dip or spike (far swaps, nearly sorted) belongs
/// to the repair paths and is refused at the first break.
template <class T, class Comp>
inline bool try_record_runs_merge(T* p, std::size_t n, Comp comp, bool par) {
    if constexpr (!std::is_trivially_copyable<T>::value || std::is_arithmetic<T>::value) {
        (void)p; (void)n; (void)comp; (void)par;
        return false;
    } else {
        if (n < 4096) return false;
#if FYX_ENABLE_PARALLEL
        par = par && n >= (std::size_t(1) << 17) && parallel_available();
#else
        par = false;
#endif
        auto before = adaptive_order<T>(comp);
        constexpr unsigned kMaxBreaks = 8;
        unsigned coarse = 0;
        for (std::size_t j = 1; j <= 64; ++j) {
            const std::size_t i = (j * (n - 1)) / 65 + 1;
            if (before(p[i], p[i - 1])) return false;
            const std::size_t a = ((j - 1) * (n - 1)) / 64, b = (j * (n - 1)) / 64;
            if (before(p[b], p[a]) && ++coarse > kMaxBreaks) return false;
        }
        std::size_t bounds[kMaxBreaks + 2];
        unsigned nb = 0;
        bounds[0] = 0;
        constexpr std::size_t kBlk = 2048;
        for (std::size_t b = 1; b < n; b += kBlk) {
            const std::size_t e = std::min(n, b + kBlk);
            if (!any_adjacent_pair(p, b, e, before, true)) continue;
            for (std::size_t i = b; i < e; ++i) {
                if (!before(p[i], p[i - 1])) continue;
                if (nb == kMaxBreaks) return false;
                // Dip (p[i] alone below its left neighbour) or spike (p[i-1]
                // alone above both sides): not a boundary between runs.
                if (i + 1 < n && !before(p[i + 1], p[i - 1])) return false;
                if (i >= 2 && !before(p[i], p[i - 2])) return false;
                bounds[++nb] = i;
            }
        }
        if (nb == 0) return true;
        if (nb == 1) {
            const std::size_t m = bounds[1];
            if (!before(p[0], p[n - 1])) {
                // Wrap: [m, n) <= p[0] <= [0, m) -> rotate.
                const std::size_t na = m, nr = n - m;
                ScratchLease<T> lease(std::min(na, nr));
                if (!lease.valid()) return false;
                T* t = lease.get();
                if (na <= nr) {
                    std::memcpy(static_cast<void*>(t), static_cast<const void*>(p), na * sizeof(T));
                    std::memmove(static_cast<void*>(p), static_cast<const void*>(p + m), nr * sizeof(T));
                    std::memcpy(static_cast<void*>(p + nr), static_cast<const void*>(t), na * sizeof(T));
                } else {
                    std::memcpy(static_cast<void*>(t), static_cast<const void*>(p + m), nr * sizeof(T));
                    std::memmove(static_cast<void*>(p + nr), static_cast<const void*>(p), na * sizeof(T));
                    std::memcpy(static_cast<void*>(p), static_cast<const void*>(t), nr * sizeof(T));
                }
                return true;
            }
            // Trim: [0, a) <= p[m] and [e, n) >= p[m-1] are already final.
            const std::size_t a = static_cast<std::size_t>(std::upper_bound(p, p + m, p[m], before) - p);
            const std::size_t e = static_cast<std::size_t>(std::lower_bound(p + m, p + n, p[m - 1], before) - p);
            ScratchLease<T> lease(e - a);
            if (!lease.valid()) return false;
            T* t = lease.get();
#if FYX_ENABLE_PARALLEL
            if (par && e - a >= (std::size_t(1) << 17)) {
                T* buf = t - a;
                parallel_merge_to_buffer_rec(p, a, m, m, e, buf, a, before);
                auto copy_job = [&](std::size_t lo, std::size_t hi) {
                    std::memcpy(static_cast<void*>(p + lo), static_cast<const void*>(buf + lo),
                                (hi - lo) * sizeof(T));
                };
                parallel_for_index(a, e, std::max<std::size_t>(std::size_t(1) << 16, (e - a) / 8), copy_job);
                return true;
            }
#endif
            merge_runs_kchain<FYX_RUNS_KCHAIN>(p + a, m - a, p + m, e - m, t, before);
            std::memcpy(static_cast<void*>(p + a), static_cast<const void*>(t), (e - a) * sizeof(T));
            return true;
        }
        bounds[nb + 1] = n;
        unsigned r = nb + 1;
        ScratchLease<T> lease(n);
        if (!lease.valid()) return false;
        T* src = p;
        T* dst = lease.get();
        while (r > 1) {
            unsigned w = 0;
            for (unsigned i = 0; i < r; i += 2) {
                const std::size_t lo = bounds[i];
                if (i + 1 < r) {
                    const std::size_t mid = bounds[i + 1], hi = bounds[i + 2];
#if FYX_ENABLE_PARALLEL
                    if (par) parallel_merge_to_buffer_rec(src, lo, mid, mid, hi, dst, lo, before);
                    else
#endif
                    merge_runs_kchain<FYX_RUNS_KCHAIN>(src + lo, mid - lo, src + mid, hi - mid, dst + lo, before);
                } else {
                    std::memcpy(static_cast<void*>(dst + lo), static_cast<const void*>(src + lo),
                                (bounds[i + 1] - lo) * sizeof(T));
                }
                bounds[w++] = lo;
            }
            bounds[w] = n;
            r = w;
            T* t = src; src = dst; dst = t;
        }
        if (src != p) {
#if FYX_ENABLE_PARALLEL
            if (par) {
                auto copy_job = [&](std::size_t lo, std::size_t hi) {
                    std::memcpy(static_cast<void*>(p + lo), static_cast<const void*>(src + lo),
                                (hi - lo) * sizeof(T));
                };
                parallel_for_index(std::size_t(0), n,
                                   std::max<std::size_t>(std::size_t(1) << 16, n / 8), copy_job);
                return true;
            }
#endif
            std::memcpy(static_cast<void*>(p), static_cast<const void*>(src), n * sizeof(T));
        }
        return true;
    }
}

} // namespace detail
template <class T, class Comp>
inline void sort_pointer_core_impl(T* p, std::size_t n, Comp comp, const Options& o);
namespace detail {

template <class T, class Comp>
inline bool try_sorted_affix_sort(T* p, std::size_t n, Comp comp, bool par = false) {
#if !FYX_ENABLE_ADAPTIVE_WEAPONS
    (void)p; (void)n; (void)comp; (void)par;
    return false;
#else
    if (n < 8192) return false;
    if constexpr (!std::is_move_constructible<T>::value ||
                  !std::is_move_assignable<T>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        auto before = adaptive_order<T>(comp);
        // Affix scans: vectorized block tests, then locate inside the block.
        constexpr std::size_t kBlk = 1024;
        std::size_t head = 1;
        {
            OrderBreakHint& h = order_break_hint();
            if (h.p == static_cast<const void*>(p) && h.n == n && h.head > 0 && h.head < n &&
                before(p[h.head], p[h.head - 1]))
                head = h.head;
            h.p = nullptr;
        }
        while (head < n) {
            const std::size_t e = std::min(n, head + kBlk);
            if (any_adjacent_pair(p, head, e, before, true)) break;
            head = e;
        }
        while (head < n && !before(p[head], p[head - 1])) ++head;
        if (head == n) return true;                  // ordered already
        std::size_t tail = n - 1;
        while (tail > head) {
            const std::size_t b = tail > head + kBlk ? tail - kBlk : head + 1;
            if (b >= tail || any_adjacent_pair(p, b, tail + 1, before, true)) break;
            tail = b;
        }
        while (tail > head && !before(p[tail], p[tail - 1])) --tail;
        // No room between the two affixes, or so little order that sorting the
        // middle and merging costs about as much as sorting the whole range.
        if (tail <= head || (tail - head) * 2u > n) return false;

        // Everything that can fail fails before anything is moved, so a range
        // this weapon declines is left exactly as it was.
        const std::size_t need1 = tail < n ? std::min(tail - head, n - tail) : std::size_t(0);
        const std::size_t need2 = std::min(head, n - head);
        std::unique_ptr<ScratchLease<T>> lease;
        T* buf = nullptr;
#if FYX_ENABLE_PARALLEL
        // Parallel callers merge the big ordered head with a co-ranked
        // parallel merge into a full-size buffer (one lease, sized for it).
        const bool par_merge = std::is_trivially_copyable<T>::value && par &&
                               n >= (std::size_t(1) << 17) && parallel_available();
#else
        (void)par;
        const bool par_merge = false;
#endif
        if constexpr (std::is_trivially_copyable<T>::value) {
            lease.reset(new ScratchLease<T>(par_merge ? n : (need1 > need2 ? need1 : need2)));
            if (!lease->valid()) return false;
            buf = lease->get();
        }
        // `descending` must follow comp: radix-key kernels sort ascending and
        // reverse for a ">" comparator (with NaN the order differs from comp).
        if constexpr (std::is_arithmetic<T>::value) {
            sort_st(p + head, tail - head, comp, is_descending_v<Comp, T>);
        } else {
            // Records: the full dispatcher (sort_st has no record kernels;
            // 10k random 16-byte records: 0.10 ms there vs 0.51 ms here).
            Options mo;
            mo.parallel = par ? Tri::On : Tri::Off;
            sort_pointer_core_impl(p + head, tail - head, comp, mo);
        }
        if (tail < n) merge_adjacent_runs(p, head, tail, n, buf, before);
#if FYX_ENABLE_PARALLEL
        if constexpr (std::is_trivially_copyable<T>::value) {
            if (par_merge) {
                if (head > 0 && before(p[head], p[head - 1])) {
                    parallel_merge_to_buffer_rec(p, std::size_t(0), head, head, n, buf,
                                                 std::size_t(0), before);
                    auto copy_job = [&](std::size_t lo, std::size_t hi) {
                        std::memcpy(static_cast<void*>(p + lo), static_cast<const void*>(buf + lo),
                                    (hi - lo) * sizeof(T));
                    };
                    parallel_for_index(std::size_t(0), n,
                                       std::max<std::size_t>(std::size_t(1) << 16, n / 8), copy_job);
                }
                return true;
            }
        }
#endif
        merge_adjacent_runs(p, 0, head, n, buf, before);
        return true;
    }
#endif
}

// ---------------------------------------------------------------------------
// Vectorised quicksort driver (parts/10b_vsort.hpp holds the kernel)
// ---------------------------------------------------------------------------

/// Vectorised cousin of the monotone scan: returns the position of the first
/// pair (i-1, i) that violates the target order, or `hi` when [start, hi) is
/// monotone.  The AVX-512 implementation is isolated behind a baseline wrapper.
#if FYX_HAS_AVX512_CODE
FYX_DIAG_PUSH_SIMD
template <class T>
FYX_TARGET_AVX512 inline std::size_t radix_key_find_break_avx512(
        const T* p, std::size_t start, std::size_t hi,
        typename RadixTraits<T>::Key prev_key,
        const bool want_up, const bool descending) {
    using RT = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr unsigned lanes = (sizeof(Key) == 8) ? 8u : 16u;
    __m512i prev_vec = (sizeof(Key) == 8)
        ? _mm512_set1_epi64(static_cast<long long>(prev_key))
        : _mm512_set1_epi32(static_cast<int>(prev_key));
    const __m512i vsign = (sizeof(Key) == 8)
        ? _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL))
        : _mm512_set1_epi32(static_cast<int>(0x80000000u));
    std::size_t i = start;
    for (; i + lanes <= hi; i += lanes) {
        const __m512i k = _mm512_loadu_si512(reinterpret_cast<const void*>(p + i));
        __m512i e;
        if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_unsigned_v<T>) e = k;
            else e = _mm512_xor_si512(k, vsign);
        } else if constexpr (sizeof(Key) == 8) {
            const __m512i t = _mm512_srai_epi64(k, 63);
            e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        } else {
            const __m512i t = _mm512_srai_epi32(k, 31);
            e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        }
        const __m512i shifted = (sizeof(Key) == 8)
            ? _mm512_alignr_epi64(e, prev_vec, 7)
            : _mm512_alignr_epi32(e, prev_vec, 15);
        const unsigned bad = (want_up != descending)
            ? (sizeof(Key) == 8 ? _mm512_cmpgt_epu64_mask(shifted, e)
                                : _mm512_cmpgt_epu32_mask(shifted, e))
            : (sizeof(Key) == 8 ? _mm512_cmplt_epu64_mask(shifted, e)
                                : _mm512_cmplt_epu32_mask(shifted, e));
        if (bad) return i + static_cast<std::size_t>(__builtin_ctz(bad));
        prev_vec = e;
    }
    if (i > start) prev_key = RT::encode(p[i - 1]);
    for (; i < hi; ++i) {
        const Key cur = RT::encode(p[i]);
        const bool bad = want_up ? (descending ? (prev_key < cur) : (cur < prev_key))
                                 : (descending ? (cur < prev_key) : (prev_key < cur));
        if (bad) return i;
        prev_key = cur;
    }
    return hi;
}
FYX_DIAG_POP_SIMD
#endif

template <class T>
inline std::size_t radix_key_find_break(const T* p, std::size_t start, std::size_t hi,
                                        typename RadixTraits<T>::Key prev_key,
                                        const bool want_up, const bool descending) {
#if FYX_HAS_AVX512_CODE
    if (use_avx512())
        return radix_key_find_break_avx512(p, start, hi, prev_key, want_up, descending);
#endif
    using Key = typename RadixTraits<T>::Key;
    Key prev = prev_key;
    for (std::size_t i = start; i < hi; ++i) {
        const Key cur = RadixTraits<T>::encode(p[i]);
        const bool bad = want_up ? (descending ? (prev < cur) : (cur < prev))
                                 : (descending ? (cur < prev) : (prev < cur));
        if (bad) return i;
        prev = cur;
    }
    return hi;
}

#if FYX_ENABLE_PARALLEL
/// Parallel driver for the one-break proof: chunk the range across the pool,
/// each chunk reports (internal monotone, seam ok) with the vectorised scans,
/// and the seam algebra accepts exactly one break total -- either a bad seam
/// (the rotation point coincides with a chunk boundary) or a single
/// internally-broken chunk (refined to the break position with one extra
/// vectorised scan of its two halves).  Wrap condition and rotate repair as in
/// the serial proof; the move itself is a three-step buffered copy spread over
/// the pool (copy the short side out, memmove the long side, copy back), so
/// the whole shape costs three bandwidth passes instead of a full sort.
template <class T>
inline bool try_one_break_rotate_parallel(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T>) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        constexpr std::size_t kOneBreakParMinN = std::size_t(1) << 18;
        if (n < kOneBreakParMinN || !parallel_available()) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        const bool want_up = !descending;

        ThreadPool& pool = global_pool();
        std::size_t chunks = std::max<std::size_t>(2, static_cast<std::size_t>(pool.nworkers()) * 4);
        if (chunks * 4096 > n) chunks = std::max<std::size_t>(2, n / 4096);

        std::vector<unsigned char> internal_bad(chunks, 0), seam_bad(chunks, 0);
        std::atomic<unsigned> bad_total{0};
        std::atomic<bool>     give_up{false};

        auto job = [&](std::size_t jc_lo, std::size_t jc_hi) {
            for (std::size_t c = jc_lo; c < jc_hi; ++c) {
                if (give_up.load(std::memory_order_relaxed)) return;
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                if (hi <= lo) continue;
                if (c > 0) {
                    const Key a = RT::encode(p[lo - 1]);
                    const Key b = RT::encode(p[lo]);
                    const bool bad = want_up ? (b < a) : (a < b);
                    if (bad) {
                        seam_bad[c] = 1;
                        if (bad_total.fetch_add(1, std::memory_order_relaxed) + 1 > 1) {
                            give_up.store(true, std::memory_order_relaxed);
                            return;
                        }
                    }
                }
                const std::size_t start =
                    (c == 0) ? lo + 1 : (seam_bad[c] ? lo + 1 : lo);
                if (start >= hi) continue;
                if (!radix_key_monotone_scan(p, start, hi, RT::encode(p[start - 1]),
                                             want_up, descending)) {
                    internal_bad[c] = 1;
                    if (bad_total.fetch_add(1, std::memory_order_relaxed) + 1 > 1) {
                        give_up.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), job);
        if (give_up.load(std::memory_order_relaxed)) return false;

        std::size_t brk = n;
        unsigned    total = 0;
        for (std::size_t c = 0; c < chunks; ++c) {
            const std::size_t lo = (c * n) / chunks;
            const std::size_t hi = ((c + 1) * n) / chunks;
            if (seam_bad[c])    { ++total; brk = lo; }
            if (internal_bad[c]) {
                ++total;
                // Refine inside the broken chunk: first violating pair, then
                // prove the rest of the chunk is one more monotone stretch.
                const std::size_t b = radix_key_find_break(p, lo + 1, hi,
                                                           RT::encode(p[lo]),
                                                           want_up, descending);
                if (b == hi || !radix_key_monotone_scan(p, b + 1, hi,
                                                        RT::encode(p[b]),
                                                        want_up, descending))
                    return false;
                brk = b;
            }
        }
        if (total != 1) return false;

        const Key front = RT::encode(p[0]);
        const Key back  = RT::encode(p[n - 1]);
        const bool cyclic = want_up ? (back <= front) : (back >= front);
        if (!cyclic) return false;

        // Buffered rotate spread over the pool.  A short-side buffer with
        // in-place memmove chunks is NOT safe here: with chunk length g and
        // shift s, chunk k's write window [s+kg, s+(k+1)g) overlaps chunk
        // k + ceil(s/g)'s read window whenever s is not a multiple of g, and
        // the chunks run concurrently -- the source is overwritten before it
        // is read (measured: perm-breaking corruption at 1M).  So stage the
        // whole array and write each result chunk as two straight memcpys
        // (the source index i + brk wraps at most once per chunk).  Falls
        // back to the serial std::rotate when the arena cannot host a copy.
        ScratchLease<T> lease(n);
        if (!lease.valid()) { std::rotate(p, p + brk, p + n); return true; }
        T* buf = lease.get();
        const std::size_t grain = std::max<std::size_t>(std::size_t(1) << 16,
                                                        n / 8);
        auto copy_out = [&](std::size_t a, std::size_t b) {
            std::memcpy(buf + a, p + a, (b - a) * sizeof(T));
        };
        parallel_for_index(std::size_t(0), n, grain, copy_out);
        const std::size_t split = n - brk;   // i >= split reads buf[i + brk - n]
        auto write_back = [&](std::size_t a, std::size_t b) {
            if (a >= split) {
                std::memcpy(p + a, buf + (a + brk - n), (b - a) * sizeof(T));
            } else if (b <= split) {
                std::memcpy(p + a, buf + (a + brk), (b - a) * sizeof(T));
            } else {
                std::memcpy(p + a, buf + (a + brk), (split - a) * sizeof(T));
                std::memcpy(p + split, buf, (b - split) * sizeof(T));
            }
        };
        parallel_for_index(std::size_t(0), n, grain, write_back);
        return true;
    }
}
#endif  // FYX_ENABLE_PARALLEL


#if FYX_ENABLE_PARALLEL
/// Partition at the top, then hand the two sides to the pool.  The partition
/// itself is sequential -- a parallel partition needs a second array and a
/// prefix-sum round, which costs more than it saves at two workers -- so the
/// speedup ceiling is Amdahl over the first `depth` levels.  With two workers
/// and three levels that is ~1.8x, which is what it measures.
template <class T>
inline void vqsort_parallel_rec(T* p, std::size_t n, int budget, unsigned depth) {
    // Below this a task costs more than the sort it carries.
    constexpr std::size_t kMinParTask = std::size_t(1) << 16;
    while (true) {
        if (depth == 0 || budget <= 0 || n < kMinParTask) {
            vqsort_serial_budget(p, n, budget);
            return;
        }
        const VqStep st = vqsort_partition_step(p, n);
        --budget;
        if (st.left_done && st.right_done) return;
        if (st.left_done)  { p += st.split; n -= st.split; continue; }
        if (st.right_done) { n = st.split; continue; }
        T* const rp = p + st.split;
        const std::size_t ln = st.split;
        const std::size_t rn = n - st.split;
        const int nb = budget;
        const unsigned nd = depth - 1;
        fork_join([=] { vqsort_parallel_rec(p, ln, nb, nd); },
                  [=] { vqsort_parallel_rec(rp, rn, nb, nd); });
        return;
    }
}
#endif

/// How much disorder the patch merges may repair before the vectorised
/// quicksort becomes the better buy.  They pull the dirty positions out, sort
/// them and merge them back -- three sequential passes that move every element
/// at least twice, plus a sort of the patch, so their cost climbs with the
/// amount of dirt while the quicksort's does not.  The crossovers below are
/// measured, not guessed (random long-distance swaps, this machine):
///
/// 4M int32, sorted then a percentage of positions swapped at random:
///
///   swapped   patch merge   quicksort seq   quicksort pooled
///    0.02%      0.0077          0.0178           ~0.011
///    0.1%       0.0123          0.0169            0.0115
///    0.3%       0.0150          0.0169            ~0.011
///    1%         0.0265          0.0202            0.0108
///
/// Sequentially the patch merge holds on until the patch itself is expensive
/// to sort, around half a percent; pooled, the quicksort takes over four times
/// earlier because it is the only one of the two that uses the second core.
/// The patch merge's own dirty count for those rows falls between n/256 and
/// n/128 at 0.1% and above n/32 at 1%, which is what these two budgets pick
/// out.  Declining is cheap either way (0.0005-0.002 s here).  The cheap
/// repairs (adjacent-swap, bounded insertion) are never affected: they cost a
/// fraction of a pass and beat both.
template <class T>
inline std::size_t patch_merge_dirty_budget(std::size_t n, bool parallel) {
    if (!(vqsort_preferred<T>() && vqsort_usable<T>(n))) return kPatchDirtyDefault;
#if FYX_ENABLE_PARALLEL
    if (parallel && parallel_available()) return n / 256;
#else
    (void)parallel;
#endif
    return n / 64;
}

/// Sorts `p[0,n)` with the AVX-512 vectorised quicksort, or returns false and
/// leaves the range untouched.  Declines when: the type has no kernel, the CPU
/// has no AVX-512, the range is too small for the vector partition, the radix
/// family is faster for the type, or -- for floating point -- the range holds
/// a NaN or a -0, whose total order the hardware compare cannot reproduce.
template <class T>
inline bool try_vector_quicksort(T* p, std::size_t n, bool descending, bool parallel) {
    if constexpr (!vqsort_kernel_supported_v<T>) {
        (void)p; (void)n; (void)descending; (void)parallel;
        return false;
    } else {
        if (!vqsort_preferred<T>()) return false;
        if (!vqsort_usable<T>(n)) return false;
        if (n < kVqsortMinN) return false;
        if (!vq_range_clean_memo(p, n)) return false;
#if FYX_ENABLE_PARALLEL
        if (parallel && parallel_available()) {
            unsigned depth = 0;
            for (unsigned w = global_pool().nworkers(); w > 1; w >>= 1) ++depth;
            depth += 2;                       // a few extra levels for balance
            vqsort_parallel_rec(p, n, vqsort_budget(n), depth);
        } else {
            vqsort_serial(p, n);
        }
#else
        (void)parallel;
        vqsort_serial(p, n);
#endif
        if (descending) std::reverse(p, p + n);
        return true;
    }
}

template <class T, class Comp>
inline void sort_st(T* p, std::size_t n, Comp comp, bool descending,
                    const InputProfile<T, Comp>* known_profile) {
    constexpr bool radix_type = radix_supported_v<T>;
    const bool ascending      = is_ascending_v<Comp, T>;
    const bool radix_order    = radix_type && (ascending || descending);

    if (n <= kNetworkMax) {
        record_dispatch(DispatchDecision::Network);
        if constexpr (radix_type) {
            if (radix_order) {
                small_sort_numeric(p, n);
                if (descending) std::reverse(p, p + n);
                return;
            }
        }
        insertion_sort(p, p + n, comp);
        return;
    }

    InputProfile<T, Comp> local_profile;
    const InputProfile<T, Comp>* prof = known_profile;
    if (!prof && n >= kProfileMinN) {
        local_profile = profile_input(p, n, comp);
        prof = &local_profile;
    }

    if (prof) {
        if (apply_profile_fast_exit(p, n, *prof, true)) return;
        if (radix_order && try_one_break_rotate(p, n, descending, /*stable_wrap=*/true)) {
            record_dispatch(DispatchDecision::ProfileSorted);
            return;
        }
#if FYX_ENABLE_PARALLEL
        if (radix_order && !(parallel_available() && n >= configured_min_parallel_size()) &&
            try_proof_structured_sort(p, n, descending, /*stable_wrap=*/true)) {
            record_dispatch(DispatchDecision::ProfileSorted);
            return;
        }
#else
        if (radix_order && try_proof_structured_sort(p, n, descending, /*stable_wrap=*/true)) {
            record_dispatch(DispatchDecision::ProfileSorted);
            return;
        }
#endif
    } else {
        if (radix_order) {
            if (try_radix_monotonic_sort(p, n, descending, true)) return;
        } else {
            if (try_monotonic_sort(p, p + n, comp, true)) return;
        }
    }

    if (try_zigzag_organ_pipe_sort(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; }
    if (try_numeric_half_organ_fill(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; }
    // Adaptive natural-run merge: see sort_pointer_core.
    if (try_natural_run_merge_adaptive(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; }
    // Ordered except for one stretch in the middle: see try_sorted_affix_sort.
    if (try_sorted_affix_sort(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; }

    const bool high_entropy = prof && prof->is_high_entropy;
    const bool partial_pdq = prof && prof->is_partially_sorted &&
        n <= kProfilePartialPdqMax;
    if (partial_pdq && !(prof && prof->is_low_cardinality)) {
        if (radix_order) {
            if (try_partially_sorted_local_repair(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; }
            if (try_radix_permutation_range_sort(p, n, descending)) { record_dispatch(DispatchDecision::Radix); return; }
            // High-entropy nearly-sorted numeric data with long-distance swaps
            // is usually faster on FYX radix than on comparison pdq/patch.
            // Continue into the radix block below instead of committing here.
        } else {
            if (try_guarded_radix_order_sort(p, n, comp, false, true)) { record_dispatch(DispatchDecision::Radix); return; }
            if (partial_runs_look_independent(p, n, comp) && try_key_radix_for_partial(p, n, comp)) { record_dispatch(DispatchDecision::Radix); return; } if (try_partially_sorted_repair(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; }
            if (try_key_radix_for_partial(p, n, comp)) { record_dispatch(DispatchDecision::Radix); return; } pdqsort_for_profile_pattern(p, n, comp);
            record_dispatch(DispatchDecision::PartialPdq);
            return;
        }
    }

    if (try_bitonic_runs_sort(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; }

    if (radix_order) {
        if (!high_entropy) {
            if (try_integer_range_count_sort(p, n, descending, prof ? &prof->sample_window : nullptr)) { record_dispatch(DispatchDecision::LowCardinality); return; }
            if (!vq_beats_sparse_count(p, n)) {
                if (try_radix_key_sparse_count_sort(p, n, descending, prof ? &prof->sample_window : nullptr)) { record_dispatch(DispatchDecision::LowCardinality); return; }
                if (try_low_cardinality_count_sort(p, p + n, comp)) { record_dispatch(DispatchDecision::LowCardinality); return; }
            }
        }
        if (partial_pdq && try_partially_sorted_local_repair(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; }
        if (try_radix_permutation_range_sort(p, n, descending)) { record_dispatch(DispatchDecision::Radix); return; }
        // A narrow value range is not the same property as a low value count:
        // 4M int32 drawn from 2^18 values is high-entropy by every sample and
        // still sorts fastest by counting.  Both range kernels self-gate on
        // range vs n, so this costs a sample when it declines.
        if (high_entropy && try_integer_range_count_sort(p, n, descending, prof ? &prof->sample_window : nullptr)) { record_dispatch(DispatchDecision::LowCardinality); return; }
        if constexpr (radix_type) {
            if (n >= kRadixThreshold || std::is_floating_point<T>::value) {
#if FYX_ENABLE_PARALLEL
                // The high-prefix sort beats the 32-wide one on every size
                // measured -- 1M int32: 0.0090s against 0.0120s, 8M: 0.080s
                // against 0.154s -- so it goes first and the wide sort is the
                // fallback for the shapes whose prefix it declines.
                if (try_vector_quicksort(p, n, descending, false)) {
                    record_dispatch(DispatchDecision::VectorQuick);
                    return;
                }
                if (high_entropy && try_serial_radix_high_prefix_sort(p, n, descending)) {
                    record_dispatch(DispatchDecision::Radix);
                    return;
                }
                if (high_entropy && try_serial_radix32_wide_sort(p, n, descending)) {
                    record_dispatch(DispatchDecision::Radix);
                    return;
                }
#endif
                if (radix_sort(p, n)) {             // returns false only on OOM
                    if (descending) std::reverse(p, p + n);
                    record_dispatch(DispatchDecision::Radix);
                    return;
                }
                // allocation failed -> fall through to the comparison path
            }
        }
    } else {
        if (try_guarded_radix_order_sort(p, n, comp, false, high_entropy)) { record_dispatch(DispatchDecision::Radix); return; }
        if (!high_entropy) {
            if (try_string_value_count_sort(p, n, comp, false)) { record_dispatch(DispatchDecision::LowCardinality); return; }
            if (try_guarded_string_value_count_sort(p, n, comp)) { record_dispatch(DispatchDecision::LowCardinality); return; }
            if (try_trivial_prefix_key_count_sort(p, n, comp)) { record_dispatch(DispatchDecision::LowCardinality); return; }
            if (try_low_cardinality_count_sort(p, p + n, comp)) { record_dispatch(DispatchDecision::LowCardinality); return; }
        }
        if (partial_pdq) { if (partial_runs_look_independent(p, n, comp) && try_key_radix_for_partial(p, n, comp)) { record_dispatch(DispatchDecision::Radix); return; } if (try_partially_sorted_repair(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; } if (try_key_radix_for_partial(p, n, comp)) { record_dispatch(DispatchDecision::Radix); return; } pdqsort_for_profile_pattern(p, n, comp); record_dispatch(DispatchDecision::PartialPdq); return; }
        if (try_guarded_string_order_sort(p, n, comp, false)) { record_dispatch(DispatchDecision::Radix); return; }
        if (try_string_msd_sort(p, n, comp, descending)) { record_dispatch(DispatchDecision::Radix); return; }
        if (try_trivial_prefix_key_radix_sort(p, n, comp)) { record_dispatch(DispatchDecision::Radix); return; }
    }

    if (n <= kInsertionThreshold) { insertion_sort(p, p + n, comp); record_dispatch(DispatchDecision::Pdq); return; }
    // Generic path: strings, structs and custom comparators get an ips4o-style
    // sample sort once they are large enough; smaller ranges use pdqsort.
    if (n >= kSampleThreshold && (!std::is_arithmetic<T>::value || !radix_order)) {
        if constexpr (std::is_copy_constructible<T>::value) {
            sample_sort(p, p + n, comp);
            record_dispatch(DispatchDecision::Sample);
            return;
        }
    }
    pdqsort(p, p + n, comp);
    record_dispatch(DispatchDecision::Pdq);
}


// The moving merge below is used by both the parallel merge and the sequential
// stable merge sort, so it must stay outside the parallel guard.
/// The same stable merge as std::merge -- equal elements keep the order of the
/// first run -- except that it moves.  std::merge assigns through a const
/// lvalue, which a move-only payload (std::unique_ptr, say) cannot take, and
/// that used to make the parallel merge refuse to compile for them.
template <class T, class Comp>
inline void merge_runs_moving(T* a, std::size_t n1, T* b, std::size_t n2, T* dst, Comp comp) {
    std::size_t i = 0, j = 0, w = 0;
    while (i != n1 && j != n2) {
        if (comp(b[j], a[i])) dst[w++] = std::move(b[j++]);
        else                  dst[w++] = std::move(a[i++]);
    }
    while (i != n1) dst[w++] = std::move(a[i++]);
    while (j != n2) dst[w++] = std::move(b[j++]);
}

#if FYX_ENABLE_PARALLEL

template <class T, unsigned ActivePasses>
inline bool radix_sort_lower_passes(T* data, std::size_t n) noexcept {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    static_assert(ActivePasses <= RT::passes, "too many lower radix passes");
    if (n < 2) return true;
    if (n <= kNetworkMax) {
        small_sort_numeric(data, n);
        return true;
    }

    ScratchLease<Key> lease(n * 2);
    if (!lease.valid()) return false;
    Key* a = lease.get();
    Key* b = a + n;

    RadixHistogram<ActivePasses> hist;
    hist.clear();
    for (std::size_t i = 0; i < n; ++i) {
        const Key k = RT::encode(data[i]);
        a[i] = k;
        for (unsigned pass = 0; pass < ActivePasses; ++pass)
            ++hist.count[pass][radix_digit(k, pass)];
    }

    const RadixPlan<ActivePasses> plan = plan_radix<ActivePasses>(hist, n);
    if (plan.count == 0) return true;

    Key* src = a;
    Key* dst = b;
    constexpr std::size_t kPerLine = WcbTraits<Key>::kPerLine;
    ScratchLease<Key> wcb_lease(2 * kRadixBuckets * kPerLine + kPerLine);
    if (!wcb_lease.valid()) return false;

    RadixScatterScratch<Key> sc;
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(wcb_lease.get());
    const std::uintptr_t mis = (kCacheLine - (addr & (kCacheLine - 1))) & (kCacheLine - 1);
    sc.line = reinterpret_cast<Key*>(addr + mis);
    sc.head = sc.line + kRadixBuckets * kPerLine;
    const bool can_stream = have_nt_stores();

    std::size_t offset[kRadixBuckets];
    for (unsigned pi = 0; pi < plan.count; ++pi) {
        const unsigned pass = plan.active[pi];
        std::size_t sum = 0;
        for (unsigned d = 0; d < kRadixBuckets; ++d) {
            offset[d] = sum;
            sum += static_cast<std::size_t>(hist.count[pass][d]);
        }
        radix_scatter_pass<Key>(src, n, dst, pass * kRadixBits, offset, sc, can_stream);
        Key* t = src; src = dst; dst = t;
    }

    for (std::size_t i = 0; i < n; ++i) data[i] = RT::decode(src[i]);
    return true;
}

template <unsigned TopBits, class T>
inline bool try_msd_radix_bucket_sort_impl(T* p, std::size_t n, bool descending) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr unsigned KeyBits = sizeof(Key) * CHAR_BIT;
    static_assert(TopBits > 0 && TopBits <= KeyBits, "invalid MSD radix width");
    constexpr std::size_t Buckets = std::size_t(1) << TopBits;
    constexpr unsigned Shift = KeyBits - TopBits;

    if (n < (std::size_t(1) << 20) || !parallel_available()) return false;
    // This MSD-bucket hybrid is a bandwidth-constrained 2-worker win in the
    // sandbox, but the user's 4H8G vqsort matrix shows it loses to the normal
    // chunked radix path once four hardware threads are available.  Keep it as
    // the 2-core specialization and let 3+ cores fall through to radix.
    if (global_pool().nworkers() != 2) return false;
    const std::size_t chunks = adaptive_parallel_chunks(n);
    if ((n + chunks - 1) / chunks > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        return false;

    ScratchLease<T> tmp_lease(n);
    if (!tmp_lease.valid()) return false;
    T* tmp = tmp_lease.get();

    std::vector<std::uint32_t> local(chunks * Buckets, 0);
    auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            const std::size_t lo = (c * n) / chunks;
            const std::size_t hi = ((c + 1) * n) / chunks;
            std::uint32_t* lc = local.data() + c * Buckets;
            for (std::size_t i = lo; i < hi; ++i) {
                const Key k = RT::encode(p[i]);
                ++lc[static_cast<std::size_t>(k >> Shift)];
            }
        }
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);

    std::vector<std::size_t> off(Buckets + 1, 0);
    std::size_t nonzero = 0;
    for (std::size_t d = 0; d < Buckets; ++d) {
        std::size_t s = 0;
        for (std::size_t c = 0; c < chunks; ++c)
            s += local[c * Buckets + d];
        if (s != 0) ++nonzero;
        off[d + 1] = off[d] + s;
    }
    if (nonzero <= 1) return false;

    std::vector<std::size_t> base(chunks * Buckets);
    std::size_t run = 0;
    for (std::size_t d = 0; d < Buckets; ++d) {
        for (std::size_t c = 0; c < chunks; ++c) {
            const std::size_t idx = c * Buckets + d;
            base[idx] = run;
            run += local[idx];
        }
    }

    auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
        ScratchLease<std::size_t> pos_lease(Buckets);
        std::vector<std::size_t> pos_fallback;
        std::size_t* pos = pos_lease.valid() ? pos_lease.get() : nullptr;
        if (!pos) {
            pos_fallback.resize(Buckets);
            pos = pos_fallback.data();
        }
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            std::memcpy(pos, base.data() + c * Buckets, Buckets * sizeof(std::size_t));
            const std::size_t lo = (c * n) / chunks;
            const std::size_t hi = ((c + 1) * n) / chunks;
            for (std::size_t i = lo; i < hi; ++i) {
                const T v = p[i];
                const Key k = RT::encode(v);
                tmp[pos[static_cast<std::size_t>(k >> Shift)]++] = v;
            }
        }
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);

    auto bucket_job = [&](std::size_t d_lo, std::size_t d_hi) {
        for (std::size_t d = d_lo; d < d_hi; ++d) {
            const std::size_t lo = off[d];
            const std::size_t hi = off[d + 1];
            const std::size_t sz = hi - lo;
            if (sz <= 1) continue;
            if constexpr (TopBits == 16 && sizeof(Key) == 8) {
                if constexpr (std::is_floating_point<T>::value) {
                    if (!radix_sort_lower_passes<T, 6>(tmp + lo, sz))
                        sort_st(tmp + lo, sz, fyx::less{}, false);
                } else {
                    if (sz <= kNetworkMax) small_sort_numeric(tmp + lo, sz);
                    else if (sz < (std::size_t(1) << 16)) pdqsort(tmp + lo, tmp + hi, fyx::less{});
                    else if (!radix_sort_lower_passes<T, 6>(tmp + lo, sz))
                        pdqsort(tmp + lo, tmp + hi, fyx::less{});
                }
            } else if constexpr (TopBits == 8 && sizeof(Key) == 4 && std::is_floating_point<T>::value) {
                if (!radix_sort_lower_passes<T, 3>(tmp + lo, sz))
                    sort_st(tmp + lo, sz, fyx::less{}, false);
            } else {
                sort_st(tmp + lo, sz, fyx::less{}, false);
            }
        }
    };
    const std::size_t bucket_grain = TopBits >= 16 ? std::size_t(256) : std::size_t(1);
    parallel_for_index(std::size_t(0), Buckets, bucket_grain, bucket_job);

    auto copy_job = [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) p[i] = tmp[i];
    };
    parallel_for_index(std::size_t(0), n, kParallelThreshold, copy_job);
    if (descending) std::reverse(p, p + n);
    return true;
}

template <class T>
inline bool try_msd_radix_bucket_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        using Key = typename RadixTraits<T>::Key;
        if constexpr (sizeof(Key) == 8) {
            return try_msd_radix_bucket_sort_impl<16>(p, n, descending);
        } else if constexpr (std::is_same<T, float>::value) {
            return try_msd_radix_bucket_sort_impl<8>(p, n, descending);
        } else {
            (void)p; (void)n; (void)descending;
            return false;
        }
    }
}

template <class T, class Comp>
inline void parallel_merge_to_buffer_rec(T* src,
                                         std::size_t a0, std::size_t a1,
                                         std::size_t b0, std::size_t b1,
                                         T* dst, std::size_t out,
                                         Comp comp) {
    const std::size_t n1 = a1 - a0;
    const std::size_t n2 = b1 - b0;
    const std::size_t total = n1 + n2;
    if (total == 0) return;
    constexpr std::size_t kMergeGrain = 8192;
    if (total <= kMergeGrain || !parallel_available()) {
        merge_runs_moving(src + a0, a1 - a0, src + b0, b1 - b0, dst + out, comp);
        return;
    }

    if (n1 >= n2) {
        const std::size_t amid = a0 + n1 / 2;
        const std::size_t bmid = static_cast<std::size_t>(
            std::lower_bound(src + b0, src + b1, src[amid], comp) - src);
        const std::size_t left = (amid - a0) + (bmid - b0);
        fork_join([&] { parallel_merge_to_buffer_rec(src, a0, amid, b0, bmid, dst, out, comp); },
                  [&] { parallel_merge_to_buffer_rec(src, amid, a1, bmid, b1, dst, out + left, comp); });
    } else {
        const std::size_t bmid = b0 + n2 / 2;
        const std::size_t amid = static_cast<std::size_t>(
            std::upper_bound(src + a0, src + a1, src[bmid], comp) - src);
        const std::size_t left = (amid - a0) + (bmid - b0);
        fork_join([&] { parallel_merge_to_buffer_rec(src, a0, amid, b0, bmid, dst, out, comp); },
                  [&] { parallel_merge_to_buffer_rec(src, amid, a1, bmid, b1, dst, out + left, comp); });
    }
}

template <class T, class Comp>
inline bool parallel_merge_buffered(T* p, std::size_t mid, std::size_t n, Comp comp) {
    if constexpr (!std::is_default_constructible<T>::value || !std::is_move_assignable<T>::value) {
        (void)p; (void)mid; (void)n; (void)comp;
        return false;
    } else {
        if (mid == 0 || mid >= n) return true;
#if FYX_HAS_EXCEPTIONS
        try {
#endif
            std::vector<T> out(n);
            parallel_merge_to_buffer_rec(p, 0, mid, mid, n, out.data(), 0, comp);
            auto copy_job = [&](std::size_t lo, std::size_t hi) {
                for (std::size_t i = lo; i < hi; ++i) p[i] = std::move(out[i]);
            };
            parallel_for_index(std::size_t(0), n, kParallelThreshold, copy_job);
            return true;
#if FYX_HAS_EXCEPTIONS
        } catch (...) {
            return false;
        }
#endif
    }
}

#endif // FYX_ENABLE_PARALLEL

#if FYX_ENABLE_PARALLEL
// Task-parallel divide: sort both halves (each via sort_st, which may itself
// pick radix / network / pdqsort) and merge the two sorted runs back together.
// Correct for any comparator because both halves are sorted *by comp* and
// std::inplace_merge merges two comp-sorted runs into one.
template <class T, class Comp>
inline void parallel_sort_ptr(T* p, std::size_t n, Comp comp, bool descending,
                              const Options& o) {
    if (n <= kParallelThreshold || !parallel_available() ||
        o.parallel == Tri::Off || o.threads == 1) {
        sort_st(p, n, comp, descending);
        return;
    }
    const std::size_t mid = n / 2;
    fork_join([&] { sort_st(p, mid, comp, descending); },
              [&] { sort_st(p + mid, n - mid, comp, descending); });
    // The two halves come back in the order sort_st produces, which for
    // floating point with a default comparator is the radix total order:
    // -NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN.  Merging them with the
    // raw comparator instead would compare NaN with `<`, which is false both
    // ways, and interleave the two halves wrongly -- 70000 floats holding
    // NaNs came out with three inversions.  adaptive_order is the same
    // comparator everywhere else.
    auto order = adaptive_order<T>(comp);
    if (!parallel_merge_buffered(p, mid, n, order))
        std::inplace_merge(p, p + mid, p + n, order);
}
#endif

// Scratch storage for stable_merge_sort. Unlike vector::emplace_back, direct
// construction into reserved storage has no per-element capacity branch. The
// exact live-object count makes the buffer safe for move-only/non-default types
// and for comparator or move exceptions during a merge.
template <class T>
class StableSortBuffer {
    using Alloc = std::allocator<T>;
    using Traits = std::allocator_traits<Alloc>;
    Alloc alloc_;
    T* data_;
    std::size_t size_;
    std::size_t capacity_;

public:
    explicit StableSortBuffer(std::size_t capacity)
        : data_(Traits::allocate(alloc_, capacity)), size_(0), capacity_(capacity) {}
    StableSortBuffer(const StableSortBuffer&) = delete;
    StableSortBuffer& operator=(const StableSortBuffer&) = delete;
    ~StableSortBuffer() {
        clear();
        Traits::deallocate(alloc_, data_, capacity_);
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    T& operator[](std::size_t i) noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

    void emplace_back(T&& value) {
        // Callers keep each output pass within the buffer's allocated
        // capacity; treating that bound as a loop invariant avoids a capacity
        // branch on every element.
        Traits::construct(alloc_, data_ + size_, std::move(value));
        ++size_;
    }

    void clear() noexcept {
        while (size_ != 0) Traits::destroy(alloc_, data_ + --size_);
    }

    void swap(StableSortBuffer& other) noexcept {
        using std::swap;
        swap(data_, other.data_);
        swap(size_, other.size_);
        swap(capacity_, other.capacity_);
    }
};

// Stable top-down merge sort using one reusable buffer for the left run.
// Balanced splitting guarantees that the smaller run fits in n/2 scratch
// elements. Boundary checks skip merges when adjacent recursive ranges are
// already ordered; this retains stability while avoiding a second full buffer.
template <class It, class T, class Comp>
inline void stable_merge_sort_recursive(It first, std::size_t n,
                                        StableSortBuffer<T>& scratch, Comp& comp) {
    constexpr std::size_t kInsertionCutoff = 32;
    if (n < 2) return;
    if (n <= kInsertionCutoff) {
        for (std::size_t i = 1; i < n; ++i) {
            T value = std::move(first[i]);
            std::size_t hole = i;
            while (hole != 0 && comp(value, first[hole - 1])) {
                first[hole] = std::move(first[hole - 1]);
                --hole;
            }
            first[hole] = std::move(value);
        }
        return;
    }

    const std::size_t left_size = n / 2;
    const std::size_t middle = left_size;
    stable_merge_sort_recursive(first, left_size, scratch, comp);
    stable_merge_sort_recursive(first + middle, n - middle, scratch, comp);
    if (!comp(first[middle], first[middle - 1])) return;

    scratch.clear();
    for (std::size_t i = 0; i < left_size; ++i)
        scratch.emplace_back(std::move(first[i]));
    std::size_t left = 0, right = middle, out = 0;
    while (left < left_size && right < n) {
        if (comp(first[right], scratch[left])) first[out++] = std::move(first[right++]);
        else                                      first[out++] = std::move(scratch[left++]);
    }
    while (left < left_size) first[out++] = std::move(scratch[left++]);
    scratch.clear();
}

// ---------------------------------------------------------------------------
// Adaptive, allocation-assisted, STABLE merge sort. Works for any random-
// access range (pointers, vector/deque iterators, ...). Used by stable_sort
// whenever the radix path cannot be taken.
// ---------------------------------------------------------------------------
template <class It, class Comp>
inline void stable_merge_sort(It first, It last, Comp comp) {
    using T = iter_value_t<It>;
    const std::size_t n = static_cast<std::size_t>(last - first);
    if (n < 2) return;

    // A balanced recursion needs at most floor(n/2) live scratch objects.
    // Raw allocator-backed storage supports move-only/non-default types and
    // avoids reserving a second n-element buffer for alternating merge passes.
    StableSortBuffer<T> scratch(n / 2);
    stable_merge_sort_recursive(first, n, scratch, comp);
}

// ---------------------------------------------------------------------------
// nth_element: introspective quickselect reusing pdqsort's pivot / partition.
// A depth limit guarantees termination (on pathological inputs it falls back
// to a full pdqsort of the remaining range, which is still O(n log n)).
// ---------------------------------------------------------------------------
template <class It, class Comp>
inline void nth_select(It first, It nth, It last, Comp comp, int& depth) {
    using Diff = typename std::iterator_traits<It>::difference_type;
    while (last - first > 1) {
        const Diff sz = last - first;
        if (sz <= static_cast<Diff>(kInsertionThreshold)) {
            pdqsort(first, last, comp);   // fully sorts -> nth is now exact
            return;
        }
        choose_pivot(first, last, comp);  // places a pivot at *first
        const It p = partition_right_simple(first, last, comp).pivot_pos;
        if (nth < p)       last = p;
        else if (nth > p)  first = p + 1;
        else               return;
        if (--depth < 0) { pdqsort(first, last, comp); return; }
    }
}

template <class It, class Comp>
inline void nth_element_impl(It first, It nth, It last, Comp comp) {
    using Diff = typename std::iterator_traits<It>::difference_type;
    const Diff total = last - first;
    if (total <= 1 || nth < first || nth >= last) return;
    int depth = static_cast<int>(log2_floor(static_cast<std::uint64_t>(total))) * 2 + 16;
    nth_select(first, nth, last, comp, depth);
}

// ---------------------------------------------------------------------------
// partial_sort: build a max-heap over [first, middle), sift out every element
// in [middle, last) that is smaller than the heap root, then heapsort the
// partial range.  Matches std::partial_sort's contract for any comparator.
// ---------------------------------------------------------------------------
template <class It, class Comp>
inline void partial_sort_impl(It first, It middle, It last, Comp comp) {
    using Diff = typename std::iterator_traits<It>::difference_type;
    if (first == middle) return;   // nothing to select; leave range untouched
    const Diff m = static_cast<Diff>(middle - first);

    for (Diff i = m / 2 - 1; i >= 0; --i) sift_down(first, i, m, comp);
    for (It it = middle; it != last; ++it) {
        if (comp(*it, *first)) {
            std::iter_swap(first, it);   // old root leaves the heap range
            sift_down(first, Diff(0), m, comp);
        }
    }
    heap_sort(first, middle, comp);
}

} // namespace detail

// ---------------------------------------------------------------------------
// Core dispatchers (called by the public overloads)
// ---------------------------------------------------------------------------

template <class T, class Comp>
inline void sort_pointer_core_impl(T* p, std::size_t n, Comp comp, const Options& o) {
    (void)o;   // consumed only by the parallel branches (compiled out otherwise)
    if (n == 0) return;
#if FYX_ENABLE_GPU
    // If the caller asked for the GPU and a backend is present, try it; on any
    // failure (no device, compile error, ...) it returns false and we fall
    // through to the verified CPU path below.
    if (o.gpu && detail::gpu_sort_dispatch(p, n, comp, o)) return;
#endif
    const bool descending = detail::is_descending_v<Comp, T>;
    const bool ascending  = detail::is_ascending_v<Comp, T>;
    const bool radix_ok   = detail::radix_supported_v<T> && (ascending || descending);

    // Small numeric ranges on AVX-512 hosts: the vectorised quicksort (with
    // its bitonic-network leaves) is ~5-30x faster below kVqsortMinN than
    // the probe stack + pdq/radix below, which only pays for itself on large
    // inputs.  One branch-free order scan keeps sorted / reverse / all-equal
    // inputs O(n); random input leaves that scan after a single block.
    if constexpr (detail::vqsort_kernel_supported_v<T> && detail::vqsort_preferred<T>()) {
#if FYX_ENABLE_PARALLEL
        const bool small_vq_serial = !detail::dynamic_parallel_allowed<T>(n, o);
#else
        const bool small_vq_serial = true;
#endif
        if (radix_ok && n >= detail::kSmallVqsortMinN && n < detail::kSmallVqsortMaxN &&
            small_vq_serial && detail::use_avx512()) {
            // One fused vector pass: neighbour order + NaN / -0 scan.  When
            // the range is clean hardware order is the library's order; else
            // the radix-key scan decides.
            // Bitwise all-equal first: random input differs in the first
            // bytes, and an all-equal range costs one memcmp.
            // Then a reverse-shaped range (first key above the last in sort
            // order, 16 strided samples non-increasing) is verified and
            // reversed in one two-ended pass instead of an order scan plus a
            // reverse pass (1M uint64: one read + write instead of two reads
            // + write).  A break leaves the swapped blocks in place -- a
            // permutation the paths below sort like any other input.
            if (const int fp = detail::front_order_probe(p, n, descending)) {
                detail::record_dispatch(fp == 1 ? detail::DispatchDecision::ProfileAllEqual
                                                : detail::DispatchDecision::ProfileReverse);
                return;
            }
            std::size_t settled = 0;
            unsigned prior = 0;
            const unsigned pre = detail::vqsort_small_prescan(p, n, &settled, &prior);
            const bool clean = (pre & 4u) == 0;   // as far as scanned
            // Keys before `mono` hold no descent in the sort direction (the
            // prescan stopped in the block where its first one lies, and
            // hardware order is the library's on the clean prefix), so the
            // descent scans below start there instead of re-reading it.
            const std::size_t mono =
                (clean && (pre & 3u) == 3u && prior == (descending ? 2u : 1u)) ? settled : 0;
            detail::FastOrderKind k;
            if (clean) {
                const bool up = (pre & 1u) != 0, dn = (pre & 2u) != 0;
                k = (!up && !dn) ? detail::FastOrderKind::AllEqual
                  : (up && dn)   ? detail::FastOrderKind::None
                  : (up != descending) ? detail::FastOrderKind::Sorted
                                       : detail::FastOrderKind::Reverse;
            } else {
                k = detail::blocked_radix_order_kind(p, n, descending);
            }
            if (k == detail::FastOrderKind::AllEqual) {
                detail::record_dispatch(detail::DispatchDecision::ProfileAllEqual);
                return;
            }
            if (k == detail::FastOrderKind::Sorted) {
                detail::record_dispatch(detail::DispatchDecision::ProfileSorted);
                return;
            }
            if (k == detail::FastOrderKind::Reverse) {
                detail::reverse_range_adaptive(p, n);
                detail::record_dispatch(detail::DispatchDecision::ProfileReverse);
                return;
            }
            // At most 32 distinct keys: count + fill (a 64-key sample with
            // more than 16 distinct keys declines at once).
            if (detail::try_vfew_distinct_sort(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::LowCardinality);
                return;
            }
            // Few-run shapes (rotated, organ pipe, bitonic, permuted blocks)
            // are O(n) for the structural kernels; each rejects random data
            // within a short scan.  Up to one leaf the column network sorts
            // any shape in a few hundred vector ops, so they are skipped.
            // Many sorted runs (four deep, well-spaced drops at the front):
            // only the few-run merge can use that; skipping the descent
            // count and the repairs saves a full pass before the quicksort.
            const bool runs_like = n > detail::vqsort_leaf<T>() &&
                                   detail::descents_look_like_runs(p, n, mono > 1 ? mono : 1, descending, 16, 4);
            if (runs_like && detail::try_few_runs_merge(p, n, descending, mono)) {
                detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                return;
            }
            if (!runs_like && n > detail::vqsort_leaf<T>()) {
                // Route by the number of descents (counted up to n/16).
                std::size_t brk = n;
                // Up to 4096 keys one pass records the descent positions
                // themselves (the sparse-outlier repair reuses them).
                constexpr std::size_t kPosCap = 256;
                std::uint32_t dpos[kPosCap];
                const bool have_pos = n <= 16 * kPosCap;
                std::size_t D;
                if (have_pos) {
                    // Below 2048 keys only the few-descent repairs use an
                    // exact count; a dense head (> 8 drops in 32 pairs)
                    // already routes the range to the dense branch.
                    D = n < 2048 ? detail::descent_positions(p, 1, 33, descending, dpos, 8) : 0;
                    D = D > 8 ? n / 16 + 1 : detail::descent_positions(p, 1, n, descending, dpos, n / 16);
                    if (D != 0) brk = dpos[0];
                } else if (detail::descents_look_dense(p, n, descending)) {
                    D = n / 16 + 1;        // random-like: skip the n/8-pair count
                } else {
                    // (block starts of the first descents: positions for the
                    // outlier repair come from those blocks alone)
                    std::size_t nblk = 0;
                    D = detail::count_descents_capped(p, n, descending, n / 16, brk, mono,
                                                      n <= 0xFFFFFFFFull ? dpos : nullptr, 32, &nblk);
                    if (D <= 32 && nblk != 0 && !detail::descent_positions_in_blocks(p, n, descending, dpos, nblk, D))
                        D = 33;                    // (cannot happen) no repair then
                }
                const bool pos_ok = have_pos || (D <= 32 && n <= 0xFFFFFFFFull);   // dpos holds all D
                if (D == 1) {
                    // Two runs (concatenated sorted halves, a rotation).
                    brk = detail::radix_key_find_break(p, brk, n, detail::RadixTraits<T>::encode(p[brk - 1]),
                                                       true, descending);
                    detail::merge_two_runs(p, n, brk, descending);
                    detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                    return;
                }
                // A few monotone runs either way (organ pipe, block swaps).
                if (detail::try_few_runs_merge(p, n, descending, mono)) {
                    detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                    return;
                }
                if (D > n / 16) {
                    // Dense descents: organ pipe / bitonic / zigzag families.
                    if (detail::try_zigzag_organ_pipe_sort(p, n, comp) ||
                        (detail::likely_mid_bitonic_runs(p, n, comp) &&
                         detail::try_bitonic_runs_sort(p, n, comp)) ||
                        detail::try_numeric_half_organ_fill(p, n, comp) ||
                        detail::try_bitonic_runs_sort(p, n, comp)) {
                        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                        return;
                    }
                } else if (D > 1) {
                    // Sparse descents: mostly sorted.  Local displacements go
                    // to bounded insertion, permuted blocks to the structural
                    // proof, a minority of far / tail keys to extract-merge.
                    // Up to 32 isolated outliers (far swaps, a few misplaced
                    // keys): the outlier repair is about one scan and leaves
                    // the range untouched when it declines -- before the
                    // insertion repair, which a far key exhausts only after
                    // n / 4 moves (1M double far swaps: 2.0 -> ~0.3 ms).
                    if (D <= 32 && pos_ok && detail::sparse_outlier_repair_at(p, n, dpos, D, descending)) {
                        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                        return;
                    }
                    const std::size_t head = std::min<std::size_t>(n - 1, 256);
                    if (D <= n / 128 &&
                        detail::head_inversions_within(p, n, descending, head, head / 32) &&
                        detail::descents_look_local(p, n, brk, descending, 96, 4)) {
                        if (detail::budgeted_insertion_repair(p, n, descending, n / 4 + 64)) {
                            detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                            return;
                        }
                    }
                    if (detail::try_proof_structured_sort(p, n, descending)) {
                        detail::record_dispatch(detail::DispatchDecision::ProfileSorted);
                        return;
                    }
                    // Sorted prefix of at least half the range, disorder only
                    // behind it (appended keys): sort the tail, merge once.
                    {
                        const std::size_t cut = detail::radix_key_find_break(
                            p, brk, n, detail::RadixTraits<T>::encode(p[brk - 1]), true, descending);
                        // (Below 2048 int32 keys the vq sorts the whole range
                        // faster than tail sort + merge network; measured.)
                        if (cut >= n / 2 && cut < n && (n >= 2048 || sizeof(T) == 8)) {
                            sort_pointer_core_impl(p + cut, n - cut, comp, o);
                            detail::merge_two_runs(p, n, cut, descending);
                            detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                            return;
                        }
                    }
                    // Below 2048 keys extraction costs about what the sort
                    // does (measured: a 10% random tail of 1000 keys runs
                    // 2x slower through it); the outlier repair above already
                    // took the few-descent cases.
                    // Many sorted runs (each boundary a deep drop) defeat the
                    // extractor only after it has scanned most of the range.
                    if (n >= 2048 && !detail::descents_look_like_runs(p, n, brk, descending, 16, 4) &&
                        detail::try_extract_merge_repair(p, n, descending, [&](T* q, std::size_t m) {
                            sort_pointer_core_impl(q, m, comp, o);
                        })) {
                        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                        return;
                    }
                }
            }
            // Narrow integer domains: dense counting is O(n + range) and beats
            // any comparison sort there; the probe samples and declines fast.
            // A 64-key strided sample bounds the range from below; only a
            // range small beside n gets the (heavier) counting probe.
            if constexpr (std::is_integral<T>::value) {
                bool narrow = false;
                if (n >= detail::kCountingMinN) {
                    using UK = typename std::make_unsigned<T>::type;
                    T lo = p[0], hi = p[0];
                    for (std::size_t j = 1; j < 64; ++j) {
                        const T x = p[(j * n) >> 6];
                        lo = x < lo ? x : lo;
                        hi = hi < x ? x : hi;
                    }
                    const UK span = static_cast<UK>(static_cast<UK>(hi) - static_cast<UK>(lo));
                    narrow = static_cast<unsigned long long>(span) <=
                             static_cast<unsigned long long>(std::max<std::size_t>(4096, n / 8));
                }
                if (narrow && detail::try_integer_range_count_sort<T>(p, n, descending, nullptr)) {
                    detail::record_dispatch(detail::DispatchDecision::LowCardinality);
                    return;
                }
            }
            // A few hundred to ~2000 distinct keys: hash counting for 8-byte
            // keys.  Clean 4-byte ranges stay with the vq: with its leaf
            // screen (one- / two-valued leaves skip the network) it beats
            // the table at every measured n and key count (100k x 256 keys:
            // 1.24 against 2.27 ns/key; 1000 keys: 2.63 against 2.95).
            if ((sizeof(T) != 4 || !clean) && detail::try_hash_count_sort(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::LowCardinality);
                return;
            }
            // NaN / unclean ranges take the general path below.
            if (clean && (pre & 8u) == 0) {
                detail::vqsort_serial(p, n);
                if (descending) detail::reverse_range_adaptive(p, n);
                detail::record_dispatch(detail::DispatchDecision::VectorQuick);
                return;
            }
            // Cleanliness not yet established (floats): the first partition
            // screens for NaN / -0 as it goes; on a hit the (permuted) range
            // continues down the general path.
            if (clean && detail::vqsort_serial_checked(p, n)) {
                if (descending) detail::reverse_range_adaptive(p, n);
                detail::record_dispatch(detail::DispatchDecision::VectorQuick);
                return;
            }
        }
    }

    // The NaN / -0 scan is memoised per top-level call from here on (the
    // large-n probes may ask several times about the same range).
    const detail::VqCleanMemoScope vq_clean_scope;

#if FYX_ENABLE_FAST_PATHS
#if FYX_ENABLE_PARALLEL
    const bool kv16_par = n >= detail::kKv16EarlyParallelMinN && detail::dynamic_parallel_allowed<T>(n, o);
#else
    const bool kv16_par = false;
#endif
    if (detail::try_kv16_early_sort(p, n, comp, kv16_par)) {
        detail::record_dispatch(detail::DispatchDecision::VectorQuick);
        return;
    }
    if (n > detail::kNetworkMax && detail::try_parallel_all_equal_exit(p, n, comp)) return;
    if (n > detail::kNetworkMax && detail::try_zigzag_organ_pipe_sort(p, n, comp)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    if (n > detail::kNetworkMax && detail::likely_mid_bitonic_runs(p, n, comp) &&
        detail::try_bitonic_runs_sort(p, n, comp)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
#if FYX_ENABLE_PARALLEL
    const bool rev_parallel_ok = detail::dynamic_parallel_allowed<T>(n, o);
#else
    const bool rev_parallel_ok = false;
#endif
    if (n > detail::kNetworkMax && detail::try_fast_reverse_exit(p, n, comp, rev_parallel_ok)) {
        detail::record_dispatch(detail::DispatchDecision::ProfileReverse);
        return;
    }
    if (n > detail::kNetworkMax && detail::try_numeric_half_organ_fill(p, n, comp)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    if (n > detail::kNetworkMax && detail::try_string_half_organ_reorder(p, n, comp)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    if (n > detail::kNetworkMax && !std::is_arithmetic<T>::value &&
        detail::likely_mid_bitonic_runs(p, n, comp) &&
        detail::try_bitonic_runs_sort(p, n, comp)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    if (n > detail::kNetworkMax) {
        if (detail::try_order_exit_adaptive(p, n, comp, true, rev_parallel_ok)) return;
        // One-break structural proof: rotated sorted ranges (both runs are
        // individually sorted, so the sampled order exits cannot see them).
        // Rotating back is O(n) and stable; random input declines within the
        // first few elements after its first inversion.
        if (radix_ok) {
#if FYX_ENABLE_PARALLEL
            if (rev_parallel_ok && detail::try_one_break_rotate_parallel(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::ProfileSorted);
                return;
            }
#endif
            if (detail::try_one_break_rotate(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::ProfileSorted);
                return;
            }
#if FYX_ENABLE_PARALLEL
            if (rev_parallel_ok && detail::try_proof_structured_sort_parallel(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::ProfileSorted);
                return;
            }
#endif
            if (!rev_parallel_ok && detail::try_proof_structured_sort(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::ProfileSorted);
                return;
            }
        }
    }
    // Inputs made of a handful of monotone runs (rotated sorted arrays,
    // concatenated sorted blocks, shuffled block permutations, ...) cost
    // O(n log R) sequential moves here instead of a fixed 4-8 radix passes or
    // a full comparison recursion.  Random data is rejected inside the scan
    // after touching a couple of hundred elements.
    // Bounded insertion, thorough mode, tried before the natural-run scan.
    // Sparse *displaced runs* -- block swaps, permuted chunks -- are
    // insertion's killer case: 8M double blockswap sorts in ~0.011 s here
    // against ~0.06 s when the natural-run characterisation (a full scan that
    // then declines) and the profile scan run first.  Thorough mode rehearses
    // on a copy of the prefix before permuting anything and its shift budgets
    // refuse every shape it cannot finish within a small multiple of n:
    // concat2/rotated (natural-run's shapes) and random data fall through in
    // a few milliseconds or less, which is what keeps this win from taxing
    // them.
    if (n > detail::kNetworkMax && detail::try_record_runs_merge(p, n, comp, rev_parallel_ok)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    // Records: the affix weapon straight after the run merge (neither moves
    // anything when declining, so the order exit's break hint still holds),
    // before insertion repair and the natural-run scan pay to rescan the head.
    const bool affix_early = !std::is_arithmetic<T>::value;
    if (affix_early && n > detail::kNetworkMax && detail::try_sorted_affix_sort(p, n, comp, rev_parallel_ok)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    if (n > detail::kNetworkMax && detail::try_bounded_insertion_repair(p, n, comp, true)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
#if FYX_ENABLE_PARALLEL
    const bool fd_par = detail::dynamic_parallel_allowed<T>(n, o);
#else
    const bool fd_par = false;
#endif
    if (n > detail::kNetworkMax && detail::try_natural_run_merge_adaptive(p, n, comp)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    // Ordered except for one stretch in the middle: see
    // try_sorted_affix_sort.  Like the run merge this has to be reachable
    // before the parallel kernels are chosen, or a range that is three
    // quarters sorted pays for all of it on every worker.
    if (!affix_early && n > detail::kNetworkMax && detail::try_sorted_affix_sort(p, n, comp, fd_par)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    if (n > detail::kNetworkMax && detail::try_adjacent_swap_repair(p, n, comp)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    if (n > detail::kNetworkMax && detail::try_interleaved_runs_sort(p, n, comp)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    if (n > detail::kNetworkMax && detail::try_bitonic_runs_sort(p, n, comp)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    if (n <= detail::kProfilePartialPdqMax && detail::pdq_preferred_order_sample(p, n, comp)) {
        if (radix_ok) {
#if FYX_ENABLE_PARALLEL
            const bool par_here = detail::dynamic_parallel_allowed<T>(n, o);
#else
            const bool par_here = false;
#endif
            const std::size_t patch_budget = detail::patch_merge_dirty_budget<T>(n, par_here);
            if (detail::try_partially_sorted_local_repair(p, n, comp, patch_budget)) {
                detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                return;
            }
            if (detail::try_radix_permutation_range_sort(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::Radix);
                return;
            }
            if (detail::try_vector_quicksort(p, n, descending, par_here)) {
                detail::record_dispatch(detail::DispatchDecision::VectorQuick);
                return;
            }
            // The quicksort declined (no AVX-512, a NaN in the range, ...):
            // the patch merges are the best thing left, so give them the turn
            // that was held back for it.
            if (patch_budget != detail::kPatchDirtyDefault &&
                detail::try_partially_sorted_local_repair(p, n, comp)) {
                detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                return;
            }
            // Numeric long-distance nearly-sorted inputs should not pay the
            // full profile scan plus comparison pdq.  Once adjacent/local
            // repairs decline, send them straight to the radix family.
#if FYX_ENABLE_PARALLEL
            if (detail::dynamic_parallel_allowed<T>(n, o)) {
                if (detail::try_parallel_radix_high_prefix_sort(p, n, descending) ||
                    detail::try_parallel_radix32_wide_sort(p, n, descending) ||
                    detail::try_parallel_radix_sort(p, n, descending, true)) {
                    detail::record_dispatch(detail::DispatchDecision::Radix);
                    return;
                }
            }
            if (detail::try_serial_radix_high_prefix_sort(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::Radix);
                return;
            }
#endif
            if constexpr (detail::radix_supported_v<T>) {
                if (detail::radix_sort(p, n)) {
                    if (descending) std::reverse(p, p + n);
                    detail::record_dispatch(detail::DispatchDecision::Radix);
                    return;
                }
            }
        } else {
            if (detail::try_guarded_radix_order_sort(p, n, comp,
#if FYX_ENABLE_PARALLEL
                    detail::dynamic_parallel_allowed<T>(n, o),
#else
                    false,
#endif
                    true)) {
                detail::record_dispatch(detail::DispatchDecision::Radix);
                return;
            }
            if (detail::partial_runs_look_independent(p, n, comp) && detail::try_key_radix_for_partial(p, n, comp,
#if FYX_ENABLE_PARALLEL
                    detail::dynamic_parallel_allowed<T>(n, o)
#else
                    false
#endif
                    )) { detail::record_dispatch(detail::DispatchDecision::Radix); return; } if (detail::try_partially_sorted_repair(p, n, comp)) {
                detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                return;
            }
            if (detail::try_key_radix_for_partial(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; } detail::pdqsort_for_profile_pattern(p, n, comp);
            detail::record_dispatch(detail::DispatchDecision::PartialPdq);
            return;
        }
    }
#endif

    detail::InputProfile<T, Comp> top_profile;
    const detail::InputProfile<T, Comp>* prof = nullptr;
    if (n >= detail::kProfileMinN) {
        top_profile = detail::profile_input(p, n, comp);
        prof = &top_profile;
        if (detail::apply_profile_fast_exit(p, n, top_profile, true)) return;
    }

#if FYX_ENABLE_PARALLEL
    const bool want_parallel = detail::dynamic_parallel_allowed<T>(n, o);
    // Before creating tasks, let the unified top-level profile and O(n)
    // counting/string/key-specialized paths win the whole range.  High-entropy
    // samples skip low-cardinality probes, but still allow MSD string radix and
    // comparator-key radix fast paths.
    if (want_parallel && n > detail::kNetworkMax) {
        const bool high_entropy = prof && prof->is_high_entropy;
        const bool partial_pdq = prof && prof->is_partially_sorted &&
            n <= detail::kProfilePartialPdqMax;
        if (partial_pdq && !(prof && prof->is_low_cardinality)) {
            if (radix_ok) {
                const std::size_t patch_budget = detail::patch_merge_dirty_budget<T>(n, want_parallel);
                if (detail::try_partially_sorted_local_repair(p, n, comp, patch_budget)) {
                    detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                    return;
                }
                if (detail::try_vector_quicksort(p, n, descending, want_parallel)) {
                    detail::record_dispatch(detail::DispatchDecision::VectorQuick);
                    return;
                }
                if (patch_budget != detail::kPatchDirtyDefault &&
                detail::try_partially_sorted_local_repair(p, n, comp)) {
                    detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                    return;
                }
                if (detail::try_radix_permutation_range_sort(p, n, descending)) {
                    detail::record_dispatch(detail::DispatchDecision::Radix);
                    return;
                }
                // Numeric nearly-sorted with remote swaps falls through to
                // radix; adjacent/local repairs have already had first chance.
            } else {
                if (detail::try_guarded_radix_order_sort(p, n, comp, want_parallel, true)) {
                    detail::record_dispatch(detail::DispatchDecision::Radix);
                    return;
                }
                if (detail::partial_runs_look_independent(p, n, comp) && detail::try_key_radix_for_partial(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; } if (detail::try_partially_sorted_repair(p, n, comp)) {
                    detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                    return;
                }
                if (detail::try_key_radix_for_partial(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; } detail::pdqsort_for_profile_pattern(p, n, comp);
                detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                return;
            }
        }
        if (radix_ok) {
            if (!high_entropy) {
                const bool low_card_hint = prof &&
                    (prof->is_low_cardinality_candidate || prof->is_low_cardinality);
                if (low_card_hint) {
                    // The profile hint is only a hint: each counting path still
                    // validates the full range before committing.  Dense integer
                    // range counting is sample-gated so sparse huge-span data can
                    // decline before paying a full min/max scan.
                    if (detail::try_integer_range_count_sort_parallel(p, n, descending, prof ? &prof->sample_window : nullptr)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_dense_prefix_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_rank16_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_integer_sparse_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_sparse_count_sort_parallel(p, n, descending,
                        (prof && prof->sample_window.distinct) ? prof->sample_window.distinct : 0)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_sparse_count_sort(p, n, descending, prof ? &prof->sample_window : nullptr)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_integer_range_count_sort(p, n, descending, prof ? &prof->sample_window : nullptr)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                } else {
                    if (detail::try_integer_range_count_sort_parallel(p, n, descending, prof ? &prof->sample_window : nullptr)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_integer_range_count_sort(p, n, descending, prof ? &prof->sample_window : nullptr)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_dense_prefix_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_rank16_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_integer_sparse_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_sparse_count_sort_parallel(p, n, descending,
                        (prof && prof->sample_window.distinct) ? prof->sample_window.distinct : 0)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_sparse_count_sort(p, n, descending, prof ? &prof->sample_window : nullptr)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                }
                if (detail::try_low_cardinality_count_sort(p, p + n, comp)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
            }
            if (partial_pdq && detail::try_partially_sorted_local_repair(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::PartialPdq); return; }
            if (detail::try_radix_permutation_range_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            // See sort_st: narrow range is not low cardinality, and the range
            // kernels self-gate, so high-entropy input gets a chance too.
            if (high_entropy && detail::try_integer_range_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
            if (detail::try_vector_quicksort(p, n, descending, true)) { detail::record_dispatch(detail::DispatchDecision::VectorQuick); return; }
            if (high_entropy && detail::try_parallel_radix_high_prefix_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (high_entropy && detail::try_parallel_radix32_wide_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (high_entropy && detail::try_msd_radix_bucket_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (detail::try_parallel_radix_sort(p, n, descending, high_entropy)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
        } else {
            if (detail::try_guarded_radix_order_sort(p, n, comp, want_parallel, high_entropy)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (!high_entropy) {
                if (detail::try_string_value_count_sort_parallel(p, n, comp, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                if (detail::try_string_value_count_sort(p, n, comp, false)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                if (detail::try_guarded_string_value_count_sort(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                if (detail::try_trivial_prefix_key_count_sort_parallel(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                if (detail::try_trivial_prefix_key_count_sort(p, n, comp, true)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                if (detail::try_low_cardinality_count_sort(p, p + n, comp)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
            }
            if (partial_pdq) { if (detail::partial_runs_look_independent(p, n, comp) && detail::try_key_radix_for_partial(p, n, comp, true)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; } if (detail::try_partially_sorted_repair(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::PartialPdq); return; } if (detail::try_key_radix_for_partial(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; } detail::pdqsort_for_profile_pattern(p, n, comp); detail::record_dispatch(detail::DispatchDecision::PartialPdq); return; }
            if (detail::try_guarded_string_order_sort(p, n, comp, want_parallel)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (detail::try_string_msd_sort_parallel(p, n, comp, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (detail::try_string_msd_sort(p, n, comp, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (detail::try_trivial_prefix_key_radix_sort(p, n, comp, true)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
        }
    }
#endif

    if (radix_ok) {
#if FYX_ENABLE_PARALLEL
        if (want_parallel) {
            const bool high_entropy = prof && prof->is_high_entropy;
            if (detail::try_radix_permutation_range_sort(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::Radix);
                return;
            }
            if (detail::try_vector_quicksort(p, n, descending, true)) {
                detail::record_dispatch(detail::DispatchDecision::VectorQuick);
                return;
            }
            if (high_entropy && detail::try_parallel_radix_high_prefix_sort(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::Radix);
                return;
            }
            if (high_entropy && detail::try_parallel_radix32_wide_sort(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::Radix);
                return;
            }
            if (high_entropy && detail::try_msd_radix_bucket_sort(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::Radix);
                return;
            }
            if (detail::try_parallel_radix_sort(p, n, descending, high_entropy)) {
                detail::record_dispatch(detail::DispatchDecision::Radix);
                return;
            }
            detail::parallel_sort_ptr(p, n, comp, descending, o);
            return;
        }
#endif
        detail::sort_st(p, n, comp, descending, prof);
        return;
    }

#if FYX_ENABLE_PARALLEL
    if (want_parallel) {
        const bool high_entropy = prof && prof->is_high_entropy;
        if (detail::try_guarded_radix_order_sort(p, n, comp, true, high_entropy)) {
            detail::record_dispatch(detail::DispatchDecision::Radix);
            return;
        }
        if (detail::try_guarded_string_order_sort(p, n, comp, true)) {
            detail::record_dispatch(detail::DispatchDecision::Radix);
            return;
        }
        // The parallel sample sort samples by copying, so a move-only payload
        // takes the task-parallel divide and conquer instead, which only
        // moves -- see merge_runs_moving.
        if constexpr (std::is_copy_constructible<T>::value) {
            if (n >= detail::kSampleThreshold) {
                detail::parallel_sample_sort(p, p + n, comp);
                detail::record_dispatch(detail::DispatchDecision::ParallelSample);
                return;
            }
        }
        detail::parallel_sort_ptr(p, n, comp, descending, o);
        return;
    }
#endif
    detail::sort_st(p, n, comp, descending, prof);
}

/// The kernels above allocate: scratch for radix and merges, buffers for the
/// sample sort, worker threads for the parallel paths.  std::sort never
/// allocates and therefore never fails for want of memory; a sorter whose fast
/// paths do must not inherit that failure, so out of memory -- or no address
/// space left to start a worker -- falls back to the in-place comparison sort,
/// which needs neither.  The range is already permuted by whatever threw, and
/// pdqsort is happy to start from any permutation.
template <class T, class Comp>
inline void sort_pointer_core(T* p, std::size_t n, Comp comp, const Options& o) {
#if FYX_HAS_EXCEPTIONS
    try {
        sort_pointer_core_impl(p, n, comp, o);
    } catch (const std::bad_alloc&) {
        detail::pdqsort(p, p + n, comp);
    } catch (const std::system_error&) {
        detail::pdqsort(p, p + n, comp);
    }
#else
    sort_pointer_core_impl(p, n, comp, o);
#endif
}

template <class It, class Comp>
inline void sort_iter_core(It first, It last, Comp comp, const Options& o) {
    (void)o;
    const auto n0 = last - first;
    if (n0 == 0) return;
    // Raw pointers and libstdc++/libc++ vector/string normal iterators are
    // contiguous: route them through the pointer dispatcher so numeric radix,
    // sparse count and top-level profiling are not lost when callers write
    // fyx::sort(v.begin(), v.end()) instead of fyx::sort(v).
    if constexpr (std::is_pointer_v<It>) {
        sort_pointer_core(first, static_cast<std::size_t>(n0), comp, o);
        return;
    } else if constexpr (detail::has_mutable_base_pointer_v<It>) {
        sort_pointer_core(detail::iterator_base_pointer<It>::get(first),
                          static_cast<std::size_t>(n0), comp, o);
        return;
    }
    const std::size_t n = static_cast<std::size_t>(n0);
    if (n <= detail::kInsertionThreshold) {
        detail::insertion_sort(first, last, comp);
        return;
    }
    if (detail::try_monotonic_sort(first, last, comp, true)) return;
    // Segmented storage (std::deque) cannot be indexed, so radix, the adaptive
    // weapons and the parallel pool are all out of reach here -- and `o` used
    // to be dropped on the floor, so asking for parallel silently cost the
    // same as not asking.  Buffering gets them back: see
    // try_buffered_iter_sort.
    if (try_buffered_iter_sort(first, last, comp, o, n)) return;
    if (detail::try_low_cardinality_count_sort(first, last, comp)) return;
    using T = typename std::iterator_traits<It>::value_type;
    if (n >= detail::kSampleThreshold) {
        if constexpr (std::is_copy_constructible<T>::value) {
            detail::sample_sort(first, last, comp);
            return;
        }
    }
    detail::pdqsort(first, last, comp);
}

template <class Container, class Comp>
inline void sort_container_core(Container& c, Comp comp, const Options& o) {
    if constexpr (detail::has_std_data_v<Container>) {
        auto* p = std::data(c);
        const std::size_t n = static_cast<std::size_t>(std::size(c));
        sort_pointer_core(p, n, comp, o);
    } else if constexpr (detail::has_member_sort_with_v<Container, Comp>) {
        // Node-based: see has_member_sort.  Nothing to gain from a buffer --
        // the elements are never moved -- and Options cannot apply, since
        // splicing is not something worth spreading across workers.
        c.sort(comp);
    } else {
        sort_iter_core(std::begin(c), std::end(c), comp, o);
    }
}

/// Sorts a range that the contiguous kernels cannot address -- a segmented
/// container, or anything whose iterators cannot be indexed -- by moving the
/// elements into a buffer, sorting that, and moving them back.
///
/// Two extra passes buy everything a vector gets: radix, the adaptive weapons,
/// the profile, and the parallel pool.  1M int32 in a std::deque is 0.051s
/// through the iterator kernels and 0.017s this way, against 0.084s for
/// std::sort.
///
/// Returns false only when the buffer cannot be had, in which case nothing has
/// been moved.  If the sort itself throws, the elements go back where they
/// came from: unsorted, but the range still owns every one of them.
template <class It, class Comp>
inline bool try_buffered_iter_sort(It first, It last, Comp comp, const Options& o, std::size_t n) {
    using T = typename std::iterator_traits<It>::value_type;
    if constexpr (!std::is_move_constructible<T>::value ||
                  !std::is_move_assignable<T>::value) {
        (void)first; (void)last; (void)comp; (void)o; (void)n;
        return false;
    } else {
        std::vector<T> buf;
        // Without exceptions there is nothing to catch: an allocation that
        // cannot be served terminates, and a move that cannot be made cannot
        // report it.  The recovery arms below exist to leave the range holding
        // every element it started with, which is only reachable when a throw
        // is possible in the first place.
#if FYX_HAS_EXCEPTIONS
        try {
            buf.reserve(n);
        } catch (...) {
            return false;
        }
        std::size_t taken = 0;
        try {
            for (It it = first; it != last; ++it) { buf.push_back(std::move(*it)); ++taken; }
        } catch (...) {
            It out = first;
            for (std::size_t i = 0; i < taken; ++i) { *out = std::move(buf[i]); ++out; }
            return false;
        }
        try {
            sort_pointer_core(buf.data(), buf.size(), comp, o);
        } catch (...) {
            It out = first;
            for (std::size_t i = 0; i < n; ++i) { *out = std::move(buf[i]); ++out; }
            throw;
        }
#else
        buf.reserve(n);
        for (It it = first; it != last; ++it) buf.push_back(std::move(*it));
        sort_pointer_core(buf.data(), buf.size(), comp, o);
#endif
        It out = first;
        for (std::size_t i = 0; i < n; ++i) { *out = std::move(buf[i]); ++out; }
        return true;
    }
}

template <class It, class Comp>
inline void sort_forward_core(It first, It last, Comp comp, const Options& o) {
    std::size_t n = 0;
    for (It it = first; it != last; ++it) ++n;
    if (n < 2) return;
    if (try_buffered_iter_sort(first, last, comp, o, n)) return;
    // No buffer to be had.  Quadratic, but it is the only thing left that
    // works through a forward iterator, and it is reached only when the
    // allocation for the buffer has already failed.
    for (It i = first; i != last; ++i) {
        typename std::iterator_traits<It>::value_type v = std::move(*i);
        It j = i;
        It prev = j;
        bool done = false;
        while (!done) {
            if (j == first) { done = true; break; }
            --prev;
            if (comp(v, *prev)) { *j = std::move(*prev); j = prev; prev = j; }
            else                { done = true; }
        }
        *j = std::move(v);
    }
}

// ===========================================================================
//  fyx::sort  -- the main entry point
// ===========================================================================

// ---- pointer + length ------------------------------------------------------
template <class T>
inline void sort(T* p, std::size_t n) {
    sort_pointer_core(p, n, fyx::less{}, Options{});
}
template <class T, class Comp,
          std::enable_if_t<!detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void sort(T* p, std::size_t n, Comp comp) {
    sort_pointer_core(p, n, comp, Options{});
}
template <class T>
inline void sort(T* p, std::size_t n, const Options& o) {
    sort_pointer_core(p, n, fyx::less{}, o);
}
template <class T, class Comp,
          std::enable_if_t<!detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void sort(T* p, std::size_t n, Comp comp, const Options& o) {
    sort_pointer_core(p, n, comp, o);
}

// ---- iterator pair ---------------------------------------------------------
template <class It,
          std::enable_if_t<detail::is_random_access_v<It>, int> = 0>
inline void sort(It first, It last) {
    sort_iter_core(first, last, fyx::less{}, Options{});
}
// Anything less than random access: see sort_forward_core.
template <class It,
          std::enable_if_t<!detail::is_random_access_v<It>, int> = 0>
inline void sort(It first, It last) {
    sort_forward_core(first, last, fyx::less{}, Options{});
}
template <class It, class Comp,
          std::enable_if_t<detail::is_random_access_v<It> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void sort(It first, It last, Comp comp) {
    sort_iter_core(first, last, comp, Options{});
}
template <class It, class Comp,
          std::enable_if_t<!detail::is_random_access_v<It> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void sort(It first, It last, Comp comp) {
    sort_forward_core(first, last, comp, Options{});
}
template <class It,
          std::enable_if_t<detail::is_random_access_v<It>, int> = 0>
inline void sort(It first, It last, const Options& o) {
    sort_iter_core(first, last, fyx::less{}, o);
}
template <class It,
          std::enable_if_t<!detail::is_random_access_v<It>, int> = 0>
inline void sort(It first, It last, const Options& o) {
    sort_forward_core(first, last, fyx::less{}, o);
}
template <class It, class Comp,
          std::enable_if_t<detail::is_random_access_v<It> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void sort(It first, It last, Comp comp, const Options& o) {
    sort_iter_core(first, last, comp, o);
}
template <class It, class Comp,
          std::enable_if_t<!detail::is_random_access_v<It> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void sort(It first, It last, Comp comp, const Options& o) {
    sort_forward_core(first, last, comp, o);
}

// ---- contiguous container --------------------------------------------------
template <class Container,
          std::enable_if_t<!std::is_pointer_v<std::remove_reference_t<Container>> &&
                           !std::is_array_v<std::remove_reference_t<Container>>, int> = 0>
inline void sort(Container& c) {
    sort_container_core(c, fyx::less{}, Options{});
}
template <class Container, class Comp,
          std::enable_if_t<!std::is_pointer_v<std::remove_reference_t<Container>> &&
                           !std::is_array_v<std::remove_reference_t<Container>> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void sort(Container& c, Comp comp) {
    sort_container_core(c, comp, Options{});
}
template <class Container,
          std::enable_if_t<!std::is_pointer_v<std::remove_reference_t<Container>> &&
                           !std::is_array_v<std::remove_reference_t<Container>>, int> = 0>
inline void sort(Container& c, const Options& o) {
    sort_container_core(c, fyx::less{}, o);
}
template <class Container, class Comp,
          std::enable_if_t<!std::is_pointer_v<std::remove_reference_t<Container>> &&
                           !std::is_array_v<std::remove_reference_t<Container>> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void sort(Container& c, Comp comp, const Options& o) {
    sort_container_core(c, comp, o);
}

// ===========================================================================
//  fyx::stable_sort
// ===========================================================================

template <class T, class Comp>
inline void stable_sort_dispatch(T* p, std::size_t n, Comp comp) {
    if (n < 2) return;
    const bool ascending  = detail::is_ascending_v<Comp, T>;
    const bool descending = detail::is_descending_v<Comp, T>;
    // Integers ordered by the standard "<" / ">" comparators: elements that
    // compare equal are bit-identical, so stability is unobservable and the
    // (faster, adaptive) unstable engine produces the exact same output.
    // Kept serial: stable_sort has no Options and never spawned threads.
    if constexpr (std::is_integral<T>::value) {
        if (ascending || descending) {
            Options o;
            o.parallel = Tri::Off;
            sort_pointer_core(p, n, comp, o);
            return;
        }
    }
    if (detail::radix_supported_v<T> && (ascending || descending)) {
        if (detail::try_radix_monotonic_sort(p, n, descending, false)) return;
    } else {
        if (detail::try_monotonic_sort(p, p + n, comp, false)) return;
        if (detail::try_stable_reverse_sort(p, p + n, comp)) return;
    }
    if (ascending || descending) {
        if (detail::try_integer_range_count_sort(p, n, descending)) return;
        if (detail::try_integer_sparse_count_sort(p, n, descending)) return;
        if (detail::try_string_value_count_sort(p, n, comp, descending)) return;
        if (detail::try_string_msd_sort(p, n, comp, descending)) return;
    }
    if (detail::try_low_cardinality_count_sort(p, p + n, comp)) return;

    // Radix is naturally stable and matches the default "<" order exactly.
    if constexpr (detail::radix_supported_v<T>) {
        if (ascending) {
            if (detail::radix_sort(p, n)) return;   // OOM -> fall through
        }
    }
    detail::stable_merge_sort(p, p + n, comp);
}

// ---- pointer + length ------------------------------------------------------
template <class T>
inline void stable_sort(T* p, std::size_t n) {
    stable_sort_dispatch(p, n, fyx::less{});
}
template <class T, class Comp,
          std::enable_if_t<!detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void stable_sort(T* p, std::size_t n, Comp comp) {
    stable_sort_dispatch(p, n, comp);
}

// ---- iterator pair ---------------------------------------------------------
template <class It,
          std::enable_if_t<detail::is_random_access_v<It>, int> = 0>
inline void stable_sort(It first, It last) {
    if constexpr (std::is_pointer_v<It>) {
        stable_sort_dispatch(first, static_cast<std::size_t>(last - first), fyx::less{});
    } else {
        if (last - first < 2) return;
        if (detail::try_monotonic_sort(first, last, fyx::less{}, false)) return;
        if (detail::try_stable_reverse_sort(first, last, fyx::less{})) return;
        if (detail::try_low_cardinality_count_sort(first, last, fyx::less{})) return;
        detail::stable_merge_sort(first, last, fyx::less{});
    }
}
template <class It, class Comp,
          std::enable_if_t<detail::is_random_access_v<It> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void stable_sort(It first, It last, Comp comp) {
    if constexpr (std::is_pointer_v<It>) {
        stable_sort_dispatch(first, static_cast<std::size_t>(last - first), comp);
    } else {
        if (last - first < 2) return;
        if (detail::try_monotonic_sort(first, last, comp, false)) return;
        if (detail::try_stable_reverse_sort(first, last, comp)) return;
        if (detail::try_low_cardinality_count_sort(first, last, comp)) return;
        detail::stable_merge_sort(first, last, comp);
    }
}

// ---- contiguous container --------------------------------------------------
template <class Container,
          std::enable_if_t<detail::has_std_data_v<Container> &&
                           !std::is_pointer_v<std::remove_reference_t<Container>> && !std::is_array_v<std::remove_reference_t<Container>>, int> = 0>
inline void stable_sort(Container& c) {
    stable_sort_dispatch(std::data(c), static_cast<std::size_t>(std::size(c)), fyx::less{});
}
template <class Container, class Comp,
          std::enable_if_t<detail::has_std_data_v<Container> &&
                           !std::is_pointer_v<std::remove_reference_t<Container>> && !std::is_array_v<std::remove_reference_t<Container>> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void stable_sort(Container& c, Comp comp) {
    stable_sort_dispatch(std::data(c), static_cast<std::size_t>(std::size(c)), comp);
}

// ===========================================================================
//  fyx::partial_sort
// ===========================================================================

// ---- iterator triple -------------------------------------------------------
template <class It,
          std::enable_if_t<detail::is_random_access_v<It>, int> = 0>
inline void partial_sort(It first, It middle, It last) {
    detail::partial_sort_impl(first, middle, last, fyx::less{});
}
template <class It, class Comp,
          std::enable_if_t<detail::is_random_access_v<It> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void partial_sort(It first, It middle, It last, Comp comp) {
    detail::partial_sort_impl(first, middle, last, comp);
}

// ---- contiguous container + count -----------------------------------------
template <class Container, class Comp,
          std::enable_if_t<detail::has_std_data_v<Container> &&
                           !std::is_pointer_v<std::remove_reference_t<Container>> && !std::is_array_v<std::remove_reference_t<Container>> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void partial_sort(Container& c, std::size_t middle_n, Comp comp) {
    auto first = std::begin(c);
    detail::partial_sort_impl(first, first + middle_n, std::end(c), comp);
}
template <class Container,
          std::enable_if_t<detail::has_std_data_v<Container> &&
                           !std::is_pointer_v<std::remove_reference_t<Container>> && !std::is_array_v<std::remove_reference_t<Container>>, int> = 0>
inline void partial_sort(Container& c, std::size_t middle_n) {
    auto first = std::begin(c);
    detail::partial_sort_impl(first, first + middle_n, std::end(c), fyx::less{});
}

// ===========================================================================
//  fyx::nth_element
// ===========================================================================

// ---- iterator triple -------------------------------------------------------
template <class It,
          std::enable_if_t<detail::is_random_access_v<It>, int> = 0>
inline void nth_element(It first, It nth, It last) {
    detail::nth_element_impl(first, nth, last, fyx::less{});
}
template <class It, class Comp,
          std::enable_if_t<detail::is_random_access_v<It> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void nth_element(It first, It nth, It last, Comp comp) {
    detail::nth_element_impl(first, nth, last, comp);
}

// ---- contiguous container + count -----------------------------------------
template <class Container, class Comp,
          std::enable_if_t<detail::has_std_data_v<Container> &&
                           !std::is_pointer_v<std::remove_reference_t<Container>> && !std::is_array_v<std::remove_reference_t<Container>> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void nth_element(Container& c, std::size_t nth_n, Comp comp) {
    auto first = std::begin(c);
    detail::nth_element_impl(first, first + nth_n, std::end(c), comp);
}
template <class Container,
          std::enable_if_t<detail::has_std_data_v<Container> &&
                           !std::is_pointer_v<std::remove_reference_t<Container>> && !std::is_array_v<std::remove_reference_t<Container>>, int> = 0>
inline void nth_element(Container& c, std::size_t nth_n) {
    auto first = std::begin(c);
    detail::nth_element_impl(first, first + nth_n, std::end(c), fyx::less{});
}

} // namespace fyx

// ===========================================================================
//  C ABI -- C-linkage, inline (safe to include from multiple TUs).  To call
//  these from C, compile one .cpp that #includes this header and references
//  the symbols; they are emitted with external C linkage there.
// ===========================================================================
// The wrappers report failure through the return value, so with exceptions
// switched off (-fno-exceptions / FYX_NO_EXCEPTIONS) they simply call through:
// the library's own out-of-memory handling degrades to the in-place kernels
// there, and nothing else can throw.
#if FYX_HAS_EXCEPTIONS
#  define FYX_C_ABI_BODY(call) try { call; return 0; } catch (...) { return -1; }
#else
#  define FYX_C_ABI_BODY(call) call; return 0;
#endif

extern "C" {
inline int fyx_sort_int32 (std::int32_t*  d, std::size_t n) noexcept { FYX_C_ABI_BODY(fyx::sort(d, n)) }
inline int fyx_sort_uint32(std::uint32_t* d, std::size_t n) noexcept { FYX_C_ABI_BODY(fyx::sort(d, n)) }
inline int fyx_sort_int64 (std::int64_t*  d, std::size_t n) noexcept { FYX_C_ABI_BODY(fyx::sort(d, n)) }
inline int fyx_sort_uint64(std::uint64_t* d, std::size_t n) noexcept { FYX_C_ABI_BODY(fyx::sort(d, n)) }
inline int fyx_sort_float (float*  d, std::size_t n) noexcept { FYX_C_ABI_BODY(fyx::sort(d, n)) }
inline int fyx_sort_double(double* d, std::size_t n) noexcept { FYX_C_ABI_BODY(fyx::sort(d, n)) }
}
