// ============================================================================
//  Section 6 -- Scalar kernels
//    * branch-free insertion sort (used *inside* pdqsort, never for the
//      n <= 64 numeric entry point, which always uses a sorting network)
//    * heapsort (pdqsort's depth-limit fallback -- guarantees O(n log n))
//    * scalar branch-free bitonic network (the universal fallback for the
//      n <= 64 path on targets with no usable SIMD)
// ============================================================================

namespace fyx {
namespace detail {

// ---------------------------------------------------------------------------
// Insertion sort
// ---------------------------------------------------------------------------

/// Plain insertion sort over [first, last).
template <typename It, typename Compare>
inline void insertion_sort(It first, It last, Compare comp) {
    using T = typename std::iterator_traits<It>::value_type;
    if (first == last) return;
    for (It i = first + 1; i != last; ++i) {
        if (comp(*i, *(i - 1))) {
            T   tmp = std::move(*i);
            It  j   = i;
            do {
                *j = std::move(*(j - 1));
                --j;
            } while (j != first && comp(tmp, *(j - 1)));
            *j = std::move(tmp);
        }
    }
}

/// Insertion sort that may assume *(first - 1) is a valid element that is not
/// greater than everything in [first, last) -- i.e. a sentinel exists.  This
/// removes the `j != first` bound check from the inner loop.
template <typename It, typename Compare>
inline void insertion_sort_guarded(It first, It last, Compare comp) {
    using T = typename std::iterator_traits<It>::value_type;
    if (first == last) return;
    for (It i = first + 1; i != last; ++i) {
        if (comp(*i, *(i - 1))) {
            T  tmp = std::move(*i);
            It j   = i;
            do {
                *j = std::move(*(j - 1));
                --j;
            } while (comp(tmp, *(j - 1)));
            *j = std::move(tmp);
        }
    }
}

/// Bounded insertion sort used by pdqsort's partial-order optimisation.
/// Gives up (returning false) once it has moved more than `limit` elements,
/// which tells the caller the range is not nearly sorted.
template <typename It, typename Compare>
inline bool partial_insertion_sort(It first, It last, Compare comp) {
    using T = typename std::iterator_traits<It>::value_type;
    using Diff = typename std::iterator_traits<It>::difference_type;
    constexpr Diff kLimit = 8;

    if (first == last) return true;
    Diff moved = 0;
    for (It i = first + 1; i != last; ++i) {
        if (!comp(*i, *(i - 1))) continue;
        T  tmp = std::move(*i);
        It j   = i;
        do {
            *j = std::move(*(j - 1));
            --j;
        } while (j != first && comp(tmp, *(j - 1)));
        *j = std::move(tmp);
        moved += i - j;
        if (moved > kLimit) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Heapsort -- the worst-case guarantee behind pdqsort
// ---------------------------------------------------------------------------

template <typename It, typename Compare>
inline void sift_down(It first, typename std::iterator_traits<It>::difference_type root,
                      typename std::iterator_traits<It>::difference_type n, Compare comp) {
    using T    = typename std::iterator_traits<It>::value_type;
    using Diff = typename std::iterator_traits<It>::difference_type;

    T value = std::move(*(first + root));
    Diff child;
    // Sift the hole down to a leaf, then sift `value` back up.  This halves the
    // number of comparisons compared with the naive formulation.
    while ((child = 2 * root + 1) < n) {
        if (child + 1 < n && comp(*(first + child), *(first + child + 1))) ++child;
        if (!comp(value, *(first + child))) break;
        *(first + root) = std::move(*(first + child));
        root = child;
    }
    *(first + root) = std::move(value);
}

template <typename It, typename Compare>
inline void heap_sort(It first, It last, Compare comp) {
    using Diff = typename std::iterator_traits<It>::difference_type;
    const Diff n = last - first;
    if (n < 2) return;
    for (Diff i = n / 2 - 1; i >= 0; --i) sift_down(first, i, n, comp);
    for (Diff i = n - 1; i > 0; --i) {
        std::swap(*first, *(first + i));
        sift_down(first, Diff(0), i, comp);
    }
}

// ---------------------------------------------------------------------------
// Median selection (pdqsort pivot choice)
// ---------------------------------------------------------------------------

template <typename It, typename Compare>
FYX_FORCE_INLINE void sort2_iter(It a, It b, Compare comp) {
    if (comp(*b, *a)) std::iter_swap(a, b);
}

template <typename It, typename Compare>
FYX_FORCE_INLINE void median3(It a, It b, It c, Compare comp) {
    sort2_iter(a, b, comp);
    sort2_iter(b, c, comp);
    sort2_iter(a, b, comp);
}

/// Places a good pivot at *first.  Uses median-of-3 for small ranges and
/// median-of-9 (ninther) for larger ones.
template <typename It, typename Compare>
inline void choose_pivot(It first, It last, Compare comp) {
    using Diff = typename std::iterator_traits<It>::difference_type;
    const Diff n = last - first;
    Diff h = n / 2;
    It   a = first + 1, b = first + h, c = last - 1;
    if (n > 128) {
        median3(a - 1, a, a + 1, comp);
        median3(b - 1, b, b + 1, comp);
        median3(c - 2, c - 1, c, comp);
        // After the three sub-medians, b-1..c-1 hold them; take the median of
        // the three medians.
        median3(a, b, c - 1, comp);
        std::iter_swap(first, b);
    } else {
        median3(a, b, c, comp);
        std::iter_swap(first, b);
    }
}

// ---------------------------------------------------------------------------
// Scalar branch-free bitonic network
// ---------------------------------------------------------------------------
//
// Sorts exactly N == 2^p unsigned keys ascending, with no data-dependent
// branches.  This is the universal n <= 64 kernel on targets without SIMD, and
// the reference implementation the SIMD networks are validated against.
//
// Standard Batcher bitonic:
//     for k = 2, 4, ... N:
//         for j = k/2, k/4, ... 1:
//             for each i: partner = i ^ j; if partner > i:
//                 ascending = ((i & k) == 0)
//                 keep min at i (max at partner) iff ascending
// ---------------------------------------------------------------------------

template <typename Key>
FYX_FORCE_INLINE void cmpx_asc(Key& a, Key& b) noexcept {
    const Key lo = a < b ? a : b;
    const Key hi = a < b ? b : a;
    a = lo;
    b = hi;
}

template <typename Key>
FYX_FORCE_INLINE void cmpx_desc(Key& a, Key& b) noexcept {
    const Key lo = a < b ? a : b;
    const Key hi = a < b ? b : a;
    a = hi;
    b = lo;
}

/// N is a compile-time power of two.
template <typename Key, unsigned N>
inline void bitonic_scalar(Key* FYX_RESTRICT a) noexcept {
    static_assert(N != 0 && (N & (N - 1)) == 0, "N must be a power of two");
    for (unsigned k = 2; k <= N; k <<= 1) {
        for (unsigned j = k >> 1; j > 0; j >>= 1) {
            for (unsigned i = 0; i < N; ++i) {
                const unsigned p = i ^ j;
                if (p > i) {
                    if ((i & k) == 0) cmpx_asc(a[i], a[p]);
                    else              cmpx_desc(a[i], a[p]);
                }
            }
        }
    }
}

/// Runtime-N dispatcher over the scalar network.  `a` must have capacity for
/// the padded power-of-two length and the padding must already hold the
/// all-ones sentinel.
template <typename Key>
inline void bitonic_scalar_n(Key* a, std::size_t padded) noexcept {
    switch (padded) {
        case 1:  return;
        case 2:  bitonic_scalar<Key, 2>(a);  return;
        case 4:  bitonic_scalar<Key, 4>(a);  return;
        case 8:  bitonic_scalar<Key, 8>(a);  return;
        case 16: bitonic_scalar<Key, 16>(a); return;
        case 32: bitonic_scalar<Key, 32>(a); return;
        case 64: bitonic_scalar<Key, 64>(a); return;
        default: FYX_ASSERT_IMPL(false);     return;
    }
}

} // namespace detail
} // namespace fyx
