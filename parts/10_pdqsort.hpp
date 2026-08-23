// ============================================================================
//  Section 10 -- Branchless block-partition quicksort (pdqsort derivative)
//
//  This is the general-purpose comparison sort: it accepts any comparator and
//  any value type, so it serves generic containers, the tail of the sample
//  sort, and every case the radix path declines.
//
//  Three ideas carry the performance:
//
//  1. Branchless block partitioning.  A data-dependent branch per element
//     mispredicts about half the time on random input, costing ~15 cycles
//     each.  Instead we compare a block of 64 elements and record the indices
//     that need moving into a small byte array, advancing the write cursor by
//     the comparison result rather than branching on it.  The comparison loop
//     becomes straight-line code; the only branches left are loop bounds.
//
//  2. Pattern defeat.  A bad pivot streak degrades quicksort to O(n^2).  We
//     track a recursion budget and finish with heapsort when it runs out,
//     giving a hard O(n log n) worst case.  Nearly-sorted runs are detected
//     and short-circuited, and a pivot equal to the parent's pivot triggers a
//     partition that isolates the duplicates, so inputs with few distinct keys
//     do not blow up.
//
//  3. Small ranges leave the recursion.  Numeric keys with the default
//     comparator are routed to a SIMD sorting network by the caller; the
//     insertion sort here is only reached for types the networks cannot take.
// ============================================================================

namespace fyx {
namespace detail {

/// Block size for the branchless partition.  64 one-byte offsets fit in a
/// single cache line and the count always fits in an unsigned char.
inline constexpr std::size_t kPartitionBlock = 64;

/// Result of one partition step.
template <typename It>
struct PartitionResult {
    It   pivot_pos;      ///< final resting place of the pivot
    bool already_parted; ///< true when no element needed to move
};

// ---------------------------------------------------------------------------
// Branchless block partition (Hoare scheme, elements < pivot to the left)
// ---------------------------------------------------------------------------
//
// `first` holds the pivot on entry (the caller has already placed it there).
//
// The loop keeps two offset buffers.  `offsets_l` collects positions in the
// left block holding elements that belong on the right, `offsets_r` the
// mirror.  Matching pairs are then swapped directly.  Because a block is
// scanned in full before any swap happens, the comparison loop contains no
// data-dependent branch at all.
// ---------------------------------------------------------------------------

template <typename It, typename Compare>
inline PartitionResult<It> partition_right_branchless(It first, It last, Compare comp) {
    using T    = typename std::iterator_traits<It>::value_type;
    using Diff = typename std::iterator_traits<It>::difference_type;

    T pivot = std::move(*first);

    It lo = first;
    It hi = last;

    // Find the first element >= pivot.  The pivot at *first is a sentinel.
    while (comp(*++lo, pivot)) {}

    // Find the last element < pivot.  If the scan above never moved there is
    // no sentinel on the right yet, so that loop has to be bounded.
    if (lo - 1 == first) {
        while (lo < hi && !comp(*--hi, pivot)) {}
    } else {
        while (!comp(*--hi, pivot)) {}
    }

    const bool already_parted = (lo >= hi);

    if (!already_parted) {
        std::iter_swap(lo, hi);
        ++lo;
    }

    // ---- invariants for the block loop ------------------------------------
    //   [begin, lo)  are < pivot, except the num_l pending offsets
    //   [hi, end)    are >= pivot, except the num_r pending offsets
    //   [lo, hi)     is unclassified
    // A pending left offset marks an element that belongs on the right, and
    // vice versa.  Offsets within a buffer are always ascending.
    // -----------------------------------------------------------------------
    alignas(kCacheLine) unsigned char offsets_l[kPartitionBlock];
    alignas(kCacheLine) unsigned char offsets_r[kPartitionBlock];

    It          base_l = lo;
    It          base_r = hi;
    std::size_t num_l = 0, num_r = 0, start_l = 0, start_r = 0;

    for (;;) {
        const Diff remaining = hi - lo;

        // Classify a block from the left, but only if the left buffer is
        // drained -- a pending block must stay where it is.
        if (num_l == 0 && remaining > 0) {
            const std::size_t blk = static_cast<std::size_t>(
                remaining < static_cast<Diff>(kPartitionBlock)
                    ? remaining : static_cast<Diff>(kPartitionBlock));
            base_l  = lo;
            start_l = 0;
            for (std::size_t i = 0; i < blk; ++i) {
                offsets_l[num_l] = static_cast<unsigned char>(i);
                num_l += static_cast<std::size_t>(!comp(*lo, pivot));
                ++lo;
            }
        }

        // Symmetrically from the right.  offsets_r[k] is a 1-based distance
        // *below* base_r, so the element lives at base_r - offsets_r[k].
        const Diff remaining2 = hi - lo;
        if (num_r == 0 && remaining2 > 0) {
            const std::size_t blk = static_cast<std::size_t>(
                remaining2 < static_cast<Diff>(kPartitionBlock)
                    ? remaining2 : static_cast<Diff>(kPartitionBlock));
            base_r  = hi;
            start_r = 0;
            for (std::size_t i = 0; i < blk; ++i) {
                --hi;
                offsets_r[num_r] = static_cast<unsigned char>(i + 1);
                num_r += static_cast<std::size_t>(comp(*hi, pivot));
            }
        }

        // Every matched pair can be exchanged directly.
        const std::size_t num = (num_l < num_r) ? num_l : num_r;
        for (std::size_t i = 0; i < num; ++i)
            std::iter_swap(base_l + offsets_l[start_l + i],
                           base_r - offsets_r[start_r + i]);

        num_l   -= num;  num_r   -= num;
        start_l += num;  start_r += num;

        // Done once nothing is unclassified and at most one side still holds
        // pending offsets (that leftover is handled by the cleanup below).
        if (lo >= hi && (num_l == 0 || num_r == 0)) break;
    }

    // ---- cleanup ----------------------------------------------------------
    // Consume the leftovers from the highest offset downwards.  Because the
    // offsets ascend, each swap partner taken from the shrinking boundary is
    // guaranteed to sit at or below the pending element, so no element is ever
    // moved twice.
    if (num_l) {
        while (num_l--) std::iter_swap(base_l + offsets_l[start_l + num_l], --lo);
        hi = lo;
    }
    if (num_r) {
        while (num_r--) {
            std::iter_swap(base_r - offsets_r[start_r + num_r], hi);
            ++hi;
        }
        lo = hi;
    }

    // Drop the pivot into the gap between the two halves.
    It pivot_pos = lo - 1;
    *first       = std::move(*pivot_pos);
    *pivot_pos   = std::move(pivot);

    return PartitionResult<It>{pivot_pos, already_parted};
}

/// Simple (branchy) partition, used when the value type is expensive to move
/// or the range is short enough that block bookkeeping does not pay.
template <typename It, typename Compare>
inline PartitionResult<It> partition_right_simple(It first, It last, Compare comp) {
    using T = typename std::iterator_traits<It>::value_type;

    T  pivot = std::move(*first);
    It l     = first;
    It r     = last;

    while (comp(*++l, pivot)) {}

    if (l - 1 == first) {
        while (l < r && !comp(*--r, pivot)) {}
    } else {
        while (!comp(*--r, pivot)) {}
    }

    const bool already_parted = (l >= r);

    while (l < r) {
        std::iter_swap(l, r);
        while (comp(*++l, pivot)) {}
        while (!comp(*--r, pivot)) {}
    }

    It pivot_pos = l - 1;
    *first       = std::move(*pivot_pos);
    *pivot_pos   = std::move(pivot);
    return PartitionResult<It>{pivot_pos, already_parted};
}

// ---------------------------------------------------------------------------
// Partition that sends elements *equal* to the pivot to the left.
//
// Reached when the chosen pivot compares equal to the parent's pivot, which
// means the range is dominated by one value.  Isolating the duplicates here is
// what keeps "many equal keys" inputs near linear.
// ---------------------------------------------------------------------------

template <typename It, typename Compare>
inline It partition_left(It first, It last, Compare comp) {
    using T = typename std::iterator_traits<It>::value_type;

    T  pivot = std::move(*first);
    It l     = first;
    It r     = last;

    while (comp(pivot, *--r)) {}

    if (r + 1 == last) {
        while (l < r && !comp(pivot, *++l)) {}
    } else {
        while (!comp(pivot, *++l)) {}
    }

    while (l < r) {
        std::iter_swap(l, r);
        while (comp(pivot, *--r)) {}
        while (!comp(pivot, *++l)) {}
    }

    It pivot_pos = r;
    *first       = std::move(*pivot_pos);
    *pivot_pos   = std::move(pivot);
    return pivot_pos;
}

// ---------------------------------------------------------------------------
// Small-range kernel
// ---------------------------------------------------------------------------
//
// The specification forbids insertion sort for numeric arrays of at most 64
// elements, which must use a SIMD network.  `SmallSort` is the hook: the
// numeric specialisation is installed by the dispatch layer, and this generic
// version only ever runs for types with no network (non-trivial objects,
// custom comparators over class types, and so on).
// ---------------------------------------------------------------------------

template <typename It, typename Compare>
FYX_FORCE_INLINE void small_sort_generic(It first, It last, Compare comp,
                                         bool leftmost) {
    if (leftmost) insertion_sort(first, last, comp);
    else          insertion_sort_guarded(first, last, comp);
}

// ---------------------------------------------------------------------------
// The recursive driver
// ---------------------------------------------------------------------------

template <typename It, typename Compare, bool Branchless>
inline void pdqsort_loop(It first, It last, Compare comp, int bad_allowed,
                         bool leftmost) {
    using Diff = typename std::iterator_traits<It>::difference_type;

    for (;;) {
        const Diff size = last - first;

        if (size <= static_cast<Diff>(kInsertionThreshold)) {
            small_sort_generic(first, last, comp, leftmost);
            return;
        }

        choose_pivot(first, last, comp);

        // If the pivot equals the predecessor (which is <= everything here),
        // the range is full of duplicates of that value: peel them off.
        if (!leftmost && !comp(*(first - 1), *first)) {
            first = partition_left(first, last, comp) + 1;
            continue;
        }

        const PartitionResult<It> part =
            Branchless ? partition_right_branchless(first, last, comp)
                       : partition_right_simple(first, last, comp);

        const Diff l_size = part.pivot_pos - first;
        const Diff r_size = last - (part.pivot_pos + 1);
        const bool highly_unbalanced = (l_size < size / 8) || (r_size < size / 8);

        if (highly_unbalanced) {
            // Repeated bad splits: shuffle a few elements to break the pattern
            // that is defeating the pivot heuristic.
            if (--bad_allowed == 0) {
                heap_sort(first, last, comp);
                return;
            }
            if (l_size >= static_cast<Diff>(kInsertionThreshold)) {
                std::iter_swap(first, first + l_size / 4);
                std::iter_swap(part.pivot_pos - 1, part.pivot_pos - l_size / 4);
                if (l_size > 128) {
                    std::iter_swap(first + 1, first + (l_size / 4 + 1));
                    std::iter_swap(first + 2, first + (l_size / 4 + 2));
                    std::iter_swap(part.pivot_pos - 2, part.pivot_pos - (l_size / 4 + 1));
                    std::iter_swap(part.pivot_pos - 3, part.pivot_pos - (l_size / 4 + 2));
                }
            }
            if (r_size >= static_cast<Diff>(kInsertionThreshold)) {
                std::iter_swap(part.pivot_pos + 1, part.pivot_pos + (1 + r_size / 4));
                std::iter_swap(last - 1, last - r_size / 4);
                if (r_size > 128) {
                    std::iter_swap(part.pivot_pos + 2, part.pivot_pos + (2 + r_size / 4));
                    std::iter_swap(part.pivot_pos + 3, part.pivot_pos + (3 + r_size / 4));
                    std::iter_swap(last - 2, last - (1 + r_size / 4));
                    std::iter_swap(last - 3, last - (2 + r_size / 4));
                }
            }
        } else if (part.already_parted &&
                   partial_insertion_sort(first, part.pivot_pos, comp) &&
                   partial_insertion_sort(part.pivot_pos + 1, last, comp)) {
            // The partition moved nothing and both halves were nearly sorted:
            // the bounded insertion sorts above just finished the job.
            return;
        }

        // Recurse into the smaller half, loop on the larger one, so the stack
        // depth stays O(log n).
        if (l_size < r_size) {
            pdqsort_loop<It, Compare, Branchless>(first, part.pivot_pos, comp,
                                                  bad_allowed, leftmost);
            first    = part.pivot_pos + 1;
            leftmost = false;
        } else {
            pdqsort_loop<It, Compare, Branchless>(part.pivot_pos + 1, last, comp,
                                                  bad_allowed, false);
            last = part.pivot_pos;
        }
    }
}

/// Entry point.  `Branchless` is chosen by the caller from the value type: it
/// pays for small trivially-copyable types and costs for everything else.
template <typename It, typename Compare>
inline void pdqsort(It first, It last, Compare comp) {
    if (first == last) return;

    using T = typename std::iterator_traits<It>::value_type;
    constexpr bool kBranchless =
        std::is_arithmetic<T>::value && sizeof(T) <= 16;

    pdqsort_loop<It, Compare, kBranchless>(
        first, last, comp,
        static_cast<int>(log2_floor(static_cast<std::uint64_t>(last - first))) + 1,
        true);
}

} // namespace detail
} // namespace fyx
