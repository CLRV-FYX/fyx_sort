// ============================================================================
//  Section 12b -- Adaptive-order weapons (natural-run merge / dirty-patch merge)
//
//  Comparison sort costs O(n log n) comparisons no matter how much order the
//  input already has; radix sort costs a fixed number of passes no matter how
//  few keys actually differ.  Real inputs are usually neither random nor
//  perfectly sorted, and both extremes waste work there.  This section adds
//  two structure detectors that turn "almost sorted" into O(n) sequential
//  work:
//
//  * natural-run merge (`try_natural_run_merge`)
//      One scan finds the maximal monotone runs of the input.  Descending runs
//      are reversed in place (fyx::sort is not stable, so reversing equal keys
//      is allowed) and the runs are then merged bottom-up with a buffer no
//      larger than the smallest side of any merge.  Cost is O(n log R) moves
//      with strictly sequential access, so organ pipes, rotated sorted arrays,
//      concatenated sorted blocks and "sorted with a few blocks moved" inputs
//      cost one or two passes instead of 4-8 radix passes (or 20 partitioning
//      passes for a comparison sort).
//
//  * dirty-patch merge (`try_dirty_patch_merge`)
//      When only a small fraction of the positions take part in an inversion,
//      the clean subsequence is already sorted.  Pull the dirty positions out
//      into a tiny buffer, sort that buffer, and merge it back over the
//      compacted clean run.  Three sequential passes total, independent of the
//      key width, so 64-bit keys cost the same as 32-bit ones.
//
//  Both detectors are structure-driven, not benchmark-driven:
//    - they are read-only until the structure has been proven (the run scan and
//      the inversion scan never mutate), so a rejection costs a few hundred
//      touched elements on random data;
//    - they never change the multiset (reversal, compaction and merge are all
//      permutations), so a wrong guess cannot corrupt the caller's data;
//    - they cover whole *classes* of input (any arrangement of R monotone runs,
//      any pattern of local disorder), not one generator's output shape.
// ============================================================================

namespace fyx {
namespace detail {

#ifndef FYX_ENABLE_ADAPTIVE_WEAPONS
#  define FYX_ENABLE_ADAPTIVE_WEAPONS 1
#endif

// ---------------------------------------------------------------------------
// Natural runs
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Ordering used by every adaptive weapon.
//
// Default-order floating point is compared through its radix key, exactly like
// the radix and pdq pattern kernels: std::less<double> treats NaN as
// equivalent to everything and cannot tell -0 from +0, while the library
// documents the IEEE totalOrder relation (-NaN < ... < -0 < +0 < ... < +NaN).
// Sorting by the key produces a sequence that is still ordered under the
// caller's comparator, so this is always safe.
// ---------------------------------------------------------------------------
template <class T, class Comp>
inline auto adaptive_order(Comp comp) noexcept {
    constexpr bool radix_order = radix_supported_v<T> &&
        std::is_floating_point<T>::value &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    return [comp](const T& a, const T& b) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            const auto ka = RT::encode(a);
            const auto kb = RT::encode(b);
            if constexpr (is_descending_v<Comp, T>) return kb < ka;
            else return ka < kb;
        } else {
            (void)comp;
            return comp(a, b);
        }
    };
}

/// One maximal monotone run: elements [begin, end) are non-decreasing (once
/// descending runs have been reversed) under the sort's ordering.
struct MonotoneRun {
    std::size_t begin;
    std::size_t end;
};

/// Scans [p, p + n) and records up to `cap` maximal monotone runs.
///
/// Returns the number of runs, or `cap + 1` as soon as more than `cap` runs
/// exist (the scan then stops early -- random data is rejected after touching
/// about 2 * cap elements).  The scan is read-only.
template <class T, class Comp>
inline std::size_t scan_monotone_runs(const T* p, std::size_t n, Comp comp,
                                      std::size_t cap, MonotoneRun* out) {
    std::size_t count = 0;
    std::size_t start = 0;
    int dir = 0;                       // +1 ascending, -1 descending, 0 unknown
    for (std::size_t i = 1; i < n; ++i) {
        const bool down = comp(p[i], p[i - 1]);
        if (!down) {
            if (!comp(p[i - 1], p[i])) continue;   // equivalent: no information
        }
        const int d = down ? -1 : 1;
        if (dir == 0) { dir = d; continue; }
        if (d != dir) {
            // The extremum at i-1 keeps the closed run; the new run starts at i.
            if (count < cap) { out[count].begin = start; out[count].end = i; }
            ++count;
            if (count > cap) return cap + 1;
            start = i;
            dir = d;
        }
    }
    if (count < cap) { out[count].begin = start; out[count].end = n; }
    ++count;
    return count > cap ? cap + 1 : count;
}

/// Merges the adjacent runs [l, m) and [m, r) in place, using `buf` as scratch
/// space for the smaller of the two runs (at most min(r - m, m - l) elements).
/// `buf` may be null, in which case the merge falls back to std::inplace_merge
/// (the only correct option for types whose objects have to be constructed).
template <class T, class Comp>
inline void merge_adjacent_runs(T* p, std::size_t l, std::size_t m, std::size_t r,
                                T* buf, Comp comp) {
    const std::size_t a = m - l;
    const std::size_t b = r - m;
    if (a == 0 || b == 0) return;
    if (buf == nullptr) {
        std::inplace_merge(p + l, p + m, p + r, comp);
        return;
    }
    if (a >= b) {
        // Buffer the right run and merge backwards.  Writes always land at
        // indices >= the next unread element of the left run, so the left run
        // is never clobbered.
        for (std::size_t i = 0; i < b; ++i) buf[i] = std::move(p[m + i]);
        std::size_t ia = m, ib = b, w = r;
        while (ib != 0 && ia != l) {
            if (comp(buf[ib - 1], p[ia - 1])) {
                --ia; --w; p[w] = std::move(p[ia]);
            } else {
                --ib; --w; p[w] = std::move(buf[ib]);
            }
        }
        while (ib != 0) { --ib; --w; p[w] = std::move(buf[ib]); }
    } else {
        // Buffer the left run and merge forwards.
        for (std::size_t i = 0; i < a; ++i) buf[i] = std::move(p[l + i]);
        std::size_t ia = 0, rb = m, w = l;
        while (ia != a && rb != r) {
            if (comp(p[rb], buf[ia])) { p[w] = std::move(p[rb]); ++rb; }
            else                      { p[w] = std::move(buf[ia]); ++ia; }
            ++w;
        }
        while (ia != a) { p[w] = std::move(buf[ia]); ++ia; ++w; }
    }
}

/// Adaptive natural-merge sort.
///
/// Returns true when the range was sorted by merging its monotone runs.
/// `max_runs` bounds the run count (and therefore the scan cost on inputs that
/// have no runs at all); `max_levels` bounds the number of merge passes so the
/// O(n log R) traffic stays below what the radix/sample kernels would spend.
template <class T, class Comp>
inline bool try_natural_run_merge(T* p, std::size_t n, Comp comp,
                                  std::size_t max_runs, std::size_t max_levels) {
    if (n < 1024 || max_runs < 2) return false;
    if constexpr (!std::is_move_constructible<T>::value ||
                  !std::is_move_assignable<T>::value) {
        return false;
    } else {
        auto before = adaptive_order<T>(comp);
        std::vector<MonotoneRun> runbuf(max_runs + 1);
        const std::size_t count = scan_monotone_runs(p, n, before, max_runs, runbuf.data());
        if (count == 0 || count > max_runs) return false;

        if (count == 1) {
            // A single run: already ordered, or ordered the wrong way round.
            if (before(p[n - 1], p[0])) std::reverse(p, p + n);
            return true;
        }

        // Simulate the bottom-up schedule: count the passes and the largest
        // scratch buffer any single merge needs.  Both must stay inside budget
        // before we touch a single element.
        std::vector<std::size_t> bounds(count + 1);
        std::vector<std::size_t> next(count + 1);
        bounds[0] = 0;
        for (std::size_t i = 0; i < count; ++i) bounds[i + 1] = runbuf[i].end;
        std::size_t nruns = count;
        std::size_t levels = 0;
        std::size_t maxbuf = 1;
        {
            std::vector<std::size_t> sim = bounds;
            std::size_t m = count;
            while (m > 1) {
                std::size_t w = 1;
                std::size_t i = 0;
                next[0] = sim[0];
                for (; i + 2 <= m; i += 2) {
                    const std::size_t left  = sim[i + 1] - sim[i];
                    const std::size_t right = sim[i + 2] - sim[i + 1];
                    const std::size_t small = left < right ? left : right;
                    if (small > maxbuf) maxbuf = small;
                    next[w++] = sim[i + 2];
                }
                if (i < m) next[w++] = sim[i + 1];
                ++levels;
                if (levels > max_levels) return false;
                for (std::size_t k = 0; k < w; ++k) sim[k] = next[k];
                m = w - 1;
            }
        }

        std::unique_ptr<ScratchLease<T>> lease;
        // Descending runs become ascending; fyx::sort is unstable so reversing
        // equal keys is allowed.
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t b = runbuf[i].begin;
            const std::size_t e = runbuf[i].end;
            if (e - b > 1 && before(p[e - 1], p[b])) std::reverse(p + b, p + e);
        }

        // Raw scratch storage only holds objects that need no construction;
        // everything else uses the standard in-place merge, which manages its
        // own (properly constructed) temporary.
        T* buf = nullptr;
        if constexpr (std::is_trivially_copyable<T>::value) {
            lease.reset(new ScratchLease<T>(maxbuf));
            if (!lease->valid()) return false;  // no scratch: leave it to radix/pdq
            buf = lease->get();
        }

        std::size_t m = count;
        while (m > 1) {
            std::size_t w = 1;
            std::size_t i = 0;
            next[0] = bounds[0];
            for (; i + 2 <= m; i += 2) {
                merge_adjacent_runs(p, bounds[i], bounds[i + 1], bounds[i + 2], buf, before);
                next[w++] = bounds[i + 2];
            }
            if (i < m) next[w++] = bounds[i + 1];
            for (std::size_t k = 0; k < w; ++k) bounds[k] = next[k];
            m = w - 1;
        }
        return true;
    }
}

/// Convenience wrapper: budget the merge against the kernels it replaces.
/// Radix on 32-bit keys spends 4 passes, on 64-bit keys 8 passes (each one a
/// read + a scattered write), so 64-bit types tolerate more merge levels than
/// 32-bit ones.  Non-radix types pay comparisons, which merge saves most.
template <class T, class Comp>
inline bool try_natural_run_merge_adaptive(T* p, std::size_t n, Comp comp) {
#if !FYX_ENABLE_ADAPTIVE_WEAPONS
    (void)p; (void)n; (void)comp;
    return false;
#else
    constexpr std::size_t max_runs   = 64;
    constexpr std::size_t max_levels =
        radix_supported_v<T> ? (sizeof(T) <= 4 ? 4u : 6u) : 8u;
    return try_natural_run_merge(p, n, comp, max_runs, max_levels);
#endif
}

// ---------------------------------------------------------------------------
// Dirty-patch merge
// ---------------------------------------------------------------------------

/// Sorts a patch and merges it back over the clean run sitting at the front of
/// `p`.  Writes run backwards so the array can act as its own output: the write
/// cursor never passes the read cursor of the clean run.
template <class T, class Comp>
inline void merge_patch_back(T* p, std::size_t clean_n, std::vector<T>& patch, Comp comp) {
    const std::size_t patch_n = patch.size();
    if (patch_n == 0) return;
    pdqsort(patch.data(), patch.data() + patch_n, comp);
    if (clean_n == 0) {
        for (std::size_t i = 0; i < patch_n; ++i) p[i] = std::move(patch[i]);
        return;
    }
    std::size_t ci = clean_n, pi = patch_n, out = clean_n + patch_n;
    while (pi != 0) {
        if (ci != 0 && comp(patch[pi - 1], p[ci - 1])) {
            --ci; --out; p[out] = std::move(p[ci]);
        } else {
            --pi; --out; p[out] = std::move(patch[pi]);
        }
    }
}

/// Sorts inputs whose disorder is concentrated in a small set of positions.
///
/// A position is "dirty" when it takes part in an adjacent inversion.  If the
/// dirty set is small, the clean subsequence is almost certainly already
/// ordered; we verify that in one more scan (repairing it by dirtying the few
/// positions that break it), then compact the clean elements to the front,
/// sort the tiny dirty patch, and merge the two sorted sequences back -- the
/// merge runs backwards so it can use the array itself as the output.
///
/// Total cost: three or four sequential passes plus a sort of the patch,
/// independent of the key width.  Returns false (without having moved
/// anything) as soon as the dirty set grows past `max_dirty`.
template <class T, class Comp>
inline bool try_dirty_patch_merge(T* p, std::size_t n, Comp comp,
                                  std::size_t max_dirty) {
    if (n < 1024 || max_dirty == 0) return false;
    if constexpr (!std::is_move_constructible<T>::value ||
                  !std::is_move_assignable<T>::value ||
                  !std::is_copy_constructible<T>::value) {
        return false;
    } else {
        ScratchLease<std::uint64_t> words((n >> 6) + 2);
        if (!words.valid()) return false;
        std::uint64_t* bits = words.get();
        std::memset(bits, 0, ((n >> 6) + 2) * sizeof(std::uint64_t));
        auto before = adaptive_order<T>(comp);

        std::size_t dirty_count = 0;
        auto mark = [&](std::size_t idx) {
            const std::size_t w = idx >> 6;
            const std::uint64_t m = std::uint64_t(1) << (idx & 63);
            if ((bits[w] & m) == 0) {
                bits[w] |= m;
                ++dirty_count;
            }
        };
        auto is_dirty = [&](std::size_t idx) {
            return (bits[idx >> 6] >> (idx & 63)) & 1u;
        };

        // Pass 1: every adjacent inversion pins down two dirty positions.
        for (std::size_t i = 1; i < n; ++i) {
            if (before(p[i], p[i - 1])) {
                mark(i - 1);
                mark(i);
                if (dirty_count > max_dirty) return false;
            }
        }
        if (dirty_count == 0) return true;      // already ordered

        // Pass 2: prove the clean subsequence is ordered.  A single scan is
        // enough for genuine local disorder (every displaced element is pinned
        // by its own adjacent inversion); anything that is still out of order
        // afterwards is a block-level displacement, and the displacement merge
        // below characterises those exactly instead of growing the patch one
        // pair at a time.
        // The repair budget is deliberately tight: a shape that needs the whole
        // range re-marked (a moved block, say) is not a "local disorder" shape,
        // and the displacement merge below handles it in fewer passes than we
        // would spend discovering that here.
        const std::size_t grow_cap = std::min<std::size_t>(max_dirty, dirty_count * 4 + 64);
        bool ordered = false;
        for (int pass = 0; pass < 1 && !ordered; ++pass) {
            bool changed = false;
            std::size_t prev = n;
            for (std::size_t i = 0; i < n; ++i) {
                if (is_dirty(i)) continue;
                if (prev != n && before(p[i], p[prev])) {
                    mark(prev);
                    mark(i);
                    if (dirty_count > grow_cap) return false;
                    changed = true;
                    prev = n;
                    continue;
                }
                prev = i;
            }
            ordered = !changed;
        }
        if (!ordered) return false;

        // Compact: clean elements slide to the front in order, dirty elements
        // move into the patch buffer.
        std::vector<T> patch;
        patch.reserve(dirty_count);
        std::size_t w = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (is_dirty(i)) patch.push_back(std::move(p[i]));
            else if (w != i) p[w] = std::move(p[i]), ++w;
            else ++w;
        }
        if (w + patch.size() != n) return false;       // defensive
        merge_patch_back(p, w, patch, before);
        return true;
    }
}

/// Budget wrapper: disorder beyond an eighth of the range is no longer "local",
/// and the patch sort stops being cheaper than the kernels it replaces.
template <class T, class Comp>
inline bool try_dirty_patch_merge_adaptive(T* p, std::size_t n, Comp comp) {
#if !FYX_ENABLE_ADAPTIVE_WEAPONS
    (void)p; (void)n; (void)comp;
    return false;
#else
    return try_dirty_patch_merge(p, n, comp, n / 8);
#endif
}

// ---------------------------------------------------------------------------
// Displacement patch merge
// ---------------------------------------------------------------------------

/// Sorts inputs whose disorder is a set of *displaced* elements, wherever they
/// were moved from and however far they travelled.
///
/// Characterisation (exact, two linear scans):
///   element i may stay  <=>  it is >= every element before it  and
///                            it is <= every element after it.
/// Two elements that both satisfy the test are automatically in order (each one
/// is bounded by everything on the other side), so what remains after removing
/// the failures is already sorted -- no iteration, no guessing, and no
/// assumption about how far the failures travelled.  That covers whole classes
/// of input the adjacent-inversion test cannot see: swapped blocks, moved
/// segments, elements dragged across a long clean run, splices.
///
/// Cost: two scans plus one compaction and one merge, and a sort of the patch.
/// The only scratch memory is two bits per element, so the probe never fights
/// the radix kernels for the thread arena -- it stays affordable even when it
/// declines.
template <class T, class Comp>
inline bool try_displacement_patch_merge(T* p, std::size_t n, Comp comp,
                                         std::size_t max_dirty) {
    if (n < 1024 || max_dirty == 0) return false;
    if constexpr (!std::is_move_constructible<T>::value ||
                  !std::is_move_assignable<T>::value ||
                  !std::is_copy_constructible<T>::value) {
        return false;
    } else {
        auto before = adaptive_order<T>(comp);
        const std::size_t words = (n >> 6) + 2;
        ScratchLease<std::uint64_t> bits_lease(words * 2);
        if (!bits_lease.valid()) return false;
        std::uint64_t* suf = bits_lease.get();
        std::uint64_t* pre = suf + words;
        std::memset(suf, 0, words * 2 * sizeof(std::uint64_t));

        const std::size_t n_sentinel = n;
        std::size_t low = 0;

        // Pass 1 (backward): an element greater than the minimum of its own
        // suffix cannot stay.  One running index, one bit per element.
        {
            std::size_t smin = n_sentinel;
            for (std::size_t i = n - 1;; --i) {
                if (smin != n_sentinel && before(p[smin], p[i])) {
                    suf[i >> 6] |= std::uint64_t(1) << (i & 63);
                    if (++low > max_dirty) return false;
                }
                if (smin == n_sentinel || before(p[i], p[smin])) smin = i;
                if (i == 0) break;
            }
        }

        // Pass 2 (forward): same test against the maximum of the prefix.  The
        // union of the two bit sets is the patch; its size is known before a
        // single element moves.
        std::size_t high = 0;
        {
            std::size_t cmax = n_sentinel;
            for (std::size_t i = 0; i < n; ++i) {
                if (cmax != n_sentinel && before(p[i], p[cmax])) {
                    pre[i >> 6] |= std::uint64_t(1) << (i & 63);
                    if (++high > max_dirty) return false;
                }
                if (cmax == n_sentinel || before(p[cmax], p[i])) cmax = i;
            }
        }
        if (low + high == 0) return true;
        if (low + high > max_dirty) return false;

        // Pass 3: compact around the patch, then merge the patch back.
        std::vector<T> patch;
        patch.reserve(low + high);
        std::size_t w = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t mask = std::uint64_t(1) << (i & 63);
            if (((suf[i >> 6] | pre[i >> 6]) & mask) != 0) patch.push_back(std::move(p[i]));
            else {
                if (w != i) p[w] = std::move(p[i]);
                ++w;
            }
        }
        if (w + patch.size() != n) return false;      // defensive
        merge_patch_back(p, w, patch, before);
        return true;
    }
}

template <class T, class Comp>
inline bool try_displacement_patch_merge_adaptive(T* p, std::size_t n, Comp comp) {
#if !FYX_ENABLE_ADAPTIVE_WEAPONS
    (void)p; (void)n; (void)comp;
    return false;
#else
    return try_displacement_patch_merge(p, n, comp, n / 8);
#endif
}

} // namespace detail
} // namespace fyx
