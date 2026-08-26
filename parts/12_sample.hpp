// ===========================================================================
//  Section 12 -- Generic sample sort (ips4o-style) for the non-radix path
//
//  Radix owns numeric + default comparator.  Everything else -- strings,
//  structs, and custom comparators over any type -- lands here.  For those
//  the comparison itself is the cost, so the win is doing ~log2(k) = 8
//  comparisons per element instead of ~log2(n).  That is exactly what a
//  sample sort (introspective / ips4o style) buys over pdqsort.
//
//  Design (per DESIGN.md section 6.3):
//    * k = 256 buckets, k-1 = 255 splitters chosen as quantiles of a sample.
//    * splitters stored in an *implicit binary-search tree* (Eytzinger layout)
//      so classification is a branchless descent  b = 2*b + comp(tree[b], x).
//    * two phases: count bucket sizes, then permute through a temp buffer
//      (the library already pays O(n) for radix, so O(n) temp is consistent).
//    * recurse on large buckets; small buckets fall to pdqsort / insertion.
//    * low-cardinality guard: when the sample is dominated by few distinct
//      values pdqsort's three-way partition peels duplicates for free, so we
//      skip sample sort and use pdqsort instead (prevents the ~0.55x cliff).
// ===========================================================================

namespace fyx {
namespace detail {

// Build the Eytzinger (implicit BST) layout of `m` already-sorted splitters
// into `tree` (1-based, size 2*m+1).  Internal nodes 1..m hold splitters;
// leaves m+1..2m+1 are bucket terminals (bucket = index - m - 1).
template <class T, class Comp>
inline void build_classifier_tree(std::vector<T>& tree, const T* s, int node,
                                  int lo, int hi, Comp comp) {
    if (lo > hi) return;
    const int mid = lo + (hi - lo) / 2;
    tree[node] = s[mid];
    if (static_cast<std::size_t>(2 * node + 1) >= tree.size()) return;  // safety
    build_classifier_tree(tree, s, 2 * node,     lo, mid - 1, comp);
    build_classifier_tree(tree, s, 2 * node + 1, mid + 1, hi, comp);
}

// Branchless descent to a bucket id in [0, m].
template <class T, class Comp>
inline unsigned classify_bucket(const T& x, const T* tree, unsigned m, Comp comp) {
    unsigned b = 1;
    while (b <= m) b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    return b - m - 1u;
}

// Sample sort over a random-access range.  Falls back to pdqsort when the
// low-cardinality guard fires (cheap for caller -- this is hit before the
// O(n) permutation work).
template <class It, class Comp>
inline void sample_sort(It first, It last, Comp comp) {
    using T = typename std::iterator_traits<It>::value_type;
    const std::size_t n = static_cast<std::size_t>(last - first);

    if (n <= kInsertionThreshold) { insertion_sort(first, last, comp); return; }
    if (n <  kSampleThreshold)    { pdqsort(first, last, comp);      return; }

    const unsigned k = static_cast<unsigned>(kSampleBuckets);   // 256
    const unsigned m = k - 1u;                                  // 255 splitters

    // ---- 1. representative sample (stride across the range) ----
    // IPS4o-style oversampling: sorting a 64K string sample costs more than it
    // saves.  A few samples per bucket are enough for random/high-entropy data
    // and drastically reduce top-level overhead.
    const std::size_t oversample = std::max<std::size_t>(1, (log2_floor(static_cast<std::uint64_t>(n)) + 4) / 5);
    const std::size_t S = std::min(n, std::max<std::size_t>(k, static_cast<std::size_t>(k) * oversample));
    const std::size_t stride = n / S;
    std::vector<T> sample(S);
    for (std::size_t i = 0; i < S; ++i) sample[i] = *(first + i * stride);
    std::sort(sample.begin(), sample.end(), comp);

    // ---- 2. low-cardinality guard -----------------------------------------
    // Count distinct values in the sample.  Expensive comparators (string /
    // struct) still win sample sort down to a few-percent distinct ratio; only
    // bail out when the data is dominated by duplicates (pdqsort peels them).
    std::size_t distinct = 1;
    for (std::size_t i = 1; i < S; ++i)
        if (comp(sample[i - 1], sample[i])) ++distinct;
    const double distinct_ratio = static_cast<double>(distinct) / static_cast<double>(S);
    if (distinct_ratio == 0) { pdqsort(first, last, comp); return; }
    if (!std::is_arithmetic<T>::value && distinct_ratio < 0.05) {
        pdqsort(first, last, comp);
        return;
    }

    // ---- 3. quantile splitters + classifier tree -------------------------
    std::vector<T> splitters(m);
    for (unsigned i = 0; i < m; ++i)
        splitters[i] = sample[static_cast<std::size_t>((i + 1) * S) / k];
    std::vector<T> tree(2 * m + 1);
    build_classifier_tree(tree, splitters.data(), 1, 0, static_cast<int>(m) - 1, comp);

    // ---- 4. count bucket sizes (and remember each element's bucket) ------
    std::vector<unsigned char> bid(n);
    std::vector<std::size_t> count(k, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned b = classify_bucket(*(first + i), tree.data(), m, comp);
        bid[i] = static_cast<unsigned char>(b);
        count[b]++;
    }
    std::vector<std::size_t> offset(k + 1);
    std::size_t max_bucket = 0;
    for (unsigned b = 0; b < k; ++b) {
        offset[b + 1] = offset[b] + count[b];
        if (count[b] > max_bucket) max_bucket = count[b];
    }
    // Degenerate splitter sets can map the whole range back into one bucket;
    // recursing would make no progress, so hand it to pdqsort's robust
    // three-way partition / heapsort fallback.
    if (max_bucket == n) { pdqsort(first, last, comp); return; }

    // ---- 5. block-level bucket reorder (武器四) -------------------------
    // The previous in-place cycle follower minimized auxiliary memory for
    // trivially-copyable records, but it touched the input/output in long
    // pseudo-random cycles.  Here we use an IPS4o-style block plan: count each
    // source block's contribution to every bucket, prefix those block counts
    // inside each bucket, then scatter block by block into a temporary array.
    // Writes are confined to the block's reserved slice of each bucket, and the
    // final copy-back is a streaming pass.  Non-trivial objects already used a
    // temp buffer; this gives trivial payload records the same cache locality.
    {
        const std::size_t block_elems = std::max<std::size_t>(kSampleBlock, 1024);
        const std::size_t blocks = (n + block_elems - 1) / block_elems;
        std::vector<std::array<std::size_t, kSampleBuckets>> local(blocks);
        for (auto& a : local) a.fill(0);
        for (std::size_t i = 0; i < n; ++i)
            ++local[i / block_elems][bid[i]];

        std::vector<std::array<std::size_t, kSampleBuckets>> base(blocks);
        for (auto& a : base) a.fill(0);
        for (unsigned b = 0; b < k; ++b) {
            std::size_t run = offset[b];
            for (std::size_t blk = 0; blk < blocks; ++blk) {
                base[blk][b] = run;
                run += local[blk][b];
            }
        }

        std::vector<T> tmp(n);
        for (std::size_t blk = 0; blk < blocks; ++blk) {
            const std::size_t lo = blk * block_elems;
            const std::size_t hi = std::min(n, lo + block_elems);
            auto pos = base[blk];
            for (std::size_t i = lo; i < hi; ++i) {
                const unsigned b = bid[i];
                tmp[pos[b]++] = std::move(*(first + i));
            }
        }
        for (std::size_t i = 0; i < n; ++i) *(first + i) = std::move(tmp[i]);
    }

    // ---- 6. recurse on each bucket ---------------------------------------
    for (unsigned b = 0; b < k; ++b) {
        const std::size_t lo = offset[b], hi = offset[b + 1];
        const std::size_t sz = hi - lo;
        if (sz == 0) continue;
        if (sz >= kSampleThreshold) sample_sort(first + lo, first + hi, comp);
        else if (sz > kInsertionThreshold) pdqsort(first + lo, first + hi, comp);
        else insertion_sort(first + lo, first + hi, comp);
    }
}


#if FYX_ENABLE_PARALLEL

// Small fork-join based parallel-for over integer ranges.  `fn(lo, hi)` must be
// safe to run concurrently on disjoint ranges.
template <class Fn>
inline void parallel_for_index(std::size_t lo, std::size_t hi,
                               std::size_t grain, Fn& fn) {
    if (hi <= lo) return;
    if (hi - lo <= grain || !parallel_available()) { fn(lo, hi); return; }
    const std::size_t mid = lo + (hi - lo) / 2;
    fork_join([&] { parallel_for_index(lo, mid, grain, fn); },
              [&] { parallel_for_index(mid, hi, grain, fn); });
}

template <class It, class Comp>
inline void parallel_sample_sort_impl(It first, It last, Comp comp, unsigned depth);

template <class It, class Comp>
inline void parallel_sample_sort_one_bucket(It first, std::size_t lo, std::size_t hi,
                                            Comp comp, unsigned depth) {
    const std::size_t sz = hi - lo;
    if (sz == 0) return;
    It b = first + static_cast<typename std::iterator_traits<It>::difference_type>(lo);
    It e = first + static_cast<typename std::iterator_traits<It>::difference_type>(hi);
    if (sz >= kSampleThreshold && depth != 0)
        parallel_sample_sort_impl(b, e, comp, depth - 1);
    else if (sz > kInsertionThreshold)
        pdqsort(b, e, comp);
    else
        insertion_sort(b, e, comp);
}

template <class It, class Comp>
inline void parallel_sample_sort_bucket_range(It first,
                                              const std::vector<std::size_t>& offset,
                                              Comp comp, unsigned depth,
                                              unsigned lo, unsigned hi) {
    if (hi <= lo) return;
    if (hi - lo <= 8 || !parallel_available()) {
        for (unsigned b = lo; b < hi; ++b)
            parallel_sample_sort_one_bucket(first, offset[b], offset[b + 1], comp, depth);
        return;
    }
    const unsigned mid = lo + (hi - lo) / 2;
    fork_join([=, &offset] { parallel_sample_sort_bucket_range(first, offset, comp, depth, lo, mid); },
              [=, &offset] { parallel_sample_sort_bucket_range(first, offset, comp, depth, mid, hi); });
}

// Parallel top-level sample sort: sample and splitter construction are serial
// (small), while classification/counting, scatter/copy-back, and bucket
// recursion are split across the Chase-Lev pool.  The permutation is stable by
// chunk order, which is stronger than sort requires and harmless for payloads.
template <class It, class Comp>
inline void parallel_sample_sort_impl(It first, It last, Comp comp, unsigned depth) {
    using T = typename std::iterator_traits<It>::value_type;
    const std::size_t n = static_cast<std::size_t>(last - first);

    if (n <= kInsertionThreshold) { insertion_sort(first, last, comp); return; }
    if (n < kSampleThreshold || depth == 0 || !parallel_available()) {
        sample_sort(first, last, comp);
        return;
    }

    const unsigned k = static_cast<unsigned>(kSampleBuckets);
    const unsigned m = k - 1u;

    const std::size_t oversample = std::max<std::size_t>(1, (log2_floor(static_cast<std::uint64_t>(n)) + 4) / 5);
    const std::size_t S = std::min(n, std::max<std::size_t>(k, static_cast<std::size_t>(k) * oversample));
    const std::size_t stride = n / S;
    std::vector<T> sample(S);
    for (std::size_t i = 0; i < S; ++i) sample[i] = *(first + i * stride);
    std::sort(sample.begin(), sample.end(), comp);

    std::size_t distinct = 1;
    for (std::size_t i = 1; i < S; ++i)
        if (comp(sample[i - 1], sample[i])) ++distinct;
    const double distinct_ratio = static_cast<double>(distinct) / static_cast<double>(S);
    if (!std::is_arithmetic<T>::value && distinct_ratio < 0.05) {
        pdqsort(first, last, comp);
        return;
    }

    std::vector<T> splitters(m);
    for (unsigned i = 0; i < m; ++i)
        splitters[i] = sample[static_cast<std::size_t>((i + 1) * S) / k];
    std::vector<T> tree(2 * m + 1);
    build_classifier_tree(tree, splitters.data(), 1, 0, static_cast<int>(m) - 1, comp);

    ThreadPool& pool = global_pool();
    std::size_t chunks = (n + kParallelThreshold - 1) / kParallelThreshold;
    const std::size_t max_chunks = std::max<std::size_t>(2, static_cast<std::size_t>(pool.nworkers()) * 4);
    if (chunks > max_chunks) chunks = max_chunks;
    if (chunks < 2) { sample_sort(first, last, comp); return; }

    std::vector<unsigned char> bid(n);
    std::vector<std::array<std::size_t, kSampleBuckets>> local(chunks);
    for (auto& a : local) a.fill(0);

    auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            const std::size_t lo = (c * n) / chunks;
            const std::size_t hi = ((c + 1) * n) / chunks;
            auto& lc = local[c];
            for (std::size_t i = lo; i < hi; ++i) {
                const unsigned b = classify_bucket(*(first + i), tree.data(), m, comp);
                bid[i] = static_cast<unsigned char>(b);
                ++lc[b];
            }
        }
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);

    std::vector<std::size_t> count(k, 0), offset(k + 1, 0);
    std::size_t max_bucket = 0;
    for (unsigned b = 0; b < k; ++b) {
        for (std::size_t c = 0; c < chunks; ++c) count[b] += local[c][b];
        if (count[b] > max_bucket) max_bucket = count[b];
        offset[b + 1] = offset[b] + count[b];
    }
    if (max_bucket == n) { pdqsort(first, last, comp); return; }

    std::vector<std::array<std::size_t, kSampleBuckets>> base(chunks);
    for (auto& a : base) a.fill(0);
    for (unsigned b = 0; b < k; ++b) {
        std::size_t run = offset[b];
        for (std::size_t c = 0; c < chunks; ++c) {
            base[c][b] = run;
            run += local[c][b];
        }
    }

    {
        std::vector<T> tmp(n);
        auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                auto pos = base[c];
                for (std::size_t i = lo; i < hi; ++i) {
                    const unsigned b = bid[i];
                    tmp[pos[b]++] = std::move(*(first + i));
                }
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);

        auto copy_job = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t i = lo; i < hi; ++i) *(first + i) = std::move(tmp[i]);
        };
        parallel_for_index(std::size_t(0), n, kParallelThreshold, copy_job);
    }

    parallel_sample_sort_bucket_range(first, offset, comp, depth, 0u, k);
}

template <class It, class Comp>
inline void parallel_sample_sort(It first, It last, Comp comp) {
    const std::size_t n = static_cast<std::size_t>(last - first);
    unsigned depth = static_cast<unsigned>(2 * log2_floor(static_cast<std::uint64_t>(n ? n : 1)) + 8);
    parallel_sample_sort_impl(first, last, comp, depth);
}

#endif // FYX_ENABLE_PARALLEL

} // namespace detail
} // namespace fyx
