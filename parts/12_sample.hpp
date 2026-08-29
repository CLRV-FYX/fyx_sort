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
//    * k = 256 buckets, k-1 = 255 splitters chosen as quantiles of a sample
//      (parallel arithmetic comparator fallback uses a 64-way top partition
//      with the same 256-way sample budget to reduce random-data comparisons).
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


// Specialized unrolled classifier for the fixed 256-way top-level used by FYX.
// It removes the loop/termination branch from the Eytzinger descent while
// preserving the exact same splitter semantics (bucket = upper_bound in the
// sorted splitter set represented by the tree).
template <class T, class Comp>
FYX_FORCE_INLINE unsigned classify_bucket_256(const T& x, const T* tree, Comp comp) {
    static_assert(kSampleBuckets == 256, "unrolled classifier assumes 256 buckets");
    unsigned b = 1;
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    return b - kSampleBuckets;
}

template <class T, class Comp>
FYX_FORCE_INLINE unsigned classify_bucket_64(const T& x, const T* tree, Comp comp) {
    unsigned b = 1;
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    b = 2 * b + (comp(x, tree[b]) ? 0u : 1u);
    return b - 64u;
}


template <class T>
inline constexpr std::size_t sample_sort_threshold_for() noexcept {
    // The lower threshold helps cheap numeric/custom-comparator leaves by
    // avoiding large pdqsort tails.  Strings are comparison-expensive and the
    // 32K recursion threshold regressed the 4H2G random-string matrix, so keep
    // the previous 128K handoff for string sample-sort fallback paths.
    if constexpr (std::is_same<T, std::string>::value) return std::size_t(1) << 17;
    else return kSampleThreshold;
}

template <class T, class Comp>
FYX_FORCE_INLINE unsigned classify_bucket_sample(const T& x, const T* tree,
                                                 unsigned m, Comp comp) {
    // Unrolling the fixed-depth classifier is a win for cheap/trivial payloads.
    // For std::string and other non-trivial comparators, preserve the compact
    // looped classifier to avoid code-size/I-cache regressions and keep the
    // old random-string behaviour.
    if constexpr (std::is_arithmetic<T>::value || std::is_trivially_copyable<T>::value) {
        (void)m;
        return classify_bucket_256(x, tree, comp);
    } else {
        return classify_bucket(x, tree, m, comp);
    }
}


template <class T>
FYX_FORCE_INLINE std::uint64_t sample_value_bits(const T& v) noexcept {
    std::uint64_t u = 0;
    std::memcpy(&u, &v, sizeof(T));
    return u;
}

FYX_FORCE_INLINE std::size_t sample_hash_bits(std::uint64_t x) noexcept {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return static_cast<std::size_t>(x);
}

inline unsigned short sample_project_rank16(std::uint64_t key, unsigned kind) noexcept {
    const std::uint64_t x = key;
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

template <class It, class Comp>
inline bool sample_arithmetic_rank16_count_sort(It first, It last, Comp comp) {
    using T = typename std::iterator_traits<It>::value_type;
    if constexpr (!std::is_arithmetic<T>::value || std::is_same<T, bool>::value ||
                  !std::is_trivially_copyable<T>::value || sizeof(T) > sizeof(std::uint64_t)) {
        (void)first; (void)last; (void)comp;
        return false;
    } else {
        constexpr std::size_t Limit = 256;
        constexpr std::size_t Cap = 512;
        constexpr std::size_t Mask = Cap - 1;
        constexpr unsigned short Sentinel = std::numeric_limits<unsigned short>::max();
        const std::size_t n = static_cast<std::size_t>(last - first);
        if (n < 2) return true;

        std::array<std::uint64_t, Cap> sample_keys{};
        std::array<unsigned char, Cap> sample_used{};
        std::vector<T> values;
        std::vector<std::uint64_t> keys;
        values.reserve(Limit);
        keys.reserve(Limit);
        const std::size_t s = std::min<std::size_t>(n, 4096);
        for (std::size_t j = 0; j < s; ++j) {
            const std::size_t idx = (j * n) / s;
            const T v = *(first + static_cast<typename std::iterator_traits<It>::difference_type>(idx));
            const std::uint64_t key = sample_value_bits(v);
            std::size_t h = sample_hash_bits(key) & Mask;
            for (;;) {
                if (!sample_used[h]) {
                    if (keys.size() >= Limit) return false;
                    sample_used[h] = 1;
                    sample_keys[h] = key;
                    keys.push_back(key);
                    values.push_back(v);
                    break;
                }
                if (sample_keys[h] == key) break;
                h = (h + 1) & Mask;
            }
        }
        if (keys.empty()) return false;

        std::vector<unsigned> order(keys.size());
        for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
            return comp(values[a], values[b]);
        });
        std::vector<T> sorted_values;
        std::vector<std::uint64_t> sorted_keys;
        sorted_values.reserve(order.size());
        sorted_keys.reserve(order.size());
        for (unsigned idx : order) {
            sorted_values.push_back(values[idx]);
            sorted_keys.push_back(keys[idx]);
        }

        std::vector<unsigned short> rank_of(65536, Sentinel);
        unsigned chosen = std::numeric_limits<unsigned>::max();
        for (unsigned kind = 0; kind < 13; ++kind) {
            std::fill(rank_of.begin(), rank_of.end(), Sentinel);
            bool ok = true;
            for (std::size_t r = 0; r < sorted_keys.size(); ++r) {
                const unsigned short q = sample_project_rank16(sorted_keys[r], kind);
                if (rank_of[q] != Sentinel) { ok = false; break; }
                rank_of[q] = static_cast<unsigned short>(r);
            }
            if (ok) { chosen = kind; break; }
        }
        if (chosen == std::numeric_limits<unsigned>::max()) return false;

        std::vector<std::size_t> counts(sorted_keys.size(), 0);
        for (std::size_t i = 0; i < n; ++i) {
            const T v = *(first + static_cast<typename std::iterator_traits<It>::difference_type>(i));
            const std::uint64_t key = sample_value_bits(v);
            const unsigned short r = rank_of[sample_project_rank16(key, chosen)];
            if (r == Sentinel || sorted_keys[r] != key) return false;
            ++counts[r];
        }

        std::size_t out = 0;
        for (std::size_t r = 0; r < sorted_values.size(); ++r) {
            const std::size_t c = counts[r];
            std::fill_n(first + static_cast<typename std::iterator_traits<It>::difference_type>(out), c, sorted_values[r]);
            out += c;
        }
        return true;
    }
}

template <class It, class Comp>
inline bool sample_arithmetic_sparse_count_sort(It first, It last, Comp comp) {
    using T = typename std::iterator_traits<It>::value_type;
    if constexpr (!std::is_arithmetic<T>::value || std::is_same<T, bool>::value ||
                  !std::is_trivially_copyable<T>::value || sizeof(T) > sizeof(std::uint64_t)) {
        (void)first; (void)last; (void)comp;
        return false;
    } else {
        constexpr std::size_t Cap = 512;
        constexpr std::size_t Mask = Cap - 1;
        const std::size_t n = static_cast<std::size_t>(last - first);
        if (n < 2) return true;

        std::array<std::uint64_t, Cap> keys{};
        std::array<T, Cap> values{};
        std::array<std::size_t, Cap> counts{};
        std::array<unsigned char, Cap> used{};
        std::vector<unsigned> distinct;
        distinct.reserve(256);

        for (std::size_t i = 0; i < n; ++i) {
            const T v = *(first + static_cast<typename std::iterator_traits<It>::difference_type>(i));
            const std::uint64_t key = sample_value_bits(v);
            std::size_t h = sample_hash_bits(key) & Mask;
            for (;;) {
                if (!used[h]) {
                    if (distinct.size() >= 256) return false;
                    used[h] = 1;
                    keys[h] = key;
                    values[h] = v;
                    counts[h] = 1;
                    distinct.push_back(static_cast<unsigned>(h));
                    break;
                }
                if (keys[h] == key) { ++counts[h]; break; }
                h = (h + 1) & Mask;
            }
        }
        if (distinct.size() <= 1) return true;

        std::sort(distinct.begin(), distinct.end(), [&](unsigned a, unsigned b) {
            return comp(values[a], values[b]);
        });

        std::size_t out = 0;
        for (unsigned slot : distinct) {
            const T v = values[slot];
            const std::size_t c = counts[slot];
            std::fill_n(first + static_cast<typename std::iterator_traits<It>::difference_type>(out), c, v);
            out += c;
        }
        return true;
    }
}


// Sample sort over a random-access range.  Falls back to pdqsort when the
// low-cardinality guard fires (cheap for caller -- this is hit before the
// O(n) permutation work).
template <class It, class Comp>
inline void sample_sort(It first, It last, Comp comp) {
    using T = typename std::iterator_traits<It>::value_type;
    const std::size_t n = static_cast<std::size_t>(last - first);

    const std::size_t sample_threshold = sample_sort_threshold_for<T>();
    if (n <= kInsertionThreshold) { insertion_sort(first, last, comp); return; }
    if (n < sample_threshold)     { pdqsort(first, last, comp);      return; }

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
    if constexpr (std::is_arithmetic<T>::value) {
        // 256-way float/double lowcard samples land around a 0.20 distinct
        // ratio with the current oversampling.  Try a collision-free rank16
        // exact counter before giving floating-point duplicates to pdqsort; it
        // still declines safely when the full input exceeds 256 raw values.
        const double count_ratio = 0.25;
        if (distinct_ratio <= count_ratio) {
            if (sample_arithmetic_rank16_count_sort(first, last, comp)) return;
            if (sample_arithmetic_sparse_count_sort(first, last, comp)) return;
            if (distinct_ratio < 0.10 || std::is_floating_point<T>::value) { pdqsort(first, last, comp); return; }
        }
    } else if (distinct_ratio < 0.05) {
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
    std::array<std::size_t, kSampleBuckets> count{};
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned b = classify_bucket_sample(*(first + i), tree.data(), m, comp);
        bid[i] = static_cast<unsigned char>(b);
        ++count[b];
    }
    std::array<std::size_t, kSampleBuckets + 1> offset{};
    std::size_t max_bucket = 0;
    for (unsigned b = 0; b < k; ++b) {
        offset[b + 1] = offset[b] + count[b];
        if (count[b] > max_bucket) max_bucket = count[b];
    }
    // Degenerate splitter sets can map the whole range back into one bucket;
    // recursing would make no progress, so hand it to pdqsort's robust
    // three-way partition / heapsort fallback.
    if (max_bucket == n) { pdqsort(first, last, comp); return; }

    // ---- 5. bucket reorder -----------------------------------------------
    // Non-string payloads use one prefix-position scatter: a single thread owns
    // every bucket cursor and avoids the old block-prefix tables.  String
    // fallback paths keep the pre-unroll block reservations, which give
    // comparison-expensive objects more localized per-bucket writes and avoid
    // the 4H2G random-string regression seen with the numeric-tuned fast path.
    {
        std::vector<T> tmp(n);
        if constexpr (!std::is_same<T, std::string>::value) {
            std::array<std::size_t, kSampleBuckets> pos{};
            for (unsigned b = 0; b < k; ++b) pos[b] = offset[b];
            for (std::size_t i = 0; i < n; ++i) {
                const unsigned b = bid[i];
                tmp[pos[b]++] = std::move(*(first + i));
            }
        } else {
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

            for (std::size_t blk = 0; blk < blocks; ++blk) {
                const std::size_t lo = blk * block_elems;
                const std::size_t hi = std::min(n, lo + block_elems);
                auto pos = base[blk];
                for (std::size_t i = lo; i < hi; ++i) {
                    const unsigned b = bid[i];
                    tmp[pos[b]++] = std::move(*(first + i));
                }
            }
        }
        for (std::size_t i = 0; i < n; ++i) *(first + i) = std::move(tmp[i]);
    }

    // ---- 6. recurse on each bucket ---------------------------------------
    for (unsigned b = 0; b < k; ++b) {
        const std::size_t lo = offset[b], hi = offset[b + 1];
        const std::size_t sz = hi - lo;
        if (sz == 0) continue;
        if (sz >= sample_threshold) sample_sort(first + lo, first + hi, comp);
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
    using T = typename std::iterator_traits<It>::value_type;
    if (sz >= sample_sort_threshold_for<T>() && depth != 0)
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

    const std::size_t sample_threshold = sample_sort_threshold_for<T>();
    if (n <= kInsertionThreshold) { insertion_sort(first, last, comp); return; }
    if (n < sample_threshold || depth == 0 || !parallel_available()) {
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
    if constexpr (std::is_arithmetic<T>::value) {
        // 256-way float/double lowcard samples land around a 0.20 distinct
        // ratio with the current oversampling.  Try a collision-free rank16
        // exact counter before giving floating-point duplicates to pdqsort; it
        // still declines safely when the full input exceeds 256 raw values.
        const double count_ratio = 0.25;
        if (distinct_ratio <= count_ratio) {
            if (sample_arithmetic_rank16_count_sort(first, last, comp)) return;
            if (sample_arithmetic_sparse_count_sort(first, last, comp)) return;
            if (distinct_ratio < 0.10 || std::is_floating_point<T>::value) { pdqsort(first, last, comp); return; }
        }
    } else if (distinct_ratio < 0.05) {
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
                const unsigned b = classify_bucket_sample(*(first + i), tree.data(), m, comp);
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
inline void parallel_sample_sort_arithmetic64_top(It first, It last, Comp comp, unsigned depth) {
    using T = typename std::iterator_traits<It>::value_type;
    constexpr unsigned k = 64;
    constexpr unsigned m = k - 1u;
    const std::size_t n = static_cast<std::size_t>(last - first);

    const std::size_t sample_threshold = sample_sort_threshold_for<T>();
    if (n <= kInsertionThreshold) { insertion_sort(first, last, comp); return; }
    if (n < sample_threshold || depth == 0 || !parallel_available()) {
        sample_sort(first, last, comp);
        return;
    }

    // Keep the 256-bucket sample budget even though the top partition uses 64
    // buckets.  This preserves the 256-way low-cardinality signal while cutting
    // random arithmetic classification from eight comparator probes to six.
    const std::size_t oversample = std::max<std::size_t>(1, (log2_floor(static_cast<std::uint64_t>(n)) + 4) / 5);
    const std::size_t S = std::min(n, std::max<std::size_t>(std::size_t(256), std::size_t(256) * oversample));
    const std::size_t stride = n / S;
    std::vector<T> sample(S);
    for (std::size_t i = 0; i < S; ++i) sample[i] = *(first + i * stride);
    std::sort(sample.begin(), sample.end(), comp);

    std::size_t distinct = 1;
    for (std::size_t i = 1; i < S; ++i)
        if (comp(sample[i - 1], sample[i])) ++distinct;
    const double distinct_ratio = static_cast<double>(distinct) / static_cast<double>(S);
    const double count_ratio = 0.25;
    if (distinct_ratio <= count_ratio) {
        if (sample_arithmetic_rank16_count_sort(first, last, comp)) return;
        if (sample_arithmetic_sparse_count_sort(first, last, comp)) return;
        if (distinct_ratio < 0.10 || std::is_floating_point<T>::value) { pdqsort(first, last, comp); return; }
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
    std::vector<std::array<std::size_t, k>> local(chunks);
    for (auto& a : local) a.fill(0);

    auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            const std::size_t lo = (c * n) / chunks;
            const std::size_t hi = ((c + 1) * n) / chunks;
            auto& lc = local[c];
            for (std::size_t i = lo; i < hi; ++i) {
                const unsigned b = classify_bucket_64(*(first + i), tree.data(), comp);
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

    std::vector<std::array<std::size_t, k>> base(chunks);
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
    using T = typename std::iterator_traits<It>::value_type;
    const std::size_t n = static_cast<std::size_t>(last - first);
    unsigned depth = static_cast<unsigned>(2 * log2_floor(static_cast<std::uint64_t>(n ? n : 1)) + 8);
#if FYX_SAMPLE_SORT_V2
    if constexpr (std::is_arithmetic<T>::value && !std::is_same<T, bool>::value) {
        parallel_sample_sort_arithmetic64_top(first, last, comp, depth);
    } else {
        parallel_sample_sort_impl(first, last, comp, depth);
    }
#else
    parallel_sample_sort_impl(first, last, comp, depth);
#endif
}

#endif // FYX_ENABLE_PARALLEL

} // namespace detail
} // namespace fyx
