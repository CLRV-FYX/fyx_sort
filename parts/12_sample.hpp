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
    const std::size_t S = std::min(n, static_cast<std::size_t>(1u << 16));
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
    for (unsigned b = 0; b < k; ++b) offset[b + 1] = offset[b] + count[b];

    // ---- 5. in-place permutation (cycle following) ----------------------
    // Each bucket b must occupy [offset[b], offset[b+1]); bid[i] records the
    // bucket of element i, so its destination is the next free slot of that
    // bucket.  We rotate elements along their permutation cycles.  This is
    // O(n) extra bookkeeping (bid[] + a visited bitmap) and moves each element
    // exactly once -- no O(n) temp buffer, no extra memory passes.
    std::vector<std::size_t> cur(offset.begin(), offset.begin() + k);
    std::vector<unsigned char> visited((n + 7) / 8, 0);
    auto vis_set = [&](std::size_t i) { visited[i >> 3] |= static_cast<unsigned char>(1u << (i & 7)); };
    auto vis_get = [&](std::size_t i) { return (visited[i >> 3] >> (i & 7)) & 1u; };
    for (std::size_t i = 0; i < n; ++i) {
        if (vis_get(i)) continue;
        std::size_t j = i;
        T val = std::move(*(first + j));
        for (;;) {
            vis_set(j);
            const unsigned b = bid[j];
            const std::size_t dst = cur[b]++;
            if (dst == i) {                 // we have come back to the cycle start
                *(first + i) = std::move(val);
                break;
            }
            T tmp = std::move(*(first + dst));
            *(first + dst) = std::move(val);
            val = std::move(tmp);
            j = dst;
            if (vis_get(j)) {               // cycle closed on an already-placed slot
                *(first + j) = std::move(val);
                break;
            }
        }
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

} // namespace detail
} // namespace fyx
