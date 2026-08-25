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

// Forward declaration of the (optionally compiled) GPU dispatch.  Defined in
// parts/15_gpu.hpp under #ifdef FYX_ENABLE_GPU; when that switch is off the
// function does not exist and the guarded call sites below compile away.
#if FYX_ENABLE_GPU
template <class T, class Comp>
inline bool gpu_sort_dispatch(T* p, std::size_t n, Comp comp, const Options& o);
#endif

// ---------------------------------------------------------------------------
// Counting sort (武器二): for integers whose value *range* is small, an O(n +
// range) pass beats both radix and comparison sorts.  Stable.  Returns false
// (caller falls back to radix) when the range is too wide to count cheaply.
// Only valid when the comparator is the default < or > (sorting by value
// order); custom comparators must not use this.
// ---------------------------------------------------------------------------
template <class T>
inline bool counting_sort(T* p, std::size_t n) {
    using U = typename std::make_unsigned<T>::type;
    const std::size_t kMaxRange = 1u << 20;   // 1M buckets cap

    // Cheap sample to estimate the range; bail early on obviously-wide data.
    T mn = p[0], mx = p[0];
    const std::size_t stride = n > 4096 ? n / 4096 : 1;
    for (std::size_t i = 0; i < n; i += stride) {
        if (p[i] < mn) mn = p[i];
        if (p[i] > mx) mx = p[i];
    }
    // Overflow-safe span check (handles signed extremes like INT64_MIN..MAX).
    if (mn < T(0) && mx > T(0)) {
        if (static_cast<U>(mx) > std::numeric_limits<U>::max() - static_cast<U>(-mn))
            return false;
    }
    if (static_cast<std::size_t>(static_cast<U>(mx) - static_cast<U>(mn)) + 1 > kMaxRange)
        return false;

    // Exact full min/max.
    mn = p[0]; mx = p[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (p[i] < mn) mn = p[i];
        if (p[i] > mx) mx = p[i];
    }
    const U lo = static_cast<U>(mn);
    std::size_t range = static_cast<std::size_t>(static_cast<U>(mx) - lo) + 1;
    if (range > kMaxRange) return false;

    ScratchLease<std::size_t> lease(range);
    if (!lease.valid()) return false;
    std::size_t* c = lease.get();
    for (std::size_t i = 0; i < range; ++i) c[i] = 0;
    for (std::size_t i = 0; i < n; ++i) c[static_cast<U>(p[i]) - lo]++;
    std::size_t total = 0;
    for (std::size_t i = 0; i < range; ++i) { std::size_t cnt = c[i]; c[i] = total; total += cnt; }
    ScratchLease<T> out(n);
    if (!out.valid()) return false;
    T* o = out.get();
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t idx = static_cast<U>(p[i]) - lo;
        o[c[idx]++] = p[i];
    }
    for (std::size_t i = 0; i < n; ++i) p[i] = o[i];
    return true;
}

// ---------------------------------------------------------------------------
// Single-threaded best-kernel selection for a contiguous pointer range.
// `descending` is only consulted for radix-encodable types (where we may have
// sorted ascending and must reverse to honour a ">" comparator).  For the
// generic comparison path it is ignored and `comp` is used directly.
// ---------------------------------------------------------------------------
template <class T, class Comp>
inline void sort_st(T* p, std::size_t n, Comp comp, bool descending) {
    if (n <= kNetworkMax) {
        if constexpr (radix_supported_v<T>) {
            small_sort_numeric(p, n);
            if (descending) std::reverse(p, p + n);
            return;
        } else {
            insertion_sort(p, p + n, comp);
            return;
        }
    }
    if constexpr (radix_supported_v<T>) {
        if (n >= kRadixThreshold) {
            if (radix_sort(p, n)) {             // returns false only on OOM
                if (descending) std::reverse(p, p + n);
                return;
            }
            // allocation failed -> fall through to the comparison path
        }
    }
    if (n <= kInsertionThreshold) { insertion_sort(p, p + n, comp); return; }
    // Generic (non-radix) path: for non-arithmetic types (strings, structs,
    // custom comparators) an ips4o-style sample sort beats pdqsort on
    // comparison cost.  sample_sort self-guards the low-cardinality case and
    // falls back to pdqsort internally.
    if (n >= kSampleThreshold && !std::is_arithmetic_v<T>) {
        sample_sort(p, p + n, comp);
        return;
    }
    pdqsort(p, p + n, comp);
}

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
    std::inplace_merge(p, p + mid, p + n, comp);
}
#endif

// ---------------------------------------------------------------------------
// Bottom-up, allocation-assisted, STABLE merge sort.  Works for any
// random-access range (pointers, vector/deque iterators, ...).  Used by
// stable_sort whenever the radix path cannot be taken (non-numeric types,
// descending order, mixed comparators).
// ---------------------------------------------------------------------------
template <class It, class Comp>
inline void stable_merge_sort(It first, It last, Comp comp) {
    using T = iter_value_t<It>;
    const std::size_t n = static_cast<std::size_t>(last - first);
    if (n < 2) return;

    std::vector<T> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = first[i];

    bool from_a = true;
    for (std::size_t width = 1; width < n; width *= 2) {
        T* src = from_a ? a.data() : b.data();
        T* dst = from_a ? b.data() : a.data();
        for (std::size_t i = 0; i < n; i += 2 * width) {
            const std::size_t m = std::min(i + width, n);
            const std::size_t r = std::min(i + 2 * width, n);
            std::merge(src + i, src + m, src + m, src + r, dst + i, comp);
        }
        from_a = !from_a;
    }
    T* final = from_a ? a.data() : b.data();
    for (std::size_t i = 0; i < n; ++i) first[i] = final[i];
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
inline void sort_pointer_core(T* p, std::size_t n, Comp comp, const Options& o) {
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

    if (radix_ok) {
#if FYX_ENABLE_PARALLEL
        if ((o.parallel == Tri::On) ||
            (o.parallel == Tri::Auto && n >= detail::kParallelThreshold &&
             detail::parallel_available())) {
            detail::parallel_sort_ptr(p, n, comp, descending, o);
            return;
        }
#endif
        detail::sort_st(p, n, comp, descending);
        return;
    }

#if FYX_ENABLE_PARALLEL
    if ((o.parallel == Tri::On) ||
        (o.parallel == Tri::Auto && n >= detail::kParallelThreshold &&
         detail::parallel_available())) {
        detail::parallel_sort_ptr(p, n, comp, false, o);
        return;
    }
#endif
    if (n <= detail::kInsertionThreshold) { detail::insertion_sort(p, p + n, comp); return; }
    detail::pdqsort(p, p + n, comp);
}

template <class It, class Comp>
inline void sort_iter_core(It first, It last, Comp comp, const Options& o) {
    const auto n = last - first;
    if (n == 0) return;
    // A raw pointer pair is contiguous: take the radix / network fast path.
    if constexpr (std::is_pointer_v<It>) {
        sort_pointer_core(first, static_cast<std::size_t>(n), comp, o);
        return;
    }
    if (static_cast<std::size_t>(n) <= detail::kInsertionThreshold) {
        detail::insertion_sort(first, last, comp);
        return;
    }
    detail::pdqsort(first, last, comp);
}

template <class Container, class Comp>
inline void sort_container_core(Container& c, Comp comp, const Options& o) {
    auto* p = std::data(c);
    const std::size_t n = static_cast<std::size_t>(std::size(c));
    sort_pointer_core(p, n, comp, o);
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
template <class It, class Comp,
          std::enable_if_t<detail::is_random_access_v<It> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void sort(It first, It last, Comp comp) {
    sort_iter_core(first, last, comp, Options{});
}
template <class It,
          std::enable_if_t<detail::is_random_access_v<It>, int> = 0>
inline void sort(It first, It last, const Options& o) {
    sort_iter_core(first, last, fyx::less{}, o);
}
template <class It, class Comp,
          std::enable_if_t<detail::is_random_access_v<It> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void sort(It first, It last, Comp comp, const Options& o) {
    sort_iter_core(first, last, comp, o);
}

// ---- contiguous container --------------------------------------------------
template <class Container,
          std::enable_if_t<detail::has_std_data_v<Container> &&
                           !std::is_pointer_v<std::remove_reference_t<Container>> && !std::is_array_v<std::remove_reference_t<Container>>, int> = 0>
inline void sort(Container& c) {
    sort_container_core(c, fyx::less{}, Options{});
}
template <class Container, class Comp,
          std::enable_if_t<detail::has_std_data_v<Container> &&
                           !std::is_pointer_v<std::remove_reference_t<Container>> && !std::is_array_v<std::remove_reference_t<Container>> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void sort(Container& c, Comp comp) {
    sort_container_core(c, comp, Options{});
}
template <class Container,
          std::enable_if_t<detail::has_std_data_v<Container> &&
                           !std::is_pointer_v<std::remove_reference_t<Container>> && !std::is_array_v<std::remove_reference_t<Container>>, int> = 0>
inline void sort(Container& c, const Options& o) {
    sort_container_core(c, fyx::less{}, o);
}
template <class Container, class Comp,
          std::enable_if_t<detail::has_std_data_v<Container> &&
                           !std::is_pointer_v<std::remove_reference_t<Container>> && !std::is_array_v<std::remove_reference_t<Container>> &&
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
    if (n == 0) return;
    // Radix is naturally stable and matches the default "<" order exactly.
    if constexpr (detail::radix_supported_v<T>) {
        if (detail::is_ascending_v<Comp, T>) {
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
    detail::stable_merge_sort(first, last, fyx::less{});
}
template <class It, class Comp,
          std::enable_if_t<detail::is_random_access_v<It> &&
                           !detail::is_fyx_options_v<Comp> &&
                           !std::is_integral_v<std::remove_reference_t<Comp>>, int> = 0>
inline void stable_sort(It first, It last, Comp comp) {
    detail::stable_merge_sort(first, last, comp);
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
extern "C" {
inline int fyx_sort_int32 (std::int32_t*  d, std::size_t n) noexcept { try { fyx::排序(d, n); return 0; } catch (...) { return -1; } }
inline int fyx_sort_uint32(std::uint32_t* d, std::size_t n) noexcept { try { fyx::排序(d, n); return 0; } catch (...) { return -1; } }
inline int fyx_sort_int64 (std::int64_t*  d, std::size_t n) noexcept { try { fyx::排序(d, n); return 0; } catch (...) { return -1; } }
inline int fyx_sort_uint64(std::uint64_t* d, std::size_t n) noexcept { try { fyx::排序(d, n); return 0; } catch (...) { return -1; } }
inline int fyx_sort_float (float*  d, std::size_t n) noexcept { try { fyx::排序(d, n); return 0; } catch (...) { return -1; } }
inline int fyx_sort_double(double* d, std::size_t n) noexcept { try { fyx::排序(d, n); return 0; } catch (...) { return -1; } }
}
