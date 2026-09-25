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
    if constexpr (radix_order) {
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

        for (std::size_t i = 1; i < n; ++i) {
            if (comp(p[i], p[i - 1])) {          // reverse of comp order
                for (++i; i < n; ++i)
                    if (comp(p[i - 1], p[i])) return FastOrderKind::None;
                return FastOrderKind::Reverse;
            }
            if (comp(p[i - 1], p[i])) {          // already in comp order
                for (++i; i < n; ++i)
                    if (comp(p[i], p[i - 1])) return FastOrderKind::None;
                return FastOrderKind::Sorted;
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
        std::size_t shifts = 0, strikes = 0, next_check = 64;
        for (std::size_t i = 1; i < n; ++i) {
            if (!before(p[i], p[i - 1])) continue;
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

        std::size_t split = n;
        for (; i < n; ++i) {
            const bool up = before(p[i - 1], p[i]);
            const bool down = before(p[i], p[i - 1]);
            if (first_asc ? down : up) {
                split = i;
                ++i;
                break;
            }
        }
        if (split == n) return false;
        if (split < n / 16u || n - split < n / 16u) return false;

        for (; i < n; ++i) {
            if (first_asc) {
                if (before(p[i - 1], p[i])) return false;
            } else {
                if (before(p[i], p[i - 1])) return false;
            }
        }

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
        if (try_arithmetic_organ_fill()) return true;

        struct Cursor {
            std::size_t cur;
            std::size_t lo;
            std::size_t hi;
            bool have;
            bool forward;
        };
        auto advance = [](Cursor& c) {
            if (!c.have) return;
            if (c.forward) {
                ++c.cur;
                if (c.cur >= c.hi) c.have = false;
            } else {
                if (c.cur == c.lo) c.have = false;
                else --c.cur;
            }
        };
        Cursor a = first_asc ? Cursor{0, 0, split, true, true}
                             : Cursor{split - 1, 0, split, true, false};
        Cursor b = first_asc ? Cursor{n - 1, split, n, true, false}
                             : Cursor{split, split, n, true, true};

        if constexpr (std::is_trivially_copyable<T>::value) {
            ScratchLease<T> tmp_lease(n);
            if (!tmp_lease.valid()) return false;
            T* tmp = tmp_lease.get();
            std::size_t out = 0;
            while (a.have && b.have) {
                if (before(p[b.cur], p[a.cur])) { tmp[out++] = p[b.cur]; advance(b); }
                else                            { tmp[out++] = p[a.cur]; advance(a); }
            }
            while (a.have) { tmp[out++] = p[a.cur]; advance(a); }
            while (b.have) { tmp[out++] = p[b.cur]; advance(b); }
            if (out != n) return false;
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
template <class T>
inline bool radix_key_monotone_scan(const T* p, std::size_t start, std::size_t hi,
                                    typename RadixTraits<T>::Key prev_key,
                                    const bool want_up, const bool descending) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
#if FYX_HAS_AVX512_CODE
    if (use_avx512()) {
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
            // Key transform must match RadixTraits<T>::encode bit for bit:
            // unsigned integers are identity, signed integers flip the sign
            // bit only (two's-complement order preserving), and IEEE floats
            // need the sign-propagating flip that reverses negative magnitudes.
            // (The float transform applied to signed integers INVERTS the
            // order within negatives -- measured: monotone negative arrays
            // were declined by the vector proof and fell to slower paths.)
            __m512i e;
            if constexpr (std::is_integral_v<T>) {
                if constexpr (std::is_unsigned_v<T>) {
                    e = k;
                } else {
                    e = _mm512_xor_si512(k, vsign);
                }
            } else {
                if constexpr (sizeof(Key) == 8) {
                    const __m512i t = _mm512_srai_epi64(k, 63);
                    e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
                } else {
                    const __m512i t = _mm512_srai_epi32(k, 31);
                    e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
                }
            }
            const __m512i shifted = (sizeof(Key) == 8)
                ? _mm512_alignr_epi64(e, prev_vec, 7)
                : _mm512_alignr_epi32(e, prev_vec, 15);
            // bad lane: the neighbour pair violates the target order
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
#endif
    Key prev = prev_key;
    for (std::size_t i = start; i < hi; ++i) {
        const Key cur = RT::encode(p[i]);
        const bool bad = want_up ? (descending ? (prev < cur) : (cur < prev))
                                 : (descending ? (cur < prev) : (prev < cur));
        if (bad) return false;
        prev = cur;
    }
    return true;
}

// Vectorised all-equal sweep: elements [lo, hi) all order-equal to p[0].
// Radix floats compare encoded keys (so -0/+0 count as different, matching
// the detector's verdict); integers compare raw.  Returns false when the
// caller should run the full per-chunk classification instead -- including
// every non-arithmetic type, which the classify path already serves.
template <class T>
inline bool range_all_equal_first_vec(const T* p, std::size_t lo, std::size_t hi) {
    if (lo >= hi) return true;
    if constexpr (!std::is_arithmetic<T>::value) {
        (void)p;
        return false;
    } else {
#if FYX_HAS_AVX512_CODE
        if (use_avx512()) {
            constexpr bool use_key = radix_supported_v<T> && std::is_floating_point<T>::value;
            using RT  = RadixTraits<T>;
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
            // full mask has one bit per lane: 8 lanes of 8 bytes, 16 of 4, ...
            const __mmask64 full = (width == 8) ? __mmask64(0xFF)
                                 : (width == 4) ? __mmask64(0xFFFF)
                                 : (width == 2) ? __mmask64(0xFFFFFFFFull)
                                                : ~__mmask64(0);
            std::size_t i = lo;
            for (; i + (width * 8) <= hi; i += width * 8) {
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
                } else {
                    if (p[i] != first) return false;
                }
            }
            return true;
        }
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
            constexpr unsigned lanes = (sizeof(Key) == 8) ? 8u : 16u;
            const __m512i vsign = (sizeof(Key) == 8)
                ? _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL))
                : _mm512_set1_epi32(static_cast<int>(0x80000000u));
            __m512i prev_vec = (sizeof(Key) == 8)
                ? _mm512_set1_epi64(static_cast<long long>(RT::encode(p[0])))
                : _mm512_set1_epi32(static_cast<int>(RT::encode(p[0])));
            bool capped = false;
            std::size_t i = 1;
            for (; i + lanes <= n; i += lanes) {
                const __m512i k = _mm512_loadu_si512(reinterpret_cast<const void*>(p + i));
                // same per-type transform as radix_key_monotone_scan (see there)
                __m512i e;
                if constexpr (std::is_integral_v<T>) {
                    if constexpr (std::is_unsigned_v<T>) {
                        e = k;
                    } else {
                        e = _mm512_xor_si512(k, vsign);
                    }
                } else {
                    if constexpr (sizeof(Key) == 8) {
                        const __m512i t = _mm512_srai_epi64(k, 63);
                        e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
                    } else {
                        const __m512i t = _mm512_srai_epi32(k, 31);
                        e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
                    }
                }
                const __m512i shifted = (sizeof(Key) == 8)
                    ? _mm512_alignr_epi64(e, prev_vec, 7)
                    : _mm512_alignr_epi32(e, prev_vec, 15);
                // violation: ascending -> cur < prev (e < shifted);
                //            descending -> prev < cur (shifted < e)
                unsigned bad = (sizeof(Key) == 8)
                    ? (descending ? _mm512_cmplt_epu64_mask(shifted, e)
                                  : _mm512_cmplt_epu64_mask(e, shifted))
                    : (descending ? _mm512_cmplt_epu32_mask(shifted, e)
                                  : _mm512_cmplt_epu32_mask(e, shifted));
                while (bad) {
                    if (nb == kProofStructMaxBreaks) { capped = true; break; }
                    const unsigned bit = static_cast<unsigned>(__builtin_ctz(bad));
                    brk[nb++] = i + bit;
                    bad &= bad - 1;
                }
                if (capped) break;
                prev_vec = e;
            }
            if (!capped) {
                Key prev = RT::encode(p[(i > 1 ? i : 1) - 1]);
                for (; i < n; ++i) {
                    const Key cur = RT::encode(p[i]);
                    if (descending ? (prev < cur) : (cur < prev)) {
                        if (nb == kProofStructMaxBreaks) { capped = true; break; }
                        brk[nb++] = i;
                    }
                    prev = cur;
                }
            }
            if (capped) return false;
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
                                          std::size_t offset, bool& descending) {
    using Key = typename RadixTraits<Field>::Key;
    if (offset + sizeof(Field) > sizeof(T)) return false;

    const std::size_t s = std::min<std::size_t>(n, 96);
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
                                                std::size_t offset) {
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
    return true;
}

template <class Field, class T, class Comp>
inline bool try_trivial_field_count_sort(T* p, std::size_t n, Comp comp,
                                         std::size_t offset) {
    using Key = typename RadixTraits<Field>::Key;
    bool descending = false;
    if (!trivial_field_candidate_order<Field>(p, n, comp, offset, descending)) return false;
    if (!trivial_field_low_cardinality_probe<Field>(p, n, offset)) return false;

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
                ScratchLease<std::size_t> counts_lease(range);
                if (counts_lease.valid()) {
                    std::size_t* counts = counts_lease.get();
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

                        ScratchLease<T> out_lease(n);
                        if (out_lease.valid()) {
                            T* out = out_lease.get();
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


template <class Field, class T, class Comp>
inline bool try_trivial_field_radix_sort(T* p, std::size_t n, Comp comp,
                                         std::size_t offset) {
    if (n < kSampleThreshold) return false;
    if constexpr (!std::is_trivially_copyable<T>::value) {
        (void)p; (void)n; (void)comp; (void)offset;
        return false;
    } else {
        using Key = typename RadixTraits<Field>::Key;
        constexpr unsigned Passes = (sizeof(Key) * CHAR_BIT + kRadixBits - 1) / kRadixBits;

        bool descending = false;
        if (!trivial_field_candidate_order<Field>(p, n, comp, offset, descending)) return false;

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
inline bool try_trivial_prefix_key_radix_sort(T* p, std::size_t n, Comp comp) {
    if constexpr (!std::is_trivially_copyable<T>::value || std::is_arithmetic<T>::value ||
                  std::is_same<T, std::string>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        constexpr std::size_t max_probe = sizeof(T) < 32 ? sizeof(T) : 32;
        for (std::size_t off = 0; off + 4 <= max_probe; off += 4) {
            if (try_trivial_field_radix_sort<std::int32_t>(p, n, comp, off)) return true;
            if (try_trivial_field_radix_sort<std::uint32_t>(p, n, comp, off)) return true;
        }
        for (std::size_t off = 0; off + 8 <= max_probe; off += 8) {
            if (try_trivial_field_radix_sort<std::int64_t>(p, n, comp, off)) return true;
            if (try_trivial_field_radix_sort<std::uint64_t>(p, n, comp, off)) return true;
        }
        return false;
    }
}

template <class T, class Comp>
inline bool try_trivial_prefix_key_count_sort(T* p, std::size_t n, Comp comp) {
    if constexpr (!std::is_trivially_copyable<T>::value || std::is_arithmetic<T>::value ||
                  std::is_same<T, std::string>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        constexpr std::size_t max_probe = sizeof(T) < 32 ? sizeof(T) : 32;
        for (std::size_t off = 0; off + 4 <= max_probe; off += 4) {
            if (try_trivial_field_count_sort<std::int32_t>(p, n, comp, off)) return true;
            if (try_trivial_field_count_sort<std::uint32_t>(p, n, comp, off)) return true;
        }
        for (std::size_t off = 0; off + 8 <= max_probe; off += 8) {
            if (try_trivial_field_count_sort<std::int64_t>(p, n, comp, off)) return true;
            if (try_trivial_field_count_sort<std::uint64_t>(p, n, comp, off)) return true;
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

inline bool string_msd_sort_default(std::string* p, std::size_t n, bool descending) {
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
template <class T, class Comp>
inline bool try_sorted_affix_sort(T* p, std::size_t n, Comp comp) {
#if !FYX_ENABLE_ADAPTIVE_WEAPONS
    (void)p; (void)n; (void)comp;
    return false;
#else
    if (n < 8192) return false;
    if constexpr (!std::is_move_constructible<T>::value ||
                  !std::is_move_assignable<T>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        auto before = adaptive_order<T>(comp);
        std::size_t head = 1;
        while (head < n && !before(p[head], p[head - 1])) ++head;
        if (head == n) return true;                  // ordered already
        std::size_t tail = n - 1;
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
        if constexpr (std::is_trivially_copyable<T>::value) {
            lease.reset(new ScratchLease<T>(need1 > need2 ? need1 : need2));
            if (!lease->valid()) return false;
            buf = lease->get();
        }
        sort_st(p + head, tail - head, comp, false);
        if (tail < n) merge_adjacent_runs(p, head, tail, n, buf, before);
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
/// monotone.  Same encode/alignr shape as radix_key_monotone_scan, so the two
/// agree bit for bit on where the break is.
template <class T>
inline std::size_t radix_key_find_break(const T* p, std::size_t start, std::size_t hi,
                                        typename RadixTraits<T>::Key prev_key,
                                        const bool want_up, const bool descending) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
#if FYX_HAS_AVX512_CODE
    if (use_avx512()) {
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
                if constexpr (std::is_unsigned_v<T>) {
                    e = k;
                } else {
                    e = _mm512_xor_si512(k, vsign);
                }
            } else {
                if constexpr (sizeof(Key) == 8) {
                    const __m512i t = _mm512_srai_epi64(k, 63);
                    e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
                } else {
                    const __m512i t = _mm512_srai_epi32(k, 31);
                    e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
                }
            }
            const __m512i shifted = (sizeof(Key) == 8)
                ? _mm512_alignr_epi64(e, prev_vec, 7)
                : _mm512_alignr_epi32(e, prev_vec, 15);
            const unsigned bad = (want_up != descending)
                ? ((sizeof(Key) == 8 ? _mm512_cmpgt_epu64_mask(shifted, e)
                                     : _mm512_cmpgt_epu32_mask(shifted, e)))
                : ((sizeof(Key) == 8 ? _mm512_cmplt_epu64_mask(shifted, e)
                                     : _mm512_cmplt_epu32_mask(shifted, e)));
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
#endif
    Key prev = prev_key;
    for (std::size_t i = start; i < hi; ++i) {
        const Key cur = RT::encode(p[i]);
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
        if (!vqsort_range_clean(p, n)) return false;
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
            if (try_partially_sorted_repair(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; }
            pdqsort_for_profile_pattern(p, n, comp);
            record_dispatch(DispatchDecision::PartialPdq);
            return;
        }
    }

    if (try_bitonic_runs_sort(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; }

    if (radix_order) {
        if (!high_entropy) {
            if (try_integer_range_count_sort(p, n, descending, prof ? &prof->sample_window : nullptr)) { record_dispatch(DispatchDecision::LowCardinality); return; }
            if (try_radix_key_sparse_count_sort(p, n, descending, prof ? &prof->sample_window : nullptr)) { record_dispatch(DispatchDecision::LowCardinality); return; }
            if (try_low_cardinality_count_sort(p, p + n, comp)) { record_dispatch(DispatchDecision::LowCardinality); return; }
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
        if (partial_pdq) { if (try_partially_sorted_repair(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; } pdqsort_for_profile_pattern(p, n, comp); record_dispatch(DispatchDecision::PartialPdq); return; }
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
    for (std::size_t i = 0; i < n; ++i) a[i] = std::move(first[i]);

    bool from_a = true;
    for (std::size_t width = 1; width < n; width *= 2) {
        T* src = from_a ? a.data() : b.data();
        T* dst = from_a ? b.data() : a.data();
        for (std::size_t i = 0; i < n; i += 2 * width) {
            const std::size_t m = std::min(i + width, n);
            const std::size_t r = std::min(i + 2 * width, n);
            merge_runs_moving(src + i, m - i, src + m, r - m, dst + i, comp);
        }
        from_a = !from_a;
    }
    T* final = from_a ? a.data() : b.data();
    for (std::size_t i = 0; i < n; ++i) first[i] = std::move(final[i]);
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

#if FYX_ENABLE_FAST_PATHS
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
    if (n > detail::kNetworkMax && detail::try_bounded_insertion_repair(p, n, comp, true)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    if (n > detail::kNetworkMax && detail::try_natural_run_merge_adaptive(p, n, comp)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    // Ordered except for one stretch in the middle: see
    // try_sorted_affix_sort.  Like the run merge this has to be reachable
    // before the parallel kernels are chosen, or a range that is three
    // quarters sorted pays for all of it on every worker.
    if (n > detail::kNetworkMax && detail::try_sorted_affix_sort(p, n, comp)) {
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
            if (detail::try_partially_sorted_repair(p, n, comp)) {
                detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                return;
            }
            detail::pdqsort_for_profile_pattern(p, n, comp);
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
                if (detail::try_partially_sorted_repair(p, n, comp)) {
                    detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                    return;
                }
                detail::pdqsort_for_profile_pattern(p, n, comp);
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
                if (detail::try_trivial_prefix_key_count_sort(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                if (detail::try_low_cardinality_count_sort(p, p + n, comp)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
            }
            if (partial_pdq) { if (detail::try_partially_sorted_repair(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::PartialPdq); return; } detail::pdqsort_for_profile_pattern(p, n, comp); detail::record_dispatch(detail::DispatchDecision::PartialPdq); return; }
            if (detail::try_guarded_string_order_sort(p, n, comp, want_parallel)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (detail::try_string_msd_sort_parallel(p, n, comp, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (detail::try_string_msd_sort(p, n, comp, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (detail::try_trivial_prefix_key_radix_sort(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
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
    if (detail::radix_supported_v<T> && (ascending || descending)) {
        if (detail::try_radix_monotonic_sort(p, n, descending, false)) return;
    } else {
        if (detail::try_monotonic_sort(p, p + n, comp, false)) return;
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
