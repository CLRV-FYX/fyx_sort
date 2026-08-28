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
                    if (descending ? (prev < k) : (k < prev)) return false;
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
inline FastOrderKind detect_fast_order_kind(T* p, std::size_t n, Comp comp) {
    if (n < 2) return FastOrderKind::AllEqual;
#if !FYX_ENABLE_FAST_PATHS
    (void)p; (void)comp;
    return FastOrderKind::None;
#else
    if constexpr (std::is_arithmetic<T>::value && !std::is_same<T, bool>::value &&
                  std::is_trivially_copyable<T>::value) {
        if (std::memcmp(p, p + 1, (n - 1) * sizeof(T)) == 0)
            return FastOrderKind::AllEqual;
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
inline bool try_nearly_sorted_repair(T* p, std::size_t n, Comp comp) {
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n; (void)comp;
    return false;
#else
    if (n < std::size_t(1024)) return false;
    constexpr bool radix_order = radix_supported_v<T> &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    if constexpr (radix_order && std::is_floating_point<T>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else if constexpr (!std::is_move_constructible<T>::value || !std::is_move_assignable<T>::value) {
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
            if (comp(p[i], p[i - 1])) {
                mark_dirty(i - 1);
                mark_dirty(i);
                if (dirty_count > max_dirty) return false;
            }
        }
        if (dirty_count == 0) return true;

        const T* last_clean = nullptr;
        for (std::size_t i = 0; i < n; ++i) {
            if (dirty[i]) continue;
            if (last_clean && comp(p[i], *last_clean)) return false;
            last_clean = p + i;
        }

        std::vector<T> clean;
        std::vector<T> patch;
        clean.reserve(n - dirty_count);
        patch.reserve(dirty_count);
        for (std::size_t i = 0; i < n; ++i) {
            if (dirty[i]) patch.push_back(std::move(p[i]));
            else          clean.push_back(std::move(p[i]));
        }
        std::sort(patch.begin(), patch.end(), comp);
        std::size_t ci = 0, pi = 0, out = 0;
        while (ci < clean.size() && pi < patch.size()) {
            if (comp(patch[pi], clean[ci])) p[out++] = std::move(patch[pi++]);
            else                           p[out++] = std::move(clean[ci++]);
        }
        while (ci < clean.size())  p[out++] = std::move(clean[ci++]);
        while (pi < patch.size())  p[out++] = std::move(patch[pi++]);
        return true;
    }
#endif
}


template <class T, class Comp>
inline void pdqsort_for_profile_pattern(T* p, std::size_t n, Comp comp) {
    constexpr bool radix_order = radix_supported_v<T> &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    if constexpr (radix_order && std::is_floating_point<T>::value) {
        using RT = RadixTraits<T>;
        using Key = typename RT::Key;
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
        bool even_asc_odd_desc = true;
        bool even_desc_odd_asc = true;
        for (std::size_t i = 2; i < n && (even_asc_odd_desc || even_desc_odd_asc); i += 2) {
            if (before(p[i], p[i - 2])) even_asc_odd_desc = false;
            if (before(p[i - 2], p[i])) even_desc_odd_asc = false;
        }
        for (std::size_t i = 3; i < n && (even_asc_odd_desc || even_desc_odd_asc); i += 2) {
            if (before(p[i - 2], p[i])) even_asc_odd_desc = false;
            if (before(p[i], p[i - 2])) even_desc_odd_asc = false;
        }
        if (!even_asc_odd_desc && !even_desc_odd_asc) return false;

        std::vector<T> tmp;
        tmp.reserve(n);

        if (even_asc_odd_desc) {
            std::size_t e = 0;
            bool have_even = e < n;
            std::size_t o = (n < 2) ? 0 : ((n % 2 == 0) ? n - 1 : n - 2);
            bool have_odd = n >= 2;
            while (have_even && have_odd) {
                if (before(p[o], p[e])) {
                    tmp.push_back(std::move(p[o]));
                    if (o >= 2) o -= 2; else have_odd = false;
                } else {
                    tmp.push_back(std::move(p[e]));
                    e += 2;
                    have_even = e < n;
                }
            }
            while (have_even) {
                tmp.push_back(std::move(p[e]));
                e += 2;
                have_even = e < n;
            }
            while (have_odd) {
                tmp.push_back(std::move(p[o]));
                if (o >= 2) o -= 2; else have_odd = false;
            }
        } else {
            std::size_t e = (n % 2 == 1) ? n - 1 : n - 2;
            bool have_even = n >= 1;
            std::size_t o = 1;
            bool have_odd = o < n;
            while (have_even && have_odd) {
                if (before(p[o], p[e])) {
                    tmp.push_back(std::move(p[o]));
                    o += 2;
                    have_odd = o < n;
                } else {
                    tmp.push_back(std::move(p[e]));
                    if (e >= 2) e -= 2; else have_even = false;
                }
            }
            while (have_even) {
                tmp.push_back(std::move(p[e]));
                if (e >= 2) e -= 2; else have_even = false;
            }
            while (have_odd) {
                tmp.push_back(std::move(p[o]));
                o += 2;
                have_odd = o < n;
            }
        }

        for (std::size_t i = 0; i < n; ++i) p[i] = std::move(tmp[i]);
        return true;
    }
#endif
}

template <class T>
inline bool try_integer_range_count_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!(std::is_integral<T>::value && !std::is_same<T, bool>::value && radix_supported_v<T>)) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;

        T mn = p[0], mx = p[0];
        for (std::size_t i = 1; i < n; ++i) {
            if (p[i] < mn) mn = p[i];
            if (mx < p[i]) mx = p[i];
        }

        const Key lo   = RT::encode(mn);
        const Key hi   = RT::encode(mx);
        const Key span = static_cast<Key>(hi - lo);
        if (span == std::numeric_limits<Key>::max()) return false;

        const std::size_t adaptive = std::max<std::size_t>(std::size_t(4096), n);
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
inline constexpr std::size_t kProfileSampleLimit     = 1024;
inline constexpr std::size_t kProfilePartialDivisor  = 64;
inline constexpr std::size_t kProfilePartialPdqMax   = 1u << 20;

enum class DispatchDecision : unsigned char {
    None = 0,
    Network,
    ProfileAllEqual,
    ProfileSorted,
    ProfileReverse,
    LowCardinality,
    PartialPdq,
    Radix,
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
inline void record_dispatch(DispatchDecision d) noexcept { test_dispatch_slot() = d; }
#else
inline void record_dispatch(DispatchDecision) noexcept {}
#endif


inline std::size_t configured_min_parallel_size() noexcept;

template <class T>
inline void reverse_range_adaptive(T* p, std::size_t n) {
    // `std::reverse` is already a tight bidirectional swap loop for contiguous
    // ranges.  Earlier task-splitting experiments helped some object-heavy
    // cases but regressed 4H numeric reverse inputs, so keep the fast-path
    // detector and the reversal as one predictable serial pass.
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
    for (std::size_t j = 1; j < s; ++j) {
        const std::size_t idx = (j * (n - 1)) / (s - 1);
        const ProfileAdjacentRelation r = profile_relation(data[prev_idx], data[idx], comp);
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
inline bool try_radix_key_sparse_count_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr std::size_t Cap  = 1024;
        constexpr std::size_t Mask = Cap - 1;
        constexpr std::size_t Limit = kCountingClassLimit;

        std::array<Key, Cap> keys{};
        std::array<std::size_t, Cap> counts{};
        std::array<unsigned char, Cap> used{};
        std::vector<Key> distinct;
        distinct.reserve(Limit);

        for (std::size_t i = 0; i < n; ++i) {
            const Key k = RT::encode(p[i]);
            std::size_t h = low_card_hash_key(k) & Mask;
            for (;;) {
                if (!used[h]) {
                    if (distinct.size() >= Limit) return false;
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
        if constexpr (sizeof(Key) != 8) {
            (void)p; (void)n;
            return false;
        } else {
            constexpr unsigned Shift = 64u - PrefixBits;
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
    constexpr unsigned Shift = 64u - PrefixBits;
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
    constexpr unsigned Shift = 64u - PrefixBits;
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
            constexpr unsigned FirstShift = 64u - PrefixBits;
            if (n < (std::size_t(1) << 20) || !parallel_available()) return false;
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
    static_assert(Bits > 8 && Bits <= 13, "wide radix count is tuned for 9..13 bits");
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
    static_assert(Bits > 8 && Bits <= 13, "wide radix count is tuned for 9..13 bits");
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

template <class Key, unsigned Bits>
inline void radix_scatter_key_pass_wide(const Key* FYX_RESTRICT src, std::size_t n,
                                        Key* FYX_RESTRICT dst, unsigned shift,
                                        std::size_t* FYX_RESTRICT offset,
                                        bool can_stream) {
    static_assert(Bits > 8 && Bits <= 13, "wide radix scatter is tuned for 9..13 bits");
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
    static_assert(Bits > 8 && Bits <= 13, "wide radix scatter is tuned for 9..13 bits");
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
    static_assert(Bits > 8 && Bits <= 13, "wide radix scatter is tuned for 9..13 bits");
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

template <class T, unsigned PrefixBits>
inline void radix_sort_high_prefix_decoded_ties(T* p, std::size_t n) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr unsigned Shift = 64u - PrefixBits;
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
        if constexpr (sizeof(Key) != 8) {
            (void)p; (void)n; (void)descending;
            return false;
        } else {
            constexpr unsigned Passes = PrefixBits / Bits;
            constexpr unsigned FirstShift = 64u - PrefixBits;
            constexpr std::size_t Buckets = std::size_t(1) << Bits;
            if (n < (std::size_t(1) << 20) || !parallel_available()) return false;
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

                if (pi == 0) {
                    auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
                        for (std::size_t c = c_lo; c < c_hi; ++c) {
                            const std::size_t lo = (c * n) / chunks;
                            const std::size_t hi = ((c + 1) * n) / chunks;
                            auto pos = base[c];
                            radix_scatter_encode_pass_wide<T, Key, Bits>(p + lo, hi - lo, src, shift, pos.data(), can_stream);
                        }
                    };
                    parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);
                } else if (pi + 1 == Passes) {
                    auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
                        for (std::size_t c = c_lo; c < c_hi; ++c) {
                            const std::size_t lo = (c * n) / chunks;
                            const std::size_t hi = ((c + 1) * n) / chunks;
                            auto pos = base[c];
                            radix_scatter_key_decode_pass_wide<T, Key, Bits>(src + lo, hi - lo, p, shift, pos.data(), can_stream);
                        }
                    };
                    parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);
                } else {
                    auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
                        for (std::size_t c = c_lo; c < c_hi; ++c) {
                            const std::size_t lo = (c * n) / chunks;
                            const std::size_t hi = ((c + 1) * n) / chunks;
                            auto pos = base[c];
                            radix_scatter_key_pass_wide<Key, Bits>(src + lo, hi - lo, dst, shift, pos.data(), can_stream);
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

template <class T>
inline bool try_parallel_radix_high_prefix_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        using Key = typename RadixTraits<T>::Key;
        if constexpr (sizeof(Key) != 8) {
            (void)p; (void)n; (void)descending;
            return false;
        } else if constexpr (std::is_same<T, double>::value) {
            // IEEE doubles generated by benchmark/random workloads share many
            // exponent bits.  Sorting the top 39 encoded bits with three 13-bit
            // digits removes most tie buckets while cutting the full 8-pass radix
            // to two key-buffer passes plus a final decode scatter.
            return try_parallel_radix_high_prefix_key_sort_wide<T, 39, 13>(p, n, descending);
        } else if constexpr (std::is_integral<T>::value && sizeof(T) == 8) {
            // 64-bit integer random data has very few collisions in the top
            // encoded bits at 10M scale.  Three 11-bit prefix passes avoid the
            // fourth byte pass and repair only the remaining tiny tie groups.
            return try_parallel_radix_high_prefix_key_sort_wide<T, 33, 11>(p, n, descending);
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
    const std::size_t max_chunks = std::max<std::size_t>(2, static_cast<std::size_t>(pool.nworkers()) * 2);
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
        const bool can_stream = have_nt_stores();

        parallel_radix_wide_count_scatter<11>(chunks,
            [&](std::size_t c, std::uint32_t* out) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                radix_count_value_pass_banked_wide<T, 11>(p + lo, hi - lo, 0, out);
            },
            [&](std::size_t c, std::size_t* pos) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                radix_scatter_encode_pass_wide<T, Key, 11>(p + lo, hi - lo, a, 0, pos, can_stream);
            });

        parallel_radix_wide_count_scatter<11>(chunks,
            [&](std::size_t c, std::uint32_t* out) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                radix_count_key_pass_banked_wide<Key, 11>(a + lo, hi - lo, 11, out);
            },
            [&](std::size_t c, std::size_t* pos) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                radix_scatter_key_pass_wide<Key, 11>(a + lo, hi - lo, b, 11, pos, can_stream);
            });

        parallel_radix_wide_count_scatter<10>(chunks,
            [&](std::size_t c, std::uint32_t* out) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                radix_count_key_pass_banked_wide<Key, 10>(b + lo, hi - lo, 22, out);
            },
            [&](std::size_t c, std::size_t* pos) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                radix_scatter_key_decode_pass_wide<T, Key, 10>(b + lo, hi - lo, p, 22, pos, can_stream);
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
            if (n < (std::size_t(1) << 20) || !parallel_available()) return false;
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

        auto fill_job = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t out_rank = lo; out_rank < hi; ++out_rank) {
                const std::size_t src_rank = descending ? (d - 1 - out_rank) : out_rank;
                const T v = RT::decode(distinct[src_rank]);
                std::fill(p + offset[out_rank], p + offset[out_rank + 1], v);
            }
        };
        parallel_for_index(std::size_t(0), d, std::size_t(8), fill_job);
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

        auto fill_job = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t out_rank = lo; out_rank < hi; ++out_rank) {
                const std::size_t src_rank = descending ? (d - 1 - out_rank) : out_rank;
                const T v = RT::decode(distinct[src_rank]);
                std::fill(p + offset[out_rank], p + offset[out_rank + 1], v);
            }
        };
        parallel_for_index(std::size_t(0), d, std::size_t(8), fill_job);
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
inline bool try_radix_key_sparse_count_sort_parallel(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        if (n < kParallelThreshold || !parallel_available()) return false;
        if (!radix_key_sparse_probe_ok(p, n)) return false;

        using RT = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr std::size_t Cap = 1024;
        constexpr std::size_t Mask = Cap - 1;
        constexpr std::size_t Limit = kCountingClassLimit;

        const std::size_t chunks = adaptive_parallel_chunks(n);
        std::vector<std::array<Key, Cap>> local_keys(chunks);
        std::vector<std::array<std::size_t, Cap>> local_counts(chunks);
        std::vector<std::array<unsigned char, Cap>> local_used(chunks);
        std::vector<std::vector<Key>> local_distinct(chunks);
        std::vector<unsigned char> overflow(chunks, 0);
        for (std::size_t c = 0; c < chunks; ++c) {
            local_used[c].fill(0);
            local_counts[c].fill(0);
            local_distinct[c].reserve(Limit);
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
                            if (distinct.size() >= Limit) { overflow[c] = 1; goto done_chunk; }
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
        distinct.reserve(Limit);
        auto global_add = [&](Key k, std::size_t add) -> bool {
            std::size_t h = low_card_hash_key(k) & Mask;
            for (;;) {
                if (!used[h]) {
                    if (distinct.size() >= Limit) return false;
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
inline bool try_integer_range_count_sort_parallel(T* p, std::size_t n, bool descending) {
    if constexpr (!(std::is_integral<T>::value && !std::is_same<T, bool>::value && radix_supported_v<T>)) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        if (n < kParallelThreshold || !parallel_available()) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        constexpr std::size_t MaxParallelRange = 65536;

        const std::size_t sample_n = std::min<std::size_t>(n, kProfileSampleLimit);
        Key smn = RT::encode(p[0]);
        Key smx = smn;
        for (std::size_t j = 1; j < sample_n; ++j) {
            const std::size_t idx = (j * n) / sample_n;
            const Key k = RT::encode(p[idx]);
            if (k < smn) smn = k;
            if (smx < k) smx = k;
        }
        const Key sample_span = static_cast<Key>(smx - smn);
        if (sample_span == std::numeric_limits<Key>::max()) return false;
        const unsigned long long sample_range64 = static_cast<unsigned long long>(sample_span) + 1ull;
        if (sample_range64 < 64ull ||
            sample_range64 > static_cast<unsigned long long>(MaxParallelRange)) return false;

        const std::size_t chunks = adaptive_parallel_chunks(n);
        if ((n + chunks - 1) / chunks >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) return false;

        std::vector<Key> local_min(chunks), local_max(chunks);
        auto minmax_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                Key mn = RT::encode(p[lo]);
                Key mx = mn;
                for (std::size_t i = lo + 1; i < hi; ++i) {
                    const Key k = RT::encode(p[i]);
                    if (k < mn) mn = k;
                    if (mx < k) mx = k;
                }
                local_min[c] = mn;
                local_max[c] = mx;
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), minmax_job);

        Key mn = local_min[0];
        Key mx = local_max[0];
        for (std::size_t c = 1; c < chunks; ++c) {
            if (local_min[c] < mn) mn = local_min[c];
            if (mx < local_max[c]) mx = local_max[c];
        }
        const Key span = static_cast<Key>(mx - mn);
        if (span == std::numeric_limits<Key>::max()) return false;
        const unsigned long long range64 = static_cast<unsigned long long>(span) + 1ull;
        if (range64 == 0 || range64 > static_cast<unsigned long long>(MaxParallelRange)) return false;
        const std::size_t range = static_cast<std::size_t>(range64);

        std::vector<std::uint32_t> local(chunks * range, 0);
        auto count_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                const std::size_t lo = (c * n) / chunks;
                const std::size_t hi = ((c + 1) * n) / chunks;
                std::uint32_t* lc = local.data() + c * range;
                for (std::size_t i = lo; i < hi; ++i)
                    ++lc[static_cast<std::size_t>(RT::encode(p[i]) - mn)];
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), count_job);

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

        auto fill_job = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t out_rank = lo; out_rank < hi; ++out_rank) {
                const std::size_t src_rank = descending ? (range - 1 - out_rank) : out_rank;
                const T v = RT::decode(static_cast<Key>(mn + static_cast<Key>(src_rank)));
                std::fill(p + offset[out_rank], p + offset[out_rank + 1], v);
            }
        };
        parallel_for_index(std::size_t(0), range, std::size_t(64), fill_job);
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

#if FYX_ENABLE_PARALLEL
        if (!done && prefer_parallel && high_entropy)
            done = try_parallel_radix32_wide_sort(p, n, descending) ||
                   try_parallel_radix_high_prefix_sort(p, n, descending);
        if (!done && prefer_parallel)
            done = try_parallel_radix_sort(p, n, descending, high_entropy);
#else
        (void)prefer_parallel;
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
template <class T, class Comp>
inline void sort_st(T* p, std::size_t n, Comp comp, bool descending,
                    const InputProfile<T, Comp>* known_profile = nullptr) {
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
    } else {
        if (radix_order) {
            if (try_radix_monotonic_sort(p, n, descending, true)) return;
        } else {
            if (try_monotonic_sort(p, p + n, comp, true)) return;
        }
    }

    const bool high_entropy = prof && prof->is_high_entropy;
    const bool partial_pdq = prof && prof->is_partially_sorted &&
        n <= kProfilePartialPdqMax && !(radix_order && std::is_floating_point<T>::value);
    if (partial_pdq && !(prof && prof->is_low_cardinality)) {
        pdqsort(p, p + n, comp);
        record_dispatch(DispatchDecision::PartialPdq);
        return;
    }

    if (radix_order) {
        if (!high_entropy) {
            if (try_integer_range_count_sort(p, n, descending)) { record_dispatch(DispatchDecision::LowCardinality); return; }
            if (try_radix_key_sparse_count_sort(p, n, descending)) { record_dispatch(DispatchDecision::LowCardinality); return; }
            if (try_low_cardinality_count_sort(p, p + n, comp)) { record_dispatch(DispatchDecision::LowCardinality); return; }
        }
        if (partial_pdq) { pdqsort(p, p + n, comp); record_dispatch(DispatchDecision::PartialPdq); return; }
        if constexpr (radix_type) {
            if (n >= kRadixThreshold || std::is_floating_point<T>::value) {
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
        if (partial_pdq) { pdqsort(p, p + n, comp); record_dispatch(DispatchDecision::PartialPdq); return; }
        if (try_guarded_string_order_sort(p, n, comp, false)) { record_dispatch(DispatchDecision::Radix); return; }
        if (try_string_msd_sort(p, n, comp, descending)) { record_dispatch(DispatchDecision::Radix); return; }
        if (try_trivial_prefix_key_radix_sort(p, n, comp)) { record_dispatch(DispatchDecision::Radix); return; }
    }

    if (n <= kInsertionThreshold) { insertion_sort(p, p + n, comp); record_dispatch(DispatchDecision::Pdq); return; }
    // Generic path: strings, structs and custom comparators get an ips4o-style
    // sample sort once they are large enough; smaller ranges use pdqsort.
    if (n >= kSampleThreshold && (!std::is_arithmetic<T>::value || !radix_order)) {
        sample_sort(p, p + n, comp);
        record_dispatch(DispatchDecision::Sample);
        return;
    }
    pdqsort(p, p + n, comp);
    record_dispatch(DispatchDecision::Pdq);
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
        std::merge(src + a0, src + a1, src + b0, src + b1, dst + out, comp);
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
    if (!parallel_merge_buffered(p, mid, n, comp))
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

#if FYX_ENABLE_FAST_PATHS
    if (n > detail::kNetworkMax && detail::try_parallel_all_equal_exit(p, n, comp)) return;
    if (n > detail::kNetworkMax && detail::try_fast_order_exit(p, n, comp, true)) return;
    if (n > detail::kNetworkMax && detail::try_interleaved_runs_sort(p, n, comp)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    if (n <= detail::kProfilePartialPdqMax && detail::pdq_preferred_order_sample(p, n, comp)) {
        if constexpr (!std::is_arithmetic<T>::value) {
            if (detail::try_nearly_sorted_repair(p, n, comp)) {
                detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                return;
            }
        }
        detail::pdqsort_for_profile_pattern(p, n, comp);
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
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
            n <= detail::kProfilePartialPdqMax && !(radix_ok && std::is_floating_point<T>::value);
        if (partial_pdq && !(prof && prof->is_low_cardinality)) {
            detail::pdqsort(p, p + n, comp);
            detail::record_dispatch(detail::DispatchDecision::PartialPdq);
            return;
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
                    if (detail::try_integer_range_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_dense_prefix_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_rank16_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_integer_sparse_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_sparse_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_sparse_count_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_integer_range_count_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                } else {
                    if (detail::try_integer_range_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_integer_range_count_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_dense_prefix_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_rank16_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_integer_sparse_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_sparse_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_sparse_count_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                }
                if (detail::try_low_cardinality_count_sort(p, p + n, comp)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
            }
            if (partial_pdq) { detail::pdqsort(p, p + n, comp); detail::record_dispatch(detail::DispatchDecision::PartialPdq); return; }
            if (high_entropy && detail::try_parallel_radix32_wide_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (high_entropy && detail::try_parallel_radix_high_prefix_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
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
            if (partial_pdq) { detail::pdqsort(p, p + n, comp); detail::record_dispatch(detail::DispatchDecision::PartialPdq); return; }
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
            if (high_entropy && detail::try_parallel_radix32_wide_sort(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::Radix);
                return;
            }
            if (high_entropy && detail::try_parallel_radix_high_prefix_sort(p, n, descending)) {
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
        if (n >= detail::kSampleThreshold) {
            detail::parallel_sample_sort(p, p + n, comp);
            detail::record_dispatch(detail::DispatchDecision::ParallelSample);
            return;
        }
        detail::parallel_sort_ptr(p, n, comp, descending, o);
        return;
    }
#endif
    detail::sort_st(p, n, comp, descending, prof);
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
    if (detail::try_low_cardinality_count_sort(first, last, comp)) return;
    if (n >= detail::kSampleThreshold) {
        detail::sample_sort(first, last, comp);
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
extern "C" {
inline int fyx_sort_int32 (std::int32_t*  d, std::size_t n) noexcept { try { fyx::sort(d, n); return 0; } catch (...) { return -1; } }
inline int fyx_sort_uint32(std::uint32_t* d, std::size_t n) noexcept { try { fyx::sort(d, n); return 0; } catch (...) { return -1; } }
inline int fyx_sort_int64 (std::int64_t*  d, std::size_t n) noexcept { try { fyx::sort(d, n); return 0; } catch (...) { return -1; } }
inline int fyx_sort_uint64(std::uint64_t* d, std::size_t n) noexcept { try { fyx::sort(d, n); return 0; } catch (...) { return -1; } }
inline int fyx_sort_float (float*  d, std::size_t n) noexcept { try { fyx::sort(d, n); return 0; } catch (...) { return -1; } }
inline int fyx_sort_double(double* d, std::size_t n) noexcept { try { fyx::sort(d, n); return 0; } catch (...) { return -1; } }
}
