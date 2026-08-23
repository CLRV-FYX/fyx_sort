// ---------------------------------------------------------------------------
// The generic network, parameterised over a per-ISA "Ops" policy.
//
// Ops must provide:
//   using Vec;  static constexpr unsigned kLanes;  using Mask;
//   Vec  load(const Key*)          Vec  loadu_partial(const Key*, n, fill)
//   void store(Key*, Vec)          void store_partial(Key*, Vec, n)
//   Vec  min(Vec,Vec)              Vec  max(Vec,Vec)
//   Vec  permute_xor<J>(Vec)       // lane i <- lane i^J
//   Vec  blend(Mask keepmin, Vec mins, Vec maxs)
//   Vec  splat(Key)
// ---------------------------------------------------------------------------

/// One inter-vector compare-exchange: `lo` keeps mins, `hi` keeps maxs when
/// ascending; reversed when descending.
template <typename Ops>
FYX_FORCE_INLINE void cross_step(typename Ops::Vec& a, typename Ops::Vec& b, bool ascending) {
    const typename Ops::Vec mn = Ops::min(a, b);
    const typename Ops::Vec mx = Ops::max(a, b);
    a = ascending ? mn : mx;
    b = ascending ? mx : mn;
}

// The intra-vector steps are unrolled through a recursive template so that J
// and the blend mask are compile-time constants.
template <typename Ops, unsigned J, unsigned K, unsigned Base>
FYX_FORCE_INLINE void intra_step(typename Ops::Vec& v) {
    constexpr unsigned      L    = Ops::kLanes;
    constexpr std::uint64_t mask = lane_min_mask(L, J, K, Base);
    const typename Ops::Vec p    = Ops::template permute_xor<J>(v);
    const typename Ops::Vec mn   = Ops::min(v, p);
    const typename Ops::Vec mx   = Ops::max(v, p);
    v = Ops::blend(static_cast<typename Ops::Mask>(mask), mn, mx);
}

/// Runs the j = L/2, L/4, ..., 1 tail of a bitonic merge inside one vector.
template <typename Ops, unsigned K, unsigned Base, unsigned J>
struct IntraTail {
    FYX_FORCE_INLINE static void run(typename Ops::Vec& v) {
        intra_step<Ops, J, K, Base>(v);
        IntraTail<Ops, K, Base, (J >> 1)>::run(v);
    }
};
template <typename Ops, unsigned K, unsigned Base>
struct IntraTail<Ops, K, Base, 0> {
    FYX_FORCE_INLINE static void run(typename Ops::Vec&) {}
};

/// Full bitonic sort of V vectors (V * kLanes keys), V a power of two.
///
/// Vectors are held in a fixed-size array; every index is a compile-time
/// constant after unrolling, so the whole thing lives in registers for
/// V <= 4 (i.e. n <= 64 with 16-lane AVX-512).
template <typename Ops, unsigned V>
struct BitonicVec {
    using Vec = typename Ops::Vec;
    static constexpr unsigned L = Ops::kLanes;
    static constexpr unsigned N = V * L;

    /// Applies the whole network to `v[0..V)`.
    static FYX_FORCE_INLINE void run(Vec* v) {
        // k iterates over merge widths in *elements*.
        for_each_k(v, std::integral_constant<unsigned, 2>{});
    }

private:
    // --- k loop (compile-time recursion) -----------------------------------
    template <unsigned K>
    static FYX_FORCE_INLINE void for_each_k(Vec* v, std::integral_constant<unsigned, K>) {
        j_loop<K, K / 2>(v);
        for_each_k(v, std::integral_constant<unsigned, K * 2>{});
    }
    static FYX_FORCE_INLINE void for_each_k(Vec*, std::integral_constant<unsigned, N * 2>) {}

    // --- j loop ------------------------------------------------------------
    template <unsigned K, unsigned J>
    static FYX_FORCE_INLINE void j_loop(Vec* v) {
        if constexpr (J >= L) {
            // Partner is another vector: J/L vectors away.
            constexpr unsigned stride = J / L;
            for (unsigned i = 0; i < V; ++i) {
                const unsigned partner = i ^ stride;
                if (partner > i) {
                    // Direction is constant over the vector because K >= 2J >= 2L.
                    const bool asc = ((i * L) & K) == 0;
                    cross_step<Ops>(v[i], v[partner], asc);
                }
            }
            j_loop<K, (J >> 1)>(v);
        } else if constexpr (J > 0) {
            // Remaining j < L are all intra-vector; unroll them per vector.
            intra_all<K, J>(v, std::integral_constant<unsigned, 0>{});
        }
    }

    // --- per-vector intra tail ---------------------------------------------
    template <unsigned K, unsigned J, unsigned I>
    static FYX_FORCE_INLINE void intra_all(Vec* v, std::integral_constant<unsigned, I>) {
        IntraTail<Ops, K, I * L, J>::run(v[I]);
        intra_all<K, J>(v, std::integral_constant<unsigned, I + 1>{});
    }
    template <unsigned K, unsigned J>
    static FYX_FORCE_INLINE void intra_all(Vec*, std::integral_constant<unsigned, V>) {}
};

// ---------------------------------------------------------------------------
// Sorting a padded key array with a given Ops policy.
// ---------------------------------------------------------------------------

/// Sorts `n` keys (n <= V*kLanes) held in `keys`, padding with the sentinel.
template <typename Ops, unsigned V>
FYX_FORCE_INLINE void network_sort_v(typename Ops::Key* keys, std::size_t n) {
    using Key = typename Ops::Key;
    using Vec = typename Ops::Vec;
    constexpr unsigned L = Ops::kLanes;
    constexpr unsigned N = V * L;
    static_assert(N <= 64, "network is only used up to 64 elements");

    const Key sentinel = std::numeric_limits<Key>::max();
    Vec v[V];
    // Load full vectors, then a partial one, then sentinel-fill the rest.
    for (unsigned i = 0; i < V; ++i) {
        const std::size_t off = std::size_t(i) * L;
        if (off + L <= n) {
            v[i] = Ops::load(keys + off);
        } else if (off < n) {
            v[i] = Ops::load_partial(keys + off, unsigned(n - off), sentinel);
        } else {
            v[i] = Ops::splat(sentinel);
        }
    }
    BitonicVec<Ops, V>::run(v);
    for (unsigned i = 0; i < V; ++i) {
        const std::size_t off = std::size_t(i) * L;
        if (off + L <= n)      Ops::store(keys + off, v[i]);
        else if (off < n)      Ops::store_partial(keys + off, v[i], unsigned(n - off));
    }
}

/// Dispatches on the padded size, instantiating only the vector counts that a
/// 64-element ceiling can require.
template <typename Ops>
inline void network_sort_keys(typename Ops::Key* keys, std::size_t n) {
    constexpr unsigned L = Ops::kLanes;
    if (n < 2) return;
    const std::size_t padded = static_cast<std::size_t>(next_pow2(n));
    const std::size_t vecs   = (padded + L - 1) / L;
    switch (vecs) {
        case 1: network_sort_v<Ops, 1>(keys, n); return;
        case 2: network_sort_v<Ops, 2>(keys, n); return;
        case 4: network_sort_v<Ops, 4>(keys, n); return;
        case 8: if constexpr (L * 8 <= 64) { network_sort_v<Ops, 8>(keys, n); return; } break;
        case 16: if constexpr (L * 16 <= 64) { network_sort_v<Ops, 16>(keys, n); return; } break;
        case 32: if constexpr (L * 32 <= 64) { network_sort_v<Ops, 32>(keys, n); return; } break;
        default: break;
    }
    // vecs == 3, 5, 6, 7 ... cannot occur (padded and L are powers of two), but
    // keep a correct path rather than an assertion in release builds.
    ::fyx::detail::bitonic_pad_scalar<typename Ops::Key>(keys, n);
}
