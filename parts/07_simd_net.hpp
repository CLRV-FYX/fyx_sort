// ============================================================================
//  Section 7 -- SIMD sorting networks for n <= 64
//
//  Design
//  ------
//  Every numeric type is first mapped to an *unsigned key* via RadixTraits, so
//  a single unsigned-integer network handles int/uint/float/double and gets
//  IEEE total ordering (NaN, +-0, +-inf) for free.  The array is padded to the
//  next power of two with the all-ones sentinel (the largest possible key), so
//  padding always sinks to the top and can be discarded.
//
//  The network itself is Batcher bitonic, expressed as:
//      for k = 2,4,...,N:  for j = k/2,...,1:  compare-exchange i with i^j
//
//  With L lanes per vector and V = N/L vectors:
//    * j >= L  -> the partner lies in a *different* vector.  Since k >= 2j >=
//                 2L, the direction bit (i & k) is constant across a whole
//                 vector, so the step is a plain min/max between two vectors.
//    * j <  L  -> the partner is inside the same vector.  We permute the vector
//                 by the involution i -> i^j and blend min/max under a mask.
//                 Both the permutation index vector and the blend mask depend
//                 only on (j, k, lane count) and are built at compile time.
//
//  This gives O(log^2 N) vector ops with zero branches and zero memory traffic
//  beyond the initial load and final store.
//
//  Correctness is validated in the test suite against std::sort for every
//  n in [0,64], every supported type, and adversarial inputs (all-equal, few
//  distinct, NaN, +-0, extremes).
// ============================================================================

namespace fyx {
namespace detail {

// GCC/Clang implement the unmasked AVX-512 intrinsics in terms of the _mask_
// builtins seeded with _mm512_undefined_epi32(), which expands to the
// self-initialising `__m512i __Y = __Y;`.  From -O1 upwards that trips
// -Wuninitialized at every *inlining* site, so a user building with -Werror
// would fail through no fault of their own.  The diagnostic must therefore be
// suppressed across the whole region in which such code is inlined, not merely
// around the leaf intrinsic.  FYX_ISA_BEGIN_* below bundle that suppression
// together with the ISA selection.
#if FYX_COMPILER_GCC || FYX_COMPILER_CLANG
#  define FYX_DIAG_PUSH_SIMD                                       \
      _Pragma("GCC diagnostic push")                               \
      _Pragma("GCC diagnostic ignored \"-Wuninitialized\"")        \
      _Pragma("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
#  define FYX_DIAG_POP_SIMD _Pragma("GCC diagnostic pop")
#else
#  define FYX_DIAG_PUSH_SIMD
#  define FYX_DIAG_POP_SIMD
#endif

// ---------------------------------------------------------------------------
// ISA regions.
//
// Everything between FYX_ISA_BEGIN_xxx and FYX_ISA_END is compiled for that
// instruction set, including template instantiations.  This is what makes it
// possible to ship AVX-512 kernels in a header compiled *without*
// -march=native: the region carries its own target, and the runtime CPU check
// decides whether the entry point is ever called.
//
// MSVC has no per-function target mechanism; there the regions are inert and
// the ISA is whatever /arch selected, with the runtime check still guarding
// execution.
// ---------------------------------------------------------------------------
#if FYX_COMPILER_GCC
#  define FYX_ISA_BEGIN(isa)                                       \
      FYX_DIAG_PUSH_SIMD                                           \
      _Pragma("GCC push_options")                                  \
      _Pragma(FYX_STRINGIFY(GCC target(isa)))
#  define FYX_ISA_END                                              \
      _Pragma("GCC pop_options")                                   \
      FYX_DIAG_POP_SIMD
#elif FYX_COMPILER_CLANG
#  define FYX_ISA_BEGIN(isa)                                       \
      FYX_DIAG_PUSH_SIMD                                           \
      _Pragma("clang attribute push (__attribute__((target(isa))), apply_to = function)")
#  define FYX_ISA_END                                              \
      _Pragma("clang attribute pop")                               \
      FYX_DIAG_POP_SIMD
#else
#  define FYX_ISA_BEGIN(isa)
#  define FYX_ISA_END
#endif

// ---------------------------------------------------------------------------
// Compile-time network metadata
// ---------------------------------------------------------------------------

/// Lane index permutation for the intra-vector step: lane i exchanges with
/// lane i^j.  The permutation is an involution, so one index vector serves
/// both directions.
template <unsigned Lanes>
struct LaneXor {
    unsigned idx[Lanes];
    constexpr explicit LaneXor(unsigned j) : idx() {
        for (unsigned i = 0; i < Lanes; ++i) idx[i] = i ^ j;
    }
};

/// Blend mask for the intra-vector step.  Bit i is set when lane i must keep
/// the *minimum* of the exchanged pair.
///
/// Lane i keeps the min when it is the lower element of its pair
/// ((i & j) == 0) and the block is ascending ((i & k) == 0); when the block is
/// descending the roles invert.  `base` is the index of lane 0 of this vector
/// inside the whole network, so that (i & k) is evaluated globally.
constexpr std::uint64_t lane_min_mask(unsigned lanes, unsigned j, unsigned k, unsigned base) {
    std::uint64_t m = 0;
    for (unsigned i = 0; i < lanes; ++i) {
        const unsigned g   = base + i;
        const bool     low = (g & j) == 0;
        const bool     asc = (g & k) == 0;
        if (low == asc) m |= (std::uint64_t(1) << i);
    }
    return m;
}

/// Scalar fallback that pads into a stack buffer and runs the scalar network.
template <typename Key>
inline void bitonic_pad_scalar(Key* keys, std::size_t n) {
    if (n < 2) return;
    Key buf[64];
    const std::size_t padded = static_cast<std::size_t>(next_pow2(n));
    FYX_ASSERT_IMPL(padded <= 64);
    for (std::size_t i = 0; i < n; ++i)      buf[i] = keys[i];
    for (std::size_t i = n; i < padded; ++i) buf[i] = std::numeric_limits<Key>::max();
    bitonic_scalar_n(buf, padded);
    for (std::size_t i = 0; i < n; ++i) keys[i] = buf[i];
}

} // namespace detail
} // namespace fyx
