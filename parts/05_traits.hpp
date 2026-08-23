// ============================================================================
//  Section 5 -- Type traits
//    * radix key encoding (order-preserving map T -> unsigned integer)
//    * detection of the default comparator
//    * detection of contiguous iterators / containers
// ============================================================================

namespace fyx {

// ---------------------------------------------------------------------------
// Comparators.  fyx::less / fyx::greater are recognised by the dispatcher and
// unlock the radix path; std::less<T>, std::less<void> and std::greater are
// recognised too.
// ---------------------------------------------------------------------------

struct less {
    template <typename A, typename B>
    FYX_FORCE_INLINE constexpr bool operator()(const A& a, const B& b) const
        noexcept(noexcept(a < b)) { return a < b; }
};

struct greater {
    template <typename A, typename B>
    FYX_FORCE_INLINE constexpr bool operator()(const A& a, const B& b) const
        noexcept(noexcept(b < a)) { return b < a; }
};

namespace detail {

// --- is_ascending_comparator ------------------------------------------------
template <typename C, typename T> struct is_std_less                 : std::false_type {};
template <typename T>             struct is_std_less<fyx::less, T>   : std::true_type  {};
template <typename T>             struct is_std_less<std::less<T>, T>: std::true_type  {};
template <typename T>             struct is_std_less<std::less<void>, T> : std::true_type {};

template <typename C, typename T> struct is_std_greater                     : std::false_type {};
template <typename T>             struct is_std_greater<fyx::greater, T>    : std::true_type  {};
template <typename T>             struct is_std_greater<std::greater<T>, T> : std::true_type  {};
template <typename T>             struct is_std_greater<std::greater<void>, T> : std::true_type {};

/// True when Compare is a known "<" on T, so the radix path may replace it.
template <typename C, typename T>
inline constexpr bool is_ascending_v = is_std_less<typename std::decay<C>::type, T>::value;

/// True when Compare is a known ">" on T (radix + reverse).
template <typename C, typename T>
inline constexpr bool is_descending_v = is_std_greater<typename std::decay<C>::type, T>::value;

// --- radix key traits -------------------------------------------------------
//
// encode() maps a value to an unsigned integer of the same width such that
//     a < b   <=>   encode(a) < encode(b)
// for the total order the sort must produce.  decode() is its exact inverse.
//
//   unsigned    : identity.
//   signed      : flip the sign bit (two's-complement order -> unsigned order).
//   IEEE float  : if the sign bit is set, flip every bit; otherwise flip only
//                 the sign bit.  This yields the IEEE-754 totalOrder relation:
//                 -NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN.
//                 Note -0 sorts before +0, which std::sort does not distinguish
//                 (they compare equal), so the result is still a valid sorted
//                 sequence under operator<.
// ---------------------------------------------------------------------------

template <typename T, typename Enable = void>
struct RadixTraits {
    static constexpr bool supported = false;
};

// ---- unsigned integers -----------------------------------------------------
template <typename T>
struct RadixTraits<T, typename std::enable_if<std::is_integral<T>::value &&
                                              std::is_unsigned<T>::value &&
                                              !std::is_same<T, bool>::value>::type> {
    static constexpr bool supported = true;
    using Key = typename std::make_unsigned<T>::type;
    static constexpr unsigned bits  = sizeof(T) * CHAR_BIT;
    static constexpr unsigned passes = (bits + kRadixBits - 1) / kRadixBits;

    FYX_FORCE_INLINE static Key encode(T v) noexcept { return static_cast<Key>(v); }
    FYX_FORCE_INLINE static T   decode(Key k) noexcept { return static_cast<T>(k); }
};

// ---- signed integers -------------------------------------------------------
template <typename T>
struct RadixTraits<T, typename std::enable_if<std::is_integral<T>::value &&
                                              std::is_signed<T>::value>::type> {
    static constexpr bool supported = true;
    using Key = typename std::make_unsigned<T>::type;
    static constexpr unsigned bits   = sizeof(T) * CHAR_BIT;
    static constexpr unsigned passes = (bits + kRadixBits - 1) / kRadixBits;
    static constexpr Key      kSign  = Key(1) << (bits - 1);

    FYX_FORCE_INLINE static Key encode(T v) noexcept {
        return static_cast<Key>(static_cast<Key>(v) ^ kSign);
    }
    FYX_FORCE_INLINE static T decode(Key k) noexcept {
        return static_cast<T>(static_cast<Key>(k ^ kSign));
    }
};

// ---- IEEE-754 binary32 / binary64 -----------------------------------------
template <typename T>
struct RadixTraits<T, typename std::enable_if<std::is_floating_point<T>::value &&
                                              (sizeof(T) == 4 || sizeof(T) == 8) &&
                                              std::numeric_limits<T>::is_iec559>::type> {
    static constexpr bool supported = true;
    using Key = typename std::conditional<sizeof(T) == 4, std::uint32_t, std::uint64_t>::type;
    static constexpr unsigned bits   = sizeof(T) * CHAR_BIT;
    static constexpr unsigned passes = (bits + kRadixBits - 1) / kRadixBits;
    static constexpr Key      kSign  = Key(1) << (bits - 1);

    FYX_FORCE_INLINE static Key encode(T v) noexcept {
        Key u;
        std::memcpy(&u, &v, sizeof(Key));          // the only defined type pun
        // Arithmetic-shift the sign bit across the word: 0xFFFF.. for negative,
        // 0 for positive; then OR in the sign bit so positives still flip it.
        const Key mask = static_cast<Key>(-static_cast<Key>(u >> (bits - 1))) | kSign;
        return static_cast<Key>(u ^ mask);
    }
    FYX_FORCE_INLINE static T decode(Key k) noexcept {
        // Inverse: if the encoded top bit is set the original was positive.
        const Key mask = ((k >> (bits - 1)) != 0) ? kSign
                                                  : static_cast<Key>(~Key(0));
        const Key u = static_cast<Key>(k ^ mask);
        T v;
        std::memcpy(&v, &u, sizeof(T));
        return v;
    }
};

// bool and char-like types are handled by the integral specialisations except
// bool itself, which we exclude (a two-valued sort is a counting problem and
// the generic path handles it correctly and fast enough).

template <typename T>
inline constexpr bool radix_supported_v = RadixTraits<T>::supported;

// --- contiguous iterator detection ------------------------------------------
//
// C++17 has no contiguous_iterator_tag, so we detect the shapes that matter:
// raw pointers, and any random-access iterator whose operator-> yields a real
// pointer and whose reference is a true lvalue reference to value_type.  For
// the standard containers we care about (vector, array, string, valarray) the
// library-specific iterator types are handled by the pointer check after
// unwrapping __normal_iterator / _Vector_iterator via std::addressof on a
// dereferenced element -- but doing that requires a non-empty range, so we
// only ever call it when first != last.
// ---------------------------------------------------------------------------

template <typename It>
using iter_value_t = typename std::iterator_traits<It>::value_type;

template <typename It>
using iter_cat_t = typename std::iterator_traits<It>::iterator_category;

template <typename It>
inline constexpr bool is_random_access_v =
    std::is_base_of<std::random_access_iterator_tag, iter_cat_t<It>>::value;

template <typename It, typename = void>
struct IsContiguous : std::false_type {};

template <typename T>
struct IsContiguous<T*, void> : std::true_type {};

template <typename T>
struct IsContiguous<const T*, void> : std::true_type {};

// std::vector<T>::iterator, std::array<T,N>::iterator, std::string::iterator
// are all random-access and expose a pointer through operator->.  That is
// exactly the shape we can safely convert with std::addressof(*it).
template <typename It>
struct IsContiguous<It, typename std::enable_if<
    is_random_access_v<It> &&
    std::is_pointer<decltype(std::declval<It&>().operator->())>::value &&
    std::is_lvalue_reference<typename std::iterator_traits<It>::reference>::value
>::type> : std::true_type {};

template <typename It>
inline constexpr bool is_contiguous_v = IsContiguous<typename std::decay<It>::type>::value;

/// Converts a contiguous iterator to a raw pointer.  Only valid for a non-empty
/// range; call sites check that first.
template <typename T>
FYX_FORCE_INLINE T* to_pointer(T* p) noexcept { return p; }

template <typename It>
FYX_FORCE_INLINE auto to_pointer(It it) noexcept
    -> typename std::add_pointer<typename std::remove_reference<decltype(*it)>::type>::type {
    return std::addressof(*it);
}

// --- misc -------------------------------------------------------------------

/// Types the SIMD kernels handle natively.
template <typename T>
inline constexpr bool is_simd_sortable_v =
    (std::is_arithmetic<T>::value && !std::is_same<T, bool>::value &&
     (sizeof(T) == 4 || sizeof(T) == 8));

/// Cheap-to-move types benefit from value-based (rather than swap-based) loops.
template <typename T>
inline constexpr bool is_cheap_v =
    std::is_trivially_copyable<T>::value && sizeof(T) <= 2 * sizeof(void*) * 2;

} // namespace detail
} // namespace fyx
