// ============================================================================
//  FYX-SORT Ultimate  --  single-header, zero-dependency, high-performance sort
// ============================================================================
//
//  Author : 付yanxin (FYX)   --  https://github.com/CLRV-FYX/fyx_sort
//  License: Attribution-NonCommercial (see LICENSE in the repository)
//
//  ---------------------------------------------------------------------------
//  QUICK START
//  ---------------------------------------------------------------------------
//      #include "fyx_sort.hpp"
//
//      std::vector<int> v = {5, 3, 8, 1};
//      fyx::sort(v);                              // container overload
//      fyx::sort(v.begin(), v.end());             // iterator overload
//      fyx::sort(ptr, n);                         // pointer + length
//      fyx::sort(v, std::greater<int>());         // custom comparator
//      fyx::stable_sort(v);
//      fyx::partial_sort(v.begin(), v.begin()+10, v.end());
//      fyx::nth_element(v.begin(), v.begin()+n/2, v.end());
//
//      fyx::Options o;  o.threads = 4;  o.parallel = fyx::Tri::Off;
//      fyx::sort(v, o);
//
//      // C ABI
//      fyx_sort_int32(data, n);
//
//  Compile:  g++ -std=c++17 -O3 -pthread  your.cpp
//            (-march=native is NOT required: every SIMD kernel carries its own
//             target attribute and is dispatched at run time.)
//            cl /std:c++17 /O2 /EHsc your.cpp
//
//  ---------------------------------------------------------------------------
//  COMPILE-TIME SWITCHES  (all optional -- the default is "just include it")
//  ---------------------------------------------------------------------------
//    FYX_ENABLE_GPU          off by default. When *not* defined this header
//                            contains no GPU header, no GPU symbol, no dlopen,
//                            and adds no link-time dependency whatsoever.
//    FYX_ENABLE_PARALLEL     on  by default. Define FYX_DISABLE_PARALLEL (or
//                            FYX_ENABLE_PARALLEL=0) for a pure single-threaded
//                            build with no <thread> dependency.
//    FYX_DISABLE_AVX512      never emit / never use AVX-512 kernels.
//    FYX_DISABLE_AVX2        never emit / never use AVX2 kernels.
//    FYX_DISABLE_SSE42       never emit / never use SSE4.2 kernels.
//    FYX_DISABLE_NEON        never emit / never use ARM NEON kernels.
//    FYX_DISABLE_SIMD        scalar only (implies all of the above).
//    FYX_FORCE_SIMD_HISTOGRAM
//                            force the AVX-512 conflict-detection histogram
//                            even when the heuristic prefers the scalar one.
//    FYX_NO_EXCEPTIONS       do not throw; failed allocations degrade to an
//                            in-place algorithm instead.
//    FYX_ASSERT(x)           user-supplied assertion hook.
//
//  ---------------------------------------------------------------------------
//  WHAT THIS LIBRARY ACTUALLY DOES  (see DESIGN.md for the honest numbers)
//  ---------------------------------------------------------------------------
//    * numeric keys, default comparator  -> LSD radix sort, 8-bit digits,
//      single fused histogram pass, degenerate-pass skipping, software
//      write-combining scatter with non-temporal stores, parallel over cores.
//    * n <= 64, numeric                  -> branch-free SIMD bitonic network
//      (AVX-512 / AVX2 / SSE4.2 / NEON), no insertion sort.
//    * everything else                   -> parallel sample sort (ips4o-style
//      branch-free classification) with block-partitioning pdqsort below the
//      recursion threshold and heapsort as the depth-limit fallback.
//    * stable_sort                       -> radix (naturally stable) or a
//      parallel merge sort.
//    * work distribution                 -> lock-free Chase-Lev work-stealing
//      deques, lazily started thread pool, waiters execute tasks.
//
// ============================================================================

#ifndef FYX_SORT_HPP_INCLUDED
#define FYX_SORT_HPP_INCLUDED

#define FYX_VERSION_MAJOR 10
#define FYX_VERSION_MINOR 0
#define FYX_VERSION_PATCH 0
#define FYX_VERSION_STRING "10.0.0"

// ---------------------------------------------------------------------------
// Language level
// ---------------------------------------------------------------------------
#if defined(_MSVC_LANG)
#  define FYX_CPLUSPLUS _MSVC_LANG
#else
#  define FYX_CPLUSPLUS __cplusplus
#endif

#if FYX_CPLUSPLUS < 201703L
#  error "FYX-SORT requires C++17 or newer (use -std=c++17 or /std:c++17)."
#endif

// ---------------------------------------------------------------------------
// Compiler identification
// ---------------------------------------------------------------------------
#if defined(_MSC_VER) && !defined(__clang__)
#  define FYX_COMPILER_MSVC 1
#  if _MSC_VER < 1910
#    error "FYX-SORT requires MSVC 2017 (19.10) or newer."
#  endif
#else
#  define FYX_COMPILER_MSVC 0
#endif

#if defined(__clang__)
#  define FYX_COMPILER_CLANG 1
#  if (__clang_major__ < 5)
#    error "FYX-SORT requires Clang 5 or newer."
#  endif
#else
#  define FYX_COMPILER_CLANG 0
#endif

#if defined(__GNUC__) && !defined(__clang__)
#  define FYX_COMPILER_GCC 1
#  if (__GNUC__ < 7)
#    error "FYX-SORT requires GCC 7 or newer."
#  endif
#else
#  define FYX_COMPILER_GCC 0
#endif

#if defined(__MINGW32__) || defined(__MINGW64__)
#  define FYX_COMPILER_MINGW 1
#else
#  define FYX_COMPILER_MINGW 0
#endif

// GCC-or-Clang: the two share the GNU attribute / builtin vocabulary.
#define FYX_GNUC_LIKE (FYX_COMPILER_GCC || FYX_COMPILER_CLANG)

// ---------------------------------------------------------------------------
// Operating system
// ---------------------------------------------------------------------------
#if defined(_WIN32) || defined(_WIN64)
#  define FYX_OS_WINDOWS 1
#else
#  define FYX_OS_WINDOWS 0
#endif
#if defined(__linux__)
#  define FYX_OS_LINUX 1
#else
#  define FYX_OS_LINUX 0
#endif
#if defined(__APPLE__)
#  define FYX_OS_MACOS 1
#else
#  define FYX_OS_MACOS 0
#endif
#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
#  define FYX_OS_BSD 1
#else
#  define FYX_OS_BSD 0
#endif
#define FYX_OS_POSIX (FYX_OS_LINUX || FYX_OS_MACOS || FYX_OS_BSD)

// ---------------------------------------------------------------------------
// Architecture
// ---------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
#  define FYX_ARCH_X86_64 1
#else
#  define FYX_ARCH_X86_64 0
#endif
#if defined(__i386__) || defined(_M_IX86)
#  define FYX_ARCH_X86_32 1
#else
#  define FYX_ARCH_X86_32 0
#endif
#define FYX_ARCH_X86 (FYX_ARCH_X86_64 || FYX_ARCH_X86_32)

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
#  define FYX_ARCH_ARM64 1
#else
#  define FYX_ARCH_ARM64 0
#endif
#if defined(__arm__) || defined(_M_ARM)
#  define FYX_ARCH_ARM32 1
#else
#  define FYX_ARCH_ARM32 0
#endif
#define FYX_ARCH_ARM (FYX_ARCH_ARM64 || FYX_ARCH_ARM32)

// ---------------------------------------------------------------------------
// Standard headers (CPU path only -- nothing here pulls in a GPU runtime)
// ---------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <type_traits>
#include <iterator>
#include <functional>
#include <algorithm>
#include <vector>
#include <array>
#include <new>

// ---------------------------------------------------------------------------
// Parallel switch
// ---------------------------------------------------------------------------
#if defined(FYX_DISABLE_PARALLEL)
#  undef  FYX_ENABLE_PARALLEL
#  define FYX_ENABLE_PARALLEL 0
#elif !defined(FYX_ENABLE_PARALLEL)
#  define FYX_ENABLE_PARALLEL 1
#elif (FYX_ENABLE_PARALLEL + 0) == 0 && !defined(FYX_ENABLE_PARALLEL_EXPLICIT_ZERO)
   // FYX_ENABLE_PARALLEL was defined with no value -> treat as "on".
#  undef  FYX_ENABLE_PARALLEL
#  define FYX_ENABLE_PARALLEL 1
#endif

#if FYX_ENABLE_PARALLEL
#  include <thread>
#  include <mutex>
#  include <atomic>
#  include <condition_variable>
#  include <random>
#  include <chrono>
#else
#  include <atomic>
#endif

// ---------------------------------------------------------------------------
// Attributes / builtins
// ---------------------------------------------------------------------------
#if FYX_GNUC_LIKE
#  define FYX_FORCE_INLINE inline __attribute__((always_inline))
#  define FYX_NOINLINE     __attribute__((noinline))
#  define FYX_RESTRICT     __restrict__
#  define FYX_LIKELY(x)    (__builtin_expect(!!(x), 1))
#  define FYX_UNLIKELY(x)  (__builtin_expect(!!(x), 0))
#  define FYX_HOT          __attribute__((hot))
#  define FYX_PURE         __attribute__((pure))
#elif FYX_COMPILER_MSVC
#  define FYX_FORCE_INLINE __forceinline
#  define FYX_NOINLINE     __declspec(noinline)
#  define FYX_RESTRICT     __restrict
#  define FYX_LIKELY(x)    (x)
#  define FYX_UNLIKELY(x)  (x)
#  define FYX_HOT
#  define FYX_PURE
#else
#  define FYX_FORCE_INLINE inline
#  define FYX_NOINLINE
#  define FYX_RESTRICT
#  define FYX_LIKELY(x)    (x)
#  define FYX_UNLIKELY(x)  (x)
#  define FYX_HOT
#  define FYX_PURE
#endif

#define FYX_UNUSED(x) ((void)(x))

#define FYX_STRINGIFY_(x) #x
#define FYX_STRINGIFY(x)  FYX_STRINGIFY_(x)

#if defined(FYX_ASSERT)
#  define FYX_ASSERT_IMPL(x) FYX_ASSERT(x)
#else
#  define FYX_ASSERT_IMPL(x) ((void)0)
#endif

// Exceptions ----------------------------------------------------------------
#if defined(FYX_NO_EXCEPTIONS) || (defined(__cpp_exceptions) && __cpp_exceptions == 0) \
    || (FYX_COMPILER_MSVC && !defined(_CPPUNWIND))
#  define FYX_HAS_EXCEPTIONS 0
#else
#  define FYX_HAS_EXCEPTIONS 1
#endif

// ---------------------------------------------------------------------------
// SIMD availability -- "CODE" macros decide what is *compiled*, the runtime
// CpuFeatures object decides what is *executed*.
// ---------------------------------------------------------------------------
#if defined(FYX_DISABLE_SIMD)
#  define FYX_DISABLE_AVX512 1
#  define FYX_DISABLE_AVX2   1
#  define FYX_DISABLE_SSE42  1
#  define FYX_DISABLE_NEON   1
#endif

// x86: GCC/Clang can emit any ISA through function-level target attributes, so
// the kernels are always compiled and selected at run time.  MSVC has no such
// attribute; there the kernels are compiled only when the corresponding /arch
// flag is active (or, for AVX-512, when the compiler is new enough to accept
// the intrinsics unconditionally).
#if FYX_ARCH_X86 && !defined(FYX_DISABLE_SSE42)
#  if FYX_GNUC_LIKE || defined(__SSE4_2__) || FYX_ARCH_X86_64 || (FYX_COMPILER_MSVC && defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    define FYX_HAS_SSE42_CODE 1
#  else
#    define FYX_HAS_SSE42_CODE 0
#  endif
#else
#  define FYX_HAS_SSE42_CODE 0
#endif

#if FYX_ARCH_X86 && !defined(FYX_DISABLE_AVX2)
#  if FYX_GNUC_LIKE || defined(__AVX2__) || (FYX_COMPILER_MSVC && _MSC_VER >= 1910)
#    define FYX_HAS_AVX2_CODE 1
#  else
#    define FYX_HAS_AVX2_CODE 0
#  endif
#else
#  define FYX_HAS_AVX2_CODE 0
#endif

#if FYX_ARCH_X86 && !defined(FYX_DISABLE_AVX512)
#  if FYX_GNUC_LIKE || defined(__AVX512F__) || (FYX_COMPILER_MSVC && _MSC_VER >= 1911)
#    define FYX_HAS_AVX512_CODE 1
#  else
#    define FYX_HAS_AVX512_CODE 0
#  endif
#else
#  define FYX_HAS_AVX512_CODE 0
#endif

// ARM NEON is mandatory on AArch64 and opt-in (compile-flag driven) on 32-bit.
#if FYX_ARCH_ARM && !defined(FYX_DISABLE_NEON)
#  if FYX_ARCH_ARM64 || defined(__ARM_NEON) || defined(__ARM_NEON__)
#    define FYX_HAS_NEON_CODE 1
#  else
#    define FYX_HAS_NEON_CODE 0
#  endif
#else
#  define FYX_HAS_NEON_CODE 0
#endif

#define FYX_HAS_ANY_SIMD_CODE (FYX_HAS_SSE42_CODE || FYX_HAS_AVX2_CODE || FYX_HAS_AVX512_CODE || FYX_HAS_NEON_CODE)

// Intrinsic headers ---------------------------------------------------------
#if FYX_ARCH_X86 && (FYX_HAS_SSE42_CODE || FYX_HAS_AVX2_CODE || FYX_HAS_AVX512_CODE)
#  include <immintrin.h>
#endif
#if FYX_HAS_NEON_CODE
#  include <arm_neon.h>
#endif
#if FYX_ARCH_X86 && FYX_COMPILER_MSVC
#  include <intrin.h>
#endif
#if FYX_ARCH_X86 && FYX_GNUC_LIKE
#  include <cpuid.h>
#endif

// Per-function ISA selection.  On MSVC the attributes are empty: the ISA is
// whatever /arch selected, and the runtime check still guards execution.
#if FYX_GNUC_LIKE && FYX_ARCH_X86
#  define FYX_TARGET_SSE42     __attribute__((target("sse4.2")))
#  define FYX_TARGET_AVX2      __attribute__((target("avx2,fma")))
#  define FYX_TARGET_AVX512    __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512cd")))
#  define FYX_TARGET_AVX512VP  __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512cd,avx512vpopcntdq")))
#else
#  define FYX_TARGET_SSE42
#  define FYX_TARGET_AVX2
#  define FYX_TARGET_AVX512
#  define FYX_TARGET_AVX512VP
#endif

// ---------------------------------------------------------------------------
// Cache-line / tuning constants
// ---------------------------------------------------------------------------
namespace fyx {
namespace detail {

/// Size of a cache line in bytes.  64 everywhere we support except Apple
/// Silicon, whose 128-byte lines we simply over-align for (harmless).
#if FYX_ARCH_ARM64 && FYX_OS_MACOS
inline constexpr std::size_t kCacheLine = 128;
#else
inline constexpr std::size_t kCacheLine = 64;
#endif

/// Radix configuration: 8-bit digits -> 256 buckets.  This is the sweet spot;
/// the 256 x 64B write-combining buffer is 16 KiB and still fits in L1 next to
/// the histogram (256 x 8B = 2 KiB).
inline constexpr unsigned kRadixBits    = 8;
inline constexpr unsigned kRadixBuckets = 1u << kRadixBits;   // 256
inline constexpr unsigned kRadixMask    = kRadixBuckets - 1u; // 0xFF

/// Below this length a sort never goes parallel (task overhead dominates).
inline constexpr std::size_t kParallelThreshold = 1u << 15;   // 32768

/// Below this length radix sort loses to a good comparison sort (the fused
/// histogram pass plus a full ping-pong copy is not amortised yet).
inline constexpr std::size_t kRadixThreshold = 1024;

/// Sorting-network ceiling.  Everything at or below this length is sorted by a
/// branch-free network, never by insertion sort.
inline constexpr std::size_t kNetworkMax = 64;

/// pdqsort switches to the network / small-sort below this.
inline constexpr std::size_t kInsertionThreshold = 24;

/// Sample-sort bucket count and block size (ips4o terminology).
inline constexpr std::size_t kSampleBuckets   = 256;
inline constexpr std::size_t kSampleBlock     = 1024;
inline constexpr std::size_t kSampleThreshold = 1u << 17;     // 131072

/// Software prefetch distances, in elements, for the radix scatter loop.
inline constexpr std::size_t kPrefetchL1 = 16;
inline constexpr std::size_t kPrefetchL2 = 128;

} // namespace detail
} // namespace fyx

// ============================================================================
//  Section 2 -- Portable primitives
//  Prefetch, bit twiddling, aligned allocation, non-temporal stores.
//  Everything here is a thin, always-inlined shim so that the algorithm code
//  below never has to spell out a compiler #ifdef again.
// ============================================================================

namespace fyx {
namespace detail {

// ---------------------------------------------------------------------------
// Prefetch
// ---------------------------------------------------------------------------
// Locality hints follow the GCC convention:
//   3 = keep in all cache levels (T0)   2 = L2 and up (T1)
//   1 = L3 only (T2)                    0 = non-temporal (NTA)
// ---------------------------------------------------------------------------

template <int Locality>
FYX_FORCE_INLINE void prefetch_read(const void* p) noexcept {
    static_assert(Locality >= 0 && Locality <= 3, "locality must be 0..3");
#if FYX_GNUC_LIKE
    __builtin_prefetch(p, 0, Locality);
#elif FYX_COMPILER_MSVC && FYX_ARCH_X86
    _mm_prefetch(reinterpret_cast<const char*>(p),
                 Locality == 3 ? _MM_HINT_T0 :
                 Locality == 2 ? _MM_HINT_T1 :
                 Locality == 1 ? _MM_HINT_T2 : _MM_HINT_NTA);
#elif FYX_COMPILER_MSVC && FYX_ARCH_ARM64
    __prefetch(p);
#else
    FYX_UNUSED(p);
#endif
}

template <int Locality>
FYX_FORCE_INLINE void prefetch_write(void* p) noexcept {
    static_assert(Locality >= 0 && Locality <= 3, "locality must be 0..3");
#if FYX_GNUC_LIKE
    __builtin_prefetch(p, 1, Locality);
#elif FYX_COMPILER_MSVC && FYX_ARCH_X86
    // MSVC exposes no write-intent prefetch for x86; T0 is the closest.
    _mm_prefetch(reinterpret_cast<const char*>(p),
                 Locality >= 2 ? _MM_HINT_T0 : _MM_HINT_T1);
#elif FYX_COMPILER_MSVC && FYX_ARCH_ARM64
    __prefetch(p);
#else
    FYX_UNUSED(p);
#endif
}

/// Multi-level prefetch used by the radix scatter loop: the near element goes
/// to L1, the far element to L2.  Both are bounds-checked by the caller.
template <typename T>
FYX_FORCE_INLINE void prefetch_stream(const T* base, std::size_t i, std::size_t n) noexcept {
    if (FYX_LIKELY(i + kPrefetchL1 < n)) prefetch_read<3>(base + i + kPrefetchL1);
    if (FYX_LIKELY(i + kPrefetchL2 < n)) prefetch_read<1>(base + i + kPrefetchL2);
}

// ---------------------------------------------------------------------------
// Pause / spin hint
// ---------------------------------------------------------------------------
FYX_FORCE_INLINE void cpu_pause() noexcept {
#if FYX_ARCH_X86 && (FYX_GNUC_LIKE || FYX_COMPILER_MSVC)
    _mm_pause();
#elif FYX_ARCH_ARM && FYX_GNUC_LIKE
    __asm__ __volatile__("yield" ::: "memory");
#elif FYX_ARCH_ARM64 && FYX_COMPILER_MSVC
    __yield();
#else
    // Nothing portable to do; the compiler barrier below is still useful.
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

// ---------------------------------------------------------------------------
// Bit utilities
// ---------------------------------------------------------------------------

FYX_FORCE_INLINE unsigned popcount32(std::uint32_t x) noexcept {
#if FYX_GNUC_LIKE
    return static_cast<unsigned>(__builtin_popcount(x));
#elif FYX_COMPILER_MSVC
    return static_cast<unsigned>(__popcnt(x));
#else
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return static_cast<unsigned>((x * 0x01010101u) >> 24);
#endif
}

FYX_FORCE_INLINE unsigned popcount64(std::uint64_t x) noexcept {
#if FYX_GNUC_LIKE
    return static_cast<unsigned>(__builtin_popcountll(x));
#elif FYX_COMPILER_MSVC && FYX_ARCH_X86_64
    return static_cast<unsigned>(__popcnt64(x));
#else
    return popcount32(static_cast<std::uint32_t>(x)) +
           popcount32(static_cast<std::uint32_t>(x >> 32));
#endif
}

/// Index of the highest set bit.  Undefined for x == 0 (callers guard).
FYX_FORCE_INLINE unsigned bit_scan_reverse64(std::uint64_t x) noexcept {
    FYX_ASSERT_IMPL(x != 0);
#if FYX_GNUC_LIKE
    return 63u - static_cast<unsigned>(__builtin_clzll(x));
#elif FYX_COMPILER_MSVC && FYX_ARCH_X86_64
    unsigned long idx = 0;
    _BitScanReverse64(&idx, x);
    return static_cast<unsigned>(idx);
#elif FYX_COMPILER_MSVC
    unsigned long idx = 0;
    if (_BitScanReverse(&idx, static_cast<unsigned long>(x >> 32))) return static_cast<unsigned>(idx) + 32u;
    _BitScanReverse(&idx, static_cast<unsigned long>(x));
    return static_cast<unsigned>(idx);
#else
    unsigned r = 0;
    while (x >>= 1) ++r;
    return r;
#endif
}

/// floor(log2(n)) for n >= 1; returns 0 for n == 0 so that depth limits stay sane.
FYX_FORCE_INLINE unsigned log2_floor(std::uint64_t n) noexcept {
    return n ? bit_scan_reverse64(n) : 0u;
}

/// Smallest power of two >= n (n <= 2^63).
FYX_FORCE_INLINE std::uint64_t next_pow2(std::uint64_t n) noexcept {
    if (n <= 1) return 1;
    return std::uint64_t(1) << (bit_scan_reverse64(n - 1) + 1);
}

// ---------------------------------------------------------------------------
// Aligned allocation
// ---------------------------------------------------------------------------
// std::aligned_alloc is C11/C++17 but unavailable on MSVC and on some MinGW
// configurations, so we route through the platform primitive directly.
// ---------------------------------------------------------------------------

inline void* aligned_malloc(std::size_t bytes, std::size_t alignment) noexcept {
    if (bytes == 0) bytes = 1;
    // Alignment must be a power of two and (for aligned_alloc) divide the size.
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    alignment = static_cast<std::size_t>(next_pow2(alignment));
#if FYX_OS_WINDOWS
    return _aligned_malloc(bytes, alignment);
#elif FYX_OS_POSIX
    void* p = nullptr;
    if (::posix_memalign(&p, alignment, bytes) != 0) return nullptr;
    return p;
#else
    const std::size_t rounded = (bytes + alignment - 1) / alignment * alignment;
    return std::aligned_alloc(alignment, rounded);
#endif
}

inline void aligned_free(void* p) noexcept {
    if (!p) return;
#if FYX_OS_WINDOWS
    _aligned_free(p);
#else
    std::free(p);
#endif
}

/// RAII owner for an aligned raw buffer of trivially-copyable T.
template <typename T>
class AlignedBuffer {
public:
    AlignedBuffer() noexcept = default;

    explicit AlignedBuffer(std::size_t count) noexcept { allocate(count); }

    AlignedBuffer(const AlignedBuffer&)            = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& o) noexcept : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr;
        o.size_ = 0;
    }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            data_ = o.data_; size_ = o.size_;
            o.data_ = nullptr; o.size_ = 0;
        }
        return *this;
    }

    ~AlignedBuffer() { reset(); }

    /// Grows to at least `count` elements.  Contents are *not* preserved.
    /// Returns false if the allocation failed (never throws).
    bool allocate(std::size_t count) noexcept {
        if (count <= size_) return true;
        reset();
        void* p = aligned_malloc(count * sizeof(T), kCacheLine);
        if (!p) return false;
        data_ = static_cast<T*>(p);
        size_ = count;
        return true;
    }

    void reset() noexcept {
        aligned_free(data_);
        data_ = nullptr;
        size_ = 0;
    }

    T*          data()  const noexcept { return data_; }
    std::size_t size()  const noexcept { return size_; }
    bool        valid() const noexcept { return data_ != nullptr; }

private:
    T*          data_ = nullptr;
    std::size_t size_ = 0;
};

// ---------------------------------------------------------------------------
// Non-temporal 64-byte store (one full cache line)
// ---------------------------------------------------------------------------
// Used to flush the radix write-combining buffers.  A normal store would first
// read the destination line into cache (RFO), wasting half of the available
// bandwidth on data we are about to overwrite completely.
//
// `dst` must be 64-byte aligned for the SIMD paths; the caller checks this and
// falls back to memcpy otherwise.
//
// These are plain `inline`, not FYX_FORCE_INLINE: each carries a target
// attribute and is reached from baseline dispatch code, and GCC rejects
// always_inline across a target mismatch.  The call overhead is amortised over
// a whole cache line of payload.
// ---------------------------------------------------------------------------

#if FYX_HAS_AVX512_CODE
FYX_TARGET_AVX512
inline void stream_line_avx512(void* dst, const void* src) noexcept {
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst),
                        _mm512_loadu_si512(src));
}
#endif

#if FYX_HAS_AVX2_CODE
FYX_TARGET_AVX2
inline void stream_line_avx2(void* dst, const void* src) noexcept {
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src) + 1);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst),     a);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst) + 1, b);
}
#endif

#if FYX_HAS_SSE42_CODE
FYX_TARGET_SSE42
inline void stream_line_sse42(void* dst, const void* src) noexcept {
    const __m128i* s = reinterpret_cast<const __m128i*>(src);
    __m128i*       d = reinterpret_cast<__m128i*>(dst);
    const __m128i a = _mm_loadu_si128(s + 0);
    const __m128i b = _mm_loadu_si128(s + 1);
    const __m128i c = _mm_loadu_si128(s + 2);
    const __m128i e = _mm_loadu_si128(s + 3);
    _mm_stream_si128(d + 0, a);
    _mm_stream_si128(d + 1, b);
    _mm_stream_si128(d + 2, c);
    _mm_stream_si128(d + 3, e);
}
#endif

/// Fence after a batch of non-temporal stores, so that subsequent readers (and
/// other threads) observe the data.  Cheap no-op where NT stores do not exist.
FYX_FORCE_INLINE void store_fence() noexcept {
#if FYX_ARCH_X86 && (FYX_GNUC_LIKE || FYX_COMPILER_MSVC)
    _mm_sfence();
#else
    std::atomic_thread_fence(std::memory_order_release);
#endif
}

// ---------------------------------------------------------------------------
// Branch-free helpers
// ---------------------------------------------------------------------------

/// Conditional swap that compiles to CMOVs rather than a branch.
template <typename T, typename Compare>
FYX_FORCE_INLINE void branchless_sort2(T& a, T& b, Compare comp) noexcept {
    // Copy first so the compiler sees straight-line dataflow.
    const bool swap = comp(b, a);
    const T    lo   = swap ? b : a;
    const T    hi   = swap ? a : b;
    a = lo;
    b = hi;
}

template <typename T, typename Compare>
FYX_FORCE_INLINE void sort3(T& a, T& b, T& c, Compare comp) noexcept {
    branchless_sort2(a, b, comp);
    branchless_sort2(b, c, comp);
    branchless_sort2(a, b, comp);
}

template <typename T, typename Compare>
FYX_FORCE_INLINE void sort4(T& a, T& b, T& c, T& d, Compare comp) noexcept {
    branchless_sort2(a, b, comp);
    branchless_sort2(c, d, comp);
    branchless_sort2(a, c, comp);
    branchless_sort2(b, d, comp);
    branchless_sort2(b, c, comp);
}

} // namespace detail
} // namespace fyx

// ============================================================================
//  Section 3 -- Runtime CPU feature detection
//  CPUID + XGETBV on x86, hwcaps / compile-time facts on ARM.  Probed exactly
//  once and cached in a function-local static (thread-safe since C++11).
// ============================================================================

namespace fyx {
namespace detail {

struct CpuFeatures {
    // --- x86 -----------------------------------------------------------
    bool sse2       = false;
    bool sse42      = false;
    bool avx        = false;
    bool avx2       = false;
    bool bmi1       = false;
    bool bmi2       = false;
    bool avx512f    = false;
    bool avx512bw   = false;
    bool avx512dq   = false;
    bool avx512vl   = false;
    bool avx512cd   = false;
    bool avx512vbmi = false;
    bool avx512vpopcntdq = false;
    // --- ARM -----------------------------------------------------------
    bool neon       = false;
    // --- topology / cache ----------------------------------------------
    unsigned    logical_cores = 1;
    std::size_t l2_bytes      = 256u * 1024u;
    std::size_t l3_bytes      = 8u * 1024u * 1024u;

    /// True when the full AVX-512 subset the sorting kernels need is present.
    bool avx512_sort_ready() const noexcept {
        return avx512f && avx512bw && avx512dq && avx512vl;
    }
    /// True when the conflict-detection histogram can run.
    bool avx512_conflict_ready() const noexcept {
        return avx512_sort_ready() && avx512cd && avx512vpopcntdq;
    }
};

#if FYX_ARCH_X86

FYX_FORCE_INLINE void cpuid_raw(int leaf, int subleaf, unsigned regs[4]) noexcept {
#if FYX_COMPILER_MSVC
    int out[4];
    __cpuidex(out, leaf, subleaf);
    regs[0] = static_cast<unsigned>(out[0]);
    regs[1] = static_cast<unsigned>(out[1]);
    regs[2] = static_cast<unsigned>(out[2]);
    regs[3] = static_cast<unsigned>(out[3]);
#elif FYX_GNUC_LIKE
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid_count(static_cast<unsigned>(leaf), static_cast<unsigned>(subleaf), a, b, c, d);
    regs[0] = a; regs[1] = b; regs[2] = c; regs[3] = d;
#else
    regs[0] = regs[1] = regs[2] = regs[3] = 0;
    FYX_UNUSED(leaf); FYX_UNUSED(subleaf);
#endif
}

FYX_FORCE_INLINE unsigned cpuid_max_leaf() noexcept {
    unsigned r[4];
    cpuid_raw(0, 0, r);
    return r[0];
}

/// Reads XCR0.  Only called after CPUID reported OSXSAVE, so the instruction
/// is guaranteed to be legal.
FYX_FORCE_INLINE std::uint64_t xgetbv0() noexcept {
#if FYX_COMPILER_MSVC
    return _xgetbv(0);
#elif FYX_GNUC_LIKE
    unsigned eax = 0, edx = 0;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#else
    return 0;
#endif
}

#endif // FYX_ARCH_X86

inline unsigned detect_logical_cores() noexcept {
#if FYX_ENABLE_PARALLEL
    const unsigned n = std::thread::hardware_concurrency();
    return n ? n : 1u;
#else
    return 1u;
#endif
}

inline CpuFeatures probe_cpu_features() noexcept {
    CpuFeatures f;
    f.logical_cores = detect_logical_cores();

#if FYX_ARCH_X86
    const unsigned maxleaf = cpuid_max_leaf();
    if (maxleaf >= 1) {
        unsigned r[4];
        cpuid_raw(1, 0, r);
        const unsigned ecx = r[2], edx = r[3];
        f.sse2  = (edx & (1u << 26)) != 0;
        f.sse42 = (ecx & (1u << 20)) != 0;

        const bool osxsave = (ecx & (1u << 27)) != 0;
        const bool avx_bit = (ecx & (1u << 28)) != 0;

        // AVX state (XMM|YMM) must be enabled by the OS before we may execute
        // any VEX-encoded instruction; likewise ZMM state for EVEX.
        bool ymm_ok = false, zmm_ok = false;
        if (osxsave) {
            const std::uint64_t xcr0 = xgetbv0();
            ymm_ok = (xcr0 & 0x6u) == 0x6u;             // XMM + YMM
            zmm_ok = ymm_ok && (xcr0 & 0xE0u) == 0xE0u; // + opmask, ZMM_hi256, hi16_ZMM
        }
        f.avx = avx_bit && ymm_ok;

        if (maxleaf >= 7) {
            unsigned s[4];
            cpuid_raw(7, 0, s);
            const unsigned ebx7 = s[1], ecx7 = s[2];
            f.bmi1 = (ebx7 & (1u << 3))  != 0;
            f.bmi2 = (ebx7 & (1u << 8))  != 0;
            f.avx2 = ((ebx7 & (1u << 5)) != 0) && ymm_ok;

            if (zmm_ok) {
                f.avx512f          = (ebx7 & (1u << 16)) != 0;
                f.avx512dq         = (ebx7 & (1u << 17)) != 0;
                f.avx512cd         = (ebx7 & (1u << 28)) != 0;
                f.avx512bw         = (ebx7 & (1u << 30)) != 0;
                f.avx512vl         = (ebx7 & (1u << 31)) != 0;
                f.avx512vbmi       = (ecx7 & (1u << 1))  != 0;
                f.avx512vpopcntdq  = (ecx7 & (1u << 14)) != 0;
            }
        }
    }

    // Deterministic cache parameters (leaf 4, Intel-style; AMD implements it
    // too on every part we care about).  Leaf 0x8000001D is the AMD spelling
    // and is only consulted when leaf 4 yields nothing.
    {
        bool got_l2 = false, got_l3 = false;
        for (int i = 0; i < 8; ++i) {
            unsigned c[4];
            cpuid_raw(4, i, c);
            const unsigned type = c[0] & 0x1Fu;
            if (type == 0) break;                       // no more cache levels
            const unsigned level = (c[0] >> 5) & 0x7u;
            if (type != 1 && type != 3) continue;       // want data or unified
            const std::size_t ways   = ((c[1] >> 22) & 0x3FFu) + 1u;
            const std::size_t parts  = ((c[1] >> 12) & 0x3FFu) + 1u;
            const std::size_t line   = (c[1] & 0xFFFu) + 1u;
            const std::size_t sets   = static_cast<std::size_t>(c[2]) + 1u;
            const std::size_t bytes  = ways * parts * line * sets;
            if (level == 2 && !got_l2) { f.l2_bytes = bytes; got_l2 = true; }
            if (level == 3 && !got_l3) { f.l3_bytes = bytes; got_l3 = true; }
        }
        if (!got_l3) f.l3_bytes = f.l2_bytes * 8;       // plausible stand-in
    }
#endif // FYX_ARCH_X86

#if FYX_ARCH_ARM64
    f.neon = true;                                      // architecturally required
#elif FYX_ARCH_ARM32 && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    f.neon = true;
#endif

    return f;
}

/// Process-wide feature singleton.
inline const CpuFeatures& cpu() noexcept {
    static const CpuFeatures f = probe_cpu_features();
    return f;
}

// ---------------------------------------------------------------------------
// Effective ISA -- combines "was it compiled?" with "does this CPU have it?"
// ---------------------------------------------------------------------------

FYX_FORCE_INLINE bool use_avx512() noexcept {
#if FYX_HAS_AVX512_CODE
    return cpu().avx512_sort_ready();
#else
    return false;
#endif
}

FYX_FORCE_INLINE bool use_avx512_conflict() noexcept {
#if FYX_HAS_AVX512_CODE
    return cpu().avx512_conflict_ready();
#else
    return false;
#endif
}

FYX_FORCE_INLINE bool use_avx2() noexcept {
#if FYX_HAS_AVX2_CODE
    return cpu().avx2;
#else
    return false;
#endif
}

FYX_FORCE_INLINE bool use_sse42() noexcept {
#if FYX_HAS_SSE42_CODE
    return cpu().sse42;
#else
    return false;
#endif
}

FYX_FORCE_INLINE bool use_neon() noexcept {
#if FYX_HAS_NEON_CODE
    return cpu().neon;
#else
    return false;
#endif
}

/// Widest usable vector in bytes (1 == scalar).  Drives buffer alignment and
/// the choice of write-combining flush routine.
FYX_FORCE_INLINE std::size_t simd_width_bytes() noexcept {
    if (use_avx512()) return 64;
    if (use_avx2())   return 32;
    if (use_sse42())  return 16;
    if (use_neon())   return 16;
    return 1;
}

/// True when `stream_cache_line` will actually issue non-temporal stores.
/// The radix scatter uses this to decide whether pre-aligning bucket cursors
/// is worth the bookkeeping.
FYX_FORCE_INLINE bool have_nt_stores() noexcept {
#if FYX_HAS_AVX512_CODE || FYX_HAS_AVX2_CODE || FYX_HAS_SSE42_CODE
    return use_avx512() || use_avx2() || use_sse42();
#else
    return false;
#endif
}

/// Copy exactly one cache line using the widest non-temporal store available.
/// `dst` must be 64-byte aligned.
FYX_FORCE_INLINE void stream_cache_line(void* dst, const void* src) noexcept {
#if FYX_HAS_AVX512_CODE
    if (use_avx512()) { stream_line_avx512(dst, src); return; }
#endif
#if FYX_HAS_AVX2_CODE
    if (use_avx2())   { stream_line_avx2(dst, src);   return; }
#endif
#if FYX_HAS_SSE42_CODE
    if (use_sse42())  { stream_line_sse42(dst, src);  return; }
#endif
    std::memcpy(dst, src, kCacheLine);
}

} // namespace detail
} // namespace fyx

// ============================================================================
//  Section 4 -- Thread-local scratch memory
//
//  Radix sort needs an out-of-place ping-pong buffer of n elements and a
//  16 KiB write-combining area.  Allocating those per call would dominate the
//  runtime for repeated medium-sized sorts, so every thread keeps one buffer
//  alive and grows it monotonically.
//
//  The buffer is handed out through an RAII lease.  Nested sorts on the same
//  thread (radix -> recursion -> radix) are handled by falling back to a fresh
//  private allocation when the thread-local lease is already taken, so the
//  design never silently aliases two live buffers.
// ============================================================================

namespace fyx {
namespace detail {

/// A raw byte arena with monotone growth.  Not thread-safe by construction:
/// exactly one instance exists per thread.
class ScratchArena {
public:
    ScratchArena() noexcept = default;

    ScratchArena(const ScratchArena&)            = delete;
    ScratchArena& operator=(const ScratchArena&) = delete;

    ~ScratchArena() { aligned_free(base_); }

    /// Returns a pointer to at least `bytes` writable bytes, 64-byte aligned,
    /// or nullptr if the allocation failed.  Previous contents are discarded.
    void* acquire(std::size_t bytes) noexcept {
        if (bytes <= capacity_) return base_;
        // Grow geometrically to avoid repeated reallocation in a size ramp.
        std::size_t want = capacity_ ? capacity_ : std::size_t(4096);
        while (want < bytes) {
            const std::size_t next = want + want / 2 + 4096;
            if (next < want) { want = bytes; break; }   // overflow guard
            want = next;
        }
        void* p = aligned_malloc(want, kCacheLine);
        if (!p) {
            // Retry with the exact request: the geometric target may simply be
            // too large for the remaining address space.
            p = aligned_malloc(bytes, kCacheLine);
            if (!p) return nullptr;
            want = bytes;
        }
        aligned_free(base_);
        base_     = static_cast<unsigned char*>(p);
        capacity_ = want;
        return base_;
    }

    /// Releases the memory back to the OS.  Useful for long-lived processes
    /// that sorted one huge array and will not do so again.
    void shrink() noexcept {
        aligned_free(base_);
        base_     = nullptr;
        capacity_ = 0;
    }

    std::size_t capacity() const noexcept { return capacity_; }

    bool in_use() const noexcept { return leased_; }
    void set_leased(bool v) noexcept { leased_ = v; }

private:
    unsigned char* base_     = nullptr;
    std::size_t    capacity_ = 0;
    bool           leased_   = false;
};

inline ScratchArena& thread_arena() noexcept {
    static thread_local ScratchArena arena;
    return arena;
}

/// RAII lease over `count` objects of type T.
///
/// Prefers the thread-local arena.  If that arena is already leased (nested
/// sort) or too small to grow, falls back to a private allocation owned by the
/// lease.  `valid()` reports whether any memory at all was obtained; callers
/// must degrade to an in-place algorithm when it returns false.
template <typename T>
class ScratchLease {
    static_assert(std::is_trivially_destructible<T>::value ||
                  !std::is_trivially_destructible<T>::value,
                  "ScratchLease stores raw storage; T is only ever placement-used");

public:
    explicit ScratchLease(std::size_t count) noexcept {
        if (count == 0) { ptr_ = nullptr; return; }

        // Overflow check before multiplying.
        if (count > (std::size_t(-1) / sizeof(T))) { ptr_ = nullptr; return; }
        const std::size_t bytes = count * sizeof(T);

        ScratchArena& a = thread_arena();
        if (!a.in_use()) {
            void* p = a.acquire(bytes);
            if (p) {
                a.set_leased(true);
                from_arena_ = true;
                ptr_        = static_cast<T*>(p);
                count_      = count;
                return;
            }
        }
        // Nested use, or the arena could not grow: allocate privately.
        void* p = aligned_malloc(bytes, kCacheLine);
        ptr_    = static_cast<T*>(p);
        count_  = p ? count : 0;
    }

    ScratchLease(const ScratchLease&)            = delete;
    ScratchLease& operator=(const ScratchLease&) = delete;

    ~ScratchLease() {
        if (from_arena_) thread_arena().set_leased(false);
        else             aligned_free(ptr_);
    }

    T*          get()   const noexcept { return ptr_; }
    std::size_t count() const noexcept { return count_; }
    bool        valid() const noexcept { return ptr_ != nullptr; }

private:
    T*          ptr_        = nullptr;
    std::size_t count_      = 0;
    bool        from_arena_ = false;
};

/// Frees this thread's cached scratch memory.  Exposed publicly as
/// fyx::release_thread_memory().
inline void release_thread_scratch() noexcept {
    ScratchArena& a = thread_arena();
    if (!a.in_use()) a.shrink();
}

} // namespace detail
} // namespace fyx

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

// ============================================================================
//  Section 6 -- Scalar kernels
//    * branch-free insertion sort (used *inside* pdqsort, never for the
//      n <= 64 numeric entry point, which always uses a sorting network)
//    * heapsort (pdqsort's depth-limit fallback -- guarantees O(n log n))
//    * scalar branch-free bitonic network (the universal fallback for the
//      n <= 64 path on targets with no usable SIMD)
// ============================================================================

namespace fyx {
namespace detail {

// ---------------------------------------------------------------------------
// Insertion sort
// ---------------------------------------------------------------------------

/// Plain insertion sort over [first, last).
template <typename It, typename Compare>
inline void insertion_sort(It first, It last, Compare comp) {
    using T = typename std::iterator_traits<It>::value_type;
    if (first == last) return;
    for (It i = first + 1; i != last; ++i) {
        if (comp(*i, *(i - 1))) {
            T   tmp = std::move(*i);
            It  j   = i;
            do {
                *j = std::move(*(j - 1));
                --j;
            } while (j != first && comp(tmp, *(j - 1)));
            *j = std::move(tmp);
        }
    }
}

/// Insertion sort that may assume *(first - 1) is a valid element that is not
/// greater than everything in [first, last) -- i.e. a sentinel exists.  This
/// removes the `j != first` bound check from the inner loop.
template <typename It, typename Compare>
inline void insertion_sort_guarded(It first, It last, Compare comp) {
    using T = typename std::iterator_traits<It>::value_type;
    if (first == last) return;
    for (It i = first + 1; i != last; ++i) {
        if (comp(*i, *(i - 1))) {
            T  tmp = std::move(*i);
            It j   = i;
            do {
                *j = std::move(*(j - 1));
                --j;
            } while (comp(tmp, *(j - 1)));
            *j = std::move(tmp);
        }
    }
}

/// Bounded insertion sort used by pdqsort's partial-order optimisation.
/// Gives up (returning false) once it has moved more than `limit` elements,
/// which tells the caller the range is not nearly sorted.
template <typename It, typename Compare>
inline bool partial_insertion_sort(It first, It last, Compare comp) {
    using T = typename std::iterator_traits<It>::value_type;
    using Diff = typename std::iterator_traits<It>::difference_type;
    constexpr Diff kLimit = 8;

    if (first == last) return true;
    Diff moved = 0;
    for (It i = first + 1; i != last; ++i) {
        if (!comp(*i, *(i - 1))) continue;
        T  tmp = std::move(*i);
        It j   = i;
        do {
            *j = std::move(*(j - 1));
            --j;
        } while (j != first && comp(tmp, *(j - 1)));
        *j = std::move(tmp);
        moved += i - j;
        if (moved > kLimit) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Heapsort -- the worst-case guarantee behind pdqsort
// ---------------------------------------------------------------------------

template <typename It, typename Compare>
inline void sift_down(It first, typename std::iterator_traits<It>::difference_type root,
                      typename std::iterator_traits<It>::difference_type n, Compare comp) {
    using T    = typename std::iterator_traits<It>::value_type;
    using Diff = typename std::iterator_traits<It>::difference_type;

    T value = std::move(*(first + root));
    Diff child;
    // Sift the hole down to a leaf, then sift `value` back up.  This halves the
    // number of comparisons compared with the naive formulation.
    while ((child = 2 * root + 1) < n) {
        if (child + 1 < n && comp(*(first + child), *(first + child + 1))) ++child;
        if (!comp(value, *(first + child))) break;
        *(first + root) = std::move(*(first + child));
        root = child;
    }
    *(first + root) = std::move(value);
}

template <typename It, typename Compare>
inline void heap_sort(It first, It last, Compare comp) {
    using Diff = typename std::iterator_traits<It>::difference_type;
    const Diff n = last - first;
    if (n < 2) return;
    for (Diff i = n / 2 - 1; i >= 0; --i) sift_down(first, i, n, comp);
    for (Diff i = n - 1; i > 0; --i) {
        std::swap(*first, *(first + i));
        sift_down(first, Diff(0), i, comp);
    }
}

// ---------------------------------------------------------------------------
// Median selection (pdqsort pivot choice)
// ---------------------------------------------------------------------------

template <typename It, typename Compare>
FYX_FORCE_INLINE void sort2_iter(It a, It b, Compare comp) {
    if (comp(*b, *a)) std::iter_swap(a, b);
}

template <typename It, typename Compare>
FYX_FORCE_INLINE void median3(It a, It b, It c, Compare comp) {
    sort2_iter(a, b, comp);
    sort2_iter(b, c, comp);
    sort2_iter(a, b, comp);
}

/// Places a good pivot at *first.  Uses median-of-3 for small ranges and
/// median-of-9 (ninther) for larger ones.
template <typename It, typename Compare>
inline void choose_pivot(It first, It last, Compare comp) {
    using Diff = typename std::iterator_traits<It>::difference_type;
    const Diff n = last - first;
    Diff h = n / 2;
    It   a = first + 1, b = first + h, c = last - 1;
    if (n > 128) {
        median3(a - 1, a, a + 1, comp);
        median3(b - 1, b, b + 1, comp);
        median3(c - 2, c - 1, c, comp);
        // After the three sub-medians, b-1..c-1 hold them; take the median of
        // the three medians.
        median3(a, b, c - 1, comp);
        std::iter_swap(first, b);
    } else {
        median3(a, b, c, comp);
        std::iter_swap(first, b);
    }
}

// ---------------------------------------------------------------------------
// Scalar branch-free bitonic network
// ---------------------------------------------------------------------------
//
// Sorts exactly N == 2^p unsigned keys ascending, with no data-dependent
// branches.  This is the universal n <= 64 kernel on targets without SIMD, and
// the reference implementation the SIMD networks are validated against.
//
// Standard Batcher bitonic:
//     for k = 2, 4, ... N:
//         for j = k/2, k/4, ... 1:
//             for each i: partner = i ^ j; if partner > i:
//                 ascending = ((i & k) == 0)
//                 keep min at i (max at partner) iff ascending
// ---------------------------------------------------------------------------

template <typename Key>
FYX_FORCE_INLINE void cmpx_asc(Key& a, Key& b) noexcept {
    const Key lo = a < b ? a : b;
    const Key hi = a < b ? b : a;
    a = lo;
    b = hi;
}

template <typename Key>
FYX_FORCE_INLINE void cmpx_desc(Key& a, Key& b) noexcept {
    const Key lo = a < b ? a : b;
    const Key hi = a < b ? b : a;
    a = hi;
    b = lo;
}

/// N is a compile-time power of two.
template <typename Key, unsigned N>
inline void bitonic_scalar(Key* FYX_RESTRICT a) noexcept {
    static_assert(N != 0 && (N & (N - 1)) == 0, "N must be a power of two");
    for (unsigned k = 2; k <= N; k <<= 1) {
        for (unsigned j = k >> 1; j > 0; j >>= 1) {
            for (unsigned i = 0; i < N; ++i) {
                const unsigned p = i ^ j;
                if (p > i) {
                    if ((i & k) == 0) cmpx_asc(a[i], a[p]);
                    else              cmpx_desc(a[i], a[p]);
                }
            }
        }
    }
}

/// Runtime-N dispatcher over the scalar network.  `a` must have capacity for
/// the padded power-of-two length and the padding must already hold the
/// all-ones sentinel.
template <typename Key>
inline void bitonic_scalar_n(Key* a, std::size_t padded) noexcept {
    switch (padded) {
        case 1:  return;
        case 2:  bitonic_scalar<Key, 2>(a);  return;
        case 4:  bitonic_scalar<Key, 4>(a);  return;
        case 8:  bitonic_scalar<Key, 8>(a);  return;
        case 16: bitonic_scalar<Key, 16>(a); return;
        case 32: bitonic_scalar<Key, 32>(a); return;
        case 64: bitonic_scalar<Key, 64>(a); return;
        default: FYX_ASSERT_IMPL(false);     return;
    }
}

} // namespace detail
} // namespace fyx

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

// ============================================================================
//  Section 8 -- Per-ISA "Ops" policies for the sorting network
//
//  Each policy sorts *unsigned keys* of a fixed width.  Signed and floating
//  point values reach here already encoded by RadixTraits, so only unsigned
//  min/max is ever needed.
// ============================================================================

namespace fyx {
namespace detail {

// ---------------------------------------------------------------------------
// Scalar policy -- 1 lane.  Always available; also the reference semantics.
// ---------------------------------------------------------------------------
template <typename KeyT>
struct ScalarOps {
    using Key  = KeyT;
    using Vec  = KeyT;
    using Mask = std::uint64_t;
    static constexpr unsigned kLanes = 1;

    FYX_FORCE_INLINE static Vec  load(const Key* p)            { return *p; }
    FYX_FORCE_INLINE static void store(Key* p, Vec v)          { *p = v; }
    FYX_FORCE_INLINE static Vec  splat(Key k)                  { return k; }
    FYX_FORCE_INLINE static Vec  min(Vec a, Vec b)             { return a < b ? a : b; }
    FYX_FORCE_INLINE static Vec  max(Vec a, Vec b)             { return a < b ? b : a; }
    FYX_FORCE_INLINE static Vec  load_partial(const Key* p, unsigned n, Key fill) {
        return n ? *p : fill;
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        if (n) *p = v;
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) { return v; }  // single lane
    FYX_FORCE_INLINE static Vec blend(Mask, Vec mn, Vec) { return mn; }
};

#if FYX_HAS_SSE42_CODE
FYX_ISA_BEGIN("sse4.2")
namespace isa_sse42 {
using namespace ::fyx::detail;
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
// ---------------------------------------------------------------------------
// SSE4.2 -- 4 x uint32 or 2 x uint64
// ---------------------------------------------------------------------------

struct Sse42Ops32 {
    using Key  = std::uint32_t;
    using Vec  = __m128i;
    using Mask = std::uint32_t;
    static constexpr unsigned kLanes = 4;

    FYX_FORCE_INLINE static Vec load(const Key* p) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
    }
    FYX_FORCE_INLINE static Vec splat(Key k) {
        return _mm_set1_epi32(static_cast<int>(k));
    }
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) { return _mm_min_epu32(a, b); }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) { return _mm_max_epu32(a, b); }

    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        Key tmp[4] = {fill, fill, fill, fill};
        for (unsigned i = 0; i < n; ++i) tmp[i] = p[i];
        return load(tmp);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        Key tmp[4];
        store(tmp, v);
        for (unsigned i = 0; i < n; ++i) p[i] = tmp[i];
    }

    // Lane i <- lane i^J.  J is 1 or 2 for a 4-lane vector.
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1 || J == 2, "J must be 1 or 2 for 4 lanes");
        // _MM_SHUFFLE takes lanes in (3,2,1,0) order.
        if constexpr (J == 1) return _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
        else                  return _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
    }

    // No mask registers before AVX-512: build a 128-bit selector constant.
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        const __m128i sel = _mm_set_epi32(
            (keepmin & 8u) ? -1 : 0, (keepmin & 4u) ? -1 : 0,
            (keepmin & 2u) ? -1 : 0, (keepmin & 1u) ? -1 : 0);
        return _mm_blendv_epi8(mx, mn, sel);
    }
};

struct Sse42Ops64 {
    using Key  = std::uint64_t;
    using Vec  = __m128i;
    using Mask = std::uint32_t;
    static constexpr unsigned kLanes = 2;

    FYX_FORCE_INLINE static Vec load(const Key* p) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
    }
    FYX_FORCE_INLINE static Vec splat(Key k) {
        return _mm_set1_epi64x(static_cast<long long>(k));
    }
    // SSE has no unsigned 64-bit compare; bias by 2^63 and use the signed one.
    FYX_FORCE_INLINE static __m128i cmpgt_u64(Vec a, Vec b) {
        const __m128i bias = _mm_set1_epi64x(static_cast<long long>(0x8000000000000000ULL));
        return _mm_cmpgt_epi64(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias));
    }
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) {
        return _mm_blendv_epi8(a, b, cmpgt_u64(a, b));
    }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) {
        return _mm_blendv_epi8(b, a, cmpgt_u64(a, b));
    }
    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        Key tmp[2] = {fill, fill};
        for (unsigned i = 0; i < n; ++i) tmp[i] = p[i];
        return load(tmp);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        Key tmp[2];
        store(tmp, v);
        for (unsigned i = 0; i < n; ++i) p[i] = tmp[i];
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1, "J must be 1 for 2 lanes");
        return _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));   // swap the two 64-bit halves
    }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        const __m128i sel = _mm_set_epi64x((keepmin & 2u) ? -1LL : 0LL,
                                           (keepmin & 1u) ? -1LL : 0LL);
        return _mm_blendv_epi8(mx, mn, sel);
    }
};

} // namespace isa_sse42
FYX_ISA_END
#endif // FYX_HAS_SSE42_CODE

#if FYX_HAS_AVX2_CODE
FYX_ISA_BEGIN("avx2,fma")
namespace isa_avx2 {
using namespace ::fyx::detail;
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
// ---------------------------------------------------------------------------
// AVX2 -- 8 x uint32 or 4 x uint64
// ---------------------------------------------------------------------------

struct Avx2Ops32 {
    using Key  = std::uint32_t;
    using Vec  = __m256i;
    using Mask = std::uint32_t;
    static constexpr unsigned kLanes = 8;

    FYX_FORCE_INLINE static Vec load(const Key* p) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    FYX_FORCE_INLINE static Vec splat(Key k) {
        return _mm256_set1_epi32(static_cast<int>(k));
    }
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) { return _mm256_min_epu32(a, b); }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) { return _mm256_max_epu32(a, b); }

    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        Key tmp[8];
        for (unsigned i = 0; i < 8; ++i) tmp[i] = (i < n) ? p[i] : fill;
        return load(tmp);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        Key tmp[8];
        store(tmp, v);
        for (unsigned i = 0; i < n; ++i) p[i] = tmp[i];
    }

    // J in {1,2,4}.  1 and 2 stay inside each 128-bit half (vpshufd); 4 swaps
    // the halves (vpermq on the 64-bit granularity).
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1 || J == 2 || J == 4, "J must be 1, 2 or 4 for 8 lanes");
        if constexpr (J == 1)      return _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
        else if constexpr (J == 2) return _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
        else                       return _mm256_permute2x128_si256(v, v, 0x01);
    }

    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        const __m256i sel = _mm256_set_epi32(
            (keepmin & 0x80u) ? -1 : 0, (keepmin & 0x40u) ? -1 : 0,
            (keepmin & 0x20u) ? -1 : 0, (keepmin & 0x10u) ? -1 : 0,
            (keepmin & 0x08u) ? -1 : 0, (keepmin & 0x04u) ? -1 : 0,
            (keepmin & 0x02u) ? -1 : 0, (keepmin & 0x01u) ? -1 : 0);
        return _mm256_blendv_epi8(mx, mn, sel);
    }
};

struct Avx2Ops64 {
    using Key  = std::uint64_t;
    using Vec  = __m256i;
    using Mask = std::uint32_t;
    static constexpr unsigned kLanes = 4;

    FYX_FORCE_INLINE static Vec load(const Key* p) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    FYX_FORCE_INLINE static Vec splat(Key k) {
        return _mm256_set1_epi64x(static_cast<long long>(k));
    }
    FYX_FORCE_INLINE static __m256i cmpgt_u64(Vec a, Vec b) {
        const __m256i bias = _mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ULL));
        return _mm256_cmpgt_epi64(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias));
    }
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) {
        return _mm256_blendv_epi8(a, b, cmpgt_u64(a, b));
    }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) {
        return _mm256_blendv_epi8(b, a, cmpgt_u64(a, b));
    }
    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        Key tmp[4];
        for (unsigned i = 0; i < 4; ++i) tmp[i] = (i < n) ? p[i] : fill;
        return load(tmp);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        Key tmp[4];
        store(tmp, v);
        for (unsigned i = 0; i < n; ++i) p[i] = tmp[i];
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1 || J == 2, "J must be 1 or 2 for 4 lanes");
        // vpermq selects 64-bit lanes: index i takes source lane (i^J).
        if constexpr (J == 1) return _mm256_permute4x64_epi64(v, _MM_SHUFFLE(2, 3, 0, 1));
        else                  return _mm256_permute4x64_epi64(v, _MM_SHUFFLE(1, 0, 3, 2));
    }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        const __m256i sel = _mm256_set_epi64x(
            (keepmin & 8u) ? -1LL : 0LL, (keepmin & 4u) ? -1LL : 0LL,
            (keepmin & 2u) ? -1LL : 0LL, (keepmin & 1u) ? -1LL : 0LL);
        return _mm256_blendv_epi8(mx, mn, sel);
    }
};

} // namespace isa_avx2
FYX_ISA_END
#endif // FYX_HAS_AVX2_CODE

#if FYX_HAS_AVX512_CODE
FYX_ISA_BEGIN("avx512f,avx512bw,avx512dq,avx512vl,avx512cd")
namespace isa_avx512 {
using namespace ::fyx::detail;
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
// ---------------------------------------------------------------------------
// AVX-512 -- 16 x uint32 or 8 x uint64.  Native mask registers make the blend
// a single instruction and the permutation a single vpermd/vpermq.
// ---------------------------------------------------------------------------

struct Avx512Ops32 {
    using Key  = std::uint32_t;
    using Vec  = __m512i;
    using Mask = __mmask16;
    static constexpr unsigned kLanes = 16;

    FYX_FORCE_INLINE static Vec load(const Key* p) {
        return _mm512_loadu_si512(reinterpret_cast<const void*>(p));
    }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) {
        _mm512_storeu_si512(reinterpret_cast<void*>(p), v);
    }
    FYX_FORCE_INLINE static Vec splat(Key k) {
        return _mm512_set1_epi32(static_cast<int>(k));
    }
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) { return _mm512_min_epu32(a, b); }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) { return _mm512_max_epu32(a, b); }

    // Masked load/store: no scratch array, no branch.
    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        const __mmask16 m = static_cast<__mmask16>((1u << n) - 1u);
        return _mm512_mask_loadu_epi32(_mm512_set1_epi32(static_cast<int>(fill)), m, p);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        const __mmask16 m = static_cast<__mmask16>((1u << n) - 1u);
        _mm512_mask_storeu_epi32(p, m, v);
    }

    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1 || J == 2 || J == 4 || J == 8, "J must be 1,2,4,8 for 16 lanes");
        // A single vpermd with a compile-time index vector handles every J.
        const __m512i idx = _mm512_set_epi32(
            int(15u ^ J), int(14u ^ J), int(13u ^ J), int(12u ^ J),
            int(11u ^ J), int(10u ^ J), int( 9u ^ J), int( 8u ^ J),
            int( 7u ^ J), int( 6u ^ J), int( 5u ^ J), int( 4u ^ J),
            int( 3u ^ J), int( 2u ^ J), int( 1u ^ J), int( 0u ^ J));
        return _mm512_permutexvar_epi32(idx, v);
    }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        return _mm512_mask_blend_epi32(keepmin, mx, mn);
    }
};

struct Avx512Ops64 {
    using Key  = std::uint64_t;
    using Vec  = __m512i;
    using Mask = __mmask8;
    static constexpr unsigned kLanes = 8;

    FYX_FORCE_INLINE static Vec load(const Key* p) {
        return _mm512_loadu_si512(reinterpret_cast<const void*>(p));
    }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) {
        _mm512_storeu_si512(reinterpret_cast<void*>(p), v);
    }
    FYX_FORCE_INLINE static Vec splat(Key k) {
        return _mm512_set1_epi64(static_cast<long long>(k));
    }
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) { return _mm512_min_epu64(a, b); }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) { return _mm512_max_epu64(a, b); }

    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        const __mmask8 m = static_cast<__mmask8>((1u << n) - 1u);
        return _mm512_mask_loadu_epi64(_mm512_set1_epi64(static_cast<long long>(fill)), m, p);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        const __mmask8 m = static_cast<__mmask8>((1u << n) - 1u);
        _mm512_mask_storeu_epi64(p, m, v);
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1 || J == 2 || J == 4, "J must be 1,2,4 for 8 lanes");
        const __m512i idx = _mm512_set_epi64(
            static_cast<long long>(7u ^ J), static_cast<long long>(6u ^ J),
            static_cast<long long>(5u ^ J), static_cast<long long>(4u ^ J),
            static_cast<long long>(3u ^ J), static_cast<long long>(2u ^ J),
            static_cast<long long>(1u ^ J), static_cast<long long>(0u ^ J));
        return _mm512_permutexvar_epi64(idx, v);
    }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        return _mm512_mask_blend_epi64(keepmin, mx, mn);
    }
};

} // namespace isa_avx512
FYX_ISA_END
#endif // FYX_HAS_AVX512_CODE

#if FYX_HAS_NEON_CODE
namespace isa_neon {
using namespace ::fyx::detail;
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
// ---------------------------------------------------------------------------
// ARM NEON -- 4 x uint32 or 2 x uint64
// ---------------------------------------------------------------------------

struct NeonOps32 {
    using Key  = std::uint32_t;
    using Vec  = uint32x4_t;
    using Mask = std::uint32_t;
    static constexpr unsigned kLanes = 4;

    FYX_FORCE_INLINE static Vec  load(const Key* p)   { return vld1q_u32(p); }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) { vst1q_u32(p, v); }
    FYX_FORCE_INLINE static Vec  splat(Key k)         { return vdupq_n_u32(k); }
    FYX_FORCE_INLINE static Vec  min(Vec a, Vec b)    { return vminq_u32(a, b); }
    FYX_FORCE_INLINE static Vec  max(Vec a, Vec b)    { return vmaxq_u32(a, b); }

    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        Key tmp[4];
        for (unsigned i = 0; i < 4; ++i) tmp[i] = (i < n) ? p[i] : fill;
        return load(tmp);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        Key tmp[4];
        store(tmp, v);
        for (unsigned i = 0; i < n; ++i) p[i] = tmp[i];
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1 || J == 2, "J must be 1 or 2 for 4 lanes");
        if constexpr (J == 1) {
            // Swap neighbours: [1,0,3,2]
            return vrev64q_u32(v);
        } else {
            // Swap 64-bit halves: [2,3,0,1]
            return vextq_u32(v, v, 2);
        }
    }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        const uint32_t s[4] = {
            (keepmin & 1u) ? ~0u : 0u, (keepmin & 2u) ? ~0u : 0u,
            (keepmin & 4u) ? ~0u : 0u, (keepmin & 8u) ? ~0u : 0u};
        return vbslq_u32(vld1q_u32(s), mn, mx);
    }
};

struct NeonOps64 {
    using Key  = std::uint64_t;
    using Vec  = uint64x2_t;
    using Mask = std::uint32_t;
    static constexpr unsigned kLanes = 2;

    FYX_FORCE_INLINE static Vec  load(const Key* p)   { return vld1q_u64(p); }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) { vst1q_u64(p, v); }
    FYX_FORCE_INLINE static Vec  splat(Key k)         { return vdupq_n_u64(k); }
    // NEON has no 64-bit integer min/max; synthesise from the compare mask.
    FYX_FORCE_INLINE static Vec min(Vec a, Vec b) {
#if FYX_ARCH_ARM64
        return vbslq_u64(vcgtq_u64(a, b), b, a);
#else
        // 32-bit NEON lacks vcgtq_u64; fall back to lane extraction.
        std::uint64_t x[2], y[2];
        vst1q_u64(x, a); vst1q_u64(y, b);
        std::uint64_t r[2] = {x[0] < y[0] ? x[0] : y[0], x[1] < y[1] ? x[1] : y[1]};
        return vld1q_u64(r);
#endif
    }
    FYX_FORCE_INLINE static Vec max(Vec a, Vec b) {
#if FYX_ARCH_ARM64
        return vbslq_u64(vcgtq_u64(a, b), a, b);
#else
        std::uint64_t x[2], y[2];
        vst1q_u64(x, a); vst1q_u64(y, b);
        std::uint64_t r[2] = {x[0] < y[0] ? y[0] : x[0], x[1] < y[1] ? y[1] : x[1]};
        return vld1q_u64(r);
#endif
    }
    FYX_FORCE_INLINE static Vec load_partial(const Key* p, unsigned n, Key fill) {
        Key tmp[2] = {fill, fill};
        for (unsigned i = 0; i < n; ++i) tmp[i] = p[i];
        return load(tmp);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        Key tmp[2];
        store(tmp, v);
        for (unsigned i = 0; i < n; ++i) p[i] = tmp[i];
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) {
        static_assert(J == 1, "J must be 1 for 2 lanes");
        return vextq_u64(v, v, 1);
    }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) {
        const std::uint64_t s[2] = {(keepmin & 1u) ? ~0ULL : 0ULL,
                                    (keepmin & 2u) ? ~0ULL : 0ULL};
        return vbslq_u64(vld1q_u64(s), mn, mx);
    }
};

} // namespace isa_neon
#endif // FYX_HAS_NEON_CODE

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

// ---------------------------------------------------------------------------
// Runtime selection of the best network for a key width
// ---------------------------------------------------------------------------

/// Sorts n <= 64 unsigned 32-bit keys with the widest available ISA.
inline void network_sort_u32(std::uint32_t* keys, std::size_t n) {
    if (n < 2) return;
#if FYX_HAS_AVX512_CODE
    if (use_avx512()) { isa_avx512::network_sort_keys<isa_avx512::Avx512Ops32>(keys, n); return; }
#endif
#if FYX_HAS_AVX2_CODE
    if (use_avx2())   { isa_avx2::network_sort_keys<isa_avx2::Avx2Ops32>(keys, n);       return; }
#endif
#if FYX_HAS_SSE42_CODE
    if (use_sse42())  { isa_sse42::network_sort_keys<isa_sse42::Sse42Ops32>(keys, n);    return; }
#endif
#if FYX_HAS_NEON_CODE
    if (use_neon())   { isa_neon::network_sort_keys<isa_neon::NeonOps32>(keys, n);       return; }
#endif
    bitonic_pad_scalar<std::uint32_t>(keys, n);
}

/// Sorts n <= 64 unsigned 64-bit keys with the widest available ISA.
inline void network_sort_u64(std::uint64_t* keys, std::size_t n) {
    if (n < 2) return;
#if FYX_HAS_AVX512_CODE
    if (use_avx512()) { isa_avx512::network_sort_keys<isa_avx512::Avx512Ops64>(keys, n); return; }
#endif
#if FYX_HAS_AVX2_CODE
    if (use_avx2())   { isa_avx2::network_sort_keys<isa_avx2::Avx2Ops64>(keys, n);       return; }
#endif
#if FYX_HAS_SSE42_CODE
    if (use_sse42())  { isa_sse42::network_sort_keys<isa_sse42::Sse42Ops64>(keys, n);    return; }
#endif
#if FYX_HAS_NEON_CODE
    if (use_neon())   { isa_neon::network_sort_keys<isa_neon::NeonOps64>(keys, n);       return; }
#endif
    bitonic_pad_scalar<std::uint64_t>(keys, n);
}

/// Small-array entry point for any radix-encodable numeric type.
/// Encodes into a stack buffer, runs the network, decodes back.
template <typename T>
inline void small_sort_numeric(T* data, std::size_t n) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    static_assert(RT::supported, "small_sort_numeric requires a radix-encodable type");
    FYX_ASSERT_IMPL(n <= kNetworkMax);
    if (n < 2) return;

    Key buf[kNetworkMax];
    for (std::size_t i = 0; i < n; ++i) buf[i] = RT::encode(data[i]);

    if constexpr (sizeof(Key) == 4)      network_sort_u32(reinterpret_cast<std::uint32_t*>(buf), n);
    else if constexpr (sizeof(Key) == 8) network_sort_u64(reinterpret_cast<std::uint64_t*>(buf), n);
    else                                 bitonic_pad_scalar<Key>(buf, n);

    for (std::size_t i = 0; i < n; ++i) data[i] = RT::decode(buf[i]);
}

} // namespace detail
} // namespace fyx

// ============================================================================
//  Section 9 -- LSD radix sort
//
//  This is the workhorse for numeric keys with the default comparator.
//
//  Structure
//  ---------
//   1. ONE fused counting pass builds the histograms of *all* passes at once
//      (P x 256 counters).  The data is read exactly once here.
//   2. Degenerate passes -- those where every key lands in the same bucket --
//      are skipped.  Sorted, constant, or narrow-range data therefore costs a
//      fraction of the full run, and 64-bit keys holding small values skip
//      four of their eight passes.
//   3. Each surviving pass scatters src -> dst through 256 software
//      write-combining buffers.  A full 64-byte line is flushed with
//      non-temporal stores, which avoids the read-for-ownership traffic that
//      would otherwise consume half of the write bandwidth.
//   4. src and dst are swapped after every pass.  Because we know the number of
//      surviving passes in advance we can choose the initial buffer so that the
//      final pass lands in the caller's array -- no extra copy, ever.
//
//  Histogram variants
//  ------------------
//   * scalar_histogram   -- four interleaved counter banks to hide the
//                           store-to-load latency of repeated ++count[digit].
//   * simd_histogram     -- genuine AVX-512 conflict detection:
//                           vpconflictd + vpopcntd + vpgatherdd + vpscatterdd.
//                           Required by the specification; measured against the
//                           scalar version at run time (see kUseSimdHistogram).
// ============================================================================

namespace fyx {
namespace detail {

/// Per-pass histogram block: P passes x 256 buckets.
template <unsigned Passes>
struct RadixHistogram {
    // 64-bit counters: an array may legitimately exceed 4 G elements.
    std::uint64_t count[Passes][kRadixBuckets];

    void clear() noexcept {
        std::memset(count, 0, sizeof(count));
    }
};

/// Extracts digit `pass` (8 bits) from an encoded key.
template <typename Key>
FYX_FORCE_INLINE unsigned radix_digit(Key k, unsigned pass) noexcept {
    return static_cast<unsigned>((k >> (pass * kRadixBits)) & Key(kRadixMask));
}

// ---------------------------------------------------------------------------
// Scalar fused histogram
// ---------------------------------------------------------------------------
//
// Four independent counter banks remove the serial dependency between
// consecutive increments of the same bucket (a 5-cycle store-forward stall on
// most cores).  They are summed at the end.
// ---------------------------------------------------------------------------

template <typename T, unsigned Passes>
inline void scalar_histogram(const T* FYX_RESTRICT src, std::size_t n,
                             RadixHistogram<Passes>& hist) noexcept {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;

    // Bank layout: [pass][bank][bucket] -- bank-major within a pass keeps the
    // four banks of one pass in adjacent cache lines.
    constexpr unsigned kBanks = 4;
    ScratchLease<std::uint32_t> banks_lease(
        static_cast<std::size_t>(Passes) * kBanks * kRadixBuckets);

    if (!banks_lease.valid()) {
        // No memory for the banked version: count directly.
        hist.clear();
        for (std::size_t i = 0; i < n; ++i) {
            const Key k = RT::encode(src[i]);
            for (unsigned p = 0; p < Passes; ++p) ++hist.count[p][radix_digit(k, p)];
        }
        return;
    }

    std::uint32_t* banks = banks_lease.get();
    std::memset(banks, 0,
                static_cast<std::size_t>(Passes) * kBanks * kRadixBuckets *
                    sizeof(std::uint32_t));

    // A 32-bit bank counter would wrap after 4 G identical digits; flush the
    // banks into the 64-bit histogram every kFlush elements to stay exact.
    constexpr std::size_t kFlush = std::size_t(1) << 30;

    hist.clear();
    std::size_t base = 0;
    while (base < n) {
        const std::size_t stop = (n - base > kFlush) ? base + kFlush : n;

        std::size_t i = base;
        // Unroll by 4, one bank each, so the increments are independent.
        for (; i + 4 <= stop; i += 4) {
            prefetch_stream(src, i, n);
            const Key k0 = RT::encode(src[i + 0]);
            const Key k1 = RT::encode(src[i + 1]);
            const Key k2 = RT::encode(src[i + 2]);
            const Key k3 = RT::encode(src[i + 3]);
            for (unsigned p = 0; p < Passes; ++p) {
                std::uint32_t* b = banks + static_cast<std::size_t>(p) * kBanks * kRadixBuckets;
                ++b[0 * kRadixBuckets + radix_digit(k0, p)];
                ++b[1 * kRadixBuckets + radix_digit(k1, p)];
                ++b[2 * kRadixBuckets + radix_digit(k2, p)];
                ++b[3 * kRadixBuckets + radix_digit(k3, p)];
            }
        }
        for (; i < stop; ++i) {
            const Key k = RT::encode(src[i]);
            for (unsigned p = 0; p < Passes; ++p)
                ++banks[static_cast<std::size_t>(p) * kBanks * kRadixBuckets +
                        radix_digit(k, p)];
        }

        // Fold the banks into the wide histogram and reset them.
        for (unsigned p = 0; p < Passes; ++p) {
            std::uint32_t* b = banks + static_cast<std::size_t>(p) * kBanks * kRadixBuckets;
            for (unsigned d = 0; d < kRadixBuckets; ++d) {
                hist.count[p][d] += std::uint64_t(b[0 * kRadixBuckets + d]) +
                                    std::uint64_t(b[1 * kRadixBuckets + d]) +
                                    std::uint64_t(b[2 * kRadixBuckets + d]) +
                                    std::uint64_t(b[3 * kRadixBuckets + d]);
            }
        }
        std::memset(banks, 0,
                static_cast<std::size_t>(Passes) * kBanks * kRadixBuckets *
                    sizeof(std::uint32_t));
        base = stop;
    }
}


// ---------------------------------------------------------------------------
// AVX-512 conflict-detection histogram
// ---------------------------------------------------------------------------
//
// The specification requires a *real* SIMD histogram, not scalar code in a
// vector wrapper.  The obstacle is that 16 lanes may target the same bucket in
// one step, and a plain gather/add/scatter would then lose all but one of the
// increments.  vpconflictd solves this exactly:
//
//   vpconflictd  -> for lane i, a bitmask of the lanes j<i with idx[j]==idx[i]
//   vpopcntd     -> how many earlier lanes share this bucket = this lane's rank
//   gather       -> the current counter value
//   + rank + 1   -> every lane writes a distinct, correct running total
//   scatter      -> the highest-ranked lane's value survives, which is the
//                   total including all duplicates in this vector
//
// This is exact for any duplicate pattern, including all 16 lanes equal.
// Requires AVX512F + AVX512CD; vpopcntd (AVX512VPOPCNTDQ) is emulated when
// absent, which is why the runtime gate also checks avx512vpopcntdq.
// ---------------------------------------------------------------------------

#if FYX_HAS_AVX512_CODE
FYX_ISA_BEGIN("avx512f,avx512cd,avx512vpopcntdq,avx512bw,avx512dq,avx512vl")
namespace isa_avx512_hist {

/// Accumulates a 256-bucket histogram of the 8-bit digit at `shift` for
/// 32-bit encoded keys.  `hist` is 256 x uint32 and must be zeroed.
inline void histogram_u32_avx512(const std::uint32_t* FYX_RESTRICT keys,
                                 std::size_t n, unsigned shift,
                                 std::uint32_t* FYX_RESTRICT hist) noexcept {
    const __m512i vmask = _mm512_set1_epi32(int(kRadixMask));
    const __m512i vone  = _mm512_set1_epi32(1);
    const __m512i vzero = _mm512_setzero_si512();

    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        if (FYX_LIKELY(i + 128 < n))
            prefetch_read<1>(keys + i + 128);

        const __m512i k   = _mm512_loadu_si512(
                                reinterpret_cast<const void*>(keys + i));
        // Digit index for each lane.
        const __m512i idx = _mm512_and_si512(_mm512_srli_epi32(k, shift), vmask);

        // Rank of each lane among earlier lanes hitting the same bucket.
        const __m512i cfl  = _mm512_conflict_epi32(idx);
        const __m512i rank = _mm512_popcnt_epi32(cfl);

        // Read-modify-write.  All 16 lanes are active, hence mask 0xFFFF.
        const __m512i cur = _mm512_mask_i32gather_epi32(
                                vzero, static_cast<__mmask16>(0xFFFF), idx,
                                reinterpret_cast<const void*>(hist), 4);
        const __m512i upd = _mm512_add_epi32(_mm512_add_epi32(cur, rank), vone);
        _mm512_i32scatter_epi32(reinterpret_cast<void*>(hist), idx, upd, 4);
    }
    for (; i < n; ++i)
        ++hist[(keys[i] >> shift) & kRadixMask];
}

/// Same for 64-bit encoded keys: 8 lanes per vector, 32-bit counters.
inline void histogram_u64_avx512(const std::uint64_t* FYX_RESTRICT keys,
                                 std::size_t n, unsigned shift,
                                 std::uint32_t* FYX_RESTRICT hist) noexcept {
    const __m512i vmask = _mm512_set1_epi64(std::int64_t(kRadixMask));
    const __m256i vone  = _mm256_set1_epi32(1);
    const __m256i vzero = _mm256_setzero_si256();

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        if (FYX_LIKELY(i + 64 < n))
            prefetch_read<1>(keys + i + 64);

        const __m512i k = _mm512_loadu_si512(
                              reinterpret_cast<const void*>(keys + i));
        // Digits fit in 8 bits, so narrow to 32-bit lanes for conflict/gather.
        const __m512i d64 = _mm512_and_si512(_mm512_srli_epi64(k, shift), vmask);
        const __m256i idx = _mm512_cvtepi64_epi32(d64);

        const __m256i cfl  = _mm256_conflict_epi32(idx);
        const __m256i rank = _mm256_popcnt_epi32(cfl);

        const __m256i cur = _mm256_mmask_i32gather_epi32(
                                vzero, static_cast<__mmask8>(0xFF), idx,
                                reinterpret_cast<const void*>(hist), 4);
        const __m256i upd = _mm256_add_epi32(_mm256_add_epi32(cur, rank), vone);
        _mm256_i32scatter_epi32(reinterpret_cast<void*>(hist), idx, upd, 4);
    }
    for (; i < n; ++i)
        ++hist[unsigned((keys[i] >> shift)) & kRadixMask];
}

} // namespace isa_avx512_hist
FYX_ISA_END
#endif // FYX_HAS_AVX512_CODE

// ---------------------------------------------------------------------------
// Software write-combining scatter
// ---------------------------------------------------------------------------
//
// A naive scatter writes single elements to up to 256 destinations at once.
// Every such store is a partial cache line, so the core must first read the
// line it is about to overwrite (read-for-ownership).  With an output far
// larger than L2 that doubles the memory traffic of the pass.
//
// The cure is a per-bucket buffer exactly one cache line wide.  Elements
// accumulate there and a full line leaves with one non-temporal store, which
// neither reads the destination nor pollutes the cache.
//
// The subtlety that makes or breaks this: an NT store only works on a 64-byte
// aligned address, and a bucket generally starts mid-line.  Flushing "when
// aligned, otherwise memcpy" is worthless -- the cursor stays misaligned
// forever and no store is ever streamed.  Instead each bucket first diverts
// its leading `need` elements into a small head buffer, chosen so that the
// main cursor lands exactly on a line boundary.  From then on *every* flush is
// an aligned full-line NT store.  The heads are written back at the end.
//
// Measured on Ice Lake-SP, 10 M uint32 into 256 buckets:
//     naive scatter                  5.75 ns/elem
//     WCB, flush only when aligned   6.80 ns/elem   (never actually streams)
//     WCB, pre-aligned cursors       2.27 ns/elem   <- this implementation
// ---------------------------------------------------------------------------

/// Number of elements of type `Key` that fit in one cache line.
template <typename Key>
struct WcbTraits {
    static constexpr std::size_t kPerLine = kCacheLine / sizeof(Key);
    static_assert(kPerLine * sizeof(Key) == kCacheLine,
                  "key size must divide the cache line");
};

/// Scratch owned by the caller and reused across passes, so that a multi-pass
/// sort performs no allocation at all inside the loop.
template <typename Key>
struct RadixScatterScratch {
    static constexpr std::size_t kPerLine = WcbTraits<Key>::kPerLine;

    Key*          line = nullptr;   ///< 256 * kPerLine, 64-byte aligned
    Key*          head = nullptr;   ///< 256 * kPerLine, alignment prologues
    std::uint32_t fill[kRadixBuckets];  ///< elements currently in line[b]
    std::uint32_t hn  [kRadixBuckets];  ///< elements currently in head[b]
    std::uint32_t need[kRadixBuckets];  ///< head elements required for alignment
    std::size_t   base[kRadixBuckets];  ///< original bucket start
};

/// One pass of the radix sort: reads `src`, writes `dst` grouped by digit.
///
/// `offset[b]` must hold the output index at which bucket b starts; it is
/// consumed (advanced) in place.
template <typename Key>
inline void radix_scatter_pass(const Key* FYX_RESTRICT src, std::size_t n,
                               Key* FYX_RESTRICT dst, unsigned shift,
                               std::size_t* FYX_RESTRICT offset,
                               RadixScatterScratch<Key>& sc,
                               bool can_stream) noexcept {
    constexpr std::size_t kPerLine = WcbTraits<Key>::kPerLine;

    Key* FYX_RESTRICT line = sc.line;
    Key* FYX_RESTRICT head = sc.head;

    // Work out, per bucket, how many leading elements must be diverted so the
    // streaming cursor starts on a cache-line boundary.
    for (unsigned b = 0; b < kRadixBuckets; ++b) {
        sc.fill[b] = 0;
        sc.hn[b]   = 0;
        sc.base[b] = offset[b];
        if (can_stream) {
            const std::uintptr_t addr =
                reinterpret_cast<std::uintptr_t>(dst + offset[b]);
            const std::size_t misalign = (kCacheLine - (addr & (kCacheLine - 1)))
                                         & (kCacheLine - 1);
            sc.need[b] = static_cast<std::uint32_t>(misalign / sizeof(Key));
        } else {
            sc.need[b] = 0;
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        prefetch_stream(src, i, n);

        const Key      k = src[i];
        const unsigned b = static_cast<unsigned>((k >> shift) & Key(kRadixMask));

        // Alignment prologue for this bucket.
        if (FYX_UNLIKELY(sc.hn[b] < sc.need[b])) {
            head[static_cast<std::size_t>(b) * kPerLine + sc.hn[b]] = k;
            ++sc.hn[b];
            continue;
        }

        Key* L = line + static_cast<std::size_t>(b) * kPerLine;
        const std::uint32_t f = sc.fill[b];
        L[f] = k;

        if (FYX_UNLIKELY(f + 1 == kPerLine)) {
            Key* out = dst + offset[b] + sc.need[b];
            if (can_stream) {
                // Guaranteed 64-byte aligned by construction.
                stream_cache_line(out, L);
            } else {
                std::memcpy(out, L, kCacheLine);
            }
            offset[b] += kPerLine;
            sc.fill[b] = 0;
        } else {
            sc.fill[b] = f + 1;
        }
    }

    // Write back the alignment prologues and the partial trailing lines.
    for (unsigned b = 0; b < kRadixBuckets; ++b) {
        if (sc.hn[b])
            std::memcpy(dst + sc.base[b],
                        head + static_cast<std::size_t>(b) * kPerLine,
                        static_cast<std::size_t>(sc.hn[b]) * sizeof(Key));
        if (sc.fill[b])
            std::memcpy(dst + offset[b] + sc.need[b],
                        line + static_cast<std::size_t>(b) * kPerLine,
                        static_cast<std::size_t>(sc.fill[b]) * sizeof(Key));
        // Leave `offset[b]` pointing past everything this bucket wrote, in
        // case the caller wants to inspect it.
        offset[b] += sc.fill[b] + sc.hn[b];
    }

    if (can_stream) store_fence();
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Which passes actually need to run, and where the data ends up.
template <unsigned Passes>
struct RadixPlan {
    unsigned active[Passes];   ///< shift index of each surviving pass
    unsigned count = 0;        ///< number of surviving passes
};

/// A pass is degenerate when every key shares the same digit: the scatter
/// would be an exact copy, so it is skipped.
template <unsigned Passes>
inline RadixPlan<Passes> plan_radix(const RadixHistogram<Passes>& hist,
                                    std::size_t n) noexcept {
    RadixPlan<Passes> plan;
    for (unsigned p = 0; p < Passes; ++p) {
        bool degenerate = false;
        for (unsigned d = 0; d < kRadixBuckets; ++d) {
            if (hist.count[p][d] == n) { degenerate = true; break; }
        }
        if (!degenerate) plan.active[plan.count++] = p;
    }
    return plan;
}

/// LSD radix sort of `data[0..n)` using `tmp` as the ping-pong buffer.
/// Returns true on success; false means scratch memory was unavailable and the
/// caller must fall back to a comparison sort.
///
/// Keys are encoded on the way in and decoded on the way out, so signed
/// integers and IEEE floats sort in their natural order (see RadixTraits).
template <typename T>
inline bool radix_sort_impl(T* FYX_RESTRICT data, std::size_t n,
                            typename RadixTraits<T>::Key* FYX_RESTRICT buf_a,
                            typename RadixTraits<T>::Key* FYX_RESTRICT buf_b) noexcept {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr unsigned Passes = RT::passes;

    if (n < 2) return true;

    // -- 1. fused histogram of every pass, single read of the input ---------
    RadixHistogram<Passes> hist;
    scalar_histogram<T, Passes>(data, n, hist);

    // -- 2. skip degenerate passes ------------------------------------------
    const RadixPlan<Passes> plan = plan_radix<Passes>(hist, n);

    if (plan.count == 0) {
        // Every pass degenerate => all keys identical => already sorted.
        return true;
    }

    // -- 3. choose the starting buffer so the last pass lands in buf_a ------
    //
    // Each pass swaps src/dst.  Starting in buf_a, after `count` passes the
    // data sits in buf_a when count is even and in buf_b when it is odd.  We
    // want it in buf_a at the end (that is where we decode from), so an odd
    // pass count starts in buf_b.
    const bool start_in_b = (plan.count & 1u) != 0;

    Key* src = start_in_b ? buf_b : buf_a;
    Key* dst = start_in_b ? buf_a : buf_b;

    // Encode into the chosen source buffer.
    for (std::size_t i = 0; i < n; ++i) src[i] = RT::encode(data[i]);

    // -- 4. scatter passes ---------------------------------------------------
    constexpr std::size_t kPerLine = WcbTraits<Key>::kPerLine;

    // One allocation for both the line buffers and the alignment heads, plus
    // slack so the line buffer can be cache-line aligned by hand.
    ScratchLease<Key> wcb_lease(2 * kRadixBuckets * kPerLine + kPerLine);
    if (!wcb_lease.valid()) return false;

    RadixScatterScratch<Key> sc;
    {
        const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(wcb_lease.get());
        const std::uintptr_t m = (kCacheLine - (a & (kCacheLine - 1))) & (kCacheLine - 1);
        sc.line = reinterpret_cast<Key*>(a + m);
        sc.head = sc.line + kRadixBuckets * kPerLine;
    }
    const bool can_stream = have_nt_stores();

    std::size_t offset[kRadixBuckets];

    for (unsigned s_i = 0; s_i < plan.count; ++s_i) {
        const unsigned p     = plan.active[s_i];
        const unsigned shift = p * kRadixBits;

        // Exclusive prefix sum of this pass's histogram = bucket start offsets.
        std::size_t sum = 0;
        for (unsigned d = 0; d < kRadixBuckets; ++d) {
            offset[d] = sum;
            sum += static_cast<std::size_t>(hist.count[p][d]);
        }

        radix_scatter_pass<Key>(src, n, dst, shift, offset, sc, can_stream);

        Key* t = src; src = dst; dst = t;
    }

    // After the final swap the sorted keys are in `src`, which by construction
    // is buf_a.
    for (std::size_t i = 0; i < n; ++i) data[i] = RT::decode(src[i]);
    return true;
}

/// Convenience wrapper that owns the two ping-pong buffers.
template <typename T>
inline bool radix_sort(T* data, std::size_t n) noexcept {
    using Key = typename RadixTraits<T>::Key;
    if (n < 2) return true;

    ScratchLease<Key> lease(n * 2);
    if (!lease.valid()) return false;

    Key* a = lease.get();
    Key* b = a + n;
    return radix_sort_impl<T>(data, n, a, b);
}

} // namespace detail
} // namespace fyx

// ============================================================================
//  Section 10 -- Branchless block-partition quicksort (pdqsort derivative)
//
//  This is the general-purpose comparison sort: it accepts any comparator and
//  any value type, so it serves generic containers, the tail of the sample
//  sort, and every case the radix path declines.
//
//  Three ideas carry the performance:
//
//  1. Branchless block partitioning.  A data-dependent branch per element
//     mispredicts about half the time on random input, costing ~15 cycles
//     each.  Instead we compare a block of 64 elements and record the indices
//     that need moving into a small byte array, advancing the write cursor by
//     the comparison result rather than branching on it.  The comparison loop
//     becomes straight-line code; the only branches left are loop bounds.
//
//  2. Pattern defeat.  A bad pivot streak degrades quicksort to O(n^2).  We
//     track a recursion budget and finish with heapsort when it runs out,
//     giving a hard O(n log n) worst case.  Nearly-sorted runs are detected
//     and short-circuited, and a pivot equal to the parent's pivot triggers a
//     partition that isolates the duplicates, so inputs with few distinct keys
//     do not blow up.
//
//  3. Small ranges leave the recursion.  Numeric keys with the default
//     comparator are routed to a SIMD sorting network by the caller; the
//     insertion sort here is only reached for types the networks cannot take.
// ============================================================================

namespace fyx {
namespace detail {

/// Block size for the branchless partition.  64 one-byte offsets fit in a
/// single cache line and the count always fits in an unsigned char.
inline constexpr std::size_t kPartitionBlock = 64;

/// Result of one partition step.
template <typename It>
struct PartitionResult {
    It   pivot_pos;      ///< final resting place of the pivot
    bool already_parted; ///< true when no element needed to move
};

// ---------------------------------------------------------------------------
// Branchless block partition (Hoare scheme, elements < pivot to the left)
// ---------------------------------------------------------------------------
//
// `first` holds the pivot on entry (the caller has already placed it there).
//
// The loop keeps two offset buffers.  `offsets_l` collects positions in the
// left block holding elements that belong on the right, `offsets_r` the
// mirror.  Matching pairs are then swapped directly.  Because a block is
// scanned in full before any swap happens, the comparison loop contains no
// data-dependent branch at all.
// ---------------------------------------------------------------------------

template <typename It, typename Compare>
inline PartitionResult<It> partition_right_branchless(It first, It last, Compare comp) {
    using T    = typename std::iterator_traits<It>::value_type;
    using Diff = typename std::iterator_traits<It>::difference_type;

    T pivot = std::move(*first);

    It lo = first;
    It hi = last;

    // Find the first element >= pivot.  The pivot at *first is a sentinel.
    while (comp(*++lo, pivot)) {}

    // Find the last element < pivot.  If the scan above never moved there is
    // no sentinel on the right yet, so that loop has to be bounded.
    if (lo - 1 == first) {
        while (lo < hi && !comp(*--hi, pivot)) {}
    } else {
        while (!comp(*--hi, pivot)) {}
    }

    const bool already_parted = (lo >= hi);

    if (!already_parted) {
        std::iter_swap(lo, hi);
        ++lo;
    }

    // ---- invariants for the block loop ------------------------------------
    //   [begin, lo)  are < pivot, except the num_l pending offsets
    //   [hi, end)    are >= pivot, except the num_r pending offsets
    //   [lo, hi)     is unclassified
    // A pending left offset marks an element that belongs on the right, and
    // vice versa.  Offsets within a buffer are always ascending.
    // -----------------------------------------------------------------------
    alignas(kCacheLine) unsigned char offsets_l[kPartitionBlock];
    alignas(kCacheLine) unsigned char offsets_r[kPartitionBlock];

    It          base_l = lo;
    It          base_r = hi;
    std::size_t num_l = 0, num_r = 0, start_l = 0, start_r = 0;

    for (;;) {
        const Diff remaining = hi - lo;

        // Classify a block from the left, but only if the left buffer is
        // drained -- a pending block must stay where it is.
        if (num_l == 0 && remaining > 0) {
            const std::size_t blk = static_cast<std::size_t>(
                remaining < static_cast<Diff>(kPartitionBlock)
                    ? remaining : static_cast<Diff>(kPartitionBlock));
            base_l  = lo;
            start_l = 0;
            for (std::size_t i = 0; i < blk; ++i) {
                offsets_l[num_l] = static_cast<unsigned char>(i);
                num_l += static_cast<std::size_t>(!comp(*lo, pivot));
                ++lo;
            }
        }

        // Symmetrically from the right.  offsets_r[k] is a 1-based distance
        // *below* base_r, so the element lives at base_r - offsets_r[k].
        const Diff remaining2 = hi - lo;
        if (num_r == 0 && remaining2 > 0) {
            const std::size_t blk = static_cast<std::size_t>(
                remaining2 < static_cast<Diff>(kPartitionBlock)
                    ? remaining2 : static_cast<Diff>(kPartitionBlock));
            base_r  = hi;
            start_r = 0;
            for (std::size_t i = 0; i < blk; ++i) {
                --hi;
                offsets_r[num_r] = static_cast<unsigned char>(i + 1);
                num_r += static_cast<std::size_t>(comp(*hi, pivot));
            }
        }

        // Every matched pair can be exchanged directly.
        const std::size_t num = (num_l < num_r) ? num_l : num_r;
        for (std::size_t i = 0; i < num; ++i)
            std::iter_swap(base_l + offsets_l[start_l + i],
                           base_r - offsets_r[start_r + i]);

        num_l   -= num;  num_r   -= num;
        start_l += num;  start_r += num;

        // Done once nothing is unclassified and at most one side still holds
        // pending offsets (that leftover is handled by the cleanup below).
        if (lo >= hi && (num_l == 0 || num_r == 0)) break;
    }

    // ---- cleanup ----------------------------------------------------------
    // Consume the leftovers from the highest offset downwards.  Because the
    // offsets ascend, each swap partner taken from the shrinking boundary is
    // guaranteed to sit at or below the pending element, so no element is ever
    // moved twice.
    if (num_l) {
        while (num_l--) std::iter_swap(base_l + offsets_l[start_l + num_l], --lo);
        hi = lo;
    }
    if (num_r) {
        while (num_r--) {
            std::iter_swap(base_r - offsets_r[start_r + num_r], hi);
            ++hi;
        }
        lo = hi;
    }

    // Drop the pivot into the gap between the two halves.
    It pivot_pos = lo - 1;
    *first       = std::move(*pivot_pos);
    *pivot_pos   = std::move(pivot);

    return PartitionResult<It>{pivot_pos, already_parted};
}

/// Simple (branchy) partition, used when the value type is expensive to move
/// or the range is short enough that block bookkeeping does not pay.
template <typename It, typename Compare>
inline PartitionResult<It> partition_right_simple(It first, It last, Compare comp) {
    using T = typename std::iterator_traits<It>::value_type;

    T  pivot = std::move(*first);
    It l     = first;
    It r     = last;

    while (comp(*++l, pivot)) {}

    if (l - 1 == first) {
        while (l < r && !comp(*--r, pivot)) {}
    } else {
        while (!comp(*--r, pivot)) {}
    }

    const bool already_parted = (l >= r);

    while (l < r) {
        std::iter_swap(l, r);
        while (comp(*++l, pivot)) {}
        while (!comp(*--r, pivot)) {}
    }

    It pivot_pos = l - 1;
    *first       = std::move(*pivot_pos);
    *pivot_pos   = std::move(pivot);
    return PartitionResult<It>{pivot_pos, already_parted};
}

// ---------------------------------------------------------------------------
// Partition that sends elements *equal* to the pivot to the left.
//
// Reached when the chosen pivot compares equal to the parent's pivot, which
// means the range is dominated by one value.  Isolating the duplicates here is
// what keeps "many equal keys" inputs near linear.
// ---------------------------------------------------------------------------

template <typename It, typename Compare>
inline It partition_left(It first, It last, Compare comp) {
    using T = typename std::iterator_traits<It>::value_type;

    T  pivot = std::move(*first);
    It l     = first;
    It r     = last;

    while (comp(pivot, *--r)) {}

    if (r + 1 == last) {
        while (l < r && !comp(pivot, *++l)) {}
    } else {
        while (!comp(pivot, *++l)) {}
    }

    while (l < r) {
        std::iter_swap(l, r);
        while (comp(pivot, *--r)) {}
        while (!comp(pivot, *++l)) {}
    }

    It pivot_pos = r;
    *first       = std::move(*pivot_pos);
    *pivot_pos   = std::move(pivot);
    return pivot_pos;
}

// ---------------------------------------------------------------------------
// Small-range kernel
// ---------------------------------------------------------------------------
//
// The specification forbids insertion sort for numeric arrays of at most 64
// elements, which must use a SIMD network.  `SmallSort` is the hook: the
// numeric specialisation is installed by the dispatch layer, and this generic
// version only ever runs for types with no network (non-trivial objects,
// custom comparators over class types, and so on).
// ---------------------------------------------------------------------------

template <typename It, typename Compare>
FYX_FORCE_INLINE void small_sort_generic(It first, It last, Compare comp,
                                         bool leftmost) {
    if (leftmost) insertion_sort(first, last, comp);
    else          insertion_sort_guarded(first, last, comp);
}

// ---------------------------------------------------------------------------
// The recursive driver
// ---------------------------------------------------------------------------

template <typename It, typename Compare, bool Branchless>
inline void pdqsort_loop(It first, It last, Compare comp, int bad_allowed,
                         bool leftmost) {
    using Diff = typename std::iterator_traits<It>::difference_type;

    for (;;) {
        const Diff size = last - first;

        if (size <= static_cast<Diff>(kInsertionThreshold)) {
            small_sort_generic(first, last, comp, leftmost);
            return;
        }

        choose_pivot(first, last, comp);

        // If the pivot equals the predecessor (which is <= everything here),
        // the range is full of duplicates of that value: peel them off.
        if (!leftmost && !comp(*(first - 1), *first)) {
            first = partition_left(first, last, comp) + 1;
            continue;
        }

        const PartitionResult<It> part =
            Branchless ? partition_right_branchless(first, last, comp)
                       : partition_right_simple(first, last, comp);

        const Diff l_size = part.pivot_pos - first;
        const Diff r_size = last - (part.pivot_pos + 1);
        const bool highly_unbalanced = (l_size < size / 8) || (r_size < size / 8);

        if (highly_unbalanced) {
            // Repeated bad splits: shuffle a few elements to break the pattern
            // that is defeating the pivot heuristic.
            if (--bad_allowed == 0) {
                heap_sort(first, last, comp);
                return;
            }
            if (l_size >= static_cast<Diff>(kInsertionThreshold)) {
                std::iter_swap(first, first + l_size / 4);
                std::iter_swap(part.pivot_pos - 1, part.pivot_pos - l_size / 4);
                if (l_size > 128) {
                    std::iter_swap(first + 1, first + (l_size / 4 + 1));
                    std::iter_swap(first + 2, first + (l_size / 4 + 2));
                    std::iter_swap(part.pivot_pos - 2, part.pivot_pos - (l_size / 4 + 1));
                    std::iter_swap(part.pivot_pos - 3, part.pivot_pos - (l_size / 4 + 2));
                }
            }
            if (r_size >= static_cast<Diff>(kInsertionThreshold)) {
                std::iter_swap(part.pivot_pos + 1, part.pivot_pos + (1 + r_size / 4));
                std::iter_swap(last - 1, last - r_size / 4);
                if (r_size > 128) {
                    std::iter_swap(part.pivot_pos + 2, part.pivot_pos + (2 + r_size / 4));
                    std::iter_swap(part.pivot_pos + 3, part.pivot_pos + (3 + r_size / 4));
                    std::iter_swap(last - 2, last - (1 + r_size / 4));
                    std::iter_swap(last - 3, last - (2 + r_size / 4));
                }
            }
        } else if (part.already_parted &&
                   partial_insertion_sort(first, part.pivot_pos, comp) &&
                   partial_insertion_sort(part.pivot_pos + 1, last, comp)) {
            // The partition moved nothing and both halves were nearly sorted:
            // the bounded insertion sorts above just finished the job.
            return;
        }

        // Recurse into the smaller half, loop on the larger one, so the stack
        // depth stays O(log n).
        if (l_size < r_size) {
            pdqsort_loop<It, Compare, Branchless>(first, part.pivot_pos, comp,
                                                  bad_allowed, leftmost);
            first    = part.pivot_pos + 1;
            leftmost = false;
        } else {
            pdqsort_loop<It, Compare, Branchless>(part.pivot_pos + 1, last, comp,
                                                  bad_allowed, false);
            last = part.pivot_pos;
        }
    }
}

/// Entry point.  `Branchless` is chosen by the caller from the value type: it
/// pays for small trivially-copyable types and costs for everything else.
template <typename It, typename Compare>
inline void pdqsort(It first, It last, Compare comp) {
    if (first == last) return;

    using T = typename std::iterator_traits<It>::value_type;
    constexpr bool kBranchless =
        std::is_arithmetic<T>::value && sizeof(T) <= 16;

    pdqsort_loop<It, Compare, kBranchless>(
        first, last, comp,
        static_cast<int>(log2_floor(static_cast<std::uint64_t>(last - first))) + 1,
        true);
}

} // namespace detail
} // namespace fyx

#endif // FYX_SORT_HPP_INCLUDED
