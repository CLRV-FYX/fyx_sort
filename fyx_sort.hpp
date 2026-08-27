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

#ifndef FYX_ENABLE_TEST_HOOKS
#  define FYX_ENABLE_TEST_HOOKS 0
#endif

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
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
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

/// Sample-sort bucket count.  The serial path uses a single prefix scatter;
/// the parallel path derives chunk blocks from kParallelThreshold.
inline constexpr std::size_t kSampleBuckets   = 256;
inline constexpr std::size_t kSampleBlock     = 1024;
inline constexpr std::size_t kSampleThreshold = 1u << 15;     // 32768

/// Hard ceiling on pool size.  Guards against absurd hardware_concurrency
/// values and bounds the per-thread scratch the pool can pin.
inline constexpr unsigned kMaxThreads = 256;

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

// ============================================================================
//  Section 11 -- Parallel execution engine
//
//  A lazily-initialised thread pool over Chase-Lev work-stealing deques.
//
//  Why work stealing.  Sorting produces a highly irregular task tree: a
//  partition can split 50/50 or 1/99, and the depth varies per branch.  A
//  static split would leave most threads idle.  With work stealing each worker
//  owns a deque, pushes and pops its own tasks from the bottom (LIFO, which is
//  cache-friendly because the most recently produced task is still hot), and
//  when it runs dry it steals from the *top* of a random victim (FIFO, which
//  takes the oldest and therefore largest task, minimising steal frequency).
//
//  The Chase-Lev algorithm
//  -----------------------
//  Single-producer (the owner) / multi-consumer (the thieves), lock free.
//  Correctness rests entirely on the memory ordering, so it is spelled out:
//
//    push (owner only)
//        b = bottom.load(relaxed)          // only the owner writes bottom
//        t = top.load(acquire)             // must see thieves' top updates
//        [grow if b - t >= capacity]
//        buffer[b % cap] = task            // relaxed atomic slot stores
//        bottom.store(b + 1, release)      // publishes the task
//
//    pop (owner only)
//        b = bottom.load(relaxed) - 1
//        bottom.store(b, relaxed)
//        atomic_thread_fence(seq_cst)      // orders bottom vs top, both ways
//        t = top.load(relaxed)
//        if t <= b:
//            task = buffer[b % cap]
//            if t == b:                    // last element: race with thieves
//                if !top.compare_exchange_strong(t, t + 1, seq_cst, relaxed)
//                    task = none           // a thief won
//                bottom.store(b + 1, relaxed)
//            return task
//        else:
//            bottom.store(b + 1, relaxed)  // empty; restore
//            return none
//
//    steal (thieves)
//        t = top.load(acquire)
//        atomic_thread_fence(seq_cst)      // t must be read before b
//        b = bottom.load(acquire)
//        if t < b:
//            task = buffer[t % cap]        // speculative read
//            if !top.compare_exchange_strong(t, t + 1, seq_cst, relaxed)
//                return abort              // lost the race, task is garbage
//            return task
//        return empty
//
//  The seq_cst fences in pop and steal are what prevent the owner and a thief
//  from both taking the final element: they force a total order between the
//  bottom store and the top load on each side.
//
//  Buffer growth.  The array is replaced, never freed while readers may still
//  be inside it -- a thief can hold a pointer to the old buffer.  Retired
//  buffers are therefore kept on a list owned by the deque and released only
//  when the pool shuts down.  Sorting task counts are bounded and small, so
//  this leaks nothing in practice.
// ============================================================================

namespace fyx {
namespace detail {

#if FYX_ENABLE_PARALLEL

// ---------------------------------------------------------------------------
// A unit of work
// ---------------------------------------------------------------------------
//
// Type-erased through a plain function pointer plus an argument, rather than
// std::function: no allocation, trivially copyable, and it fits in 16 bytes so
// the deque slots stay small.
// ---------------------------------------------------------------------------

struct Task {
    void (*fn)(void*) = nullptr;
    void* arg         = nullptr;

    FYX_FORCE_INLINE bool valid() const noexcept { return fn != nullptr; }
    FYX_FORCE_INLINE void run() const { fn(arg); }
};

/// Outcome of a steal attempt: `Abort` means "lost a race, try again", which
/// is different from "the victim had nothing".
enum class StealStatus { Success, Empty, Abort };

// ---------------------------------------------------------------------------
// Ring buffer for the deque
// ---------------------------------------------------------------------------

// A slot is two pointer-sized atomics rather than one std::atomic<Task>,
// because a 16-byte atomic is *not* lock free on the mainstream ABIs (checked:
// std::atomic<Task>::is_always_lock_free == false, so it would silently take a
// mutex and defeat the whole point of the deque).  Two 8-byte atomics are
// always lock free.
//
// All slot accesses are `relaxed`: they carry no ordering themselves, the
// deque's fences and the CAS on top_ provide it.  Relaxed atomics compile to
// exactly the same plain load/store instructions as raw memory, so this costs
// nothing at run time -- it only makes the race well defined for the compiler
// and for race detectors.
//
// A thief's speculative read may tear (an `fn` from one task, an `arg` from
// another) if the owner overwrites the slot mid-read.  That is harmless: a torn
// read can only happen when the owner has wrapped around and reused the slot,
// which means top_ has moved, which means the thief's CAS fails and the value
// is thrown away.  Conversely, if the CAS succeeds the slot provably could not
// have been rewritten -- push() must grow (into a *different* buffer) before it
// can reach an index that aliases a slot the thief still owns -- so an accepted
// task is never torn.
struct AtomicSlot {
    std::atomic<void (*)(void*)> fn;
    std::atomic<void*>           arg;
};

class WsRingBuffer {
public:
    explicit WsRingBuffer(std::int64_t log_size)
        : log_size_(log_size),
          mask_((std::int64_t(1) << log_size) - 1),
          data_(static_cast<AtomicSlot*>(std::malloc(
              sizeof(AtomicSlot) *
              static_cast<std::size_t>(std::int64_t(1) << log_size)))) {
        // std::malloc gives raw storage; the atomics must be constructed.
        if (data_) {
            const std::int64_t n = mask_ + 1;
            for (std::int64_t i = 0; i < n; ++i) new (&data_[i]) AtomicSlot();
        }
    }

    ~WsRingBuffer() {
        if (data_) {
            const std::int64_t n = mask_ + 1;
            for (std::int64_t i = 0; i < n; ++i) data_[i].~AtomicSlot();
            std::free(data_);
        }
    }

    WsRingBuffer(const WsRingBuffer&)            = delete;
    WsRingBuffer& operator=(const WsRingBuffer&) = delete;

    bool         valid()    const noexcept { return data_ != nullptr; }
    std::int64_t capacity() const noexcept { return mask_ + 1; }
    std::int64_t log_size() const noexcept { return log_size_; }

    void put(std::int64_t i, Task v) noexcept {
        AtomicSlot& s = data_[i & mask_];
        s.fn.store(v.fn, std::memory_order_relaxed);
        s.arg.store(v.arg, std::memory_order_relaxed);
    }

    Task get(std::int64_t i) const noexcept {
        const AtomicSlot& s = data_[i & mask_];
        Task t;
        t.fn  = s.fn.load(std::memory_order_relaxed);
        t.arg = s.arg.load(std::memory_order_relaxed);
        return t;
    }

private:
    std::int64_t log_size_;
    std::int64_t mask_;
    AtomicSlot*  data_;
};

// ---------------------------------------------------------------------------
// Chase-Lev deque
// ---------------------------------------------------------------------------

class WorkStealingDeque {
public:
    explicit WorkStealingDeque(std::int64_t log_size = 10)
        : top_(0), bottom_(0), buffer_(nullptr) {
        WsRingBuffer* b = new (std::nothrow) WsRingBuffer(log_size);
        if (b && !b->valid()) { delete b; b = nullptr; }
        buffer_.store(b, std::memory_order_relaxed);
    }

    ~WorkStealingDeque() {
        delete buffer_.load(std::memory_order_relaxed);
        for (WsRingBuffer* r : retired_) delete r;
    }

    WorkStealingDeque(const WorkStealingDeque&)            = delete;
    WorkStealingDeque& operator=(const WorkStealingDeque&) = delete;

    bool valid() const noexcept {
        return buffer_.load(std::memory_order_relaxed) != nullptr;
    }

    /// Owner only.  Returns false if the deque could not grow.
    bool push(Task t) {
        const std::int64_t b   = bottom_.load(std::memory_order_relaxed);
        const std::int64_t top = top_.load(std::memory_order_acquire);

        WsRingBuffer* buf = buffer_.load(std::memory_order_relaxed);
        if (!buf) return false;

        if (b - top >= buf->capacity() - 1) {
            WsRingBuffer* grown =
                new (std::nothrow) WsRingBuffer(buf->log_size() + 1);
            if (!grown || !grown->valid()) { delete grown; return false; }
            for (std::int64_t i = top; i < b; ++i) grown->put(i, buf->get(i));
            // The old buffer may still be read by an in-flight thief, so it is
            // retired rather than deleted.
            retired_.push_back(buf);
            buffer_.store(grown, std::memory_order_release);
            buf = grown;
        }

        buf->put(b, t);
        // Release-store: the slot write above must be visible to any thief that
        // observes this new bottom.  A release store on bottom_ (rather than a
        // standalone release fence plus a relaxed store) pairs directly with
        // the acquire load of bottom_ in steal(), which is both the canonical
        // formulation and the one race detectors can actually see -- TSan does
        // not model std::atomic_thread_fence.  On x86 both compile to a plain
        // mov, so this is free.
        bottom_.store(b + 1, std::memory_order_release);
        return true;
    }

    /// Owner only.  Takes from the bottom (LIFO).
    bool pop(Task& out) {
        const std::int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        WsRingBuffer* buf = buffer_.load(std::memory_order_relaxed);
        if (!buf) return false;

        bottom_.store(b, std::memory_order_relaxed);
        // Full fence: the bottom store must not be reordered past the top load,
        // otherwise the owner and a thief can both claim the last task.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::int64_t t = top_.load(std::memory_order_relaxed);

        if (t <= b) {
            Task task = buf->get(b);
            if (t != b) { out = task; return true; }

            // Exactly one element: contend with the thieves for it.
            bool won = top_.compare_exchange_strong(t, t + 1,
                                                    std::memory_order_seq_cst,
                                                    std::memory_order_relaxed);
            bottom_.store(b + 1, std::memory_order_relaxed);
            if (won) { out = task; return true; }
            return false;
        }

        // Empty: undo the decrement.
        bottom_.store(b + 1, std::memory_order_relaxed);
        return false;
    }

    /// Any thread.  Takes from the top (FIFO): the oldest, biggest task.
    StealStatus steal(Task& out) {
        std::int64_t t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const std::int64_t b = bottom_.load(std::memory_order_acquire);

        if (t < b) {
            // Acquire, pairing with the release store in push()'s grow path,
            // so the copied-over slots are visible.  (memory_order_consume is
            // deprecated and every compiler promotes it to acquire anyway.)
            WsRingBuffer* buf = buffer_.load(std::memory_order_acquire);
            if (!buf) return StealStatus::Empty;

            // Speculative: only valid if the CAS below succeeds.
            Task task = buf->get(t);
            if (!top_.compare_exchange_strong(t, t + 1,
                                              std::memory_order_seq_cst,
                                              std::memory_order_relaxed))
                return StealStatus::Abort;

            out = task;
            return StealStatus::Success;
        }
        return StealStatus::Empty;
    }

    bool empty() const noexcept {
        return bottom_.load(std::memory_order_relaxed) <=
               top_.load(std::memory_order_relaxed);
    }

private:
    // top_ and bottom_ are hammered by different threads; keeping them on
    // separate cache lines removes the false sharing that would otherwise
    // dominate the steal path.
    alignas(kCacheLine) std::atomic<std::int64_t> top_;
    alignas(kCacheLine) std::atomic<std::int64_t> bottom_;
    alignas(kCacheLine) std::atomic<WsRingBuffer*> buffer_;

    std::vector<WsRingBuffer*> retired_;  ///< owner-only, freed at exit
};


// ---------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------
//
// Lazily created: a program that never sorts in parallel never spawns a
// thread.  Construction happens once, guarded by a function-local static,
// which C++11 guarantees is thread-safe.
//
// Idle policy.  Workers spin over the victim deques for a bounded number of
// rounds (cheap when work arrives promptly, which is the common case mid-sort)
// and only then block on a condition variable.  Spinning forever would burn a
// core per idle worker; blocking immediately would add a futex round trip to
// every task.
//
// Shutdown.  `stop_` is set, all workers are woken, and each is joined.  The
// destructor runs at static destruction time; because every sort call blocks
// until its own tasks are finished, no task can outlive the pool.
// ---------------------------------------------------------------------------

class ThreadPool {
public:
    /// Number of worker threads, excluding the calling thread.
    static unsigned default_threads() noexcept {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc == 0) hc = 1;
        if (hc > kMaxThreads) hc = kMaxThreads;
        return hc;
    }

    explicit ThreadPool(unsigned nthreads)
        : nworkers_(nthreads == 0 ? 1u : nthreads) {
        queues_.reserve(nworkers_);
        for (unsigned i = 0; i < nworkers_; ++i) {
            queues_.emplace_back(new (std::nothrow) WorkStealingDeque(10));
            if (!queues_.back() || !queues_.back()->valid()) { broken_ = true; return; }
        }
        // Worker 0 is the submitting thread; only 1..n-1 get an OS thread.
        threads_.reserve(nworkers_ > 0 ? nworkers_ - 1 : 0);
#if FYX_HAS_EXCEPTIONS
        try {
#endif
            for (unsigned i = 1; i < nworkers_; ++i)
                threads_.emplace_back([this, i] { worker_loop(i); });
#if FYX_HAS_EXCEPTIONS
        } catch (...) {
            // Fewer threads than requested is survivable; run with what we got.
        }
#endif
    }

    ~ThreadPool() {
        stop_.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(sleep_mu_);
            ++wake_epoch_;
        }
        sleep_cv_.notify_all();
        for (std::thread& t : threads_)
            if (t.joinable()) t.join();
        for (WorkStealingDeque* q : queues_) delete q;
    }

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    bool     broken()   const noexcept { return broken_; }
    unsigned nworkers() const noexcept { return nworkers_; }

    /// Index of the calling thread within the pool, or 0 for outsiders.
    static unsigned this_worker() noexcept { return tls_worker_id(); }

    /// Submit a task to the calling thread's queue.  Returns false when the
    /// queue could not grow, in which case the caller must run it inline.
    bool submit(unsigned worker, Task t) {
        if (broken_ || worker >= nworkers_) return false;
        if (!queues_[worker]->push(t)) return false;
        // A sleeping worker will not see the new bottom, so wake the pool.
        if (sleepers_.load(std::memory_order_acquire) != 0) {
            {
                std::lock_guard<std::mutex> lk(sleep_mu_);
                ++wake_epoch_;
            }
            sleep_cv_.notify_all();
        }
        return true;
    }

    /// Run tasks until `pending` reaches zero.  Used by the submitting thread
    /// to participate instead of blocking, which keeps all cores busy and
    /// makes nested parallelism deadlock-free.
    void wait_for(std::atomic<std::size_t>& pending, unsigned worker) {
        while (pending.load(std::memory_order_acquire) != 0) {
            Task t;
            if (try_get_task(worker, t)) t.run();
            else                          cpu_pause();
        }
    }

private:
    static unsigned& tls_worker_id() noexcept {
        static thread_local unsigned id = 0;
        return id;
    }

    /// Pop locally, else steal from a random victim.
    bool try_get_task(unsigned self, Task& out) {
        if (self < nworkers_ && queues_[self]->pop(out)) return true;

        const unsigned n = nworkers_;
        if (n <= 1) return false;

        // xorshift keeps victim selection cheap and unbiased enough.
        unsigned& st = tls_rng_state();
        for (unsigned attempt = 0; attempt < n * 2; ++attempt) {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            const unsigned v = st % n;
            if (v == self) continue;
            const StealStatus s = queues_[v]->steal(out);
            if (s == StealStatus::Success) return true;
            // Abort means a lost race: worth retrying elsewhere immediately.
        }
        return false;
    }

    static unsigned& tls_rng_state() noexcept {
        static thread_local unsigned s = 0x9E3779B9u;
        if (s == 0) s = 0x9E3779B9u;
        return s;
    }

    void worker_loop(unsigned self) {
        tls_worker_id() = self;

        while (!stop_.load(std::memory_order_acquire)) {
            Task t;
            if (try_get_task(self, t)) { t.run(); continue; }

            // Nothing found: spin briefly, then sleep.
            bool got = false;
            for (unsigned spin = 0; spin < kSpinRounds; ++spin) {
                cpu_pause();
                if (try_get_task(self, t)) { got = true; break; }
            }
            if (got) { t.run(); continue; }

            std::unique_lock<std::mutex> lk(sleep_mu_);
            const std::uint64_t epoch = wake_epoch_;
            sleepers_.fetch_add(1, std::memory_order_acq_rel);
            sleep_cv_.wait_for(lk, std::chrono::milliseconds(2), [&] {
                return stop_.load(std::memory_order_acquire) || wake_epoch_ != epoch;
            });
            sleepers_.fetch_sub(1, std::memory_order_acq_rel);
        }
        // Drain whatever is left so no submitted task is dropped.
        Task t;
        while (try_get_task(self, t)) t.run();
    }

    static constexpr unsigned kSpinRounds = 64;

    unsigned                         nworkers_;
    bool                             broken_ = false;
    std::vector<WorkStealingDeque*>  queues_;
    std::vector<std::thread>         threads_;

    std::atomic<bool>        stop_{false};
    std::atomic<unsigned>    sleepers_{0};
    std::mutex               sleep_mu_;
    std::condition_variable  sleep_cv_;
    std::uint64_t            wake_epoch_ = 0;
};

/// The process-wide pool, created on first use.
inline ThreadPool& global_pool() {
    static ThreadPool pool(ThreadPool::default_threads());
    return pool;
}

/// True when the pool is usable for parallel work.
inline bool parallel_available() {
    ThreadPool& p = global_pool();
    return !p.broken() && p.nworkers() > 1;
}

// ---------------------------------------------------------------------------
// Fork-join helper
// ---------------------------------------------------------------------------
//
// Runs `a` and `b` concurrently when a worker is free, otherwise inline.  The
// second half is pushed and the first is executed directly, so the common case
// costs one push and one pop with no synchronisation beyond the deque itself.
// ---------------------------------------------------------------------------

template <typename FnA, typename FnB>
inline void fork_join(FnA&& a, FnB&& b) {
    ThreadPool& pool = global_pool();
    if (pool.broken() || pool.nworkers() <= 1) { a(); b(); return; }

    const unsigned self = ThreadPool::this_worker();

    struct Job {
        FnB*                      fn;
        std::atomic<std::size_t>* pending;
        static void run(void* p) {
            Job* j = static_cast<Job*>(p);
            (*j->fn)();
            j->pending->fetch_sub(1, std::memory_order_release);
        }
    };

    std::atomic<std::size_t> pending{1};
    Job job{&b, &pending};

    if (!pool.submit(self, Task{&Job::run, &job})) {
        // Queue full: just do it here.
        a();
        b();
        return;
    }

    a();
    pool.wait_for(pending, self);
}

#endif // FYX_ENABLE_PARALLEL

} // namespace detail
} // namespace fyx

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
    if ((std::is_arithmetic<T>::value && distinct_ratio < 0.10) ||
        (!std::is_arithmetic<T>::value && distinct_ratio < 0.05)) {
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
        const unsigned b = classify_bucket_256(*(first + i), tree.data(), comp);
        bid[i] = static_cast<unsigned char>(b);
        ++count[b];
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

    // ---- 5. bucket reorder -----------------------------------------------
    // Serial sample sort does not need per-block reservation: one thread owns
    // every bucket cursor.  A single prefix-position scatter removes an extra
    // full scan over bid[] and two large block-prefix tables from the previous
    // block plan, while the parallel path below keeps per-chunk bases for
    // thread safety.
    {
        std::vector<T> tmp(n);
        std::vector<std::size_t> pos(offset.begin(), offset.begin() + k);
        for (std::size_t i = 0; i < n; ++i) {
            const unsigned b = bid[i];
            tmp[pos[b]++] = std::move(*(first + i));
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
    if ((std::is_arithmetic<T>::value && distinct_ratio < 0.10) ||
        (!std::is_arithmetic<T>::value && distinct_ratio < 0.05)) {
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
                const unsigned b = classify_bucket_256(*(first + i), tree.data(), comp);
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
        std::reverse(p, p + n);
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

template <class T, class Comp>
inline bool try_string_value_count_sort(T* p, std::size_t n, Comp comp, bool descending) {
    if constexpr (!std::is_same<T, std::string>::value) {
        (void)p; (void)n; (void)comp; (void)descending;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        if (!(is_ascending_v<Comp, T> || is_descending_v<Comp, T>)) return false;

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
            for (std::size_t j = 0; j < c; ++j) p[out++] = s;
        }
        (void)descending;
        return true;
    }
}



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
    }

    for (std::size_t i = 0; i < n; ++i) {
        prefetch_stream(src, i, n);
        const T v = src[i];
        const Key k = RT::encode(v);
        const unsigned b = static_cast<unsigned>((k >> shift) & Key(kRadixMask));

        if (FYX_UNLIKELY(sc.hn[b] < sc.need[b])) {
            head[static_cast<std::size_t>(b) * kPerLine + sc.hn[b]] = v;
            ++sc.hn[b];
            continue;
        }

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
    }

    for (std::size_t i = 0; i < n; ++i) {
        prefetch_stream(src, i, n);
        const Key k = src[i];
        const unsigned b = static_cast<unsigned>((k >> shift) & Key(kRadixMask));
        const T v = RT::decode(k);

        if (FYX_UNLIKELY(sc.hn[b] < sc.need[b])) {
            head[static_cast<std::size_t>(b) * kPerLine + sc.hn[b]] = v;
            ++sc.hn[b];
            continue;
        }

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
inline bool radix_key_sparse_probe_ok(T* p, std::size_t n) {
    if constexpr (!radix_supported_v<T> || std::is_same<T, bool>::value) {
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
                done = try_integer_sparse_count_sort_parallel(p, n, descending) ||
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

    detail::InputProfile<T, Comp> top_profile;
    const detail::InputProfile<T, Comp>* prof = nullptr;
    if (n >= detail::kProfileMinN) {
        top_profile = detail::profile_input(p, n, comp);
        prof = &top_profile;
        if (detail::apply_profile_fast_exit(p, n, top_profile, true)) return;
    }

#if FYX_ENABLE_PARALLEL
    const bool want_parallel = (o.threads != 1) &&
        ((o.parallel == Tri::On) ||
         (o.parallel == Tri::Auto && n >= detail::kParallelThreshold &&
          detail::parallel_available()));
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
                    // Sparse random low-cardinality data often has a huge min/max
                    // span; trying dense range counting first costs a full extra
                    // scan before the sparse path wins.  The profile hint is only
                    // a hint: the sparse implementations still validate <=256
                    // distinct keys before committing.
                    if (detail::try_integer_sparse_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_sparse_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_sparse_count_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_integer_range_count_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                } else {
                    if (detail::try_integer_range_count_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_integer_sparse_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_sparse_count_sort_parallel(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                    if (detail::try_radix_key_sparse_count_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                }
                if (detail::try_low_cardinality_count_sort(p, p + n, comp)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
            }
            if (partial_pdq) { detail::pdqsort(p, p + n, comp); detail::record_dispatch(detail::DispatchDecision::PartialPdq); return; }
            if (high_entropy && detail::try_msd_radix_bucket_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (detail::try_parallel_radix_sort(p, n, descending, high_entropy)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
        } else {
            if (detail::try_guarded_radix_order_sort(p, n, comp, want_parallel, high_entropy)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (!high_entropy) {
                if (detail::try_string_value_count_sort(p, n, comp, false)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
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

// ===========================================================================
//  Section 15 -- GPU layer (skeleton + CPU fallback)
//
//  SCOPE / HONESTY NOTE: this sandbox has no GPU and no network, so the GPU
//  compute path CANNOT be compiled or run here.  Per DESIGN.md section 16 the
//  GPU layer's agreed scope is "skeleton + CPU fallback", so this file:
//    * probes for a CUDA driver at runtime via dlopen (no GPU headers needed
//      to compile -- symbols are resolved through dlsym);
//    * exposes a device-buffer / stream abstraction;
//    * routes fyx::sort through gpu_sort_dispatch, which returns false on any
//      failure so the caller falls back to the verified CPU kernels.
//  The actual device kernel is opt-in behind FYX_GPU_COMPUTE (off by default)
//  because it is UNVERIFIED without a GPU; enabling it is for GPU boxes where
//  the kernel can be debugged against real hardware.
//
//  Build: define FYX_ENABLE_GPU to include this file.  Default build skips it.
// ===========================================================================

#if FYX_ENABLE_GPU
#include <dlfcn.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace fyx {
namespace detail {

// Opaque CUDA driver / NVRTC types (we do not include cuda.h; approximate
// definitions are enough for passing pointers through dlsym).
typedef int                 CUdevice;
typedef void*               CUcontext;
typedef void*               CUmodule;
typedef void*               CUfunction;
typedef void*               CUstream;
typedef unsigned long long  CUdeviceptr;
typedef void*               nvrtcProgram;

// ---- resolved driver symbols ----------------------------------------------
struct CudaSyms {
    void* lib = nullptr;
    void* nvrtc = nullptr;
    // driver API
    int (*cuInit)(unsigned int) = nullptr;
    int (*cuDeviceGet)(CUdevice*, int) = nullptr;
    int (*cuCtxCreate)(CUcontext*, unsigned int, CUdevice) = nullptr;
    int (*cuMemAlloc)(CUdeviceptr*, std::size_t) = nullptr;
    int (*cuMemcpyHtoD)(CUdeviceptr, const void*, std::size_t) = nullptr;
    int (*cuMemcpyDtoH)(void*, CUdeviceptr, std::size_t) = nullptr;
    int (*cuModuleLoadData)(CUmodule*, const char*) = nullptr;
    int (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    int (*cuLaunchKernel)(CUfunction, unsigned,unsigned,unsigned,
                          unsigned,unsigned,unsigned,
                          unsigned, CUstream, void**, void**) = nullptr;
    int (*cuMemFree)(CUdeviceptr) = nullptr;
    int (*cuCtxDestroy)(CUcontext) = nullptr;
    // NVRTC
    int (*nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*,
                              int, const char**, const char**) = nullptr;
    int (*nvrtcCompileProgram)(nvrtcProgram, int, const char**) = nullptr;
    int (*nvrtcGetPTX)(nvrtcProgram, char*) = nullptr;
    int (*nvrtcDestroyProgram)(nvrtcProgram*) = nullptr;

    bool ok = false;
};

inline CudaSyms& cuda_syms() {
    static CudaSyms s;
    if (s.ok) return s;
    // dlopen the driver + compiler; if either is missing we simply stay disabled.
    s.lib   = dlopen("libcuda.so",      RTLD_LAZY | RTLD_LOCAL);
    s.nvrtc = dlopen("libnvrtc.so",     RTLD_LAZY | RTLD_LOCAL);
    if (!s.lib) { if (s.nvrtc) dlclose(s.nvrtc); return s; }
    auto sym = [](void* h, const char* n) -> void* {
        return h ? dlsym(h, n) : nullptr;
    };
    s.cuInit              = (decltype(s.cuInit))             sym(s.lib, "cuInit");
    s.cuDeviceGet         = (decltype(s.cuDeviceGet))        sym(s.lib, "cuDeviceGet");
    s.cuCtxCreate         = (decltype(s.cuCtxCreate))        sym(s.lib, "cuCtxCreate");
    s.cuMemAlloc          = (decltype(s.cuMemAlloc))          sym(s.lib, "cuMemAlloc");
    s.cuMemcpyHtoD        = (decltype(s.cuMemcpyHtoD))        sym(s.lib, "cuMemcpyHtoD");
    s.cuMemcpyDtoH        = (decltype(s.cuMemcpyDtoH))        sym(s.lib, "cuMemcpyDtoH");
    s.cuModuleLoadData    = (decltype(s.cuModuleLoadData))    sym(s.lib, "cuModuleLoadData");
    s.cuModuleGetFunction = (decltype(s.cuModuleGetFunction)) sym(s.lib, "cuModuleGetFunction");
    s.cuLaunchKernel      = (decltype(s.cuLaunchKernel))      sym(s.lib, "cuLaunchKernel");
    s.cuMemFree           = (decltype(s.cuMemFree))           sym(s.lib, "cuMemFree");
    s.cuCtxDestroy        = (decltype(s.cuCtxDestroy))        sym(s.lib, "cuCtxDestroy");
    if (s.nvrtc) {
        s.nvrtcCreateProgram   = (decltype(s.nvrtcCreateProgram))   sym(s.nvrtc, "nvrtcCreateProgram");
        s.nvrtcCompileProgram  = (decltype(s.nvrtcCompileProgram))  sym(s.nvrtc, "nvrtcCompileProgram");
        s.nvrtcGetPTX          = (decltype(s.nvrtcGetPTX))          sym(s.nvrtc, "nvrtcGetPTX");
        s.nvrtcDestroyProgram  = (decltype(s.nvrtcDestroyProgram))  sym(s.nvrtc, "nvrtcDestroyProgram");
    }
    s.ok = s.cuInit && s.cuDeviceGet && s.cuCtxCreate && s.cuMemAlloc &&
           s.cuMemcpyHtoD && s.cuMemcpyDtoH && s.cuModuleLoadData &&
           s.cuModuleGetFunction && s.cuLaunchKernel && s.cuMemFree && s.cuCtxDestroy &&
           s.nvrtcCreateProgram && s.nvrtcCompileProgram && s.nvrtcGetPTX && s.nvrtcDestroyProgram;
    return s;
}

// ---- device buffer ---------------------------------------------------------
template <class T>
struct GpuBuffer {
    CudaSyms*   s = nullptr;
    CUdeviceptr  dev = 0;
    std::size_t  n = 0;
    ~GpuBuffer() { if (s && dev && s->cuMemFree) s->cuMemFree(dev); }
    bool alloc(CudaSyms& syms, std::size_t count) {
        s = &syms; n = count;
        return syms.cuMemAlloc(&dev, count * sizeof(T)) == 0;
    }
    bool upload(const T* host) { return s->cuMemcpyHtoD(dev, host, n * sizeof(T)) == 0; }
    bool download(T* host)     { return s->cuMemcpyDtoH(host, dev, n * sizeof(T)) == 0; }
};

// ---- the (opt-in, UNVERIFIED) device radix kernel -------------------------
// One LSD pass: histogram with atomics, then a host prefix-sum, then an atomic
// scatter into the output buffer.  Repeats for every 8-bit digit.  This is the
// structure DESIGN.md section 2.6 describes; it is NOT run in CI (no GPU) and
// is provided so a GPU owner can enable FYX_GPU_COMPUTE and debug it there.
#if defined(FYX_GPU_COMPUTE)
inline std::string gpu_radix_kernel_src(std::size_t key_bytes) {
    const char* ktype = key_bytes == 8 ? "unsigned long long"
                      : key_bytes == 4 ? "unsigned int"
                      : key_bytes == 2 ? "unsigned short"
                      :                  "unsigned char";
    return std::string(R"CUDA(
extern "C" __global__ void fyx_hist(const )") + ktype + R"CUDA( *__restrict__ in,
                                  unsigned int* __restrict__ hist,
                                  unsigned int shift, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int d = (unsigned int)((in[i] >> shift) & 0xFFu);
    atomicAdd(&hist[d], 1u);
}
extern "C" __global__ void fyx_scatter(const )") + ktype + R"CUDA( *__restrict__ in,
                                    )" + ktype + R"CUDA( *__restrict__ out,
                                    unsigned int* __restrict__ base,
                                    unsigned int shift, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int d = (unsigned int)((in[i] >> shift) & 0xFFu);
    unsigned int pos = atomicAdd(&base[d], 1u);
    out[pos] = in[i];
}
)CUDA";
}
#endif

// ---- dispatch entry (returns true on success, false to fall back to CPU) ---
template <class T, class Comp>
inline bool gpu_sort_dispatch(T* p, std::size_t n, Comp, const Options&) {
    // Only numeric, default-ascending keys can take the GPU radix path.
    if (!radix_supported_v<T> || !is_ascending_v<Comp, T>) return false;
    CudaSyms& s = cuda_syms();
    if (!s.ok) return false;   // no driver -> CPU fallback

#if defined(FYX_GPU_COMPUTE)
    // UNVERIFIED ON THIS BOX (no GPU).  Wrapped so any failure falls back.
    try {
        CUdevice dev = 0;
        CUcontext ctx = nullptr;
        if (s.cuInit(0) != 0) return false;
        if (s.cuDeviceGet(&dev, 0) != 0) return false;
        if (s.cuCtxCreate(&ctx, 0, dev) != 0) return false;

        GpuBuffer<T> d_in, d_out;
        GpuBuffer<unsigned int> d_hist, d_base;
        if (!d_in.alloc(s, n)  || !d_out.alloc(s, n) ||
            !d_hist.alloc(s, 256) || !d_base.alloc(s, 256)) { s.cuCtxDestroy(ctx); return false; }
        if (!d_in.upload(p)) { s.cuCtxDestroy(ctx); return false; }

        const std::size_t digits = sizeof(T);  // 8-bit digits per byte
        const std::size_t threads = 256, blocks = (n + threads - 1) / threads;
        unsigned int nn = static_cast<unsigned int>(n);
        CUmodule mod = nullptr;
        CUfunction fhist = nullptr, fscat = nullptr;
        std::string ptx;
        {
            nvrtcProgram prog = nullptr;
            std::string src = gpu_radix_kernel_src(sizeof(T));
            if (s.nvrtcCreateProgram(&prog, src.c_str(), "fyx_radix", 0, nullptr, nullptr) != 0)
                { s.cuCtxDestroy(ctx); return false; }
            const char* opts[] = { "--gpu-architecture=compute_70" };
            if (s.nvrtcCompileProgram(prog, 1, opts) != 0)
                { s.nvrtcDestroyProgram(&prog); s.cuCtxDestroy(ctx); return false; }
            std::size_t sz = 0; char* buf = nullptr;
            s.nvrtcGetPTX(prog, buf); /* buf points into prog; load below */
            ptx = std::string(buf ? buf : "");
            s.nvrtcDestroyProgram(&prog);
        }
        if (s.cuModuleLoadData(&mod, ptx.c_str()) != 0) { s.cuCtxDestroy(ctx); return false; }
        s.cuModuleGetFunction(&fhist, mod, "fyx_hist");
        s.cuModuleGetFunction(&fscat, mod, "fyx_scatter");

        std::vector<unsigned int> host_hist(256), host_base(256);
        std::vector<T> dbl_buf(n);  // host scratch for the ping-pong
        const T* cur_in = p;        // we copy through dbl_buf on host each pass
        // (device ping-pong uses d_in/d_out; simplified to a single in/out swap)
        for (std::size_t d = 0; d < digits; ++d) {
            unsigned int shift = static_cast<unsigned int>(d * 8);
            std::memset(host_hist.data(), 0, 256 * sizeof(unsigned int));
            if (s.cuMemcpyHtoD(d_hist.dev, host_hist.data(), 256 * sizeof(unsigned int)) != 0) break;
            void* hargs[] = { &d_in.dev, &d_hist.dev, &shift, &nn };
            s.cuLaunchKernel(fhist, blocks,1,1, threads,1,1, 0, nullptr, hargs, nullptr);
            if (s.cuMemcpyDtoH(host_hist.data(), d_hist.dev, 256 * sizeof(unsigned int)) != 0) break;
            unsigned int sum = 0;
            for (int b = 0; b < 256; ++b) { host_base[b] = sum; sum += host_hist[b]; }
            if (s.cuMemcpyHtoD(d_base.dev, host_base.data(), 256 * sizeof(unsigned int)) != 0) break;
            void* sargs[] = { &d_in.dev, &d_out.dev, &d_base.dev, &shift, &nn };
            s.cuLaunchKernel(fscat, blocks,1,1, threads,1,1, 0, nullptr, sargs, nullptr);
            // swap in/out for next digit
            CUdeviceptr tmp = d_in.dev; d_in.dev = d_out.dev; d_out.dev = tmp;
        }
        if (s.cuMemcpyDtoH(const_cast<T*>(cur_in), d_in.dev, n * sizeof(T)) != 0) { s.cuCtxDestroy(ctx); return false; }
        (void)dbl_buf;
        s.cuCtxDestroy(ctx);
        return true;   // GPU path completed
    } catch (...) {
        return false;  // any failure -> CPU fallback
    }
#else
    (void)p; (void)n;
    return false;      // compute path disabled: CPU fallback (the documented default)
#endif
}

} // namespace detail
} // namespace fyx

#endif // FYX_ENABLE_GPU

#endif // FYX_SORT_HPP_INCLUDED
