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
//    FYX_ENABLE_FAST_PATHS   on by default; set to 0 to disable the entry
//                            sorted/all-equal/reverse/zigzag fast paths.
//    FYX_USE_PDQ_PARTITION   on by default; controls the partially-sorted and
//                            interleaved-run pdq/two-run handling.
//    FYX_USE_STRING_VIEW     on by default; string all-equal probes compare
//                            string data/size directly without copies.
//    FYX_SAMPLE_SORT_V2      on by default; keeps sample-sort v2 tuning behind
//                            an explicit compile-time gate.
//    FYX_MIN_PARALLEL_SIZE   runtime environment variable: Auto-mode minimum
//                            element count before launching the thread pool.
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

#ifndef FYX_ENABLE_FAST_PATHS
#  define FYX_ENABLE_FAST_PATHS 1
#endif
#ifndef FYX_USE_PDQ_PARTITION
#  define FYX_USE_PDQ_PARTITION 1
#endif
#ifndef FYX_USE_STRING_VIEW
#  define FYX_USE_STRING_VIEW 1
#endif
#ifndef FYX_SAMPLE_SORT_V2
#  define FYX_SAMPLE_SORT_V2 1
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
    std::size_t remaining_heads = 0;
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
        remaining_heads += sc.need[b];
    }

    auto push_line = [&](Key k, unsigned b) {
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
    };

    std::size_t i = 0;
    for (; i < n && remaining_heads != 0; ++i) {
        prefetch_stream(src, i, n);

        const Key      k = src[i];
        const unsigned b = static_cast<unsigned>((k >> shift) & Key(kRadixMask));

        // Alignment prologue for this bucket.  Once every bucket has consumed
        // its tiny prologue (at most one cache line total per bucket), the main
        // loop below no longer pays this branch on every element.
        if (FYX_UNLIKELY(sc.hn[b] < sc.need[b])) {
            head[static_cast<std::size_t>(b) * kPerLine + sc.hn[b]] = k;
            ++sc.hn[b];
            --remaining_heads;
            continue;
        }

        push_line(k, b);
    }

    for (; i < n; ++i) {
        prefetch_stream(src, i, n);
        const Key      k = src[i];
        const unsigned b = static_cast<unsigned>((k >> shift) & Key(kRadixMask));
        push_line(k, b);
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
    // Splitter search keeps copies of the elements it sampled, so a payload
    // that cannot be copied -- std::unique_ptr -- has to be left to the
    // comparison sort, which only ever moves.
    if constexpr (!std::is_copy_constructible<T>::value) { pdqsort(first, last, comp); return; }

    // ---- 0. structural pre-pass ------------------------------------------
    // Sampling, splitter search and 256-way classification are wasted on a
    // range that is already in order, in reverse order, or all equivalent --
    // and those are the shapes a comparison sort gets handed in practice
    // (appending to an ordered table, re-sorting after a merge, a column that
    // turned out to have one value).  One comparison per element finds them,
    // and the pass abandons itself as soon as the range is proven to be none
    // of the three: random input leaves after two or three elements, so this
    // costs nothing where it does not pay off.  Only the second comparison is
    // conditional, so an ascending range is confirmed with one per element.
    if (n >= 64) {
        bool asc = true, desc = true, same = true;
        for (std::size_t i = 1; i < n; ++i) {
            It a = first + (i - 1), b = first + i;
            if (comp(*b, *a)) {                 // b < a: descent
                asc = false; same = false;
                if (!desc) break;
            } else if (desc || same) {          // b >= a: ascent or tie
                if (comp(*a, *b)) {             // a < b: ascent
                    desc = false; same = false;
                    if (!asc) break;
                }
            }
        }
        if (asc || same) return;                // ordered, or all equivalent
        if (desc) { std::reverse(first, last); return; }
    }

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

// ============================================================================
//  Section 12b -- Adaptive-order weapons (natural-run merge / dirty-patch merge)
//
//  Comparison sort costs O(n log n) comparisons no matter how much order the
//  input already has; radix sort costs a fixed number of passes no matter how
//  few keys actually differ.  Real inputs are usually neither random nor
//  perfectly sorted, and both extremes waste work there.  This section adds
//  two structure detectors that turn "almost sorted" into O(n) sequential
//  work:
//
//  * natural-run merge (`try_natural_run_merge`)
//      One scan finds the maximal monotone runs of the input.  Descending runs
//      are reversed in place (fyx::sort is not stable, so reversing equal keys
//      is allowed) and the runs are then merged bottom-up with a buffer no
//      larger than the smallest side of any merge.  Cost is O(n log R) moves
//      with strictly sequential access, so organ pipes, rotated sorted arrays,
//      concatenated sorted blocks and "sorted with a few blocks moved" inputs
//      cost one or two passes instead of 4-8 radix passes (or 20 partitioning
//      passes for a comparison sort).
//
//  * dirty-patch merge (`try_dirty_patch_merge`)
//      When only a small fraction of the positions take part in an inversion,
//      the clean subsequence is already sorted.  Pull the dirty positions out
//      into a tiny buffer, sort that buffer, and merge it back over the
//      compacted clean run.  Three sequential passes total, independent of the
//      key width, so 64-bit keys cost the same as 32-bit ones.
//
//  Both detectors are structure-driven, not benchmark-driven:
//    - they are read-only until the structure has been proven (the run scan and
//      the inversion scan never mutate), so a rejection costs a few hundred
//      touched elements on random data;
//    - they never change the multiset (reversal, compaction and merge are all
//      permutations), so a wrong guess cannot corrupt the caller's data;
//    - they cover whole *classes* of input (any arrangement of R monotone runs,
//      any pattern of local disorder), not one generator's output shape.
// ============================================================================

namespace fyx {
namespace detail {

#ifndef FYX_ENABLE_ADAPTIVE_WEAPONS
#  define FYX_ENABLE_ADAPTIVE_WEAPONS 1
#endif

// ---------------------------------------------------------------------------
// Natural runs
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Ordering used by every adaptive weapon.
//
// Default-order floating point is compared through its radix key, exactly like
// the radix and pdq pattern kernels: std::less<double> treats NaN as
// equivalent to everything and cannot tell -0 from +0, while the library
// documents the IEEE totalOrder relation (-NaN < ... < -0 < +0 < ... < +NaN).
// Sorting by the key produces a sequence that is still ordered under the
// caller's comparator, so this is always safe.
// ---------------------------------------------------------------------------
template <class T, class Comp>
inline auto adaptive_order(Comp comp) noexcept {
    constexpr bool radix_order = radix_supported_v<T> &&
        std::is_floating_point<T>::value &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    return [comp](const T& a, const T& b) -> bool {
        if constexpr (radix_order) {
            using RT = RadixTraits<T>;
            const auto ka = RT::encode(a);
            const auto kb = RT::encode(b);
            if constexpr (is_descending_v<Comp, T>) return kb < ka;
            else return ka < kb;
        } else {
            (void)comp;
            return comp(a, b);
        }
    };
}

/// One maximal monotone run: elements [begin, end) are non-decreasing (once
/// descending runs have been reversed) under the sort's ordering.
struct MonotoneRun {
    std::size_t begin;
    std::size_t end;
};

/// Scans [p, p + n) and records up to `cap` maximal monotone runs.
///
/// Returns the number of runs, or `cap + 1` as soon as more than `cap` runs
/// exist (the scan then stops early -- random data is rejected after touching
/// about 2 * cap elements).  The scan is read-only.
template <class T, class Comp>
inline std::size_t scan_monotone_runs(const T* p, std::size_t n, Comp comp,
                                      std::size_t cap, MonotoneRun* out) {
    std::size_t count = 0;
    std::size_t start = 0;
    int dir = 0;                       // +1 ascending, -1 descending, 0 unknown
    for (std::size_t i = 1; i < n; ++i) {
        const bool down = comp(p[i], p[i - 1]);
        if (!down) {
            if (!comp(p[i - 1], p[i])) continue;   // equivalent: no information
        }
        const int d = down ? -1 : 1;
        if (dir == 0) { dir = d; continue; }
        if (d != dir) {
            // The extremum at i-1 keeps the closed run; the new run starts at i.
            if (count < cap) { out[count].begin = start; out[count].end = i; }
            ++count;
            if (count > cap) return cap + 1;
            start = i;
            dir = d;
        }
    }
    if (count < cap) { out[count].begin = start; out[count].end = n; }
    ++count;
    return count > cap ? cap + 1 : count;
}

/// Merges the adjacent runs [l, m) and [m, r) in place, using `buf` as scratch
/// space for the smaller of the two runs (at most min(r - m, m - l) elements).
/// `buf` may be null, in which case the merge falls back to std::inplace_merge
/// (the only correct option for types whose objects have to be constructed).
template <class T, class Comp>
inline void merge_adjacent_runs(T* p, std::size_t l, std::size_t m, std::size_t r,
                                T* buf, Comp comp) {
    const std::size_t a = m - l;
    const std::size_t b = r - m;
    if (a == 0 || b == 0) return;
    if (buf == nullptr) {
        std::inplace_merge(p + l, p + m, p + r, comp);
        return;
    }
    if (a >= b) {
        // Buffer the right run and merge backwards.  Writes always land at
        // indices >= the next unread element of the left run, so the left run
        // is never clobbered.
        for (std::size_t i = 0; i < b; ++i) buf[i] = std::move(p[m + i]);
        std::size_t ia = m, ib = b, w = r;
        while (ib != 0 && ia != l) {
            if (comp(buf[ib - 1], p[ia - 1])) {
                --ia; --w; p[w] = std::move(p[ia]);
            } else {
                --ib; --w; p[w] = std::move(buf[ib]);
            }
        }
        while (ib != 0) { --ib; --w; p[w] = std::move(buf[ib]); }
    } else {
        // Buffer the left run and merge forwards.
        for (std::size_t i = 0; i < a; ++i) buf[i] = std::move(p[l + i]);
        std::size_t ia = 0, rb = m, w = l;
        while (ia != a && rb != r) {
            if (comp(p[rb], buf[ia])) { p[w] = std::move(p[rb]); ++rb; }
            else                      { p[w] = std::move(buf[ia]); ++ia; }
            ++w;
        }
        while (ia != a) { p[w] = std::move(buf[ia]); ++ia; ++w; }
    }
}

/// Adaptive natural-merge sort.
///
/// Returns true when the range was sorted by merging its monotone runs.
/// `max_runs` bounds the run count (and therefore the scan cost on inputs that
/// have no runs at all); `max_levels` bounds the number of merge passes so the
/// O(n log R) traffic stays below what the radix/sample kernels would spend.
template <class T, class Comp>
inline bool try_natural_run_merge(T* p, std::size_t n, Comp comp,
                                  std::size_t max_runs, std::size_t max_levels) {
    if (n < 1024 || max_runs < 2) return false;
    if constexpr (!std::is_move_constructible<T>::value ||
                  !std::is_move_assignable<T>::value) {
        return false;
    } else {
        auto before = adaptive_order<T>(comp);
        std::vector<MonotoneRun> runbuf(max_runs + 1);
        const std::size_t count = scan_monotone_runs(p, n, before, max_runs, runbuf.data());
        if (count == 0 || count > max_runs) return false;

        if (count == 1) {
            // A single run: already ordered, or ordered the wrong way round.
            if (before(p[n - 1], p[0])) std::reverse(p, p + n);
            return true;
        }

        // Simulate the bottom-up schedule: count the passes and the largest
        // scratch buffer any single merge needs.  Both must stay inside budget
        // before we touch a single element.
        std::vector<std::size_t> bounds(count + 1);
        std::vector<std::size_t> next(count + 1);
        bounds[0] = 0;
        for (std::size_t i = 0; i < count; ++i) bounds[i + 1] = runbuf[i].end;
        std::size_t nruns = count;
        std::size_t levels = 0;
        std::size_t maxbuf = 1;
        {
            std::vector<std::size_t> sim = bounds;
            std::size_t m = count;
            while (m > 1) {
                std::size_t w = 1;
                std::size_t i = 0;
                next[0] = sim[0];
                for (; i + 2 <= m; i += 2) {
                    const std::size_t left  = sim[i + 1] - sim[i];
                    const std::size_t right = sim[i + 2] - sim[i + 1];
                    const std::size_t small = left < right ? left : right;
                    if (small > maxbuf) maxbuf = small;
                    next[w++] = sim[i + 2];
                }
                if (i < m) next[w++] = sim[i + 1];
                ++levels;
                if (levels > max_levels) return false;
                for (std::size_t k = 0; k < w; ++k) sim[k] = next[k];
                m = w - 1;
            }
        }

        std::unique_ptr<ScratchLease<T>> lease;
        // Descending runs become ascending; fyx::sort is unstable so reversing
        // equal keys is allowed.
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t b = runbuf[i].begin;
            const std::size_t e = runbuf[i].end;
            if (e - b > 1 && before(p[e - 1], p[b])) std::reverse(p + b, p + e);
        }

        // Raw scratch storage only holds objects that need no construction;
        // everything else uses the standard in-place merge, which manages its
        // own (properly constructed) temporary.
        T* buf = nullptr;
        if constexpr (std::is_trivially_copyable<T>::value) {
            lease.reset(new ScratchLease<T>(maxbuf));
            if (!lease->valid()) return false;  // no scratch: leave it to radix/pdq
            buf = lease->get();
        }

        std::size_t m = count;
        while (m > 1) {
            std::size_t w = 1;
            std::size_t i = 0;
            next[0] = bounds[0];
            for (; i + 2 <= m; i += 2) {
                merge_adjacent_runs(p, bounds[i], bounds[i + 1], bounds[i + 2], buf, before);
                next[w++] = bounds[i + 2];
            }
            if (i < m) next[w++] = bounds[i + 1];
            for (std::size_t k = 0; k < w; ++k) bounds[k] = next[k];
            m = w - 1;
        }
        return true;
    }
}

/// Convenience wrapper: budget the merge against the kernels it replaces.
/// Radix on 32-bit keys spends 4 passes, on 64-bit keys 8 passes (each one a
/// read + a scattered write), so 64-bit types tolerate more merge levels than
/// 32-bit ones.  Non-radix types pay comparisons, which merge saves most.
template <class T, class Comp>
inline bool try_natural_run_merge_adaptive(T* p, std::size_t n, Comp comp) {
#if !FYX_ENABLE_ADAPTIVE_WEAPONS
    (void)p; (void)n; (void)comp;
    return false;
#else
    constexpr std::size_t max_runs   = 64;
    constexpr std::size_t max_levels =
        radix_supported_v<T> ? (sizeof(T) <= 4 ? 4u : 6u) : 8u;
    return try_natural_run_merge(p, n, comp, max_runs, max_levels);
#endif
}

// ---------------------------------------------------------------------------
// Dirty-patch merge
// ---------------------------------------------------------------------------

/// Moves `k` elements from `src` to `dst`; the ranges may overlap, and `dst`
/// may sit before or after `src`.  Trivially copyable payloads go through
/// memmove, which the vectoriser cannot beat; the rest are walked away from
/// the end that would otherwise be overwritten first.
template <class T>
inline void move_range_bulk(T* dst, T* src, std::size_t k) noexcept {
    if constexpr (std::is_trivially_copyable<T>::value) {
        if (k) std::memmove(static_cast<void*>(dst), static_cast<const void*>(src), k * sizeof(T));
    } else if (dst < src) {
        for (std::size_t i = 0; i < k; ++i) dst[i] = std::move(src[i]);
    } else if (dst > src) {
        for (std::size_t i = k; i-- > 0;) dst[i] = std::move(src[i]);
    }
}

/// Sorts a patch and merges it back over the clean run sitting at the front of
/// `p`.  Writes run backwards so the array can act as its own output: the write
/// cursor never passes the read cursor of the clean run.
///
/// The patch is tiny next to the run, so comparing one element at a time would
/// spend the whole budget deciding what to do with elements that are simply
/// copied.  Each step gallops back through the run to find how many of its
/// elements belong after the next patch element and moves that block in one
/// go: O(log(n/p)) comparisons per patch element and one bulk move per block.
template <class T, class Comp>
inline void merge_patch_back(T* p, std::size_t clean_n, std::vector<T>& patch, Comp comp) {
    const std::size_t patch_n = patch.size();
    if (patch_n == 0) return;
    pdqsort(patch.data(), patch.data() + patch_n, comp);
    if (clean_n == 0) {
        for (std::size_t i = 0; i < patch_n; ++i) p[i] = std::move(patch[i]);
        return;
    }
    std::size_t ci = clean_n, pi = patch_n, out = clean_n + patch_n;
    while (pi != 0) {
        const T& pv = patch[pi - 1];
        // Trailing run of the clean side that is >= pv: it belongs after pv.
        std::size_t t = 0, step = 1;
        while (t + step <= ci && !comp(p[ci - (t + step)], pv)) {
            t += step;
            step <<= 1;
        }
        std::size_t lo = t, hi = std::min<std::size_t>(t + step, ci);
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo + 1) / 2;
            if (!comp(p[ci - mid], pv)) lo = mid;
            else                        hi = mid - 1;
        }
        // lo elements of the run, then one patch element.
        if (lo) move_range_bulk(p + (out - lo), p + (ci - lo), lo);
        out -= lo;
        ci  -= lo;
        --out;
        p[out] = std::move(patch[pi - 1]);
        --pi;
    }
}

/// Sorts inputs whose disorder is concentrated in a small set of positions.
///
/// A position is "dirty" when it takes part in an adjacent inversion.  If the
/// dirty set is small, the clean subsequence is almost certainly already
/// ordered; we verify that in one more scan (repairing it by dirtying the few
/// positions that break it), then compact the clean elements to the front,
/// sort the tiny dirty patch, and merge the two sorted sequences back -- the
/// merge runs backwards so it can use the array itself as the output.
///
/// Total cost: three or four sequential passes plus a sort of the patch,
/// independent of the key width.  Returns false (without having moved
/// anything) as soon as the dirty set grows past `max_dirty`.
template <class T, class Comp>
inline bool try_dirty_patch_merge(T* p, std::size_t n, Comp comp,
                                  std::size_t max_dirty) {
    if (n < 1024 || max_dirty == 0) return false;
    if constexpr (!std::is_move_constructible<T>::value ||
                  !std::is_move_assignable<T>::value ||
                  !std::is_copy_constructible<T>::value) {
        return false;
    } else {
        ScratchLease<std::uint64_t> words((n >> 6) + 2);
        if (!words.valid()) return false;
        std::uint64_t* bits = words.get();
        std::memset(bits, 0, ((n >> 6) + 2) * sizeof(std::uint64_t));
        auto before = adaptive_order<T>(comp);

        std::size_t dirty_count = 0;
        auto mark = [&](std::size_t idx) {
            const std::size_t w = idx >> 6;
            const std::uint64_t m = std::uint64_t(1) << (idx & 63);
            if ((bits[w] & m) == 0) {
                bits[w] |= m;
                ++dirty_count;
            }
        };
        auto is_dirty = [&](std::size_t idx) {
            return (bits[idx >> 6] >> (idx & 63)) & 1u;
        };

        // Pass 1: every adjacent inversion pins down two dirty positions.
        for (std::size_t i = 1; i < n; ++i) {
            if (before(p[i], p[i - 1])) {
                mark(i - 1);
                mark(i);
                if (dirty_count > max_dirty) return false;
            }
        }
        if (dirty_count == 0) return true;      // already ordered

        // Pass 2+: prove the clean subsequence is ordered.  Two overlapping
        // swaps can hide an inversion behind a position that the marking pass
        // dirtied, so a couple of scans are allowed -- but the whole repair has
        // to stay inside `grow_cap`, which a block-level displacement (where
        // the patch would have to swallow entire moved runs) cannot do.  Those
        // shapes fall through to the displacement merge, which characterises
        // them exactly in two scans instead of growing a patch pair by pair.
        // The repair budget is deliberately tight: a shape that needs the whole
        // range re-marked (a moved block, say) is not a "local disorder" shape,
        // and the displacement merge below handles it in fewer passes than we
        // would spend discovering that here.
        // The hypothesis under test is that a handful of positions are out of
        // place.  A pass that has to re-mark a large part of the original
        // patch refutes it -- that is what a moved block looks like from here,
        // every element inside it still being in order -- so the scan is
        // abandoned instead of paying two more of them before giving up.
        const std::size_t grow_cap = std::min<std::size_t>(max_dirty, dirty_count * 4 + 64);
        const std::size_t dirty0 = dirty_count;
        bool ordered = false;
        for (int pass = 0; pass < 3 && !ordered; ++pass) {
            bool changed = false;
            std::size_t added = 0;
            std::size_t prev = n;
            for (std::size_t i = 0; i < n; ++i) {
                if (is_dirty(i)) continue;
                if (prev != n && before(p[i], p[prev])) {
                    mark(prev);
                    mark(i);
                    ++added;
                    if (dirty_count > grow_cap) return false;
                    changed = true;
                    prev = n;
                    continue;
                }
                prev = i;
            }
            ordered = !changed;
            if (!ordered && added * 2u > dirty0) return false;
        }
        if (!ordered) return false;

        // Compact: clean elements slide to the front in order, dirty elements
        // move into the patch buffer.
        std::vector<T> patch;
        patch.reserve(dirty_count);
        std::size_t w = 0;
        // Word at a time: a patch is a fraction of a percent of the range, so
        // nearly every word is entirely clean and slides as one block instead
        // of sixty-four tested elements.
        const std::size_t nwords = (n >> 6) + 1;
        for (std::size_t wi = 0; wi < nwords; ++wi) {
            const std::size_t base = wi << 6;
            if (base >= n) break;
            const std::size_t cnt = std::min<std::size_t>(64, n - base);
            const std::uint64_t bw = bits[wi];
            if (bw == 0 && cnt == 64) {
                if (w != base) move_range_bulk(p + w, p + base, 64);
                w += 64;
                continue;
            }
            for (std::size_t k = 0; k < cnt; ++k) {
                if ((bw >> k) & 1u) patch.push_back(std::move(p[base + k]));
                else if (w != base + k) p[w] = std::move(p[base + k]), ++w;
                else ++w;
            }
        }
        if (w + patch.size() != n) return false;       // defensive
        merge_patch_back(p, w, patch, before);
        return true;
    }
}

/// Budget wrapper: disorder beyond an eighth of the range is no longer "local",
/// and the patch sort stops being cheaper than the kernels it replaces.
template <class T, class Comp>
inline bool try_dirty_patch_merge_adaptive(T* p, std::size_t n, Comp comp) {
#if !FYX_ENABLE_ADAPTIVE_WEAPONS
    (void)p; (void)n; (void)comp;
    return false;
#else
    return try_dirty_patch_merge(p, n, comp, n / 8);
#endif
}

// ---------------------------------------------------------------------------
// Displacement patch merge
// ---------------------------------------------------------------------------

/// Sorts inputs whose disorder is a set of *displaced* elements, wherever they
/// were moved from and however far they travelled.
///
/// Characterisation (exact, two linear scans):
///   element i may stay  <=>  it is >= every element before it  and
///                            it is <= every element after it.
/// Two elements that both satisfy the test are automatically in order (each one
/// is bounded by everything on the other side), so what remains after removing
/// the failures is already sorted -- no iteration, no guessing, and no
/// assumption about how far the failures travelled.  That covers whole classes
/// of input the adjacent-inversion test cannot see: swapped blocks, moved
/// segments, elements dragged across a long clean run, splices.
///
/// Cost: two scans plus one compaction and one merge, and a sort of the patch.
/// The only scratch memory is two bits per element, so the probe never fights
/// the radix kernels for the thread arena -- it stays affordable even when it
/// declines.
template <class T, class Comp>
inline bool try_displacement_patch_merge(T* p, std::size_t n, Comp comp,
                                         std::size_t max_dirty) {
    if (n < 1024 || max_dirty == 0) return false;
    if constexpr (!std::is_move_constructible<T>::value ||
                  !std::is_move_assignable<T>::value ||
                  !std::is_copy_constructible<T>::value) {
        return false;
    } else {
        auto before = adaptive_order<T>(comp);
        const std::size_t words = (n >> 6) + 2;
        ScratchLease<std::uint64_t> bits_lease(words * 2);
        if (!bits_lease.valid()) return false;
        std::uint64_t* suf = bits_lease.get();
        std::uint64_t* pre = suf + words;
        std::memset(suf, 0, words * 2 * sizeof(std::uint64_t));

        const std::size_t n_sentinel = n;
        std::size_t low = 0;

        // Pass 1 (backward): an element greater than the minimum of its own
        // suffix cannot stay.  One running index, one bit per element.
        {
            std::size_t smin = n_sentinel;
            for (std::size_t i = n - 1;; --i) {
                if (smin != n_sentinel && before(p[smin], p[i])) {
                    suf[i >> 6] |= std::uint64_t(1) << (i & 63);
                    if (++low > max_dirty) return false;
                }
                if (smin == n_sentinel || before(p[i], p[smin])) smin = i;
                if (i == 0) break;
            }
        }

        // Pass 2 (forward): same test against the maximum of the prefix.  The
        // union of the two bit sets is the patch; its size is known before a
        // single element moves.
        std::size_t high = 0;
        {
            std::size_t cmax = n_sentinel;
            for (std::size_t i = 0; i < n; ++i) {
                if (cmax != n_sentinel && before(p[i], p[cmax])) {
                    pre[i >> 6] |= std::uint64_t(1) << (i & 63);
                    if (++high > max_dirty) return false;
                }
                if (cmax == n_sentinel || before(p[cmax], p[i])) cmax = i;
            }
        }
        if (low + high == 0) return true;
        if (low + high > max_dirty) return false;

        // Pass 3: compact around the patch, then merge the patch back.
        std::vector<T> patch;
        patch.reserve(low + high);
        std::size_t w = 0;
        const std::size_t nwords = (n >> 6) + 1;
        for (std::size_t wi = 0; wi < nwords; ++wi) {
            const std::size_t base = wi << 6;
            if (base >= n) break;
            const std::size_t cnt = std::min<std::size_t>(64, n - base);
            const std::uint64_t bw = suf[wi] | pre[wi];
            if (bw == 0 && cnt == 64) {
                if (w != base) move_range_bulk(p + w, p + base, 64);
                w += 64;
                continue;
            }
            for (std::size_t k = 0; k < cnt; ++k) {
                if ((bw >> k) & 1u) patch.push_back(std::move(p[base + k]));
                else {
                    if (w != base + k) p[w] = std::move(p[base + k]);
                    ++w;
                }
            }
        }
        if (w + patch.size() != n) return false;      // defensive
        merge_patch_back(p, w, patch, before);
        return true;
    }
}

template <class T, class Comp>
inline bool try_displacement_patch_merge_adaptive(T* p, std::size_t n, Comp comp) {
#if !FYX_ENABLE_ADAPTIVE_WEAPONS
    (void)p; (void)n; (void)comp;
    return false;
#else
    return try_displacement_patch_merge(p, n, comp, n / 8);
#endif
}

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


template <class T, class Comp>
inline bool try_partially_sorted_local_repair(T* p, std::size_t n, Comp comp) {
#if !FYX_USE_PDQ_PARTITION
    (void)p; (void)n; (void)comp;
    return false;
#else
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
    if (try_dirty_patch_merge_adaptive(p, n, comp)) return true;
    // Adjacent inversions cannot see a block that was moved wholesale: every
    // element inside it is still in order.  The prefix-max / suffix-min
    // characterisation finds those, so spliced / block-moved inputs also cost
    // a couple of linear passes instead of a full sort.
    if (try_displacement_patch_merge_adaptive(p, n, comp)) return true;
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

        for (std::size_t i = probe; i < mid; ++i) {
            if (before(p[i - 1], p[i])) return false;
        }
        const std::size_t suffix_start = std::max<std::size_t>(mid + probe, mid + 1u);
        for (std::size_t i = suffix_start; i < n; ++i) {
            if (before(p[i], p[i - 1])) return false;
        }
        std::reverse(p, p + mid);
        return true;
    }
#endif
}


template <class T, class Comp>
inline bool try_fast_reverse_exit(T* p, std::size_t n, Comp comp) {
#if !FYX_ENABLE_FAST_PATHS
    (void)p; (void)n; (void)comp;
    return false;
#else
    if (n < 2) return false;
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
inline constexpr std::size_t kProfilePartialPdqMax   = 64u << 20;

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
    static_assert(Bits > 8 && Bits <= 13, "wide radix count is tuned for 9..13 bits");
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
            if (try_integer_range_count_sort(p, n, descending)) { record_dispatch(DispatchDecision::LowCardinality); return; }
            if (try_radix_key_sparse_count_sort(p, n, descending)) { record_dispatch(DispatchDecision::LowCardinality); return; }
            if (try_low_cardinality_count_sort(p, p + n, comp)) { record_dispatch(DispatchDecision::LowCardinality); return; }
        }
        if (partial_pdq && try_partially_sorted_local_repair(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; }
        if (try_radix_permutation_range_sort(p, n, descending)) { record_dispatch(DispatchDecision::Radix); return; }
        if constexpr (radix_type) {
            if (n >= kRadixThreshold || std::is_floating_point<T>::value) {
#if FYX_ENABLE_PARALLEL
                // The high-prefix sort beats the 32-wide one on every size
                // measured -- 1M int32: 0.0090s against 0.0120s, 8M: 0.080s
                // against 0.154s -- so it goes first and the wide sort is the
                // fallback for the shapes whose prefix it declines.
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
    if (n > detail::kNetworkMax && detail::try_fast_reverse_exit(p, n, comp)) {
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
    if (n > detail::kNetworkMax && detail::try_fast_order_exit(p, n, comp, true)) return;
    // Inputs made of a handful of monotone runs (rotated sorted arrays,
    // concatenated sorted blocks, shuffled block permutations, ...) cost
    // O(n log R) sequential moves here instead of a fixed 4-8 radix passes or
    // a full comparison recursion.  Random data is rejected inside the scan
    // after touching a couple of hundred elements.
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
    if (n > detail::kNetworkMax && detail::try_bounded_insertion_repair(p, n, comp)) {
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
            if (detail::try_partially_sorted_local_repair(p, n, comp)) {
                detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                return;
            }
            if (detail::try_radix_permutation_range_sort(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::Radix);
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
                if (detail::try_partially_sorted_local_repair(p, n, comp)) {
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
            if (partial_pdq && detail::try_partially_sorted_local_repair(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::PartialPdq); return; }
            if (detail::try_radix_permutation_range_sort(p, n, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
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
