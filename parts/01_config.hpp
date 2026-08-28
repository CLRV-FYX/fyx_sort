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
