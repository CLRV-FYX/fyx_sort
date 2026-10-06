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
#include <atomic>
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
#include <deque>
#  include <mutex>
#  include <atomic>
#  include <condition_variable>
#  include <random>
#  include <chrono>
#else
#  include <atomic>
#endif

// ---------------------------------------------------------------------------
// GCC at -O2 (the most common release flag) uses the "very cheap" vector cost
// model and neither unswitches nor peels loops.  The library's branch-free
// order / structure scans then run 2-4x slower than at -O3 (Ice Lake-SP,
// GCC 12.2, 200K int32: sorted-input exit 0.71 vs 0.15 ns/elem, rotated 1.8
// vs 0.85, block-swap 2.1 vs 1.2).  Enabling just those three optimisations
// for the library's own functions recovers the -O3 numbers without changing
// the caller's flags.  Clang (vectorises at -O2) and MSVC are unaffected;
// define FYX_NO_OPTIMIZE_PRAGMA to opt out.  Popped at the end of the header.
// ---------------------------------------------------------------------------
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && \
    defined(__OPTIMIZE__) && !defined(__OPTIMIZE_SIZE__) && !defined(FYX_NO_OPTIMIZE_PRAGMA)
#  define FYX_OPTIMIZE_PRAGMA_ACTIVE 1
#  pragma GCC push_options
#  pragma GCC optimize("vect-cost-model=dynamic", "unswitch-loops", "peel-loops")
#else
#  define FYX_OPTIMIZE_PRAGMA_ACTIVE 0
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
/// Silences -Wunused-local-typedefs for a typedef that only some compile-time
/// configurations look at (e.g. a kernel switched off by a FYX_* macro).
#define FYX_UNUSED_TYPE(T) ((void)sizeof(T*))

#define FYX_STRINGIFY_(x) #x
#define FYX_STRINGIFY(x)  FYX_STRINGIFY_(x)

#if defined(FYX_ASSERT)
#  define FYX_ASSERT_IMPL(x) FYX_ASSERT(x)
#else
#  define FYX_ASSERT_IMPL(x) ((void)0)
#endif

// Exceptions ----------------------------------------------------------------
// GCC and Clang do not define __cpp_exceptions as 0 under -fno-exceptions --
// they leave it undefined, along with __EXCEPTIONS -- so testing it for zero
// never fired and a plain -fno-exceptions build (no FYX_NO_EXCEPTIONS) failed
// to compile on the first try/catch it reached.
#if defined(FYX_NO_EXCEPTIONS) \
    || (defined(__cpp_exceptions) && __cpp_exceptions == 0) \
    || (FYX_GNUC_LIKE && !defined(__EXCEPTIONS)) \
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
#elif FYX_ARCH_X86 && FYX_GNUC_LIKE
// FYX_DISABLE_SIMD compiles every vector kernel out, but the spin hint
// (_mm_pause) and the non-temporal store fence (_mm_sfence) are not vector
// kernels and are still used.  They live in the SSE headers, which are
// available on any x86 target without an -m switch, so a scalar-only build
// stays a scalar-only build and still compiles.
#  include <xmmintrin.h>
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

/// "Leave the patch merges their own budget" -- see patch_merge_dirty_budget.
inline constexpr std::size_t kPatchDirtyDefault = static_cast<std::size_t>(-1);

/// Digit width of the wide radix passes (10/11/11 for 32-bit keys).  Used to
/// convert a key span into "how many passes would radix actually run".
inline constexpr unsigned kRadixWidePassBits = 11;

/// Evenly spaced elements read by every sampling probe (input profile,
/// distinct estimate, pivot gates).  Lives here rather than next to the
/// profile because the counting kernels sample before the profile exists.
inline constexpr std::size_t kProfileSampleLimit = 1024;

/// Sorting-network ceiling.  Everything at or below this length is sorted by a
/// branch-free network, never by insertion sort.
inline constexpr std::size_t kNetworkMax = 64;

/// Below this length the AVX-512 vectorised quicksort (parts/10b_vsort.hpp)
/// does not pay: a range that fits in L2 is where the radix passes are cheap,
/// and the quicksort still has to walk log(n/leaf) levels over it.
inline constexpr std::size_t kVqsortMinN = 1u << 14;          // 16384
// Memory-form vpcompress in the AVX-512 quicksort partition (1, default):
// ~15% faster than register compress + masked store on Ice Lake-SP.  AMD
// Zen 4 microcodes the memory form; build with 0 there (unmeasured here).
#ifndef FYX_VQ_COMPRESS_TO_MEMORY
#  define FYX_VQ_COMPRESS_TO_MEMORY 1
#endif
// Full unrolling of fixed-trip loops over register arrays / merge chains: GCC -O2 does
// not unroll them on its own, which leaves the arrays in memory and cost the
// partition ~30% at -O2 versus -O3 (Ice Lake-SP, GCC 12.2).
#if defined(__clang__)
#  define FYX_VQ_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#  define FYX_VQ_UNROLL _Pragma("GCC unroll 16")
#else
#  define FYX_VQ_UNROLL
#endif

// Independent merge chains in the AVX-512 two-run merge (latency hiding).
#ifndef FYX_VMERGE_CHAINS
#  define FYX_VMERGE_CHAINS 2
#endif

// Lower bound of the small-range AVX-512 vector quicksort fast path.
inline constexpr std::size_t kSmallVqsortMinN = 2;
// Upper bound (exclusive) of that fast path.  It is the serial path for
// every size the pool does not take (Auto mode parallelises from 1M); with
// the column-network leaf the vector quicksort beats the probe stack of the
// general path on random / duplicate-heavy input up to there (Ice Lake-SP,
// GCC 12.2, int32 20K random 4.5 -> 2.3 ns/elem).
#ifndef FYX_SMALL_VQ_MAX_N
#  define FYX_SMALL_VQ_MAX_N (1u << 20)
#endif
inline constexpr std::size_t kSmallVqsortMaxN = FYX_SMALL_VQ_MAX_N;

// Records keyed by a sampled integer field: 1 = MSD record radix (one scatter,
// cache-resident buckets), 0 = legacy LSD pass-per-digit radix.
#ifndef FYX_RECORD_MSD
#  define FYX_RECORD_MSD 1
#endif

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
      _Pragma(FYX_STRINGIFY(clang attribute push (__attribute__((target(isa))), apply_to = function)))
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
//  Section 10b -- AVX-512 vectorised quicksort
//
//  Why this exists
//  ---------------
//  `tools/dev/NOTES.md` closes the accounting on the LSD radix for
//  high-entropy numeric input: one count+scatter pass costs ~3.1 ns/elem and
//  does not respond to blocking, wider digits or a deeper write-combining
//  buffer, and 32 bits at <=13 bits per pass means three passes is the floor.
//  ~9.4 ns/elem of CPU work is therefore the floor for LSD radix on int32
//  here, against ~7.2 for a vectorised quicksort of the vqsort class.  Every
//  cheaper idea was measured and lost.  This is that vectorised quicksort.
//
//  Structure (the same one vqsort and x86-simd-sort use)
//  -----------------------------------------------------
//    * partition with `vpcompressd`/`vcompressps`: one compare gives a mask,
//      two masked compress-stores place the low and high halves at the two
//      ends of the free gap, and no element is ever written twice;
//    * four vectors are consumed per iteration and the side to read from is
//      chosen *branchlessly*.  That choice depends on how the data splits, so
//      a branch there mispredicts on roughly every other iteration -- at one
//      vector per iteration the misprediction alone costs more than the
//      partition (measured: 1 vector/branchy 0.0062 s vs 4 vectors/branchless
//      0.0042 s for 1M int32);
//    * the leaf (<=16 vectors) is a Batcher bitonic network run entirely in
//      registers, reusing the generic template from `parts/_network_body.inc`
//      with policies over the *native* type -- inside a quicksort the
//      comparisons are already in the value domain, so the encode and decode
//      passes the radix networks need are pure loss here.
//
//  Ordering semantics
//  ------------------
//  Integers compare natively.  Floating point uses `_CMP_GE_OQ`, which orders
//  everything except NaN and cannot separate -0 from +0, while the rest of the
//  library promises the radix total order (-0 before +0, NaN at one end).  The
//  entry point therefore screens the range first (a read-only SIMD pass) and
//  declines when it holds a NaN or a -0, leaving those inputs to the radix
//  path that already handles them.
// ============================================================================

#if FYX_HAS_AVX512_CODE

FYX_ISA_BEGIN("avx512f,avx512bw,avx512dq,avx512vl,avx512cd")
namespace fyx {
namespace detail {
namespace isa_avx512 {

// ---------------------------------------------------------------------------
// Per-type policies.  Everything is expressed in the native type: no encode.
// ---------------------------------------------------------------------------
// FYX_VQ_COMPRESS_TO_MEMORY=0 selects register-form compress + masked store:
// the memory form is microcoded on AMD Zen 4, but on Ice Lake-SP the register
// form measured ~15% slower, so the memory form stays the default.
// vfpclass categories whose hardware order differs from the radix total
// order: QNaN (0x01), -0 (0x04), SNaN (0x80).  One instruction instead of an
// unordered compare plus a -0 bit compare.
constexpr int kFpClassUnclean = 0x01 | 0x04 | 0x80;

FYX_FORCE_INLINE std::uint64_t bzhi_mask(unsigned c) {
    return c >= 64 ? ~0ull : ((1ull << c) - 1ull);
}


template <class T>
struct VOps;

template <>
struct VOps<std::int32_t> {
    using T    = std::int32_t;
    using reg  = __m512i;
    using mask = __mmask16;
    static constexpr int V = 16;
    FYX_FORCE_INLINE static reg  loadu(const T* p) { return _mm512_loadu_si512(reinterpret_cast<const void*>(p)); }
    FYX_FORCE_INLINE static void storeu(T* p, reg v) { _mm512_storeu_si512(reinterpret_cast<void*>(p), v); }
    FYX_FORCE_INLINE static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_epi32(k, p); }
    FYX_FORCE_INLINE static reg  mask_loadu(reg s, mask k, const T* p) { return _mm512_mask_loadu_epi32(s, k, p); }
    FYX_FORCE_INLINE static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_epi32(p, k, v); }
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) {
#if FYX_VQ_COMPRESS_TO_MEMORY
        _mm512_mask_compressstoreu_epi32(p, k, v);
#else
        _mm512_mask_storeu_epi32(p, static_cast<__mmask16>(bzhi_mask(popcount64(static_cast<std::uint64_t>(k)))), _mm512_maskz_compress_epi32(k, v));
#endif
    }
    FYX_FORCE_INLINE static mask ge(reg a, reg b) { return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLT); }
    FYX_FORCE_INLINE static mask gt(reg a, reg b) { return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLE); }
    FYX_FORCE_INLINE static reg  min(reg a, reg b) { return _mm512_min_epi32(a, b); }
    FYX_FORCE_INLINE static reg  max(reg a, reg b) { return _mm512_max_epi32(a, b); }
    FYX_FORCE_INLINE static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_epi32(s, k, a, b); }
    FYX_FORCE_INLINE static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_epi32(s, k, a, b); }
    FYX_FORCE_INLINE static reg  set1(T x) { return _mm512_set1_epi32(x); }
    FYX_FORCE_INLINE static T    reduce_min(reg v) { return _mm512_reduce_min_epi32(v); }
    FYX_FORCE_INLINE static T    reduce_max(reg v) { return _mm512_reduce_max_epi32(v); }
    FYX_FORCE_INLINE static T    hi() { return (std::numeric_limits<T>::max)(); }
    FYX_FORCE_INLINE static T    lo() { return (std::numeric_limits<T>::lowest)(); }
    template <unsigned J>
    FYX_FORCE_INLINE static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi32(
            int(15u ^ J), int(14u ^ J), int(13u ^ J), int(12u ^ J),
            int(11u ^ J), int(10u ^ J), int( 9u ^ J), int( 8u ^ J),
            int( 7u ^ J), int( 6u ^ J), int( 5u ^ J), int( 4u ^ J),
            int( 3u ^ J), int( 2u ^ J), int( 1u ^ J), int( 0u ^ J));
        return _mm512_permutexvar_epi32(idx, v);
    }
    FYX_FORCE_INLINE static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_epi32(keepmin, mx, mn); }
};

template <>
struct VOps<std::uint32_t> {
    using T    = std::uint32_t;
    using reg  = __m512i;
    using mask = __mmask16;
    static constexpr int V = 16;
    FYX_FORCE_INLINE static reg  loadu(const T* p) { return _mm512_loadu_si512(reinterpret_cast<const void*>(p)); }
    FYX_FORCE_INLINE static void storeu(T* p, reg v) { _mm512_storeu_si512(reinterpret_cast<void*>(p), v); }
    FYX_FORCE_INLINE static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_epi32(k, p); }
    FYX_FORCE_INLINE static reg  mask_loadu(reg s, mask k, const T* p) { return _mm512_mask_loadu_epi32(s, k, p); }
    FYX_FORCE_INLINE static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_epi32(p, k, v); }
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) {
#if FYX_VQ_COMPRESS_TO_MEMORY
        _mm512_mask_compressstoreu_epi32(p, k, v);
#else
        _mm512_mask_storeu_epi32(p, static_cast<__mmask16>(bzhi_mask(popcount64(static_cast<std::uint64_t>(k)))), _mm512_maskz_compress_epi32(k, v));
#endif
    }
    FYX_FORCE_INLINE static mask ge(reg a, reg b) { return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_NLT); }
    FYX_FORCE_INLINE static mask gt(reg a, reg b) { return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_NLE); }
    FYX_FORCE_INLINE static reg  min(reg a, reg b) { return _mm512_min_epu32(a, b); }
    FYX_FORCE_INLINE static reg  max(reg a, reg b) { return _mm512_max_epu32(a, b); }
    FYX_FORCE_INLINE static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_epu32(s, k, a, b); }
    FYX_FORCE_INLINE static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_epu32(s, k, a, b); }
    FYX_FORCE_INLINE static reg  set1(T x) { return _mm512_set1_epi32(static_cast<int>(x)); }
    FYX_FORCE_INLINE static T    reduce_min(reg v) { return static_cast<T>(_mm512_reduce_min_epu32(v)); }
    FYX_FORCE_INLINE static T    reduce_max(reg v) { return static_cast<T>(_mm512_reduce_max_epu32(v)); }
    FYX_FORCE_INLINE static T    hi() { return (std::numeric_limits<T>::max)(); }
    FYX_FORCE_INLINE static T    lo() { return (std::numeric_limits<T>::lowest)(); }
    template <unsigned J>
    FYX_FORCE_INLINE static reg permute_xor(reg v) { return VOps<std::int32_t>::permute_xor<J>(v); }
    FYX_FORCE_INLINE static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_epi32(keepmin, mx, mn); }
};

template <>
struct VOps<std::int64_t> {
    using T    = std::int64_t;
    using reg  = __m512i;
    using mask = __mmask8;
    static constexpr int V = 8;
    FYX_FORCE_INLINE static reg  loadu(const T* p) { return _mm512_loadu_si512(reinterpret_cast<const void*>(p)); }
    FYX_FORCE_INLINE static void storeu(T* p, reg v) { _mm512_storeu_si512(reinterpret_cast<void*>(p), v); }
    FYX_FORCE_INLINE static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_epi64(k, p); }
    FYX_FORCE_INLINE static reg  mask_loadu(reg s, mask k, const T* p) { return _mm512_mask_loadu_epi64(s, k, p); }
    FYX_FORCE_INLINE static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_epi64(p, k, v); }
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) {
#if FYX_VQ_COMPRESS_TO_MEMORY
        _mm512_mask_compressstoreu_epi64(p, k, v);
#else
        _mm512_mask_storeu_epi64(p, static_cast<__mmask8>(bzhi_mask(popcount64(static_cast<std::uint64_t>(k)))), _mm512_maskz_compress_epi64(k, v));
#endif
    }
    FYX_FORCE_INLINE static mask ge(reg a, reg b) { return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_NLT); }
    FYX_FORCE_INLINE static mask gt(reg a, reg b) { return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_NLE); }
    FYX_FORCE_INLINE static reg  min(reg a, reg b) { return _mm512_min_epi64(a, b); }
    FYX_FORCE_INLINE static reg  max(reg a, reg b) { return _mm512_max_epi64(a, b); }
    FYX_FORCE_INLINE static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_epi64(s, k, a, b); }
    FYX_FORCE_INLINE static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_epi64(s, k, a, b); }
    FYX_FORCE_INLINE static reg  set1(T x) { return _mm512_set1_epi64(x); }
    FYX_FORCE_INLINE static T    reduce_min(reg v) { return _mm512_reduce_min_epi64(v); }
    FYX_FORCE_INLINE static T    reduce_max(reg v) { return _mm512_reduce_max_epi64(v); }
    FYX_FORCE_INLINE static T    hi() { return (std::numeric_limits<T>::max)(); }
    FYX_FORCE_INLINE static T    lo() { return (std::numeric_limits<T>::lowest)(); }
    template <unsigned J>
    FYX_FORCE_INLINE static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi64(
            static_cast<long long>(7u ^ J), static_cast<long long>(6u ^ J),
            static_cast<long long>(5u ^ J), static_cast<long long>(4u ^ J),
            static_cast<long long>(3u ^ J), static_cast<long long>(2u ^ J),
            static_cast<long long>(1u ^ J), static_cast<long long>(0u ^ J));
        return _mm512_permutexvar_epi64(idx, v);
    }
    FYX_FORCE_INLINE static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_epi64(keepmin, mx, mn); }
};

template <>
struct VOps<std::uint64_t> {
    using T    = std::uint64_t;
    using reg  = __m512i;
    using mask = __mmask8;
    static constexpr int V = 8;
    FYX_FORCE_INLINE static reg  loadu(const T* p) { return _mm512_loadu_si512(reinterpret_cast<const void*>(p)); }
    FYX_FORCE_INLINE static void storeu(T* p, reg v) { _mm512_storeu_si512(reinterpret_cast<void*>(p), v); }
    FYX_FORCE_INLINE static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_epi64(k, p); }
    FYX_FORCE_INLINE static reg  mask_loadu(reg s, mask k, const T* p) { return _mm512_mask_loadu_epi64(s, k, p); }
    FYX_FORCE_INLINE static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_epi64(p, k, v); }
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) {
#if FYX_VQ_COMPRESS_TO_MEMORY
        _mm512_mask_compressstoreu_epi64(p, k, v);
#else
        _mm512_mask_storeu_epi64(p, static_cast<__mmask8>(bzhi_mask(popcount64(static_cast<std::uint64_t>(k)))), _mm512_maskz_compress_epi64(k, v));
#endif
    }
    FYX_FORCE_INLINE static mask ge(reg a, reg b) { return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_NLT); }
    FYX_FORCE_INLINE static mask gt(reg a, reg b) { return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_NLE); }
    FYX_FORCE_INLINE static reg  min(reg a, reg b) { return _mm512_min_epu64(a, b); }
    FYX_FORCE_INLINE static reg  max(reg a, reg b) { return _mm512_max_epu64(a, b); }
    FYX_FORCE_INLINE static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_epu64(s, k, a, b); }
    FYX_FORCE_INLINE static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_epu64(s, k, a, b); }
    FYX_FORCE_INLINE static reg  set1(T x) { return _mm512_set1_epi64(static_cast<long long>(x)); }
    FYX_FORCE_INLINE static T    reduce_min(reg v) { return _mm512_reduce_min_epu64(v); }
    FYX_FORCE_INLINE static T    reduce_max(reg v) { return _mm512_reduce_max_epu64(v); }
    FYX_FORCE_INLINE static T    hi() { return (std::numeric_limits<T>::max)(); }
    FYX_FORCE_INLINE static T    lo() { return (std::numeric_limits<T>::lowest)(); }
    template <unsigned J>
    FYX_FORCE_INLINE static reg permute_xor(reg v) { return VOps<std::int64_t>::permute_xor<J>(v); }
    FYX_FORCE_INLINE static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_epi64(keepmin, mx, mn); }
};

template <>
struct VOps<float> {
    using T    = float;
    using reg  = __m512;
    using mask = __mmask16;
    static constexpr int V = 16;
    FYX_FORCE_INLINE static reg  loadu(const T* p) { return _mm512_loadu_ps(p); }
    FYX_FORCE_INLINE static void storeu(T* p, reg v) { _mm512_storeu_ps(p, v); }
    FYX_FORCE_INLINE static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_ps(k, p); }
    FYX_FORCE_INLINE static reg  mask_loadu(reg s, mask k, const T* p) { return _mm512_mask_loadu_ps(s, k, p); }
    FYX_FORCE_INLINE static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_ps(p, k, v); }
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) {
#if FYX_VQ_COMPRESS_TO_MEMORY
        _mm512_mask_compressstoreu_ps(p, k, v);
#else
        _mm512_mask_storeu_ps(p, static_cast<__mmask16>(bzhi_mask(popcount64(static_cast<std::uint64_t>(k)))), _mm512_maskz_compress_ps(k, v));
#endif
    }
    FYX_FORCE_INLINE static mask ge(reg a, reg b) { return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ); }
    FYX_FORCE_INLINE static mask gt(reg a, reg b) { return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ); }
    FYX_FORCE_INLINE static reg  min(reg a, reg b) { return _mm512_min_ps(a, b); }
    FYX_FORCE_INLINE static reg  max(reg a, reg b) { return _mm512_max_ps(a, b); }
    FYX_FORCE_INLINE static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_ps(s, k, a, b); }
    FYX_FORCE_INLINE static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_ps(s, k, a, b); }
    FYX_FORCE_INLINE static reg  set1(T x) { return _mm512_set1_ps(x); }
    FYX_FORCE_INLINE static T    reduce_min(reg v) { return _mm512_reduce_min_ps(v); }
    FYX_FORCE_INLINE static T    reduce_max(reg v) { return _mm512_reduce_max_ps(v); }
    FYX_FORCE_INLINE static T    hi() { return (std::numeric_limits<T>::infinity)(); }
    FYX_FORCE_INLINE static T    lo() { return -(std::numeric_limits<T>::infinity)(); }
    template <unsigned J>
    FYX_FORCE_INLINE static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi32(
            int(15u ^ J), int(14u ^ J), int(13u ^ J), int(12u ^ J),
            int(11u ^ J), int(10u ^ J), int( 9u ^ J), int( 8u ^ J),
            int( 7u ^ J), int( 6u ^ J), int( 5u ^ J), int( 4u ^ J),
            int( 3u ^ J), int( 2u ^ J), int( 1u ^ J), int( 0u ^ J));
        return _mm512_permutexvar_ps(idx, v);
    }
    FYX_FORCE_INLINE static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_ps(keepmin, mx, mn); }
};

template <>
struct VOps<double> {
    using T    = double;
    using reg  = __m512d;
    using mask = __mmask8;
    static constexpr int V = 8;
    FYX_FORCE_INLINE static reg  loadu(const T* p) { return _mm512_loadu_pd(p); }
    FYX_FORCE_INLINE static void storeu(T* p, reg v) { _mm512_storeu_pd(p, v); }
    FYX_FORCE_INLINE static reg  maskz_loadu(mask k, const T* p) { return _mm512_maskz_loadu_pd(k, p); }
    FYX_FORCE_INLINE static reg  mask_loadu(reg s, mask k, const T* p) { return _mm512_mask_loadu_pd(s, k, p); }
    FYX_FORCE_INLINE static void mask_storeu(T* p, mask k, reg v) { _mm512_mask_storeu_pd(p, k, v); }
    FYX_FORCE_INLINE static void compressstore(T* p, mask k, reg v) {
#if FYX_VQ_COMPRESS_TO_MEMORY
        _mm512_mask_compressstoreu_pd(p, k, v);
#else
        _mm512_mask_storeu_pd(p, static_cast<__mmask8>(bzhi_mask(popcount64(static_cast<std::uint64_t>(k)))), _mm512_maskz_compress_pd(k, v));
#endif
    }
    FYX_FORCE_INLINE static mask ge(reg a, reg b) { return _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ); }
    FYX_FORCE_INLINE static mask gt(reg a, reg b) { return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ); }
    FYX_FORCE_INLINE static reg  min(reg a, reg b) { return _mm512_min_pd(a, b); }
    FYX_FORCE_INLINE static reg  max(reg a, reg b) { return _mm512_max_pd(a, b); }
    FYX_FORCE_INLINE static reg  mask_min(reg s, mask k, reg a, reg b) { return _mm512_mask_min_pd(s, k, a, b); }
    FYX_FORCE_INLINE static reg  mask_max(reg s, mask k, reg a, reg b) { return _mm512_mask_max_pd(s, k, a, b); }
    FYX_FORCE_INLINE static reg  set1(T x) { return _mm512_set1_pd(x); }
    FYX_FORCE_INLINE static T    reduce_min(reg v) { return _mm512_reduce_min_pd(v); }
    FYX_FORCE_INLINE static T    reduce_max(reg v) { return _mm512_reduce_max_pd(v); }
    FYX_FORCE_INLINE static T    hi() { return (std::numeric_limits<T>::infinity)(); }
    FYX_FORCE_INLINE static T    lo() { return -(std::numeric_limits<T>::infinity)(); }
    template <unsigned J>
    FYX_FORCE_INLINE static reg permute_xor(reg v) {
        const __m512i idx = _mm512_set_epi64(
            static_cast<long long>(7u ^ J), static_cast<long long>(6u ^ J),
            static_cast<long long>(5u ^ J), static_cast<long long>(4u ^ J),
            static_cast<long long>(3u ^ J), static_cast<long long>(2u ^ J),
            static_cast<long long>(1u ^ J), static_cast<long long>(0u ^ J));
        return _mm512_permutexvar_pd(idx, v);
    }
    FYX_FORCE_INLINE static reg blend(mask keepmin, reg mn, reg mx) { return _mm512_mask_blend_pd(keepmin, mx, mn); }
};

// ---------------------------------------------------------------------------
// Leaf: the generic bitonic network over native values.
// ---------------------------------------------------------------------------
template <class T>
struct VNetOps {
    using P    = VOps<T>;
    using Key  = T;
    using Vec  = typename P::reg;
    using Mask = typename P::mask;
    static constexpr unsigned kLanes = static_cast<unsigned>(P::V);

    FYX_FORCE_INLINE static Vec  load(const Key* p)   { return P::loadu(p); }
    FYX_FORCE_INLINE static void store(Key* p, Vec v) { P::storeu(p, v); }
    FYX_FORCE_INLINE static Vec  splat(Key k)         { return P::set1(k); }
    FYX_FORCE_INLINE static Vec  min(Vec a, Vec b)    { return P::min(a, b); }
    FYX_FORCE_INLINE static Vec  max(Vec a, Vec b)    { return P::max(a, b); }
    FYX_FORCE_INLINE static Vec  load_partial(const Key* p, unsigned n, Key fill) {
        const Mask m = static_cast<Mask>((1ull << n) - 1ull);
        return P::mask_loadu(P::set1(fill), m, p);
    }
    FYX_FORCE_INLINE static void store_partial(Key* p, Vec v, unsigned n) {
        P::mask_storeu(p, static_cast<Mask>((1ull << n) - 1ull), v);
    }
    template <unsigned J>
    FYX_FORCE_INLINE static Vec permute_xor(Vec v) { return P::template permute_xor<J>(v); }
    FYX_FORCE_INLINE static Vec blend(Mask keepmin, Vec mn, Vec mx) { return P::blend(keepmin, mn, mx); }
};

/// Like `network_sort_v`, but the padding sentinel comes from the policy (a
/// float network must pad with +inf, not FLT_MAX) and the leaf may hold up to
/// 16 vectors rather than 64 elements.
template <class T, unsigned NV>
FYX_FORCE_INLINE void vnet_sort_v(T* keys, std::size_t n) {
    using Ops = VNetOps<T>;
    using Vec = typename Ops::Vec;
    constexpr unsigned L = Ops::kLanes;
    const T sentinel = VOps<T>::hi();
    Vec v[NV];
    for (unsigned i = 0; i < NV; ++i) {
        const std::size_t off = static_cast<std::size_t>(i) * L;
        if (off + L <= n)  v[i] = Ops::load(keys + off);
        else if (off < n)  v[i] = Ops::load_partial(keys + off, static_cast<unsigned>(n - off), sentinel);
        else               v[i] = Ops::splat(sentinel);
    }
    BitonicVec<Ops, NV>::run(v);
    for (unsigned i = 0; i < NV; ++i) {
        const std::size_t off = static_cast<std::size_t>(i) * L;
        if (off + L <= n)  Ops::store(keys + off, v[i]);
        else if (off < n)  Ops::store_partial(keys + off, v[i], static_cast<unsigned>(n - off));
    }
}

// ---------------------------------------------------------------------------
// Column-network leaf (the layout vqsort uses): V registers x L lanes.
//
//   1. a size-optimal sorting network across the V registers sorts every
//      lane ("column") with min/max only -- no shuffles at all;
//   2. log2(L) merge levels join lane groups pairwise.  Each level is one
//      reversed compare (two lane permutes per register pair), lane
//      half-cleaners (one permute per register) and register half-cleaners
//      (plain min/max);
//   3. one in-register transpose restores memory order.
//
// For 16 x int32 (256 keys) that is ~1000 vector ops against ~1800 for the
// row-wise Batcher network above, whose every intra-vector stage pays a
// permute, a min, a max and a blend.
// ---------------------------------------------------------------------------
template <unsigned V> struct ColNet;
template <> struct ColNet<4> {
    static constexpr unsigned n = 5;
    static constexpr unsigned char a[n] = {0, 2, 0, 1, 1};
    static constexpr unsigned char b[n] = {1, 3, 2, 3, 2};
};
template <> struct ColNet<8> {
    static constexpr unsigned n = 19;
    static constexpr unsigned char a[n] = {0, 1, 4, 5, 0, 1, 2, 3, 0, 2, 4, 6, 2, 3, 1, 3, 1, 3, 5};
    static constexpr unsigned char b[n] = {2, 3, 6, 7, 4, 5, 6, 7, 1, 3, 5, 7, 4, 5, 4, 6, 2, 4, 6};
};
template <> struct ColNet<16> {   // 60 comparators, depth 10
    static constexpr unsigned n = 60;
    static constexpr unsigned char a[n] = {
        0, 1, 2, 3, 4, 5, 7, 9,   0, 1, 2, 3, 6, 8, 10, 11,   0, 2, 4, 6, 7, 10, 12, 14,
        0, 1, 4, 5, 6, 8, 12, 13,  1, 3, 4, 5, 8, 9, 13,      1, 2, 5, 7, 9, 11,
        2, 3, 9, 11,               3, 6, 7, 10,               3, 5, 7, 9, 11,   6, 8};
    static constexpr unsigned char b[n] = {
        13, 12, 15, 14, 8, 6, 11, 10,   5, 7, 9, 4, 13, 14, 15, 12,   1, 3, 5, 8, 9, 11, 13, 15,
        2, 3, 10, 11, 7, 9, 14, 15,      2, 12, 6, 7, 10, 11, 14,      4, 6, 8, 10, 13, 14,
        4, 6, 12, 13,                    5, 8, 9, 12,                  4, 6, 8, 10, 12,  7, 9};
};

template <> struct ColNet<32> {   // Batcher odd-even merge sort, 191 comparators
    static constexpr unsigned n = 191;
    static constexpr unsigned char a[n] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 1, 5, 9, 13, 17, 21, 25, 29, 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 2, 3, 10, 11, 18, 19, 26, 27, 1, 3, 5, 9, 11, 13, 17, 19, 21, 25, 27, 29, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 4, 5, 6, 7, 20, 21, 22, 23, 2, 3, 6, 7, 10, 11, 18, 19, 22, 23, 26, 27, 1, 3, 5, 7, 9, 11, 13, 17, 19, 21, 23, 25, 27, 29, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29};
    static constexpr unsigned char b[n] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 2, 6, 10, 14, 18, 22, 26, 30, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 4, 5, 12, 13, 20, 21, 28, 29, 2, 4, 6, 10, 12, 14, 18, 20, 22, 26, 28, 30, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 8, 9, 10, 11, 24, 25, 26, 27, 4, 5, 8, 9, 12, 13, 20, 21, 24, 25, 28, 29, 2, 4, 6, 8, 10, 12, 14, 18, 20, 22, 24, 26, 28, 30, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
};

template <class T>
FYX_FORCE_INLINE typename VOps<T>::reg vp2v(typename VOps<T>::reg a, __m512i idx,
                                            typename VOps<T>::reg b) {
    if constexpr (std::is_same<T, float>::value)       return _mm512_permutex2var_ps(a, idx, b);
    else if constexpr (std::is_same<T, double>::value) return _mm512_permutex2var_pd(a, idx, b);
    else if constexpr (sizeof(T) == 4)                 return _mm512_permutex2var_epi32(a, idx, b);
    else                                               return _mm512_permutex2var_epi64(a, idx, b);
}

/// Index vectors for one transpose round at block size B over L lanes.
template <unsigned L, unsigned B, bool Hi>
struct VTransIdx {
    using I = typename std::conditional<L == 16, std::int32_t, std::int64_t>::type;
    struct Tab { alignas(64) I v[L]; };
    static constexpr Tab make() {
        Tab t{};
        for (unsigned l = 0; l < L; ++l) {
            const bool up = (l & B) != 0;
            t.v[l] = static_cast<I>(Hi ? (up ? L + l : l + B) : (up ? L + (l - B) : l));
        }
        return t;
    }
    static constexpr Tab tab = make();
    FYX_FORCE_INLINE static __m512i load() {
        return _mm512_load_si512(reinterpret_cast<const void*>(tab.v));
    }
};

template <class T>
struct VColLeaf {
    using P   = VOps<T>;
    using reg = typename P::reg;
    using M   = typename P::mask;
    static constexpr unsigned L = static_cast<unsigned>(P::V);

    FYX_FORCE_INLINE static void ce(reg& x, reg& y) {
        const reg mn = P::min(x, y);
        y = P::max(x, y);
        x = mn;
    }
    static constexpr std::uint64_t low_lanes(unsigned bit) {
        std::uint64_t m = 0;
        for (unsigned l = 0; l < L; ++l) if (((l >> bit) & 1u) == 0) m |= 1ull << l;
        return m;
    }
    template <unsigned V, std::size_t... I>
    FYX_FORCE_INLINE static void columns(reg* v, std::index_sequence<I...>) {
        (ce(v[ColNet<V>::a[I]], v[ColNet<V>::b[I]]), ...);
    }
    template <unsigned V, unsigned K>
    FYX_FORCE_INLINE static void reversed_compare(reg* v) {
        constexpr unsigned Mx = (2u << K) - 1u;
        const M low = static_cast<M>(low_lanes(K));
        rc_each<V, Mx>(v, low, std::make_index_sequence<V / 2>{});
    }
    template <unsigned V, unsigned Mx, unsigned R>
    FYX_FORCE_INLINE static void rc_one(reg* v, M low) {
        const reg x = v[R];
        const reg s = P::template permute_xor<Mx>(v[V - 1 - R]);
        const reg mn = P::min(x, s);
        const reg mx = P::max(x, s);
        v[R]         = P::blend(low, mn, mx);
        v[V - 1 - R] = P::template permute_xor<Mx>(P::blend(low, mx, mn));
    }
    template <unsigned V, unsigned Mx, std::size_t... R>
    FYX_FORCE_INLINE static void rc_each(reg* v, M low, std::index_sequence<R...>) {
        (rc_one<V, Mx, static_cast<unsigned>(R)>(v, low), ...);
    }
    template <unsigned V, unsigned Bit>
    FYX_FORCE_INLINE static void lane_clean(reg* v) {
        const M low = static_cast<M>(low_lanes(Bit));
        lc_each<Bit>(v, low, std::make_index_sequence<V>{});
    }
    template <unsigned Bit, std::size_t... R>
    FYX_FORCE_INLINE static void lc_each(reg* v, M low, std::index_sequence<R...>) {
        ((v[R] = lc_one<Bit>(v[R], low)), ...);
    }
    template <unsigned Bit>
    FYX_FORCE_INLINE static reg lc_one(reg x, M low) {
        const reg q = P::template permute_xor<(1u << Bit)>(x);
        return P::blend(low, P::min(x, q), P::max(x, q));
    }
    template <unsigned V, unsigned K, unsigned Bit>
    FYX_FORCE_INLINE static void lane_cleans(reg* v) {
        if constexpr (Bit < K) {
            lane_cleans<V, K, Bit + 1>(v);       // larger bits first
            lane_clean<V, Bit>(v);
        }
    }
    template <unsigned V, unsigned J>
    FYX_FORCE_INLINE static void reg_cleans(reg* v) {
        if constexpr (J > 0) {
            rcl_each<J>(v, std::make_index_sequence<V>{});
            reg_cleans<V, J / 2>(v);
        }
    }
    template <unsigned J, std::size_t... R>
    FYX_FORCE_INLINE static void rcl_each(reg* v, std::index_sequence<R...>) {
        ((((R & J) == 0) ? ce(v[R], v[R | J]) : void()), ...);
    }
    template <unsigned V, unsigned K>
    FYX_FORCE_INLINE static void levels(reg* v) {
        if constexpr ((1u << K) < L) {
            reversed_compare<V, K>(v);
            lane_cleans<V, K, 0>(v);
            reg_cleans<V, V / 2>(v);
            levels<V, K + 1>(v);
        }
    }
    template <unsigned B, unsigned N = L>
    FYX_FORCE_INLINE static void transpose_round(reg* v) {
        if constexpr (B > 0) {
            const __m512i lo = VTransIdx<L, B, false>::load();
            const __m512i hi = VTransIdx<L, B, true>::load();
            tr_each<B>(v, lo, hi, std::make_index_sequence<N>{});
            transpose_round<B / 2, N>(v);
        }
    }

    template <unsigned B, std::size_t... R>
    FYX_FORCE_INLINE static void tr_each(reg* v, __m512i lo, __m512i hi, std::index_sequence<R...>) {
        ((((R & B) == 0) ? tr_one<B, static_cast<unsigned>(R)>(v, lo, hi) : void()), ...);
    }
    template <unsigned B, unsigned R>
    FYX_FORCE_INLINE static void tr_one(reg* v, __m512i lo, __m512i hi) {
        {
            const reg x = v[R], y = v[R + B];
            v[R]     = vp2v<T>(x, lo, y);
            v[R + B] = vp2v<T>(x, hi, y);
        }
    }

    FYX_FORCE_INLINE static reg load_one(const T* keys, std::size_t n, std::size_t off) {
        if (off + L <= n) return P::loadu(keys + off);
        const reg s = P::set1(P::hi());
        if (off < n) return P::mask_loadu(s, static_cast<M>((1ull << (n - off)) - 1ull), keys + off);
        return s;
    }
    FYX_FORCE_INLINE static void store_one(T* keys, std::size_t n, std::size_t off, reg x) {
        if (off + L <= n)  P::storeu(keys + off, x);
        else if (off < n)  P::mask_storeu(keys + off, static_cast<M>((1ull << (n - off)) - 1ull), x);
    }
    template <std::size_t... I>
    FYX_FORCE_INLINE static void load_all(reg* v, const T* keys, std::size_t n, std::index_sequence<I...>) {
        ((v[I] = load_one(keys, n, I * L)), ...);
    }
    // Logical order is lane-major: key (lane l, register r) sits at l*V + r.
    // After transposing each L x L register block c, its register l holds
    // output vector l*C + c.
    template <unsigned C, std::size_t... I>
    FYX_FORCE_INLINE static void store_all(const reg* v, T* keys, std::size_t n, std::index_sequence<I...>) {
        (store_one(keys, n, ((I % L) * C + I / L) * L, v[I]), ...);
    }

    // V == L/2: each register's halves are transposed as two independent
    // V x V blocks, then registers (2i, 2i+1) are recombined into output
    // vectors i and i + V/2.
    template <unsigned V, std::size_t... I>
    FYX_FORCE_INLINE static void store_half(const reg* v, T* keys, std::size_t n, std::index_sequence<I...>) {
        (store_one(keys, n, ((I % 2 == 0) ? I / 2 : I / 2 + V / 2) * L, v[I]), ...);
    }
    template <unsigned R, std::size_t... I>
    FYX_FORCE_INLINE static void pair_each(reg* v, __m512i lo, __m512i hi, std::index_sequence<I...>) {
        ((tr_pair(v[2 * I], v[2 * I + 1], lo, hi)), ...);
    }
    FYX_FORCE_INLINE static void tr_pair(reg& a, reg& b, __m512i lo, __m512i hi) {
        const reg x = a, y = b;
        a = vp2v<T>(x, lo, y);
        b = vp2v<T>(x, hi, y);
    }

    /// Sorts n <= V*L keys; V is a multiple of L, or exactly L/2.
    template <unsigned V>
    FYX_FORCE_INLINE static void sort(T* keys, std::size_t n) {
        static_assert(V % L == 0 || 2 * V == L || (4 * V == L && V == 4), "unsupported column leaf shape");
        reg v[V];
        load_all(v, keys, n, std::make_index_sequence<V>{});
        columns<V>(v, std::make_index_sequence<ColNet<V>::n>{});
        levels<V, 0>(v);
        if constexpr (4 * V == L) {
            // 4 registers x 16 lanes: transpose the 4x4 blocks inside each
            // lane quarter, then the 4x4 matrix of quarters.
            transpose_round<V / 2, V>(v);
            tr_each<2>(v, VTransIdx<L, 8, false>::load(), VTransIdx<L, 8, true>::load(),
                       std::make_index_sequence<V>{});
            tr_each<1>(v, VTransIdx<L, 4, false>::load(), VTransIdx<L, 4, true>::load(),
                       std::make_index_sequence<V>{});
            store_all<1>(v, keys, n, std::make_index_sequence<V>{});
            return;
        }
        if constexpr (2 * V == L) {
            transpose_round<V / 2, V>(v);
            pair_each<0>(v, VTransIdx<L, L / 2, false>::load(), VTransIdx<L, L / 2, true>::load(),
                         std::make_index_sequence<V / 2>{});
            store_half<V>(v, keys, n, std::make_index_sequence<V>{});
            return;
        }
        transpose_round<L / 2>(v);
        if constexpr (V > L)     transpose_round<L / 2>(v + L);
        if constexpr (V > 2 * L) transpose_round<L / 2>(v + 2 * L);
        if constexpr (V > 3 * L) transpose_round<L / 2>(v + 3 * L);
        static_assert(V <= 4 * L, "column leaf supports at most four blocks");
        store_all<V / L>(v, keys, n, std::make_index_sequence<V>{});
    }
};

#ifndef FYX_VQ_COLUMN_LEAF
#  define FYX_VQ_COLUMN_LEAF 1
#endif

/// Leaf sort for n <= 16 vectors' worth of elements.
template <class T>
inline void vnet_sort(T* a, std::size_t n) {
    constexpr unsigned L = static_cast<unsigned>(VOps<T>::V);
    if (n < 2) return;
    const std::size_t vecs = (static_cast<std::size_t>(next_pow2(n)) + L - 1) / L;
    switch (vecs) {
        case 1:  vnet_sort_v<T, 1>(a, n);  return;
        case 2:  vnet_sort_v<T, 2>(a, n);  return;
        case 4:
#if FYX_VQ_COLUMN_LEAF
            VColLeaf<T>::template sort<4>(a, n); return;
#endif
            vnet_sort_v<T, 4>(a, n);  return;
        case 8:
#if FYX_VQ_COLUMN_LEAF
            VColLeaf<T>::template sort<8>(a, n); return;
#endif
            vnet_sort_v<T, 8>(a, n);  return;
        case 16:
#if FYX_VQ_COLUMN_LEAF
            VColLeaf<T>::template sort<16>(a, n); return;
#endif
            vnet_sort_v<T, 16>(a, n); return;
#if FYX_VQ_COLUMN_LEAF
        case 32: VColLeaf<T>::template sort<32>(a, n); return;
#endif
        default: break;
    }
    pdqsort(a, a + n, std::less<T>());
}

// ---------------------------------------------------------------------------
// Partition
// ---------------------------------------------------------------------------
/// The high side takes `>= pivot` normally and `> pivot` in the Strict form,
/// which is what separates a pivot-valued block from the rest.
template <class T, bool Strict>
FYX_FORCE_INLINE typename VOps<T>::mask vhigh_mask(typename VOps<T>::reg curr,
                                                   typename VOps<T>::reg pivot) {
    if constexpr (Strict) return VOps<T>::gt(curr, pivot);
    else                  return VOps<T>::ge(curr, pivot);
}

template <class T, bool Strict>
FYX_FORCE_INLINE void vpart_vec(T* a, std::size_t& ls, std::size_t& rs,
                                typename VOps<T>::reg curr, typename VOps<T>::reg pivot,
                                typename VOps<T>::reg& vmin, typename VOps<T>::reg& vmax) {
    using P = VOps<T>;
    const typename P::mask gek = vhigh_mask<T, Strict>(curr, pivot);
    const int gc = static_cast<int>(popcount64(static_cast<std::uint64_t>(gek)));
    P::compressstore(a + ls, static_cast<typename P::mask>(~gek), curr);
    P::compressstore(a + rs - static_cast<std::size_t>(gc), gek, curr);
    ls += static_cast<std::size_t>(P::V - gc);
    rs -= static_cast<std::size_t>(gc);
    vmin = P::min(vmin, curr);
    vmax = P::max(vmax, curr);
}

template <class T, bool Strict>
FYX_FORCE_INLINE void vpart_vec_masked(T* a, std::size_t& ls, std::size_t& rs,
                                       typename VOps<T>::reg curr, typename VOps<T>::reg pivot,
                                       typename VOps<T>::mask valid,
                                       typename VOps<T>::reg& vmin, typename VOps<T>::reg& vmax) {
    using P = VOps<T>;
    const typename P::mask gek =
        static_cast<typename P::mask>(vhigh_mask<T, Strict>(curr, pivot) & valid);
    const typename P::mask ltk = static_cast<typename P::mask>((~gek) & valid);
    const int gc = static_cast<int>(popcount64(static_cast<std::uint64_t>(gek)));
    const int lc = static_cast<int>(popcount64(static_cast<std::uint64_t>(ltk)));
    P::compressstore(a + ls, ltk, curr);
    P::compressstore(a + rs - static_cast<std::size_t>(gc), gek, curr);
    ls += static_cast<std::size_t>(lc);
    rs -= static_cast<std::size_t>(gc);
    vmin = P::mask_min(vmin, valid, vmin, curr);
    vmax = P::mask_max(vmax, valid, vmax, curr);
}

/// Scalar partition for ranges too small to hold the two boundary vectors.
template <class T, bool Strict>
inline std::size_t vpartition_scalar(T* a, std::size_t n, T pivot, T& out_min, T& out_max) {
    T mn = VOps<T>::hi();
    T mx = VOps<T>::lo();
    for (std::size_t i = 0; i < n; ++i) {
        const T x = a[i];
        mn = x < mn ? x : mn;
        mx = mx < x ? x : mx;
    }
    std::size_t i = 0, j = n;
    while (i < j) {
        const bool low = Strict ? !(pivot < a[i]) : (a[i] < pivot);
        if (low) { ++i; continue; }
        --j;
        const T t = a[i]; a[i] = a[j]; a[j] = t;
    }
    out_min = mn;
    out_max = mx;
    return i;
}

/// In-place partition of `a[0,n)` around `pivot`: everything `< pivot` moves
/// to the front, everything `>= pivot` to the back, and the count of the
/// former is returned.  `out_min`/`out_max` are the extremes of the whole
/// range, which is what lets the caller prune a side that is all pivot.
///
/// `U` vectors are consumed per iteration; the side read from is selected
/// without a branch (see the file header).
template <class T, bool Strict = false, int U = 4>
inline std::size_t vpartition(T* a, std::size_t n, T pivot, T& out_min, T& out_max) {
    using P   = VOps<T>;
    using reg = typename P::reg;
    constexpr int V = P::V;
    constexpr std::size_t CH = static_cast<std::size_t>(U) * V;

    if (n < 2 * CH) {
        if constexpr (U > 1) return vpartition<T, Strict, U / 2>(a, n, pivot, out_min, out_max);
        else                 return vpartition_scalar<T, Strict>(a, n, pivot, out_min, out_max);
    }

    const reg pv = P::set1(pivot);
    reg vmin = P::set1(P::hi());
    reg vmax = P::set1(P::lo());

    std::size_t l = 0, r = n;      // unread region [l, r)
    std::size_t ls = 0, rs = n;    // free gap: [ls, l) on the left, [r, rs) on the right

    // Lift the U boundary vectors on each side into registers to make room.
    reg vl[U], vr[U];
    for (int i = 0; i < U; ++i) vl[i] = P::loadu(a + static_cast<std::size_t>(i) * V);
    for (int i = 0; i < U; ++i) vr[i] = P::loadu(a + n - static_cast<std::size_t>(i + 1) * V);
    l += CH;
    r -= CH;

    // Reading from the side with less free space keeps at least CH free on
    // both sides, which is what a worst-case split of one iteration needs.
    while (r - l >= CH) {
        const bool from_right = (rs - r) < (l - ls);
        const std::size_t src = from_right ? (r - CH) : l;
        reg cur[U];
        for (int i = 0; i < U; ++i) cur[i] = P::loadu(a + src + static_cast<std::size_t>(i) * V);
        r -= from_right ? CH : 0;
        l += from_right ? 0 : CH;
        for (int i = 0; i < U; ++i) vpart_vec<T, Strict>(a, ls, rs, cur[i], pv, vmin, vmax);
    }
    while (r - l >= static_cast<std::size_t>(V)) {
        const bool from_right = (rs - r) < (l - ls);
        const std::size_t src = from_right ? (r - V) : l;
        const reg cur = P::loadu(a + src);
        r -= from_right ? V : 0;
        l += from_right ? 0 : V;
        vpart_vec<T, Strict>(a, ls, rs, cur, pv, vmin, vmax);
    }
    if (r != l) {  // 0 < r - l < V; once these are read the gap is contiguous
        const unsigned m = static_cast<unsigned>(r - l);
        const typename P::mask valid = static_cast<typename P::mask>((1ull << m) - 1ull);
        const reg cur = P::maskz_loadu(valid, a + l);
        vpart_vec_masked<T, Strict>(a, ls, rs, cur, pv, valid, vmin, vmax);
    }
    for (int i = 0; i < U; ++i) vpart_vec<T, Strict>(a, ls, rs, vl[i], pv, vmin, vmax);
    for (int i = 0; i < U; ++i) vpart_vec<T, Strict>(a, ls, rs, vr[i], pv, vmin, vmax);

    out_min = P::reduce_min(vmin);
    out_max = P::reduce_max(vmax);
    return ls;
}

template <class T, bool Strict>
FYX_FORCE_INLINE void vpart_vec_lean(T* a, std::size_t& ls, std::size_t& rs,
                                     typename VOps<T>::reg curr, typename VOps<T>::reg pivot) {
    using P = VOps<T>;
    const typename P::mask gek = vhigh_mask<T, Strict>(curr, pivot);
    const int gc = static_cast<int>(popcount64(static_cast<std::uint64_t>(gek)));
    P::compressstore(a + ls, static_cast<typename P::mask>(~gek), curr);
    P::compressstore(a + rs - static_cast<std::size_t>(gc), gek, curr);
    ls += static_cast<std::size_t>(P::V - gc);
    rs -= static_cast<std::size_t>(gc);
}

template <class T, bool Strict>
FYX_FORCE_INLINE void vpart_vec_lean_masked(T* a, std::size_t& ls, std::size_t& rs,
                                            typename VOps<T>::reg curr, typename VOps<T>::reg pivot,
                                            typename VOps<T>::mask valid) {
    using P = VOps<T>;
    const typename P::mask gek =
        static_cast<typename P::mask>(vhigh_mask<T, Strict>(curr, pivot) & valid);
    const typename P::mask ltk = static_cast<typename P::mask>((~gek) & valid);
    const int gc = static_cast<int>(popcount64(static_cast<std::uint64_t>(gek)));
    const int lc = static_cast<int>(popcount64(static_cast<std::uint64_t>(ltk)));
    P::compressstore(a + ls, ltk, curr);
    P::compressstore(a + rs - static_cast<std::size_t>(gc), gek, curr);
    ls += static_cast<std::size_t>(lc);
    rs -= static_cast<std::size_t>(gc);
}

template <class T, bool Strict>
FYX_FORCE_INLINE void vpart_vec_lean_max(T* a, std::size_t& ls, std::size_t& rs,
                                         typename VOps<T>::reg curr, typename VOps<T>::reg pivot,
                                         typename VOps<T>::reg& vmax) {
    using P = VOps<T>;
    const typename P::mask gek = vhigh_mask<T, Strict>(curr, pivot);
    const int gc = static_cast<int>(popcount64(static_cast<std::uint64_t>(gek)));
    P::compressstore(a + ls, static_cast<typename P::mask>(~gek), curr);
    P::compressstore(a + rs - static_cast<std::size_t>(gc), gek, curr);
    ls += static_cast<std::size_t>(P::V - gc);
    rs -= static_cast<std::size_t>(gc);
    vmax = P::max(vmax, curr);
}

/// Lean partition for the serial recursion: like `vpartition`, but it tracks
/// only the range maximum, not the minimum (one vector op per partitioned
/// vector instead of two).  The minimum is redundant: `split == 0` already
/// proves the pivot is the range minimum, for free.  The maximum pays for
/// itself on duplicate-heavy inputs: `pivot == max` marks the whole range one
/// value (return, no extra pass) and marks the high side all-pivot (its
/// elements are in their final place; the recursion drops it without sorting
/// it), which costs two full passes to discover structurally.
/// NaN or -0 lanes of a vector (the values whose hardware order differs from
/// the library's radix total order); always 0 for integers.
template <class T>
FYX_FORCE_INLINE std::uint64_t vbad_lanes(typename VOps<T>::reg v) {
    if constexpr (!std::is_floating_point<T>::value) {
        (void)v;
        return 0;
    } else if constexpr (sizeof(T) == 4) {
        return static_cast<std::uint64_t>(_mm512_fpclass_ps_mask(v, kFpClassUnclean));
    } else {
        return static_cast<std::uint64_t>(_mm512_fpclass_pd_mask(v, kFpClassUnclean));
    }
}

template <class T>
inline bool vrange_clean(const T* a, std::size_t n);

template <class T, bool Strict = false, int U = 4, bool TrackMax = false, bool CC = false>
inline std::size_t vpartition_lean(T* a, std::size_t n, T pivot, T& out_max,
                                   std::uint64_t* bad_out = nullptr) {
    using P   = VOps<T>;
    using reg = typename P::reg;
    constexpr int V = P::V;
    constexpr std::size_t CH = static_cast<std::size_t>(U) * V;

    if (n < 2 * CH) {
        if constexpr (U > 1) return vpartition_lean<T, Strict, U / 2, TrackMax, CC>(a, n, pivot, out_max, bad_out);
        T mn, mx;
        if constexpr (U == 1) {
            if constexpr (CC) *bad_out |= vrange_clean(a, n) ? 0u : 1u;
            const std::size_t s = vpartition_scalar<T, Strict>(a, n, pivot, mn, mx);
            if constexpr (TrackMax) out_max = mx;
            return s;
        }
    }

    const reg pv = P::set1(pivot);

    std::size_t l = 0, r = n;
    std::size_t ls = 0, rs = n;

    reg vmax;
    if constexpr (TrackMax) vmax = P::set1(pivot);
    std::uint64_t bad = 0;                 // CC: NaN / -0 lanes seen

    reg vl[U], vr[U];
    FYX_VQ_UNROLL for (int i = 0; i < U; ++i) vl[i] = P::loadu(a + static_cast<std::size_t>(i) * V);
    FYX_VQ_UNROLL for (int i = 0; i < U; ++i) vr[i] = P::loadu(a + n - static_cast<std::size_t>(i + 1) * V);
    l += CH;
    r -= CH;

    while (r - l >= CH) {
        const bool from_right = (rs - r) < (l - ls);
        const std::size_t src = from_right ? (r - CH) : l;
        reg cur[U];
        FYX_VQ_UNROLL for (int i = 0; i < U; ++i) cur[i] = P::loadu(a + src + static_cast<std::size_t>(i) * V);
        r -= from_right ? CH : 0;
        l += from_right ? 0 : CH;
        if constexpr (CC) { FYX_VQ_UNROLL for (int i = 0; i < U; ++i) bad |= vbad_lanes<T>(cur[i]); }
        if constexpr (TrackMax) {
            FYX_VQ_UNROLL for (int i = 0; i < U; ++i) vpart_vec_lean_max<T, Strict>(a, ls, rs, cur[i], pv, vmax);
        } else {
            FYX_VQ_UNROLL for (int i = 0; i < U; ++i) vpart_vec_lean<T, Strict>(a, ls, rs, cur[i], pv);
        }
    }
    while (r - l >= static_cast<std::size_t>(V)) {
        const bool from_right = (rs - r) < (l - ls);
        const std::size_t src = from_right ? (r - V) : l;
        const reg cur = P::loadu(a + src);
        r -= from_right ? V : 0;
        l += from_right ? 0 : V;
        if constexpr (CC) bad |= vbad_lanes<T>(cur);
        if constexpr (TrackMax) vpart_vec_lean_max<T, Strict>(a, ls, rs, cur, pv, vmax);
        else                    vpart_vec_lean<T, Strict>(a, ls, rs, cur, pv);
    }
    if (r != l) {
        const unsigned m = static_cast<unsigned>(r - l);
        const typename P::mask valid = static_cast<typename P::mask>((1ull << m) - 1ull);
        const reg cur = P::maskz_loadu(valid, a + l);   // zero lanes are +0: clean
        if constexpr (CC) bad |= vbad_lanes<T>(cur);
        // The masked variant keeps the junk lanes out of the counts and the
        // stores; the max update runs separately over a copy whose junk lanes
        // hold the pivot (<= every right-side value, and the pivot itself is
        // in the range, so this cannot exceed the true maximum).
        if constexpr (TrackMax) {
            vpart_vec_lean_masked<T, Strict>(a, ls, rs, cur, pv, valid);
            vmax = P::max(vmax, P::blend(valid, cur, pv));
        } else {
            vpart_vec_lean_masked<T, Strict>(a, ls, rs, cur, pv, valid);
        }
    }
    FYX_VQ_UNROLL for (int i = 0; i < U; ++i) {
        if constexpr (TrackMax) vpart_vec_lean_max<T, Strict>(a, ls, rs, vl[i], pv, vmax);
        else                    vpart_vec_lean<T, Strict>(a, ls, rs, vl[i], pv);
    }
    FYX_VQ_UNROLL for (int i = 0; i < U; ++i) {
        if constexpr (TrackMax) vpart_vec_lean_max<T, Strict>(a, ls, rs, vr[i], pv, vmax);
        else                    vpart_vec_lean<T, Strict>(a, ls, rs, vr[i], pv);
    }
    if constexpr (CC) {
        FYX_VQ_UNROLL for (int i = 0; i < U; ++i) bad |= vbad_lanes<T>(vl[i]) | vbad_lanes<T>(vr[i]);
        *bad_out |= bad;
    }
    if constexpr (TrackMax) out_max = P::reduce_max(vmax);
    return ls;
}

/// Pivot: two vectors' worth of strided samples, sorted in registers.
template <class T>
inline T vpick_pivot(const T* a, std::size_t n) {
#ifndef FYX_VQ_PIVOT_VECS
    constexpr int S = 2 * VOps<T>::V;
#else
    constexpr int S = FYX_VQ_PIVOT_VECS * VOps<T>::V;
#endif
    T s[S];
    const std::size_t step = n / S;
    for (int i = 0; i < S; ++i) s[i] = a[static_cast<std::size_t>(i) * step + (step >> 1)];
    vnet_sort<T>(s, static_cast<std::size_t>(S));
    return s[S / 2];
}

/// vpick_pivot plus `uniform`: every sample equal (then the range is likely
/// single-valued -- vall_equal_to decides with one read-only pass).
template <class T>
inline T vpick_pivot_u(const T* a, std::size_t n, bool& uniform) {
#ifndef FYX_VQ_PIVOT_VECS
    constexpr int S = 2 * VOps<T>::V;
#else
    constexpr int S = FYX_VQ_PIVOT_VECS * VOps<T>::V;
#endif
    T s[S];
    const std::size_t step = n / S;
    for (int i = 0; i < S; ++i) s[i] = a[static_cast<std::size_t>(i) * step + (step >> 1)];
    vnet_sort<T>(s, static_cast<std::size_t>(S));
    uniform = !(s[0] < s[S - 1]);
    return s[S / 2];
}

/// Every element equal to x (early exit on the first mismatching vector).
template <class T>
FYX_FORCE_INLINE bool vall_equal_to(const T* a, std::size_t n, T x) {
    using P = VOps<T>;
    using reg = typename P::reg;
    constexpr std::size_t V = static_cast<std::size_t>(P::V);
    const reg vx = P::set1(x);
    std::size_t i = 0;
    for (; i + 4 * V <= n; i += 4 * V) {
        const reg a0 = P::loadu(a + i), a1 = P::loadu(a + i + V);
        const reg a2 = P::loadu(a + i + 2 * V), a3 = P::loadu(a + i + 3 * V);
        // x <= v <= x per lane, folded: max(v) and min(v) against x
        const reg mx = P::max(P::max(a0, a1), P::max(a2, a3));
        const reg mn = P::min(P::min(a0, a1), P::min(a2, a3));
        if (P::gt(mx, vx) | P::gt(vx, mn)) return false;
    }
    for (; i < n; ++i)
        if (a[i] < x || x < a[i]) return false;
    return true;
}

/// Leaf size: 32 vectors, i.e. 512 elements for 4-byte types and 256 for
/// 8-byte ones.  With the column-network leaf (VColLeaf) a 32-register leaf
/// beat 16 registers for every width in the standalone kernel benchmark; the
/// row-wise Batcher leaf it replaced was best at 16.
#ifndef FYX_VQ_LEAF_VECS
#  define FYX_VQ_LEAF_VECS 32
#endif
template <class T>
struct VqLeaf {
    static constexpr std::size_t value = static_cast<std::size_t>(FYX_VQ_LEAF_VECS) * static_cast<std::size_t>(VOps<T>::V);
};

/// Leaf screen for duplicate-heavy inputs: one min/max pass (in L1).  A
/// single-valued leaf is done; a two-valued one (every key equals the min
/// or the max) is rewritten from its count.  Few-distinct inputs (a few
/// hundred keys, each repeated hundreds of times) end most partitions in
/// such leaves, where the sorting network did thousands of vector ops for
/// nothing.  True: the leaf is sorted.  Needs n >= V.
template <class T>
FYX_FORCE_INLINE bool vleaf_few_values(T* a, std::size_t n) {
    using P = VOps<T>;
    using reg = typename P::reg;
    constexpr std::size_t V = static_cast<std::size_t>(P::V);
    reg mn = P::loadu(a + n - V), mx = mn;
    for (std::size_t i = 0; i + V <= n; i += V) {
        const reg v = P::loadu(a + i);
        mn = P::min(mn, v);
        mx = P::max(mx, v);
    }
    const T lo = P::reduce_min(mn), hi = P::reduce_max(mx);
    if (!(lo < hi)) return true;
    const reg vlo = P::set1(lo), vhi = P::set1(hi);
    std::size_t nlo = 0, i = 0;
    for (; i + V <= n; i += V) {
        const reg v = P::loadu(a + i);
        // a middle value (lo < v < hi) means three or more keys
        if (P::gt(v, vlo) & P::gt(vhi, v)) return false;
        nlo += popcount64(static_cast<std::uint64_t>(static_cast<typename P::mask>(~P::gt(v, vlo))));
    }
    for (; i < n; ++i) {
        if (lo < a[i] && a[i] < hi) return false;
        nlo += !(lo < a[i]);
    }
    std::size_t k = 0;
    for (; k + V <= nlo; k += V) P::storeu(a + k, vlo);
    for (; k < nlo; ++k) a[k] = lo;
    for (; k < n && (k % V) != 0; ++k) a[k] = hi;
    for (; k + V <= n; k += V) P::storeu(a + k, vhi);
    for (; k < n; ++k) a[k] = hi;
    return true;
}

template <class T>
inline void vqsort_rec(T* a, std::size_t n, int budget) {
    while (n > VqLeaf<T>::value) {
        if (budget <= 0) {           // pathological splits: hand over to pdqsort
            pdqsort(a, a + n, std::less<T>());
            return;
        }
        --budget;
        bool uniform = false;
        const T pivot = vpick_pivot_u<T>(a, n, uniform);
        // All samples equal: on few-distinct inputs most ranges below the
        // top levels are single-valued, and a read-only check is far cheaper
        // than the partition pass that would otherwise discover it.
        if (uniform && vall_equal_to<T>(a, n, pivot)) return;
        // Lean partition: the hot loop tracks only the range maximum (one
        // vector op per partitioned vector, not the two that min+max cost).
        // The minimum is redundant -- `split == 0` proves the pivot is the
        // range minimum for free.  The maximum keeps the two cheap exits the
        // tracked partition had: a range whose every element equals the pivot
        // is finished, and a high side made entirely of pivot values is in
        // its final place and is dropped without being recursed into.  On
        // duplicate-heavy inputs those two cases are common, and discovering
        // them structurally costs two extra passes each.
        T hi;
        std::size_t split = vpartition_lean<T, false, 4, true>(a, n, pivot, hi);
        if (split == 0) {
            // The pivot is the range minimum, so the low side came out empty
            // and splitting there again would not move.
            if (!(pivot < hi)) return;            // the whole range is one value
            // Re-partition with the strict test instead: that puts the
            // pivot-valued elements -- at least one, and they are already in
            // their final place -- in front of everything else, so the range
            // always shrinks.
            T hi2;
            const std::size_t eq = vpartition_lean<T, true>(a, n, pivot, hi2);
            a += eq;
            n -= eq;
            continue;
        }
        if (!(pivot < hi)) {          // the high side is all pivot: already done
            n = split;
            continue;
        }
        if (split < n - split) {      // recurse into the smaller side
            vqsort_rec<T>(a, split, budget);
            a += split;
            n -= split;
        } else {
            vqsort_rec<T>(a + split, n - split, budget);
            n = split;
        }
    }
    if (n >= 4 * static_cast<std::size_t>(VOps<T>::V) && vleaf_few_values<T>(a, n)) return;
    vnet_sort<T>(a, n);
}

/// vqsort_rec whose first partition also screens for NaN / -0 (the screen
/// rides on loads the partition makes anyway, instead of a separate pass).
/// Returns false -- the range then holds a permutation of its input -- when
/// one was seen; the caller sorts it another way.
template <class T>
inline bool vqsort_rec_checked(T* a, std::size_t n, int budget) {
    if (n <= VqLeaf<T>::value) {
        if (!vrange_clean(a, n)) return false;
        vnet_sort<T>(a, n);
        return true;
    }
    const T pivot = vpick_pivot<T>(a, n);
    T hi;
    std::uint64_t bad = 0;
    const std::size_t split = vpartition_lean<T, false, 4, true, true>(a, n, pivot, hi, &bad);
    if (bad) return false;
    --budget;
    if (split == 0) {
        if (!(pivot < hi)) return true;
        T hi2;
        const std::size_t eq = vpartition_lean<T, true>(a, n, pivot, hi2);
        vqsort_rec<T>(a + eq, n - eq, budget);
        return true;
    }
    vqsort_rec<T>(a, split, budget);
    if (pivot < hi) vqsort_rec<T>(a + split, n - split, budget);
    return true;
}

/// True when the range holds no NaN and no negative zero, i.e. when the
/// hardware's floating-point order is the same total order the radix path
/// would have produced.  Integers are always clean.
template <class T>
inline bool vrange_clean(const T* a, std::size_t n) {
    if constexpr (!std::is_floating_point<T>::value) {
        (void)a; (void)n;
        return true;
    } else if constexpr (sizeof(T) == 4) {
        const std::size_t V = 16;
        __mmask16 bad = 0;
        std::size_t i = 0;
        for (; i + V <= n; i += V) {
            const __m512 v = _mm512_loadu_ps(a + i);
            bad = static_cast<__mmask16>(bad | _mm512_fpclass_ps_mask(v, kFpClassUnclean));
            if (bad) return false;
        }
        if (i < n) {
            const __mmask16 k = static_cast<__mmask16>((1u << (n - i)) - 1u);
            const __m512 v = _mm512_maskz_loadu_ps(k, a + i);
            bad = static_cast<__mmask16>(bad | (k & _mm512_fpclass_ps_mask(v, kFpClassUnclean)));
        }
        return bad == 0;
    } else {
        const std::size_t V = 8;
        __mmask8 bad = 0;
        std::size_t i = 0;
        for (; i + V <= n; i += V) {
            const __m512d v = _mm512_loadu_pd(a + i);
            bad = static_cast<__mmask8>(bad | _mm512_fpclass_pd_mask(v, kFpClassUnclean));
            if (bad) return false;
        }
        if (i < n) {
            const __mmask8 k = static_cast<__mmask8>((1u << (n - i)) - 1u);
            const __m512d v = _mm512_maskz_loadu_pd(k, a + i);
            bad = static_cast<__mmask8>(bad | (k & _mm512_fpclass_pd_mask(v, kFpClassUnclean)));
        }
        return bad == 0;
    }
}

template <class T>
FYX_FORCE_INLINE typename VOps<T>::mask vunclean_lanes(typename VOps<T>::reg v) {
    using M = typename VOps<T>::mask;
    if constexpr (std::is_same<T, float>::value) {
        return static_cast<M>(_mm512_fpclass_ps_mask(v, kFpClassUnclean));
    } else if constexpr (std::is_same<T, double>::value) {
        return static_cast<M>(_mm512_fpclass_pd_mask(v, kFpClassUnclean));
    } else {
        (void)v;
        return 0;
    }
}

/// Lanes 1..V-1 of cur followed by lane 0 of nx: the right neighbours of
/// cur's lanes from a single new load (valignd / valignq).
template <class T, class R>
FYX_FORCE_INLINE R vnext_lanes(R nx, R cur) {
    if constexpr (std::is_same<T, float>::value)
        return _mm512_castsi512_ps(_mm512_alignr_epi32(_mm512_castps_si512(nx), _mm512_castps_si512(cur), 1));
    else if constexpr (std::is_same<T, double>::value)
        return _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(nx), _mm512_castpd_si512(cur), 1));
    else if constexpr (sizeof(T) == 4)
        return _mm512_alignr_epi32(nx, cur, 1);
    else
        return _mm512_alignr_epi64(nx, cur, 1);
}

// 256-bit (EVEX ymm) lanes for streaming order scans past L2: a 512-bit
// loop entered after a few hundred microseconds of non-512-bit work (a copy,
// the caller's own code) runs its first ~50 us at reduced throughput while
// the upper vector half powers up -- 1M sorted int32 measured 186 us cold
// against 128 for the same scan on ymm.  Memory-bound, the narrower lanes
// cost nothing there; inside L2 the warm 512-bit loop stays faster.  Half
// a typical L2: 100k uint64 (800 KB, batch-copied) 26.6 -> 21.8 us on ymm;
// 100k int32 (400 KB) stays faster at 512 bits.
#ifndef FYX_PRESCAN_YMM_BYTES
#define FYX_PRESCAN_YMM_BYTES (std::size_t(1) << 19)
#endif
inline constexpr std::size_t kPrescanYmmBytes = FYX_PRESCAN_YMM_BYTES;

// Register type by (floating, width) via specialisation: std::conditional
// over vector types trips -Wignored-attributes.
template <bool F, std::size_t W> struct YReg { using type = __m256i; };
template <> struct YReg<true, 4> { using type = __m256; };
template <> struct YReg<true, 8> { using type = __m256d; };

// Order flags without compare-into-mask (port 5 only on Ice Lake, like
// valign): min(v, w) ^ v is nonzero exactly where v > w (max: v < w);
// min / max / xor / or issue on three ports.
template <class T> struct YOps {
    static constexpr bool fp = std::is_floating_point<T>::value;
    static constexpr bool w4 = sizeof(T) == 4;
    static constexpr std::size_t V = 32 / sizeof(T);
    using reg = typename YReg<fp, sizeof(T)>::type;
    FYX_FORCE_INLINE static __m256i bits(__m256i v) { return v; }
    FYX_FORCE_INLINE static __m256i bits(__m256 v) { return _mm256_castps_si256(v); }
    FYX_FORCE_INLINE static __m256i bits(__m256d v) { return _mm256_castpd_si256(v); }
    FYX_FORCE_INLINE static __m256 vmin(__m256 a, __m256 b) { return _mm256_min_ps(a, b); }
    FYX_FORCE_INLINE static __m256 vmax(__m256 a, __m256 b) { return _mm256_max_ps(a, b); }
    FYX_FORCE_INLINE static __m256d vmin(__m256d a, __m256d b) { return _mm256_min_pd(a, b); }
    FYX_FORCE_INLINE static __m256d vmax(__m256d a, __m256d b) { return _mm256_max_pd(a, b); }
    FYX_FORCE_INLINE static __m256i vmin(__m256i a, __m256i b) {
        if constexpr (w4) return std::is_signed<T>::value ? _mm256_min_epi32(a, b) : _mm256_min_epu32(a, b);
        else return std::is_signed<T>::value ? _mm256_min_epi64(a, b) : _mm256_min_epu64(a, b);
    }
    FYX_FORCE_INLINE static __m256i vmax(__m256i a, __m256i b) {
        if constexpr (w4) return std::is_signed<T>::value ? _mm256_max_epi32(a, b) : _mm256_max_epu32(a, b);
        else return std::is_signed<T>::value ? _mm256_max_epi64(a, b) : _mm256_max_epu64(a, b);
    }
    FYX_FORCE_INLINE static reg loadu(const T* p) {
        if constexpr (fp && w4) return _mm256_loadu_ps(reinterpret_cast<const float*>(p));
        else if constexpr (fp) return _mm256_loadu_pd(reinterpret_cast<const double*>(p));
        else return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    // lane order reversed (a member: lambdas do not inherit the target)
    FYX_FORCE_INLINE static reg rev(reg x) {
        if constexpr (w4) {
            const __m256i idx = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            if constexpr (fp) return _mm256_permutevar8x32_ps(x, idx);
            else return _mm256_permutevar8x32_epi32(x, idx);
        } else {
            if constexpr (fp) return _mm256_permute4x64_pd(x, 0x1B);
            else return _mm256_permute4x64_epi64(x, 0x1B);
        }
    }
    FYX_FORCE_INLINE static __mmask8 unclean(reg v) {
        if constexpr (fp && w4) return _mm256_fpclass_ps_mask(v, kFpClassUnclean);
        else if constexpr (fp) return _mm256_fpclass_pd_mask(v, kFpClassUnclean);
        else { (void)v; return 0; }
    }
};

/// Single-stream ymm order scan of the pairs (j, j + 1), j in [from, to),
/// to <= n - 1.  Accumulates into up / dn; on a mixed block stores
/// settled = block start, prior = flags before it, returns false.
/// Floats need no NaN / -0 screen here: min / max return their second
/// operand on unordered or +-0 pairs, so a pair of differing bit patterns
/// that is neither < nor > raises both flags.  Hence "only up" means every
/// pair is bitwise equal or increasing in key order (the library's order),
/// and the sorted / reverse / all-equal verdicts are exact without bit 2.
template <class T>
inline bool vprescan_ymm_run(const T* a, std::size_t from, std::size_t to, unsigned& up,
                             unsigned& dn, std::size_t* settled, unsigned* prior) {
    using Y = YOps<T>;
    constexpr std::size_t L = Y::V, BL = 8 * L;
    constexpr bool fp = std::is_floating_point<T>::value;
    std::size_t i = from;
    for (; i + BL <= to; i += BL) {
        __m256i xu = _mm256_setzero_si256(), xd = xu;
        FYX_VQ_UNROLL for (std::size_t k = 0; k < BL; k += L) {
            const auto v = Y::loadu(a + i + k);
            const auto w = Y::loadu(a + i + k + 1);
            const __m256i bv = Y::bits(v);
            xd = _mm256_or_si256(xd, _mm256_xor_si256(Y::bits(Y::vmin(v, w)), bv));   // v > w somewhere
            xu = _mm256_or_si256(xu, _mm256_xor_si256(Y::bits(Y::vmax(v, w)), bv));   // v < w somewhere
        }
        const unsigned before = (up ? 1u : 0u) | (dn ? 2u : 0u);
        up |= _mm256_testz_si256(xu, xu) ? 0u : 1u;
        dn |= _mm256_testz_si256(xd, xd) ? 0u : 1u;
        if (up && dn) {
            if (settled) *settled = i;
            if (prior) *prior = before;
            return false;
        }
    }
    for (; i < to; ++i) {
        const unsigned before = (up ? 1u : 0u) | (dn ? 2u : 0u);
        const bool lt = a[i] < a[i + 1], gt = a[i + 1] < a[i];
        up |= lt ? 1u : 0u;
        dn |= gt ? 1u : 0u;
        if constexpr (fp)   // unordered / +-0 pair of differing bits: both, as min / max above
            if (!lt && !gt && std::memcmp(&a[i], &a[i + 1], sizeof(T)) != 0) up = dn = 1u;
        if (up && dn) {
            if (settled) *settled = i;
            if (prior) *prior = before;
            return false;
        }
    }
    return true;
}

/// [lo, n) non-increasing in sort order (no p[k] < p[k+1] ascending, no
/// p[k] > p[k+1] descending)?  Then reverse it, checking and swapping blocks
/// from both ends in one pass (ymm: streaming, no 512-bit warm-up).
/// 0: reversed.  1: order break, input restored.  2 (floats): a violation in
/// hardware order -- possibly only a NaN / +-0 pair, which min / max flag
/// like a break -- so the caller's key-order scalar path decides; input
/// restored.  No NaN / -0 screen otherwise: a pass means every pair is
/// bitwise equal or strictly ordered, i.e. ordered in key order too.
/// Undo = false: on 1 / 2 the blocks already swapped stay swapped (still a
/// permutation of the input -- for callers that sort it anyway).  A template
/// parameter, so the few-run merge's instance is unchanged code.
template <class T, bool Undo = true>
inline int vreverse_tail_checked(T* p, std::size_t lo, std::size_t n, bool descending) {
    using Y = YOps<T>;
    constexpr std::size_t L = Y::V, B = 8 * L;
    constexpr bool fp = std::is_floating_point<T>::value;
    std::size_t l = lo, r = n;
    auto undo = [&]() {
        if constexpr (!Undo) return;
        for (std::size_t k = 0; k < l - lo; ++k) std::swap(p[lo + k], p[n - 1 - k]);
    };
    while (r - l >= 2 * B) {
        __m256i x = _mm256_setzero_si256();
        FYX_VQ_UNROLL for (std::size_t k = 0; k < B; k += L) {
            const auto v1 = Y::loadu(p + l + k), w1 = Y::loadu(p + l + k + 1);
            const auto v2 = Y::loadu(p + r - B - 1 + k), w2 = Y::loadu(p + r - B + k);
            // ascending sort forbids v < w (max(v, w) != v), descending v > w
            const auto m1 = descending ? Y::vmin(v1, w1) : Y::vmax(v1, w1);
            const auto m2 = descending ? Y::vmin(v2, w2) : Y::vmax(v2, w2);
            x = _mm256_or_si256(x, _mm256_or_si256(_mm256_xor_si256(Y::bits(m1), Y::bits(v1)),
                                                   _mm256_xor_si256(Y::bits(m2), Y::bits(v2))));
        }
        if (!_mm256_testz_si256(x, x)) { undo(); return fp ? 2 : 1; }
        FYX_VQ_UNROLL for (std::size_t k = 0; k < B; k += L) {
            const auto a = Y::loadu(p + l + k), b = Y::loadu(p + r - L - k);
            if constexpr (fp && sizeof(T) == 4) {
                _mm256_storeu_ps(reinterpret_cast<float*>(p + l + k), Y::rev(b));
                _mm256_storeu_ps(reinterpret_cast<float*>(p + r - L - k), Y::rev(a));
            } else if constexpr (fp) {
                _mm256_storeu_pd(reinterpret_cast<double*>(p + l + k), Y::rev(b));
                _mm256_storeu_pd(reinterpret_cast<double*>(p + r - L - k), Y::rev(a));
            } else {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + l + k), Y::rev(b));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + r - L - k), Y::rev(a));
            }
        }
        l += B;
        r -= B;
    }
    for (std::size_t k = l; k + 1 < r; ++k) {
        const bool bad = descending ? p[k + 1] < p[k] : p[k] < p[k + 1];
        if constexpr (fp) {
            const bool ord = descending ? p[k] < p[k + 1] : p[k + 1] < p[k];
            if (bad || (!ord && std::memcmp(&p[k], &p[k + 1], sizeof(T)) != 0)) { undo(); return 2; }
        } else if (bad) { undo(); return 1; }
    }
    std::reverse(p + l, p + r);
    return 0;
}

/// vsmall_prescan on ymm lanes (same result bits, same settled / prior).
/// Past L2 the scan is fetch-bound: four interleaved quarter streams keep
/// more prefetches in flight (1M uint64 sorted, Ice Lake: 262 -> 237 us; one
/// stream at 512 bits after a copy: ~268 us, 512-bit warm-up penalty).  On a
/// mixed block the exact sequential settled / prior is rebuilt from the
/// per-stream flags, scanning only the unverified gaps.
template <class T>
inline unsigned vsmall_prescan_ymm(const T* a, std::size_t n, std::size_t* settled, unsigned* prior) {
    using Y = YOps<T>;
    constexpr std::size_t L = Y::V, SB = 4 * L;
    constexpr bool fp = std::is_floating_point<T>::value;
    // Bit 2 never set (see vprescan_ymm_run): a full scan's order bits are
    // exact in key order; a mixed float range reports 11 (cleanliness left to
    // the checked quicksort).
    constexpr unsigned kMixed = fp ? 11u : 3u;
    const std::size_t q = ((n - 1) / 4) / SB * SB;
    __m256i U[4], D[4];
    for (int t = 0; t < 4; ++t) U[t] = D[t] = _mm256_setzero_si256();
    std::size_t i = 0;
    for (; i < q; i += SB) {
        __m256i nu[4], nd[4];
        FYX_VQ_UNROLL for (int t = 0; t < 4; ++t) {
            const T* p = a + std::size_t(t) * q + i;
            nu[t] = U[t]; nd[t] = D[t];
            FYX_VQ_UNROLL for (std::size_t k = 0; k < SB; k += L) {
                const auto v = Y::loadu(p + k);
                const auto w = Y::loadu(p + k + 1);
                const __m256i bv = Y::bits(v);
                nd[t] = _mm256_or_si256(nd[t], _mm256_xor_si256(Y::bits(Y::vmin(v, w)), bv));
                nu[t] = _mm256_or_si256(nu[t], _mm256_xor_si256(Y::bits(Y::vmax(v, w)), bv));
            }
        }
        const __m256i au = _mm256_or_si256(_mm256_or_si256(nu[0], nu[1]), _mm256_or_si256(nu[2], nu[3]));
        const __m256i ad = _mm256_or_si256(_mm256_or_si256(nd[0], nd[1]), _mm256_or_si256(nd[2], nd[3]));
        if (!_mm256_testz_si256(au, au) && !_mm256_testz_si256(ad, ad)) break;
        for (int t = 0; t < 4; ++t) { U[t] = nu[t]; D[t] = nd[t]; }
    }
    // Sequential rebuild: stream t verified [t*q, t*q + i); gaps scanned.
    unsigned up = 0, dn = 0;
    for (std::size_t t = 0; t < 4; ++t) {
        const std::size_t s0 = t * q, e = (t == 3) ? n - 1 : (t + 1) * q;
        if (i > 0) {
            const unsigned fu = _mm256_testz_si256(U[t], U[t]) ? 0u : 1u;
            const unsigned fd = _mm256_testz_si256(D[t], D[t]) ? 0u : 1u;
            if (((up | fu) != 0) && ((dn | fd) != 0)) {
                if (!vprescan_ymm_run<T>(a, s0, s0 + i, up, dn, settled, prior)) return kMixed;
            } else {
                up |= fu;
                dn |= fd;
            }
        }
        if (!vprescan_ymm_run<T>(a, s0 + i, e, up, dn, settled, prior)) return kMixed;
    }
    return (up ? 1u : 0u) | (dn ? 2u : 0u);
}

template <class T>
inline unsigned vsmall_prescan(const T* a, std::size_t n, std::size_t* settled = nullptr,
                               unsigned* prior = nullptr) {
    if (n * sizeof(T) >= kPrescanYmmBytes) return vsmall_prescan_ymm<T>(a, n, settled, prior);
    using P = VOps<T>;
    using M = typename P::mask;
    constexpr std::size_t L = static_cast<std::size_t>(P::V);
    constexpr bool fp = std::is_floating_point<T>::value;
    M up = 0, dn = 0, bad = 0;
    std::size_t i = 0;
    if (n >= 2) {
        // Blocks of 8 vectors, settled test once per block: the per-vector
        // test-and-branch held a sorted scan ~20-25% under the plain stream
        // (100k int32: 15.6 -> 10.5 us; 1M uint64: 332 -> 267 us).
        // One load per vector: the neighbour vector is cur shifted by a lane
        // with the next load's first lane (100k uint64: 27 -> 25 us).
        constexpr std::size_t BL = 8 * L;
        auto cur = P::loadu(a);
        for (; i + BL + L <= n; i += BL) {
            M bu = 0, bd = 0, bb = 0;
            FYX_VQ_UNROLL for (std::size_t k = 0; k < BL; k += L) {
                const auto nx = P::loadu(a + i + k + L);
                const auto w = vnext_lanes<T>(nx, cur);
                bu = static_cast<M>(bu | P::gt(w, cur));
                bd = static_cast<M>(bd | P::gt(cur, w));
                if constexpr (fp) bb = static_cast<M>(bb | vunclean_lanes<T>(cur));
                cur = nx;
            }
            const unsigned before = (up ? 1u : 0u) | (dn ? 2u : 0u);
            up  = static_cast<M>(up | bu);
            dn  = static_cast<M>(dn | bd);
            bad = static_cast<M>(bad | bb);
            if (up && dn) {
                if (settled) *settled = i;
                if (prior) *prior = before;
                if constexpr (!fp) return 3u;
                else return bad ? 7u : 11u;
            }
        }
        for (; i + L + 1 <= n; i += L) {
            const auto v = P::loadu(a + i);
            const auto w = P::loadu(a + i + 1);
            const unsigned before = (up ? 1u : 0u) | (dn ? 2u : 0u);
            up  = static_cast<M>(up | P::gt(w, v));
            dn  = static_cast<M>(dn | P::gt(v, w));
            bad = static_cast<M>(bad | vunclean_lanes<T>(v));
            if (up && dn) {
                // Pairs (j, j+1), j < i, saw only the `before` direction(s).
                if (settled) *settled = i;
                if (prior) *prior = before;
                // Order is settled; what is left is the cleanliness scan.
                // Floating point: the rest of the cleanliness scan is left to
                // the caller (bit 3), which needs it only if the structural
                // probes decline.
                if constexpr (!fp) return 3u;
                else return bad ? 7u : 11u;
            }
        }
        // Remaining pairs (j, j+1) for j in [i, n-1): fewer than L + 1 keys.
        const std::size_t k = n - 1 - i;            // pairs left, <= L
        const M m = static_cast<M>(k >= L ? ~0ull : ((1ull << k) - 1ull));
        const auto v = P::maskz_loadu(m, a + i);
        const auto w = P::maskz_loadu(m, a + i + 1);
        up  = static_cast<M>(up | (P::gt(w, v) & m));
        dn  = static_cast<M>(dn | (P::gt(v, w) & m));
        bad = static_cast<M>(bad | (vunclean_lanes<T>(v) & m));
        i += k;                                      // a[n-1] not scanned yet
    }
    if constexpr (fp) {
        if (n > 0 && i < n) {
            const std::size_t r = n - i;             // 1 key (the last)
            const M m = static_cast<M>((1ull << r) - 1ull);
            bad = static_cast<M>(bad | (vunclean_lanes<T>(P::maskz_loadu(m, a + i)) & m));
        }
    }
    return (up ? 1u : 0u) | (dn ? 2u : 0u) | (bad ? 4u : 0u);
}



// ---------------------------------------------------------------------------
// Two-run merge on radix keys: a 2V-element bitonic merge network per output
// vector (one vector of carry against the next vector from whichever run has
// the smaller head).  ~0.4 ns/key for int32 on Ice Lake-SP against ~1.5 for
// the best scalar merge.  Works on encoded keys, so NaN / -0 / descending
// follow the radix total order exactly.
// ---------------------------------------------------------------------------
template <class T>
struct VMergeKeys {
    static constexpr bool k64 = sizeof(T) == 8;
    static constexpr int  V   = k64 ? 8 : 16;
    using Key = typename std::conditional<k64, std::uint64_t, std::uint32_t>::type;
    FYX_FORCE_INLINE static __m512i sign() {
        return k64 ? _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL))
                   : _mm512_set1_epi32(static_cast<int>(0x80000000u));
    }
    FYX_FORCE_INLINE static __m512i srai_top(__m512i x) {
        if constexpr (k64) return _mm512_srai_epi64(x, 63); else return _mm512_srai_epi32(x, 31);
    }
    FYX_FORCE_INLINE static __m512i enc(__m512i x, __m512i flip) {
        if constexpr (std::is_floating_point<T>::value)
            x = _mm512_xor_si512(x, _mm512_or_si512(srai_top(x), sign()));
        else if constexpr (std::is_signed<T>::value)
            x = _mm512_xor_si512(x, sign());
        return _mm512_xor_si512(x, flip);
    }
    FYX_FORCE_INLINE static __m512i dec(__m512i k, __m512i flip) {
        k = _mm512_xor_si512(k, flip);
        if constexpr (std::is_floating_point<T>::value)
            k = _mm512_xor_si512(k, _mm512_or_si512(_mm512_andnot_si512(srai_top(k), _mm512_set1_epi32(-1)), sign()));
        else if constexpr (std::is_signed<T>::value)
            k = _mm512_xor_si512(k, sign());
        return k;
    }
    FYX_FORCE_INLINE static __m512i vmin(__m512i a, __m512i b) {
        if constexpr (k64) return _mm512_min_epu64(a, b); else return _mm512_min_epu32(a, b);
    }
    FYX_FORCE_INLINE static __m512i vmax(__m512i a, __m512i b) {
        if constexpr (k64) return _mm512_max_epu64(a, b); else return _mm512_max_epu32(a, b);
    }
    FYX_FORCE_INLINE static __m512i stage(__m512i v, __m512i q, unsigned m) {
        if constexpr (k64) return _mm512_mask_blend_epi64(static_cast<__mmask8>(m), vmin(v, q), vmax(v, q));
        else return _mm512_mask_blend_epi32(static_cast<__mmask16>(m), vmin(v, q), vmax(v, q));
    }
    // sort one bitonic vector
    FYX_FORCE_INLINE static __m512i bitonic(__m512i v) {
        if constexpr (k64) {
            v = stage(v, _mm512_shuffle_i64x2(v, v, 0x4E), 0xF0u);
            v = stage(v, _mm512_permutex_epi64(v, 0x4E), 0xCCu);
            v = stage(v, _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(0x4E)), 0xAAu);
        } else {
            v = stage(v, _mm512_shuffle_i32x4(v, v, 0x4E), 0xFF00u);
            v = stage(v, _mm512_shuffle_i32x4(v, v, 0xB1), 0xF0F0u);
            v = stage(v, _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(0x4E)), 0xCCCCu);
            v = stage(v, _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(0xB1)), 0xAAAAu);
        }
        return v;
    }
    FYX_FORCE_INLINE static void merge2(__m512i a, __m512i b, __m512i& lo, __m512i& hi) {
        const __m512i rev = k64 ? _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7)
                                : _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        b  = k64 ? _mm512_permutexvar_epi64(rev, b) : _mm512_permutexvar_epi32(rev, b);
        lo = bitonic(vmin(a, b));
        hi = bitonic(vmax(a, b));
    }
};

template <class T>
struct VMergeChain {
    const T* A; std::size_t na, ia;
    const T* B; std::size_t nb, ib;
    T* out; std::size_t o;
    __m512i hi;
};

template <class T>
FYX_FORCE_INLINE void vmerge_chain_init(VMergeChain<T>& c, __m512i flip) {
    using M = VMergeKeys<T>;
    constexpr std::size_t V = M::V;
    __m512i lo;
    M::merge2(M::enc(_mm512_loadu_si512(reinterpret_cast<const void*>(c.A)), flip),
              M::enc(_mm512_loadu_si512(reinterpret_cast<const void*>(c.B)), flip), lo, c.hi);
    _mm512_storeu_si512(reinterpret_cast<void*>(c.out), M::dec(lo, flip));
    c.ia = V; c.ib = V; c.o = V;
}

template <class T, class KeyF>
FYX_FORCE_INLINE bool vmerge_chain_live(const VMergeChain<T>& c, KeyF) {
    constexpr std::size_t V = VMergeKeys<T>::V;
    return c.ia + V <= c.na && c.ib + V <= c.nb;
}

template <class T, class KeyF>
FYX_FORCE_INLINE void vmerge_chain_step(VMergeChain<T>& c, __m512i flip, KeyF key) {
    using M = VMergeKeys<T>;
    constexpr std::size_t V = M::V;
    // Branch-free source select: which run feeds the next vector is a coin
    // flip on interleaved data, so a branch here mispredicts every other step.
    const bool fa = key(c.A[c.ia]) < key(c.B[c.ib]);
    const std::uintptr_t msk = std::uintptr_t(0) - static_cast<std::uintptr_t>(fa);
    const std::uintptr_t src = (reinterpret_cast<std::uintptr_t>(c.A + c.ia) & msk) |
                               (reinterpret_cast<std::uintptr_t>(c.B + c.ib) & ~msk);
    const __m512i nx = _mm512_loadu_si512(reinterpret_cast<const void*>(src));
    c.ia += V & msk;
    c.ib += V & ~msk;
    __m512i lo;
    M::merge2(c.hi, M::enc(nx, flip), lo, c.hi);
    _mm512_storeu_si512(reinterpret_cast<void*>(c.out + c.o), M::dec(lo, flip));
    c.o += V;
}

template <class T, class KeyF>
inline void vmerge_chain_finish(VMergeChain<T>& c, __m512i flip, KeyF key) {
    using M = VMergeKeys<T>;
    constexpr std::size_t V = M::V;
    while (vmerge_chain_live(c, key)) vmerge_chain_step(c, flip, key);
    // carry (V keys) + the short rest -> S (< 2V keys); then S against the
    // long rest by binary search and block copies.
    alignas(64) T carry[V];
    _mm512_store_si512(reinterpret_cast<void*>(carry), M::dec(c.hi, flip));
    const bool a_short = c.ia + V > c.na;
    const T* sh = a_short ? c.A + c.ia : c.B + c.ib;
    const std::size_t nsh = a_short ? c.na - c.ia : c.nb - c.ib;
    const T* lg = a_short ? c.B + c.ib : c.A + c.ia;
    const std::size_t nlg = a_short ? c.nb - c.ib : c.na - c.ia;
    T S[2 * V];
    std::size_t ns = 0, ic = 0, is = 0;
    while (ic < V && is < nsh) S[ns++] = key(sh[is]) < key(carry[ic]) ? sh[is++] : carry[ic++];
    while (ic < V) S[ns++] = carry[ic++];
    while (is < nsh) S[ns++] = sh[is++];
    T* out = c.out;
    std::size_t o = c.o, il = 0;
    for (std::size_t j = 0; j < ns; ++j) {
        const auto ks = key(S[j]);
        std::size_t lo2 = il, hi2 = nlg;                // first long key > S[j]
        while (lo2 < hi2) {
            const std::size_t mid = lo2 + (hi2 - lo2) / 2;
            if (ks < key(lg[mid])) hi2 = mid; else lo2 = mid + 1;
        }
        std::memcpy(static_cast<void*>(out + o), static_cast<const void*>(lg + il), (lo2 - il) * sizeof(T));
        o += lo2 - il;
        il = lo2;
        out[o++] = S[j];
    }
    std::memcpy(static_cast<void*>(out + o), static_cast<const void*>(lg + il), (nlg - il) * sizeof(T));
}

// The network of one step is a ~20-cycle dependency chain on the carry, so
// the output is cut at co-ranks into independent chains that are stepped
// round-robin (the out-of-order core overlaps them).
template <class T, class KeyF>
inline void vmerge_runs_impl(const T* A, std::size_t na, const T* B, std::size_t nb, T* out,
                             bool descending, KeyF key) {
    using M = VMergeKeys<T>;
    constexpr std::size_t V = M::V;
    constexpr int C = FYX_VMERGE_CHAINS;
    const __m512i flip = descending ? _mm512_set1_epi32(-1) : _mm512_setzero_si512();
    const std::size_t N = na + nb;
    VMergeChain<T> ch[C];
    std::size_t sa_prev = 0, sb_prev = 0;
    bool split_ok = true;
    for (int c = 0; c < C; ++c) {
        std::size_t sa = na, sb = nb;
        if (c + 1 < C) {
            const std::size_t sidx = N / C * static_cast<std::size_t>(c + 1);
            std::size_t lo = sidx > nb ? sidx - nb : 0, hi = sidx < na ? sidx : na;
            while (lo < hi) {                       // smallest i with B[s-i-1] < A[i]
                const std::size_t i = lo + (hi - lo) / 2;
                const std::size_t j = sidx - i;
                if (j > 0 && !(key(B[j - 1]) < key(A[i]))) lo = i + 1; else hi = i;
            }
            sa = lo; sb = sidx - lo;
        }
        ch[c].A = A + sa_prev; ch[c].na = sa - sa_prev;
        ch[c].B = B + sb_prev; ch[c].nb = sb - sb_prev;
        ch[c].out = out + sa_prev + sb_prev;
        if (ch[c].na < V || ch[c].nb < V) split_ok = false;
        sa_prev = sa; sb_prev = sb;
    }
    if (!split_ok) {
        VMergeChain<T> one{A, na, 0, B, nb, 0, out, 0, _mm512_setzero_si512()};
        vmerge_chain_init(one, flip);
        vmerge_chain_finish(one, flip, key);
        return;
    }
    for (int c = 0; c < C; ++c) vmerge_chain_init(ch[c], flip);
    for (;;) {
        bool live = true;
        for (int c = 0; c < C; ++c) live = live && vmerge_chain_live(ch[c], key);
        if (!live) break;
        FYX_VQ_UNROLL for (int c = 0; c < C; ++c) vmerge_chain_step(ch[c], flip, key);
    }
    for (int c = 0; c < C; ++c) vmerge_chain_finish(ch[c], flip, key);
}


// ---------------------------------------------------------------------------
// Few distinct keys (<= 32): count each table entry with one compare + one
// masked add per vector, then write the runs.  O(n K / V) and no data
// movement besides the final fill.  Keys are matched bitwise (exact for
// -0 / +0 and NaN payloads); the caller orders the table by radix key.
// Returns false (range untouched) when some key is not in the table.
// ---------------------------------------------------------------------------
template <class T, int KB>
inline bool vfew_count(const T* p, std::size_t n, const T* vals, unsigned K, std::uint64_t* cnt) {
    constexpr bool k64 = sizeof(T) == 8;
    constexpr std::size_t V = k64 ? 8 : 16;
    using U = typename std::conditional<k64, std::uint64_t, std::uint32_t>::type;
    U bits[KB];
    for (int t = 0; t < KB; ++t) {
        T x = vals[t < static_cast<int>(K) ? t : 0];
        std::memcpy(&bits[t], &x, sizeof(U));
    }
    std::uint64_t total = 0;
    std::size_t i = 0;
    // lane counters are 32-bit: flush every 2^20 vectors at the latest
    while (i + V <= n) {
        const std::size_t stop = std::min(n - (n - i) % V, i + (std::size_t(1) << 20) * V);
        __m512i acc[KB];
        for (int t = 0; t < KB; ++t) acc[t] = _mm512_setzero_si512();
        for (; i < stop; i += V) {
            const __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(p + i));
            FYX_VQ_UNROLL for (int t = 0; t < KB; ++t) {
                if constexpr (k64) {
                    const __mmask8 m = _mm512_cmpeq_epi64_mask(v, _mm512_set1_epi64(static_cast<long long>(bits[t])));
                    acc[t] = _mm512_mask_sub_epi64(acc[t], m, acc[t], _mm512_set1_epi64(-1));
                } else {
                    const __mmask16 m = _mm512_cmpeq_epi32_mask(v, _mm512_set1_epi32(static_cast<int>(bits[t])));
                    acc[t] = _mm512_mask_sub_epi32(acc[t], m, acc[t], _mm512_set1_epi32(-1));
                }
            }
        }
        for (int t = 0; t < static_cast<int>(K); ++t) {
            std::uint64_t c;
            if constexpr (k64) c = static_cast<std::uint64_t>(_mm512_reduce_add_epi64(acc[t]));
            else c = static_cast<std::uint64_t>(static_cast<std::uint32_t>(_mm512_reduce_add_epi32(acc[t])));
            cnt[t] += c;
            total += c;
        }
    }
    for (; i < n; ++i) {
        U x;
        std::memcpy(&x, p + i, sizeof(U));
        unsigned t = 0;
        while (t < K && bits[t] != x) ++t;
        if (t == K) return false;
        ++cnt[t];
        ++total;
    }
    return total == n;
}

template <class T>
inline void vfill(T* p, std::size_t n, T x) {
    __m512i v;
    if constexpr (sizeof(T) == 8) { std::uint64_t b; std::memcpy(&b, &x, 8); v = _mm512_set1_epi64(static_cast<long long>(b)); }
    else { std::uint32_t b; std::memcpy(&b, &x, 4); v = _mm512_set1_epi32(static_cast<int>(b)); }
    constexpr std::size_t V = 64 / sizeof(T);
    std::size_t i = 0;
    for (; i + V <= n; i += V) _mm512_storeu_si512(reinterpret_cast<void*>(p + i), v);
    for (; i < n; ++i) p[i] = x;
}

/// Writes runs vals[t] x cnt[t] back to back from p (sum of cnt = n).  Each
/// run is stored in whole vectors that may spill into the next run's slots
/// (the next run overwrites them); only the final run is trimmed to p + n.
/// One loop exit per run instead of a vector loop plus a scalar tail.
template <class T, class C>
inline void vfill_runs(T* p, std::size_t n, const T* vals, const C* cnt, std::size_t m) {
    constexpr std::size_t V = 64 / sizeof(T);
    T* const end = p + n;
    for (std::size_t t = 0; t < m; ++t) {
        __m512i v;
        if constexpr (sizeof(T) == 8) { std::uint64_t b; std::memcpy(&b, &vals[t], 8); v = _mm512_set1_epi64(static_cast<long long>(b)); }
        else { std::uint32_t b; std::memcpy(&b, &vals[t], 4); v = _mm512_set1_epi32(static_cast<int>(b)); }
        T* const stop = p + cnt[t];
        if (static_cast<std::size_t>(end - p) >= V + static_cast<std::size_t>(stop - p)) {
            T* q = p;
            do { _mm512_storeu_si512(reinterpret_cast<void*>(q), v); q += V; } while (q < stop);
        } else {
            vfill(p, static_cast<std::size_t>(stop - p), vals[t]);
        }
        p = stop;
    }
}

// Vector helpers as plain functions: a lambda returning __m512i gets no
// target attribute from the ISA pragma and trips -Wpsabi in generic builds.
template <class U>
FYX_FORCE_INLINE __m512i vbcast_bits(U b) {
    if constexpr (sizeof(U) == 8) return _mm512_set1_epi64(static_cast<long long>(b));
    else return _mm512_set1_epi32(static_cast<int>(b));
}
template <class T>
FYX_FORCE_INLINE __m512i vencode_radix(__m512i v) {
    constexpr bool k64 = sizeof(T) == 8;
    if constexpr (std::is_floating_point<T>::value) {
        if constexpr (k64) {
            const __m512i m = _mm512_or_si512(_mm512_srai_epi64(v, 63), _mm512_set1_epi64(static_cast<long long>(1ull << 63)));
            return _mm512_xor_si512(v, m);
        } else {
            const __m512i m = _mm512_or_si512(_mm512_srai_epi32(v, 31), _mm512_set1_epi32(static_cast<int>(0x80000000u)));
            return _mm512_xor_si512(v, m);
        }
    } else if constexpr (std::is_signed<T>::value) {
        if constexpr (k64) return _mm512_xor_si512(v, _mm512_set1_epi64(static_cast<long long>(1ull << 63)));
        else return _mm512_xor_si512(v, _mm512_set1_epi32(static_cast<int>(0x80000000u)));
    } else {
        return v;
    }
}

// Distinct-key table from strided samples: returns K, or 0 when the first 64
// samples already show more than `cap1` distinct keys or all samples more
// than 32.  The table lives in vector registers (padded with entry 0).
template <class T>
inline unsigned vfew_sample(const T* p, std::size_t n, unsigned cap1, std::size_t extra, T* vals) {
    constexpr bool k64 = sizeof(T) == 8;
    constexpr int NV = k64 ? 4 : 2;            // 32 entries
    using U = typename std::conditional<k64, std::uint64_t, std::uint32_t>::type;
    alignas(64) U tb[32];
    unsigned K = 0;
    __m512i tv[NV];
    auto probe = [&](std::size_t idx, unsigned cap) -> bool {
        U b;
        std::memcpy(&b, p + idx, sizeof(U));
        if (K != 0) {
            const __m512i x = vbcast_bits<U>(b);
            bool hit = false;
            for (int v = 0; v < NV; ++v) {
                if constexpr (k64) hit |= _mm512_cmpeq_epi64_mask(tv[v], x) != 0;
                else hit |= _mm512_cmpeq_epi32_mask(tv[v], x) != 0;
            }
            if (hit) return true;
        }
        if (K == cap) return false;
        if (K == 0) for (int t = 0; t < 32; ++t) tb[t] = b;
        tb[K] = b;
        vals[K++] = p[idx];
        for (int v = 0; v < NV; ++v) tv[v] = _mm512_load_si512(reinterpret_cast<const void*>(tb + v * (64 / sizeof(U))));
        return true;
    };
    for (std::size_t j = 0; j < 64; ++j)
        if (!probe((j * n) >> 6, cap1)) return 0;
    for (std::size_t j = 0; j < extra; ++j)
        if (!probe(((j * n) / extra + n / (2 * extra)) % n, 32)) return 0;
    return K;
}

// Bitwise all-equal test: one read stream against a broadcast of p[0]
// (memcmp(p, p + 1) reads two misaligned streams).  Stops at the first
// differing vector.
// Bitwise all-equal test against a broadcast of p[0]; stops at the first
// differing group.  One unaligned vector first (random input leaves there),
// then aligned loads from the next vector boundary: an unaligned zmm load
// splits a cache line every time (100k int32 from a 16-byte aligned
// buffer: 5.9 -> 3.1 us).  x |= v ^ b is one ternlog (no compare-to-mask on
// port 5), two independent accumulators, one test per 16 vectors.  Past
// kPrescanYmmBytes the scan is fetch-bound: ymm (no 512-bit warm-up), four
// interleaved quarter streams, one accumulator each (800 KB uint64:
// 17.1 -> 15.8 us).
#define FYX_ORXOR256(a, v, b) _mm256_ternarylogic_epi32((a), (v), (b), 0xF6)
#define FYX_ORXOR512(a, v, b) _mm512_ternarylogic_epi32((a), (v), (b), 0xF6)
template <class T>
inline bool vall_equal(const T* p, std::size_t n) {
    constexpr std::size_t V = 64 / sizeof(T);
    if (n < V) {
        for (std::size_t i = 1; i < n; ++i) if (std::memcmp(p + i, p, sizeof(T)) != 0) return false;
        return true;
    }
    const auto addr = reinterpret_cast<std::uintptr_t>(p);
    if (n * sizeof(T) < kPrescanYmmBytes) {
        __m512i b;
        if constexpr (sizeof(T) == 8) { std::uint64_t x; std::memcpy(&x, p, 8); b = _mm512_set1_epi64(static_cast<long long>(x)); }
        else { std::uint32_t x; std::memcpy(&x, p, 4); b = _mm512_set1_epi32(static_cast<int>(x)); }
        {
            const __m512i x = _mm512_xor_si512(_mm512_loadu_si512(reinterpret_cast<const void*>(p)), b);
            if (_mm512_test_epi64_mask(x, x)) return false;
        }
        // next 64-byte boundary (element-aligned buffers; else unaligned
        // loads from V on, still correct)
        std::size_t i = (addr % sizeof(T)) ? V : ((((addr + 64) & ~std::uintptr_t(63)) - addr) / sizeof(T));
        __m512i a0 = _mm512_setzero_si512(), a1 = a0;
        for (; i + 16 * V <= n; i += 16 * V) {
            FYX_VQ_UNROLL for (std::size_t k = 0; k < 16 * V; k += 2 * V) {
                a0 = FYX_ORXOR512(a0, _mm512_loadu_si512(reinterpret_cast<const void*>(p + i + k)), b);
                a1 = FYX_ORXOR512(a1, _mm512_loadu_si512(reinterpret_cast<const void*>(p + i + k + V)), b);
            }
            const __m512i x = _mm512_or_si512(a0, a1);
            if (_mm512_test_epi64_mask(x, x)) return false;
        }
        __m512i x = _mm512_or_si512(a0, a1);
        for (; i + V <= n; i += V) x = FYX_ORXOR512(x, _mm512_loadu_si512(reinterpret_cast<const void*>(p + i)), b);
        if (i < n) x = FYX_ORXOR512(x, _mm512_loadu_si512(reinterpret_cast<const void*>(p + n - V)), b);
        return _mm512_test_epi64_mask(x, x) == 0;
    }
    constexpr std::size_t L = 32 / sizeof(T), SB = 8 * L;
    __m256i b;
    if constexpr (sizeof(T) == 8) { std::uint64_t x; std::memcpy(&x, p, 8); b = _mm256_set1_epi64x(static_cast<long long>(x)); }
    else { std::uint32_t x; std::memcpy(&x, p, 4); b = _mm256_set1_epi32(static_cast<int>(x)); }
#define FYX_YLD(i) _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + (i)))
    {
        const __m256i x = _mm256_xor_si256(FYX_YLD(0), b);
        if (!_mm256_testz_si256(x, x)) return false;
    }
    // aligned stream starts: h on a 32-byte boundary, q a multiple of SB
    const std::size_t h = (addr % sizeof(T)) ? L : ((((addr + 32) & ~std::uintptr_t(31)) - addr) / sizeof(T));
    const std::size_t q = (n - h) / 4 / SB * SB;
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (std::size_t i = h; i < h + q; i += SB) {
        FYX_VQ_UNROLL for (std::size_t k = 0; k < SB; k += L) {
            a0 = FYX_ORXOR256(a0, FYX_YLD(i + k), b);
            a1 = FYX_ORXOR256(a1, FYX_YLD(q + i + k), b);
            a2 = FYX_ORXOR256(a2, FYX_YLD(2 * q + i + k), b);
            a3 = FYX_ORXOR256(a3, FYX_YLD(3 * q + i + k), b);
        }
        const __m256i x = _mm256_or_si256(_mm256_or_si256(a0, a1), _mm256_or_si256(a2, a3));
        if (!_mm256_testz_si256(x, x)) return false;
    }
    __m256i x = _mm256_or_si256(_mm256_or_si256(a0, a1), _mm256_or_si256(a2, a3));
    std::size_t i = h + 4 * q;
    for (; i + L <= n; i += L) x = FYX_ORXOR256(x, FYX_YLD(i), b);
    if (i < n) x = FYX_ORXOR256(x, FYX_YLD(n - L), b);
#undef FYX_YLD
    return _mm256_testz_si256(x, x);
}
#undef FYX_ORXOR256
#undef FYX_ORXOR512

// Positions j in [from, n) with key(p[j]) < key(p[j-1]) in target order,
// keys encoded as the radix total order (so NaN / -0 agree with the scalar
// paths).  Returns the count, or maxd + 1 as soon as it exceeds maxd.
template <class T>
inline std::size_t vdescent_positions(const T* p, std::size_t from, std::size_t n, bool descending,
                                      std::uint32_t* pos, std::size_t maxd) {
    constexpr bool k64 = sizeof(T) == 8;
    constexpr std::size_t V = 64 / sizeof(T);
    auto desc_mask = [&](std::size_t j) -> std::uint64_t {     // lanes j .. j+V-1
        const __m512i c = vencode_radix<T>(_mm512_loadu_si512(reinterpret_cast<const void*>(p + j)));
        const __m512i q = vencode_radix<T>(_mm512_loadu_si512(reinterpret_cast<const void*>(p + j - 1)));
        if constexpr (k64) return descending ? _mm512_cmpgt_epu64_mask(c, q) : _mm512_cmplt_epu64_mask(c, q);
        else return descending ? _mm512_cmpgt_epu32_mask(c, q) : _mm512_cmplt_epu32_mask(c, q);
    };
    std::size_t d = 0;
    std::size_t j = from < 1 ? 1 : from;
    // Branch-free recording: compress-store the lane indices of a mask.
    const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    auto take = [&](std::uint64_t m, std::size_t base) -> bool {
        const std::size_t c = static_cast<std::size_t>(__builtin_popcountll(m));
        if (d + c > maxd) { d = maxd + 1; return false; }
        const __m512i idx = _mm512_add_epi32(iota, _mm512_set1_epi32(static_cast<int>(base)));
        _mm512_mask_compressstoreu_epi32(pos + d, static_cast<__mmask16>(m), idx);
        d += c;
        return true;
    };
    for (; j + 2 * V <= n; j += 2 * V) {
        const std::uint64_t m0 = desc_mask(j), m1 = desc_mask(j + V);
        if ((m0 | m1) == 0) continue;
        if (!take(m0, j) || !take(m1, j + V)) return d;
    }
    for (; j + V <= n; j += V)
        if (!take(desc_mask(j), j)) return d;
    if (j < n && n > V) {
        // Tail: re-test the last V pairs, keep only lanes not yet covered.
        const std::size_t b = n - V;
        const std::uint64_t m = desc_mask(b) >> (j - b);
        if (!take(m, j)) return d;
    } else if (j < n) {
        using RT = RadixTraits<T>;
        for (; j < n; ++j) {
            const bool lt = descending ? RT::encode(p[j - 1]) < RT::encode(p[j]) : RT::encode(p[j]) < RT::encode(p[j - 1]);
            if (lt && !take(1, j)) return d;
        }
    }
    return d;
}


// ---------------------------------------------------------------------------
// 16-byte record quicksort (key-value records keyed by one 64-bit field at
// byte offset 0 or 8).  The same partition as vpartition, run on 64-bit lanes
// with *pair* masks: the key lane's compare bit is copied onto its payload
// lane, so one compress-store moves whole records.  Duplicate-heavy inputs
// finish without leaves (a range whose min key equals its max key is done),
// which is where the out-of-place counting / radix record paths -- two or
// more full read+write passes plus a copy back -- lose to vqsort's K64V64.
// Leaves (<= 32 records) are scalar insertion sorts.
// ---------------------------------------------------------------------------
template <bool KeyHi, bool Signed>
struct KvOps {
    static constexpr unsigned KM = KeyHi ? 0xAAu : 0x55u;
    FYX_FORCE_INLINE static __mmask8 pair(__mmask8 m) {
        const unsigned k = static_cast<unsigned>(m) & KM;
        return static_cast<__mmask8>(KeyHi ? (k | (k >> 1)) : (k | (k << 1)));
    }
    FYX_FORCE_INLINE static __mmask8 ge(__m512i a, __m512i b) {
        if constexpr (Signed) return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_NLT);
        else                  return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_NLT);
    }
    FYX_FORCE_INLINE static __mmask8 gt(__m512i a, __m512i b) {
        if constexpr (Signed) return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_NLE);
        else                  return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_NLE);
    }
    FYX_FORCE_INLINE static __mmask8 ge_k(__mmask8 k, __m512i a, __m512i b) {
        if constexpr (Signed) return _mm512_mask_cmp_epi64_mask(k, a, b, _MM_CMPINT_NLT);
        else                  return _mm512_mask_cmp_epu64_mask(k, a, b, _MM_CMPINT_NLT);
    }
    FYX_FORCE_INLINE static __mmask8 gt_k(__mmask8 k, __m512i a, __m512i b) {
        if constexpr (Signed) return _mm512_mask_cmp_epi64_mask(k, a, b, _MM_CMPINT_NLE);
        else                  return _mm512_mask_cmp_epu64_mask(k, a, b, _MM_CMPINT_NLE);
    }
    FYX_FORCE_INLINE static __m512i vmin(__m512i s, __mmask8 k, __m512i a) {
        if constexpr (Signed) return _mm512_mask_min_epi64(s, k, s, a);
        else                  return _mm512_mask_min_epu64(s, k, s, a);
    }
    FYX_FORCE_INLINE static __m512i vmax(__m512i s, __mmask8 k, __m512i a) {
        if constexpr (Signed) return _mm512_mask_max_epi64(s, k, s, a);
        else                  return _mm512_mask_max_epu64(s, k, s, a);
    }
    FYX_FORCE_INLINE static std::uint64_t rmin(__m512i v) {
        if constexpr (Signed) return static_cast<std::uint64_t>(_mm512_reduce_min_epi64(v));
        else                  return _mm512_reduce_min_epu64(v);
    }
    FYX_FORCE_INLINE static std::uint64_t rmax(__m512i v) {
        if constexpr (Signed) return static_cast<std::uint64_t>(_mm512_reduce_max_epi64(v));
        else                  return _mm512_reduce_max_epu64(v);
    }
    static constexpr std::uint64_t hi() { return Signed ? 0x7fffffffffffffffull : ~0ull; }
    static constexpr std::uint64_t lo() { return Signed ? 0x8000000000000000ull : 0ull; }
    FYX_FORCE_INLINE static bool lt(std::uint64_t x, std::uint64_t y) {
        if constexpr (Signed) return static_cast<std::int64_t>(x) < static_cast<std::int64_t>(y);
        else                  return x < y;
    }
    FYX_FORCE_INLINE static std::uint64_t key(const std::uint64_t* a, std::size_t i) {
        std::uint64_t k;
        std::memcpy(&k, a + 2 * i + (KeyHi ? 1 : 0), 8);
        return k;
    }
    FYX_FORCE_INLINE static void cstore(std::uint64_t* p, __mmask8 m, __m512i v) {
#if FYX_VQ_COMPRESS_TO_MEMORY
        _mm512_mask_compressstoreu_epi64(p, m, v);
#else
        _mm512_mask_storeu_epi64(p, static_cast<__mmask8>(bzhi_mask(popcount64(static_cast<std::uint64_t>(m)))),
                                 _mm512_maskz_compress_epi64(m, v));
#endif
    }
};

struct KvRec { std::uint64_t w[2]; };
#ifndef FYX_KVQ_LEAF
#  define FYX_KVQ_LEAF 64
#endif
static_assert(FYX_KVQ_LEAF <= 64, "kv_leaf packs a 6-bit record index");

template <bool KeyHi, bool Signed, bool Strict>
FYX_FORCE_INLINE void kv_part_vec(std::uint64_t* a, std::size_t& ls, std::size_t& rs, __m512i cur,
                                  __m512i pv, __m512i& vmn, __mmask8 valid) {
    using O = KvOps<KeyHi, Signed>;
    const __mmask8 hm = static_cast<__mmask8>(O::pair(Strict ? O::gt(cur, pv) : O::ge(cur, pv)) & valid);
    const __mmask8 lm = static_cast<__mmask8>(~hm & valid);
    const unsigned gc = static_cast<unsigned>(popcount64(hm));
    const unsigned lc = static_cast<unsigned>(popcount64(lm));
    O::cstore(a + ls, lm, cur);
    O::cstore(a + rs - gc, hm, cur);
    ls += lc;
    rs -= gc;
    vmn = O::vmin(vmn, static_cast<__mmask8>(O::KM & valid), cur);
}

/// Record permutation indexed by the even-lane "goes high" bits of a compare
/// mask (bit 2r: record r): low records first, then high records, each in
/// original order.  Only the 16 rows with even bits are used, so the direct
/// index costs no bit gathering (pext is not in the AVX-512 target set).
struct KvPermTable {
    alignas(64) std::int64_t idx[86][8];
    constexpr KvPermTable() : idx{} {
        for (int m = 0; m < 86; ++m) {
            int o = 0;
            for (int pass = 0; pass < 2; ++pass)
                for (int r = 0; r < 4; ++r)
                    if (((m >> (2 * r)) & 1) == pass) { idx[m][o++] = 2 * r; idx[m][o++] = 2 * r + 1; }
        }
    }
};
inline constexpr KvPermTable kKvPerm{};

/// Unmasked variant for the partition main loop: one permute, two full
/// stores (callers guarantee >= 8 lanes of already-read slack on each side).
template <bool KeyHi, bool Signed, bool Strict>
FYX_FORCE_INLINE void kv_part_vec_full(std::uint64_t* a, std::size_t& ls, std::size_t& rs, __m512i cur,
                                       __m512i pv, __m512i& vmn) {
    using O = KvOps<KeyHi, Signed>;
    const __mmask8 km = static_cast<__mmask8>(O::KM);
    unsigned m = static_cast<unsigned>(Strict ? O::gt_k(km, cur, pv) : O::ge_k(km, cur, pv));
    if constexpr (KeyHi) m >>= 1;
    const __m512i perm = _mm512_permutexvar_epi64(_mm512_load_si512(kKvPerm.idx[m]), cur);
    const unsigned gc = 2u * static_cast<unsigned>(popcount64(m));
    _mm512_storeu_si512(a + ls, perm);
    _mm512_storeu_si512(a + rs - 8, perm);
    ls += 8u - gc;
    rs -= gc;
    vmn = O::vmin(vmn, static_cast<__mmask8>(O::KM), cur);
}

template <bool KeyHi, bool Signed, bool Strict>
inline std::size_t kv_partition_scalar(std::uint64_t* a, std::size_t nr, std::uint64_t pivot,
                                       std::uint64_t& kmin) {
    using O = KvOps<KeyHi, Signed>;
    std::uint64_t mn = O::hi();
    for (std::size_t i = 0; i < nr; ++i) {
        const std::uint64_t k = O::key(a, i);
        mn = O::lt(k, mn) ? k : mn;
    }
    std::size_t i = 0, j = nr;
    while (i < j) {
        const std::uint64_t k = O::key(a, i);
        const bool low = Strict ? !O::lt(pivot, k) : O::lt(k, pivot);
        if (low) { ++i; continue; }
        --j;
        KvRec x, y;
        std::memcpy(&x, a + 2 * i, 16);
        std::memcpy(&y, a + 2 * j, 16);
        std::memcpy(a + 2 * i, &y, 16);
        std::memcpy(a + 2 * j, &x, 16);
    }
    kmin = mn;
    return i;
}

/// Partition records a[0, nr) (2*nr lanes): key < pivot (Strict: key <= pivot)
/// to the front; returns that record count.  kmin: the smallest key.
template <bool KeyHi, bool Signed, bool Strict>
inline std::size_t kv_partition(std::uint64_t* a, std::size_t nr, std::uint64_t pivot,
                                std::uint64_t& kmin) {
    using O = KvOps<KeyHi, Signed>;
    constexpr std::size_t V = 8, U = 4, CH = U * V;
    const std::size_t n = 2 * nr;
    if (n < 2 * CH) return kv_partition_scalar<KeyHi, Signed, Strict>(a, nr, pivot, kmin);
    const __m512i pv = _mm512_set1_epi64(static_cast<long long>(pivot));
    __m512i vmn = _mm512_set1_epi64(static_cast<long long>(O::hi()));
    std::size_t l = 0, r = n, ls = 0, rs = n;
    __m512i vl[U], vr[U];
    for (std::size_t i = 0; i < U; ++i) vl[i] = _mm512_loadu_si512(a + i * V);
    for (std::size_t i = 0; i < U; ++i) vr[i] = _mm512_loadu_si512(a + n - (i + 1) * V);
    l += CH;
    r -= CH;
    while (r - l >= CH) {
        // A real branch, not a select: the next loads then issue on the
        // predicted side instead of waiting on this iteration's counts.
        __m512i cur[U];
        if ((rs - r) < (l - ls)) {
            r -= CH;
            for (std::size_t i = 0; i < U; ++i) cur[i] = _mm512_loadu_si512(a + r + i * V);
        } else {
            for (std::size_t i = 0; i < U; ++i) cur[i] = _mm512_loadu_si512(a + l + i * V);
            l += CH;
        }
        for (std::size_t i = 0; i < U; ++i)
            kv_part_vec_full<KeyHi, Signed, Strict>(a, ls, rs, cur[i], pv, vmn);
    }
    while (r - l >= V) {
        const bool from_right = (rs - r) < (l - ls);
        const std::size_t src = from_right ? (r - V) : l;
        const __m512i cur = _mm512_loadu_si512(a + src);
        r -= from_right ? V : 0;
        l += from_right ? 0 : V;
        kv_part_vec_full<KeyHi, Signed, Strict>(a, ls, rs, cur, pv, vmn);
    }
    if (r != l) {
        const __mmask8 valid = static_cast<__mmask8>((1u << (r - l)) - 1u);
        const __m512i cur = _mm512_maskz_loadu_epi64(valid, a + l);
        kv_part_vec<KeyHi, Signed, Strict>(a, ls, rs, cur, pv, vmn, valid);
    }
    for (std::size_t i = 0; i < U; ++i) kv_part_vec<KeyHi, Signed, Strict>(a, ls, rs, vl[i], pv, vmn, 0xFF);
    for (std::size_t i = 0; i < U; ++i) kv_part_vec<KeyHi, Signed, Strict>(a, ls, rs, vr[i], pv, vmn, 0xFF);
    kmin = O::rmin(vmn);
    return ls / 2;
}

template <bool KeyHi, bool Signed>
inline void kv_insertion(std::uint64_t* a, std::size_t nr) {
    using O = KvOps<KeyHi, Signed>;
    for (std::size_t i = 1; i < nr; ++i) {
        const std::uint64_t k = O::key(a, i);
        if (!O::lt(k, O::key(a, i - 1))) continue;
        KvRec x;
        std::memcpy(&x, a + 2 * i, 16);
        std::size_t j = i;
        do { std::memmove(a + 2 * j, a + 2 * (j - 1), 16); --j; }
        while (j > 0 && O::lt(k, O::key(a, j - 1)));
        std::memcpy(a + 2 * j, &x, 16);
    }
}

/// Proof for a range whose keys are bit-identical: every record equivalent
/// to the reference record `ref` (a strict weak order makes that transitive,
/// hence sorted).  One load stream against a broadcast reference vectorizes
/// ~2x faster than the adjacent-pair proof (0.56 vs 1.0 cycles / record, L1).
/// Fin types without equiv() fall back to the adjacent-pair verify.
template <class Fin, class = void>
struct kv_fin_has_equiv : std::false_type {};
template <class Fin>
struct kv_fin_has_equiv<Fin, decltype(void(std::declval<Fin&>().equiv(std::size_t(), std::size_t(), std::size_t())))>
    : std::true_type {};
template <class Fin>
inline bool kv_fin_equiv(Fin& fin, std::size_t ref, std::size_t lo, std::size_t cnt) {
    if constexpr (kv_fin_has_equiv<Fin>::value) return fin.equiv(ref, lo, cnt);
    else return fin.verify(lo, cnt, lo != ref);
}
/// A final range of equal keys: seam plus equivalence proof.
template <class Fin>
inline bool kv_fin_equal(Fin& fin, std::size_t lo, std::size_t cnt) {
    fin.seam(lo);
    return kv_fin_equiv(fin, lo, lo, cnt);
}

/// Uniform-sample check: 1 when records [lo, lo + nr) are proven sorted
/// (all equivalent to the first under fin's comparator), 2 when fin rejected
/// a block whose keys all equal `key`, 0 when some key differs (partition
/// on).  Block by block in L1.  The comparator proof runs first: when it
/// passes the block is final whatever its keys are, so the key scan is paid
/// only on a failing block, to tell "keys differ" from "comparator rejects".
template <bool KeyHi, bool Signed, class Fin>
inline int kv_all_equal_fin(const std::uint64_t* base, std::size_t lo, std::size_t nr, std::uint64_t key,
                            Fin& fin) {
    using O = KvOps<KeyHi, Signed>;
    const __m512i kv = _mm512_set1_epi64(static_cast<long long>(key));
    const __mmask8 km = static_cast<__mmask8>(O::KM);
    constexpr std::size_t B = 128;                       // records per block
    for (std::size_t b = 0; b < nr; b += B) {
        const std::size_t e = nr - b < B ? nr : b + B;
        if (kv_fin_equiv(fin, lo, lo + b, e - b)) continue;
        const std::uint64_t* a = base + 2 * (lo + b);
        const std::size_t n = 2 * (e - b);
        __mmask8 d = 0;
        for (std::size_t i = 0; i < n; i += 8) {
            const std::size_t m = n - i < 8 ? n - i : 8;
            const __mmask8 v = static_cast<__mmask8>((1u << m) - 1u);
            d |= _mm512_mask_cmpneq_epi64_mask(static_cast<__mmask8>(km & v), _mm512_maskz_loadu_epi64(v, a + i), kv);
        }
        // A differing key may also sit in an earlier block that comp let
        // pass: harmless, those blocks are equivalent to the reference.
        return d != 0 ? 0 : 2;
    }
    fin.seam(lo);
    return 1;
}

/// Pivot by sample rank (after vqsort's PivotRank): keys equal to the pivot
/// go low, so the pivot is never the largest sample, and between the median
/// and the previous distinct sample the better-balanced split wins.
/// uniform: every sample equal (pivot is that key).
/// Leaf sort for nr <= 64 records: keys rebased to the leaf minimum and
/// packed with the record index ((key - min) << 6 | i) sort as plain u64 in a
/// register network, then one gather pass places the records.  Falls back to
/// insertion when the leaf's key span needs more than 58 bits.
template <bool KeyHi, bool Signed>
inline void kv_leaf(std::uint64_t* a, std::size_t nr) {
    using O = KvOps<KeyHi, Signed>;
    if (nr < 2) return;
    constexpr std::uint64_t flip = Signed ? 0x8000000000000000ull : 0ull;
    alignas(64) std::uint64_t k[64];
    std::uint64_t mn = ~0ull, mx = 0;
    for (std::size_t i = 0; i < nr; ++i) {
        const std::uint64_t u = O::key(a, i) ^ flip;
        k[i] = u;
        mn = u < mn ? u : mn;
        mx = u > mx ? u : mx;
    }
    if (mx - mn >= (1ull << 58)) { kv_insertion<KeyHi, Signed>(a, nr); return; }
    for (std::size_t i = 0; i < nr; ++i) k[i] = ((k[i] - mn) << 6) | i;
    vnet_sort<std::uint64_t>(k, nr);
    alignas(64) std::uint64_t tmp[128];
    std::memcpy(tmp, a, nr * 16);
    for (std::size_t j = 0; j < nr; ++j) {
        const std::size_t i = static_cast<std::size_t>(k[j] & 63u);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(a + 2 * j),
                         _mm_load_si128(reinterpret_cast<const __m128i*>(tmp + 2 * i)));
    }
}

template <bool KeyHi, bool Signed>
inline std::uint64_t kv_pick_pivot(const std::uint64_t* a, std::size_t nr, bool& uniform) {
    using O = KvOps<KeyHi, Signed>;
    constexpr std::size_t SMAX = 64;
    const std::size_t S = nr >= 2048 ? SMAX : 16;
    alignas(64) std::uint64_t s[SMAX];
    const std::size_t step = nr / S;
    // Signed keys sort as unsigned with the sign bit flipped (and back).
    constexpr std::uint64_t flip = Signed ? 0x8000000000000000ull : 0ull;
    // One sample per stratum at a pseudo-random offset inside it: a fixed
    // stride aliases with periodic inputs (runs of a length dividing into
    // the stride sample the same run offsets -- the same quantiles).
    std::uint64_t h = 0x9E3779B97F4A7C15ull * (static_cast<std::uint64_t>(nr) | 1u);
    for (std::size_t i = 0; i < S; ++i) {
        h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 32;
        s[i] = O::key(a, i * step + static_cast<std::size_t>((h >> 11) % step)) ^ flip;
    }
    vnet_sort<std::uint64_t>(s, S);
    if constexpr (Signed) for (std::size_t i = 0; i < S; ++i) s[i] ^= flip;
    uniform = s[0] == s[S - 1];
    if (uniform) return s[0];
    const std::size_t mid = S / 2;
    std::size_t prev = mid;
    while (prev > 0 && s[prev - 1] == s[mid]) --prev;
    if (prev == 0) return s[mid];          // s[mid] is the sample minimum ...
    --prev;                                 // last sample below the median
    std::size_t next = mid + 1;
    while (next < S && s[next] == s[mid]) ++next;
    if (next == S) return s[prev];          // median is the sample maximum
    return (next - mid) < (mid - prev) ? s[mid] : s[prev];
}

/// One quicksort step on records [lo, lo + nr) (nr > leaf size).  Returns
/// 0: the range is final and proven; -1: fin rejected a final range; 1: one
/// range is left, lo / nr updated; 2: split into [lo, lo + split) and
/// [lo + split, lo + nr), both non-empty and unproven.
template <bool KeyHi, bool Signed, class Fin>
inline int kvq_step(std::uint64_t* base, std::size_t& lo, std::size_t& nr, std::size_t& split, Fin& fin) {
    std::uint64_t* a = base + 2 * lo;
    bool uniform;
    const std::uint64_t pivot = kv_pick_pivot<KeyHi, Signed>(a, nr, uniform);
    if (uniform) {
        const int e = kv_all_equal_fin<KeyHi, Signed>(base, lo, nr, pivot, fin);
        if (e != 0) return e == 1 ? 0 : -1;
    }
    std::uint64_t mn;
    // Keys <= pivot low.
    const std::size_t s = kv_partition<KeyHi, Signed, true>(a, nr, pivot, mn);
    if (s == nr) {
        // pivot >= every key (only via a misleading uniform sample).
        if (mn == pivot) return kv_fin_equal(fin, lo, nr) ? 0 : -1;
        const std::size_t lt = kv_partition<KeyHi, Signed, false>(a, nr, pivot, mn);
        if (!kv_fin_equal(fin, lo + lt, nr - lt)) return -1;   // all == pivot
        nr = lt;
        return 1;
    }
    if (mn == pivot) {                                   // low side all == pivot
        if (!kv_fin_equal(fin, lo, s)) return -1;
        lo += s;
        nr -= s;
        return 1;
    }
    split = s;
    return 2;
}

/// false: depth budget exhausted or fin rejected a finished range (the range
/// then holds a permutation of its input).  fin(first_record, count) is
/// called on every final range while it is still cache-hot; it proves the
/// adjacent pairs inside the range under the caller's comparator and records
/// the seam at first_record.  fin.verify(first, count, with_prev) proves the
/// pairs inside [first, first + count), plus (first - 1, first) when with_prev;
/// fin.seam(first) records a seam.
template <bool KeyHi, bool Signed, class Fin>
inline bool kvq_rec(std::uint64_t* base, std::size_t lo, std::size_t nr, int budget, Fin& fin) {
    while (nr > FYX_KVQ_LEAF) {
        if (budget <= 0) return false;
        --budget;
        std::size_t split = 0;
        const int r = kvq_step<KeyHi, Signed>(base, lo, nr, split, fin);
        if (r == 0) return true;
        if (r < 0) return false;
        if (r == 1) continue;
        if (split < nr - split) {
            if (!kvq_rec<KeyHi, Signed>(base, lo, split, budget, fin)) return false;
            lo += split;
            nr -= split;
        } else {
            if (!kvq_rec<KeyHi, Signed>(base, lo + split, nr - split, budget, fin)) return false;
            nr = split;
        }
    }
    kv_leaf<KeyHi, Signed>(base + 2 * lo, nr);
    return fin(lo, nr);
}
} // namespace isa_avx512
} // namespace detail
} // namespace fyx
FYX_ISA_END

#endif // FYX_HAS_AVX512_CODE

namespace fyx {
namespace detail {

/// Bitwise all-equal (AVX-512 single stream; memcmp elsewhere).
template <class T>
inline bool range_bitwise_all_equal(const T* p, std::size_t n) {
    // Ends first (two scalar loads): random input leaves before any vector
    // setup, which measured ~30 ns of the small-n probe budget.
    if (n >= 2 && std::memcmp(static_cast<const void*>(p), static_cast<const void*>(p + n - 1), sizeof(T)) != 0)
        return false;
#if FYX_HAS_AVX512_CODE
    if constexpr (radix_supported_v<T> && (sizeof(T) == 4 || sizeof(T) == 8)) {
        if (use_avx512()) return isa_avx512::vall_equal(p, n);
    }
#endif
    return n < 2 || std::memcmp(static_cast<const void*>(p), static_cast<const void*>(p + 1), (n - 1) * sizeof(T)) == 0;
}

/// Checked tail reverse (see isa_avx512::vreverse_tail_checked); 2 (caller
/// decides) where no kernel applies.
template <class T, bool Undo = true>
inline int reverse_tail_checked(T* p, std::size_t lo, std::size_t n, bool descending) {
#if FYX_HAS_AVX512_CODE
    if constexpr (radix_supported_v<T> && (sizeof(T) == 4 || sizeof(T) == 8)) {
        if (use_avx512()) return isa_avx512::vreverse_tail_checked<T, Undo>(p, lo, n, descending);
    }
#endif
    (void)p; (void)lo; (void)n; (void)descending;
    return 2;
}

/// Descent positions (see isa_avx512::vdescent_positions); scalar elsewhere.
template <class T>
inline std::size_t descent_positions(const T* p, std::size_t from, std::size_t n, bool descending,
                                     std::uint32_t* pos, std::size_t maxd) {
#if FYX_HAS_AVX512_CODE
    if constexpr (radix_supported_v<T> && (sizeof(T) == 4 || sizeof(T) == 8)) {
        if (use_avx512() && n <= 0xFFFFFFFFull) return isa_avx512::vdescent_positions(p, from, n, descending, pos, maxd);
    }
#endif
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    std::size_t d = 0;
    for (std::size_t j = from < 1 ? 1 : from; j < n; ++j) {
        if (static_cast<Key>(RT::encode(p[j]) ^ flip) < static_cast<Key>(RT::encode(p[j - 1]) ^ flip)) {
            if (d == maxd) return maxd + 1;
            pos[d++] = static_cast<std::uint32_t>(j);
        }
    }
    return d;
}

/// Few-distinct-keys sort (<= 32 distinct, bitwise): a 64-key sample must
/// show <= 16 distinct keys, 64 more samples refine the table, then one
/// counting pass (declines, range untouched, if a key is missing) and a fill.
template <class T>
inline bool try_vfew_distinct_sort(T* p, std::size_t n, bool descending) {
#if FYX_HAS_AVX512_CODE
    if constexpr (radix_supported_v<T> && (sizeof(T) == 4 || sizeof(T) == 8)) {
        if (!use_avx512() || n < 1024) return false;
        T vals[32];
        const unsigned K = isa_avx512::vfew_sample(p, n, 16, 64, vals);
        if (K == 0) return false;
        std::uint64_t cnt[32] = {};
        bool ok;
        switch ((K + 3) / 4) {             // table padded to a multiple of 4
            case 1:  ok = isa_avx512::vfew_count<T, 4>(p, n, vals, K, cnt); break;
            case 2:  ok = isa_avx512::vfew_count<T, 8>(p, n, vals, K, cnt); break;
            case 3:  ok = isa_avx512::vfew_count<T, 12>(p, n, vals, K, cnt); break;
            case 4:  ok = isa_avx512::vfew_count<T, 16>(p, n, vals, K, cnt); break;
            case 5:  ok = isa_avx512::vfew_count<T, 20>(p, n, vals, K, cnt); break;
            case 6:  ok = isa_avx512::vfew_count<T, 24>(p, n, vals, K, cnt); break;
            case 7:  ok = isa_avx512::vfew_count<T, 28>(p, n, vals, K, cnt); break;
            default: ok = isa_avx512::vfew_count<T, 32>(p, n, vals, K, cnt); break;
        }
        if (!ok) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
        unsigned ord[32];
        for (unsigned t = 0; t < K; ++t) ord[t] = t;
        std::sort(ord, ord + K, [&](unsigned a, unsigned b) {
            return static_cast<Key>(RT::encode(vals[a]) ^ flip) < static_cast<Key>(RT::encode(vals[b]) ^ flip);
        });
        std::size_t o = 0;
        for (unsigned t = 0; t < K; ++t) {
            isa_avx512::vfill(p + o, static_cast<std::size_t>(cnt[ord[t]]), vals[ord[t]]);
            o += static_cast<std::size_t>(cnt[ord[t]]);
        }
        return true;
    } else {
        (void)p; (void)n; (void)descending;
        return false;
    }
#else
    (void)p; (void)n; (void)descending;
    return false;
#endif
}

/// SIMD merge of two sorted runs (radix-key order, `descending` flips it)
/// into out[0..na+nb).  Returns false when no vector kernel applies; the
/// caller then merges itself.  Requires na, nb >= one vector.
template <class T>
inline bool vmerge_runs(const T* A, std::size_t na, const T* B, std::size_t nb, T* out,
                        bool descending) {
#if FYX_HAS_AVX512_CODE
    if constexpr (radix_supported_v<T> && (sizeof(T) == 4 || sizeof(T) == 8)) {
        if (!use_avx512() || na < 16 || nb < 16) return false;
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
        isa_avx512::vmerge_runs_impl(A, na, B, nb, out, descending,
            [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); });
        return true;
    } else {
        (void)A; (void)na; (void)B; (void)nb; (void)out; (void)descending;
        return false;
    }
#else
    (void)A; (void)na; (void)B; (void)nb; (void)out; (void)descending;
    return false;
#endif
}

/// Types the vectorised quicksort has a kernel for.
template <class T>
struct vqsort_kernel_supported : std::integral_constant<bool,
#if FYX_HAS_AVX512_CODE
    std::is_same<T, std::int32_t>::value  || std::is_same<T, std::uint32_t>::value ||
    std::is_same<T, std::int64_t>::value  || std::is_same<T, std::uint64_t>::value ||
    std::is_same<T, float>::value         || std::is_same<T, double>::value
#else
    false
#endif
> {};

template <class T>
inline constexpr bool vqsort_kernel_supported_v = vqsort_kernel_supported<T>::value;

/// One partition step, for callers that want to drive the recursion
/// themselves (the parallel driver).  `left_done`/`right_done` mark a side
/// that holds one value only and therefore needs no sort.  `split` is always
/// a real reduction: see the strict re-partition in vqsort_rec.
struct VqStep {
    std::size_t split      = 0;
    bool        left_done  = false;
    bool        right_done = false;
};

#if FYX_HAS_AVX512_CODE

template <class T>
inline bool vqsort_range_clean(const T* p, std::size_t n) {
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)p; (void)n; return false; }
    else return isa_avx512::vrange_clean<T>(p, n);
}

/// One fused pass for the small-n front door: neighbour order flags plus the
/// NaN / -0 scan.  Bit 0 of the result: some a[j] < a[j+1]; bit 1: some
/// a[j+1] < a[j]; bit 2: the range is NOT clean (then the order bits are
/// meaningless -- hardware order differs from the library's total order);
/// bit 3: cleanliness not established -- the scan stopped once both
/// directions were seen (always with bits 0 and 1 set, bit 2 clear).
/// settled / prior (optional): when the scan stopped early (bit 3, or the
/// return value 3 for integers), the block start i where both directions
/// were first seen and the order bits of pairs (j, j+1), j < i (the
/// monotone prefix); untouched otherwise.
template <class T>
inline unsigned vqsort_small_prescan(const T* p, std::size_t n, std::size_t* settled = nullptr,
                                     unsigned* prior = nullptr) {
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)p; (void)n; (void)settled; (void)prior; return 4u; }
    else return isa_avx512::vsmall_prescan<T>(p, n, settled, prior);
}

template <class T, class C>
inline void vfill_runs_dispatch(T* p, std::size_t n, const T* vals, const C* cnt, std::size_t m) {
    isa_avx512::vfill_runs<T, C>(p, n, vals, cnt, m);
}

/// Depth budget: 2 log2(n) partitions is what a correct pivot stream needs;
/// past that the range goes to pdqsort, which has its own introsort guard.
inline int vqsort_budget(std::size_t n) {
    int budget = 2;
    for (std::size_t m = n; m > 1; m >>= 1) budget += 2;
    return budget;
}

/// 16-byte record quicksort entry (see isa_avx512::kvq_rec).  `p` holds nr
/// records; the key is the 64-bit field at byte 8 (key_hi) or 0.  Ascending
/// key order on true; on false p holds a permutation of its input.
template <class Fin>
inline bool kv16_vqsort_range(void* p, std::size_t lo, std::size_t nr, int budget, bool key_hi, bool is_signed,
                              Fin& fin) {
    std::uint64_t* a = static_cast<std::uint64_t*>(p);
    if (key_hi) return is_signed ? isa_avx512::kvq_rec<true, true>(a, lo, nr, budget, fin)
                                 : isa_avx512::kvq_rec<true, false>(a, lo, nr, budget, fin);
    return is_signed ? isa_avx512::kvq_rec<false, true>(a, lo, nr, budget, fin)
                     : isa_avx512::kvq_rec<false, false>(a, lo, nr, budget, fin);
}
template <class Fin>
inline bool kv16_vqsort(void* p, std::size_t nr, bool key_hi, bool is_signed, Fin& fin) {
    return kv16_vqsort_range(p, 0, nr, vqsort_budget(nr), key_hi, is_signed, fin);
}
/// One step of the record quicksort (see isa_avx512::kvq_step); nr must
/// exceed FYX_KVQ_LEAF.
template <class Fin>
inline int kv16_vqsort_step(void* p, std::size_t& lo, std::size_t& nr, std::size_t& split, bool key_hi,
                            bool is_signed, Fin& fin) {
    std::uint64_t* a = static_cast<std::uint64_t*>(p);
    if (key_hi) return is_signed ? isa_avx512::kvq_step<true, true>(a, lo, nr, split, fin)
                                 : isa_avx512::kvq_step<true, false>(a, lo, nr, split, fin);
    return is_signed ? isa_avx512::kvq_step<false, true>(a, lo, nr, split, fin)
                     : isa_avx512::kvq_step<false, false>(a, lo, nr, split, fin);
}

template <class T>
inline void vqsort_serial(T* p, std::size_t n) {
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)p; (void)n; }
    else isa_avx512::vqsort_rec<T>(p, n, vqsort_budget(n));
}

/// Clean-screening vector quicksort: false (range permuted, unsorted) when
/// the range holds a NaN or -0.
template <class T>
inline bool vqsort_serial_checked(T* p, std::size_t n) {
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)p; (void)n; return false; }
    else return isa_avx512::vqsort_rec_checked<T>(p, n, vqsort_budget(n));
}

template <class T>
inline void vqsort_serial_budget(T* p, std::size_t n, int budget) {
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)p; (void)n; (void)budget; }
    else isa_avx512::vqsort_rec<T>(p, n, budget);
}

template <class T>
inline VqStep vqsort_partition_step(T* p, std::size_t n) {
    VqStep out;
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)p; (void)n; return out; }
    else {
        const T pivot = isa_avx512::vpick_pivot<T>(p, n);
        T lo, hi;
        out.split = isa_avx512::vpartition<T>(p, n, pivot, lo, hi);
        if (!(lo < pivot)) {                    // pivot is the range minimum
            if (!(pivot < hi)) {                // ... and its maximum
                out.split      = n;
                out.left_done  = true;
                out.right_done = true;
                return out;
            }
            T lo2, hi2;
            out.split     = isa_avx512::vpartition<T, true>(p, n, pivot, lo2, hi2);
            out.left_done = true;               // the prefix is all pivot
            return out;
        }
        out.right_done = !(pivot < hi);
        return out;
    }
}

/// Leaf size of the kernel, exported so drivers can stop splitting in time.
template <class T>
inline constexpr std::size_t vqsort_leaf() {
#if FYX_HAS_AVX512_CODE
    if constexpr (vqsort_kernel_supported_v<T>) return isa_avx512::VqLeaf<T>::value;
    else return 0;
#else
    return 0;
#endif
}

#else  // no AVX-512 kernel compiled in

template <class T> inline unsigned vqsort_small_prescan(const T*, std::size_t, std::size_t* = nullptr, unsigned* = nullptr) { return 4u; }
template <class T> inline bool   vqsort_serial_checked(T*, std::size_t) { return false; }
template <class T> inline bool   vqsort_range_clean(const T*, std::size_t) { return false; }
inline int                       vqsort_budget(std::size_t) { return 0; }
template <class Fin>
inline bool                      kv16_vqsort(void*, std::size_t, bool, bool, Fin&) { return false; }
template <class Fin>
inline bool kv16_vqsort_range(void*, std::size_t, std::size_t, int, bool, bool, Fin&) { return false; }
template <class Fin>
inline int kv16_vqsort_step(void*, std::size_t&, std::size_t&, std::size_t&, bool, bool, Fin&) { return -1; }
template <class T> inline void   vqsort_serial(T*, std::size_t) {}
template <class T> inline void   vqsort_serial_budget(T*, std::size_t, int) {}
template <class T> inline VqStep vqsort_partition_step(T*, std::size_t) { return VqStep{}; }
template <class T> inline constexpr std::size_t vqsort_leaf() { return 0; }

#endif // FYX_HAS_AVX512_CODE

/// Which types the vector quicksort is actually *faster* on than the radix
/// family, decided by head-to-head kernel races on whole ranges (numbers in
/// tools/dev/NOTES.md, tables in BENCHMARKS.md):
///
///   int32   radix 0.0070 -> vq 0.0044 s (8M serial, old host)   take it
///   float   radix 0.147  -> vq 0.038  s (8M serial, old host)   take it
///   double  radix 0.114  -> vq 0.088  s (8M serial, old host)   take it
///   int64   radix 0.089  -> vq 0.096  s (8M serial, old host)   was radix
///
/// int64 was the lone "keep radix" row -- by 7% on the old host, serially at
/// 8M.  On the current host the same race runs the other way by 2-3x (1M
/// random full-range: 0.0061-0.0094 against 0.014-0.030), and the high-prefix
/// kernel owns a pathological tie case: 8M values over a 40-bit span sort in
/// 0.73 s against the vectorised quicksort's 0.083 -- nine times -- on the
/// tie-repair pass it pays whenever the top prefix bits are not near-unique.
/// A default whose worst measured case is -7% cannot stand against one whose
/// worst measured case is -90%, so 64-bit integers now take the vectorised
/// quicksort on every host; the radix family keeps the shapes only it can do
/// (permutation ranges, prefix-guarded count sorts).
template <class T>
inline constexpr bool vqsort_preferred() {
    return std::is_same<T, std::int32_t>::value || std::is_same<T, std::uint32_t>::value ||
           std::is_same<T, std::int64_t>::value || std::is_same<T, std::uint64_t>::value ||
           std::is_same<T, float>::value        || std::is_same<T, double>::value;
}

/// Runtime gate: the kernel exists for this type, the CPU has AVX-512, and
/// the range is big enough for the vector partition to pay for itself.
template <class T>
inline bool vqsort_usable(std::size_t n) {
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)n; return false; }
    else return use_avx512() && n > vqsort_leaf<T>();
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

    /// Index of the calling thread within the pool.  Pool threads own their
    /// slot; the first external thread to ask claims slot 0 -- the submitting
    /// thread the deque design reserves -- and every further external thread
    /// gets kNoWorker, which routes fork_join through the safe foreign queue.
    /// A Chase-Lev deque has exactly one owner; two external threads both
    /// pretending to be worker 0 pushed and popped the same deque and
    /// occasionally lost a task, hanging the waiter forever.
    static unsigned this_worker() noexcept {
        const unsigned id = tls_worker_id();
        if (id != kNoWorker) return id;
        static std::atomic<bool> primary_taken{false};
        bool expect = false;
        if (primary_taken.compare_exchange_strong(expect, true)) {
            tls_worker_id() = 0;
            return 0;
        }
        return kNoWorker;
    }

    static constexpr unsigned kNoWorker = static_cast<unsigned>(-1);

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

    /// Submit a task from a thread that owns no deque (every external caller
    /// after the first).  Idle workers drain this FIFO after their steal
    /// attempts miss, and foreign waiters help empty it while they wait.
    bool submit_foreign(Task t) {
        if (broken_) return false;
        {
            std::lock_guard<std::mutex> lk(foreign_mu_);
            foreign_q_.push_back(t);
        }
        foreign_pending_.fetch_add(1, std::memory_order_acq_rel);
        if (sleepers_.load(std::memory_order_acquire) != 0) {
            {
                std::lock_guard<std::mutex> lk(sleep_mu_);
                ++wake_epoch_;
            }
            sleep_cv_.notify_all();
        }
        return true;
    }

    bool foreign_pop(Task& out) {
        if (foreign_pending_.load(std::memory_order_acquire) == 0) return false;
        std::lock_guard<std::mutex> lk(foreign_mu_);
        if (foreign_q_.empty()) return false;
        out = foreign_q_.front();
        foreign_q_.pop_front();
        foreign_pending_.fetch_sub(1, std::memory_order_acq_rel);
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
        static thread_local unsigned id = kNoWorker;
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
        // Only a foreign waiter drains the foreign FIFO: this function sits
        // inside every spin-wait of every parallel sort, and an extra shared
        // atomic read there measurably slowed the striped swaps.
        if (self == kNoWorker) return foreign_pop(out);
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
            if (foreign_pop(t)) { t.run(); continue; }
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

    // Tasks handed in by foreign threads (callers that own no deque).  See
    // submit_foreign -- this exists so several application threads can sort
    // concurrently without any two of them owning one work-stealing deque.
    std::mutex               foreign_mu_;
    std::deque<Task>         foreign_q_;
    std::atomic<std::size_t> foreign_pending_{0};
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

    // A foreign thread owns no deque; its half goes through the mutexed FIFO
    // that workers and foreign waiters drain.  Only the primary external
    // thread and pool threads use the Chase-Lev deques directly.
    const bool queued = (self != ThreadPool::kNoWorker)
        ? pool.submit(self, Task{&Job::run, &job})
        : pool.submit_foreign(Task{&Job::run, &job});
    if (!queued) {
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
    FYX_UNUSED_TYPE(T);        // only the v2 top-level split looks at it
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
        // Inside an ascending run "not down" never changes anything, so the
        // equivalence test (a second comparison) is only paid elsewhere.
        if (dir == 1) {
            if (!down) continue;
        } else {
            if (!down && !comp(p[i - 1], p[i])) continue;   // equivalent: no information
            const int d = down ? -1 : 1;
            if (dir == 0 || d == dir) { dir = d; continue; }
        }
        // The extremum at i-1 keeps the closed run; the new run starts at i.
        if (count < cap) { out[count].begin = start; out[count].end = i; }
        ++count;
        if (count > cap) return cap + 1;
        start = i;
        dir = -dir;
    }
    if (count < cap) { out[count].begin = start; out[count].end = n; }
    ++count;
    return count > cap ? cap + 1 : count;
}

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

/// Merges the adjacent runs [l, m) and [m, r) in place, using `buf` as scratch
/// space for the smaller of the two runs (at most min(r - m, m - l) elements).
/// `buf` may be null, in which case the merge falls back to std::inplace_merge
/// (the only correct option for types whose objects have to be constructed).
template <class T, class Comp>
inline void merge_adjacent_runs(T* p, std::size_t l, std::size_t m, std::size_t r,
                                T* buf, Comp comp) {
    if (m == l || m == r) return;
    // Already in order across the seam: nothing to do.
    if (!comp(p[m], p[m - 1])) return;
    // Trim what is already in place: the left prefix not above p[m] and the
    // right suffix not below p[m-1] (TimSort's pre-merge gallop).
    l = static_cast<std::size_t>(std::upper_bound(p + l, p + m, p[m], comp) - p);
    r = static_cast<std::size_t>(std::lower_bound(p + m, p + r, p[m - 1], comp) - p);
    const std::size_t a = m - l;
    const std::size_t b = r - m;
    if (a == 0 || b == 0) return;
    // Every remaining right element below every remaining left one: a
    // rotation (moved blocks, rotated ranges), no comparisons needed.
    if (comp(p[r - 1], p[l])) {
        if (buf == nullptr) { std::rotate(p + l, p + m, p + r); return; }
        if (a <= b) {
            for (std::size_t i = 0; i < a; ++i) buf[i] = std::move(p[l + i]);
            move_range_bulk(p + l, p + m, b);
            for (std::size_t i = 0; i < a; ++i) p[l + b + i] = std::move(buf[i]);
        } else {
            for (std::size_t i = 0; i < b; ++i) buf[i] = std::move(p[m + i]);
            move_range_bulk(p + l + b, p + l, a);
            for (std::size_t i = 0; i < b; ++i) p[l + i] = std::move(buf[i]);
        }
        return;
    }
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

/// Sparse-displacement repair.
///
/// "Sorted except for a few elements that sit far from home" (a handful of
/// long-distance swaps, say) splits into a few dozen long runs, and merging
/// those drags every displaced element across the array one merge level at a
/// time.  Instead, one read-only scan keeps a sorted subsequence and extracts
/// whatever breaks it: when x is below the last kept element, x is extracted,
/// unless x still fits after the kept element before that -- then the last
/// kept element was the intruder and is extracted instead.  The extracted
/// elements are sorted and streamed back in a single left-to-right rewrite in
/// which records only move where the number of insertions so far differs from
/// the number of extractions; for a swap the two cancel, so the stretch
/// between the two swapped positions is never touched.  Returns false without
/// modifying anything when more than `cap` elements would be extracted.
template <class T, class Before>
inline bool try_sparse_extract_repair(T* p, std::size_t n, Before before, std::size_t cap,
                                      const MonotoneRun* runs = nullptr, std::size_t nruns = 0) {
    if constexpr (!std::is_move_constructible<T>::value || !std::is_move_assignable<T>::value) {
        (void)p; (void)n; (void)before; (void)cap; (void)runs; (void)nruns;
        return false;
    } else {
        std::vector<std::size_t> ext;
        ext.reserve(cap < 256 ? cap + 1 : 256);
        std::size_t ring[8];
        unsigned head = 0, depth = 0;      // depth = valid entries in the ring
        // With the (ascending) run list the scan skips run interiors: once
        // i and i-1 are both kept, everything up to the end of i's run is
        // non-decreasing from i and would be kept without a decision.
        std::size_t ri = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (depth != 0 && before(p[i], p[ring[head]])) {
                if (depth >= 2 && !before(p[i], p[ring[(head - 1u) & 7u]])) {
                    ext.push_back(ring[head]);
                    ring[head] = i;            // pop the intruder, keep i
                } else {
                    ext.push_back(i);
                }
                if (ext.size() > cap) return false;
                continue;
            }
            const bool chained = depth != 0 && ring[head] + 1 == i;
            head = (head + 1u) & 7u;
            ring[head] = i;
            if (depth < 8) ++depth;
            if (runs && chained) {
                while (ri < nruns && runs[ri].end <= i) ++ri;
                if (ri < nruns && runs[ri].begin < i && runs[ri].end > i + 2) {
                    const std::size_t last = runs[ri].end - 1;
                    head = (head + 1u) & 7u; ring[head] = last - 1;
                    head = (head + 1u) & 7u; ring[head] = last;
                    depth = depth + 2 < 8 ? depth + 2 : 8;
                    i = last;
                }
            }
        }
        if (ext.empty()) return true;
        std::sort(ext.begin(), ext.end());

        std::vector<T> buf;
        buf.reserve(ext.size());
        for (std::size_t idx : ext) buf.push_back(std::move(p[idx]));
        pdqsort(buf.data(), buf.data() + buf.size(), before);

        // Streaming rewrite.  Kept elements are read in order from the queue
        // (elements saved because a write overtook them) and then from p[r..];
        // positions listed in `ext` are holes that may be overwritten freely.
        std::vector<T> q;                  // FIFO: q[qh..)
        std::size_t qh = 0;
        std::size_t r = 0, e = 0, w = 0, b = 0;
        const std::size_t ne = ext.size(), nb = buf.size();
        auto skip_holes = [&]() { while (e < ne && ext[e] == r) { ++r; ++e; } };
        auto save_upto = [&](std::size_t pos) {
            while (r <= pos && r < n) {
                if (e < ne && ext[e] == r) { ++e; ++r; continue; }
                q.push_back(std::move(p[r]));
                ++r;
            }
        };
        while (w < n) {
            const bool q_empty = qh == q.size();
            if (q_empty) skip_holes();
            const bool have_kept = !q_empty || r < n;
            if (q_empty && w == r && r < n) {
                // In-place stretch: kept elements before the next hole and
                // before the next extracted element stay where they are.
                const std::size_t lim = e < ne ? ext[e] : n;
                std::size_t t;
                if (b < nb) {
                    const T& v = buf[b];
                    std::size_t lo = r, step = 1;
                    while (lo + step < lim && !before(v, p[lo + step])) { lo += step; step <<= 1; }
                    std::size_t hi = lo + step < lim ? lo + step : lim;
                    if (before(v, p[lo])) hi = lo;
                    else {
                        ++lo;
                        while (lo < hi) {
                            const std::size_t mid = lo + (hi - lo) / 2;
                            if (before(v, p[mid])) hi = mid; else lo = mid + 1;
                        }
                    }
                    t = hi;
                } else {
                    t = lim;
                }
                if (t != r) {
                    w = r = t;
                    if (w >= n) break;
                    continue;
                }
            }
            const bool take_buf = b < nb &&
                (!have_kept || before(buf[b], q_empty ? p[r] : q[qh]));
            if (take_buf) {
                save_upto(w);
                p[w++] = std::move(buf[b++]);
            } else if (!q_empty) {
                T v = std::move(q[qh++]);
                save_upto(w);
                p[w++] = std::move(v);
                if (qh == q.size()) { q.clear(); qh = 0; }
            } else {
                // q empty, kept element is p[r] with r > w: slide it down.
                p[w++] = std::move(p[r++]);
            }
        }
        return true;
    }
}

/// Ascending runs whose merge interleaves them only a few times are a block
/// permutation: the merged output is a short sequence of chunks, each a
/// contiguous slice of one run.  A k-way gallop lists those chunks (one
/// binary search each) and gives up past `max_chunks`; then the sort is a
/// rearrangement of whole slices, and slices that already sit at their final
/// offset -- the middle of a block swap, the untouched ends -- never move.
template <class T, class Before>
inline bool try_disjoint_run_permutation(T* p, std::size_t n, const std::vector<std::size_t>& bounds,
                                         std::size_t count, Before before) {
    if constexpr (!std::is_move_constructible<T>::value || !std::is_move_assignable<T>::value) {
        (void)p; (void)n; (void)bounds; (void)count; (void)before;
        return false;
    } else {
        constexpr std::size_t max_chunks = 256;
        std::vector<std::size_t> head(bounds.begin(), bounds.begin() + static_cast<std::ptrdiff_t>(count));
        std::vector<std::size_t> csrc, clen;
        csrc.reserve(64); clen.reserve(64);
        for (;;) {
            std::size_t best = count, second = count;
            for (std::size_t r = 0; r < count; ++r) {
                if (head[r] == bounds[r + 1]) continue;
                if (best == count || before(p[head[r]], p[head[best]])) { second = best; best = r; }
                else if (second == count || before(p[head[r]], p[head[second]])) second = r;
            }
            if (best == count) break;
            std::size_t e = bounds[best + 1];
            if (second != count)
                e = static_cast<std::size_t>(std::upper_bound(p + head[best], p + e, p[head[second]], before) - p);
            if (!csrc.empty() && csrc.back() + clen.back() == head[best]) {
                clen.back() += e - head[best];      // same run continues
            } else {
                if (csrc.size() >= max_chunks) return false;
                csrc.push_back(head[best]);
                clen.push_back(e - head[best]);
            }
            head[best] = e;
        }
        const std::size_t cc = csrc.size();
        std::vector<std::size_t> cdst(cc);
        std::size_t off = 0, moved = 0;
        for (std::size_t c = 0; c < cc; ++c) {
            cdst[c] = off;
            if (off != csrc[c]) moved += clen[c];
            off += clen[c];
        }
        (void)n;
        if (moved == 0) return true;
        if constexpr (std::is_trivially_copyable<T>::value) {
            ScratchLease<T> lease(moved);
            if (lease.valid()) {
                T* tmp = lease.get();
                std::size_t t = 0;
                for (std::size_t c = 0; c < cc; ++c) {
                    if (cdst[c] == csrc[c]) continue;
                    std::memcpy(static_cast<void*>(tmp + t), static_cast<const void*>(p + csrc[c]), clen[c] * sizeof(T));
                    t += clen[c];
                }
                t = 0;
                for (std::size_t c = 0; c < cc; ++c) {
                    if (cdst[c] == csrc[c]) continue;
                    std::memcpy(static_cast<void*>(p + cdst[c]), static_cast<const void*>(tmp + t), clen[c] * sizeof(T));
                    t += clen[c];
                }
                return true;
            }
        }
        std::vector<T> tmp;
        tmp.reserve(moved);
        for (std::size_t c = 0; c < cc; ++c) {
            if (cdst[c] == csrc[c]) continue;
            for (std::size_t i = csrc[c]; i < csrc[c] + clen[c]; ++i) tmp.push_back(std::move(p[i]));
        }
        std::size_t t = 0;
        for (std::size_t c = 0; c < cc; ++c) {
            if (cdst[c] == csrc[c]) continue;
            for (std::size_t i = 0; i < clen[c]; ++i) p[cdst[c] + i] = std::move(tmp[t + i]);
            t += clen[c];
        }
        return true;
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
        // Every run ascending and every break explained by one element on
        // either side of it: a few far-displaced elements, not interleaved
        // runs.  Extract and re-insert them instead of merging.
        {
            bool all_up = true, single = true;
            for (std::size_t i = 0; i < count && all_up; ++i) {
                const std::size_t b = runbuf[i].begin, e = runbuf[i].end;
                if (e - b > 1 && before(p[e - 1], p[b])) all_up = false;
                if (i == 0) continue;
                const bool left_ok  = b >= 2 && !before(p[b], p[b - 2]);
                const bool right_ok = b + 1 < n && !before(p[b + 1], p[b - 1]);
                if (!left_ok && !right_ok) single = false;
            }
            if (all_up && single &&
                try_sparse_extract_repair(p, n, before, 2 * count + 16, runbuf.data(), count)) return true;
        }
        {
            // Disjoint runs: reverse the descending ones (a permutation, so
            // harmless if this declines) and rearrange whole runs.
            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t b = runbuf[i].begin, e = runbuf[i].end;
                if (e - b > 1 && before(p[e - 1], p[b])) std::reverse(p + b, p + e);
            }
            if (count > 2 && try_disjoint_run_permutation(p, n, bounds, count, before)) return true;
        }
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
inline bool try_dirty_patch_merge_adaptive(T* p, std::size_t n, Comp comp,
                                           std::size_t max_dirty = 0) {
#if !FYX_ENABLE_ADAPTIVE_WEAPONS
    (void)p; (void)n; (void)comp; (void)max_dirty;
    return false;
#else
    return try_dirty_patch_merge(p, n, comp, max_dirty ? max_dirty : n / 8);
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
inline bool try_displacement_patch_merge_adaptive(T* p, std::size_t n, Comp comp,
                                                  std::size_t max_dirty = 0) {
#if !FYX_ENABLE_ADAPTIVE_WEAPONS
    (void)p; (void)n; (void)comp; (void)max_dirty;
    return false;
#else
    return try_displacement_patch_merge(p, n, comp, max_dirty ? max_dirty : n / 8);
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
//      * numeric, AVX-512, n < kVqsortMinN              -> order scan + vector quicksort
//      * stable_sort of integers, "<" or ">"            -> unstable engine (equal ints
//                                                          are indistinguishable)
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

// Stable sorting may reverse only a *strictly* descending range: if equivalent
// adjacent elements exist, reversing them would violate stability. Kept out of
// line so the common already-sorted monotonic exit remains compact.
template <class It, class Comp>
FYX_NOINLINE inline bool try_stable_reverse_sort(It first, It last, Comp comp) {
    const std::size_t n = static_cast<std::size_t>(last - first);
    if (n < 2) return false;
    for (std::size_t i = 1; i < n; ++i)
        if (!comp(first[i], first[i - 1])) return false;
    std::reverse(first, last);
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

        // Branch-free blocked pre-check for the common "already in target
        // order" case.  Each block reduces a violation flag without an early
        // exit, so compilers vectorise it and sorted input is verified at
        // close to memory bandwidth; the first violating block falls through
        // to the exact scalar classifier below (which also handles reverse
        // and one-break rotations).  Random input fails in the first block.
        {
            constexpr std::size_t kBlock = 256;
            bool in_order = true;
            for (std::size_t b = 1; b < n && in_order; b += kBlock) {
                const std::size_t e = std::min(n, b + kBlock);
                unsigned bad = 0;
                if (descending) {
                    for (std::size_t j = b; j < e; ++j)
                        bad |= static_cast<unsigned>(RT::encode(p[j - 1]) < RT::encode(p[j]));
                } else {
                    for (std::size_t j = b; j < e; ++j)
                        bad |= static_cast<unsigned>(RT::encode(p[j]) < RT::encode(p[j - 1]));
                }
                in_order = bad == 0;
            }
            if (in_order) return true;
        }

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

// Exact order classification over radix keys with branch-free 256-element
// blocks: both direction flags are reduced per block (vectorisable), and the
// scan stops at the first block where both directions have been violated.
// Sorted input therefore runs at close to memory bandwidth instead of paying
// a data-dependent branch per element; random input stops after one block.
template <class T>
inline FastOrderKind blocked_radix_order_kind(const T* p, std::size_t n, bool descending) {
    using RT = RadixTraits<T>;
    if (n < 2) return FastOrderKind::AllEqual;
    // The first block is short: random input shows both directions within
    // a handful of pairs, and on small ranges a full 256-pair block would
    // cost a sizeable fraction of the whole sort.
    constexpr std::size_t kBlock = 256;
    unsigned up = 0, down = 0;   // saw k[j] > k[j-1] / k[j] < k[j-1]
    for (std::size_t b = 1, blk = 16; b < n && !(up && down); b += blk, blk = kBlock) {
        const std::size_t e = std::min(n, b + blk);
        unsigned bu = 0, bd = 0;
        for (std::size_t j = b; j < e; ++j) {
            const auto k0 = RT::encode(p[j - 1]);
            const auto k1 = RT::encode(p[j]);
            bu |= static_cast<unsigned>(k0 < k1);
            bd |= static_cast<unsigned>(k1 < k0);
        }
        up |= bu;
        down |= bd;
    }
    if (!up && !down) return FastOrderKind::AllEqual;
    if (up && down) return FastOrderKind::None;
    const bool ascending_run = up != 0;
    return ascending_run != descending ? FastOrderKind::Sorted : FastOrderKind::Reverse;
}

// Merge two sorted runs A[0..na) and B[0..nb) into out[0..na+nb) as four
// independent branch-free merges over co-ranked output quarters.  A single
// branch-free merge is bound by its load -> compare -> index dependency
// (~3 ns/key on Ice Lake-SP); four interleaved chains run at ~1.5 ns/key.
// Reads A[na] and B[nb] (never uses them): both must be addressable.
template <class T, class KeyF>
inline void merge_runs_4way(const T* A, std::size_t na, const T* B, std::size_t nb, T* out, KeyF key) {
    constexpr int C = 4;
    using Key  = decltype(key(*A));
    using Bits = typename std::conditional<sizeof(T) == 8, std::uint64_t,
                 typename std::conditional<sizeof(T) == 4, std::uint32_t,
                 typename std::conditional<sizeof(T) == 2, std::uint16_t, std::uint8_t>::type>::type>::type;
    static_assert(sizeof(Bits) == sizeof(T) && std::is_trivially_copyable<T>::value, "raw-bit merge");
    const std::size_t N = na + nb;
    std::size_t sa[C + 1], sb[C + 1];
    for (int c = 0; c <= C; ++c) {
        const std::size_t s = N / C * static_cast<std::size_t>(c) + (c == C ? N % C : 0);
        std::size_t lo = s > nb ? s - nb : 0, hi = s < na ? s : na;
        while (lo < hi) {                  // smallest i with B[s-i-1] < A[i]
            const std::size_t i = lo + (hi - lo) / 2;
            const std::size_t j = s - i;
            if (j > 0 && !(key(B[j - 1]) < key(A[i]))) lo = i + 1; else hi = i;
        }
        sa[c] = lo;
        sb[c] = s - lo;
    }
    std::size_t a[C], ae[C], b[C], be[C], o[C];
    std::size_t steps = N;
    for (int c = 0; c < C; ++c) {
        a[c] = sa[c]; ae[c] = sa[c + 1];
        b[c] = sb[c]; be[c] = sb[c + 1];
        o[c] = sa[c] + sb[c];
        steps = std::min(steps, (sa[c + 1] + sb[c + 1]) - o[c]);
    }
    for (std::size_t t = 0; t < steps; ++t) {
        FYX_VQ_UNROLL for (int c = 0; c < C; ++c) {
            const Key ka = key(A[a[c]]);
            const Key kb = key(B[b[c]]);
            const bool ta = (a[c] < ae[c]) & ((b[c] >= be[c]) | !(kb < ka));
            // Mask select on raw bits: GCC turns value selects on FP into
            // branches, which mispredict on every interleaving.
            if constexpr (std::is_integral<T>::value) {
                out[o[c]++] = ta ? A[a[c]] : B[b[c]];
            } else {
            Bits va, vb;
            std::memcpy(&va, A + a[c], sizeof(T));
            std::memcpy(&vb, B + b[c], sizeof(T));
            const Bits mk = static_cast<Bits>(Bits(0) - static_cast<Bits>(ta));
            const Bits v = static_cast<Bits>((va & mk) | (vb & static_cast<Bits>(~mk)));
            std::memcpy(out + o[c]++, &v, sizeof(T));
            }
            a[c] += static_cast<std::size_t>(ta);
            b[c] += static_cast<std::size_t>(!ta);
        }
    }
    for (int c = 0; c < C; ++c) {
        while (a[c] < ae[c] && b[c] < be[c]) {
            const bool ta = !(key(B[b[c]]) < key(A[a[c]]));
            out[o[c]++] = ta ? A[a[c]] : B[b[c]];
            a[c] += static_cast<std::size_t>(ta);
            b[c] += static_cast<std::size_t>(!ta);
        }
        while (a[c] < ae[c]) out[o[c]++] = A[a[c]++];
        while (b[c] < be[c]) out[o[c]++] = B[b[c]++];
    }
}

template <class T>
inline std::size_t radix_key_find_break(const T* p, std::size_t start, std::size_t hi,
                                        typename RadixTraits<T>::Key prev_key,
                                        const bool want_up, const bool descending);

// Neighbour descents (in target order) over the whole range, counted in
// branch-free 256-pair blocks; stops once the count exceeds `cap` and then
// returns cap + 1.  `first` gets the start of the 256-pair block holding the
// first descent (exact position: radix_key_find_break from there).  With
// `blk`, the starts of the first blk_cap blocks holding a descent go there
// (*nblk of them): every such block when the result is <= blk_cap, so a
// caller can extract the positions from those blocks alone.
template <class T>
inline std::size_t count_descents_capped(const T* p, std::size_t n, bool descending,
                                         std::size_t cap, std::size_t& first, std::size_t from = 1,
                                         std::uint32_t* blk = nullptr, std::size_t blk_cap = 0,
                                         std::size_t* nblk = nullptr) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    std::size_t d = 0, nb = 0;
    first = n;
    for (std::size_t b = from < 1 ? 1 : from; b < n; b += 256) {
        const std::size_t e = std::min(n, b + 256);
        Key bd = 0;                        // same lane width as the keys: vectorizes
        for (std::size_t j = b; j < e; ++j)
            bd = static_cast<Key>(bd + static_cast<Key>(static_cast<Key>(RT::encode(p[j]) ^ flip) <
                                                        static_cast<Key>(RT::encode(p[j - 1]) ^ flip)));
        if (bd != 0) {
            if (first == n) first = b;     // block of the first descent
            if (nb < blk_cap) blk[nb++] = static_cast<std::uint32_t>(b);
        }
        d += bd;
        if (d > cap) break;
    }
    if (nblk != nullptr) *nblk = nb;
    return d > cap ? cap + 1 : d;
}

// Positions of all D descents from the nblk block starts that
// count_descents_capped left in io[] (overwritten with the positions).  Out
// of line: inlined, the extra call site shifts GCC's inlining of the
// numeric dispatcher (1M uint32 nearly-sorted 1.7x slower).
template <class T>
FYX_NOINLINE bool descent_positions_in_blocks(const T* p, std::size_t n, bool descending,
                                              std::uint32_t* io, std::size_t nblk, std::size_t D) {
    std::uint32_t blk[32];
    if (nblk > 32) return false;
    std::memcpy(blk, io, nblk * sizeof(std::uint32_t));
    std::size_t m = 0;
    for (std::size_t t = 0; t < nblk && m < D; ++t)
        m += descent_positions(p, blk[t], std::min<std::size_t>(n, blk[t] + 256u), descending, io + m, D - m);
    return m == D;
}

// Cheap stand-in for "more than n/16 descents" on random-like input: three
// 512-pair windows (head, middle, tail) each over a quarter descents, and
// at least 8 of 64 strided neighbour pairs descending.  A sorted range with
// a few disordered windows fails the strided test and gets the exact count.
// (The exact capped count scans n/8 pairs of a random range: ~26k cycles at
// n = 200000, 1% of its sort.)
template <class T>
inline bool descents_look_dense(const T* p, std::size_t n, bool descending) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    if (n < 8192) return false;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    std::size_t sd = 0;
    for (std::size_t j = 0; j < 64; ++j) {
        const std::size_t i = (j * (n - 1)) >> 6;
        sd += key(p[i + 1]) < key(p[i]);
    }
    if (sd < 8) return false;
    const std::size_t starts[3] = {1, n / 2, n - 512};
    for (std::size_t w = 0; w < 3; ++w) {
        Key bd = 0;
        for (std::size_t j = starts[w]; j < starts[w] + 512; ++j)
            bd = static_cast<Key>(bd + static_cast<Key>(key(p[j]) < key(p[j - 1])));
        if (bd <= 128) return false;
    }
    return true;
}

// True when the first few descents (from `from`) are local: the key after
// each drop still sits at or above the key `reach` places back.  Near-sorted
// input (short displacements) passes; sorted runs, far swaps and random
// tails, whose drops fall far, fail on the first descent.
template <class T>
inline bool descents_look_local(const T* p, std::size_t n, std::size_t from, bool descending,
                                std::size_t reach, int samples) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    std::size_t i = from < 1 ? 1 : from;
    for (int s = 0; s < samples && i < n; ++s) {
        i = radix_key_find_break(p, i, n, RT::encode(p[i - 1]), true, descending);
        if (i >= n) break;
        const std::size_t back = i > reach ? i - reach : 0;
        if (key(p[i]) < key(p[back])) return false;
        ++i;
    }
    return true;
}

// True when the first `samples` descents (from `from`) look like boundaries
// between sorted runs: spaced at least `gap` apart and not explained by one
// misplaced key (dropping neither neighbour restores the order).  Far swaps,
// sparse outliers and dense random stretches all fail it.
template <class T>
inline bool descents_look_like_runs(const T* p, std::size_t n, std::size_t from, bool descending,
                                    std::size_t gap, int samples) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    std::size_t i = from < 1 ? 1 : from, last = 0;
    for (int s = 0; s < samples; ++s) {
        if (i >= n) return false;
        i = radix_key_find_break(p, i, n, RT::encode(p[i - 1]), true, descending);
        if (i + 1 >= n || i < 2) return false;
        if (s != 0 && i - last < gap) return false;
        if (!(key(p[i]) < key(p[i - 2])) || !(key(p[i + 1]) < key(p[i - 1]))) return false;
        last = i;
        ++i;
    }
    return true;
}

// Exactly two ascending runs p[0..brk) and p[brk..n): rotate when the second
// run lies entirely below the first, otherwise one four-way merge.
template <class T>
inline void merge_two_runs(T* p, std::size_t n, std::size_t brk, bool descending) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    if (!(key(p[0]) < key(p[n - 1]))) {          // back <= front: a rotation
        std::rotate(p, p + brk, p + n);
        return;
    }
    const std::size_t nb = n - brk;
    {
        // The thread arena keeps the scratch mapped across calls (a fresh
        // allocation of this size page-faults on every call).
        ScratchLease<T> out(n);
        if (out.valid() && vmerge_runs(p, brk, p + brk, nb, out.get(), descending)) {
            std::memcpy(static_cast<void*>(p), static_cast<const void*>(out.get()), n * sizeof(T));
            return;
        }
    }
    std::unique_ptr<T[]> tmp(new T[nb + 1]);
    std::memcpy(static_cast<void*>(tmp.get()), static_cast<const void*>(p + brk), nb * sizeof(T));
    std::unique_ptr<T[]> out(new T[n]);
    merge_runs_4way(p, brk, tmp.get(), nb, out.get(), key);   // reads p[brk], tmp[nb]
    std::memcpy(static_cast<void*>(p), static_cast<const void*>(out.get()), n * sizeof(T));
}

// p[0..cnt) -> p[s..s+cnt), s >= 1: backward 64-byte chunks through
// registers.  Short overlapping shifts are where a libc memmove call per
// outlier dominated (~100 cycles each on 1000-key inputs).
template <class T>
FYX_FORCE_INLINE void shift_up_small(T* p, std::size_t cnt, std::size_t s) {
    constexpr std::size_t C = 64 / sizeof(T) ? 64 / sizeof(T) : 1;
    std::size_t e = cnt;
    while (e >= C) {
        unsigned char tmp[C * sizeof(T)];
        std::memcpy(tmp, static_cast<const void*>(p + e - C), sizeof(tmp));
        std::memcpy(static_cast<void*>(p + e - C + s), tmp, sizeof(tmp));
        e -= C;
    }
    while (e > 0) { --e; p[e + s] = p[e]; }
}

// Merge of two sorted runs that first tries block moves: while the runs
// interleave in long stretches (block swaps, rotations, concatenations of
// disjoint ranges) each stretch is found by a galloping search and copied
// whole; after a few short stretches the rest goes through the SIMD merge
// (or std::merge).  Radix-key order, so NaN / -0 / descending agree with
// the other run kernels.
template <class T>
inline void merge_runs_galloping(const T* A, std::size_t na, const T* B, std::size_t nb, T* out,
                                 bool descending) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    // Count of leading x in R[0..m) with key(x) < kv (strict) or <= kv.
    auto gallop = [&](const T* R, std::size_t m, Key kv, bool strict) -> std::size_t {
        auto before = [&](std::size_t t) { const Key kr = key(R[t]); return strict ? kr < kv : !(kv < kr); };
        std::size_t hi = 1;
        while (hi < m && before(hi)) hi = 2 * hi + 1;
        std::size_t lo = (hi - 1) / 2 + (hi > 1 ? 1 : 0);
        if (hi > m) hi = m;
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (before(mid)) lo = mid + 1; else hi = mid;
        }
        return lo;
    };
    std::size_t i = 0, j = 0, o = 0;
    int shorts = 0;
    while (i < na && j < nb) {
        std::size_t len;
        if (key(B[j]) < key(A[i])) {
            len = gallop(B + j, nb - j, key(A[i]), true);
            std::memcpy(static_cast<void*>(out + o), static_cast<const void*>(B + j), len * sizeof(T));
            j += len;
        } else {
            len = gallop(A + i, na - i, key(B[j]), false);
            std::memcpy(static_cast<void*>(out + o), static_cast<const void*>(A + i), len * sizeof(T));
            i += len;
        }
        o += len;
        if (len < 32 && ++shorts > 3) break;
    }
    if (i < na && j < nb) {
        auto less = [&](const T& a, const T& b) { return key(a) < key(b); };
        if (!vmerge_runs(A + i, na - i, B + j, nb - j, out + o, descending))
            std::merge(A + i, A + na, B + j, B + nb, out + o, less);
        return;
    }
    if (i < na) std::memcpy(static_cast<void*>(out + o), static_cast<const void*>(A + i), (na - i) * sizeof(T));
    if (j < nb) std::memcpy(static_cast<void*>(out + o), static_cast<const void*>(B + j), (nb - j) * sizeof(T));
}

// Sample gate of try_hash_count_sort (kept out of line: inlined, it
// perturbed the counting loop's code by ~15%).  256 keys in 16 blocks of 16
// consecutive keys (on cold data ~2 lines per block instead of one miss per
// key; block-local clustering can only inflate repeats, i.e. pass more
// inputs on to the prefix check, which decides for itself).  The Chao1
// estimate D + f1^2 / (2 f2) of the total key count (f1 / f2: keys seen
// once / twice) is returned (all-ones: nearly all distinct); the caller
// declines long-tailed inputs on it before any table is touched.  The keys are sorted in registers and their runs counted
// without branches: a hash-probe gate cost ~7000-16000 cycles per call on
// fresh inputs (mispredicted probes; a heap table also arrives cold).
template <class U, std::size_t S>
FYX_NOINLINE std::size_t hash_count_sample_gate(const unsigned char* base, std::size_t n) {
    constexpr std::size_t B = 16;                       // keys per block (64 measured slower)
    static_assert(S % B == 0, "whole blocks");
    alignas(64) U smp[S];
    for (std::size_t j = 0; j < S; ++j)
        std::memcpy(&smp[j], base + (((j / B) * (n - B)) / (S / B) + (j % B)) * sizeof(U), sizeof(U));
    // Random-like input leaves on the first 128 keys holding no repeat
    // (1536 uniform keys: P(no repeat) ~ 0.5%); one leaf-sized sort.
    auto sort_n = [&](std::size_t m) {
#if FYX_HAS_AVX512_CODE
        if (use_avx512()) { vqsort_serial<U>(smp, m); return; }
#endif
        std::sort(smp, smp + m);
    };
    sort_n(128);
    {
        unsigned rep = 0;
        for (std::size_t j = 1; j < 128; ++j) rep |= static_cast<unsigned>(smp[j] == smp[j - 1]);
        if (!rep) return ~std::size_t(0);
    }
    sort_n(S);
    // A run of length L ends at j when smp[j] != smp[j + 1]; it adds one to
    // D, and to f1 / f2 when L is 1 / 2 -- i.e. when smp[j - 1] / smp[j - 2]
    // also differ from smp[j].
    std::size_t sd = 0, f1 = 0, f2 = 0;
    for (std::size_t j = 0; j < S; ++j) {
        const bool end = j + 1 == S || smp[j] != smp[j + 1];
        const bool s1  = j < 1 || smp[j - 1] != smp[j];
        const bool s2  = !s1 && (j < 2 || smp[j - 2] != smp[j]);
        sd += end;
        f1 += end & s1;
        f2 += end & s2;
    }
    if (sd > S - S / 16) return ~std::size_t(0);
    return sd + f1 * f1 / (2 * (f2 ? f2 : 1));
}

// Moderate-cardinality sort (a few hundred to ~2000 distinct keys, beyond
// the <= 32-key vector counter): one read-only pass counts bit patterns in
// an open-addressing table, then the distinct keys are ordered by radix key
// and written back as runs.  ~1.5 ns/key against ~3 ns/key for a vectorised
// quicksort on 64-bit keys, and no NaN / -0 special cases (bit patterns are
// counted, radix keys give the library's total order).  A 256-key strided
// sample must show duplicates first; the pass itself declines -- range
// untouched -- once the table holds more than kCap keys.
template <class T>
inline bool try_hash_count_sort(T* p, std::size_t n, bool descending) {
    if constexpr (!radix_supported_v<T> || !(sizeof(T) == 4 || sizeof(T) == 8)) {
        (void)p; (void)n; (void)descending;
        return false;
    } else {
        using RT  = RadixTraits<T>;
        using Key = typename RT::Key;
        using U   = typename std::conditional<sizeof(T) == 8, std::uint64_t, std::uint32_t>::type;
        if (n < 4096 || n > 0xFFFFFFFFull) return false;
        auto bits = [](const T& x) { U u; std::memcpy(&u, &x, sizeof(U)); return u; };
        auto mix = [](U k, unsigned b) {
            return static_cast<std::size_t>((static_cast<std::uint64_t>(k) * 0x9E3779B97F4A7C15ull) >> (64 - b));
        };
        constexpr unsigned kBits = 13;
        constexpr std::size_t kSlots = std::size_t(1) << kBits, kMask = kSlots - 1, kCap = 2048;
        // Estimated key count (Chao1 over a sample; larger inputs afford 512
        // keys, which separate a long tail (zipf) from ~1500 uniform keys
        // better than 256 do) must fit the table (3/4 of kCap) and leave
        // enough keys per distinct value: below n/24 (8-byte) or n/32
        // (4-byte) the cold table and the distinct-key sort cost more than
        // the quicksort they replace.  4-byte keys also need K > n/640: the
        // 16-lane quicksort finishes cheaply once partitions turn
        // single-valued above its 512-key leaf (n = 200000, 256 keys: 1.4
        // ns/key against 2.0 for the table; 447 keys -- leaf-sized values --
        // 3.3 against 2.2).
        const unsigned char* const raw = reinterpret_cast<const unsigned char*>(p);
        const std::size_t est = n >= 65536 ? hash_count_sample_gate<U, 512>(raw, n)
                                           : hash_count_sample_gate<U, 256>(raw, n);
        const std::size_t limit = std::min<std::size_t>(kCap * 3 / 4, n / (sizeof(T) == 8 ? 24 : 32));
        if (est > limit) return false;
        if (sizeof(T) == 4 && est <= n / 640) return false;
        // Tables from the thread arena (a fresh 96 KB new/delete pair per call
        // can be trimmed back to the OS and page-faulted in again).
        // Layout: keys | cnt | ord (Key) | vals (T) | runs | slot list.
        constexpr std::size_t oCnt  = kSlots * sizeof(U);
        constexpr std::size_t oOrd  = oCnt + kSlots * sizeof(std::uint32_t);
        constexpr std::size_t oVals = oOrd + kCap * sizeof(Key);
        constexpr std::size_t oRuns = oVals + kCap * sizeof(T);
        constexpr std::size_t oLst  = oRuns + kCap * sizeof(std::uint32_t);
        ScratchLease<unsigned char> lease(oLst + (kCap + 1) * sizeof(std::uint16_t));
        if (!lease.valid()) return false;
        unsigned char* const mem = lease.get();
        U* const keys = reinterpret_cast<U*>(mem);
        std::uint32_t* const cnt = reinterpret_cast<std::uint32_t*>(mem + oCnt);
        Key* const ord = reinterpret_cast<Key*>(mem + oOrd);
        T* const vals = reinterpret_cast<T*>(mem + oVals);
        std::uint32_t* const runs = reinterpret_cast<std::uint32_t*>(mem + oRuns);
        std::uint16_t* const lst = reinterpret_cast<std::uint16_t*>(mem + oLst);
        std::memset(cnt, 0, kSlots * sizeof(std::uint32_t));
        std::size_t D = 0;
        // A fixed key set stops adding keys early; a long-tailed one (zipf)
        // keeps producing singletons.  At checkpoints (a prefix of
        // max(n/64, 512) keys, then each doubling up to n/4) the Chao1
        // estimate D + f1^2 / (2 f2) of the total key count (f1 / f2: keys
        // seen once / twice) must stay within the table, so a doomed pass
        // stops early.  A single prefix check let borderline zipf inputs run
        // on to the cap (~20 us at n = 20000).
        // lst: slots in insertion order, so extraction never scans the table.
        std::size_t f1 = 0, f2 = 0;
        std::size_t i = 0;
        std::size_t chk = std::min(n, std::max<std::size_t>(n / 64, 512));
        for (;;) {
            for (; i < chk; ++i) {
                const U k = bits(p[i]);
                std::size_t h = mix(k, kBits);
                std::uint32_t c = cnt[h];
                // Short-circuit on purpose: an empty slot's key line is
                // never read, so first touches of the (usually cold) key
                // table are stores, not load misses.
                while (c != 0 && keys[h] != k) {
                    h = (h + 1) & kMask;
                    c = cnt[h];
                }
                lst[D] = static_cast<std::uint16_t>(h);
                D += c == 0;
                if (D > kCap) return false;            // table stays <= 1/4 full
                keys[h] = k;
                f1 += static_cast<std::size_t>(c == 0) - static_cast<std::size_t>(c == 1);
                f2 += static_cast<std::size_t>(c == 1) - static_cast<std::size_t>(c == 2);
                cnt[h] = c + 1;
            }
            if (D + f1 * f1 / (2 * (f2 ? f2 : 1)) > limit) return false;
            if (chk >= n / 4) break;
            chk = std::min(n / 4, chk * 2);
        }
        for (; i < n; ++i) {
            const U k = bits(p[i]);
            std::size_t h = mix(k, kBits);
            while (cnt[h] != 0 && keys[h] != k) h = (h + 1) & kMask;
            if (cnt[h] == 0) {
                if (D >= kCap) return false;
                lst[D++] = static_cast<std::uint16_t>(h);
                keys[h] = k;
            }
            ++cnt[h];
        }
        // Distinct keys in output order: radix-encode, sort the D keys (one
        // SIMD sort when available), then decode and re-probe for the count.
        // (A table scan plus std::sort of key/count records cost ~40 us at
        // n = 20000, 600 keys.)
        const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
        for (std::size_t t = 0; t < D; ++t) {
            T x;
            std::memcpy(&x, &keys[lst[t]], sizeof(U));
            ord[t] = static_cast<Key>(RT::encode(x) ^ flip);
        }
#if FYX_HAS_AVX512_CODE
        if (use_avx512()) vqsort_serial<Key>(ord, D);
        else
#endif
            std::sort(ord, ord + D);
        // Decode and re-probe for counts; the probe chain from mix(k) holds
        // only occupied slots up to k's.
        for (std::size_t t = 0; t < D; ++t) {
            const T x = RT::decode(static_cast<Key>(ord[t] ^ flip));
            const U k = bits(x);
            std::size_t h = mix(k, kBits);
            while (keys[h] != k) h = (h + 1) & kMask;
            vals[t] = x;
            runs[t] = cnt[h];
        }
#if FYX_HAS_AVX512_CODE
        if (use_avx512()) {
            vfill_runs_dispatch<T, std::uint32_t>(p, n, vals, runs, D);
            return true;
        }
#endif
        T* q = p;
        for (std::size_t t = 0; t < D; ++t) {
            std::fill(q, q + runs[t], vals[t]);
            q += runs[t];
        }
        return true;
    }
}

#ifndef FYX_FUSED_REVERSE_MIN_BYTES
#define FYX_FUSED_REVERSE_MIN_BYTES (std::size_t(64) << 10)
#endif
inline constexpr std::size_t kFusedReverseMinBytes = FYX_FUSED_REVERSE_MIN_BYTES;

/// First key strictly above the last in sort order and 16 strided samples
/// non-increasing: worth a fused verify+reverse attempt.
template <class T>
inline bool looks_reversed(const T* p, std::size_t n, bool descending) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    if (n < 64 || !(key(p[n - 1]) < key(p[0]))) return false;
    constexpr std::size_t S = 16;
    Key prev = key(p[0]);
    for (std::size_t s = 1; s <= S; ++s) {
        const Key k = key(p[(n - 1) * s / S]);
        if (prev < k) return false;
        prev = k;
    }
    return true;
}

/// Front probe of the small-vq dispatcher: 1 bitwise all-equal; 2 reversed
/// in place (looks_reversed, then the checked two-ended reverse without
/// undo -- a break leaves a permutation); 0 neither.  Out of line, one call
/// in place of the inlined all-equal test: an extra call site in the
/// dispatcher shifted GCC's inlining of the paths after it (i32 two-run / V
/// shapes 15-20% slower with the reverse attempt never taken).
template <class T>
FYX_NOINLINE int front_order_probe(T* p, std::size_t n, bool descending) {
    if (range_bitwise_all_equal(p, n)) return 1;
    if (n * sizeof(T) >= kFusedReverseMinBytes && looks_reversed(p, n, descending) &&
        reverse_tail_checked<T, false>(p, 0, n, descending) == 0)
        return 2;
    return 0;
}

// A handful of monotone runs (organ pipe, block swaps, concatenations, a
// rotation, mixed ascending / descending stretches): split into maximal runs
// (descending ones reversed), then pairwise merges ping-ponging through one
// leased buffer, each through the SIMD two-run merge when it applies.
// Detection is read-only and gives up past `max_runs` runs, so a declined
// range is untouched; random data declines within a few dozen keys.
template <class T>
inline bool try_few_runs_merge(T* p, std::size_t n, bool descending, std::size_t mono = 0) {
    constexpr std::size_t kMaxRuns = 8;
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    std::size_t bnd[kMaxRuns + 1];
    bool rev[kMaxRuns];
    const std::size_t min_run = std::max<std::size_t>(32, n / 64);
    // [lo, n) non-increasing?  Then reverse it in the same pass: blocks from
    // both ends are checked (OR-reduced, vectorizes) and swapped.  A break
    // swaps the done blocks back.  A strided sample gates the attempt.
    auto fused_reverse_tail = [&](std::size_t lo) -> bool {
        constexpr std::size_t S = 16, B = 64;
        if (n - lo < 4 * B) return false;
        Key prev = key(p[lo]);
        for (std::size_t s = 1; s <= S; ++s) {
            const Key kq = key(p[lo + (n - 1 - lo) * s / S]);
            if (prev < kq) return false;
            prev = kq;
        }
        // Vector kernel: hardware order on clean keys; a NaN / -0 hands
        // back to the key-order loop below (input restored).
        const int vk = reverse_tail_checked(p, lo, n, descending);
        if (vk != 2) return vk == 0;
        std::size_t l = lo, r = n;
        auto undo = [&]() {
            for (std::size_t k = 0; k < l - lo; ++k) std::swap(p[lo + k], p[n - 1 - k]);
        };
        while (r - l >= 2 * B) {
            Key bad = 0;
            for (std::size_t k = 0; k < B; ++k)
                bad = static_cast<Key>(bad | static_cast<Key>(key(p[l + k]) < key(p[l + k + 1])));
            for (std::size_t k = 0; k < B; ++k)
                bad = static_cast<Key>(bad | static_cast<Key>(key(p[r - B + k - 1]) < key(p[r - B + k])));
            if (bad) { undo(); return false; }
            for (std::size_t k = 0; k < B; ++k) std::swap(p[l + k], p[r - 1 - k]);
            l += B;
            r -= B;
        }
        for (std::size_t k = l; k + 1 < r; ++k)
            if (key(p[k]) < key(p[k + 1])) { undo(); return false; }
        std::reverse(p + l, p + r);
        return true;
    };
    std::size_t R = 0, i = 0;
    while (i < n) {
        if (R == kMaxRuns) return false;
        std::size_t start = i;
        std::size_t j = i + 1;
        const Key k0 = key(p[i]);
        while (j < n && key(p[j]) == k0) ++j;
        const bool down = j < n && key(p[j]) < k0;
        // A peak (ascending run, then a descending one) joins the
        // descending run: p[start - 1] >= p[start], so that run stays
        // non-increasing, and once reversed it starts at the peak -- the
        // classic organ pipe (sort, reverse the back half) then coalesces
        // with no data movement.
        if (down && R > 0 && !rev[R - 1] && start - bnd[R - 1] > min_run) --start;
        bool done = false;
        if (down && j < n && fused_reverse_tail(start)) { j = n; done = true; }
        // mono: no descent among the first `mono` keys (caller's scan).
        if (i == 0 && !down && j < mono) j = mono;
        // vectorised run end (non-strict in the run's own direction)
        if (j < n) j = radix_key_find_break(p, j, n, RT::encode(p[j - 1]), !down, descending);
        // Short runs mean local disorder (insertion's job) or noise.
        if (j - start < min_run) {
            if (done) std::reverse(p + start, p + n);   // leave a permutation of the input
            return false;
        }
        bnd[R] = start;
        rev[R] = down && !done;
        ++R;
        i = j;
    }
    bnd[R] = n;
    if (R < 2) return false;
    for (std::size_t r = 0; r < R; ++r)
        if (rev[r]) std::reverse(p + bnd[r], p + bnd[r + 1]);
    // Neighbours already in order (value-disjoint runs: organ pipe built by
    // reversing a sorted half, concatenations of ascending blocks) coalesce
    // for free; a pair in swapped order is one rotation.  Saves the merge
    // pass and the copy-back.
    {
        std::size_t w = 1;
        for (std::size_t r = 1; r < R; ++r) {
            if (!(key(p[bnd[r]]) < key(p[bnd[r] - 1]))) continue;
            bnd[w++] = bnd[r];
        }
        bnd[w] = n;
        R = w;
        if (R == 1) return true;
        if (R == 2 && !(key(p[0]) < key(p[n - 1]))) {
            // max(second) <= min(first): swap the blocks.
            std::rotate(p, p + bnd[1], p + n);
            return true;
        }
    }
    if (R == 2) {
        // Two runs: keys of the first run up to min(second) and of the
        // second from max(first) on are already final (binary searches);
        // merge only the overlap in place, buffering its smaller side.  An
        // organ pipe whose peak landed in the first run overlaps in one
        // key: one block move instead of merge-to-buffer plus copy-back.
        const std::size_t mid = bnd[1];
        const Key kb0 = key(p[mid]), ka1 = key(p[mid - 1]);
        std::size_t lo = 0, hi = mid;                      // first key(p[i]) > kb0 in [0, mid)
        while (lo < hi) { const std::size_t m = lo + (hi - lo) / 2; if (kb0 < key(p[m])) hi = m; else lo = m + 1; }
        const std::size_t s = lo;
        lo = mid; hi = n;                                  // first key(p[i]) >= ka1 in [mid, n)
        while (lo < hi) { const std::size_t m = lo + (hi - lo) / 2; if (key(p[m]) < ka1) lo = m + 1; else hi = m; }
        const std::size_t t = lo;
        const std::size_t a = mid - s, b = t - mid;        // both >= 1 (runs did not coalesce)
        const std::size_t m = std::min(a, b);
        ScratchLease<T> lease(m);
        std::unique_ptr<T[]> priv;
        if (!lease.valid()) priv.reset(new T[m]);
        T* tmp = lease.valid() ? lease.get() : priv.get();
        auto lower = [&](std::size_t l, std::size_t h, Key k) {   // first key >= k
            while (l < h) { const std::size_t q = l + (h - l) / 2; if (key(p[q]) < k) l = q + 1; else h = q; }
            return l;
        };
        auto upper = [&](std::size_t l, std::size_t h, Key k) {   // first key > k
            while (l < h) { const std::size_t q = l + (h - l) / 2; if (k < key(p[q])) h = q; else l = q + 1; }
            return l;
        };
        const bool gallop = m * 16 < std::max(a, b);
        if (a <= b) {
            // Forward: write index o never passes the next unread B slot.
            std::memcpy(static_cast<void*>(tmp), static_cast<const void*>(p + s), a * sizeof(T));
            std::size_t o = s, i = 0, j = mid;
            if (gallop) {
                for (; i < a; ++i) {
                    const std::size_t e = lower(j, t, key(tmp[i]));
                    std::memmove(static_cast<void*>(p + o), static_cast<const void*>(p + j), (e - j) * sizeof(T));
                    o += e - j; j = e;
                    p[o++] = tmp[i];
                }
            } else {
                while (i < a && j < t) {
                    const bool tb = key(p[j]) < key(tmp[i]);
                    p[o++] = tb ? p[j] : tmp[i];
                    j += tb ? 1u : 0u;
                    i += tb ? 0u : 1u;
                }
                if (i < a) std::memcpy(static_cast<void*>(p + o), static_cast<const void*>(tmp + i), (a - i) * sizeof(T));
            }
        } else {
            // Backward: write index o never passes below the next unread A slot.
            std::memcpy(static_cast<void*>(tmp), static_cast<const void*>(p + mid), b * sizeof(T));
            std::size_t o = t, i = b, j = mid;              // counts / ends
            if (gallop) {
                for (; i > 0; --i) {
                    const std::size_t e = upper(s, j, key(tmp[i - 1]));
                    std::memmove(static_cast<void*>(p + o - (j - e)), static_cast<const void*>(p + e), (j - e) * sizeof(T));
                    o -= j - e; j = e;
                    p[--o] = tmp[i - 1];
                }
            } else {
                while (i > 0 && j > s) {
                    const bool ta = key(tmp[i - 1]) < key(p[j - 1]);
                    p[--o] = ta ? p[j - 1] : tmp[i - 1];
                    j -= ta ? 1u : 0u;
                    i -= ta ? 0u : 1u;
                }
                if (i > 0) std::memcpy(static_cast<void*>(p + s), static_cast<const void*>(tmp), i * sizeof(T));
            }
        }
        return true;
    }
    ScratchLease<T> lease(n);
    std::unique_ptr<T[]> priv;
    if (!lease.valid()) priv.reset(new T[n]);
    T* buf = lease.valid() ? lease.get() : priv.get();
    T* src = p;
    T* dst = buf;
    while (R > 1) {
        std::size_t w = 0;
        for (std::size_t r = 0; r < R; r += 2) {
            const std::size_t lo = bnd[r];
            if (r + 1 == R) {
                std::memcpy(static_cast<void*>(dst + lo), static_cast<const void*>(src + lo),
                            (bnd[r + 1] - lo) * sizeof(T));
            } else {
                const std::size_t mid = bnd[r + 1], hi = bnd[r + 2];
                merge_runs_galloping(src + lo, mid - lo, src + mid, hi - mid, dst + lo, descending);
            }
            bnd[w++] = lo;
        }
        bnd[w] = n;
        R = w;
        std::swap(src, dst);
    }
    if (src != p) std::memcpy(static_cast<void*>(p), static_cast<const void*>(src), n * sizeof(T));
    return true;
}

// Extract-and-merge repair for "sorted with a minority out of place": a
// sorted range with a few far displacements, a sorted prefix followed by an
// unsorted tail, two (or a few) concatenated sorted runs.  One pass keeps a
// non-decreasing subsequence in place (compacted to the front) and moves the
// rest to a side buffer: on a descent either the new key or the previous kept
// key -- whichever breaks the order -- is set aside.  The side buffer is then
// sorted (if it is not already) and merged back from the right, branch-free.
// Cost on success: one scan + sort(side) + one merge, against a full sort.
//
// Random-like input is rejected after a few hundred keys: the abort tests are
// (a) the kept subsequence falls below half of what was scanned, (b) the side
// buffer turns disordered (descents inside it beyond i/8 + 64 or n/16 + 64),
// (c) the side buffer exceeds n/2.  On abort the side keys are written back
// behind the kept ones, so the range is still a permutation of the input.
template <class T, class SideSort>
inline bool try_extract_merge_repair(T* p, std::size_t n, bool descending, SideSort&& side_sort) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    if (n < 256) return false;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };

    const std::size_t cap = n / 2 + 1;
    // One leased block holds the side buffer and the merge output; the side
    // sort below nests and so allocates privately if it needs scratch.
    ScratchLease<T> lease(cap + n);
    std::unique_ptr<T[]> priv;
    if (!lease.valid()) priv.reset(new T[cap + n]);
    T* const sd  = lease.valid() ? lease.get() : priv.get();
    T* const obuf = sd + cap;
    std::size_t m = 0;
    std::size_t k = 1, side_desc = 0;
    Key klast = key(p[0]);
    Key sidelast = 0;
    bool ok = true;
    bool kept_last = true;                 // retry the block skip only after a keep
    constexpr std::size_t kBlk = 16;
    std::size_t i = 1;
    for (; i < n; ++i) {
        // Ordered stretches move 16 keys at a time (branch-free test, then a
        // block copy -- nothing to copy at all before the first extraction).
        while (kept_last && i + kBlk <= n) {
            // Same-width lane sum: vectorizes (a shifted OR mask does not).
            Key bd = static_cast<Key>(key(p[i]) < klast);
            for (std::size_t j = 1; j < kBlk; ++j)
                bd = static_cast<Key>(bd + static_cast<Key>(key(p[i + j]) < key(p[i + j - 1])));
            std::size_t f = kBlk;
            if (bd != 0) {
                // Keep the ordered prefix in one go and hand the first
                // descent to the scalar step (re-testing the same block
                // key by key cost ~16x near every outlier).
                f = 0;
                if (!(key(p[i]) < klast)) {
                    f = 1;
                    while (!(key(p[i + f]) < key(p[i + f - 1]))) ++f;
                }
            }
            if (f != 0) {
                if (k != i) {
                    // k < i (may overlap): forward element copy, no libc call.
                    for (std::size_t j = 0; j < f; ++j) p[k + j] = p[i + j];
                }
                k += f;
                i += f;
                klast = key(p[k - 1]);
            }
            if (f != kBlk) break;
        }
        if (i >= n) break;
        const T x = p[i];
        const Key kx = key(x);
        if (!(kx < klast)) {               // in order: keep
            p[k++] = x;
            klast = kx;
            kept_last = true;
            continue;
        }
        kept_last = false;
        // Descent.  If at most 8 kept keys exceed x (and at least one kept
        // key does not), those are the outliers: move them aside and keep x.
        // Otherwise x is the outlier.
        std::size_t j = 1;
        while (j < 8 && j < k && kx < key(p[k - 1 - j])) ++j;
        if (j < k && !(kx < key(p[k - 1 - j]))) {
            for (std::size_t q = k - j; q < k; ++q) {
                const Key kq = key(p[q]);
                if (m != 0 && kq < sidelast) ++side_desc;
                sidelast = kq;
                sd[m++] = p[q];
            }
            k -= j;
            p[k++] = x;
            klast = kx;
        } else {
            if (m != 0 && kx < sidelast) ++side_desc;
            sidelast = kx;
            sd[m++] = x;
        }
        if ((i & 255u) == 0 || m + 8 >= cap) {
            if (2 * k + 64 < i || m + 8 >= cap ||
                side_desc > (i >> 3) + 64 || side_desc > (n >> 4) + 64) {
                ok = false;
                ++i;
                break;
            }
        }
    }
    if (ok && side_desc > (n >> 4) + 64) ok = false;
    if (!ok) {
        // p[0..k) kept, side holds the rest of [0, i), p[i..n) untouched.
        std::copy(sd, sd + m, p + k);
        return false;
    }
    if (m == 0) return true;
    if (side_desc != 0) side_sort(sd, m);
    // Backward merge: kept run in p[0..k), side run in side[0..m), k + m == n.
    if (m * 32 < k) {
        // Few side keys: binary-search each one's slot and shift whole kept
        // blocks (memmove bandwidth instead of a compare per key).
        std::size_t hi = k;                // kept keys [0, hi) not yet placed
        for (std::size_t b = m; b-- > 0;) {
            const Key kb = key(sd[b]);
            std::size_t lo = 0, h = hi;    // first kept index with key > kb
            while (lo < h) {
                const std::size_t mid = lo + (h - lo) / 2;
                if (kb < key(p[mid])) h = mid; else lo = mid + 1;
            }
            const std::size_t cnt = hi - lo;
            if (cnt) std::memmove(static_cast<void*>(p + lo + b + 1), static_cast<const void*>(p + lo), cnt * sizeof(T));
            p[lo + b] = sd[b];
            hi = lo;
        }
        return true;
    }
    // Large side: four-way merge into a scratch buffer, then copy back.
    // (A[k] = p[k] is addressable since m >= 1; sd has cap >= m + 1.)
    if (!vmerge_runs(p, k, sd, m, obuf, descending))
        merge_runs_4way(p, k, sd, m, obuf, key);
    std::memcpy(static_cast<void*>(p), static_cast<const void*>(obuf), n * sizeof(T));
    return true;
}


// Few isolated outliers in an otherwise sorted range (far swaps, a handful
// of misplaced keys): locate every descent with a vectorized block test,
// blame one key per descent, verify the remainder is ordered, then compact,
// sort the (<= 2 * kMaxD) outliers and insert them.  Only a constant number of
// data-dependent branches per descent, so it stays fast on cold predictors
// where the streaming extractor pays several mispredictions per outlier.
// Returns false (range untouched) when the blame/verify step does not hold.
template <class T>
inline bool sparse_outlier_repair_at(T* p, std::size_t n, const std::uint32_t* pos, std::size_t D,
                                     bool descending) {
    using RT  = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr std::size_t kMaxD = 32;
    const Key flip = descending ? static_cast<Key>(~Key(0)) : Key(0);
    auto key = [flip](const T& x) -> Key { return static_cast<Key>(RT::encode(x) ^ flip); };
    if (n < 64 || D > kMaxD) return false;
    if (D == 0) return true;
    // Blame: p[i-1] when dropping it leaves p[i-2] <= p[i], else p[i] when
    // dropping it leaves p[i-1] <= p[i+1].  rem[] stays ascending.
    std::size_t rem[kMaxD];
    std::size_t r = 0;
    for (std::size_t d = 0; d < D; ++d) {
        const std::size_t i = pos[d];
        const bool prev_removed = r != 0 && rem[r - 1] == i - 1;
        if (!prev_removed && (i < 2 || !(key(p[i]) < key(p[i - 2])))) {
            rem[r++] = i - 1;
        } else if (i + 1 >= n || !(key(p[i + 1]) < key(p[i - 1]))) {
            rem[r++] = i;
        } else {
            return false;
        }
    }
    // Verify: every kept neighbour pair spanning a removed index is ordered
    // (pairs not touching one were not descents).
    for (std::size_t q = 0; q < r;) {
        std::size_t q2 = q;
        while (q2 + 1 < r && rem[q2 + 1] == rem[q2] + 1) ++q2;    // consecutive
        const std::size_t lo = rem[q], hi = rem[q2];
        if (lo > 0 && hi + 1 < n && key(p[hi + 1]) < key(p[lo - 1])) return false;
        q = q2 + 1;
    }
    // Outliers to a small buffer, insertion-sorted by key.
    T out[kMaxD];
    for (std::size_t q = 0; q < r; ++q) out[q] = p[rem[q]];
    for (std::size_t a = 1; a < r; ++a) {
        const T x = out[a];
        const Key kx = key(x);
        std::size_t b = a;
        while (b > 0 && kx < key(out[b - 1])) { out[b] = out[b - 1]; --b; }
        out[b] = x;
    }
    // Kept keys are addressed in place: kept index k lives at k plus the
    // number of removed indices at or before it (rem[] ascending, few).
    auto orig = [&](std::size_t k) {
        for (std::size_t q = 0; q < r && rem[q] <= k; ++q) ++k;
        return k;
    };
    // Insertion slots (first kept index with key > outlier), all searches
    // in lockstep so their load latencies overlap.
    const std::size_t kept = n - r;
    std::size_t slot[kMaxD];
    Key okey[kMaxD];
    for (std::size_t b = 0; b < r; ++b) { slot[b] = 0; okey[b] = key(out[b]); }
    std::size_t len = kept;
    while (len > 1) {
        const std::size_t half = len / 2;
        for (std::size_t b = 0; b < r; ++b)
            slot[b] += (std::size_t(0) - static_cast<std::size_t>(!(okey[b] < key(p[orig(slot[b] + half)])))) & half;
        len -= half;
    }
    if (len == 1)
        for (std::size_t b = 0; b < r; ++b) slot[b] += static_cast<std::size_t>(!(okey[b] < key(p[orig(slot[b])])));
    // Kept key k moves from k + R(k) (removed before it) to k + C(k)
    // (outliers placed before it): piecewise constant between the
    // breakpoints rem[q] - q and slot[b].  Segments with no net shift --
    // everything between the two ends of a far swap -- stay where they
    // are; left-moving segments go left to right, then right-moving ones
    // right to left (C, R non-decreasing: no move overwrites a source that
    // is still to be read).
    std::size_t bp[2 * kMaxD + 2];
    std::size_t nb = 0;
    for (std::size_t q = 0; q < r; ++q) {
        const std::size_t x[2] = {rem[q] - q, slot[q]};
        for (std::size_t x1 : x) {
            if (x1 == 0 || x1 >= kept) continue;
            std::size_t j = nb++;
            while (j > 0 && bp[j - 1] > x1) { bp[j] = bp[j - 1]; --j; }
            bp[j] = x1;
        }
    }
    {
        std::size_t u = 0;
        for (std::size_t j = 0; j < nb; ++j) if (u == 0 || bp[u - 1] != bp[j]) bp[u++] = bp[j];
        nb = u;
    }
    struct Seg { std::size_t k0, len, src, dst; };
    Seg seg[2 * kMaxD + 2];
    std::size_t ns = 0;
    {
        std::size_t k0 = 0, R = 0, C = 0;
        for (std::size_t t = 0; t <= nb; ++t) {
            const std::size_t k1 = t < nb ? bp[t] : kept;
            while (R < r && rem[R] - R <= k0) ++R;
            while (C < r && slot[C] <= k0) ++C;
            if (k1 > k0) seg[ns++] = Seg{k0, k1 - k0, k0 + R, k0 + C};
            k0 = k1;
        }
    }
    for (std::size_t t = 0; t < ns; ++t)
        if (seg[t].dst < seg[t].src)
            std::memmove(static_cast<void*>(p + seg[t].dst), static_cast<const void*>(p + seg[t].src), seg[t].len * sizeof(T));
    for (std::size_t t = ns; t-- > 0;)
        if (seg[t].dst > seg[t].src)
            std::memmove(static_cast<void*>(p + seg[t].dst), static_cast<const void*>(p + seg[t].src), seg[t].len * sizeof(T));
    for (std::size_t b = 0; b < r; ++b) p[slot[b] + b] = out[b];
    return true;
}

template <class T>
inline bool try_sparse_outlier_repair(T* p, std::size_t n, std::size_t from, bool descending) {
    if (n > 0xFFFFFFFFull) return false;
    std::uint32_t pos[32];
    const std::size_t D = descent_positions(p, from, n, descending, pos, 32);
    return D <= 32 && sparse_outlier_repair_at(p, n, pos, D, descending);
}

// Insertion repair over radix keys (the library's total order, so NaN / -0
// are placed exactly as the radix paths place them) with a hard budget on
// element moves.  Near-sorted input -- a few local displacements -- finishes
// in one predictable pass; anything else exhausts the budget early and
// returns false, leaving a permutation of the input for the caller's sort.
template <class T>
inline bool budgeted_insertion_repair(T* p, std::size_t n, bool descending,
                                      std::size_t budget) {
    using RT = RadixTraits<T>;
    constexpr std::size_t kSkip = 16;
    std::size_t moves = 0;
    std::size_t i = 1;
    while (i < n) {
        // Branch-free skip over ordered stretches, 16 neighbours at a time.
        while (i + kSkip <= n) {
            unsigned bad = 0;
            for (std::size_t k = 0; k < kSkip; ++k) {
                const auto a = RT::encode(p[i + k - 1]);
                const auto b = RT::encode(p[i + k]);
                bad |= static_cast<unsigned>(descending ? (a < b) : (b < a));
            }
            if (bad) break;
            i += kSkip;
        }
        const std::size_t e = std::min(n, i + kSkip);
        for (; i < e; ++i) {
        const auto ki = RT::encode(p[i]);
        const auto kp = RT::encode(p[i - 1]);
        if (descending ? !(kp < ki) : !(ki < kp)) continue;
        const T x = p[i];
        std::size_t j = i;
        do {
            p[j] = p[j - 1];
            --j;
        } while (j > 0 && (descending ? (RT::encode(p[j - 1]) < ki) : (ki < RT::encode(p[j - 1]))));
        p[j] = x;
        moves += i - j;
        if (moves > budget) return false;
        }
    }
    return true;
}

// Same hint as a predicate: true when the first `limit` neighbour pairs hold
// at most `max_inv` inversions.  Counts in blocks of 32 pairs and stops as
// soon as the allowance is exceeded, so random input -- half its pairs
// inverted -- is rejected after the first block instead of after `limit`.
template <class T>
inline bool head_inversions_within(const T* p, std::size_t n, bool descending,
                                   std::size_t limit, std::size_t max_inv) {
    using RT = RadixTraits<T>;
    const std::size_t m = std::min(n == 0 ? std::size_t(0) : n - 1, limit);
    std::size_t c = 0;
    for (std::size_t b = 0; b < m; b += 32) {
        const std::size_t e = std::min(m, b + 32);
        unsigned bc = 0;
        if (descending) {
            for (std::size_t i = b; i < e; ++i)
                bc += static_cast<unsigned>(RT::encode(p[i]) < RT::encode(p[i + 1]));
        } else {
            for (std::size_t i = b; i < e; ++i)
                bc += static_cast<unsigned>(RT::encode(p[i + 1]) < RT::encode(p[i]));
        }
        c += bc;
        if (c > max_inv) return false;
    }
    return true;
}

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

// First descent found by the serial order exit's ascending scan, so the
// affix weapon tried right after it (nothing in between moves elements)
// need not rescan the ordered head.  Reset with the vq memo per top-level
// call; consumed (cleared) by its reader; checked against (p, n).
struct OrderBreakHint {
    const void* p = nullptr;
    std::size_t n = 0;
    std::size_t head = 0;
};
inline OrderBreakHint& order_break_hint() noexcept {
    static thread_local OrderBreakHint h;
    return h;
}

// any_adjacent_pair: does pred(p[j], p[j-1]) (asc) / pred(p[j-1], p[j])
// (!asc) hold for some j in [lo, hi)?  OR-reduced with no early exit so an
// inlined key comparator vectorizes; the ISA-targeted copies let a portable
// build reach 64-bit vector compares (SSE2 has none).
#define FYX_ANY_ADJ_BODY                                                                       \
    unsigned char bad = 0;                                                                    \
    if (asc) for (std::size_t j = lo; j < hi; ++j) bad |= static_cast<unsigned char>(pred(p[j], p[j - 1])); \
    else     for (std::size_t j = lo; j < hi; ++j) bad |= static_cast<unsigned char>(pred(p[j - 1], p[j])); \
    return bad != 0;
template <class T, class Pred>
inline bool any_adjacent_pair_default(const T* p, std::size_t lo, std::size_t hi, Pred& pred, bool asc) {
    FYX_ANY_ADJ_BODY
}
#if FYX_ARCH_X86 && FYX_GNUC_LIKE && !defined(__AVX512BW__)
template <class T, class Pred>
FYX_TARGET_AVX512 bool any_adjacent_pair_avx512(const T* p, std::size_t lo, std::size_t hi, Pred& pred, bool asc) {
    FYX_ANY_ADJ_BODY
}
template <class T, class Pred>
FYX_TARGET_AVX2 bool any_adjacent_pair_avx2(const T* p, std::size_t lo, std::size_t hi, Pred& pred, bool asc) {
    FYX_ANY_ADJ_BODY
}
#endif
#undef FYX_ANY_ADJ_BODY
template <class T, class Pred>
inline bool any_adjacent_pair(const T* p, std::size_t lo, std::size_t hi, Pred& pred, bool asc) {
#if FYX_ARCH_X86 && FYX_GNUC_LIKE && !defined(__AVX512BW__)
    if (use_avx512()) return any_adjacent_pair_avx512(p, lo, hi, pred, asc);
    if (use_avx2())   return any_adjacent_pair_avx2(p, lo, hi, pred, asc);
#endif
    return any_adjacent_pair_default(p, lo, hi, pred, asc);
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
    constexpr bool blocked_order = radix_supported_v<T> &&
        (is_ascending_v<Comp, T> || is_descending_v<Comp, T>);
    if constexpr (blocked_order) {
        return blocked_radix_order_kind(p, n, is_descending_v<Comp, T>);
    } else if constexpr (radix_order) {
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

        // Once the direction is known the rest is a block-wise OR-reduced
        // scan (vectorizes for plain key comparators, ISA-dispatched), with
        // an exit per block.
        // Returns the block holding the first break (n if none).
        auto any_break = [&](std::size_t from, bool asc) -> std::size_t {
            constexpr std::size_t B = 1024;
            for (std::size_t b = from; b < n; b += B)
                if (any_adjacent_pair(p, b, std::min(n, b + B), comp, asc)) return b;
            return n;
        };
        for (std::size_t i = 1; i < n; ++i) {
            if (comp(p[i], p[i - 1]))            // reverse of comp order
                return any_break(i + 1, false) != n ? FastOrderKind::None : FastOrderKind::Reverse;
            if (comp(p[i - 1], p[i])) {          // already in comp order
                std::size_t b = any_break(i + 1, true);
                if (b == n) return FastOrderKind::Sorted;
                while (!comp(p[b], p[b - 1])) ++b;
                OrderBreakHint& h = order_break_hint();
                h.p = p; h.n = n; h.head = b;
                return FastOrderKind::None;
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
        std::size_t shifts = 0, strikes = 0, next_check = 64, walk_until = 0;
        for (std::size_t i = 1; i < n; ++i) {
            if (!before(p[i], p[i - 1])) {
                // Skip an ordered stretch with one vectorized block test
                // (only positions ahead of i are read; none were moved yet).
                // A block that holds a descent is walked element-wise.
                if (i >= walk_until) {
                    const std::size_t e = std::min(n, i + 128u);
                    if (i + 16u < e && !any_adjacent_pair(p, i + 1, e, before, true)) i = e - 1;
                    else walk_until = e;
                }
                continue;
            }
            // Refuse long travel before paying for it: p[i] below the element
            // `reach` back must cross more than that, and a lone intruder at
            // i-1 above the element `reach` ahead would be bubbled forward one
            // shift per element all the way (far swaps: ~2 ms of shifts per
            // 1M records before the old in-loop check fired).
            if (i >= reach && before(p[i], p[i - reach])) return false;
            // Two neighbours that both travel more than 4096 back are the
            // front of a moved block (each one would pay the whole distance).
            if (i >= 4096 && i + 1 < n && before(p[i + 1], p[i - 4096]) &&
                before(p[i], p[i - 4096])) return false;
            if (i >= 2 && i + reach < n && !before(p[i], p[i - 2]) &&
                before(p[i + reach], p[i - 1])) return false;
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

// try_bitonic_runs_sort: above this footprint merge in place from a buffer
// holding only the descending run.
inline constexpr std::size_t kBitonicHalfBufferBytes = std::size_t(4) << 20;

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

        // Block OR-reductions without early exits (they vectorize for plain
        // key comparators); a hit is located inside its block afterwards.
        constexpr std::size_t kScanBlock = 512;
        auto any_break = [&](std::size_t lo, std::size_t hi, bool asc) -> bool {
            return any_adjacent_pair(p, lo, hi, before, asc);
        };
        std::size_t split = n;
        for (std::size_t b = i; b < n; b += kScanBlock) {
            const std::size_t e = std::min(n, b + kScanBlock);
            if (!any_break(b, e, first_asc)) continue;
            for (std::size_t j = b; j < e; ++j)
                if (first_asc ? before(p[j], p[j - 1]) : before(p[j - 1], p[j])) { split = j; break; }
            break;
        }
        if (split == n) return false;
        if (split < n / 16u || n - split < n / 16u) return false;
        for (std::size_t b = split + 1; b < n; b += kScanBlock)
            if (any_break(b, std::min(n, b + kScanBlock), !first_asc)) return false;

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
        // Value-disjoint runs (organ pipe built as sort-then-reverse-half,
        // and its mirror): after the descending run is reversed in place the
        // range is sorted, or one rotation from sorted -- no merge pass.
        // X = ascending run, Y = descending run; the test reads their ends.
        {
            const std::size_t xl = first_asc ? 0 : split, xh = first_asc ? split - 1 : n - 1;
            const std::size_t yl = first_asc ? split : 0, yh = first_asc ? n - 1 : split - 1;
            // Y's minimum is p[yh], its maximum p[yl].
            const bool x_then_y = !before(p[yh], p[xh]);   // max X <= min Y
            const bool y_then_x = !before(p[xl], p[yl]);   // max Y <= min X
            if (x_then_y || y_then_x) {
                std::reverse(p + yl, p + yh + 1);
                if (first_asc ? y_then_x && !x_then_y : x_then_y && !y_then_x)
                    std::rotate(p, p + split, p + n);
                return true;
            }
        }

        if (try_arithmetic_organ_fill()) return true;

        if constexpr (std::is_trivially_copyable<T>::value) {
          if (n * sizeof(T) > kBitonicHalfBufferBytes) {
            // Memory-bound size: save only the descending run (reversed, so
            // ascending) and merge in place toward it -- about 40% less
            // traffic than merge-to-buffer plus copy back.  first_asc: the
            // back chain fills p from n-1 down (write index = X tail + buffer
            // left, never past an unread X slot).  Otherwise the front chain
            // fills from 0 up (write index <= next unread Y slot).
            const std::size_t m = first_asc ? n - split : split;
            ScratchLease<T> tmp_lease(m);
            if (!tmp_lease.valid()) return false;
            T* tmp = tmp_lease.get();
            const T* src = first_asc ? p + split : p;
            for (std::size_t k = 0; k < m; ++k) tmp[k] = src[m - 1 - k];
            if (first_asc) {
                std::size_t xr = split, tr = m, o = n;   // counts left
                while (xr != 0 && tr != 0) {
                    const bool tx = before(tmp[tr - 1], p[xr - 1]);
                    p[--o] = *(tx ? p + (xr - 1) : tmp + (tr - 1));
                    xr -= tx ? 1u : 0u;
                    tr -= tx ? 0u : 1u;
                }
                if (tr != 0) std::memcpy(p, tmp, tr * sizeof(T));
            } else {
                std::size_t t = 0, y = split, o = 0;
                while (t != m && y != n) {
                    const bool ty = before(p[y], tmp[t]);
                    p[o++] = *(ty ? p + y : tmp + t);
                    y += ty ? 1u : 0u;
                    t += ty ? 0u : 1u;
                }
                if (t != m) std::memcpy(p + o, tmp + t, (m - t) * sizeof(T));
            }
            return true;
          }
            ScratchLease<T> tmp_lease(n);
            if (!tmp_lease.valid()) return false;
            T* tmp = tmp_lease.get();
            // X ascends over [x, xe], Y descends over [ye, y] (so p[y] is its
            // smallest).  Two branchless chains: the front emits the smallest
            // (ties -> X), the back the largest (ties -> Y), which is one
            // consistent total order.  Each round runs h steps per chain with
            // h = min(remaining X, remaining Y) / 2, so neither chain can
            // drain a run inside the round: no bounds checks in the loop.
            std::size_t x  = first_asc ? 0 : split;
            std::size_t xe = first_asc ? split - 1 : n - 1;
            std::size_t ye = first_asc ? split : 0;
            std::size_t y  = first_asc ? n - 1 : split - 1;
            std::size_t of = 0, ob = n - 1;
            while (true) {
                const std::size_t xr = xe - x + 1, yr = y - ye + 1;
                const std::size_t h = std::min(xr, yr) / 2;
                if (h < 4) break;
                for (std::size_t k = 0; k < h; ++k) {
                    const bool ty = before(p[y], p[x]);
                    tmp[of + k] = *(ty ? p + y : p + x);
                    x += ty ? 0u : 1u;
                    y -= ty ? 1u : 0u;
                    const bool tx = before(p[ye], p[xe]);
                    tmp[ob - k] = *(tx ? p + xe : p + ye);
                    xe -= tx ? 1u : 0u;
                    ye += tx ? 0u : 1u;
                }
                of += h;
                ob -= h;
            }
            // Tail, front only.  Runs stay non-empty here (each round left
            // at least half of the smaller one); x > xe marks X drained.
            while (x <= xe) {
                const bool ty = before(p[y], p[x]);
                tmp[of++] = *(ty ? p + y : p + x);
                if (!ty) { ++x; continue; }
                if (y == ye) { ye = y + 1; break; }
                --y;
            }
            while (x <= xe) tmp[of++] = p[x++];
            if (ye <= y) for (std::size_t k = y + 1; k-- > ye;) tmp[of++] = p[k];
            if (of != ob + 1) return false;
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

// The vector quicksort's NaN / -0 screen is a property of the multiset, so it
// survives every permutation the declining probes may leave behind.  One
// top-level sort call screens a given (pointer, length) at most once: the
// memo is cleared on entry to and exit from sort_pointer_core_impl (a full
// extra read of a 200K double range is ~55us, and up to three gates asked).
struct VqCleanMemo {
    const void* p = nullptr;
    std::size_t n = 0;
    bool clean = false;
};
inline VqCleanMemo& vq_clean_memo() noexcept {
    static thread_local VqCleanMemo m;
    return m;
}
struct VqCleanMemoScope {
    VqCleanMemoScope() noexcept { vq_clean_memo().p = nullptr; order_break_hint().p = nullptr; }
    ~VqCleanMemoScope() { vq_clean_memo().p = nullptr; order_break_hint().p = nullptr; }
    VqCleanMemoScope(const VqCleanMemoScope&) = delete;
    VqCleanMemoScope& operator=(const VqCleanMemoScope&) = delete;
};
template <class T>
inline bool vq_range_clean_memo(const T* p, std::size_t n) {
    VqCleanMemo& m = vq_clean_memo();
    if (m.p == static_cast<const void*>(p) && m.n == n) return m.clean;
    const bool c = vqsort_range_clean(p, n);
    m.p = p;
    m.n = n;
    m.clean = c;
    return c;
}

// True when the AVX-512 vector quicksort will accept this range.  Its
// equal-key handling beats the hash-probed sparse counters on every
// few-distinct shape measured (Ice Lake-SP, 200K serial: int32 16 values
// 133us vs 523us, 256 values 247us vs 517us; int64 16 values 298us vs
// 363us), so sparse counting is skipped where the quicksort is available.
// Dense O(n + range) counting is unaffected.
template <class T>
inline bool vq_beats_sparse_count(const T* p, std::size_t n) {
    if constexpr (!vqsort_kernel_supported_v<T>) { (void)p; (void)n; return false; }
    else {
        return vqsort_preferred<T>() && vqsort_usable<T>(n) && n >= kVqsortMinN &&
               vq_range_clean_memo(p, n);
    }
}

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
                if (vq_beats_sparse_count(p, n)) return false;
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
// try_kv16_early_sort: below kKv16EarlyMinN the probe stack's small-n paths
// win; from kKv16EarlyParallelMinN up parallel-eligible calls fork the
// quicksort's top levels.
inline constexpr std::size_t kKv16EarlyMinN          = 2048;
inline constexpr std::size_t kKv16EarlyParallelMinN  = std::size_t(1) << 17;
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
#if FYX_HAS_AVX512_CODE
FYX_DIAG_PUSH_SIMD
template <class T>
FYX_TARGET_AVX512 inline bool radix_key_monotone_scan_avx512(
        const T* p, std::size_t start, std::size_t hi,
        typename RadixTraits<T>::Key prev_key,
        const bool want_up, const bool descending) {
    using RT = RadixTraits<T>;
    using Key = typename RT::Key;
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
            if constexpr (std::is_unsigned_v<T>) e = k;
            else e = _mm512_xor_si512(k, vsign);
        } else if constexpr (sizeof(Key) == 8) {
            const __m512i t = _mm512_srai_epi64(k, 63);
            e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        } else {
            const __m512i t = _mm512_srai_epi32(k, 31);
            e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        }
        const __m512i shifted = (sizeof(Key) == 8)
            ? _mm512_alignr_epi64(e, prev_vec, 7)
            : _mm512_alignr_epi32(e, prev_vec, 15);
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
FYX_DIAG_POP_SIMD
#endif

template <class T>
inline bool radix_key_monotone_scan(const T* p, std::size_t start, std::size_t hi,
                                    typename RadixTraits<T>::Key prev_key,
                                    const bool want_up, const bool descending) {
#if FYX_HAS_AVX512_CODE
    if (use_avx512())
        return radix_key_monotone_scan_avx512(p, start, hi, prev_key, want_up, descending);
#endif
    using Key = typename RadixTraits<T>::Key;
    Key prev = prev_key;
    for (std::size_t i = start; i < hi; ++i) {
        const Key cur = RadixTraits<T>::encode(p[i]);
        const bool bad = want_up ? (descending ? (prev < cur) : (cur < prev))
                                 : (descending ? (cur < prev) : (prev < cur));
        if (bad) return false;
        prev = cur;
    }
    return true;
}

// Vectorised all-equal sweep. The AVX-512 body is isolated in a target-specific
// helper; its baseline wrapper dispatches only after the runtime feature check.
#if FYX_HAS_AVX512_CODE
FYX_DIAG_PUSH_SIMD
template <class T>
FYX_TARGET_AVX512 inline bool range_all_equal_first_avx512(
        const T* p, std::size_t lo, std::size_t hi) {
    constexpr bool use_key = radix_supported_v<T> && std::is_floating_point<T>::value;
    using RT = RadixTraits<T>;
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
    const __mmask64 full = (width == 8) ? __mmask64(0xFF)
                         : (width == 4) ? __mmask64(0xFFFF)
                         : (width == 2) ? __mmask64(0xFFFFFFFFull)
                                        : ~__mmask64(0);
    std::size_t i = lo;
    for (; i + width * 8 <= hi; i += width * 8) {
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
        } else if (p[i] != first) return false;
    }
    return true;
}
FYX_DIAG_POP_SIMD
#endif

template <class T>
inline bool range_all_equal_first_vec(const T* p, std::size_t lo, std::size_t hi) {
    if (lo >= hi) return true;
    if constexpr (!std::is_arithmetic<T>::value) {
        (void)p;
        return false;
    } else {
#if FYX_HAS_AVX512_CODE
        if (use_avx512()) return range_all_equal_first_avx512(p, lo, hi);
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
#if FYX_HAS_AVX512_CODE
FYX_DIAG_PUSH_SIMD
template <class T>
FYX_TARGET_AVX512 inline bool proof_structured_scan_avx512(
        const T* p, std::size_t n, bool descending,
        std::size_t* brk, unsigned& nb) {
    using RT = RadixTraits<T>;
    using Key = typename RT::Key;
    constexpr unsigned lanes = (sizeof(Key) == 8) ? 8u : 16u;
    const __m512i vsign = (sizeof(Key) == 8)
        ? _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL))
        : _mm512_set1_epi32(static_cast<int>(0x80000000u));
    __m512i prev_vec = (sizeof(Key) == 8)
        ? _mm512_set1_epi64(static_cast<long long>(RT::encode(p[0])))
        : _mm512_set1_epi32(static_cast<int>(RT::encode(p[0])));
    std::size_t i = 1;
    for (; i + lanes <= n; i += lanes) {
        const __m512i k = _mm512_loadu_si512(reinterpret_cast<const void*>(p + i));
        __m512i e;
        if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_unsigned_v<T>) e = k;
            else e = _mm512_xor_si512(k, vsign);
        } else if constexpr (sizeof(Key) == 8) {
            const __m512i t = _mm512_srai_epi64(k, 63);
            e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        } else {
            const __m512i t = _mm512_srai_epi32(k, 31);
            e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        }
        const __m512i shifted = (sizeof(Key) == 8)
            ? _mm512_alignr_epi64(e, prev_vec, 7)
            : _mm512_alignr_epi32(e, prev_vec, 15);
        unsigned bad = (sizeof(Key) == 8)
            ? (descending ? _mm512_cmplt_epu64_mask(shifted, e)
                          : _mm512_cmplt_epu64_mask(e, shifted))
            : (descending ? _mm512_cmplt_epu32_mask(shifted, e)
                          : _mm512_cmplt_epu32_mask(e, shifted));
        while (bad) {
            if (nb == kProofStructMaxBreaks) return false;
            const unsigned bit = static_cast<unsigned>(__builtin_ctz(bad));
            brk[nb++] = i + bit;
            bad &= bad - 1;
        }
        prev_vec = e;
    }
    Key prev = RT::encode(p[i - 1]);
    for (; i < n; ++i) {
        const Key cur = RT::encode(p[i]);
        if (descending ? (prev < cur) : (cur < prev)) {
            if (nb == kProofStructMaxBreaks) return false;
            brk[nb++] = i;
        }
        prev = cur;
    }
    return true;
}
FYX_DIAG_POP_SIMD
#endif

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
            if (!proof_structured_scan_avx512(p, n, descending, brk, nb)) return false;
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
                                          std::size_t offset, bool& descending,
                                          std::size_t samples = 96) {
    using Key = typename RadixTraits<Field>::Key;
    if (offset + sizeof(Field) > sizeof(T)) return false;

    const std::size_t s = std::min<std::size_t>(n, std::min<std::size_t>(samples, 96));
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
                                                std::size_t offset,
                                                std::size_t* distinct_out = nullptr) {
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
    if (distinct_out) *distinct_out = distinct;
    return true;
}

// In-place counting for records with <= 256 distinct sampled-equivalent
// keys.  The out-of-place version below reads and writes the records three
// times (scatter, copy back, check); here a counting pass also records each
// record's class in a byte array, an American-flag cycle permutation moves
// every record once, and a read-only pass proves the order with `comp`.
// Returns 0 = declined untouched, 1 = sorted, 2 = permuted but the proof
// failed (the caller must sort it some other way).
// Class bytes for p[lo, hi): ids[i] = class of p[i] (first-seen order),
// id_key[c] = key of class c, cnt[c] = its count.  Returns the class count,
// or SIZE_MAX past kCountingClassLimit.  Nothing in p moves.
template <class Field, class T>
inline std::size_t flag_classify_range(const T* p, std::size_t lo, std::size_t hi,
                                       std::size_t offset, unsigned char* ids,
                                       typename RadixTraits<Field>::Key* id_key,
                                       std::size_t* cnt) {
    using Key = typename RadixTraits<Field>::Key;
    static_assert(kCountingClassLimit <= 256, "class ids are bytes");
    constexpr std::size_t Cap = 1024, Mask = Cap - 1;
    std::array<Key, Cap> keys{};
    std::array<unsigned short, Cap> slot_id{};      // 0 = empty, else id + 1
    std::size_t distinct = 0;
    if (lo >= hi) return 0;
    Key last_k = load_trivial_field_key<Field>(p[lo], offset);
    unsigned last_id = 256;
    for (std::size_t i = lo; i < hi; ++i) {
        const Key k = load_trivial_field_key<Field>(p[i], offset);
        if (k == last_k && last_id < 256) { ids[i] = static_cast<unsigned char>(last_id); ++cnt[last_id]; continue; }
        std::size_t h = low_card_hash_key(k) & Mask;
        unsigned id;
        for (;;) {
            if (slot_id[h] == 0) {
                if (distinct >= kCountingClassLimit) return static_cast<std::size_t>(-1);
                id = static_cast<unsigned>(distinct);
                slot_id[h] = static_cast<unsigned short>(id + 1u);
                keys[h] = k;
                id_key[distinct++] = k;
                break;
            }
            if (keys[h] == k) { id = slot_id[h] - 1u; break; }
            h = (h + 1) & Mask;
        }
        ids[i] = static_cast<unsigned char>(id);
        ++cnt[id];
        last_k = k; last_id = id;
    }
    return distinct;
}

// Scatter p -> out by class cursor (nx indexed by class byte), with a write
// prefetch per stream: past ~16-32 destination streams the hardware
// prefetchers lose track and every fresh line costs an exposed
// read-for-ownership (few256 kv16 1M: 19.8 -> ~11 ms).  The iterations are
// independent, so this runs at store throughput; an in-place cycle
// permutation is a chain of dependent loads and measured 3x slower.
template <class T>
FYX_FORCE_INLINE void flag_scatter_range(const T* p, std::size_t lo, std::size_t hi,
                                         const unsigned char* ids, std::size_t* nx, T* out) {
    constexpr std::size_t kAheadElems = (256 + sizeof(T) - 1) / sizeof(T);
    for (std::size_t i = lo; i < hi; ++i) {
        const std::size_t d = nx[ids[i]]++;
        prefetch_write<3>(out + d + kAheadElems);
        out[d] = p[i];
    }
}

template <class Field, class T, class Comp>
inline int trivial_field_flag_sort(T* p, std::size_t n, Comp comp, std::size_t offset,
                                   bool descending) {
    using Key = typename RadixTraits<Field>::Key;
    std::array<Key, kCountingClassLimit> id_key{};
    // One lease for both the scatter buffer and the class bytes: a second
    // concurrent lease misses the thread arena and pays fresh page faults on
    // every call (16 MiB of them for 1M 16-byte records).
    const std::size_t out_bytes = (n * sizeof(T) + 63u) & ~std::size_t(63);
    ScratchLease<unsigned char> lease(out_bytes + n);
    if (!lease.valid()) return 0;
    T* out = reinterpret_cast<T*>(lease.get());
    unsigned char* ids = lease.get() + out_bytes;
    std::array<std::size_t, 256> cnt{};
    const std::size_t distinct = flag_classify_range<Field>(p, 0, n, offset, ids, id_key.data(), cnt.data());
    if (distinct == static_cast<std::size_t>(-1)) return 0;
    if (distinct <= 1) return std::is_sorted(p, p + n, comp) ? 1 : 2;
    std::array<unsigned char, kCountingClassLimit> order{};
    for (std::size_t i = 0; i < distinct; ++i) order[i] = static_cast<unsigned char>(i);
    std::sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(distinct),
              [&](unsigned char a, unsigned char b) {
                  return descending ? id_key[b] < id_key[a] : id_key[a] < id_key[b];
              });
    std::array<std::size_t, 256> nx{};
    std::size_t sum = 0;
    for (std::size_t r = 0; r < distinct; ++r) {
        const unsigned id = order[r];
        nx[id] = sum;
        sum += cnt[id];
    }
    flag_scatter_range(p, 0, n, ids, nx.data(), out);
    // Copy back while proving the order.
    bool ok = true;
    p[0] = out[0];
    for (std::size_t i = 1; i < n; ++i) {
        p[i] = out[i];
        ok &= !comp(out[i], out[i - 1]);
    }
    return ok ? 1 : 2;
}

#if FYX_ENABLE_PARALLEL
// Parallel flag sort: chunks classify independently (local class tables),
// the tables are merged into one key-ordered class list, and each chunk
// scatters through its local->global cursor map.  Same return contract.
template <class Field, class T, class Comp>
inline int trivial_field_flag_sort_parallel(T* p, std::size_t n, Comp comp, std::size_t offset,
                                            bool descending) {
    using Key = typename RadixTraits<Field>::Key;
    const std::size_t chunks = adaptive_parallel_chunks(n);
    if (chunks < 2) return trivial_field_flag_sort<Field>(p, n, comp, offset, descending);
    const std::size_t out_bytes = (n * sizeof(T) + 63u) & ~std::size_t(63);
    ScratchLease<unsigned char> lease(out_bytes + n);
    if (!lease.valid()) return 0;
    T* out = reinterpret_cast<T*>(lease.get());
    unsigned char* ids = lease.get() + out_bytes;
    std::vector<Key> lkey(chunks * 256);
    std::vector<std::size_t> lcnt(chunks * 256, 0), ldist(chunks, 0);
    auto classify_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c)
            ldist[c] = flag_classify_range<Field>(p, (c * n) / chunks, ((c + 1) * n) / chunks, offset,
                                                  ids, lkey.data() + c * 256, lcnt.data() + c * 256);
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), classify_job);
    std::vector<Key> gkey;
    gkey.reserve(256);
    for (std::size_t c = 0; c < chunks; ++c) {
        if (ldist[c] == static_cast<std::size_t>(-1)) return 0;
        for (std::size_t j = 0; j < ldist[c]; ++j) gkey.push_back(lkey[c * 256 + j]);
    }
    if (descending) std::sort(gkey.begin(), gkey.end(), [](Key a, Key b) { return b < a; });
    else std::sort(gkey.begin(), gkey.end());
    gkey.erase(std::unique(gkey.begin(), gkey.end()), gkey.end());
    if (gkey.size() > kCountingClassLimit) return 0;
    const std::size_t G = gkey.size();
    // Global rank of every local class, then per-chunk cursors in rank order.
    std::vector<unsigned short> rank(chunks * 256, 0);
    std::vector<std::size_t> gcnt(chunks * G, 0);
    for (std::size_t c = 0; c < chunks; ++c)
        for (std::size_t j = 0; j < ldist[c]; ++j) {
            const Key k = lkey[c * 256 + j];
            const auto it = descending
                ? std::lower_bound(gkey.begin(), gkey.end(), k, [](Key a, Key b) { return b < a; })
                : std::lower_bound(gkey.begin(), gkey.end(), k);
            const std::size_t g = static_cast<std::size_t>(it - gkey.begin());
            rank[c * 256 + j] = static_cast<unsigned short>(g);
            gcnt[c * G + g] = lcnt[c * 256 + j];
        }
    std::vector<std::size_t> gstart(chunks * G, 0);
    std::size_t sum = 0;
    for (std::size_t g = 0; g < G; ++g)
        for (std::size_t c = 0; c < chunks; ++c) { gstart[c * G + g] = sum; sum += gcnt[c * G + g]; }
    auto scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            std::array<std::size_t, 256> nx{};
            for (std::size_t j = 0; j < ldist[c]; ++j) nx[j] = gstart[c * G + rank[c * 256 + j]];
            flag_scatter_range(p, (c * n) / chunks, ((c + 1) * n) / chunks, ids, nx.data(), out);
        }
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), scatter_job);
    std::atomic<bool> bad{false};
    auto back_job = [&](std::size_t c_lo, std::size_t c_hi) {
        for (std::size_t c = c_lo; c < c_hi; ++c) {
            const std::size_t lo = (c * n) / chunks, hi = ((c + 1) * n) / chunks;
            bool ok = true;
            for (std::size_t i = lo; i < hi; ++i) {
                p[i] = out[i];
                if (i > 0) ok &= !comp(out[i], out[i - 1]);
            }
            if (!ok) bad.store(true, std::memory_order_relaxed);
        }
    };
    parallel_for_index(std::size_t(0), chunks, std::size_t(1), back_job);
    return bad.load() ? 2 : 1;
}
#endif

// 16-byte records keyed by a 64-bit field: in-place AVX-512 record quicksort
// (no scratch, no copy back; equal-key ranges end without leaves).  Returns
// 1 sorted, 2 permuted but not sorted under comp (caller re-sorts), 0 not
// applicable (untouched).
template <class T, class Comp>
struct Kv16Proof {
    // Seams per task slot (task tree ids < kSlots): no locking, each slot is
    // written by one task at a time.
    static constexpr unsigned kSlots = 64;
    T* p;
    Comp* comp;
    bool descending;
    std::vector<std::size_t> seams[kSlots];

    bool verify(std::size_t lo, std::size_t cnt, bool with_prev) const {
        // OR-reduction with no early exit: the compiler vectorizes it for
        // plain key comparators (strided key loads).
        if (cnt == 0) return true;
        const T* q = p + (with_prev ? lo - 1 : lo);
        const std::size_t m = cnt - (with_prev ? 0 : 1);
        const Comp& c = *comp;
        unsigned char bad = 0;
        if (descending) for (std::size_t i = 0; i < m; ++i) bad |= static_cast<unsigned char>(c(q[i], q[i + 1]));
        else            for (std::size_t i = 0; i < m; ++i) bad |= static_cast<unsigned char>(c(q[i + 1], q[i]));
        return bad == 0;
    }
    bool equiv(std::size_t ref, std::size_t lo, std::size_t cnt) const {
        const T r = p[ref];
        const T* q = p + lo;
        const Comp& c = *comp;
        unsigned char bad = 0;
        for (std::size_t i = 0; i < cnt; ++i)
            bad |= static_cast<unsigned char>(c(q[i], r) | c(r, q[i]));
        return bad == 0;
    }
    bool seams_ok() const {
        for (unsigned s = 0; s < kSlots; ++s)
            for (const std::size_t i : seams[s])
                if (descending ? (*comp)(p[i - 1], p[i]) : (*comp)(p[i], p[i - 1])) return false;
        return true;
    }
};

template <class T, class Comp>
struct Kv16Fin {
    Kv16Proof<T, Comp>* pr;
    unsigned slot;
    bool verify(std::size_t lo, std::size_t cnt, bool with_prev) { return pr->verify(lo, cnt, with_prev); }
    bool equiv(std::size_t ref, std::size_t lo, std::size_t cnt) { return pr->equiv(ref, lo, cnt); }
    void seam(std::size_t lo) { if (lo != 0) pr->seams[slot].push_back(lo); }
    bool operator()(std::size_t lo, std::size_t cnt) {
        seam(lo);
        return pr->verify(lo, cnt, false);
    }
};

#if FYX_ENABLE_PARALLEL
template <class T, class Comp>
inline bool kv16_vqsort_parallel_rec(T* p, std::size_t lo, std::size_t nr, int budget, unsigned depth,
                                     bool key_hi, bool is_signed, Kv16Proof<T, Comp>& pr, unsigned slot) {
    constexpr std::size_t kMinParTask = std::size_t(1) << 15;
    Kv16Fin<T, Comp> fin{&pr, slot};
    while (true) {
        if (depth == 0 || budget <= 0 || nr < kMinParTask)
            return kv16_vqsort_range(static_cast<void*>(p), lo, nr, budget, key_hi, is_signed, fin);
        std::size_t split = 0;
        const int r = kv16_vqsort_step(static_cast<void*>(p), lo, nr, split, key_hi, is_signed, fin);
        --budget;
        if (r == 0) return true;
        if (r < 0) return false;
        if (r == 1) continue;
        bool okl = false, okr = false;
        const std::size_t l0 = lo, ln = split, r0 = lo + split, rn = nr - split;
        const int nb = budget;
        const unsigned nd = depth - 1;
        fork_join([&] { okl = kv16_vqsort_parallel_rec(p, l0, ln, nb, nd, key_hi, is_signed, pr, 2 * slot); },
                  [&] { okr = kv16_vqsort_parallel_rec(p, r0, rn, nb, nd, key_hi, is_signed, pr, 2 * slot + 1); });
        return okl && okr;
    }
}
#endif

// 16-byte records keyed by a 64-bit field: in-place AVX-512 record quicksort
// (no scratch, no copy back; equal-key ranges end without leaves).  Every
// final range is proven under comp while hot and the seams between ranges
// afterwards (ascending, so a descending comp is checked mirrored before the
// reverse).  par: fork the top levels over the pool.  Returns 1 sorted, 2
// permuted but not sorted under comp (caller re-sorts), 0 not applicable
// (untouched).
template <class Field, class T, class Comp>
FYX_NOINLINE int trivial_field_kv16_vqsort(T* p, std::size_t n, Comp comp, std::size_t offset,
                                           bool descending, bool par = false) {
    if constexpr (sizeof(T) != 16 || sizeof(Field) != 8 || !std::is_trivially_copyable<T>::value) {
        (void)p; (void)n; (void)comp; (void)offset; (void)descending; (void)par;
        return 0;
    } else {
        if ((offset != 0 && offset != 8) || !use_avx512()) return 0;
        const bool key_hi = offset == 8;
        constexpr bool is_signed = std::is_signed<Field>::value;
        std::unique_ptr<Kv16Proof<T, Comp>> pr(new Kv16Proof<T, Comp>{p, &comp, descending, {}});
        bool ok;
#if FYX_ENABLE_PARALLEL
        if (par && parallel_available()) {
            unsigned depth = 1;
            for (unsigned t = 2; t < global_pool().nworkers() * 2u && depth < 5; t *= 2) ++depth;
            ok = kv16_vqsort_parallel_rec(p, 0, n, vqsort_budget(n), depth, key_hi, is_signed, *pr, 1u);
        } else
#else
        (void)par;
#endif
        {
            Kv16Fin<T, Comp> fin{pr.get(), 0};
            ok = kv16_vqsort(static_cast<void*>(p), n, key_hi, is_signed, fin);
        }
        if (!ok || !pr->seams_ok()) return 2;
        if (descending) std::reverse(p, p + n);
        return 1;
    }
}

// Early route for unordered 16-byte records keyed by a 64-bit field (AVX-512
// hosts): the in-place record quicksort beats the profile -> counting / MSD
// chain on every unordered shape measured (random, few-unique, sqrt, zipf),
// so it runs before the probe stack -- after a 64-sample order gate (both
// directions, frequent turns) that leaves presorted, few-run and pattern
// shapes (organ pipe, rotations, sorted prefixes) to the adaptive paths.
// Many short sorted runs pass the gate on purpose: at sqrt(n) runs the
// quicksort beats run merging (1M: 18.1 vs 20.5 ms; 10k: 0.10 vs 0.49 ms).
// true: sorted.  false: p holds a permutation of its input (maybe unchanged).
// Equivalence of every element to p[0] (a strict weak order makes that
// transitive): one load stream, OR-reduced per block.  AVX-512 target so a
// portable build still vectorizes the inlined comparator 512 bits wide (only
// called after use_avx512()).
template <class T, class Comp>
FYX_TARGET_AVX512 bool kv16_all_equiv_avx512(const T* p, std::size_t n, Comp& comp) {
    // Blocks run back to front: a producer that just wrote the range left
    // its tail in L1/L2, and a forward scan of a range larger than L2 would
    // evict exactly the lines it reads next (LRU), so tail-first takes the
    // cache hits (2-4% at 100k / 1M right after a copy).
    constexpr std::size_t B = 2048;
    const T ref = p[0];
    std::size_t e = n;
    while (e > 1) {
        const std::size_t b = e > B + 1 ? e - B : 1;
        const T* q = p + b;
        const std::size_t m = e - b;
        unsigned char bad = 0;
        for (std::size_t i = 0; i < m; ++i)
            bad |= static_cast<unsigned char>(comp(q[i], ref) | comp(ref, q[i]));
        if (bad) return false;
        e = b;
    }
    return true;
}

template <class Field, class T, class Comp>
inline int kv16_early_field(T* p, std::size_t n, Comp comp, std::size_t off, bool par) {
    bool desc = false;
    if (!trivial_field_candidate_order<Field>(p, n, comp, off, desc, 48)) return 0;
    return trivial_field_kv16_vqsort<Field>(p, n, comp, off, desc, par);
}

template <class T, class Comp>
inline bool try_kv16_early_sort(T* p, std::size_t n, Comp comp, bool par = false) {
    if constexpr (sizeof(T) != 16 || !std::is_trivially_copyable<T>::value || std::is_arithmetic<T>::value) {
        (void)p; (void)n; (void)comp; (void)par;
        return false;
    } else {
        if (n < kKv16EarlyMinN || !use_avx512()) return false;
        constexpr std::size_t S = 64;
        std::size_t asc = 0, dsc = 0, turns = 0;
        int last = 0;
        const T* prev = p;
        for (std::size_t j = 1; j < S; ++j) {
            const T* cur = p + (j * (n - 1)) / (S - 1);
            const bool up = comp(*prev, *cur), down = comp(*cur, *prev);
            asc += up ? 1u : 0u;
            dsc += down ? 1u : 0u;
            const int dir = up ? 1 : (down ? -1 : 0);
            turns += (dir != 0 && last != 0 && dir != last) ? 1u : 0u;
            last = dir != 0 ? dir : last;
            prev = cur;
        }
        if (asc == 0 && dsc == 0) {
            // Every sample equivalent: prove the range all-equivalent (hence
            // sorted) in one vectorizable pass, OR-reduced per block -- the
            // profile's tracker scan costs ~3x more.
            return kv16_all_equiv_avx512(p, n, comp);
        }
        // Both directions often, and alternating (organ pipes / bitonic
        // shapes have both but turn once or twice).
        if (asc < S / 5 || dsc < S / 5 || turns < S / 5) return false;
        for (std::size_t off = 0; off <= 8; off += 8) {
            int r = kv16_early_field<std::int64_t>(p, n, comp, off, par);
            if (r == 0) r = kv16_early_field<std::uint64_t>(p, n, comp, off, par);
            if (r != 0) return r == 1;
        }
        return false;
    }
}

template <class Field, class T, class Comp>
inline bool try_trivial_field_count_sort(T* p, std::size_t n, Comp comp,
                                         std::size_t offset, bool par = false) {
    using Key = typename RadixTraits<Field>::Key;
    bool descending = false;
    if (!trivial_field_candidate_order<Field>(p, n, comp, offset, descending)) return false;
    std::size_t sample_distinct = 0;
    if (!trivial_field_low_cardinality_probe<Field>(p, n, offset, &sample_distinct)) return false;

    // Class-byte flag sort for sparse key domains with up to 255 sampled
    // classes (the scatter write-prefetches each stream, so 256 streams no
    // longer fall off the prefetchers' cliff).
    if (sample_distinct <= 255) {
        const int r = trivial_field_kv16_vqsort<Field>(p, n, comp, offset, descending,
                                                       par && n >= kKv16EarlyParallelMinN);
        if (r != 0) return r == 1;
    }
    if constexpr (std::is_trivially_copyable<T>::value) if (sample_distinct <= 255) {
        // Sample span first: a sparse key domain (hashes, random 64-bit
        // palettes) can skip the dense-range min/max pass altogether.
        Key smn = load_trivial_field_key<Field>(p[0], offset), smx = smn;
        for (std::size_t j = 0; j < 256; ++j) {
            const Key k = load_trivial_field_key<Field>(p[(j * n) / 256], offset);
            smn = k < smn ? k : smn;
            smx = smx < k ? k : smx;
        }
        const unsigned long long sspan = static_cast<unsigned long long>(static_cast<Key>(smx - smn));
        if (sspan >= static_cast<unsigned long long>(std::min<std::size_t>(kCountingRangeLimit, std::max<std::size_t>(n, 4096)))) {
#if FYX_ENABLE_PARALLEL
            const int r = par && n >= kParallelThreshold && parallel_available()
                ? trivial_field_flag_sort_parallel<Field>(p, n, comp, offset, descending)
                : trivial_field_flag_sort<Field>(p, n, comp, offset, descending);
#else
            (void)par;
            const int r = trivial_field_flag_sort<Field>(p, n, comp, offset, descending);
#endif
            if (r != 0) return r == 1;
        }
    }

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
                // Counters and scatter buffer share one lease (a second,
                // concurrent lease would miss the thread arena and fault in
                // fresh pages on every call).
                const std::size_t out_bytes = (n * sizeof(T) + 63u) & ~std::size_t(63);
                ScratchLease<unsigned char> both_lease(out_bytes + range * sizeof(std::size_t));
                if (both_lease.valid()) {
                    std::size_t* counts = reinterpret_cast<std::size_t*>(both_lease.get() + out_bytes);
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

                        {
                            T* out = reinterpret_cast<T*>(both_lease.get());
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


// ---------------------------------------------------------------------------
// MSD radix for trivially copyable records keyed by one integer field.
// The LSD record radix above pays a full read+write of every record per active
// digit -- eight passes for a random 64-bit key, which left 1M 16-byte records
// 3-4x behind ips2ra / vqsort's key-value mode.  This version spends one
// streaming pass (min/max), one histogram pass and one scatter into scratch;
// everything after that works on buckets that fit in L1/L2: a range-adaptive
// second digit, insertion-sorted leaves, and per-bucket comparator checks
// while the bucket is still cached (the field/comparator equivalence is only
// sampled, so the final order is always proven with `comp`).
// ---------------------------------------------------------------------------
template <class Key>
FYX_FORCE_INLINE unsigned record_msd_key_bits(Key d) noexcept {
    unsigned b = 0;
    while (d != 0) { ++b; d = static_cast<Key>(d >> 1); }
    return b;
}

template <class Field, class T>
FYX_FORCE_INLINE typename RadixTraits<Field>::Key
record_msd_key(const T& x, std::size_t offset, bool descending) noexcept {
    using Key = typename RadixTraits<Field>::Key;
    const Key k = load_trivial_field_key<Field>(x, offset);
    return descending ? static_cast<Key>(~k) : k;
}

template <class Field, class T>
inline void record_msd_insertion(T* p, std::size_t m, std::size_t offset, bool descending) {
    using Key = typename RadixTraits<Field>::Key;
    for (std::size_t i = 1; i < m; ++i) {
        const Key k = record_msd_key<Field>(p[i], offset, descending);
        if (!(k < record_msd_key<Field>(p[i - 1], offset, descending))) continue;
        T tmp = p[i];
        std::size_t j = i;
        do { p[j] = p[j - 1]; --j; }
        while (j > 0 && k < record_msd_key<Field>(p[j - 1], offset, descending));
        p[j] = tmp;
    }
}

#ifndef FYX_RMSD_TOP_BITS
#  define FYX_RMSD_TOP_BITS 6
#endif
#ifndef FYX_RMSD_LEAF
#  define FYX_RMSD_LEAF 24
#endif
#ifndef FYX_RMSD_SUB_SHIFT
#  define FYX_RMSD_SUB_SHIFT 3
#endif
#ifndef FYX_RMSD_MAX_BITS
#  define FYX_RMSD_MAX_BITS 11
#endif
inline constexpr unsigned kRecordMsdMaxBits  = FYX_RMSD_MAX_BITS;
inline constexpr unsigned kRecordMsdMaxDepth = 10;
#ifndef FYX_RMSD_RANK_LEAF
#  define FYX_RMSD_RANK_LEAF 1
#endif

// Branch-free leaf: each record's final slot is its rank (keys below it plus
// equal keys earlier in the bucket), so a bucket of s records costs s*s
// predictable compares instead of insertion sort's data-dependent branches.
// `scratch` (the bucket's slot in the other buffer) is free at this point.
template <class Field, class T>
inline void record_msd_rank_leaf(T* p, T* scratch, std::size_t s, std::size_t offset,
                                 bool descending) {
    using Key = typename RadixTraits<Field>::Key;
    Key ks[FYX_RMSD_LEAF];
    for (std::size_t i = 0; i < s; ++i) ks[i] = record_msd_key<Field>(p[i], offset, descending);
    std::memcpy(static_cast<void*>(scratch), static_cast<const void*>(p), s * sizeof(T));
    for (std::size_t i = 0; i < s; ++i) {
        const Key k = ks[i];
        std::size_t r = 0;
        for (std::size_t j = 0; j < i; ++j) r += ks[j] <= k;
        for (std::size_t j = i + 1; j < s; ++j) r += ks[j] < k;
        p[r] = scratch[i];
    }
}
inline constexpr std::size_t kRecordMsdFrame = (std::size_t(2) << kRecordMsdMaxBits) + 1;

// Sorts src[0,m) into dst[0,m); src is scratch afterwards.  `work` holds
// kRecordMsdFrame counters per remaining recursion level.
template <class Field, class T>
inline void record_msd_rec(T* src, T* dst, std::size_t m, std::size_t offset,
                           bool descending, unsigned depth, std::size_t* work) {
    using Key = typename RadixTraits<Field>::Key;
    constexpr std::size_t kLeaf = FYX_RMSD_LEAF;
    if (m <= kLeaf || depth >= kRecordMsdMaxDepth) {
        std::memcpy(static_cast<void*>(dst), static_cast<const void*>(src), m * sizeof(T));
        if (m <= kLeaf) record_msd_insertion<Field>(dst, m, offset, descending);
        else {
            std::sort(dst, dst + m, [&](const T& a, const T& b) {
                return record_msd_key<Field>(a, offset, descending) <
                       record_msd_key<Field>(b, offset, descending);
            });
        }
        return;
    }
    Key mn = record_msd_key<Field>(src[0], offset, descending), mx = mn;
    for (std::size_t i = 1; i < m; ++i) {
        const Key k = record_msd_key<Field>(src[i], offset, descending);
        mn = k < mn ? k : mn;
        mx = mx < k ? k : mx;
    }
    if (mn == mx) {
        std::memcpy(static_cast<void*>(dst), static_cast<const void*>(src), m * sizeof(T));
        return;
    }
    const unsigned bits = record_msd_key_bits(static_cast<Key>(mx - mn));
    unsigned lg = 0;
    while ((std::size_t(1) << (lg + 1)) <= m) ++lg;
    unsigned B = lg > FYX_RMSD_SUB_SHIFT ? lg - FYX_RMSD_SUB_SHIFT : 1;
    if (B > kRecordMsdMaxBits) B = kRecordMsdMaxBits;
    if (B > bits) B = bits;
    const unsigned shift = bits - B;
    const std::size_t nb = std::size_t(1) << B;

    std::size_t* cnt = work;
    std::size_t* pos = work + nb + 1;
    std::memset(cnt, 0, (nb + 1) * sizeof(std::size_t));
    for (std::size_t i = 0; i < m; ++i)
        ++cnt[static_cast<std::size_t>((record_msd_key<Field>(src[i], offset, descending) - mn) >> shift) + 1];
    for (std::size_t b = 1; b <= nb; ++b) cnt[b] += cnt[b - 1];
    std::memcpy(pos, cnt, nb * sizeof(std::size_t));
    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t b = static_cast<std::size_t>(
            (record_msd_key<Field>(src[i], offset, descending) - mn) >> shift);
        dst[pos[b]++] = src[i];
    }
    // Large sub-buckets recurse; the small ones are finished by one
    // insertion pass over the whole range (keys never cross a sub-bucket
    // boundary, so it only moves records inside their own sub-bucket).
    for (std::size_t b = 0; b < nb; ++b) {
        const std::size_t lo = cnt[b], s = cnt[b + 1] - lo;
        if (s <= 1) continue;
        if (s <= kLeaf) {
#if FYX_RMSD_RANK_LEAF
            record_msd_rank_leaf<Field>(dst + lo, src + lo, s, offset, descending);
#endif
            continue;
        }
        record_msd_rec<Field>(dst + lo, src + lo, s, offset, descending, depth + 1,
                              work + kRecordMsdFrame);
        std::memcpy(static_cast<void*>(dst + lo), static_cast<const void*>(src + lo), s * sizeof(T));
    }
#if !FYX_RMSD_RANK_LEAF
    record_msd_insertion<Field>(dst, m, offset, descending);
#endif
}

template <class Field, class T, class Comp>
inline bool record_msd_sort(T* p, std::size_t n, Comp comp, std::size_t offset,
                            bool descending, bool par = false) {
    using Key = typename RadixTraits<Field>::Key;
    // The top digit is planned from a sample (out-of-sample keys clamp to the
    // edge buckets, which only makes those buckets larger) and kept narrow:
    // a full-array scatter into hundreds of streams is several times slower
    // than into a few dozen once the destination spans more pages than the
    // TLB covers, while the second level runs on L2-resident buckets.
    constexpr std::size_t S = 256;
    Key mn = record_msd_key<Field>(p[0], offset, descending), mx = mn;
    for (std::size_t j = 0; j < S; ++j) {
        const Key k = record_msd_key<Field>(p[(j * n) / S], offset, descending);
        mn = k < mn ? k : mn;
        mx = mx < k ? k : mx;
    }
    if (mn == mx) {
        mn = record_msd_key<Field>(p[0], offset, descending); mx = mn;
        for (std::size_t i = 1; i < n; ++i) {
            const Key k = record_msd_key<Field>(p[i], offset, descending);
            mn = k < mn ? k : mn;
            mx = mx < k ? k : mx;
        }
        if (mn == mx) return true;   // every key equal under a sampled-equivalent comparator
    }
    const unsigned bits = record_msd_key_bits(static_cast<Key>(mx - mn));
    unsigned lg = 0;
    while ((std::size_t(1) << (lg + 1)) <= n) ++lg;
    unsigned B = FYX_RMSD_TOP_BITS;
    if (lg < 14) B = lg > 8 ? lg - 8 : 1;
    if (B > bits) B = bits;
    const unsigned shift = bits - B;
    const std::size_t nb = std::size_t(1) << B;
    auto bucket_of = [&](Key k) noexcept -> std::size_t {
        if (k < mn) return 0;
        const std::size_t b = static_cast<std::size_t>((k - mn) >> shift);
        return b < nb ? b : nb - 1;
    };

    ScratchLease<T> tmp_lease(n);
    if (!tmp_lease.valid()) return false;
    T* tmp = tmp_lease.get();
#if FYX_ENABLE_PARALLEL
    // Parallel form: per-chunk histograms and scatter for the top digit, then
    // the buckets (independent, L2-sized after the first sub-digit) are
    // finished and proven by whichever worker picks them up.
    if (par && n >= kParallelThreshold && parallel_available()) {
        const std::size_t chunks = adaptive_parallel_chunks(n);
        std::vector<std::size_t> local(chunks * nb, 0), start(nb + 1, 0);
        auto rmsd_count_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                std::size_t* lc = local.data() + c * nb;
                const std::size_t lo = (c * n) / chunks, hi = ((c + 1) * n) / chunks;
                for (std::size_t i = lo; i < hi; ++i)
                    ++lc[bucket_of(record_msd_key<Field>(p[i], offset, descending))];
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), rmsd_count_job);
        std::size_t sum = 0;
        for (std::size_t b = 0; b < nb; ++b) {
            start[b] = sum;
            for (std::size_t c = 0; c < chunks; ++c) {
                const std::size_t v = local[c * nb + b];
                local[c * nb + b] = sum;
                sum += v;
            }
        }
        start[nb] = sum;
        auto rmsd_scatter_job = [&](std::size_t c_lo, std::size_t c_hi) {
            for (std::size_t c = c_lo; c < c_hi; ++c) {
                std::size_t* pos = local.data() + c * nb;
                const std::size_t lo = (c * n) / chunks, hi = ((c + 1) * n) / chunks;
                for (std::size_t i = lo; i < hi; ++i)
                    tmp[pos[bucket_of(record_msd_key<Field>(p[i], offset, descending))]++] = p[i];
            }
        };
        parallel_for_index(std::size_t(0), chunks, std::size_t(1), rmsd_scatter_job);
        std::atomic<bool> bad{false};
        auto rmsd_bucket_job = [&](std::size_t b_lo, std::size_t b_hi) {
            thread_local std::vector<std::size_t> tl_work;
            if (tl_work.size() < kRecordMsdFrame * (kRecordMsdMaxDepth + 1))
                tl_work.resize(kRecordMsdFrame * (kRecordMsdMaxDepth + 1));
            for (std::size_t b = b_lo; b < b_hi; ++b) {
                const std::size_t lo = start[b], s = start[b + 1] - lo;
                if (s == 0) continue;
                if (bad.load(std::memory_order_relaxed)) {
                    // Declining still has to hand back a permutation of the input.
                    std::memcpy(static_cast<void*>(p + lo), static_cast<const void*>(tmp + lo), s * sizeof(T));
                    continue;
                }
                record_msd_rec<Field>(tmp + lo, p + lo, s, offset, descending, 1, tl_work.data());
                for (std::size_t i = lo + 1; i < lo + s; ++i)
                    if (comp(p[i], p[i - 1])) { bad.store(true, std::memory_order_relaxed); break; }
            }
        };
        parallel_for_index(std::size_t(0), nb, std::size_t(1), rmsd_bucket_job);
        if (bad.load()) return false;
        const T* prev = nullptr;
        for (std::size_t b = 0; b < nb; ++b) {
            const std::size_t lo = start[b], s = start[b + 1] - lo;
            if (s == 0) continue;
            if (prev && comp(p[lo], *prev)) return false;
            prev = p + lo + s - 1;
        }
        return true;
    }
#else
    (void)par;
#endif
    std::vector<std::size_t> work(kRecordMsdFrame * (kRecordMsdMaxDepth + 1) + 2 * nb + 2, 0);
    std::size_t* cnt = work.data() + kRecordMsdFrame * (kRecordMsdMaxDepth + 1);
    std::size_t* pos = cnt + nb + 1;
    for (std::size_t i = 0; i < n; ++i)
        ++cnt[bucket_of(record_msd_key<Field>(p[i], offset, descending)) + 1];
    for (std::size_t b = 1; b <= nb; ++b) cnt[b] += cnt[b - 1];
    std::memcpy(pos, cnt, nb * sizeof(std::size_t));
    for (std::size_t i = 0; i < n; ++i)
        tmp[pos[bucket_of(record_msd_key<Field>(p[i], offset, descending))]++] = p[i];
    const T* prev = nullptr;
    for (std::size_t b = 0; b < nb; ++b) {
        const std::size_t lo = cnt[b], s = cnt[b + 1] - lo;
        if (s == 0) continue;
        record_msd_rec<Field>(tmp + lo, p + lo, s, offset, descending, 1, work.data());
        bool ok = !(prev && comp(p[lo], *prev));
        for (std::size_t i = lo + 1; ok && i < lo + s; ++i) ok = !comp(p[i], p[i - 1]);
        if (!ok) {
            // The comparator is not the field order.  The later buckets still
            // live in scratch: put them back so the caller gets a permutation
            // of its input (it falls back to a comparison sort).
            const std::size_t rest = cnt[b + 1];
            std::memcpy(static_cast<void*>(p + rest), static_cast<const void*>(tmp + rest), (n - rest) * sizeof(T));
            return false;
        }
        prev = p + lo + s - 1;
    }
    return true;
}

/// 512 strided key samples hold >= 32 repeats (distinct-key data: ~0).
template <class Field, class T>
inline bool trivial_field_dup_heavy(const T* p, std::size_t n, std::size_t offset) {
    using Key = typename RadixTraits<Field>::Key;
    if (!use_avx512() || n < 4096) return false;
    constexpr std::size_t S = 512;
    std::array<Key, S> k;
    for (std::size_t j = 0; j < S; ++j) k[j] = load_trivial_field_key<Field>(p[(j * n) / S], offset);
    std::sort(k.begin(), k.end());
    std::size_t rep = 0;
    for (std::size_t j = 1; j < S; ++j) rep += k[j] == k[j - 1] ? 1u : 0u;
    return rep >= 32;
}

template <class Field, class T, class Comp>
inline bool try_trivial_field_radix_sort(T* p, std::size_t n, Comp comp,
                                         std::size_t offset, bool par = false) {
    if (n < kSampleThreshold) return false;
    if constexpr (!std::is_trivially_copyable<T>::value) {
        (void)p; (void)n; (void)comp; (void)offset;
        return false;
    } else {
        using Key = typename RadixTraits<Field>::Key;
        constexpr unsigned Passes = (sizeof(Key) * CHAR_BIT + kRadixBits - 1) / kRadixBits;

        bool descending = false;
        if (!trivial_field_candidate_order<Field>(p, n, comp, offset, descending)) return false;
        // Duplicate-heavy 16-byte records (sqrt-n / zipf-like key counts,
        // past the counting paths' class limit): the in-place record
        // quicksort finishes equal-key ranges without leaves and beats the
        // MSD passes; distinct-key data stays on the radix path.
        if (sizeof(T) == 16 && sizeof(Field) == 8 && trivial_field_dup_heavy<Field>(p, n, offset)) {
            const int r = trivial_field_kv16_vqsort<Field>(p, n, comp, offset, descending,
                                                           par && n >= kKv16EarlyParallelMinN);
            if (r == 1) return true;
        }
        if (FYX_RECORD_MSD) return record_msd_sort<Field>(p, n, comp, offset, descending, par);

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
inline bool try_trivial_prefix_key_radix_sort(T* p, std::size_t n, Comp comp, bool par = false) {
    if constexpr (!std::is_trivially_copyable<T>::value || std::is_arithmetic<T>::value ||
                  std::is_same<T, std::string>::value) {
        (void)p; (void)n; (void)comp; (void)par;
        return false;
    } else {
        // 8-byte fields first: the upper half of a 64-bit key is itself a
        // 32-bit field that orders a duplicate-free sample exactly like the
        // key, so probing 4-byte fields first used to pick it, radix-sort on
        // half the key and then fail the final check on the first tie.
        constexpr std::size_t max_probe = sizeof(T) < 32 ? sizeof(T) : 32;
        for (std::size_t off = 0; off + 8 <= max_probe; off += 8) {
            if (try_trivial_field_radix_sort<std::int64_t>(p, n, comp, off, par)) return true;
            if (try_trivial_field_radix_sort<std::uint64_t>(p, n, comp, off, par)) return true;
        }
        for (std::size_t off = 0; off + 4 <= max_probe; off += 4) {
            if (try_trivial_field_radix_sort<std::int32_t>(p, n, comp, off, par)) return true;
            if (try_trivial_field_radix_sort<std::uint32_t>(p, n, comp, off, par)) return true;
        }
        return false;
    }
}

template <class T, class Comp>
inline bool try_trivial_prefix_key_count_sort(T* p, std::size_t n, Comp comp, bool par = false) {
    if constexpr (!std::is_trivially_copyable<T>::value || std::is_arithmetic<T>::value ||
                  std::is_same<T, std::string>::value) {
        (void)p; (void)n; (void)comp; (void)par;
        return false;
    } else {
        if (n < kCountingMinN) return false;
        constexpr std::size_t max_probe = sizeof(T) < 32 ? sizeof(T) : 32;
        // 8-byte fields first, as in the radix probe (a 64-bit key's upper
        // half samples like a 32-bit key).
        for (std::size_t off = 0; off + 8 <= max_probe; off += 8) {
            if (try_trivial_field_count_sort<std::int64_t>(p, n, comp, off, par)) return true;
            if (try_trivial_field_count_sort<std::uint64_t>(p, n, comp, off, par)) return true;
        }
        for (std::size_t off = 0; off + 4 <= max_probe; off += 4) {
            if (try_trivial_field_count_sort<std::int32_t>(p, n, comp, off, par)) return true;
            if (try_trivial_field_count_sort<std::uint32_t>(p, n, comp, off, par)) return true;
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

// ---------------------------------------------------------------------------
// Prefix-key string sort.  The byte-wise MSD below moves every std::string
// object twice per level (about three levels for random text); here each
// string contributes one 8-byte big-endian chunk of its bytes plus its index,
// the 16-byte (index, key) records go through the record MSD radix, equal
// chunks are refined with the next 8 bytes (strings that end inside a chunk
// go first, ordered by length -- they are prefixes of the longer ones), and
// the strings themselves move exactly twice at the end.  Exact for
// std::string's unsigned-byte lexicographic order.
// ---------------------------------------------------------------------------
#ifndef FYX_STRING_PREFIX_SORT
#  define FYX_STRING_PREFIX_SORT 1
#endif
struct StrKeyIdx {
    std::uint32_t idx;
    std::uint32_t len;     // the string's length (inputs with longer strings decline)
    std::uint64_t key;
};
struct StrKeyIdxLess {
    bool operator()(const StrKeyIdx& a, const StrKeyIdx& b) const noexcept { return a.key < b.key; }
};

inline std::uint64_t string_chunk_be(const std::string& s, std::size_t d) noexcept {
    const std::size_t len = s.size();
    if (d >= len) return 0;
    const unsigned char* q = reinterpret_cast<const unsigned char*>(s.data()) + d;
    const std::size_t m = len - d;
    std::uint64_t v = 0;
    if (m >= 8) {
        for (std::size_t i = 0; i < 8; ++i) v = (v << 8) | q[i];
        return v;
    }
    for (std::size_t i = 0; i < m; ++i) v |= static_cast<std::uint64_t>(q[i]) << (56 - 8 * i);
    return v;
}

inline void string_prefix_sort_keys(StrKeyIdx* a, std::size_t m) {
    if (m <= 48) {
        for (std::size_t i = 1; i < m; ++i) {
            const StrKeyIdx x = a[i];
            std::size_t j = i;
            while (j > 0 && x.key < a[j - 1].key) { a[j] = a[j - 1]; --j; }
            a[j] = x;
        }
        return;
    }
    if (!record_msd_sort<std::uint64_t>(a, m, StrKeyIdxLess{}, 8, false))
        std::sort(a, a + m, StrKeyIdxLess{});
}

// a[0, m) agree on bytes [0, depth) (zero-padded).  Orders them completely.
inline void string_prefix_refine(StrKeyIdx* a, std::size_t m, const std::string* p,
                                 std::size_t depth) {
    if (depth > 512) {
        std::sort(a, a + m, [p](const StrKeyIdx& x, const StrKeyIdx& y) {
            return p[x.idx] < p[y.idx];
        });
        return;
    }
    std::size_t e = 0;
    for (std::size_t i = 0; i < m; ++i)
        if (a[i].len <= depth) std::swap(a[i], a[e++]);
    if (e > 1) {
        std::sort(a, a + e, [](const StrKeyIdx& x, const StrKeyIdx& y) { return x.len < y.len; });
    }
    StrKeyIdx* b = a + e;
    const std::size_t k = m - e;
    if (k <= 1) return;
    // Duplicate-heavy text: a group is usually one string repeated.  One
    // memcmp per member settles that, instead of a cache-missing chunk load
    // per member per 8 bytes until every copy has ended.
    {
        const std::uint32_t l0 = b[0].len;
        const char* f0 = p[b[0].idx].data();
        bool same = true;
        for (std::size_t i = 1; i < k && same; ++i)
            same = b[i].len == l0 &&
                   std::memcmp(p[b[i].idx].data() + depth, f0 + depth, l0 - depth) == 0;
        if (same) return;
    }
    for (std::size_t i = 0; i < k; ++i) b[i].key = string_chunk_be(p[b[i].idx], depth);
    string_prefix_sort_keys(b, k);
    for (std::size_t lo = 0; lo < k;) {
        std::size_t hi = lo + 1;
        while (hi < k && b[hi].key == b[lo].key) ++hi;
        if (hi - lo > 1) string_prefix_refine(b + lo, hi - lo, p, depth + 8);
        lo = hi;
    }
}

inline bool string_prefix_sort(std::string* p, std::size_t n, bool descending) {
    ScratchLease<StrKeyIdx> lease(n);
    if (!lease.valid()) return false;
    StrKeyIdx* a = lease.get();
    for (std::size_t i = 0; i < n; ++i) {
        if (i + 16 < n) prefetch_read<3>(p[i + 16].data());
        const std::size_t len = p[i].size();
        if (len >= 0xffffffffu) return false;      // nothing moved yet
        a[i].idx = static_cast<std::uint32_t>(i);
        a[i].len = static_cast<std::uint32_t>(len);
        a[i].key = string_chunk_be(p[i], 0);
    }
    string_prefix_sort_keys(a, n);
    for (std::size_t lo = 0; lo < n;) {
        std::size_t hi = lo + 1;
        while (hi < n && a[hi].key == a[lo].key) ++hi;
        if (hi - lo > 1) string_prefix_refine(a + lo, hi - lo, p, 8);
        lo = hi;
    }
    std::vector<std::string> tmp;
    tmp.reserve(n);
    // The gather is a dependent-free random read per string: prefetch it a
    // few dozen iterations ahead so the misses overlap.
    constexpr std::size_t kAhead = 24;
    if (!descending) {
        for (std::size_t k = 0; k < n; ++k) {
            if (k + kAhead < n) prefetch_read<3>(p + a[k + kAhead].idx);
            tmp.emplace_back(std::move(p[a[k].idx]));
        }
    } else {
        for (std::size_t k = n; k-- > 0;) {
            if (k >= kAhead) prefetch_read<3>(p + a[k - kAhead].idx);
            tmp.emplace_back(std::move(p[a[k].idx]));
        }
    }
    for (std::size_t k = 0; k < n; ++k) p[k] = std::move(tmp[k]);
    return true;
}

inline bool string_msd_sort_default(std::string* p, std::size_t n, bool descending) {
#if FYX_STRING_PREFIX_SORT
    if (n >= 4096 && n < (std::size_t(1) << 32) && string_prefix_sort(p, n, descending)) return true;
#endif
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
FYX_DIAG_PUSH_SIMD
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
FYX_DIAG_POP_SIMD
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

// Partially sorted input that the patch repairs declined (many short runs,
// e.g. sqrt(n)-length sorted runs) used to fall to comparison pdqsort.  For a
// default-ordered std::string or a record whose comparator samples as an
// integer key field, the key radix sorts are several times cheaper than that.
// The patch repairs decline a "many independent sorted runs" input only after
// several full comparison passes (1M sqrt-length runs: ~20 ms for 16-byte
// records, ~60 ms for strings).  Probe 16 windows instead: if they hold many
// descents and almost every descent is *deep and wide* -- the element 8 past
// the break is still below the one 2 before it, which a lone displaced
// element never produces -- the input is run-structured and the key radix
// should go first.
template <class T, class Comp>
inline bool partial_runs_look_independent(const T* p, std::size_t n, Comp comp) {
    constexpr std::size_t kWindows = 16, kWin = 2048, kW = 8;
    if constexpr (!std::is_same<T, std::string>::value &&
                  !(std::is_trivially_copyable<T>::value && !std::is_arithmetic<T>::value)) {
        (void)p; (void)n; (void)comp;
        return false;
    }
    if (n < kWindows * kWin * 2) return false;
    std::size_t desc = 0, deep = 0;
    for (std::size_t w = 0; w < kWindows; ++w) {
        const std::size_t lo = 2 + (w * (n - kWin - kW - 4)) / kWindows;
        for (std::size_t i = lo; i < lo + kWin; ++i) {
            if (!comp(p[i], p[i - 1])) continue;
            ++desc;
            deep += comp(p[i + kW], p[i - 2]) ? 1u : 0u;
        }
    }
    return desc >= 24 && deep * 8 >= desc * 7;
}

template <class T, class Comp>
inline bool try_key_radix_for_partial(T* p, std::size_t n, Comp comp, bool par = false) {
    if constexpr (std::is_same<T, std::string>::value) {
        (void)par;
        return try_string_msd_sort(p, n, comp, is_descending_v<Comp, T>);
    } else if constexpr (std::is_trivially_copyable<T>::value && !std::is_arithmetic<T>::value) {
        return try_trivial_prefix_key_radix_sort(p, n, comp, par);
    } else {
        (void)p; (void)n; (void)comp; (void)par;
        return false;
    }
}

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
#ifndef FYX_RUNS_KCHAIN
#define FYX_RUNS_KCHAIN 8
#endif

// Co-rank: how many of the first k merged outputs come from A (ties: A
// first, i.e. a stable merge of A then B).
template <class T, class Less>
inline std::size_t merge_corank(const T* A, std::size_t na, const T* B, std::size_t nb, std::size_t k,
                                Less& before) {
    std::size_t lo = k > nb ? k - nb : 0, hi = std::min(k, na);
    while (lo < hi) {
        const std::size_t i = lo + (hi - lo) / 2;
        if (before(B[k - i - 1], A[i])) hi = i;
        else lo = i + 1;
    }
    return lo;
}

// Stable merge of sorted A and B into out with K independent branchless
// chains (output split by co-rank).  A two-way merge is latency-bound --
// index -> load -> compare -> index, ~10 cycles a record -- so K chains in
// one loop overlap K of those (measured 2x at 100k 16-byte records).  The
// select is an address mask: GCC turns a value ternary on a record back into
// a branch, which mispredicts half the time on interleaved runs.
template <unsigned K, class T, class Less>
inline void merge_runs_kchain(const T* A, std::size_t na, const T* B, std::size_t nb, T* out, Less& before) {
    const std::size_t n = na + nb;
    std::size_t i[K], j[K], ie[K], je[K], o[K];
    std::size_t prev_i = 0, prev_k = 0;
    for (unsigned c = 0; c < K; ++c) {
        const std::size_t k1 = (n * (c + 1)) / K;
        const std::size_t i1 = c + 1 == K ? na : merge_corank(A, na, B, nb, k1, before);
        i[c] = prev_i; j[c] = prev_k - prev_i; o[c] = prev_k;
        ie[c] = i1; je[c] = k1 - i1;
        prev_i = i1; prev_k = k1;
    }
    while (true) {
        std::size_t h = ~std::size_t(0);
        for (unsigned c = 0; c < K; ++c) h = std::min(h, std::min(ie[c] - i[c], je[c] - j[c]));
        if (h == 0) break;
        for (std::size_t s = 0; s < h; ++s) {
            for (unsigned c = 0; c < K; ++c) {
                const T* pa = A + i[c];
                const T* pb = B + j[c];
                const std::size_t tb = before(*pb, *pa) ? 1u : 0u;
                const std::uintptr_t mk = std::uintptr_t(0) - static_cast<std::uintptr_t>(tb);
                const std::uintptr_t ua = reinterpret_cast<std::uintptr_t>(pa);
                const std::uintptr_t ub = reinterpret_cast<std::uintptr_t>(pb);
                out[o[c]++] = *reinterpret_cast<const T*>(ua ^ ((ua ^ ub) & mk));
                j[c] += tb;
                i[c] += 1 - tb;
            }
        }
    }
    for (unsigned c = 0; c < K; ++c) {
        std::size_t x = i[c], y = j[c], w = o[c];
        while (x < ie[c] && y < je[c]) {
            if (before(B[y], A[x])) out[w++] = B[y++];
            else                    out[w++] = A[x++];
        }
        while (x < ie[c]) out[w++] = A[x++];
        while (y < je[c]) out[w++] = B[y++];
    }
}

/// Records (trivially copyable, comparator-ordered) made of at most nine
/// ascending runs -- rotated sorted ranges, concatenated sorted batches.
///  * one break and the second run wholly <= the first's head: rotated back
///    through a buffer the size of the smaller run;
///  * one break otherwise: trimmed (elements already final stay) and merged;
///  * up to eight breaks: merged pairwise, ping-ponging with one buffer.
/// Merges are co-ranked parallel merges when `par`, else K-chain merges.
/// The break scan is a vectorized block reduction.  Gates, all before
/// anything moves: 64 sampled adjacent pairs ordered, at most eight
/// descents in the 64-point coarse sample, and every break a real run
/// boundary -- an isolated dip or spike (far swaps, nearly sorted) belongs
/// to the repair paths and is refused at the first break.
template <class T, class Comp>
inline bool try_record_runs_merge(T* p, std::size_t n, Comp comp, bool par) {
    if constexpr (!std::is_trivially_copyable<T>::value || std::is_arithmetic<T>::value) {
        (void)p; (void)n; (void)comp; (void)par;
        return false;
    } else {
        if (n < 4096) return false;
#if FYX_ENABLE_PARALLEL
        par = par && n >= (std::size_t(1) << 17) && parallel_available();
#else
        par = false;
#endif
        auto before = adaptive_order<T>(comp);
        constexpr unsigned kMaxBreaks = 8;
        unsigned coarse = 0;
        for (std::size_t j = 1; j <= 64; ++j) {
            const std::size_t i = (j * (n - 1)) / 65 + 1;
            if (before(p[i], p[i - 1])) return false;
            const std::size_t a = ((j - 1) * (n - 1)) / 64, b = (j * (n - 1)) / 64;
            if (before(p[b], p[a]) && ++coarse > kMaxBreaks) return false;
        }
        std::size_t bounds[kMaxBreaks + 2];
        unsigned nb = 0;
        bounds[0] = 0;
        constexpr std::size_t kBlk = 2048;
        for (std::size_t b = 1; b < n; b += kBlk) {
            const std::size_t e = std::min(n, b + kBlk);
            if (!any_adjacent_pair(p, b, e, before, true)) continue;
            for (std::size_t i = b; i < e; ++i) {
                if (!before(p[i], p[i - 1])) continue;
                if (nb == kMaxBreaks) return false;
                // Dip (p[i] alone below its left neighbour) or spike (p[i-1]
                // alone above both sides): not a boundary between runs.
                if (i + 1 < n && !before(p[i + 1], p[i - 1])) return false;
                if (i >= 2 && !before(p[i], p[i - 2])) return false;
                bounds[++nb] = i;
            }
        }
        if (nb == 0) return true;
        if (nb == 1) {
            const std::size_t m = bounds[1];
            if (!before(p[0], p[n - 1])) {
                // Wrap: [m, n) <= p[0] <= [0, m) -> rotate.
                const std::size_t na = m, nr = n - m;
                ScratchLease<T> lease(std::min(na, nr));
                if (!lease.valid()) return false;
                T* t = lease.get();
                if (na <= nr) {
                    std::memcpy(static_cast<void*>(t), static_cast<const void*>(p), na * sizeof(T));
                    std::memmove(static_cast<void*>(p), static_cast<const void*>(p + m), nr * sizeof(T));
                    std::memcpy(static_cast<void*>(p + nr), static_cast<const void*>(t), na * sizeof(T));
                } else {
                    std::memcpy(static_cast<void*>(t), static_cast<const void*>(p + m), nr * sizeof(T));
                    std::memmove(static_cast<void*>(p + nr), static_cast<const void*>(p), na * sizeof(T));
                    std::memcpy(static_cast<void*>(p), static_cast<const void*>(t), nr * sizeof(T));
                }
                return true;
            }
            // Trim: [0, a) <= p[m] and [e, n) >= p[m-1] are already final.
            const std::size_t a = static_cast<std::size_t>(std::upper_bound(p, p + m, p[m], before) - p);
            const std::size_t e = static_cast<std::size_t>(std::lower_bound(p + m, p + n, p[m - 1], before) - p);
            ScratchLease<T> lease(e - a);
            if (!lease.valid()) return false;
            T* t = lease.get();
#if FYX_ENABLE_PARALLEL
            if (par && e - a >= (std::size_t(1) << 17)) {
                T* buf = t - a;
                parallel_merge_to_buffer_rec(p, a, m, m, e, buf, a, before);
                auto copy_job = [&](std::size_t lo, std::size_t hi) {
                    std::memcpy(static_cast<void*>(p + lo), static_cast<const void*>(buf + lo),
                                (hi - lo) * sizeof(T));
                };
                parallel_for_index(a, e, std::max<std::size_t>(std::size_t(1) << 16, (e - a) / 8), copy_job);
                return true;
            }
#endif
            merge_runs_kchain<FYX_RUNS_KCHAIN>(p + a, m - a, p + m, e - m, t, before);
            std::memcpy(static_cast<void*>(p + a), static_cast<const void*>(t), (e - a) * sizeof(T));
            return true;
        }
        bounds[nb + 1] = n;
        unsigned r = nb + 1;
        ScratchLease<T> lease(n);
        if (!lease.valid()) return false;
        T* src = p;
        T* dst = lease.get();
        while (r > 1) {
            unsigned w = 0;
            for (unsigned i = 0; i < r; i += 2) {
                const std::size_t lo = bounds[i];
                if (i + 1 < r) {
                    const std::size_t mid = bounds[i + 1], hi = bounds[i + 2];
#if FYX_ENABLE_PARALLEL
                    if (par) parallel_merge_to_buffer_rec(src, lo, mid, mid, hi, dst, lo, before);
                    else
#endif
                    merge_runs_kchain<FYX_RUNS_KCHAIN>(src + lo, mid - lo, src + mid, hi - mid, dst + lo, before);
                } else {
                    std::memcpy(static_cast<void*>(dst + lo), static_cast<const void*>(src + lo),
                                (bounds[i + 1] - lo) * sizeof(T));
                }
                bounds[w++] = lo;
            }
            bounds[w] = n;
            r = w;
            T* t = src; src = dst; dst = t;
        }
        if (src != p) {
#if FYX_ENABLE_PARALLEL
            if (par) {
                auto copy_job = [&](std::size_t lo, std::size_t hi) {
                    std::memcpy(static_cast<void*>(p + lo), static_cast<const void*>(src + lo),
                                (hi - lo) * sizeof(T));
                };
                parallel_for_index(std::size_t(0), n,
                                   std::max<std::size_t>(std::size_t(1) << 16, n / 8), copy_job);
                return true;
            }
#endif
            std::memcpy(static_cast<void*>(p), static_cast<const void*>(src), n * sizeof(T));
        }
        return true;
    }
}

} // namespace detail
template <class T, class Comp>
inline void sort_pointer_core_impl(T* p, std::size_t n, Comp comp, const Options& o);
namespace detail {

template <class T, class Comp>
inline bool try_sorted_affix_sort(T* p, std::size_t n, Comp comp, bool par = false) {
#if !FYX_ENABLE_ADAPTIVE_WEAPONS
    (void)p; (void)n; (void)comp; (void)par;
    return false;
#else
    if (n < 8192) return false;
    if constexpr (!std::is_move_constructible<T>::value ||
                  !std::is_move_assignable<T>::value) {
        (void)p; (void)n; (void)comp;
        return false;
    } else {
        auto before = adaptive_order<T>(comp);
        // Affix scans: vectorized block tests, then locate inside the block.
        constexpr std::size_t kBlk = 1024;
        std::size_t head = 1;
        {
            OrderBreakHint& h = order_break_hint();
            if (h.p == static_cast<const void*>(p) && h.n == n && h.head > 0 && h.head < n &&
                before(p[h.head], p[h.head - 1]))
                head = h.head;
            h.p = nullptr;
        }
        while (head < n) {
            const std::size_t e = std::min(n, head + kBlk);
            if (any_adjacent_pair(p, head, e, before, true)) break;
            head = e;
        }
        while (head < n && !before(p[head], p[head - 1])) ++head;
        if (head == n) return true;                  // ordered already
        std::size_t tail = n - 1;
        while (tail > head) {
            const std::size_t b = tail > head + kBlk ? tail - kBlk : head + 1;
            if (b >= tail || any_adjacent_pair(p, b, tail + 1, before, true)) break;
            tail = b;
        }
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
#if FYX_ENABLE_PARALLEL
        // Parallel callers merge the big ordered head with a co-ranked
        // parallel merge into a full-size buffer (one lease, sized for it).
        const bool par_merge = std::is_trivially_copyable<T>::value && par &&
                               n >= (std::size_t(1) << 17) && parallel_available();
#else
        (void)par;
        const bool par_merge = false;
#endif
        if constexpr (std::is_trivially_copyable<T>::value) {
            lease.reset(new ScratchLease<T>(par_merge ? n : (need1 > need2 ? need1 : need2)));
            if (!lease->valid()) return false;
            buf = lease->get();
        }
        // `descending` must follow comp: radix-key kernels sort ascending and
        // reverse for a ">" comparator (with NaN the order differs from comp).
        if constexpr (std::is_arithmetic<T>::value) {
            sort_st(p + head, tail - head, comp, is_descending_v<Comp, T>);
        } else {
            // Records: the full dispatcher (sort_st has no record kernels;
            // 10k random 16-byte records: 0.10 ms there vs 0.51 ms here).
            Options mo;
            mo.parallel = par ? Tri::On : Tri::Off;
            sort_pointer_core_impl(p + head, tail - head, comp, mo);
        }
        if (tail < n) merge_adjacent_runs(p, head, tail, n, buf, before);
#if FYX_ENABLE_PARALLEL
        if constexpr (std::is_trivially_copyable<T>::value) {
            if (par_merge) {
                if (head > 0 && before(p[head], p[head - 1])) {
                    parallel_merge_to_buffer_rec(p, std::size_t(0), head, head, n, buf,
                                                 std::size_t(0), before);
                    auto copy_job = [&](std::size_t lo, std::size_t hi) {
                        std::memcpy(static_cast<void*>(p + lo), static_cast<const void*>(buf + lo),
                                    (hi - lo) * sizeof(T));
                    };
                    parallel_for_index(std::size_t(0), n,
                                       std::max<std::size_t>(std::size_t(1) << 16, n / 8), copy_job);
                }
                return true;
            }
        }
#endif
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
/// monotone.  The AVX-512 implementation is isolated behind a baseline wrapper.
#if FYX_HAS_AVX512_CODE
FYX_DIAG_PUSH_SIMD
template <class T>
FYX_TARGET_AVX512 inline std::size_t radix_key_find_break_avx512(
        const T* p, std::size_t start, std::size_t hi,
        typename RadixTraits<T>::Key prev_key,
        const bool want_up, const bool descending) {
    using RT = RadixTraits<T>;
    using Key = typename RT::Key;
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
            if constexpr (std::is_unsigned_v<T>) e = k;
            else e = _mm512_xor_si512(k, vsign);
        } else if constexpr (sizeof(Key) == 8) {
            const __m512i t = _mm512_srai_epi64(k, 63);
            e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        } else {
            const __m512i t = _mm512_srai_epi32(k, 31);
            e = _mm512_xor_si512(k, _mm512_or_si512(t, vsign));
        }
        const __m512i shifted = (sizeof(Key) == 8)
            ? _mm512_alignr_epi64(e, prev_vec, 7)
            : _mm512_alignr_epi32(e, prev_vec, 15);
        const unsigned bad = (want_up != descending)
            ? (sizeof(Key) == 8 ? _mm512_cmpgt_epu64_mask(shifted, e)
                                : _mm512_cmpgt_epu32_mask(shifted, e))
            : (sizeof(Key) == 8 ? _mm512_cmplt_epu64_mask(shifted, e)
                                : _mm512_cmplt_epu32_mask(shifted, e));
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
FYX_DIAG_POP_SIMD
#endif

template <class T>
inline std::size_t radix_key_find_break(const T* p, std::size_t start, std::size_t hi,
                                        typename RadixTraits<T>::Key prev_key,
                                        const bool want_up, const bool descending) {
#if FYX_HAS_AVX512_CODE
    if (use_avx512())
        return radix_key_find_break_avx512(p, start, hi, prev_key, want_up, descending);
#endif
    using Key = typename RadixTraits<T>::Key;
    Key prev = prev_key;
    for (std::size_t i = start; i < hi; ++i) {
        const Key cur = RadixTraits<T>::encode(p[i]);
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
        if (!vq_range_clean_memo(p, n)) return false;
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
            if (partial_runs_look_independent(p, n, comp) && try_key_radix_for_partial(p, n, comp)) { record_dispatch(DispatchDecision::Radix); return; } if (try_partially_sorted_repair(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; }
            if (try_key_radix_for_partial(p, n, comp)) { record_dispatch(DispatchDecision::Radix); return; } pdqsort_for_profile_pattern(p, n, comp);
            record_dispatch(DispatchDecision::PartialPdq);
            return;
        }
    }

    if (try_bitonic_runs_sort(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; }

    if (radix_order) {
        if (!high_entropy) {
            if (try_integer_range_count_sort(p, n, descending, prof ? &prof->sample_window : nullptr)) { record_dispatch(DispatchDecision::LowCardinality); return; }
            if (!vq_beats_sparse_count(p, n)) {
                if (try_radix_key_sparse_count_sort(p, n, descending, prof ? &prof->sample_window : nullptr)) { record_dispatch(DispatchDecision::LowCardinality); return; }
                if (try_low_cardinality_count_sort(p, p + n, comp)) { record_dispatch(DispatchDecision::LowCardinality); return; }
            }
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
        if (partial_pdq) { if (partial_runs_look_independent(p, n, comp) && try_key_radix_for_partial(p, n, comp)) { record_dispatch(DispatchDecision::Radix); return; } if (try_partially_sorted_repair(p, n, comp)) { record_dispatch(DispatchDecision::PartialPdq); return; } if (try_key_radix_for_partial(p, n, comp)) { record_dispatch(DispatchDecision::Radix); return; } pdqsort_for_profile_pattern(p, n, comp); record_dispatch(DispatchDecision::PartialPdq); return; }
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

// Scratch storage for stable_merge_sort. Unlike vector::emplace_back, direct
// construction into reserved storage has no per-element capacity branch. The
// exact live-object count makes the buffer safe for move-only/non-default types
// and for comparator or move exceptions during a merge.
template <class T>
class StableSortBuffer {
    using Alloc = std::allocator<T>;
    using Traits = std::allocator_traits<Alloc>;
    Alloc alloc_;
    T* data_;
    std::size_t size_;
    std::size_t capacity_;

public:
    explicit StableSortBuffer(std::size_t capacity)
        : data_(Traits::allocate(alloc_, capacity)), size_(0), capacity_(capacity) {}
    StableSortBuffer(const StableSortBuffer&) = delete;
    StableSortBuffer& operator=(const StableSortBuffer&) = delete;
    ~StableSortBuffer() {
        clear();
        Traits::deallocate(alloc_, data_, capacity_);
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    T& operator[](std::size_t i) noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

    void emplace_back(T&& value) {
        // Callers keep each output pass within the buffer's allocated
        // capacity; treating that bound as a loop invariant avoids a capacity
        // branch on every element.
        Traits::construct(alloc_, data_ + size_, std::move(value));
        ++size_;
    }

    void clear() noexcept {
        while (size_ != 0) Traits::destroy(alloc_, data_ + --size_);
    }

    void swap(StableSortBuffer& other) noexcept {
        using std::swap;
        swap(data_, other.data_);
        swap(size_, other.size_);
        swap(capacity_, other.capacity_);
    }
};

// Stable top-down merge sort using one reusable buffer for the left run.
// Balanced splitting guarantees that the smaller run fits in n/2 scratch
// elements. Boundary checks skip merges when adjacent recursive ranges are
// already ordered; this retains stability while avoiding a second full buffer.
template <class It, class T, class Comp>
inline void stable_merge_sort_recursive(It first, std::size_t n,
                                        StableSortBuffer<T>& scratch, Comp& comp) {
    constexpr std::size_t kInsertionCutoff = 32;
    if (n < 2) return;
    if (n <= kInsertionCutoff) {
        for (std::size_t i = 1; i < n; ++i) {
            T value = std::move(first[i]);
            std::size_t hole = i;
            while (hole != 0 && comp(value, first[hole - 1])) {
                first[hole] = std::move(first[hole - 1]);
                --hole;
            }
            first[hole] = std::move(value);
        }
        return;
    }

    const std::size_t left_size = n / 2;
    const std::size_t middle = left_size;
    stable_merge_sort_recursive(first, left_size, scratch, comp);
    stable_merge_sort_recursive(first + middle, n - middle, scratch, comp);
    if (!comp(first[middle], first[middle - 1])) return;

    scratch.clear();
    for (std::size_t i = 0; i < left_size; ++i)
        scratch.emplace_back(std::move(first[i]));
    std::size_t left = 0, right = middle, out = 0;
    while (left < left_size && right < n) {
        if (comp(first[right], scratch[left])) first[out++] = std::move(first[right++]);
        else                                      first[out++] = std::move(scratch[left++]);
    }
    while (left < left_size) first[out++] = std::move(scratch[left++]);
    scratch.clear();
}

// ---------------------------------------------------------------------------
// Adaptive, allocation-assisted, STABLE merge sort. Works for any random-
// access range (pointers, vector/deque iterators, ...). Used by stable_sort
// whenever the radix path cannot be taken.
// ---------------------------------------------------------------------------
template <class It, class Comp>
inline void stable_merge_sort(It first, It last, Comp comp) {
    using T = iter_value_t<It>;
    const std::size_t n = static_cast<std::size_t>(last - first);
    if (n < 2) return;

    // A balanced recursion needs at most floor(n/2) live scratch objects.
    // Raw allocator-backed storage supports move-only/non-default types and
    // avoids reserving a second n-element buffer for alternating merge passes.
    StableSortBuffer<T> scratch(n / 2);
    stable_merge_sort_recursive(first, n, scratch, comp);
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

    // Small numeric ranges on AVX-512 hosts: the vectorised quicksort (with
    // its bitonic-network leaves) is ~5-30x faster below kVqsortMinN than
    // the probe stack + pdq/radix below, which only pays for itself on large
    // inputs.  One branch-free order scan keeps sorted / reverse / all-equal
    // inputs O(n); random input leaves that scan after a single block.
    if constexpr (detail::vqsort_kernel_supported_v<T> && detail::vqsort_preferred<T>()) {
#if FYX_ENABLE_PARALLEL
        const bool small_vq_serial = !detail::dynamic_parallel_allowed<T>(n, o);
#else
        const bool small_vq_serial = true;
#endif
        if (radix_ok && n >= detail::kSmallVqsortMinN && n < detail::kSmallVqsortMaxN &&
            small_vq_serial && detail::use_avx512()) {
            // One fused vector pass: neighbour order + NaN / -0 scan.  When
            // the range is clean hardware order is the library's order; else
            // the radix-key scan decides.
            // Bitwise all-equal first: random input differs in the first
            // bytes, and an all-equal range costs one memcmp.
            // Then a reverse-shaped range (first key above the last in sort
            // order, 16 strided samples non-increasing) is verified and
            // reversed in one two-ended pass instead of an order scan plus a
            // reverse pass (1M uint64: one read + write instead of two reads
            // + write).  A break leaves the swapped blocks in place -- a
            // permutation the paths below sort like any other input.
            if (const int fp = detail::front_order_probe(p, n, descending)) {
                detail::record_dispatch(fp == 1 ? detail::DispatchDecision::ProfileAllEqual
                                                : detail::DispatchDecision::ProfileReverse);
                return;
            }
            std::size_t settled = 0;
            unsigned prior = 0;
            const unsigned pre = detail::vqsort_small_prescan(p, n, &settled, &prior);
            const bool clean = (pre & 4u) == 0;   // as far as scanned
            // Keys before `mono` hold no descent in the sort direction (the
            // prescan stopped in the block where its first one lies, and
            // hardware order is the library's on the clean prefix), so the
            // descent scans below start there instead of re-reading it.
            const std::size_t mono =
                (clean && (pre & 3u) == 3u && prior == (descending ? 2u : 1u)) ? settled : 0;
            detail::FastOrderKind k;
            if (clean) {
                const bool up = (pre & 1u) != 0, dn = (pre & 2u) != 0;
                k = (!up && !dn) ? detail::FastOrderKind::AllEqual
                  : (up && dn)   ? detail::FastOrderKind::None
                  : (up != descending) ? detail::FastOrderKind::Sorted
                                       : detail::FastOrderKind::Reverse;
            } else {
                k = detail::blocked_radix_order_kind(p, n, descending);
            }
            if (k == detail::FastOrderKind::AllEqual) {
                detail::record_dispatch(detail::DispatchDecision::ProfileAllEqual);
                return;
            }
            if (k == detail::FastOrderKind::Sorted) {
                detail::record_dispatch(detail::DispatchDecision::ProfileSorted);
                return;
            }
            if (k == detail::FastOrderKind::Reverse) {
                detail::reverse_range_adaptive(p, n);
                detail::record_dispatch(detail::DispatchDecision::ProfileReverse);
                return;
            }
            // At most 32 distinct keys: count + fill (a 64-key sample with
            // more than 16 distinct keys declines at once).
            if (detail::try_vfew_distinct_sort(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::LowCardinality);
                return;
            }
            // Few-run shapes (rotated, organ pipe, bitonic, permuted blocks)
            // are O(n) for the structural kernels; each rejects random data
            // within a short scan.  Up to one leaf the column network sorts
            // any shape in a few hundred vector ops, so they are skipped.
            // Many sorted runs (four deep, well-spaced drops at the front):
            // only the few-run merge can use that; skipping the descent
            // count and the repairs saves a full pass before the quicksort.
            const bool runs_like = n > detail::vqsort_leaf<T>() &&
                                   detail::descents_look_like_runs(p, n, mono > 1 ? mono : 1, descending, 16, 4);
            if (runs_like && detail::try_few_runs_merge(p, n, descending, mono)) {
                detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                return;
            }
            if (!runs_like && n > detail::vqsort_leaf<T>()) {
                // Route by the number of descents (counted up to n/16).
                std::size_t brk = n;
                // Up to 4096 keys one pass records the descent positions
                // themselves (the sparse-outlier repair reuses them).
                constexpr std::size_t kPosCap = 256;
                std::uint32_t dpos[kPosCap];
                const bool have_pos = n <= 16 * kPosCap;
                std::size_t D;
                if (have_pos) {
                    // Below 2048 keys only the few-descent repairs use an
                    // exact count; a dense head (> 8 drops in 32 pairs)
                    // already routes the range to the dense branch.
                    D = n < 2048 ? detail::descent_positions(p, 1, 33, descending, dpos, 8) : 0;
                    D = D > 8 ? n / 16 + 1 : detail::descent_positions(p, 1, n, descending, dpos, n / 16);
                    if (D != 0) brk = dpos[0];
                } else if (detail::descents_look_dense(p, n, descending)) {
                    D = n / 16 + 1;        // random-like: skip the n/8-pair count
                } else {
                    // (block starts of the first descents: positions for the
                    // outlier repair come from those blocks alone)
                    std::size_t nblk = 0;
                    D = detail::count_descents_capped(p, n, descending, n / 16, brk, mono,
                                                      n <= 0xFFFFFFFFull ? dpos : nullptr, 32, &nblk);
                    if (D <= 32 && nblk != 0 && !detail::descent_positions_in_blocks(p, n, descending, dpos, nblk, D))
                        D = 33;                    // (cannot happen) no repair then
                }
                const bool pos_ok = have_pos || (D <= 32 && n <= 0xFFFFFFFFull);   // dpos holds all D
                if (D == 1) {
                    // Two runs (concatenated sorted halves, a rotation).
                    brk = detail::radix_key_find_break(p, brk, n, detail::RadixTraits<T>::encode(p[brk - 1]),
                                                       true, descending);
                    detail::merge_two_runs(p, n, brk, descending);
                    detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                    return;
                }
                // A few monotone runs either way (organ pipe, block swaps).
                if (detail::try_few_runs_merge(p, n, descending, mono)) {
                    detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                    return;
                }
                if (D > n / 16) {
                    // Dense descents: organ pipe / bitonic / zigzag families.
                    if (detail::try_zigzag_organ_pipe_sort(p, n, comp) ||
                        (detail::likely_mid_bitonic_runs(p, n, comp) &&
                         detail::try_bitonic_runs_sort(p, n, comp)) ||
                        detail::try_numeric_half_organ_fill(p, n, comp) ||
                        detail::try_bitonic_runs_sort(p, n, comp)) {
                        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                        return;
                    }
                } else if (D > 1) {
                    // Sparse descents: mostly sorted.  Local displacements go
                    // to bounded insertion, permuted blocks to the structural
                    // proof, a minority of far / tail keys to extract-merge.
                    // Up to 32 isolated outliers (far swaps, a few misplaced
                    // keys): the outlier repair is about one scan and leaves
                    // the range untouched when it declines -- before the
                    // insertion repair, which a far key exhausts only after
                    // n / 4 moves (1M double far swaps: 2.0 -> ~0.3 ms).
                    if (D <= 32 && pos_ok && detail::sparse_outlier_repair_at(p, n, dpos, D, descending)) {
                        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                        return;
                    }
                    const std::size_t head = std::min<std::size_t>(n - 1, 256);
                    if (D <= n / 128 &&
                        detail::head_inversions_within(p, n, descending, head, head / 32) &&
                        detail::descents_look_local(p, n, brk, descending, 96, 4)) {
                        if (detail::budgeted_insertion_repair(p, n, descending, n / 4 + 64)) {
                            detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                            return;
                        }
                    }
                    if (detail::try_proof_structured_sort(p, n, descending)) {
                        detail::record_dispatch(detail::DispatchDecision::ProfileSorted);
                        return;
                    }
                    // Sorted prefix of at least half the range, disorder only
                    // behind it (appended keys): sort the tail, merge once.
                    {
                        const std::size_t cut = detail::radix_key_find_break(
                            p, brk, n, detail::RadixTraits<T>::encode(p[brk - 1]), true, descending);
                        // (Below 2048 int32 keys the vq sorts the whole range
                        // faster than tail sort + merge network; measured.)
                        if (cut >= n / 2 && cut < n && (n >= 2048 || sizeof(T) == 8)) {
                            sort_pointer_core_impl(p + cut, n - cut, comp, o);
                            detail::merge_two_runs(p, n, cut, descending);
                            detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                            return;
                        }
                    }
                    // Below 2048 keys extraction costs about what the sort
                    // does (measured: a 10% random tail of 1000 keys runs
                    // 2x slower through it); the outlier repair above already
                    // took the few-descent cases.
                    // Many sorted runs (each boundary a deep drop) defeat the
                    // extractor only after it has scanned most of the range.
                    if (n >= 2048 && !detail::descents_look_like_runs(p, n, brk, descending, 16, 4) &&
                        detail::try_extract_merge_repair(p, n, descending, [&](T* q, std::size_t m) {
                            sort_pointer_core_impl(q, m, comp, o);
                        })) {
                        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                        return;
                    }
                }
            }
            // Narrow integer domains: dense counting is O(n + range) and beats
            // any comparison sort there; the probe samples and declines fast.
            // A 64-key strided sample bounds the range from below; only a
            // range small beside n gets the (heavier) counting probe.
            if constexpr (std::is_integral<T>::value) {
                bool narrow = false;
                if (n >= detail::kCountingMinN) {
                    using UK = typename std::make_unsigned<T>::type;
                    T lo = p[0], hi = p[0];
                    for (std::size_t j = 1; j < 64; ++j) {
                        const T x = p[(j * n) >> 6];
                        lo = x < lo ? x : lo;
                        hi = hi < x ? x : hi;
                    }
                    const UK span = static_cast<UK>(static_cast<UK>(hi) - static_cast<UK>(lo));
                    narrow = static_cast<unsigned long long>(span) <=
                             static_cast<unsigned long long>(std::max<std::size_t>(4096, n / 8));
                }
                if (narrow && detail::try_integer_range_count_sort<T>(p, n, descending, nullptr)) {
                    detail::record_dispatch(detail::DispatchDecision::LowCardinality);
                    return;
                }
            }
            // A few hundred to ~2000 distinct keys: hash counting for 8-byte
            // keys.  Clean 4-byte ranges stay with the vq: with its leaf
            // screen (one- / two-valued leaves skip the network) it beats
            // the table at every measured n and key count (100k x 256 keys:
            // 1.24 against 2.27 ns/key; 1000 keys: 2.63 against 2.95).
            if ((sizeof(T) != 4 || !clean) && detail::try_hash_count_sort(p, n, descending)) {
                detail::record_dispatch(detail::DispatchDecision::LowCardinality);
                return;
            }
            // NaN / unclean ranges take the general path below.
            if (clean && (pre & 8u) == 0) {
                detail::vqsort_serial(p, n);
                if (descending) detail::reverse_range_adaptive(p, n);
                detail::record_dispatch(detail::DispatchDecision::VectorQuick);
                return;
            }
            // Cleanliness not yet established (floats): the first partition
            // screens for NaN / -0 as it goes; on a hit the (permuted) range
            // continues down the general path.
            if (clean && detail::vqsort_serial_checked(p, n)) {
                if (descending) detail::reverse_range_adaptive(p, n);
                detail::record_dispatch(detail::DispatchDecision::VectorQuick);
                return;
            }
        }
    }

    // The NaN / -0 scan is memoised per top-level call from here on (the
    // large-n probes may ask several times about the same range).
    const detail::VqCleanMemoScope vq_clean_scope;

#if FYX_ENABLE_FAST_PATHS
#if FYX_ENABLE_PARALLEL
    const bool kv16_par = n >= detail::kKv16EarlyParallelMinN && detail::dynamic_parallel_allowed<T>(n, o);
#else
    const bool kv16_par = false;
#endif
    if (detail::try_kv16_early_sort(p, n, comp, kv16_par)) {
        detail::record_dispatch(detail::DispatchDecision::VectorQuick);
        return;
    }
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
    if (n > detail::kNetworkMax && detail::try_record_runs_merge(p, n, comp, rev_parallel_ok)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    // Records: the affix weapon straight after the run merge (neither moves
    // anything when declining, so the order exit's break hint still holds),
    // before insertion repair and the natural-run scan pay to rescan the head.
    const bool affix_early = !std::is_arithmetic<T>::value;
    if (affix_early && n > detail::kNetworkMax && detail::try_sorted_affix_sort(p, n, comp, rev_parallel_ok)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    if (n > detail::kNetworkMax && detail::try_bounded_insertion_repair(p, n, comp, true)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
#if FYX_ENABLE_PARALLEL
    const bool fd_par = detail::dynamic_parallel_allowed<T>(n, o);
#else
    const bool fd_par = false;
#endif
    if (n > detail::kNetworkMax && detail::try_natural_run_merge_adaptive(p, n, comp)) {
        detail::record_dispatch(detail::DispatchDecision::PartialPdq);
        return;
    }
    // Ordered except for one stretch in the middle: see
    // try_sorted_affix_sort.  Like the run merge this has to be reachable
    // before the parallel kernels are chosen, or a range that is three
    // quarters sorted pays for all of it on every worker.
    if (!affix_early && n > detail::kNetworkMax && detail::try_sorted_affix_sort(p, n, comp, fd_par)) {
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
            if (detail::partial_runs_look_independent(p, n, comp) && detail::try_key_radix_for_partial(p, n, comp,
#if FYX_ENABLE_PARALLEL
                    detail::dynamic_parallel_allowed<T>(n, o)
#else
                    false
#endif
                    )) { detail::record_dispatch(detail::DispatchDecision::Radix); return; } if (detail::try_partially_sorted_repair(p, n, comp)) {
                detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                return;
            }
            if (detail::try_key_radix_for_partial(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; } detail::pdqsort_for_profile_pattern(p, n, comp);
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
                if (detail::partial_runs_look_independent(p, n, comp) && detail::try_key_radix_for_partial(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; } if (detail::try_partially_sorted_repair(p, n, comp)) {
                    detail::record_dispatch(detail::DispatchDecision::PartialPdq);
                    return;
                }
                if (detail::try_key_radix_for_partial(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; } detail::pdqsort_for_profile_pattern(p, n, comp);
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
                if (detail::try_trivial_prefix_key_count_sort(p, n, comp, true)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
                if (detail::try_low_cardinality_count_sort(p, p + n, comp)) { detail::record_dispatch(detail::DispatchDecision::LowCardinality); return; }
            }
            if (partial_pdq) { if (detail::partial_runs_look_independent(p, n, comp) && detail::try_key_radix_for_partial(p, n, comp, true)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; } if (detail::try_partially_sorted_repair(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::PartialPdq); return; } if (detail::try_key_radix_for_partial(p, n, comp)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; } detail::pdqsort_for_profile_pattern(p, n, comp); detail::record_dispatch(detail::DispatchDecision::PartialPdq); return; }
            if (detail::try_guarded_string_order_sort(p, n, comp, want_parallel)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (detail::try_string_msd_sort_parallel(p, n, comp, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (detail::try_string_msd_sort(p, n, comp, descending)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
            if (detail::try_trivial_prefix_key_radix_sort(p, n, comp, true)) { detail::record_dispatch(detail::DispatchDecision::Radix); return; }
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
    // Integers ordered by the standard "<" / ">" comparators: elements that
    // compare equal are bit-identical, so stability is unobservable and the
    // (faster, adaptive) unstable engine produces the exact same output.
    // Kept serial: stable_sort has no Options and never spawned threads.
    if constexpr (std::is_integral<T>::value) {
        if (ascending || descending) {
            Options o;
            o.parallel = Tri::Off;
            sort_pointer_core(p, n, comp, o);
            return;
        }
    }
    if (detail::radix_supported_v<T> && (ascending || descending)) {
        if (detail::try_radix_monotonic_sort(p, n, descending, false)) return;
    } else {
        if (detail::try_monotonic_sort(p, p + n, comp, false)) return;
        if (detail::try_stable_reverse_sort(p, p + n, comp)) return;
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
        if (detail::try_stable_reverse_sort(first, last, fyx::less{})) return;
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
        if (detail::try_stable_reverse_sort(first, last, comp)) return;
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
#if FYX_OPTIMIZE_PRAGMA_ACTIVE
#  pragma GCC pop_options
#endif
#include <dlfcn.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#if FYX_OPTIMIZE_PRAGMA_ACTIVE
#  pragma GCC push_options
#  pragma GCC optimize("vect-cost-model=dynamic", "unswitch-loops", "peel-loops")
#endif

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
extern "C" __global__ void fyx_hist(const )CUDA") + ktype + R"CUDA( *__restrict__ in,
                                  unsigned int* __restrict__ hist,
                                  unsigned int shift, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int d = (unsigned int)((in[i] >> shift) & 0xFFu);
    atomicAdd(&hist[d], 1u);
}
extern "C" __global__ void fyx_scatter(const )CUDA" + ktype + R"CUDA( *__restrict__ in,
                                    )CUDA" + ktype + R"CUDA( *__restrict__ out,
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
    // UNVERIFIED ON THIS BOX (no GPU).  Wrapped so any failure falls back;
    // with exceptions switched off there is nothing to wrap, and every error
    // path here already returns false explicitly.
#if FYX_HAS_EXCEPTIONS
    try {
#endif
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
            char* buf = nullptr;
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
#if FYX_HAS_EXCEPTIONS
    } catch (...) {
        return false;  // any failure -> CPU fallback
    }
#endif
#else
    (void)p; (void)n;
    return false;      // compute path disabled: CPU fallback (the documented default)
#endif
}

} // namespace detail
} // namespace fyx

#endif // FYX_ENABLE_GPU

// ===========================================================================
//  Section 99 -- epilogue: restore the caller's optimisation options.
// ===========================================================================
#if FYX_OPTIMIZE_PRAGMA_ACTIVE
#  pragma GCC pop_options
#endif

#endif // FYX_SORT_HPP_INCLUDED
