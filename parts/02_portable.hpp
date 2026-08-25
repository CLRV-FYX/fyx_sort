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
