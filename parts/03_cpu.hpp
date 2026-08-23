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
