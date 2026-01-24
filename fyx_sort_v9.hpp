/*
 * FYX-SORT v9.0 - 终极高性能排序库
 * 
 * ████████╗██╗   ██╗██╗  ██╗    ███████╗ ██████╗ ██████╗ ████████╗
 * ██╔═════╝╚██╗ ██╔╝╚██╗██╔╝    ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝
 * █████╗    ╚████╔╝  ╚███╔╝     ███████╗██║   ██║██████╔╝   ██║   
 * ██╔══╝     ╚██╔╝   ██╔██╗     ╚════██║██║   ██║██╔══██╗   ██║   
 * ██║         ██║   ██╔╝ ██╗    ███████║╚██████╔╝██║  ██║   ██║   
 * ╚═╝         ╚═╝   ╚═╝  ╚═╝    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   
 * 
 * 版本: 9.0.0 (终极版)
 * 
 * 核心特性:
 *   - 无锁工作窃取并行调度器 (Chase-Lev算法)
 *   - 分层自适应基数排序 (L1/L2/L3缓存优化)
 *   - 预计算SIMD置换表 (零运行时开销)
 *   - 流式多级预取 (隐藏内存延迟)
 *   - CPU运行时特征检测 (CPUID)
 *   - 数据分布学习 (自适应算法选择)
 *   - American Flag原地基数排序
 *   - 缓存无关算法 (自动适应缓存层次)
 *   - 超标量分区 (分支预测友好)
 *   - NUMA感知内存分配
 * 
 * 许可证: 
 *   ┌─────────────────────────────────────────────────────────────┐
 *   │ 本软件采用【署名-非商业性使用】许可协议                     │
 *   │                                                             │
 *   │ 您可以自由地：                                              │
 *   │   - 复制、分发、展示和表演本作品                           │
 *   │   - 基于本作品创作演绎作品                                 │
 *   │                                                             │
 *   │ 但必须遵守以下条件：                                        │
 *   │   1. 署名 - 您必须保留原作者署名 (FYX/付宇轩)              │
 *   │   2. 非商业性使用 - 您不得将本作品用于商业目的             │
 *   │                                                             │
 *   │ 如需商业使用，请联系原作者获取授权。                        │
 *   └─────────────────────────────────────────────────────────────┘
 * 
 * 原作者: FYX (付宇轩)
 * 版权所有 (C) 2024-2025
 */
 
#ifndef FYX_SORT_V9_HPP
#define FYX_SORT_V9_HPP

// 编译器版本检查
#if defined(_MSC_VER) && _MSC_VER < 1914
    #error "FYX-SORT requires Visual Studio 2017 15.7 or later"
#endif

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 7
    #error "FYX-SORT requires GCC 7 or later"
#endif

#if defined(__clang__) && __clang_major__ < 5
    #error "FYX-SORT requires Clang 5 or later"
#endif

// C++标准检查
#if __cplusplus < 201703L && !defined(_MSVC_LANG)
    #error "FYX-SORT requires C++17 or later"
#elif defined(_MSVC_LANG) && _MSVC_LANG < 201703L
    #error "FYX-SORT requires C++17 or later"
#endif

#define FYX_VERSION "9.1.0"
#define FYX_VERSION_MAJOR 9
#define FYX_VERSION_MINOR 1
#define FYX_VERSION_PATCH 0

// ═══════════════════════════════════════════════════════════════════════════
// 第一部分: 编译器和平台配置
// ═══════════════════════════════════════════════════════════════════════════

#if defined(_MSC_VER)
    #define FYX_MSVC 1
    #define FYX_INLINE __forceinline
    #define FYX_NOINLINE __declspec(noinline)
    #define FYX_RESTRICT __restrict
    #define FYX_LIKELY(x) (x)
    #define FYX_UNLIKELY(x) (x)
    #define FYX_ASSUME(x) __assume(x)
    #define FYX_ALIGN(n) __declspec(align(n))
    #define FYX_HOT
    #define FYX_COLD
    #define FYX_FLATTEN
    #define FYX_UNREACHABLE() __assume(0)
    #define FYX_THREAD_LOCAL __declspec(thread)
    #include <intrin.h>
    #pragma warning(disable: 4324 4127 4701 4244 4146 4267 4702 4706)
#elif defined(__clang__) || defined(__GNUC__)
    #define FYX_GCC_COMPATIBLE 1
    #define FYX_INLINE inline __attribute__((always_inline))
    #define FYX_NOINLINE __attribute__((noinline))
    #define FYX_RESTRICT __restrict__
    #define FYX_LIKELY(x) __builtin_expect(!!(x), 1)
    #define FYX_UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define FYX_ASSUME(x) do { if(!(x)) __builtin_unreachable(); } while(0)
    #define FYX_ALIGN(n) __attribute__((aligned(n)))
    #define FYX_HOT __attribute__((hot))
    #define FYX_COLD __attribute__((cold))
    #define FYX_FLATTEN __attribute__((flatten))
    #define FYX_UNREACHABLE() __builtin_unreachable()
    #define FYX_THREAD_LOCAL __thread
#else
    #define FYX_INLINE inline
    #define FYX_NOINLINE
    #define FYX_RESTRICT
    #define FYX_LIKELY(x) (x)
    #define FYX_UNLIKELY(x) (x)
    #define FYX_ASSUME(x) ((void)0)
    #define FYX_ALIGN(n)
    #define FYX_HOT
    #define FYX_COLD
    #define FYX_FLATTEN
    #define FYX_UNREACHABLE() ((void)0)
    #define FYX_THREAD_LOCAL thread_local
#endif

// 预取宏 - 多级预取支持
#ifdef _MSC_VER
    #define FYX_PREFETCH_T0(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define FYX_PREFETCH_T1(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T1)
    #define FYX_PREFETCH_T2(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T2)
    #define FYX_PREFETCH_NTA(addr) _mm_prefetch((const char*)(addr), _MM_HINT_NTA)
    #define FYX_PREFETCH_W(addr)   _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
    #define FYX_PREFETCH_T0(addr)  __builtin_prefetch((addr), 0, 3)
    #define FYX_PREFETCH_T1(addr)  __builtin_prefetch((addr), 0, 2)
    #define FYX_PREFETCH_T2(addr)  __builtin_prefetch((addr), 0, 1)
    #define FYX_PREFETCH_NTA(addr) __builtin_prefetch((addr), 0, 0)
    #define FYX_PREFETCH_W(addr)   __builtin_prefetch((addr), 1, 3)
#else
    #define FYX_PREFETCH_T0(addr)  ((void)0)
    #define FYX_PREFETCH_T1(addr)  ((void)0)
    #define FYX_PREFETCH_T2(addr)  ((void)0)
    #define FYX_PREFETCH_NTA(addr) ((void)0)
    #define FYX_PREFETCH_W(addr)   ((void)0)
#endif

// 并行开关
#ifndef FYX_ENABLE_PARALLEL
    #define FYX_ENABLE_PARALLEL 1
#endif

// 标准库包含
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <type_traits>
namespace detail {
namespace static_checks {
    // 验证对齐数组正确性
    static_assert(sizeof(mem::AlignedArray<int, 64>) >= sizeof(int) * 64,
                  "AlignedArray size incorrect");
    static_assert(alignof(mem::AlignedArray<int, 64>) >= config::CACHE_LINE,
                  "AlignedArray alignment incorrect");
    
    // 验证类型特征
    static_assert(traits::is_radix_sortable_v<int32_t>, "int32_t should be radix sortable");
    static_assert(traits::is_radix_sortable_v<uint64_t>, "uint64_t should be radix sortable");
    static_assert(traits::is_radix_sortable_v<float>, "float should be radix sortable");
    static_assert(traits::is_radix_sortable_v<double>, "double should be radix sortable");
    
    // 验证键映射正确性
    static_assert(sizeof(typename keymap::Mapper<float>::Key) == sizeof(float),
                  "float key size mismatch");
    static_assert(sizeof(typename keymap::Mapper<double>::Key) == sizeof(double),
                  "double key size mismatch");
}
} // namespace detail



#include <iterator>
#include <vector>
#include <array>
#include <functional>
#include <utility>
#include <numeric>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <limits>
#include <new>
#include <deque>
#include <random>
#include <tuple>
#include <cassert>
#include <chrono>

#if defined(_MSC_VER) || defined(__MINGW32__)
    #include <malloc.h>
#endif

// ═══════════════════════════════════════════════════════════════════════════
// 第二部分: SIMD检测和CPU特性
// ═══════════════════════════════════════════════════════════════════════════

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define FYX_X86 1
    #include <immintrin.h>
    
    // AVX-512完整支持
    #if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__)
        #define FYX_AVX512 1
        #define FYX_AVX512_FULL 1
        #define FYX_AVX2 1
        #define FYX_SSE42 1
        #define FYX_SIMD_WIDTH 64
    // AVX-512基础支持
    #elif defined(__AVX512F__) && defined(__AVX512DQ__)
        #define FYX_AVX512 1
        #define FYX_AVX512_BASIC 1
        #define FYX_AVX2 1
        #define FYX_SSE42 1
        #define FYX_SIMD_WIDTH 64
    // 仅AVX-512F
    #elif defined(__AVX512F__)
        #define FYX_AVX512 1
        #define FYX_AVX512_MINIMAL 1
        #define FYX_AVX2 1
        #define FYX_SSE42 1
        #define FYX_SIMD_WIDTH 64
    #elif defined(__AVX2__)
        #define FYX_AVX2 1
        #define FYX_SSE42 1
        #define FYX_SIMD_WIDTH 32
    #elif defined(__SSE4_2__)
        #define FYX_SSE42 1
        #define FYX_SIMD_WIDTH 16
    #elif defined(__SSE2__) || defined(_M_X64)
        #define FYX_SSE2 1
        #define FYX_SIMD_WIDTH 16
    #else
        #define FYX_SIMD_WIDTH 8
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define FYX_NEON 1
    #define FYX_ARM64 1
    #define FYX_SIMD_WIDTH 16
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define FYX_NEON 1
    #define FYX_SIMD_WIDTH 16
#else
    #define FYX_SIMD_WIDTH 8
#endif

// C++版本检测
#if __cplusplus >= 202002L
    #define FYX_CPP20 1
#endif
#if __cplusplus >= 201703L
    #define FYX_CPP17 1
#endif

namespace fyx {

// ═══════════════════════════════════════════════════════════════════════════
// 第三部分: 运行时CPU特性检测
// ═══════════════════════════════════════════════════════════════════════════

namespace cpu {
    struct Features {
        bool sse2 = false;
        bool sse42 = false;
        bool avx = false;
        bool avx2 = false;
        bool avx512f = false;
        bool avx512dq = false;
        bool avx512bw = false;
        bool avx512vl = false;
        bool popcnt = false;
        bool bmi1 = false;
        bool bmi2 = false;
        
        size_t l1_cache_size = 32 * 1024;
        size_t l2_cache_size = 256 * 1024;
        size_t l3_cache_size = 8 * 1024 * 1024;
        size_t cache_line_size = 64;
        
        bool initialized = false;
    };
    
    inline Features& get_features() {
        static Features f;
        if (!f.initialized) {
#ifdef FYX_X86
    #ifdef _MSC_VER
            int cpuinfo[4] = {0};
            __cpuid(cpuinfo, 0);
            int max_id = cpuinfo[0];
            
            if (max_id >= 1) {
                __cpuid(cpuinfo, 1);
                f.sse2 = (cpuinfo[3] & (1 << 26)) != 0;
                f.sse42 = (cpuinfo[2] & (1 << 20)) != 0;
                f.avx = (cpuinfo[2] & (1 << 28)) != 0;
                f.popcnt = (cpuinfo[2] & (1 << 23)) != 0;
            }
            
            if (max_id >= 7) {
                __cpuidex(cpuinfo, 7, 0);
                f.avx2 = (cpuinfo[1] & (1 << 5)) != 0;
                f.bmi1 = (cpuinfo[1] & (1 << 3)) != 0;
                f.bmi2 = (cpuinfo[1] & (1 << 8)) != 0;
                f.avx512f = (cpuinfo[1] & (1 << 16)) != 0;
                f.avx512dq = (cpuinfo[1] & (1 << 17)) != 0;
                f.avx512bw = (cpuinfo[1] & (1 << 30)) != 0;
                f.avx512vl = (cpuinfo[1] & (1u << 31)) != 0;
            }
    #elif defined(__GNUC__) || defined(__clang__)
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
                unsigned int max_id = eax;
                
                if (max_id >= 1) {
                    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
                    f.sse2 = (edx & (1 << 26)) != 0;
                    f.sse42 = (ecx & (1 << 20)) != 0;
                    f.avx = (ecx & (1 << 28)) != 0;
                    f.popcnt = (ecx & (1 << 23)) != 0;
                }
                
                if (max_id >= 7) {
                    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
                    f.avx2 = (ebx & (1 << 5)) != 0;
                    f.bmi1 = (ebx & (1 << 3)) != 0;
                    f.bmi2 = (ebx & (1 << 8)) != 0;
                    f.avx512f = (ebx & (1 << 16)) != 0;
                    f.avx512dq = (ebx & (1 << 17)) != 0;
                    f.avx512bw = (ebx & (1 << 30)) != 0;
                    f.avx512vl = (ebx & (1u << 31)) != 0;
                }
            }
    #endif
#endif
            // 使用编译时检测作为后备
#ifdef FYX_SSE2
            f.sse2 = true;
#endif
#ifdef FYX_SSE42
            f.sse42 = true;
#endif
#ifdef FYX_AVX2
            f.avx2 = true;
            f.avx = true;
#endif
#ifdef FYX_AVX512
            f.avx512f = true;
    #ifdef FYX_AVX512_FULL
            f.avx512dq = true;
            f.avx512bw = true;
            f.avx512vl = true;
    #endif
#endif
            f.initialized = true;
        }
        return f;
    }
    
    inline bool has_avx512() { return get_features().avx512f && get_features().avx512dq; }
    inline bool has_avx2() { return get_features().avx2; }
    inline bool has_sse42() { return get_features().sse42; }
}

// ═══════════════════════════════════════════════════════════════════════════
// 第四部分: 配置参数
// ═══════════════════════════════════════════════════════════════════════════

namespace config {
    // 缓存参数
    inline constexpr size_t CACHE_LINE = 64;
    inline constexpr size_t L1_SIZE = 32 * 1024;
    inline constexpr size_t L2_SIZE = 256 * 1024;
    inline constexpr size_t L3_SIZE = 8 * 1024 * 1024;
    
    // 算法阈值
    inline constexpr size_t TINY = 16;
    inline constexpr size_t SMALL = 32;
    inline constexpr size_t MEDIUM = 128;
    inline constexpr size_t INSERTION_LIMIT = 24;
    inline constexpr size_t NETWORK_SORT_LIMIT = 64;
    
    // 基数排序参数
    inline constexpr size_t RADIX_BITS = 8;
    inline constexpr size_t NUM_BUCKETS = 256;
    
    // 超标量采样参数
    inline constexpr size_t OVERSAMPLING_FACTOR = 16;
    inline constexpr size_t MIN_SAMPLES_PER_BUCKET = 4;
    inline constexpr size_t MAX_BUCKETS = 32;
    
    // 预取参数 - 多级预取
    inline constexpr size_t PREFETCH_DISTANCE_L1 = 64;
    inline constexpr size_t PREFETCH_DISTANCE_L2 = 256;
    inline constexpr size_t PREFETCH_DISTANCE_L3 = 1024;
    inline constexpr size_t BLOCK_SIZE = 4096;
    
    // 并行参数
    inline constexpr size_t PARALLEL_THRESHOLD = 16384;
    inline constexpr size_t MIN_PARALLEL_BLOCK = 4096;
    inline constexpr size_t WORK_STEAL_THRESHOLD = 2048;
    inline constexpr size_t MAX_STEAL_ATTEMPTS = 32;
    
    // 计数排序阈值
    inline constexpr size_t COUNTING_SORT_RATIO = 4;
    inline constexpr size_t COUNTING_MAX_RANGE = 1000000;
    inline constexpr size_t COUNTING_MIN_SIZE = 256;
    
    // 内存安全阈值
    inline constexpr size_t MAX_ALLOC_SIZE = size_t(1) << 40; // 1TB
    
    // 分层阈值
    inline constexpr size_t HIERARCHICAL_L1_THRESHOLD = 8000;    // L1缓存友好
    inline constexpr size_t HIERARCHICAL_L2_THRESHOLD = 100000;  // L2缓存友好
    inline constexpr size_t HIERARCHICAL_L3_THRESHOLD = 1000000; // L3缓存友好
    
    inline size_t num_threads() noexcept {
        static const size_t n = []() {
            unsigned hw = std::thread::hardware_concurrency();
            return hw > 0 ? static_cast<size_t>(hw) : 1;
        }();
        return n;
    }
    
    // 【修复#9】: 提供可配置的内存限制
    inline std::atomic<size_t>& max_memory_limit() {
        static std::atomic<size_t> limit{L3_SIZE * 8}; // 默认64MB
        return limit;
    }
    
    inline size_t available_memory() noexcept {
        return max_memory_limit().load(std::memory_order_relaxed);
    }
    
    inline void set_memory_limit(size_t bytes) noexcept {
        max_memory_limit().store(bytes, std::memory_order_relaxed);
    }
    
    template<typename T>
    inline constexpr size_t l1_block_size() noexcept {
        return (L1_SIZE / 2) / sizeof(T);
    }
    
    template<typename T>
    inline constexpr size_t l2_block_size() noexcept {
        return (L2_SIZE / 2) / sizeof(T);
    }
    
    template<typename T>
    inline constexpr size_t l3_block_size() noexcept {
        return (L3_SIZE / 2) / sizeof(T);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 第五部分: 排序选项
// ═══════════════════════════════════════════════════════════════════════════

struct Options {
    bool parallel = true;
    bool stable = false;
    size_t parallel_threshold = config::PARALLEL_THRESHOLD;
    size_t max_threads = 0;
    bool force_radix = false;
    bool force_comparison = false;
    bool adaptive = true;          // 自适应算法选择
    bool prefetch_aggressive = true; // 激进预取
    
    static Options defaults() { return {}; }
    static Options sequential() { Options o; o.parallel = false; return o; }
    static Options stable_sort() { Options o; o.stable = true; return o; }
    static Options radix_only() { Options o; o.force_radix = true; return o; }
    static Options comparison_only() { Options o; o.force_comparison = true; return o; }
    static Options maximum_performance() { 
        Options o; 
        o.adaptive = true; 
        o.prefetch_aggressive = true; 
        return o; 
    }
};

namespace detail {

// ═══════════════════════════════════════════════════════════════════════════
// 修复的内存管理 (替换原第六部分)
// ═══════════════════════════════════════════════════════════════════════════

namespace mem {
    // 安全的对齐内存分配
    inline void* aligned_alloc(size_t size, size_t align = config::CACHE_LINE) noexcept {
        if (size == 0) return nullptr;
        if (size > config::MAX_ALLOC_SIZE) return nullptr;
        
        // 修复：确保对齐值有效
        if (align == 0) align = config::CACHE_LINE;
        if (align < sizeof(void*)) align = sizeof(void*);
        // 确保是2的幂
        if ((align & (align - 1)) != 0) {
            // 向上取整到2的幂
            size_t p = 1;
            while (p < align) p <<= 1;
            align = p;
        }
        
        // 溢出检查
        size_t aligned_size = (size + align - 1) & ~(align - 1);
        if (aligned_size < size) return nullptr;
        
#if defined(_MSC_VER) || defined(__MINGW32__)
        void* p = _aligned_malloc(aligned_size, align);
        return p;
#else
        void* p = nullptr;
        if (posix_memalign(&p, align, aligned_size) != 0) {
            return nullptr;
        }
        return p;
#endif
    }
    
    inline void aligned_free(void* p) noexcept {
        if (!p) return;
#if defined(_MSC_VER) || defined(__MINGW32__)
        _aligned_free(p);
#else
        ::free(p);
#endif
    }
    
    enum class AllocStatus {
        Success,
        ZeroSize,
        TooLarge,
        OutOfMemory
    };
    
    template<typename T>
    class alignas(config::CACHE_LINE) Buffer {
    public:
        T* ptr = nullptr;
        size_t len = 0;
        size_t capacity_ = 0;  // 修复：添加容量跟踪
        AllocStatus status = AllocStatus::ZeroSize;
        
        Buffer() = default;
        
        explicit Buffer(size_t n) {
            allocate(n);
        }
        
        ~Buffer() { 
            aligned_free(ptr); 
        }
        
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        
        Buffer(Buffer&& o) noexcept 
            : ptr(o.ptr), len(o.len), capacity_(o.capacity_), status(o.status) { 
            o.ptr = nullptr; 
            o.len = 0;
            o.capacity_ = 0;
            o.status = AllocStatus::ZeroSize;
        }
        
        Buffer& operator=(Buffer&& o) noexcept {
            if (this != &o) {
                aligned_free(ptr);
                ptr = o.ptr; 
                len = o.len;
                capacity_ = o.capacity_;
                status = o.status;
                o.ptr = nullptr; 
                o.len = 0;
                o.capacity_ = 0;
                o.status = AllocStatus::ZeroSize;
            }
            return *this;
        }
        
        T& operator[](size_t i) noexcept { return ptr[i]; }
        const T& operator[](size_t i) const noexcept { return ptr[i]; }
        
        T* data() noexcept { return ptr; }
        const T* data() const noexcept { return ptr; }
        size_t size() const noexcept { return len; }
        size_t capacity() const noexcept { return capacity_; }
        
        explicit operator bool() const noexcept { 
            return ptr != nullptr && status == AllocStatus::Success; 
        }
        
        bool valid() const noexcept {
            return status == AllocStatus::Success && ptr != nullptr;
        }
        
        AllocStatus get_status() const noexcept { return status; }
        
        void clear() { 
            aligned_free(ptr); 
            ptr = nullptr; 
            len = 0;
            capacity_ = 0;
            status = AllocStatus::ZeroSize;
        }
        
    private:
        bool allocate(size_t n) {
            if (n == 0) {
                status = AllocStatus::ZeroSize;
                return true;
            }
            
            if (n > config::MAX_ALLOC_SIZE / sizeof(T)) {
                status = AllocStatus::TooLarge;
                return false;
            }
            
            ptr = static_cast<T*>(aligned_alloc(n * sizeof(T)));
            if (ptr) {
                len = n;
                capacity_ = n;
                status = AllocStatus::Success;
                return true;
            } else {
                len = 0;
                capacity_ = 0;
                status = AllocStatus::OutOfMemory;
                return false;
            }
        }
        
    public:
        // 修复：resize保留数据选项
        bool resize(size_t n, bool preserve_data = false) {
            if (n == 0) {
                clear();
                return true;
            }
            
            // 如果容量足够，只调整长度
            if (n <= capacity_ && ptr) {
                len = n;
                return true;
            }
            
            // 需要重新分配
            if (n > config::MAX_ALLOC_SIZE / sizeof(T)) {
                status = AllocStatus::TooLarge;
                return false;
            }
            
            T* new_ptr = static_cast<T*>(aligned_alloc(n * sizeof(T)));
            if (!new_ptr) {
                status = AllocStatus::OutOfMemory;
                return false;
            }
            
            // 修复：可选保留数据
            if (preserve_data && ptr && len > 0) {
                size_t copy_count = std::min(len, n);
                if constexpr (std::is_trivially_copyable_v<T>) {
                    std::memcpy(new_ptr, ptr, copy_count * sizeof(T));
                } else {
                    for (size_t i = 0; i < copy_count; ++i) {
                        new (new_ptr + i) T(std::move(ptr[i]));
                        ptr[i].~T();
                    }
                }
            }
            
            aligned_free(ptr);
            ptr = new_ptr;
            len = n;
            capacity_ = n;
            status = AllocStatus::Success;
            return true;
        }
        
        // 保留数据的resize
        bool resize_preserve(size_t n) {
            return resize(n, true);
        }
    };
    
    template<typename T, size_t N>
    struct alignas(config::CACHE_LINE) AlignedArray {
        T data[N];
        
        T& operator[](size_t i) noexcept { return data[i]; }
        const T& operator[](size_t i) const noexcept { return data[i]; }
        
        T* begin() noexcept { return data; }
        T* end() noexcept { return data + N; }
        const T* begin() const noexcept { return data; }
        const T* end() const noexcept { return data + N; }
        
        static constexpr size_t size() noexcept { return N; }
        
        void fill(const T& val) noexcept {
            for (size_t i = 0; i < N; ++i) data[i] = val;
        }
        
        void zero() noexcept {
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memset(data, 0, sizeof(data));
            } else {
                for (size_t i = 0; i < N; ++i) data[i] = T{};
            }
        }
    };
    
    template<typename T, size_t StackSize = 256>
    class SmallBuffer {
        alignas(config::CACHE_LINE) T stack_buf[StackSize];
        T* heap_ptr = nullptr;
        size_t len = 0;
        bool using_heap = false;
        
    public:
        SmallBuffer() = default;
        
        explicit SmallBuffer(size_t n) : len(n) {
            if (n <= StackSize) {
                using_heap = false;
            } else {
                heap_ptr = static_cast<T*>(aligned_alloc(n * sizeof(T)));
                using_heap = heap_ptr != nullptr;
                if (!using_heap) len = 0;
            }
        }
        
        ~SmallBuffer() {
            if (using_heap && heap_ptr) {
                aligned_free(heap_ptr);
            }
        }
        
        SmallBuffer(const SmallBuffer&) = delete;
        SmallBuffer& operator=(const SmallBuffer&) = delete;
        
        // 移动构造
        SmallBuffer(SmallBuffer&& o) noexcept : len(o.len), using_heap(o.using_heap) {
            if (using_heap) {
                heap_ptr = o.heap_ptr;
                o.heap_ptr = nullptr;
            } else {
                for (size_t i = 0; i < len; ++i) {
                    stack_buf[i] = std::move(o.stack_buf[i]);
                }
            }
            o.len = 0;
            o.using_heap = false;
        }
        
        T* data() noexcept { return using_heap ? heap_ptr : stack_buf; }
        const T* data() const noexcept { return using_heap ? heap_ptr : stack_buf; }
        size_t size() const noexcept { return len; }
        explicit operator bool() const noexcept { return len > 0; }
        
        T& operator[](size_t i) noexcept { return data()[i]; }
        const T& operator[](size_t i) const noexcept { return data()[i]; }
    };
} // namespace mem

// ═══════════════════════════════════════════════════════════════════════════
// 第七部分: 流式多级预取
// ═══════════════════════════════════════════════════════════════════════════

namespace prefetch {
    template<typename T>
    FYX_INLINE void stream_prefetch(const T* ptr, size_t count) noexcept {
        // L1预取: 最近的数据
        FYX_PREFETCH_T0(ptr);
        if (count > 16) {
            FYX_PREFETCH_T0(ptr + 8);
        }
        
        // L2预取: 中等距离
        if (count > 64) {
            FYX_PREFETCH_T1(ptr + config::PREFETCH_DISTANCE_L1 / sizeof(T));
        }
        
        // L3预取: 远距离
        if (count > 256) {
            FYX_PREFETCH_T2(ptr + config::PREFETCH_DISTANCE_L2 / sizeof(T));
        }
    }
    
    template<typename T>
    FYX_INLINE void prefetch_range(const T* ptr, size_t n, size_t stride = 8) noexcept {
        for (size_t i = 0; i < n && i < 4 * stride; i += stride) {
            FYX_PREFETCH_T0(ptr + i);
        }
    }
    
    template<typename T>
    FYX_INLINE void prefetch_write(T* ptr) noexcept {
        FYX_PREFETCH_W(ptr);
    }
    
    // 连续预取多个缓存行
    template<typename T>
    FYX_INLINE void prefetch_continuous(const T* ptr, size_t cache_lines) noexcept {
        for (size_t i = 0; i < cache_lines; ++i) {
            FYX_PREFETCH_T0(reinterpret_cast<const char*>(ptr) + i * config::CACHE_LINE);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 第八部分: 迭代器和类型特征
// ═══════════════════════════════════════════════════════════════════════════

namespace iter_traits {
    template<typename It>
    inline constexpr bool is_random_access_v = 
        std::is_same_v<typename std::iterator_traits<It>::iterator_category, 
                       std::random_access_iterator_tag>;
    
    template<typename It, typename = void>
    struct is_contiguous_impl : std::false_type {};
    
    template<typename It>
    struct is_contiguous_impl<It, std::enable_if_t<std::is_pointer_v<It>>> : std::true_type {};
    
    template<typename It>
    inline constexpr bool is_contiguous_v = is_contiguous_impl<It>::value || std::is_pointer_v<It>;
    
    template<typename It>
    using value_type_t = typename std::iterator_traits<It>::value_type;
    
    template<typename It>
    using difference_type_t = typename std::iterator_traits<It>::difference_type;
}

namespace traits {
    template<typename T>
    inline constexpr bool is_radix_sortable_v = 
        std::is_integral_v<T> || std::is_floating_point_v<T>;
    
    template<typename T>
    inline constexpr bool use_indirect_v = sizeof(T) > 64;
    
    template<typename T, typename = void>
    struct has_data : std::false_type {};
    
    template<typename T>
    struct has_data<T, std::void_t<decltype(std::declval<T>().data())>> : std::true_type {};
    
    template<typename T>
    inline constexpr bool is_contiguous_v = has_data<T>::value;
    
    template<typename T>
    inline constexpr bool is_trivially_copyable_v = std::is_trivially_copyable<T>::value;
    
    template<typename T>
    inline constexpr bool can_use_avx512_v = 
        (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
         std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> ||
         std::is_same_v<T, float> || std::is_same_v<T, double>);
    
    template<typename T, typename Cmp>
    inline constexpr bool is_default_less_v = 
        std::is_same_v<std::decay_t<Cmp>, std::less<T>> ||
        std::is_same_v<std::decay_t<Cmp>, std::less<>>;
    
    template<typename T>
    inline constexpr bool is_trivially_movable_v = 
        std::is_trivially_copyable_v<T> || 
        (std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_assignable_v<T>);
    
    template<typename T>
    inline constexpr bool can_use_simd_v = 
        std::is_trivially_copyable_v<T> && 
        (sizeof(T) == 4 || sizeof(T) == 8) &&
        (std::is_integral_v<T> || std::is_floating_point_v<T>);
}

// ═══════════════════════════════════════════════════════════════════════════
// 第九部分: 基本操作
// ═══════════════════════════════════════════════════════════════════════════

namespace ops {
    template<typename T>
    FYX_INLINE void swap(T& a, T& b) noexcept(std::is_nothrow_move_constructible_v<T> && 
                                               std::is_nothrow_move_assignable_v<T>) {
        T t = std::move(a); 
        a = std::move(b); 
        b = std::move(t);
    }
    
    template<typename T, typename Cmp>
    FYX_INLINE void cswap(T& a, T& b, Cmp& cmp) noexcept {
        if (cmp(b, a)) swap(a, b);
    }
    
    // 无分支条件交换
    template<typename T>
    FYX_INLINE void cswap_branchless(T& a, T& b) noexcept {
        if constexpr (std::is_integral_v<T> && sizeof(T) <= 8) {
            using U = std::make_unsigned_t<T>;
            U ua = static_cast<U>(a);
            U ub = static_cast<U>(b);
            U diff = ua - ub;
            U mask = static_cast<U>(static_cast<std::make_signed_t<U>>(diff) >> (sizeof(T) * 8 - 1));
            U tmp = (ua ^ ub) & mask;
            a = static_cast<T>(ua ^ tmp);
            b = static_cast<T>(ub ^ tmp);
        } else {
            if (b < a) swap(a, b);
        }
    }
    
    template<typename T>
    FYX_INLINE T min_val(T a, T b) noexcept {
        return a < b ? a : b;
    }
    
    template<typename T>
    FYX_INLINE T max_val(T a, T b) noexcept {
        return a > b ? a : b;
    }
    
    // 批量移动（带多级预取）
    template<typename T>
    FYX_INLINE void move_range(T* FYX_RESTRICT dst, T* FYX_RESTRICT src, size_t n) noexcept {
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memmove(dst, src, n * sizeof(T));
        } else {
            for (size_t i = 0; i < n; ++i) {
                if (i + 8 < n) FYX_PREFETCH_T0(src + i + 8);
                if (i + 32 < n) FYX_PREFETCH_T1(src + i + 32);
                dst[i] = std::move(src[i]);
            }
        }
    }
    
    template<typename T>
    FYX_INLINE void copy_range(T* FYX_RESTRICT dst, const T* FYX_RESTRICT src, size_t n) noexcept {
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memcpy(dst, src, n * sizeof(T));
        } else {
            for (size_t i = 0; i < n; ++i) {
                if (i + 8 < n) FYX_PREFETCH_T0(src + i + 8);
                dst[i] = src[i];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 第十部分: 键映射（基数排序用）
// ═══════════════════════════════════════════════════════════════════════════

namespace keymap {
    template<typename T, typename = void> struct Mapper;
    
    template<typename T>
    struct Mapper<T, std::enable_if_t<std::is_unsigned_v<T>>> {
        using Key = T;
        static FYX_INLINE Key to_key(T v) noexcept { return v; }
    };
    
    template<typename T>
    struct Mapper<T, std::enable_if_t<std::is_signed_v<T> && std::is_integral_v<T>>> {
        using Key = std::make_unsigned_t<T>;
        static constexpr Key FLIP = Key(1) << (sizeof(T) * 8 - 1);
        static FYX_INLINE Key to_key(T v) noexcept { 
            return static_cast<Key>(v) ^ FLIP; 
        }
    };
    
    template<> struct Mapper<float> {
        using Key = uint32_t;
        static FYX_INLINE Key to_key(float v) noexcept {
            Key k; 
            std::memcpy(&k, &v, sizeof(k));
            Key mask = -static_cast<Key>(k >> 31) | 0x80000000U;
            return k ^ mask;
        }
    };
    
    template<> struct Mapper<double> {
        using Key = uint64_t;
        static FYX_INLINE Key to_key(double v) noexcept {
            Key k; 
            std::memcpy(&k, &v, sizeof(k));
            Key mask = -static_cast<Key>(k >> 63) | 0x8000000000000000ULL;
            return k ^ mask;
        }
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// 第十一部分: 预计算SIMD置换表 (修复版 - 条件编译)
// ═══════════════════════════════════════════════════════════════════════════

#ifdef FYX_AVX512
namespace simd_tables {
    struct alignas(64) PrecomputedTables {
        __m512i rev_idx_16;
        __m512i rev_idx_8;
        uint16_t blend_AAAA;
        uint16_t blend_CCCC;
        uint16_t blend_F0F0;
        uint16_t blend_FF00;
        uint8_t blend_AA;
        uint8_t blend_CC;
        uint8_t blend_F0;
        
        PrecomputedTables() noexcept {
            rev_idx_16 = _mm512_set_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
            rev_idx_8 = _mm512_set_epi64(0,1,2,3,4,5,6,7);
            blend_AAAA = 0xAAAA;
            blend_CCCC = 0xCCCC;
            blend_F0F0 = 0xF0F0;
            blend_FF00 = 0xFF00;
            blend_AA = 0xAA;
            blend_CC = 0xCC;
            blend_F0 = 0xF0;
        }
    };
    
    // 线程安全的单例获取
    inline const PrecomputedTables& get_tables() noexcept {
        // C++11保证静态局部变量线程安全初始化
        static const PrecomputedTables tables;
        return tables;
    }
}
#endif // FYX_AVX512

// 为非AVX512平台提供空实现
#ifndef FYX_AVX512
namespace simd_tables {
    struct PrecomputedTables {};
    inline const PrecomputedTables& get_tables() noexcept {
        static const PrecomputedTables tables;
        return tables;
    }
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
// 第十二部分: AVX-512 SIMD核心
// ═══════════════════════════════════════════════════════════════════════════

#ifdef FYX_AVX512
namespace simd512 {
    using namespace simd_tables;
    
    // 16x int32 排序网络 (优化版)
    FYX_INLINE __m512i sort_16xi32(__m512i v) noexcept {
        __m512i t, mn, mx;
        const auto& tbl = get_tables();
        
        // Layer 1
        t = _mm512_shuffle_epi32(v, _MM_PERM_CDAB);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_AAAA, mn, mx);
        
        // Layer 2
        t = _mm512_shuffle_epi32(v, _MM_PERM_BADC);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_CCCC, mn, mx);
        
        // Layer 3
        t = _mm512_shuffle_epi32(v, _MM_PERM_CDAB);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_AAAA, mn, mx);
        
        // Layer 4
        t = _mm512_shuffle_i32x4(v, v, 0xB1);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_F0F0, mn, mx);
        
        // Layer 5
        t = _mm512_shuffle_epi32(v, _MM_PERM_BADC);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_CCCC, mn, mx);
        
        // Layer 6
        t = _mm512_shuffle_epi32(v, _MM_PERM_CDAB);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_AAAA, mn, mx);
        
        // Layer 7
        t = _mm512_shuffle_i32x4(v, v, 0x4E);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_FF00, mn, mx);
        
        // Layer 8
        t = _mm512_shuffle_i32x4(v, v, 0xB1);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_F0F0, mn, mx);
        
        // Layer 9
        t = _mm512_shuffle_epi32(v, _MM_PERM_BADC);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_CCCC, mn, mx);
        
        // Layer 10
        t = _mm512_shuffle_epi32(v, _MM_PERM_CDAB);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_AAAA, mn, mx);
        
        return v;
    }
    
    // 16元素合并 (优化版)
    FYX_INLINE void bitonic_merge_16(__m512i& v) noexcept {
        __m512i t, mn, mx;
        const auto& tbl = get_tables();
        
        t = _mm512_shuffle_i32x4(v, v, 0x4E);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_FF00, mn, mx);
        
        t = _mm512_shuffle_i32x4(v, v, 0xB1);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_F0F0, mn, mx);
        
        t = _mm512_shuffle_epi32(v, _MM_PERM_BADC);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_CCCC, mn, mx);
        
        t = _mm512_shuffle_epi32(v, _MM_PERM_CDAB);
        mn = _mm512_min_epi32(v, t);
        mx = _mm512_max_epi32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_AAAA, mn, mx);
    }
    
    // 32x int32 排序 (优化版)
    FYX_INLINE void sort_32xi32(int32_t* arr) noexcept {
        const auto& tbl = get_tables();
        
        __m512i v0 = _mm512_loadu_si512(arr);
        __m512i v1 = _mm512_loadu_si512(arr + 16);
        
        v0 = sort_16xi32(v0);
        v1 = sort_16xi32(v1);
        
        v1 = _mm512_permutexvar_epi32(tbl.rev_idx_16, v1);
        
        __m512i mn = _mm512_min_epi32(v0, v1);
        __m512i mx = _mm512_max_epi32(v0, v1);
        v0 = mn;
        v1 = mx;
        
        bitonic_merge_16(v0);
        bitonic_merge_16(v1);
        
        _mm512_storeu_si512(arr, v0);
        _mm512_storeu_si512(arr + 16, v1);
    }
    
    // 64x int32 排序 (优化版)
    FYX_NOINLINE void sort_64xi32(int32_t* arr) noexcept {
        const auto& tbl = get_tables();
        
        // 预取
        FYX_PREFETCH_T0(arr + 64);
        
        __m512i v0 = sort_16xi32(_mm512_loadu_si512(arr));
        __m512i v1 = sort_16xi32(_mm512_loadu_si512(arr + 16));
        __m512i v2 = sort_16xi32(_mm512_loadu_si512(arr + 32));
        __m512i v3 = sort_16xi32(_mm512_loadu_si512(arr + 48));
        
        // 阶段1
        v1 = _mm512_permutexvar_epi32(tbl.rev_idx_16, v1);
        v3 = _mm512_permutexvar_epi32(tbl.rev_idx_16, v3);
        
        __m512i t0 = _mm512_min_epi32(v0, v1);
        __m512i t1 = _mm512_max_epi32(v0, v1);
        __m512i t2 = _mm512_min_epi32(v2, v3);
        __m512i t3 = _mm512_max_epi32(v2, v3);
        
        bitonic_merge_16(t0);
        bitonic_merge_16(t1);
        bitonic_merge_16(t2);
        bitonic_merge_16(t3);
        
        v0 = t0; v1 = t1; v2 = t2; v3 = t3;
        
        // 阶段2
        v2 = _mm512_permutexvar_epi32(tbl.rev_idx_16, v2);
        v3 = _mm512_permutexvar_epi32(tbl.rev_idx_16, v3);
        
        t0 = _mm512_min_epi32(v0, v3);
        t3 = _mm512_max_epi32(v0, v3);
        t1 = _mm512_min_epi32(v1, v2);
        t2 = _mm512_max_epi32(v1, v2);
        
        v0 = t0; v1 = t1; v2 = t2; v3 = t3;
        
        v1 = _mm512_permutexvar_epi32(tbl.rev_idx_16, v1);
        __m512i mn01 = _mm512_min_epi32(v0, v1);
        __m512i mx01 = _mm512_max_epi32(v0, v1);
        bitonic_merge_16(mn01);
        bitonic_merge_16(mx01);
        v0 = mn01;
        v1 = mx01;
        
        v3 = _mm512_permutexvar_epi32(tbl.rev_idx_16, v3);
        __m512i mn23 = _mm512_min_epi32(v2, v3);
        __m512i mx23 = _mm512_max_epi32(v2, v3);
        bitonic_merge_16(mn23);
        bitonic_merge_16(mx23);
        v2 = mn23;
        v3 = mx23;
        
        _mm512_storeu_si512(arr, v0);
        _mm512_storeu_si512(arr + 16, v1);
        _mm512_storeu_si512(arr + 32, v2);
        _mm512_storeu_si512(arr + 48, v3);
    }
    
    // 128x int32 排序
    FYX_NOINLINE void sort_128xi32(int32_t* arr) noexcept {
        sort_64xi32(arr);
        sort_64xi32(arr + 64);
        
        // 归并两个64元素排序块
        const auto& tbl = get_tables();
        
        for (int i = 3; i >= 0; --i) {
            __m512i a = _mm512_loadu_si512(arr + 48 + i * 16 - 48);
            __m512i b = _mm512_loadu_si512(arr + 64 + (3 - i) * 16);
            b = _mm512_permutexvar_epi32(tbl.rev_idx_16, b);
            
            __m512i mn = _mm512_min_epi32(a, b);
            __m512i mx = _mm512_max_epi32(a, b);
            
            bitonic_merge_16(mn);
            bitonic_merge_16(mx);
            
            _mm512_storeu_si512(arr + i * 16, mn);
            _mm512_storeu_si512(arr + 64 + (3 - i) * 16, 
                _mm512_permutexvar_epi32(tbl.rev_idx_16, mx));
        }
        
        // 最终合并
        sort_64xi32(arr);
        sort_64xi32(arr + 64);
    }
    
    // 16x uint32 排序
    FYX_INLINE __m512i sort_16xu32(__m512i v) noexcept {
        __m512i t, mn, mx;
        const auto& tbl = get_tables();
        
        t = _mm512_shuffle_epi32(v, _MM_PERM_CDAB);
        mn = _mm512_min_epu32(v, t); mx = _mm512_max_epu32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_AAAA, mn, mx);
        
        t = _mm512_shuffle_epi32(v, _MM_PERM_BADC);
        mn = _mm512_min_epu32(v, t); mx = _mm512_max_epu32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_CCCC, mn, mx);
        
        t = _mm512_shuffle_epi32(v, _MM_PERM_CDAB);
        mn = _mm512_min_epu32(v, t); mx = _mm512_max_epu32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_AAAA, mn, mx);
        
        t = _mm512_shuffle_i32x4(v, v, 0xB1);
        mn = _mm512_min_epu32(v, t); mx = _mm512_max_epu32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_F0F0, mn, mx);
        
        t = _mm512_shuffle_epi32(v, _MM_PERM_BADC);
        mn = _mm512_min_epu32(v, t); mx = _mm512_max_epu32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_CCCC, mn, mx);
        
        t = _mm512_shuffle_epi32(v, _MM_PERM_CDAB);
        mn = _mm512_min_epu32(v, t); mx = _mm512_max_epu32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_AAAA, mn, mx);
        
        t = _mm512_shuffle_i32x4(v, v, 0x4E);
        mn = _mm512_min_epu32(v, t); mx = _mm512_max_epu32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_FF00, mn, mx);
        
        t = _mm512_shuffle_i32x4(v, v, 0xB1);
        mn = _mm512_min_epu32(v, t); mx = _mm512_max_epu32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_F0F0, mn, mx);
        
        t = _mm512_shuffle_epi32(v, _MM_PERM_BADC);
        mn = _mm512_min_epu32(v, t); mx = _mm512_max_epu32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_CCCC, mn, mx);
        
        t = _mm512_shuffle_epi32(v, _MM_PERM_CDAB);
        mn = _mm512_min_epu32(v, t); mx = _mm512_max_epu32(v, t);
        v = _mm512_mask_blend_epi32(tbl.blend_AAAA, mn, mx);
        
        return v;
    }
    
    // 8x int64 排序
    FYX_INLINE __m512i sort_8xi64(__m512i v) noexcept {
        __m512i t, mn, mx;
        const auto& tbl = get_tables();
        
        t = _mm512_shuffle_i64x2(v, v, 0xB1);
        t = _mm512_permutex_epi64(t, 0x4E);
        mn = _mm512_min_epi64(v, t);
        mx = _mm512_max_epi64(v, t);
        v = _mm512_mask_blend_epi64(tbl.blend_AA, mn, mx);
        
        t = _mm512_permutex_epi64(v, 0x4E);
        mn = _mm512_min_epi64(v, t);
        mx = _mm512_max_epi64(v, t);
        v = _mm512_mask_blend_epi64(tbl.blend_CC, mn, mx);
        
        t = _mm512_shuffle_i64x2(v, v, 0xB1);
        t = _mm512_permutex_epi64(t, 0x4E);
        mn = _mm512_min_epi64(v, t);
        mx = _mm512_max_epi64(v, t);
        v = _mm512_mask_blend_epi64(tbl.blend_AA, mn, mx);
        
        t = _mm512_shuffle_i64x2(v, v, 0x4E);
        mn = _mm512_min_epi64(v, t);
        mx = _mm512_max_epi64(v, t);
        v = _mm512_mask_blend_epi64(tbl.blend_F0, mn, mx);
        
        t = _mm512_permutex_epi64(v, 0x4E);
        mn = _mm512_min_epi64(v, t);
        mx = _mm512_max_epi64(v, t);
        v = _mm512_mask_blend_epi64(tbl.blend_CC, mn, mx);
        
        t = _mm512_shuffle_i64x2(v, v, 0xB1);
        t = _mm512_permutex_epi64(t, 0x4E);
        mn = _mm512_min_epi64(v, t);
        mx = _mm512_max_epi64(v, t);
        v = _mm512_mask_blend_epi64(tbl.blend_AA, mn, mx);
        
        return v;
    }
    
    // 8x uint64 排序
    FYX_INLINE __m512i sort_8xu64(__m512i v) noexcept {
        __m512i t, mn, mx;
        const auto& tbl = get_tables();
        
        t = _mm512_shuffle_i64x2(v, v, 0xB1);
        t = _mm512_permutex_epi64(t, 0x4E);
        mn = _mm512_min_epu64(v, t);
        mx = _mm512_max_epu64(v, t);
        v = _mm512_mask_blend_epi64(tbl.blend_AA, mn, mx);
        
        t = _mm512_permutex_epi64(v, 0x4E);
        mn = _mm512_min_epu64(v, t);
        mx = _mm512_max_epu64(v, t);
        v = _mm512_mask_blend_epi64(tbl.blend_CC, mn, mx);
        
        t = _mm512_shuffle_i64x2(v, v, 0xB1);
        t = _mm512_permutex_epi64(t, 0x4E);
        mn = _mm512_min_epu64(v, t);
        mx = _mm512_max_epu64(v, t);
        v = _mm512_mask_blend_epi64(tbl.blend_AA, mn, mx);
        
        t = _mm512_shuffle_i64x2(v, v, 0x4E);
        mn = _mm512_min_epu64(v, t);
        mx = _mm512_max_epu64(v, t);
        v = _mm512_mask_blend_epi64(tbl.blend_F0, mn, mx);
        
        t = _mm512_permutex_epi64(v, 0x4E);
        mn = _mm512_min_epu64(v, t);
        mx = _mm512_max_epu64(v, t);
        v = _mm512_mask_blend_epi64(tbl.blend_CC, mn, mx);
        
        t = _mm512_shuffle_i64x2(v, v, 0xB1);
        t = _mm512_permutex_epi64(t, 0x4E);
        mn = _mm512_min_epu64(v, t);
        mx = _mm512_max_epu64(v, t);
        v = _mm512_mask_blend_epi64(tbl.blend_AA, mn, mx);
        
        return v;
    }
    
    // 8x double 排序
    FYX_INLINE __m512d sort_8xf64(__m512d v) noexcept {
        __m512d t, mn, mx;
        const auto& tbl = get_tables();
        
        t = _mm512_shuffle_pd(v, v, 0x55);
        mn = _mm512_min_pd(v, t);
        mx = _mm512_max_pd(v, t);
        v = _mm512_mask_blend_pd(tbl.blend_AA, mn, mx);
        
        t = _mm512_permutex_pd(v, 0x4E);
        mn = _mm512_min_pd(v, t);
        mx = _mm512_max_pd(v, t);
        v = _mm512_mask_blend_pd(tbl.blend_CC, mn, mx);
        
        t = _mm512_shuffle_pd(v, v, 0x55);
        mn = _mm512_min_pd(v, t);
        mx = _mm512_max_pd(v, t);
        v = _mm512_mask_blend_pd(tbl.blend_AA, mn, mx);
        
        t = _mm512_shuffle_f64x2(v, v, 0x4E);
        mn = _mm512_min_pd(v, t);
        mx = _mm512_max_pd(v, t);
        v = _mm512_mask_blend_pd(tbl.blend_F0, mn, mx);
        
        t = _mm512_permutex_pd(v, 0x4E);
        mn = _mm512_min_pd(v, t);
        mx = _mm512_max_pd(v, t);
        v = _mm512_mask_blend_pd(tbl.blend_CC, mn, mx);
        
        t = _mm512_shuffle_pd(v, v, 0x55);
        mn = _mm512_min_pd(v, t);
        mx = _mm512_max_pd(v, t);
        v = _mm512_mask_blend_pd(tbl.blend_AA, mn, mx);
        
        return v;
    }
    
    // 16x float 排序
    FYX_INLINE __m512 sort_16xf32(__m512 v) noexcept {
        __m512 t, mn, mx;
        const auto& tbl = get_tables();
        
        t = _mm512_shuffle_ps(v, v, 0xB1);
        mn = _mm512_min_ps(v, t);
        mx = _mm512_max_ps(v, t);
        v = _mm512_mask_blend_ps(tbl.blend_AAAA, mn, mx);
        
        t = _mm512_shuffle_ps(v, v, 0x4E);
        mn = _mm512_min_ps(v, t);
        mx = _mm512_max_ps(v, t);
        v = _mm512_mask_blend_ps(tbl.blend_CCCC, mn, mx);
        
        t = _mm512_shuffle_ps(v, v, 0xB1);
        mn = _mm512_min_ps(v, t);
        mx = _mm512_max_ps(v, t);
        v = _mm512_mask_blend_ps(tbl.blend_AAAA, mn, mx);
        
        t = _mm512_shuffle_f32x4(v, v, 0xB1);
        mn = _mm512_min_ps(v, t);
        mx = _mm512_max_ps(v, t);
        v = _mm512_mask_blend_ps(tbl.blend_F0F0, mn, mx);
        
        t = _mm512_shuffle_ps(v, v, 0x4E);
        mn = _mm512_min_ps(v, t);
        mx = _mm512_max_ps(v, t);
        v = _mm512_mask_blend_ps(tbl.blend_CCCC, mn, mx);
        
        t = _mm512_shuffle_ps(v, v, 0xB1);
        mn = _mm512_min_ps(v, t);
        mx = _mm512_max_ps(v, t);
        v = _mm512_mask_blend_ps(tbl.blend_AAAA, mn, mx);
        
        t = _mm512_shuffle_f32x4(v, v, 0x4E);
        mn = _mm512_min_ps(v, t);
        mx = _mm512_max_ps(v, t);
        v = _mm512_mask_blend_ps(tbl.blend_FF00, mn, mx);
        
        t = _mm512_shuffle_f32x4(v, v, 0xB1);
        mn = _mm512_min_ps(v, t);
        mx = _mm512_max_ps(v, t);
        v = _mm512_mask_blend_ps(tbl.blend_F0F0, mn, mx);
        
        t = _mm512_shuffle_ps(v, v, 0x4E);
        mn = _mm512_min_ps(v, t);
        mx = _mm512_max_ps(v, t);
        v = _mm512_mask_blend_ps(tbl.blend_CCCC, mn, mx);
        
        t = _mm512_shuffle_ps(v, v, 0xB1);
        mn = _mm512_min_ps(v, t);
        mx = _mm512_max_ps(v, t);
        v = _mm512_mask_blend_ps(tbl.blend_AAAA, mn, mx);
        
        return v;
    }
    
    // 16x double 排序
    FYX_INLINE void sort_16xf64(double* arr) noexcept {
        const auto& tbl = get_tables();
        
        __m512d v0 = _mm512_loadu_pd(arr);
        __m512d v1 = _mm512_loadu_pd(arr + 8);
        
        v0 = sort_8xf64(v0);
        v1 = sort_8xf64(v1);
        
        v1 = _mm512_permutexvar_pd(tbl.rev_idx_8, v1);
        
        __m512d mn = _mm512_min_pd(v0, v1);
        __m512d mx = _mm512_max_pd(v0, v1);
        v0 = mn;
        v1 = mx;
        
        auto merge8 = [&tbl](__m512d& v) {
            __m512d t, mn, mx;
            const __m512i idx4 = _mm512_set_epi64(3,2,1,0,7,6,5,4);
            
            t = _mm512_permutexvar_pd(idx4, v);
            mn = _mm512_min_pd(v, t);
            mx = _mm512_max_pd(v, t);
            v = _mm512_mask_blend_pd(tbl.blend_F0, mn, mx);
            
            t = _mm512_permutex_pd(v, 0x4E);
            mn = _mm512_min_pd(v, t);
            mx = _mm512_max_pd(v, t);
            v = _mm512_mask_blend_pd(tbl.blend_CC, mn, mx);
            
            t = _mm512_shuffle_pd(v, v, 0x55);
            mn = _mm512_min_pd(v, t);
            mx = _mm512_max_pd(v, t);
            v = _mm512_mask_blend_pd(tbl.blend_AA, mn, mx);
        };
        
        merge8(v0);
        merge8(v1);
        
        _mm512_storeu_pd(arr, v0);
        _mm512_storeu_pd(arr + 8, v1);
    }
    
    // MinMax (优化版 - 带多级预取)
    template<typename T>
    std::pair<T, T> find_minmax_512(const T* data, size_t n) noexcept {
        if (FYX_UNLIKELY(n == 0)) return {T{}, T{}};
        if (FYX_UNLIKELY(n == 1)) return {data[0], data[0]};
        
        auto scalar_minmax = [](const T* d, size_t cnt) -> std::pair<T, T> {
            T mn = d[0], mx = d[0];
            for (size_t i = 1; i < cnt; ++i) {
                if (d[i] < mn) mn = d[i];
                if (d[i] > mx) mx = d[i];
            }
            return {mn, mx};
        };
        
        if constexpr (std::is_same_v<T, int32_t>) {
            if (n < 16) return scalar_minmax(data, n);
            
            __m512i vmin = _mm512_loadu_si512(data);
            __m512i vmax = vmin;
            
            size_t i = 16;
            for (; i + 64 <= n; i += 64) {
                // 多级预取
                FYX_PREFETCH_T0(data + i + 64);
                FYX_PREFETCH_T1(data + i + 256);
                
                __m512i v0 = _mm512_loadu_si512(data + i);
                __m512i v1 = _mm512_loadu_si512(data + i + 16);
                __m512i v2 = _mm512_loadu_si512(data + i + 32);
                __m512i v3 = _mm512_loadu_si512(data + i + 48);
                
                __m512i mn01 = _mm512_min_epi32(v0, v1);
                __m512i mx01 = _mm512_max_epi32(v0, v1);
                __m512i mn23 = _mm512_min_epi32(v2, v3);
                __m512i mx23 = _mm512_max_epi32(v2, v3);
                
                vmin = _mm512_min_epi32(vmin, _mm512_min_epi32(mn01, mn23));
                vmax = _mm512_max_epi32(vmax, _mm512_max_epi32(mx01, mx23));
            }
            
            for (; i + 16 <= n; i += 16) {
                __m512i v = _mm512_loadu_si512(data + i);
                vmin = _mm512_min_epi32(vmin, v);
                vmax = _mm512_max_epi32(vmax, v);
            }
            
            T mn = _mm512_reduce_min_epi32(vmin);
            T mx = _mm512_reduce_max_epi32(vmax);
            
            for (; i < n; ++i) {
                if (data[i] < mn) mn = data[i];
                if (data[i] > mx) mx = data[i];
            }
            return {mn, mx};
        }
        else if constexpr (std::is_same_v<T, uint32_t>) {
            if (n < 16) return scalar_minmax(data, n);
            
            __m512i vmin = _mm512_loadu_si512(data);
            __m512i vmax = vmin;
            
            size_t i = 16;
            for (; i + 64 <= n; i += 64) {
                FYX_PREFETCH_T0(data + i + 64);
                FYX_PREFETCH_T1(data + i + 256);
                
                __m512i v0 = _mm512_loadu_si512(data + i);
                __m512i v1 = _mm512_loadu_si512(data + i + 16);
                __m512i v2 = _mm512_loadu_si512(data + i + 32);
                __m512i v3 = _mm512_loadu_si512(data + i + 48);
                
                __m512i mn01 = _mm512_min_epu32(v0, v1);
                __m512i mx01 = _mm512_max_epu32(v0, v1);
                __m512i mn23 = _mm512_min_epu32(v2, v3);
                __m512i mx23 = _mm512_max_epu32(v2, v3);
                
                vmin = _mm512_min_epu32(vmin, _mm512_min_epu32(mn01, mn23));
                vmax = _mm512_max_epu32(vmax, _mm512_max_epu32(mx01, mx23));
            }
            
            for (; i + 16 <= n; i += 16) {
                __m512i v = _mm512_loadu_si512(data + i);
                vmin = _mm512_min_epu32(vmin, v);
                vmax = _mm512_max_epu32(vmax, v);
            }
            
            T mn = _mm512_reduce_min_epu32(vmin);
            T mx = _mm512_reduce_max_epu32(vmax);
            
            for (; i < n; ++i) {
                if (data[i] < mn) mn = data[i];
                if (data[i] > mx) mx = data[i];
            }
            return {mn, mx};
        }
        else if constexpr (std::is_same_v<T, int64_t>) {
            if (n < 8) return scalar_minmax(data, n);
            
            __m512i vmin = _mm512_loadu_si512(data);
            __m512i vmax = vmin;
            
            size_t i = 8;
            for (; i + 32 <= n; i += 32) {
                FYX_PREFETCH_T0(data + i + 32);
                FYX_PREFETCH_T1(data + i + 128);
                
                __m512i v0 = _mm512_loadu_si512(data + i);
                __m512i v1 = _mm512_loadu_si512(data + i + 8);
                __m512i v2 = _mm512_loadu_si512(data + i + 16);
                __m512i v3 = _mm512_loadu_si512(data + i + 24);
                
                __m512i mn01 = _mm512_min_epi64(v0, v1);
                __m512i mx01 = _mm512_max_epi64(v0, v1);
                __m512i mn23 = _mm512_min_epi64(v2, v3);
                __m512i mx23 = _mm512_max_epi64(v2, v3);
                
                vmin = _mm512_min_epi64(vmin, _mm512_min_epi64(mn01, mn23));
                vmax = _mm512_max_epi64(vmax, _mm512_max_epi64(mx01, mx23));
            }
            
            for (; i + 8 <= n; i += 8) {
                __m512i v = _mm512_loadu_si512(data + i);
                vmin = _mm512_min_epi64(vmin, v);
                vmax = _mm512_max_epi64(vmax, v);
            }
            
            T mn = _mm512_reduce_min_epi64(vmin);
            T mx = _mm512_reduce_max_epi64(vmax);
            
            for (; i < n; ++i) {
                if (data[i] < mn) mn = data[i];
                if (data[i] > mx) mx = data[i];
            }
            return {mn, mx};
        }
        else if constexpr (std::is_same_v<T, uint64_t>) {
            if (n < 8) return scalar_minmax(data, n);
            
            __m512i vmin = _mm512_loadu_si512(data);
            __m512i vmax = vmin;
            
            size_t i = 8;
            for (; i + 32 <= n; i += 32) {
                FYX_PREFETCH_T0(data + i + 32);
                FYX_PREFETCH_T1(data + i + 128);
                
                __m512i v0 = _mm512_loadu_si512(data + i);
                __m512i v1 = _mm512_loadu_si512(data + i + 8);
                __m512i v2 = _mm512_loadu_si512(data + i + 16);
                __m512i v3 = _mm512_loadu_si512(data + i + 24);
                
                __m512i mn01 = _mm512_min_epu64(v0, v1);
                __m512i mx01 = _mm512_max_epu64(v0, v1);
                __m512i mn23 = _mm512_min_epu64(v2, v3);
                __m512i mx23 = _mm512_max_epu64(v2, v3);
                
                vmin = _mm512_min_epu64(vmin, _mm512_min_epu64(mn01, mn23));
                vmax = _mm512_max_epu64(vmax, _mm512_max_epu64(mx01, mx23));
            }
            
            for (; i + 8 <= n; i += 8) {
                __m512i v = _mm512_loadu_si512(data + i);
                vmin = _mm512_min_epu64(vmin, v);
                vmax = _mm512_max_epu64(vmax, v);
            }
            
            T mn = _mm512_reduce_min_epu64(vmin);
            T mx = _mm512_reduce_max_epu64(vmax);
            
            for (; i < n; ++i) {
                if (data[i] < mn) mn = data[i];
                if (data[i] > mx) mx = data[i];
            }
            return {mn, mx};
        }
        else if constexpr (std::is_same_v<T, float>) {
            if (n < 16) return scalar_minmax(data, n);
            
            __m512 vmin = _mm512_loadu_ps(data);
            __m512 vmax = vmin;
            
            size_t i = 16;
            for (; i + 64 <= n; i += 64) {
                FYX_PREFETCH_T0(data + i + 64);
                FYX_PREFETCH_T1(data + i + 256);
                
                __m512 v0 = _mm512_loadu_ps(data + i);
                __m512 v1 = _mm512_loadu_ps(data + i + 16);
                __m512 v2 = _mm512_loadu_ps(data + i + 32);
                __m512 v3 = _mm512_loadu_ps(data + i + 48);
                
                __m512 mn01 = _mm512_min_ps(v0, v1);
                __m512 mx01 = _mm512_max_ps(v0, v1);
                __m512 mn23 = _mm512_min_ps(v2, v3);
                __m512 mx23 = _mm512_max_ps(v2, v3);
                
                vmin = _mm512_min_ps(vmin, _mm512_min_ps(mn01, mn23));
                vmax = _mm512_max_ps(vmax, _mm512_max_ps(mx01, mx23));
            }
            
            for (; i + 16 <= n; i += 16) {
                __m512 v = _mm512_loadu_ps(data + i);
                vmin = _mm512_min_ps(vmin, v);
                vmax = _mm512_max_ps(vmax, v);
            }
            
            T mn = _mm512_reduce_min_ps(vmin);
            T mx = _mm512_reduce_max_ps(vmax);
            
            for (; i < n; ++i) {
                if (data[i] < mn) mn = data[i];
                if (data[i] > mx) mx = data[i];
            }
            return {mn, mx};
        }
        else if constexpr (std::is_same_v<T, double>) {
            if (n < 8) return scalar_minmax(data, n);
            
            __m512d vmin = _mm512_loadu_pd(data);
            __m512d vmax = vmin;
            
            size_t i = 8;
            for (; i + 32 <= n; i += 32) {
                FYX_PREFETCH_T0(data + i + 32);
                FYX_PREFETCH_T1(data + i + 128);
                
                __m512d v0 = _mm512_loadu_pd(data + i);
                __m512d v1 = _mm512_loadu_pd(data + i + 8);
                __m512d v2 = _mm512_loadu_pd(data + i + 16);
                __m512d v3 = _mm512_loadu_pd(data + i + 24);
                
                __m512d mn01 = _mm512_min_pd(v0, v1);
                __m512d mx01 = _mm512_max_pd(v0, v1);
                __m512d mn23 = _mm512_min_pd(v2, v3);
                __m512d mx23 = _mm512_max_pd(v2, v3);
                
                vmin = _mm512_min_pd(vmin, _mm512_min_pd(mn01, mn23));
                vmax = _mm512_max_pd(vmax, _mm512_max_pd(mx01, mx23));
            }
            
            for (; i + 8 <= n; i += 8) {
                __m512d v = _mm512_loadu_pd(data + i);
                vmin = _mm512_min_pd(vmin, v);
                vmax = _mm512_max_pd(vmax, v);
            }
            
            T mn = _mm512_reduce_min_pd(vmin);
            T mx = _mm512_reduce_max_pd(vmax);
            
            for (; i < n; ++i) {
                if (data[i] < mn) mn = data[i];
                if (data[i] > mx) mx = data[i];
            }
            return {mn, mx};
        }
        else {
            return scalar_minmax(data, n);
        }
    }
    
    // 直方图 (优化版)
    template<typename T>
    FYX_NOINLINE void histogram_512(const T* FYX_RESTRICT data, size_t n, 
                                     size_t* FYX_RESTRICT counts, int shift) noexcept {
        using M = keymap::Mapper<T>;
        constexpr size_t MASK = 255;
        
        // 使用4路并行计数减少冲突
        alignas(64) size_t local_counts[4][256] = {};
        
        size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            FYX_PREFETCH_T0(data + i + 64);
            FYX_PREFETCH_T1(data + i + 256);
            
            for (size_t j = 0; j < 32; j += 4) {
                ++local_counts[0][(M::to_key(data[i + j + 0]) >> shift) & MASK];
                ++local_counts[1][(M::to_key(data[i + j + 1]) >> shift) & MASK];
                ++local_counts[2][(M::to_key(data[i + j + 2]) >> shift) & MASK];
                ++local_counts[3][(M::to_key(data[i + j + 3]) >> shift) & MASK];
            }
        }
        
        for (; i < n; ++i) {
            ++local_counts[0][(M::to_key(data[i]) >> shift) & MASK];
        }
        
        // 合并计数
        for (size_t b = 0; b < 256; ++b) {
            counts[b] = local_counts[0][b] + local_counts[1][b] + 
                        local_counts[2][b] + local_counts[3][b];
        }
    }
    
    // 散布 (优化版)
    template<typename T>
    FYX_NOINLINE void scatter_512(const T* FYX_RESTRICT src, T* FYX_RESTRICT dst, 
                                   size_t n, size_t* FYX_RESTRICT offsets, int shift) noexcept {
        using M = keymap::Mapper<T>;
        constexpr size_t MASK = 255;
        
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            FYX_PREFETCH_T0(&src[i + 64]);
            
            T v0 = src[i+0], v1 = src[i+1], v2 = src[i+2], v3 = src[i+3];
            T v4 = src[i+4], v5 = src[i+5], v6 = src[i+6], v7 = src[i+7];
            T v8 = src[i+8], v9 = src[i+9], v10 = src[i+10], v11 = src[i+11];
            T v12 = src[i+12], v13 = src[i+13], v14 = src[i+14], v15 = src[i+15];
            
            size_t b0 = (M::to_key(v0) >> shift) & MASK;
            size_t b1 = (M::to_key(v1) >> shift) & MASK;
            size_t b2 = (M::to_key(v2) >> shift) & MASK;
            size_t b3 = (M::to_key(v3) >> shift) & MASK;
            size_t b4 = (M::to_key(v4) >> shift) & MASK;
            size_t b5 = (M::to_key(v5) >> shift) & MASK;
            size_t b6 = (M::to_key(v6) >> shift) & MASK;
            size_t b7 = (M::to_key(v7) >> shift) & MASK;
            size_t b8 = (M::to_key(v8) >> shift) & MASK;
            size_t b9 = (M::to_key(v9) >> shift) & MASK;
            size_t b10 = (M::to_key(v10) >> shift) & MASK;
            size_t b11 = (M::to_key(v11) >> shift) & MASK;
            size_t b12 = (M::to_key(v12) >> shift) & MASK;
            size_t b13 = (M::to_key(v13) >> shift) & MASK;
            size_t b14 = (M::to_key(v14) >> shift) & MASK;
            size_t b15 = (M::to_key(v15) >> shift) & MASK;
            
            // 预取目标位置
            FYX_PREFETCH_W(dst + offsets[b0]);
            FYX_PREFETCH_W(dst + offsets[b8]);
            
            dst[offsets[b0]++] = v0;
            dst[offsets[b1]++] = v1;
            dst[offsets[b2]++] = v2;
            dst[offsets[b3]++] = v3;
            dst[offsets[b4]++] = v4;
            dst[offsets[b5]++] = v5;
            dst[offsets[b6]++] = v6;
            dst[offsets[b7]++] = v7;
            dst[offsets[b8]++] = v8;
            dst[offsets[b9]++] = v9;
            dst[offsets[b10]++] = v10;
            dst[offsets[b11]++] = v11;
            dst[offsets[b12]++] = v12;
            dst[offsets[b13]++] = v13;
            dst[offsets[b14]++] = v14;
            dst[offsets[b15]++] = v15;
        }
        
        for (; i < n; ++i) {
            T v = src[i];
            size_t b = (M::to_key(v) >> shift) & MASK;
            dst[offsets[b]++] = v;
        }
    }
}
#endif // FYX_AVX512

// ═══════════════════════════════════════════════════════════════════════════
// 第十三部分: AVX2 SIMD核心
// ═══════════════════════════════════════════════════════════════════════════

#ifdef FYX_AVX2
namespace simd256 {
    FYX_INLINE __m256i sort_8xi32(__m256i v) noexcept {
        __m256i t, mn, mx;
        
        t = _mm256_shuffle_epi32(v, 0xB1);
        mn = _mm256_min_epi32(v, t);
        mx = _mm256_max_epi32(v, t);
        v = _mm256_blend_epi32(mn, mx, 0xAA);
        
        t = _mm256_shuffle_epi32(v, 0x1B);
        mn = _mm256_min_epi32(v, t);
        mx = _mm256_max_epi32(v, t);
        v = _mm256_blend_epi32(mn, mx, 0xCC);
        
        t = _mm256_shuffle_epi32(v, 0xB1);
        mn = _mm256_min_epi32(v, t);
        mx = _mm256_max_epi32(v, t);
        v = _mm256_blend_epi32(mn, mx, 0xAA);
        
        t = _mm256_permute2x128_si256(v, v, 0x01);
        t = _mm256_shuffle_epi32(t, 0x1B);
        mn = _mm256_min_epi32(v, t);
        mx = _mm256_max_epi32(v, t);
        v = _mm256_blend_epi32(mn, mx, 0xF0);
        
        t = _mm256_shuffle_epi32(v, 0x4E);
        mn = _mm256_min_epi32(v, t);
        mx = _mm256_max_epi32(v, t);
        v = _mm256_blend_epi32(mn, mx, 0xCC);
        
        t = _mm256_shuffle_epi32(v, 0xB1);
        mn = _mm256_min_epi32(v, t);
        mx = _mm256_max_epi32(v, t);
        v = _mm256_blend_epi32(mn, mx, 0xAA);
        
        return v;
    }
    
    FYX_INLINE void sort_16xi32(int32_t* arr) noexcept {
        __m256i v0 = _mm256_loadu_si256((__m256i*)arr);
        __m256i v1 = _mm256_loadu_si256((__m256i*)(arr + 8));
        
        v0 = sort_8xi32(v0);
        v1 = sort_8xi32(v1);
        
        v1 = _mm256_permute2x128_si256(v1, v1, 0x01);
        v1 = _mm256_shuffle_epi32(v1, 0x1B);
        
        __m256i mn = _mm256_min_epi32(v0, v1);
        __m256i mx = _mm256_max_epi32(v0, v1);
        v0 = mn;
        v1 = mx;
        
        auto merge8 = [](__m256i& v) {
            __m256i t, mn, mx;
            
            t = _mm256_permute2x128_si256(v, v, 0x01);
            mn = _mm256_min_epi32(v, t);
            mx = _mm256_max_epi32(v, t);
            v = _mm256_blend_epi32(mn, mx, 0xF0);
            
            t = _mm256_shuffle_epi32(v, 0x4E);
            mn = _mm256_min_epi32(v, t);
            mx = _mm256_max_epi32(v, t);
            v = _mm256_blend_epi32(mn, mx, 0xCC);
            
            t = _mm256_shuffle_epi32(v, 0xB1);
            mn = _mm256_min_epi32(v, t);
            mx = _mm256_max_epi32(v, t);
            v = _mm256_blend_epi32(mn, mx, 0xAA);
        };
        
        merge8(v0);
        merge8(v1);
        
        _mm256_storeu_si256((__m256i*)arr, v0);
        _mm256_storeu_si256((__m256i*)(arr + 8), v1);
    }
    
    FYX_INLINE void sort_32xi32(int32_t* arr) noexcept {
        sort_16xi32(arr);
        sort_16xi32(arr + 16);
        
        __m256i a0 = _mm256_loadu_si256((__m256i*)(arr));
        __m256i a1 = _mm256_loadu_si256((__m256i*)(arr + 8));
        __m256i b0 = _mm256_loadu_si256((__m256i*)(arr + 16));
        __m256i b1 = _mm256_loadu_si256((__m256i*)(arr + 24));
        
        __m256i rb0 = _mm256_permute2x128_si256(b1, b1, 0x01);
        rb0 = _mm256_shuffle_epi32(rb0, 0x1B);
        __m256i rb1 = _mm256_permute2x128_si256(b0, b0, 0x01);
        rb1 = _mm256_shuffle_epi32(rb1, 0x1B);
        
        b0 = rb0;
        b1 = rb1;
        
        __m256i mn0 = _mm256_min_epi32(a0, b0);
        __m256i mx0 = _mm256_max_epi32(a0, b0);
        __m256i mn1 = _mm256_min_epi32(a1, b1);
        __m256i mx1 = _mm256_max_epi32(a1, b1);
        
        a0 = mn0; a1 = mn1;
        b0 = mx0; b1 = mx1;
        
        auto merge8 = [](__m256i& v) {
            __m256i t, mn, mx;
            
            t = _mm256_permute2x128_si256(v, v, 0x01);
            mn = _mm256_min_epi32(v, t);
            mx = _mm256_max_epi32(v, t);
            v = _mm256_blend_epi32(mn, mx, 0xF0);
            
            t = _mm256_shuffle_epi32(v, 0x4E);
            mn = _mm256_min_epi32(v, t);
            mx = _mm256_max_epi32(v, t);
            v = _mm256_blend_epi32(mn, mx, 0xCC);
            
            t = _mm256_shuffle_epi32(v, 0xB1);
            mn = _mm256_min_epi32(v, t);
            mx = _mm256_max_epi32(v, t);
            v = _mm256_blend_epi32(mn, mx, 0xAA);
        };
        
        __m256i t0 = _mm256_permute2x128_si256(a0, a1, 0x20);
        __m256i t1 = _mm256_permute2x128_si256(a0, a1, 0x31);
        __m256i mnt = _mm256_min_epi32(t0, t1);
        __m256i mxt = _mm256_max_epi32(t0, t1);
        a0 = _mm256_permute2x128_si256(mnt, mxt, 0x20);
        a1 = _mm256_permute2x128_si256(mnt, mxt, 0x31);
        merge8(a0);
        merge8(a1);
        
        t0 = _mm256_permute2x128_si256(b0, b1, 0x20);
        t1 = _mm256_permute2x128_si256(b0, b1, 0x31);
        mnt = _mm256_min_epi32(t0, t1);
        mxt = _mm256_max_epi32(t0, t1);
        b0 = _mm256_permute2x128_si256(mnt, mxt, 0x20);
        b1 = _mm256_permute2x128_si256(mnt, mxt, 0x31);
        merge8(b0);
        merge8(b1);
        
        _mm256_storeu_si256((__m256i*)(arr), a0);
        _mm256_storeu_si256((__m256i*)(arr + 8), a1);
        _mm256_storeu_si256((__m256i*)(arr + 16), b0);
        _mm256_storeu_si256((__m256i*)(arr + 24), b1);
    }
    
    template<typename T>
    std::pair<T, T> find_minmax_256(const T* data, size_t n) noexcept {
        if (n == 0) return {T{}, T{}};
        if (n == 1) return {data[0], data[0]};
        
        if constexpr (std::is_same_v<T, int32_t>) {
            if (n < 8) {
                T mn = data[0], mx = data[0];
                for (size_t i = 1; i < n; ++i) {
                    if (data[i] < mn) mn = data[i];
                    if (data[i] > mx) mx = data[i];
                }
                return {mn, mx};
            }
            
            __m256i vmin = _mm256_loadu_si256((__m256i*)data);
            __m256i vmax = vmin;
            
            size_t i = 8;
            
            for (; i + 32 <= n; i += 32) {
                FYX_PREFETCH_T0(data + i + 64);
                FYX_PREFETCH_T1(data + i + 256);
                
                __m256i v0 = _mm256_loadu_si256((__m256i*)(data + i));
                __m256i v1 = _mm256_loadu_si256((__m256i*)(data + i + 8));
                __m256i v2 = _mm256_loadu_si256((__m256i*)(data + i + 16));
                __m256i v3 = _mm256_loadu_si256((__m256i*)(data + i + 24));
                
                __m256i mn01 = _mm256_min_epi32(v0, v1);
                __m256i mx01 = _mm256_max_epi32(v0, v1);
                __m256i mn23 = _mm256_min_epi32(v2, v3);
                __m256i mx23 = _mm256_max_epi32(v2, v3);
                
                __m256i mn0123 = _mm256_min_epi32(mn01, mn23);
                __m256i mx0123 = _mm256_max_epi32(mx01, mx23);
                
                vmin = _mm256_min_epi32(vmin, mn0123);
                vmax = _mm256_max_epi32(vmax, mx0123);
            }
            
            for (; i + 8 <= n; i += 8) {
                __m256i v = _mm256_loadu_si256((__m256i*)(data + i));
                vmin = _mm256_min_epi32(vmin, v);
                vmax = _mm256_max_epi32(vmax, v);
            }
            
            __m128i lo = _mm256_castsi256_si128(vmin);
            __m128i hi = _mm256_extracti128_si256(vmin, 1);
            lo = _mm_min_epi32(lo, hi);
            lo = _mm_min_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
            lo = _mm_min_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
            T mn = _mm_cvtsi128_si32(lo);
            
            lo = _mm256_castsi256_si128(vmax);
            hi = _mm256_extracti128_si256(vmax, 1);
            lo = _mm_max_epi32(lo, hi);
            lo = _mm_max_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
            lo = _mm_max_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
            T mx = _mm_cvtsi128_si32(lo);
            
            for (; i < n; ++i) {
                if (data[i] < mn) mn = data[i];
                if (data[i] > mx) mx = data[i];
            }
            
            return {mn, mx};
        }
        else {
            T mn = data[0], mx = data[0];
            for (size_t i = 1; i < n; ++i) {
                if (data[i] < mn) mn = data[i];
                if (data[i] > mx) mx = data[i];
            }
            return {mn, mx};
        }
    }
    
    template<typename T>
    FYX_NOINLINE void histogram_256(const T* FYX_RESTRICT data, size_t n, 
                                     size_t* FYX_RESTRICT counts, int shift) noexcept {
        using M = keymap::Mapper<T>;
        constexpr size_t MASK = 255;
        
        // 4路并行计数
        alignas(64) size_t local_counts[4][256] = {};
        
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            FYX_PREFETCH_T0(data + i + 64);
            
            for (size_t j = 0; j < 16; j += 4) {
                ++local_counts[0][(M::to_key(data[i + j + 0]) >> shift) & MASK];
                ++local_counts[1][(M::to_key(data[i + j + 1]) >> shift) & MASK];
                ++local_counts[2][(M::to_key(data[i + j + 2]) >> shift) & MASK];
                ++local_counts[3][(M::to_key(data[i + j + 3]) >> shift) & MASK];
            }
        }
        
        for (; i < n; ++i) {
            ++local_counts[0][(M::to_key(data[i]) >> shift) & MASK];
        }
        
        for (size_t b = 0; b < 256; ++b) {
            counts[b] = local_counts[0][b] + local_counts[1][b] + 
                        local_counts[2][b] + local_counts[3][b];
        }
    }
    
    template<typename T>
    FYX_NOINLINE void scatter_256(const T* FYX_RESTRICT src, T* FYX_RESTRICT dst, 
                                   size_t n, size_t* FYX_RESTRICT offsets, int shift) noexcept {
        using M = keymap::Mapper<T>;
        constexpr size_t MASK = 255;
        
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            FYX_PREFETCH_T0(&src[i + 64]);
            
            T v0 = src[i+0], v1 = src[i+1], v2 = src[i+2], v3 = src[i+3];
            T v4 = src[i+4], v5 = src[i+5], v6 = src[i+6], v7 = src[i+7];
            
            size_t b0 = (M::to_key(v0) >> shift) & MASK;
            size_t b1 = (M::to_key(v1) >> shift) & MASK;
            size_t b2 = (M::to_key(v2) >> shift) & MASK;
            size_t b3 = (M::to_key(v3) >> shift) & MASK;
            size_t b4 = (M::to_key(v4) >> shift) & MASK;
            size_t b5 = (M::to_key(v5) >> shift) & MASK;
            size_t b6 = (M::to_key(v6) >> shift) & MASK;
            size_t b7 = (M::to_key(v7) >> shift) & MASK;
            
            dst[offsets[b0]++] = v0;
            dst[offsets[b1]++] = v1;
            dst[offsets[b2]++] = v2;
            dst[offsets[b3]++] = v3;
            dst[offsets[b4]++] = v4;
            dst[offsets[b5]++] = v5;
            dst[offsets[b6]++] = v6;
            dst[offsets[b7]++] = v7;
        }
        
        for (; i < n; ++i) {
            T v = src[i];
            size_t b = (M::to_key(v) >> shift) & MASK;
            dst[offsets[b]++] = v;
        }
    }
}
#endif // FYX_AVX2

// ═══════════════════════════════════════════════════════════════════════════
// 第十四部分: 统一SIMD接口
// ═══════════════════════════════════════════════════════════════════════════

namespace simd {
    template<typename T>
    FYX_INLINE std::pair<T, T> find_minmax(const T* data, size_t n) noexcept {
#ifdef FYX_AVX512
        return simd512::find_minmax_512(data, n);
#elif defined(FYX_AVX2)
        return simd256::find_minmax_256(data, n);
#else
        if (n == 0) return {T{}, T{}};
        T mn = data[0], mx = data[0];
        for (size_t i = 1; i < n; ++i) {
            if (data[i] < mn) mn = data[i];
            if (data[i] > mx) mx = data[i];
        }
        return {mn, mx};
#endif
    }
    
    template<typename T>
    FYX_INLINE void histogram(const T* data, size_t n, size_t* counts, int shift) noexcept {
#ifdef FYX_AVX512
        simd512::histogram_512(data, n, counts, shift);
#elif defined(FYX_AVX2)
        simd256::histogram_256(data, n, counts, shift);
#else
        using M = keymap::Mapper<T>;
        for (size_t i = 0; i < n; ++i) {
            ++counts[(M::to_key(data[i]) >> shift) & 255];
        }
#endif
    }
    
    template<typename T>
    FYX_INLINE void scatter(const T* src, T* dst, size_t n, size_t* offsets, int shift) noexcept {
#ifdef FYX_AVX512
        simd512::scatter_512(src, dst, n, offsets, shift);
#elif defined(FYX_AVX2)
        simd256::scatter_256(src, dst, n, offsets, shift);
#else
        using M = keymap::Mapper<T>;
        for (size_t i = 0; i < n; ++i) {
            T v = src[i];
            size_t b = (M::to_key(v) >> shift) & 255;
            dst[offsets[b]++] = v;
        }
#endif
    }
    
    template<typename T>
    FYX_INLINE void sort_small_simd(T* arr, size_t n) noexcept {
        if constexpr (std::is_same_v<T, int32_t>) {
#ifdef FYX_AVX512
            if (n == 128) { simd512::sort_128xi32(arr); return; }
            if (n == 64) { simd512::sort_64xi32(arr); return; }
            if (n == 32) { simd512::sort_32xi32(arr); return; }
            if (n == 16) {
                __m512i v = simd512::sort_16xi32(_mm512_loadu_si512(arr));
                _mm512_storeu_si512(arr, v);
                return;
            }
#endif
#ifdef FYX_AVX2
            if (n == 32) { simd256::sort_32xi32(arr); return; }
            if (n == 16) { simd256::sort_16xi32(arr); return; }
            if (n == 8) {
                __m256i v = simd256::sort_8xi32(_mm256_loadu_si256((__m256i*)arr));
                _mm256_storeu_si256((__m256i*)arr, v);
                return;
            }
#endif
        }
        else if constexpr (std::is_same_v<T, double>) {
#ifdef FYX_AVX512
            if (n == 16) { simd512::sort_16xf64(arr); return; }
            if (n == 8) {
                __m512d v = simd512::sort_8xf64(_mm512_loadu_pd(arr));
                _mm512_storeu_pd(arr, v);
                return;
            }
#endif
        }
        else if constexpr (std::is_same_v<T, float>) {
#ifdef FYX_AVX512
            if (n == 16) {
                __m512 v = simd512::sort_16xf32(_mm512_loadu_ps(arr));
                _mm512_storeu_ps(arr, v);
                return;
            }
#endif
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 第十五部分: 排序网络
// ═══════════════════════════════════════════════════════════════════════════

namespace sortnet {
    template<typename T, typename Cmp>
    FYX_INLINE void sort2(T* a, Cmp& c) noexcept { 
        ops::cswap(a[0], a[1], c); 
    }
    
    template<typename T, typename Cmp>
    FYX_INLINE void sort3(T* a, Cmp& c) noexcept {
        ops::cswap(a[1], a[2], c); 
        ops::cswap(a[0], a[2], c); 
        ops::cswap(a[0], a[1], c);
    }
    
    template<typename T, typename Cmp>
    FYX_INLINE void sort4(T* a, Cmp& c) noexcept {
        ops::cswap(a[0], a[1], c); ops::cswap(a[2], a[3], c);
        ops::cswap(a[0], a[2], c); ops::cswap(a[1], a[3], c);
        ops::cswap(a[1], a[2], c);
    }
    
    template<typename T, typename Cmp>
    FYX_INLINE void sort5(T* a, Cmp& c) noexcept {
        ops::cswap(a[0], a[1], c); ops::cswap(a[3], a[4], c);
        ops::cswap(a[2], a[4], c); ops::cswap(a[2], a[3], c);
        ops::cswap(a[1], a[4], c); ops::cswap(a[0], a[3], c);
        ops::cswap(a[0], a[2], c); ops::cswap(a[1], a[3], c);
        ops::cswap(a[1], a[2], c);
    }
    
    template<typename T, typename Cmp>
    FYX_INLINE void sort6(T* a, Cmp& c) noexcept {
        ops::cswap(a[1], a[2], c); ops::cswap(a[4], a[5], c);
        ops::cswap(a[0], a[2], c); ops::cswap(a[3], a[5], c);
        ops::cswap(a[0], a[1], c); ops::cswap(a[3], a[4], c);
        ops::cswap(a[2], a[5], c); ops::cswap(a[0], a[3], c);
        ops::cswap(a[1], a[4], c); ops::cswap(a[2], a[4], c);
        ops::cswap(a[1], a[3], c); ops::cswap(a[2], a[3], c);
    }
    
    template<typename T, typename Cmp>
    FYX_INLINE void sort8(T* a, Cmp& c) noexcept {
        ops::cswap(a[0], a[1], c); ops::cswap(a[2], a[3], c);
        ops::cswap(a[4], a[5], c); ops::cswap(a[6], a[7], c);
        ops::cswap(a[0], a[2], c); ops::cswap(a[1], a[3], c);
        ops::cswap(a[4], a[6], c); ops::cswap(a[5], a[7], c);
        ops::cswap(a[1], a[2], c); ops::cswap(a[5], a[6], c);
        ops::cswap(a[0], a[4], c); ops::cswap(a[1], a[5], c);
        ops::cswap(a[2], a[6], c); ops::cswap(a[3], a[7], c);
        ops::cswap(a[2], a[4], c); ops::cswap(a[3], a[5], c);
        ops::cswap(a[1], a[2], c); ops::cswap(a[3], a[4], c);
        ops::cswap(a[5], a[6], c);
    }
    
    template<typename T, typename Cmp>
    void small_sort(T* a, size_t n, Cmp& c) noexcept {
        switch (n) {
            case 0: case 1: return;
            case 2: sort2(a, c); return;
            case 3: sort3(a, c); return;
            case 4: sort4(a, c); return;
            case 5: sort5(a, c); return;
            case 6: sort6(a, c); return;
            case 7: case 8: sort8(a, c); return;
            default:
                if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, double> || 
                              std::is_same_v<T, float>) {
                    if constexpr (traits::is_default_less_v<T, Cmp>) {
                        simd::sort_small_simd(a, n);
                        if (n <= 128) return;
                    }
                }
                for (size_t i = 1; i < n; ++i) {
                    T key = std::move(a[i]);
                    size_t j = i;
                    while (j > 0 && c(key, a[j-1])) {
                        a[j] = std::move(a[j-1]);
                        --j;
                    }
                    a[j] = std::move(key);
                }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 第十六部分: 插入排序
// ═══════════════════════════════════════════════════════════════════════════

namespace insertion {
    template<typename T, typename Cmp>
    FYX_NOINLINE void sort(T* a, size_t n, Cmp& cmp) noexcept {
        if (n <= 8) { 
            sortnet::small_sort(a, n, cmp); 
            return; 
        }
        
        size_t min_idx = 0;
        for (size_t i = 1; i < n; ++i) {
            if (cmp(a[i], a[min_idx])) min_idx = i;
        }
        ops::swap(a[0], a[min_idx]);
        
        for (size_t i = 2; i < n; ++i) {
            T key = std::move(a[i]);
            size_t j = i;
            while (cmp(key, a[j-1])) { 
                a[j] = std::move(a[j-1]); 
                --j; 
            }
            a[j] = std::move(key);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 第十七部分: 堆排序
// ═══════════════════════════════════════════════════════════════════════════

namespace heap {
    template<typename T, typename Cmp>
    FYX_NOINLINE void sort(T* a, size_t n, Cmp& cmp) noexcept {
        if (n <= 1) return;
        
        auto sift = [&](size_t i, size_t len) {
            while (true) {
                size_t l = 2*i+1;
                if (l >= len) break;
                size_t r = l + 1;
                size_t largest = i;
                if (cmp(a[largest], a[l])) largest = l;
                if (r < len && cmp(a[largest], a[r])) largest = r;
                if (largest == i) break;
                ops::swap(a[i], a[largest]); 
                i = largest;
            }
        };
        
        for (size_t i = n/2; i > 0; --i) sift(i-1, n);
        for (size_t i = n-1; i > 0; --i) { 
            ops::swap(a[0], a[i]); 
            sift(0, i); 
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 第十八部分: 计数排序 (修复版)
// ═══════════════════════════════════════════════════════════════════════════

namespace counting {
    template<typename T>
    bool try_sort(T* a, size_t n) noexcept {
        // 只对整数类型有效
        if constexpr (!std::is_integral_v<T>) {
            return false;
        }
        
        if (n < config::COUNTING_MIN_SIZE) return false;
        
        using U = std::make_unsigned_t<T>;
        
        auto [true_min, true_max] = simd::find_minmax(a, n);
        
        // 修复：安全的范围计算
        if (true_max < true_min) return false;
        
        // 使用更大的类型避免溢出
        uint64_t range64;
        if constexpr (std::is_signed_v<T>) {
            // 对于有符号类型，转换为无符号后计算
            // 确保不会因为符号位导致问题
            U u_min = static_cast<U>(true_min);
            U u_max = static_cast<U>(true_max);
            
            // 处理有符号到无符号的映射
            // 对于有符号整数，最小值映射后反而更大
            if (true_min < 0 && true_max >= 0) {
                // 跨越零点
                uint64_t neg_range = static_cast<uint64_t>(-static_cast<int64_t>(true_min));
                uint64_t pos_range = static_cast<uint64_t>(true_max);
                range64 = neg_range + pos_range;
            } else if (true_max < 0) {
                // 都是负数
                range64 = static_cast<uint64_t>(true_max - true_min);
            } else {
                // 都是非负
                range64 = static_cast<uint64_t>(true_max) - static_cast<uint64_t>(true_min);
            }
        } else {
            // 无符号类型直接计算
            range64 = static_cast<uint64_t>(true_max) - static_cast<uint64_t>(true_min);
        }
        
        if (range64 > config::COUNTING_MAX_RANGE) return false;
        if (range64 > static_cast<uint64_t>(n) * config::COUNTING_SORT_RATIO) return false;
        
        size_t range = static_cast<size_t>(range64) + 1;
        size_t required_mem = range * sizeof(size_t);
        if (required_mem > config::available_memory()) return false;
        
        mem::Buffer<size_t> count_buf(range);
        if (!count_buf) return false;
        
        size_t* count = count_buf.data();
        std::memset(count, 0, range * sizeof(size_t));
        
        // 计数
        for (size_t i = 0; i < n; ++i) {
            size_t idx;
            if constexpr (std::is_signed_v<T>) {
                // 有符号类型：将值映射到 [0, range)
                idx = static_cast<size_t>(static_cast<int64_t>(a[i]) - static_cast<int64_t>(true_min));
            } else {
                idx = static_cast<size_t>(a[i] - true_min);
            }
            ++count[idx];
        }
        
        // 重建数组
        size_t pos = 0;
        for (size_t i = 0; i < range; ++i) {
            T val;
            if constexpr (std::is_signed_v<T>) {
                val = static_cast<T>(static_cast<int64_t>(true_min) + static_cast<int64_t>(i));
            } else {
                val = static_cast<T>(true_min + static_cast<U>(i));
            }
            size_t c = count[i];
            while (c-- > 0) {
                a[pos++] = val; 
            }
        }
        
        return true;
    }
} // namespace counting

// ═══════════════════════════════════════════════════════════════════════════
// 第十九部分: 分层自适应基数排序 (修复版)
// ═══════════════════════════════════════════════════════════════════════════

namespace radix {
    // L1缓存优化的小块排序
    template<typename T>
    FYX_NOINLINE void sort_l1_block(T* a, size_t n) {
        using Map = keymap::Mapper<T>;
        using Key = typename Map::Key;
        
        constexpr size_t NB = 256;
        constexpr size_t KEY_BITS = sizeof(Key) * 8;
        constexpr size_t NUM_PASSES = KEY_BITS / 8;
        
        // 使用栈上缓冲区避免堆分配
        if (n > config::HIERARCHICAL_L1_THRESHOLD) {
            std::less<T> cmp;
            insertion::sort(a, n, cmp);
            return;
        }
        
        // 栈上分配小缓冲区
        alignas(64) T buffer[8192]; // 足够L1友好的大小
        if (n > 8192) {
            lsd_sort(a, n);
            return;
        }
        
        T* src = a;
        T* dst = buffer;
        
        alignas(64) size_t count[NB + 1];
        alignas(64) size_t offsets[NB];
        
        for (size_t pass = 0; pass < NUM_PASSES; ++pass) {
            int shift = static_cast<int>(pass * 8);
            
            std::memset(count, 0, sizeof(count));
            
            // 计数
            for (size_t i = 0; i < n; ++i) {
                ++count[(Map::to_key(src[i]) >> shift) & 0xFF];
            }
            
            // 检查是否需要这一轮
            size_t non_empty = 0;
            for (size_t i = 0; i < NB; ++i) {
                if (count[i] > 0) ++non_empty;
            }
            if (non_empty <= 1) continue;
            
            // 前缀和
            size_t sum = 0;
            for (size_t i = 0; i < NB; ++i) {
                size_t c = count[i];
                count[i] = sum;
                sum += c;
            }
            std::memcpy(offsets, count, sizeof(offsets));
            
            // 分发
            for (size_t i = 0; i < n; ++i) {
                size_t b = (Map::to_key(src[i]) >> shift) & 0xFF;
                dst[offsets[b]++] = src[i];
            }
            
            T* tmp = src; src = dst; dst = tmp;
        }
        
        if (src != a) {
            std::memcpy(a, src, n * sizeof(T));
        }
    }
    
    // LSD基数排序 (优化版)
    template<typename T>
    FYX_NOINLINE void lsd_sort(T* a, size_t n) {
        using Map = keymap::Mapper<T>;
        using Key = typename Map::Key;
        
        constexpr size_t RADIX = 8;
        constexpr size_t NUM_BUCKETS = 256;
        constexpr size_t KEY_BITS = sizeof(Key) * 8;
        constexpr size_t NUM_PASSES = (KEY_BITS + RADIX - 1) / RADIX;
        
        mem::Buffer<T> buffer(n);
        if (!buffer) {
            std::less<T> cmp;
            heap::sort(a, n, cmp);
            return;
        }
        
        T* src = a;
        T* dst = buffer.data();
        
        alignas(64) size_t count[NUM_BUCKETS + 1];
        alignas(64) size_t offsets[NUM_BUCKETS];
        
        for (size_t pass = 0; pass < NUM_PASSES; ++pass) {
            int shift = static_cast<int>(pass * RADIX);
            
            std::memset(count, 0, sizeof(count));
            
            simd::histogram(src, n, count, shift);
            
            // 检查是否需要这一轮
            size_t non_empty = 0;
            for (size_t i = 0; i < NUM_BUCKETS; ++i) {
                if (count[i] > 0) ++non_empty;
            }
            if (non_empty <= 1) continue;
            
            size_t sum = 0;
            for (size_t i = 0; i < NUM_BUCKETS; ++i) {
                size_t c = count[i]; 
                count[i] = sum; 
                sum += c;
            }
            
            std::memcpy(offsets, count, sizeof(offsets));
            
            simd::scatter(src, dst, n, offsets, shift);
            
            T* tmp = src; src = dst; dst = tmp;
        }
        
        if (src != a) {
            std::memcpy(a, src, n * sizeof(T));
        }
    }
    
    // MSD递归基数排序
    template<typename T>
    void msd_recursive(T* a, size_t n, int shift, mem::Buffer<T>& buf) {
        using Map = keymap::Mapper<T>;
        
        constexpr size_t NB = 256;
        
        if (n <= config::SMALL) {
            std::less<T> cmp;
            insertion::sort(a, n, cmp);
            return;
        }
        if (shift < 0) return;
        
        alignas(64) size_t count[NB + 1] = {};
        
        for (size_t i = 0; i < n; ++i) {
            ++count[(Map::to_key(a[i]) >> shift) & 0xFF];
        }
        
        size_t sum = 0;
        for (size_t i = 0; i < NB; ++i) {
            size_t c = count[i]; 
            count[i] = sum; 
            sum += c;
        }
        count[NB] = sum;
        
        alignas(64) size_t off[NB];
        std::memcpy(off, count, sizeof(off));
        
        for (size_t i = 0; i < n; ++i) {
            size_t b = (Map::to_key(a[i]) >> shift) & 0xFF;
            buf[off[b]++] = a[i];
        }
        std::memcpy(a, buf.data(), n * sizeof(T));
        
        if (shift >= 8) {
            for (size_t b = 0; b < NB; ++b) {
                size_t start = count[b];
                size_t len = count[b+1] - start;
                if (len > 1) {
                    msd_recursive(a + start, len, shift - 8, buf);
                }
            }
        }
    }
    
    template<typename T>
    void msd_sort(T* a, size_t n) {
        using Key = typename keymap::Mapper<T>::Key;
        mem::Buffer<T> buf(n);
        if (!buf) {
            std::less<T> cmp;
            heap::sort(a, n, cmp);
            return;
        }
        int start_shift = static_cast<int>(sizeof(Key) * 8 - 8);
        msd_recursive(a, n, start_shift, buf);
    }
    
    // American Flag原地基数排序 (修复版 - 防止死循环)
    template<typename T>
    FYX_NOINLINE void american_flag_sort(T* a, size_t n, int shift = -1) {
        using Map = keymap::Mapper<T>;
        using Key = typename Map::Key;
        
        if (shift < 0) {
            shift = static_cast<int>(sizeof(Key) * 8 - 8);
        }
        
        if (n <= config::SMALL || shift < 0) {
            std::less<T> cmp;
            insertion::sort(a, n, cmp);
            return;
        }
        
        constexpr size_t NB = 256;
        alignas(64) size_t count[NB] = {};
        alignas(64) size_t offset[NB + 1];
        
        // 计数
        for (size_t i = 0; i < n; ++i) {
            ++count[(Map::to_key(a[i]) >> shift) & 0xFF];
        }
        
        // 计算偏移
        offset[0] = 0;
        for (size_t i = 0; i < NB; ++i) {
            offset[i + 1] = offset[i] + count[i];
        }
        
        // 原地分区 (修复版)
        alignas(64) size_t current[NB];
        std::memcpy(current, offset, sizeof(current));
        
        for (size_t b = 0; b < NB; ++b) {
            size_t end = offset[b + 1];
            while (current[b] < end) {
                size_t bucket = (Map::to_key(a[current[b]]) >> shift) & 0xFF;
                if (bucket == b) {
                    ++current[b];
                } else {
                    // 循环交换直到当前位置放入正确元素
                    while (bucket != b) {
                        size_t target = current[bucket]++;
                        ops::swap(a[current[b]], a[target]);
                        bucket = (Map::to_key(a[current[b]]) >> shift) & 0xFF;
                    }
                    ++current[b];
                }
            }
        }
        
        // 递归处理各桶
        if (shift >= 8) {
            for (size_t b = 0; b < NB; ++b) {
                if (count[b] > 1) {
                    american_flag_sort(a + offset[b], count[b], shift - 8);
                }
            }
        }
    }
    
    // 分层自适应基数排序入口
    template<typename T>
    void hierarchical_sort(T* a, size_t n) {
        if (n <= config::SMALL) {
            std::less<T> cmp;
            insertion::sort(a, n, cmp);
            return;
        }
        
        // 尝试计数排序
        if (counting::try_sort(a, n)) return;
        
        // 根据数据大小选择最优算法
        if (n <= config::HIERARCHICAL_L1_THRESHOLD) {
            sort_l1_block(a, n);
        } else if (n <= config::HIERARCHICAL_L2_THRESHOLD) {
            lsd_sort(a, n);
        } else if (n <= config::HIERARCHICAL_L3_THRESHOLD) {
            lsd_sort(a, n);
        } else {
            msd_sort(a, n);
        }
    }
    
    // 主入口
    template<typename T>
    void sort(T* a, size_t n) {
        hierarchical_sort(a, n);
    }
} // namespace radix

// ═══════════════════════════════════════════════════════════════════════════
// 第二十部分: 无锁工作窃取队列 (修复版 - 增强线程安全)
// ═══════════════════════════════════════════════════════════════════════════

#if FYX_ENABLE_PARALLEL
namespace parallel {
    
    struct WorkItem {
        void* data = nullptr;
        size_t start = 0;
        size_t length = 0;
        int depth = 0;
        
        bool valid() const { return data != nullptr && length > 0; }
    };
    
    // Chase-Lev无锁工作窃取双端队列 (修复版)
    class LockFreeWorkStealQueue {
        static constexpr size_t INITIAL_CAPACITY = 1024;
        static constexpr size_t MAX_CAPACITY = 1 << 20;
        
        struct CircularArray {
            std::vector<std::atomic<WorkItem>> items;
            size_t mask;
            
            explicit CircularArray(size_t cap) : items(cap), mask(cap - 1) {}
            
            size_t capacity() const { return mask + 1; }
            
            WorkItem get(size_t i) const {
                return items[i & mask].load(std::memory_order_relaxed);
            }
            
            void put(size_t i, WorkItem item) {
                items[i & mask].store(item, std::memory_order_relaxed);
            }
        };
        
        alignas(config::CACHE_LINE) std::atomic<size_t> bottom{0};
        alignas(config::CACHE_LINE) std::atomic<size_t> top{0};
        alignas(config::CACHE_LINE) std::atomic<CircularArray*> array;
        
        // 修复：使用shared_ptr自动管理旧数组
        std::vector<std::shared_ptr<CircularArray>> garbage;
        mutable std::mutex garbage_mutex;
        
        CircularArray* grow(CircularArray* old_arr, size_t b, size_t t) {
            size_t new_cap = old_arr->capacity() * 2;
            if (new_cap > MAX_CAPACITY) new_cap = MAX_CAPACITY;
            
            auto new_arr = new CircularArray(new_cap);
            for (size_t i = t; i < b; ++i) {
                new_arr->put(i, old_arr->get(i));
            }
            return new_arr;
        }
        
    public:
        LockFreeWorkStealQueue() {
            array.store(new CircularArray(INITIAL_CAPACITY), std::memory_order_relaxed);
        }
        
        ~LockFreeWorkStealQueue() {
            std::lock_guard<std::mutex> lock(garbage_mutex);
            delete array.load(std::memory_order_relaxed);
            // shared_ptr自动释放garbage中的数组
        }
        
        LockFreeWorkStealQueue(const LockFreeWorkStealQueue&) = delete;
        LockFreeWorkStealQueue& operator=(const LockFreeWorkStealQueue&) = delete;
        
        // 所有者推入
        void push(WorkItem item) {
            size_t b = bottom.load(std::memory_order_relaxed);
            size_t t = top.load(std::memory_order_acquire);
            CircularArray* arr = array.load(std::memory_order_relaxed);
            
            if (b - t >= arr->capacity() - 1) {
                CircularArray* new_arr = grow(arr, b, t);
                {
                    std::lock_guard<std::mutex> lock(garbage_mutex);
                    garbage.push_back(std::shared_ptr<CircularArray>(arr));
                }
                array.store(new_arr, std::memory_order_release);
                arr = new_arr;
            }
            
            arr->put(b, item);
            std::atomic_thread_fence(std::memory_order_release);
            bottom.store(b + 1, std::memory_order_relaxed);
        }
        
        // 所有者弹出
        bool pop(WorkItem& item) {
            size_t b = bottom.load(std::memory_order_relaxed);
            if (b == 0) return false;
            
            b = b - 1;
            bottom.store(b, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            
            size_t t = top.load(std::memory_order_relaxed);
            
            if (t <= b) {
                CircularArray* arr = array.load(std::memory_order_relaxed);
                item = arr->get(b);
                
                if (t == b) {
                    if (!top.compare_exchange_strong(t, t + 1, 
                            std::memory_order_seq_cst, std::memory_order_relaxed)) {
                        bottom.store(b + 1, std::memory_order_relaxed);
                        return false;
                    }
                    bottom.store(b + 1, std::memory_order_relaxed);
                }
                return true;
            } else {
                bottom.store(b + 1, std::memory_order_relaxed);
                return false;
            }
        }
        
        // 窃取
        bool steal(WorkItem& item) {
            size_t t = top.load(std::memory_order_acquire);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            size_t b = bottom.load(std::memory_order_acquire);
            
            if (t >= b) return false;
            
            CircularArray* arr = array.load(std::memory_order_consume);
            item = arr->get(t);
            
            if (!top.compare_exchange_strong(t, t + 1,
                    std::memory_order_seq_cst, std::memory_order_relaxed)) {
                return false;
            }
            
            return true;
        }
        
        bool empty() const {
            size_t t = top.load(std::memory_order_relaxed);
            size_t b = bottom.load(std::memory_order_relaxed);
            return t >= b;
        }
        
        size_t size() const {
            size_t t = top.load(std::memory_order_relaxed);
            size_t b = bottom.load(std::memory_order_relaxed);
            return (b > t) ? (b - t) : 0;
        }
    };
    
    // 并行基数排序
    template<typename T>
    void parallel_radix(T* a, size_t n, const Options& opts) {
        size_t nt = opts.max_threads > 0 ? opts.max_threads : config::num_threads();
        if (n < opts.parallel_threshold * 2 || nt <= 1) {
            radix::sort(a, n);
            return;
        }
        
        nt = std::min(nt, config::MAX_BUCKETS);
        
        using Map = keymap::Mapper<T>;
        using Key = typename Map::Key;
        constexpr size_t NB = 256;
        constexpr int SHIFT = sizeof(Key) * 8 - 8;
        
        // 第一阶段：并行计数
        std::vector<mem::AlignedArray<size_t, NB>> local_counts(nt);
        size_t chunk = (n + nt - 1) / nt;
        std::vector<std::thread> threads;
        threads.reserve(nt);
        
        for (size_t t = 0; t < nt; ++t) {
            size_t lo = t * chunk;
            size_t hi = std::min(lo + chunk, n);
            threads.emplace_back([&, t, lo, hi]() {
                local_counts[t].zero();
                for (size_t i = lo; i < hi; ++i) {
                    ++local_counts[t][(Map::to_key(a[i]) >> SHIFT) & 0xFF];
                }
            });
        }
        for (auto& th : threads) th.join();
        threads.clear();
        
        // 合并计数
        alignas(64) size_t counts[NB + 1] = {};
        for (size_t b = 0; b < NB; ++b) {
            for (size_t t = 0; t < nt; ++t) {
                counts[b] += local_counts[t][b];
            }
        }
        
        size_t sum = 0;
        for (size_t b = 0; b < NB; ++b) {
            size_t c = counts[b]; 
            counts[b] = sum; 
            sum += c;
        }
        counts[NB] = sum;
        
        // 分配缓冲区
        mem::Buffer<T> buf(n);
        if (!buf) { 
            radix::sort(a, n); 
            return; 
        }
        
        // 计算每个线程的偏移
        std::vector<mem::AlignedArray<size_t, NB>> offsets(nt);
        for (size_t b = 0; b < NB; ++b) {
            size_t off = counts[b];
            for (size_t t = 0; t < nt; ++t) {
                offsets[t][b] = off;
                off += local_counts[t][b];
            }
        }
        
        // 第二阶段：并行分发
        for (size_t t = 0; t < nt; ++t) {
            size_t lo = t * chunk;
            size_t hi = std::min(lo + chunk, n);
            threads.emplace_back([&, t, lo, hi]() {
                for (size_t i = lo; i < hi; ++i) {
                    size_t b = (Map::to_key(a[i]) >> SHIFT) & 0xFF;
                    buf[offsets[t][b]++] = a[i];
                }
            });
        }
        for (auto& th : threads) th.join();
        threads.clear();
        
        // 第三阶段：工作窃取并行递归排序
        std::vector<LockFreeWorkStealQueue> queues(nt);
        std::atomic<bool> done{false};
        std::atomic<size_t> active_workers{0};
        
        // 初始化工作项
        size_t total_work = 0;
        for (size_t b = 0; b < NB; ++b) {
            size_t start = counts[b];
            size_t len = counts[b+1] - start;
            if (len > 1) {
                WorkItem item;
                item.data = buf.data();
                item.start = start;
                item.length = len;
                item.depth = 0;
                queues[b % nt].push(item);
                ++total_work;
            }
        }
        
        if (total_work == 0) {
            std::memcpy(a, buf.data(), n * sizeof(T));
            return;
        }
        
        // 工作线程
        auto worker = [&](size_t tid) {
            active_workers.fetch_add(1, std::memory_order_relaxed);
            WorkItem item;
            size_t steal_attempts = 0;
            
            while (!done.load(std::memory_order_relaxed)) {
                // 先尝试从自己的队列弹出
                if (queues[tid].pop(item)) {
                    steal_attempts = 0;
                    T* data = static_cast<T*>(item.data);
                    
                    if (item.length > config::WORK_STEAL_THRESHOLD && item.depth < 3) {
                        size_t mid = item.length / 2;
                        WorkItem right;
                        right.data = data;
                        right.start = item.start + mid;
                        right.length = item.length - mid;
                        right.depth = item.depth + 1;
                        queues[tid].push(right);
                        radix::sort(data + item.start, mid);
                    } else {
                        radix::sort(data + item.start, item.length);
                    }
                    continue;
                }
                
                // 尝试从其他线程窃取
                bool stolen = false;
                for (size_t i = 1; i <= nt && !stolen; ++i) {
                    size_t victim = (tid + i) % nt;
                    if (queues[victim].steal(item)) {
                        stolen = true;
                        steal_attempts = 0;
                        T* data = static_cast<T*>(item.data);
                        radix::sort(data + item.start, item.length);
                    }
                }
                
                if (!stolen) {
                    ++steal_attempts;
                    if (steal_attempts > config::MAX_STEAL_ATTEMPTS) {
                        bool all_empty = true;
                        for (size_t i = 0; i < nt; ++i) {
                            if (!queues[i].empty()) {
                                all_empty = false;
                                break;
                            }
                        }
                        if (all_empty) {
                            break;
                        }
                        steal_attempts = 0;
                    }
                    std::this_thread::yield();
                }
            }
            
            if (active_workers.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                done.store(true, std::memory_order_release);
            }
        };
        
        for (size_t t = 0; t < nt; ++t) {
            threads.emplace_back(worker, t);
        }
        for (auto& th : threads) th.join();
        
        std::memcpy(a, buf.data(), n * sizeof(T));
    }
    
    // 并行超采样排序
    template<typename T, typename Cmp>
    void parallel_supersample(T* a, size_t n, Cmp cmp, const Options& opts);
    
} // namespace parallel
#endif // FYX_ENABLE_PARALLEL

// ═══════════════════════════════════════════════════════════════════════════
// 第二十一部分: 数据分布学习与自适应选择 (修复版)
// ═══════════════════════════════════════════════════════════════════════════

namespace adaptive {
    struct DataProfile {
        bool is_sorted = false;
        bool is_reverse = false;
        bool is_nearly_sorted = false;
        bool has_many_duplicates = false;
        bool is_small_range = false;
        double entropy = 0.0;
        size_t unique_estimate = 0;
        size_t run_count = 0;
    };
    
    template<typename T, typename Cmp>
    DataProfile analyze(const T* a, size_t n, Cmp& cmp) {
        DataProfile profile;
        if (n <= 1) {
            profile.is_sorted = true;
            return profile;
        }
        
        // 采样分析
        size_t sample_size = std::min(n - 1, size_t(256));
        size_t step = (n > sample_size) ? ((n - 1) / sample_size) : 1;
        if (step == 0) step = 1;  // 修复：防止除零
        
        size_t asc = 0, desc = 0, eq = 0;
        size_t runs = 1;
        bool last_asc = true;
        
        alignas(64) size_t bucket_counts[16] = {};
        
        size_t actual_samples = 0;
        for (size_t i = 0; i < sample_size && i * step + 1 < n; ++i) {
            size_t idx = i * step;
            ++actual_samples;
            
            if (cmp(a[idx + 1], a[idx])) {
                ++desc;
                if (last_asc) { ++runs; last_asc = false; }
            } else if (cmp(a[idx], a[idx + 1])) {
                ++asc;
                if (!last_asc) { ++runs; last_asc = true; }
            } else {
                ++eq;
            }
            
            if constexpr (std::is_integral_v<T>) {
                size_t hash_val;
                if constexpr (sizeof(T) <= sizeof(size_t)) {
                    hash_val = static_cast<size_t>(a[idx]) * 2654435761ULL;
                } else {
                    hash_val = static_cast<size_t>(a[idx] >> 32) ^ static_cast<size_t>(a[idx]);
                }
                ++bucket_counts[hash_val & 0xF];
            }
        }
        
        if (actual_samples == 0) {
            profile.is_sorted = true;
            return profile;
        }
        
        profile.run_count = runs;
        
        // 判断排序状态
        if (asc == actual_samples) {
            profile.is_sorted = true;
            // 完整验证
            for (size_t i = 1; i < n; ++i) {
                if (cmp(a[i], a[i-1])) { 
                    profile.is_sorted = false; 
                    break; 
                }
            }
        } else if (desc == actual_samples) {
            profile.is_reverse = true;
            for (size_t i = 1; i < n; ++i) {
                if (cmp(a[i-1], a[i])) { 
                    profile.is_reverse = false; 
                    break; 
                }
            }
        }
        
        profile.is_nearly_sorted = (asc > actual_samples * 9 / 10) || 
                                    (desc > actual_samples * 9 / 10);
        profile.has_many_duplicates = eq > actual_samples / 4;
        
        // 计算熵
        if constexpr (std::is_integral_v<T>) {
            double total = static_cast<double>(actual_samples);
            for (size_t i = 0; i < 16; ++i) {
                if (bucket_counts[i] > 0) {
                    double p = bucket_counts[i] / total;
                    profile.entropy -= p * std::log2(p);
                }
            }
            
            size_t non_empty_buckets = 0;
            for (size_t i = 0; i < 16; ++i) {
                if (bucket_counts[i] > 0) ++non_empty_buckets;
            }
            profile.unique_estimate = non_empty_buckets * n / 16;
        }
        
        // 检查是否小范围
        if constexpr (std::is_integral_v<T>) {
            size_t check_size = std::min(n, size_t(1000));
            auto [mn, mx] = simd::find_minmax(a, check_size);
            
            // 安全计算范围
            uint64_t range;
            if constexpr (std::is_signed_v<T>) {
                if (mx >= mn) {
                    range = static_cast<uint64_t>(static_cast<int64_t>(mx) - static_cast<int64_t>(mn));
                } else {
                    range = UINT64_MAX;
                }
            } else {
                range = static_cast<uint64_t>(mx) - static_cast<uint64_t>(mn);
            }
            profile.is_small_range = range < static_cast<uint64_t>(n) * config::COUNTING_SORT_RATIO;
        }
        
        return profile;
    }
    
    enum class Algorithm {
        InsertionSort,
        HeapSort,
        PDQSort,
        MergeSort,
        RadixSort,
        CountingSort,
        AmericanFlagSort
    };
    
    template<typename T, typename Cmp>
    Algorithm select_algorithm(size_t n, const DataProfile& profile, const Options& opts) {
        if (n <= config::INSERTION_LIMIT) {
            return Algorithm::InsertionSort;
        }
        
        if (opts.force_radix && traits::is_radix_sortable_v<T>) {
            return Algorithm::RadixSort;
        }
        if (opts.force_comparison) {
            return Algorithm::PDQSort;
        }
        if (opts.stable) {
            return Algorithm::MergeSort;
        }
        
        if constexpr (traits::is_radix_sortable_v<T>) {
            if constexpr (traits::is_default_less_v<T, Cmp>) {
                if (profile.is_small_range && n >= config::COUNTING_MIN_SIZE) {
                    return Algorithm::CountingSort;
                }
                if (n >= 10000) {
                    return Algorithm::RadixSort;
                }
            }
        }
        
        if (profile.is_nearly_sorted && profile.run_count < n / 32) {
            return Algorithm::InsertionSort;
        }
        
        return Algorithm::PDQSort;
    }
} // namespace adaptive

// ═══════════════════════════════════════════════════════════════════════════
// 第二十二部分: 超标量采样分区
// ═══════════════════════════════════════════════════════════════════════════

namespace supersample {
    template<typename T, typename Cmp>
    struct Sampler {
        static constexpr size_t OVERSAMPLING = config::OVERSAMPLING_FACTOR;
        
        static void sample(const T* data, size_t n, std::vector<T>& samples, 
                          size_t num_buckets, Cmp& cmp) {
            size_t num_samples = num_buckets * OVERSAMPLING;
            samples.resize(num_samples);
            
            // 均匀采样
            for (size_t i = 0; i < num_samples; ++i) {
                size_t idx = (i * n) / num_samples;
                samples[i] = data[idx];
            }
            
            std::sort(samples.begin(), samples.end(), cmp);
            
            // 选择分位点作为枢纽
            std::vector<T> pivots(num_buckets - 1);
            for (size_t i = 0; i < num_buckets - 1; ++i) {
                pivots[i] = samples[(i + 1) * OVERSAMPLING];
            }
            samples = std::move(pivots);
        }
    };
    
    template<typename T, typename Cmp>
    struct Classifier {
        const std::vector<T>& pivots;
        Cmp& cmp;
        
        Classifier(const std::vector<T>& p, Cmp& c) : pivots(p), cmp(c) {}
        
        FYX_INLINE size_t classify(const T& val) const noexcept {
            // 对于小数量枢纽使用线性搜索（分支预测友好）
            size_t num_pivots = pivots.size();
            if (num_pivots <= 7) {
                size_t b = 0;
                while (b < num_pivots && !cmp(val, pivots[b])) ++b;
                return b;
            }
            
            // 二分搜索
            size_t lo = 0, hi = num_pivots;
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (cmp(pivots[mid], val)) lo = mid + 1;
                else hi = mid;
            }
            return lo;
        }
        
        void classify_batch(const T* data, size_t n, size_t* buckets) const noexcept {
            for (size_t i = 0; i < n; ++i) {
                buckets[i] = classify(data[i]);
            }
        }
    };
    
    template<typename T, typename Cmp>
    void partition(T* data, size_t n, size_t num_buckets, Cmp& cmp,
                   std::vector<size_t>& bucket_starts, 
                   std::vector<size_t>& bucket_sizes) {
        
        num_buckets = std::min(num_buckets, config::MAX_BUCKETS);
        
        if (num_buckets < 2) {
            bucket_starts = {0};
            bucket_sizes = {n};
            return;
        }
        
        std::vector<T> pivots;
        Sampler<T, Cmp>::sample(data, n, pivots, num_buckets, cmp);
        
        Classifier<T, Cmp> classifier(pivots, cmp);
        
        bucket_sizes.resize(num_buckets, 0);
        
        // 计数
        constexpr size_t BLOCK = 4096;
        mem::Buffer<size_t> bucket_indices(n);
        
        if (bucket_indices) {
            for (size_t i = 0; i < n; i += BLOCK) {
                size_t block_size = std::min(BLOCK, n - i);
                classifier.classify_batch(data + i, block_size, bucket_indices.data() + i);
                for (size_t j = 0; j < block_size; ++j) {
                    ++bucket_sizes[bucket_indices[i + j]];
                }
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                ++bucket_sizes[classifier.classify(data[i])];
            }
        }
        
        // 计算偏移
        bucket_starts.resize(num_buckets + 1);
        bucket_starts[0] = 0;
        for (size_t i = 0; i < num_buckets; ++i) {
            bucket_starts[i + 1] = bucket_starts[i] + bucket_sizes[i];
        }
        
        // 分发
        mem::Buffer<T> buffer(n);
        if (!buffer) {
            std::stable_sort(data, data + n, cmp);
            return;
        }
        
        std::vector<size_t> write_pos(bucket_starts.begin(), bucket_starts.end() - 1);
        
        if (bucket_indices) {
            for (size_t i = 0; i < n; ++i) {
                size_t b = bucket_indices[i];
                buffer[write_pos[b]++] = std::move(data[i]);
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                size_t b = classifier.classify(data[i]);
                buffer[write_pos[b]++] = std::move(data[i]);
            }
        }
        
        std::memcpy(data, buffer.data(), n * sizeof(T));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 第二十三部分: PDQSort
// ═══════════════════════════════════════════════════════════════════════════

namespace pdq {
    template<typename T, typename Cmp>
    FYX_INLINE T& median3(T* a, T* b, T* c, Cmp& cmp) {
        if (cmp(*b, *a)) ops::swap(*a, *b);
        if (cmp(*c, *b)) ops::swap(*b, *c);
        if (cmp(*b, *a)) ops::swap(*a, *b);
        return *b;
    }
    
    template<typename T, typename Cmp>
    FYX_INLINE T& ninther(T* a, size_t n, Cmp& cmp) {
        size_t s = n / 8;
        median3(a, a + s, a + 2*s, cmp);
        median3(a + (n/2 - s), a + n/2, a + (n/2 + s), cmp);
        median3(a + (n - 2*s - 1), a + (n - s - 1), a + (n - 1), cmp);
        return median3(a, a + n/2, a + n - 1, cmp);
    }
    
    template<typename T, typename Cmp>
    std::pair<T*, T*> partition3(T* first, T* last, Cmp& cmp) {
        size_t n = static_cast<size_t>(last - first);
        
        T* piv;
        if (n >= 128) {
            piv = &ninther(first, n, cmp);
        } else {
            piv = &median3(first, first + n/2, last - 1, cmp);
        }
        
        T pivot = *piv;
        ops::swap(*first, *piv);
        
        T* lt = first;
        T* gt = last;
        T* i = first + 1;
        
        while (i < gt) {
            if (cmp(*i, pivot)) {
                ops::swap(*lt++, *i++);
            } else if (cmp(pivot, *i)) {
                ops::swap(*i, *--gt);
            } else {
                ++i;
            }
        }
        
        return {lt, gt};
    }
    
    template<typename T, typename Cmp>
    void sort_impl(T* first, T* last, Cmp& cmp, int depth, bool left) {
        while (true) {
            size_t n = static_cast<size_t>(last - first);
            
            if (n <= config::INSERTION_LIMIT) {
                insertion::sort(first, n, cmp);
                return;
            }
            
            if (depth == 0) { 
                heap::sort(first, n, cmp); 
                return; 
            }
            --depth;
            
            auto [lt, gt] = partition3(first, last, cmp);
            size_t ln = static_cast<size_t>(lt - first);
            size_t rn = static_cast<size_t>(last - gt);
            
            if (ln < rn) { 
                sort_impl(first, lt, cmp, depth, left); 
                first = gt; 
                left = false; 
            } else { 
                sort_impl(gt, last, cmp, depth, false); 
                last = lt; 
            }
        }
    }
    
    template<typename T, typename Cmp>
    void sort(T* a, size_t n, Cmp cmp) {
        if (n <= 1) return;
        int depth = 0; 
        for (size_t m = n; m > 1; m >>= 1) ++depth;
        sort_impl(a, a + n, cmp, depth * 2, true);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 第二十四部分: 归并排序
// ═══════════════════════════════════════════════════════════════════════════

namespace merge {
    template<typename T, typename Cmp>
    void sort(T* a, size_t n, Cmp& cmp) {
        if (n <= config::SMALL) { 
            insertion::sort(a, n, cmp); 
            return; 
        }
        
        mem::Buffer<T> buf(n);
        if (!buf) { 
            pdq::sort(a, n, cmp); 
            return; 
        }
        
        for (size_t width = 1; width < n; width *= 2) {
            for (size_t i = 0; i < n; i += 2 * width) {
                size_t left = i;
                size_t mid = std::min(i + width, n);
                size_t right = std::min(i + 2 * width, n);
                
                // 已经有序则跳过
                if (mid < right && !cmp(a[mid], a[mid-1])) continue;
                
                size_t l = left, r = mid, k = left;
                
                for (size_t j = left; j < right; ++j) {
                    buf[j] = std::move(a[j]);
                }
                
                while (l < mid && r < right) {
                    if (!cmp(buf[r], buf[l])) {
                        a[k++] = std::move(buf[l++]);
                    } else {
                        a[k++] = std::move(buf[r++]);
                    }
                }
                while (l < mid) a[k++] = std::move(buf[l++]);
                while (r < right) a[k++] = std::move(buf[r++]);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 第二十五部分: 并行超采样排序实现 (修复版)
// ═══════════════════════════════════════════════════════════════════════════

#if FYX_ENABLE_PARALLEL
namespace parallel {
    template<typename T, typename Cmp>
    void parallel_supersample(T* a, size_t n, Cmp cmp, const Options& opts) {
        size_t nt = opts.max_threads > 0 ? opts.max_threads : config::num_threads();
        if (n < opts.parallel_threshold * 2 || nt <= 1) {
            pdq::sort(a, n, cmp);
            return;
        }
        
        nt = std::min(nt, config::MAX_BUCKETS);
        
        std::vector<size_t> bucket_starts, bucket_sizes;
        supersample::partition(a, n, nt, cmp, bucket_starts, bucket_sizes);
        
        // 修复：处理空桶和边界情况
        if (bucket_sizes.empty()) {
            pdq::sort(a, n, cmp);
            return;
        }
        
        // 计算有效桶数
        size_t valid_buckets = 0;
        for (size_t sz : bucket_sizes) {
            if (sz > 0) ++valid_buckets;
        }
        
        if (valid_buckets == 0) {
            return; // 数据已排序
        }
        
        // 动态调整线程数
        size_t actual_threads = std::min(nt, valid_buckets);
        actual_threads = std::max(actual_threads, size_t(1));
        
        std::vector<std::thread> threads;
        threads.reserve(actual_threads);
        
        std::atomic<size_t> next_bucket{0};
        
        auto worker = [&]() {
            while (true) {
                size_t b = next_bucket.fetch_add(1, std::memory_order_relaxed);
                if (b >= bucket_sizes.size()) break;
                
                size_t start = bucket_starts[b];
                size_t len = bucket_sizes[b];
                
                if (len > 1) {
                    pdq::sort(a + start, len, cmp);
                }
            }
        };
        
        for (size_t t = 0; t < actual_threads; ++t) {
            threads.emplace_back(worker);
        }
        
        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }
    }
} // namespace parallel
#endif // FYX_ENABLE_PARALLEL

// ═══════════════════════════════════════════════════════════════════════════
// 第二十六部分: 间接排序
// ═══════════════════════════════════════════════════════════════════════════

namespace indirect {
    template<typename T, typename Cmp>
    void sort(T* a, size_t n, Cmp& cmp) {
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), size_t(0));
        
        pdq::sort(idx.data(), idx.size(), 
            [&](size_t i, size_t j) { return cmp(a[i], a[j]); });
        
        std::vector<bool> done(n, false);
        for (size_t i = 0; i < n; ++i) {
            if (done[i] || idx[i] == i) continue;
            T tmp = std::move(a[i]);
            size_t j = i;
            while (idx[j] != i) {
                a[j] = std::move(a[idx[j]]);
                done[j] = true;
                j = idx[j];
            }
            a[j] = std::move(tmp);
            done[j] = true;
        }
    }
    
    template<typename T, typename Cmp>
    void stable_sort(T* a, size_t n, Cmp& cmp) {
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), size_t(0));
        
        std::stable_sort(idx.begin(), idx.end(), 
            [&](size_t i, size_t j) { return cmp(a[i], a[j]); });
        
        std::vector<bool> done(n, false);
        for (size_t i = 0; i < n; ++i) {
            if (done[i] || idx[i] == i) continue;
            T tmp = std::move(a[i]);
            size_t j = i;
            while (idx[j] != i) {
                a[j] = std::move(a[idx[j]]);
                done[j] = true;
                j = idx[j];
            }
            a[j] = std::move(tmp);
            done[j] = true;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 第二十七部分: 迭代器适配器
// ═══════════════════════════════════════════════════════════════════════════

namespace iter_adapter {
    template<typename It, typename Cmp>
    void sort_non_contiguous(It first, It last, Cmp cmp) {
        using T = iter_traits::value_type_t<It>;
        auto n = std::distance(first, last);
        if (n < 2) return;
        
        std::vector<T> tmp(first, last);
        
        if constexpr (traits::is_radix_sortable_v<T> && traits::is_default_less_v<T, Cmp>) {
            radix::sort(tmp.data(), static_cast<size_t>(n));
        } else {
            pdq::sort(tmp.data(), static_cast<size_t>(n), cmp);
        }
        
        std::copy(tmp.begin(), tmp.end(), first);
    }
    
    template<typename It, typename Cmp>
    void stable_sort_non_contiguous(It first, It last, Cmp cmp) {
        using T = iter_traits::value_type_t<It>;
        auto n = std::distance(first, last);
        if (n < 2) return;
        
        std::vector<T> tmp(first, last);
        merge::sort(tmp.data(), static_cast<size_t>(n), cmp);
        std::copy(tmp.begin(), tmp.end(), first);
    }
}

} // namespace detail


// ═══════════════════════════════════════════════════════════════════════════
// 第二十八部分: 主排序器 (修复版)
// ═══════════════════════════════════════════════════════════════════════════

template<typename T>
struct Sorter {
    template<typename Cmp = std::less<T>>
    static void sort(T* a, size_t n, Cmp cmp = Cmp(), const Options& opts = Options::defaults()) {
        if (n < 2) return;
        
        // 极小数组
        if (n <= config::TINY) {
            detail::sortnet::small_sort(a, n, cmp);
            return;
        }
        
        // 自适应分析
        auto profile = detail::adaptive::analyze(a, n, cmp);
        
        if (profile.is_sorted) return;
        
        // 修复：二次确认逆序
        if (profile.is_reverse) { 
            std::reverse(a, a + n);
            // 完整验证
            bool sorted = true;
            for (size_t i = 1; i < n && sorted; ++i) {
                if (cmp(a[i], a[i-1])) {
                    sorted = false;
                }
            }
            if (sorted) return;
            // 不是真正逆序，继续排序（此时数据已反转，可能需要再排）
        }
        
        // 强制比较排序
        if (opts.force_comparison) {
#if FYX_ENABLE_PARALLEL
            if (opts.parallel && n >= opts.parallel_threshold * 2 && config::num_threads() > 1) {
                detail::parallel::parallel_supersample(a, n, cmp, opts);
                return;
            }
#endif
            detail::pdq::sort(a, n, cmp);
            return;
        }
        
        // 稳定排序
        if (opts.stable) {
            if constexpr (detail::traits::use_indirect_v<T>) {
                detail::indirect::stable_sort(a, n, cmp);
            } else {
                detail::merge::sort(a, n, cmp);
            }
            return;
        }
        
        // 数值类型优化
        if constexpr (detail::traits::is_radix_sortable_v<T>) {
            if constexpr (detail::traits::is_default_less_v<T, Cmp>) {
                if (!opts.force_comparison) {
#if FYX_ENABLE_PARALLEL
                    if (opts.parallel && n >= opts.parallel_threshold * 2 && config::num_threads() > 1) {
                        detail::parallel::parallel_radix(a, n, opts);
                        return;
                    }
#endif
                    detail::radix::sort(a, n);
                    return;
                }
            }
        }
        
        // 大对象间接排序
        if constexpr (detail::traits::use_indirect_v<T>) {
            detail::indirect::sort(a, n, cmp);
            return;
        }
        
        // 并行快排
#if FYX_ENABLE_PARALLEL
        if (opts.parallel && n >= opts.parallel_threshold * 2 && config::num_threads() > 1) {
            detail::parallel::parallel_supersample(a, n, cmp, opts);
            return;
        }
#endif
        
        detail::pdq::sort(a, n, cmp);
    }
    
    template<typename Cmp = std::less<T>>
    static void stable_sort(T* a, size_t n, Cmp cmp = Cmp(), const Options& = Options::defaults()) {
        if (n < 2) return;
        if constexpr (detail::traits::use_indirect_v<T>) {
            detail::indirect::stable_sort(a, n, cmp);
        } else {
            detail::merge::sort(a, n, cmp);
        }
    }
};


// ═══════════════════════════════════════════════════════════════════════════
// 第二十九部分: 公共API
// ═══════════════════════════════════════════════════════════════════════════

// 容器排序
template<typename Container>
void sort(Container& c, const Options& opts = Options::defaults()) {
    using T = typename Container::value_type;
    if (c.size() < 2) return;
    if constexpr (detail::traits::is_contiguous_v<Container>)
        Sorter<T>::sort(c.data(), c.size(), std::less<T>{}, opts);
    else {
        std::vector<T> tmp(c.begin(), c.end());
        Sorter<T>::sort(tmp.data(), tmp.size(), std::less<T>{}, opts);
        std::copy(tmp.begin(), tmp.end(), c.begin());
    }
}

template<typename Container, typename Cmp>
void sort(Container& c, Cmp cmp, const Options& opts = Options::defaults()) {
    using T = typename Container::value_type;
    if (c.size() < 2) return;
    if constexpr (detail::traits::is_contiguous_v<Container>)
        Sorter<T>::sort(c.data(), c.size(), cmp, opts);
    else {
        std::vector<T> tmp(c.begin(), c.end());
        Sorter<T>::sort(tmp.data(), tmp.size(), cmp, opts);
        std::copy(tmp.begin(), tmp.end(), c.begin());
    }
}

// 迭代器排序
template<typename It>
void sort(It first, It last, const Options& opts = Options::defaults()) {
    using T = typename std::iterator_traits<It>::value_type;
    auto n = std::distance(first, last);
    if (n < 2) return;
    
    if constexpr (detail::iter_traits::is_contiguous_v<It> || std::is_pointer_v<It>) {
        Sorter<T>::sort(&(*first), static_cast<size_t>(n), std::less<T>{}, opts);
    } else if constexpr (detail::iter_traits::is_random_access_v<It>) {
        detail::iter_adapter::sort_non_contiguous(first, last, std::less<T>{});
    } else {
        std::vector<T> tmp(first, last);
        Sorter<T>::sort(tmp.data(), tmp.size(), std::less<T>{}, opts);
        std::copy(tmp.begin(), tmp.end(), first);
    }
}

template<typename It, typename Cmp>
void sort(It first, It last, Cmp cmp, const Options& opts = Options::defaults()) {
    using T = typename std::iterator_traits<It>::value_type;
    auto n = std::distance(first, last);
    if (n < 2) return;
    
    if constexpr (detail::iter_traits::is_contiguous_v<It> || std::is_pointer_v<It>) {
        Sorter<T>::sort(&(*first), static_cast<size_t>(n), cmp, opts);
    } else if constexpr (detail::iter_traits::is_random_access_v<It>) {
        detail::iter_adapter::sort_non_contiguous(first, last, cmp);
    } else {
        std::vector<T> tmp(first, last);
        Sorter<T>::sort(tmp.data(), tmp.size(), cmp, opts);
        std::copy(tmp.begin(), tmp.end(), first);
    }
}

// 返回排序副本
template<typename Container>
Container sorted(Container c, const Options& opts = Options::defaults()) {
    sort(c, opts); 
    return c;
}

template<typename Container, typename Cmp>
Container sorted(Container c, Cmp cmp, const Options& opts = Options::defaults()) {
    sort(c, cmp, opts); 
    return c;
}

// 稳定排序
template<typename Container>
void stable_sort(Container& c, const Options& opts = Options::defaults()) {
    using T = typename Container::value_type;
    if (c.size() < 2) return;
    if constexpr (detail::traits::is_contiguous_v<Container>)
        Sorter<T>::stable_sort(c.data(), c.size(), std::less<T>{}, opts);
    else {
        std::vector<T> tmp(c.begin(), c.end());
        Sorter<T>::stable_sort(tmp.data(), tmp.size(), std::less<T>{}, opts);
        std::copy(tmp.begin(), tmp.end(), c.begin());
    }
}

template<typename Container, typename Cmp>
void stable_sort(Container& c, Cmp cmp, const Options& opts = Options::defaults()) {
    using T = typename Container::value_type;
    if (c.size() < 2) return;
    if constexpr (detail::traits::is_contiguous_v<Container>)
        Sorter<T>::stable_sort(c.data(), c.size(), cmp, opts);
    else {
        std::vector<T> tmp(c.begin(), c.end());
        Sorter<T>::stable_sort(tmp.data(), tmp.size(), cmp, opts);
        std::copy(tmp.begin(), tmp.end(), c.begin());
    }
}

template<typename It>
void stable_sort(It first, It last, const Options& opts = Options::defaults()) {
    using T = typename std::iterator_traits<It>::value_type;
    auto n = std::distance(first, last);
    if (n < 2) return;
    
    if constexpr (std::is_pointer_v<It>) {
        Sorter<T>::stable_sort(first, static_cast<size_t>(n), std::less<T>{}, opts);
    } else {
        std::vector<T> tmp(first, last);
        Sorter<T>::stable_sort(tmp.data(), tmp.size(), std::less<T>{}, opts);
        std::copy(tmp.begin(), tmp.end(), first);
    }
}

template<typename It, typename Cmp>
void stable_sort(It first, It last, Cmp cmp, const Options& opts = Options::defaults()) {
    using T = typename std::iterator_traits<It>::value_type;
    auto n = std::distance(first, last);
    if (n < 2) return;
    
    if constexpr (std::is_pointer_v<It>) {
        Sorter<T>::stable_sort(first, static_cast<size_t>(n), cmp, opts);
    } else {
        std::vector<T> tmp(first, last);
        Sorter<T>::stable_sort(tmp.data(), tmp.size(), cmp, opts);
        std::copy(tmp.begin(), tmp.end(), first);
    }
}

// 部分排序
template<typename It>
void partial_sort(It first, It middle, It last) {
    std::partial_sort(first, middle, last);
}

template<typename It, typename Cmp>
void partial_sort(It first, It middle, It last, Cmp cmp) {
    std::partial_sort(first, middle, last, cmp);
}

// nth_element
template<typename It>
void nth_element(It first, It nth, It last) {
    std::nth_element(first, nth, last);
}

template<typename It, typename Cmp>
void nth_element(It first, It nth, It last, Cmp cmp) {
    std::nth_element(first, nth, last, cmp);
}

// 检查是否已排序
template<typename Container>
bool is_sorted(const Container& c) {
    if (c.size() <= 1) return true;
    auto it = c.begin(); 
    auto prev = *it++;
    while (it != c.end()) { 
        if (*it < prev) return false; 
        prev = *it++; 
    }
    return true;
}

template<typename Container, typename Cmp>
bool is_sorted(const Container& c, Cmp cmp) {
    if (c.size() <= 1) return true;
    auto it = c.begin(); 
    auto prev = *it++;
    while (it != c.end()) { 
        if (cmp(*it, prev)) return false; 
        prev = *it++; 
    }
    return true;
}

template<typename It>
bool is_sorted(It first, It last) {
    if (first == last) return true;
    auto prev = *first++;
    while (first != last) {
        if (*first < prev) return false;
        prev = *first++;
    }
    return true;
}

template<typename It, typename Cmp>
bool is_sorted(It first, It last, Cmp cmp) {
    if (first == last) return true;
    auto prev = *first++;
    while (first != last) {
        if (cmp(*first, prev)) return false;
        prev = *first++;
    }
    return true;
}

// argsort
template<typename Container>
std::vector<size_t> argsort(const Container& c) {
    std::vector<size_t> idx(c.size());
    std::iota(idx.begin(), idx.end(), size_t(0));
    std::sort(idx.begin(), idx.end(), 
        [&](size_t a, size_t b) { return c[a] < c[b]; });
    return idx;
}

template<typename Container, typename Cmp>
std::vector<size_t> argsort(const Container& c, Cmp cmp) {
    std::vector<size_t> idx(c.size());
    std::iota(idx.begin(), idx.end(), size_t(0));
    std::sort(idx.begin(), idx.end(), 
        [&](size_t a, size_t b) { return cmp(c[a], c[b]); });
    return idx;
}

// 按索引重排
template<typename Container>
void reorder(Container& c, const std::vector<size_t>& indices) {
    using T = typename Container::value_type;
    size_t n = c.size();
    if (n != indices.size()) return;
    
    std::vector<bool> done(n, false);
    for (size_t i = 0; i < n; ++i) {
        if (done[i] || indices[i] == i) continue;
        T tmp = std::move(c[i]);
        size_t j = i;
        while (indices[j] != i) {
            c[j] = std::move(c[indices[j]]);
            done[j] = true;
            j = indices[j];
        }
        c[j] = std::move(tmp);
        done[j] = true;
    }
}

// 唯一化
template<typename Container>
size_t unique(Container& c) {
    if (c.size() <= 1) return c.size();
    auto it = std::unique(c.begin(), c.end());
    size_t new_size = static_cast<size_t>(std::distance(c.begin(), it));
    c.resize(new_size);
    return new_size;
}

template<typename Container, typename Pred>
size_t unique(Container& c, Pred pred) {
    if (c.size() <= 1) return c.size();
    auto it = std::unique(c.begin(), c.end(), pred);
    size_t new_size = static_cast<size_t>(std::distance(c.begin(), it));
    c.resize(new_size);
    return new_size;
}

// 排序并唯一化
template<typename Container>
size_t sort_unique(Container& c, const Options& opts = Options::defaults()) {
    sort(c, opts);
    return unique(c);
}

template<typename Container, typename Cmp>
size_t sort_unique(Container& c, Cmp cmp, const Options& opts = Options::defaults()) {
    sort(c, cmp, opts);
    return unique(c, [&](const auto& a, const auto& b) { 
        return !cmp(a, b) && !cmp(b, a); 
    });
}

// 中位数
template<typename Container>
auto median(Container& c) -> typename Container::value_type {
    if (c.empty()) return typename Container::value_type{};
    size_t n = c.size();
    auto mid = c.begin() + static_cast<std::ptrdiff_t>(n / 2);
    std::nth_element(c.begin(), mid, c.end());
    if (n % 2 == 1) return *mid;
    auto mid_prev = std::max_element(c.begin(), mid);
    return (*mid_prev + *mid) / 2;
}

// 第k小元素
template<typename Container>
auto kth_element(Container& c, size_t k) -> typename Container::value_type {
    if (c.empty() || k >= c.size()) return typename Container::value_type{};
    auto it = c.begin() + static_cast<std::ptrdiff_t>(k);
    std::nth_element(c.begin(), it, c.end());
    return *it;
}

template<typename Container, typename Cmp>
auto kth_element(Container& c, size_t k, Cmp cmp) -> typename Container::value_type {
    if (c.empty() || k >= c.size()) return typename Container::value_type{};
    auto it = c.begin() + static_cast<std::ptrdiff_t>(k);
    std::nth_element(c.begin(), it, c.end(), cmp);
    return *it;
}

// 版本信息
inline const char* version() { return FYX_VERSION; }
inline int version_major() { return FYX_VERSION_MAJOR; }
inline int version_minor() { return FYX_VERSION_MINOR; }
inline int version_patch() { return FYX_VERSION_PATCH; }

} // namespace fyx

#endif // FYX_SORT_V9_HPP

// ═══════════════════════════════════════════════════════════════════════════
// 测试程序
// ═══════════════════════════════════════════════════════════════════════════

#ifdef FYX_MAIN

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <deque>

struct Large { 
    int key; 
    char data[252]; 
    bool operator<(const Large& o) const { return key < o.key; }
    bool operator==(const Large& o) const { return key == o.key; }
};

template<typename T, typename Gen>
void bench(const char* name, size_t n, Gen gen, int runs = 5) {
    double fyx_time = 0, std_time = 0;
    bool correct = true;
    std::mt19937 rng(42);
    
    for (int r = 0; r < runs; ++r) {
        std::vector<T> data(n);
        for (auto& x : data) x = gen(rng);
        auto a = data, b = data;
        
        auto t1 = std::chrono::high_resolution_clock::now();
        fyx::sort(a);
        auto t2 = std::chrono::high_resolution_clock::now();
        fyx_time += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        t1 = std::chrono::high_resolution_clock::now();
        std::sort(b.begin(), b.end());
        t2 = std::chrono::high_resolution_clock::now();
        std_time += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        if (!fyx::is_sorted(a) || a != b) correct = false;
    }
    
    double speedup = std_time / fyx_time;
    const char* status = correct ? "? OK" : "? FAIL";
    
    std::cout << std::setw(22) << name << " │ " 
              << std::setw(10) << n << " │ "
              << std::fixed << std::setprecision(2)
              << std::setw(10) << fyx_time/runs << " ms │ "
              << std::setw(10) << std_time/runs << " ms │ "
              << std::setw(7) << speedup << "x │ "
              << status << "\n";
}

int main() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ████████╗██╗   ██╗██╗  ██╗    ███████╗ ██████╗ ██████╗ ████████╗           ║
║   ██╔═════╝╚██╗ ██╔╝╚██╗██╔╝    ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝           ║
║   █████╗    ╚████╔╝  ╚███╔╝     ███████╗██║   ██║██████╔╝   ██║              ║
║   ██╔══╝     ╚██╔╝   ██╔██╗     ╚════██║██║   ██║██╔══██╗   ██║              ║
║   ██║         ██║   ██╔╝ ██╗    ███████║╚██████╔╝██║  ██║   ██║              ║
║   ╚═╝         ╚═╝   ╚═╝  ╚═╝    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝              ║
║                                                                               ║
║                        v9.0.0 - 终极高性能排序库                              ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
)" << "\n";
    
    std::cout << "系统配置:\n";
    std::cout << "  版本: " << fyx::version() << "\n";
    std::cout << "  线程数: " << fyx::config::num_threads() << "\n";
    std::cout << "  SIMD宽度: " << FYX_SIMD_WIDTH << " 字节\n";
#ifdef FYX_AVX512_FULL
    std::cout << "  SIMD: AVX-512 (完整)\n";
#elif defined(FYX_AVX512)
    std::cout << "  SIMD: AVX-512\n";
#elif defined(FYX_AVX2)
    std::cout << "  SIMD: AVX2\n";
#elif defined(FYX_SSE42)
    std::cout << "  SIMD: SSE4.2\n";
#else
    std::cout << "  SIMD: 标量\n";
#endif
    std::cout << "\n";
    
    // ═══════════════════════════════════════════════════════════════════
    // 正确性测试
    // ═══════════════════════════════════════════════════════════════════
    std::cout << "═══════════════════════ 正确性测试 ═══════════════════════\n\n";
    
    bool all_ok = true;
    std::mt19937 rng(12345);
    
    auto test = [&](const char* name, auto fn) {
        std::cout << "  " << std::setw(35) << std::left << name;
        std::cout.flush();
        bool ok = fn();
        std::cout << (ok ? "? 通过" : "? 失败") << "\n";
        all_ok &= ok;
    };
    
    test("小数组 (1-64)", [&]() {
        for (int n = 1; n <= 64; ++n) {
            std::vector<int> a(n), b;
            for (auto& x : a) x = static_cast<int>(rng() % 1000);
            b = a; fyx::sort(a); std::sort(b.begin(), b.end());
            if (a != b) return false;
        }
        return true;
    });
    
    test("SIMD排序 (16/32/64/128)", [&]() {
        for (int n : {16, 32, 64, 128}) {
            for (int iter = 0; iter < 100; ++iter) {
                std::vector<int32_t> a(n);
                for (auto& x : a) x = static_cast<int32_t>(rng());
                auto b = a;
                fyx::sort(a);
                std::sort(b.begin(), b.end());
                if (a != b) return false;
            }
        }
        return true;
    });
    
    test("大数组 (1M)", [&]() {
        std::vector<int> a(1000000);
        for (auto& x : a) x = static_cast<int>(rng());
        auto b = a; fyx::sort(a); std::sort(b.begin(), b.end());
        return a == b;
    });
    
    test("float排序", [&]() {
        std::vector<float> a(50000);
        std::uniform_real_distribution<float> dist(-1e6f, 1e6f);
        for (auto& x : a) x = dist(rng);
        auto b = a; fyx::sort(a); std::sort(b.begin(), b.end());
        return a == b;
    });
    
    test("double排序", [&]() {
        std::vector<double> a(50000);
        std::uniform_real_distribution<double> dist(-1e9, 1e9);
        for (auto& x : a) x = dist(rng);
        auto b = a; fyx::sort(a); std::sort(b.begin(), b.end());
        return a == b;
    });
    
    test("int64排序", [&]() {
        std::vector<int64_t> a(50000);
        std::uniform_int_distribution<int64_t> dist(INT64_MIN, INT64_MAX);
        for (auto& x : a) x = dist(rng);
        auto b = a; fyx::sort(a); std::sort(b.begin(), b.end());
        return a == b;
    });
    
    test("已排序", [&]() {
        std::vector<int> a(100000);
        std::iota(a.begin(), a.end(), 0);
        auto b = a; fyx::sort(a);
        return a == b;
    });
    
    test("逆序", [&]() {
        std::vector<int> a(100000);
        for (int i = 0; i < 100000; ++i) a[i] = 100000 - i;
        fyx::sort(a);
        return fyx::is_sorted(a);
    });
    
    test("全相同元素", [&]() {
        std::vector<int> a(10000, 42);
        auto b = a; fyx::sort(a);
        return a == b;
    });
    
    test("稳定排序", [&]() {
        struct S { int k, v; bool operator<(const S& o) const { return k < o.k; } };
        std::vector<S> a(5000);
        for (int i = 0; i < 5000; ++i) a[i] = {static_cast<int>(rng() % 100), i};
        fyx::stable_sort(a);
        for (size_t i = 1; i < a.size(); ++i)
            if (a[i].k == a[i-1].k && a[i].v < a[i-1].v) return false;
        return true;
    });
    
    test("并行排序", [&]() {
        std::vector<int> a(1000000);
        for (auto& x : a) x = static_cast<int>(rng());
        auto b = a; 
        fyx::sort(a, fyx::Options::defaults());
        std::sort(b.begin(), b.end());
        return a == b;
    });
    
    test("自定义比较器", [&]() {
        std::vector<int> a(10000);
        for (auto& x : a) x = static_cast<int>(rng() % 10000);
        auto b = a;
        fyx::sort(a, std::greater<int>{});
        std::sort(b.begin(), b.end(), std::greater<int>{});
        return a == b;
    });
    
    test("大对象排序", [&]() {
        std::vector<Large> a(1000);
        for (int i = 0; i < 1000; ++i) a[i].key = static_cast<int>(rng() % 10000);
        auto b = a;
        fyx::sort(a);
        std::sort(b.begin(), b.end());
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].key != b[i].key) return false;
        }
        return true;
    });
    
    test("American Flag原地排序", [&]() {
        std::vector<int> a(10000);
        for (auto& x : a) x = static_cast<int>(rng());
        auto b = a;
        fyx::detail::radix::american_flag_sort(a.data(), a.size());
        std::sort(b.begin(), b.end());
        return a == b;
    });
    
    if (!all_ok) {
        std::cout << "\n!!! 测试失败 !!!\n";
        return 1;
    }
    std::cout << "\n所有测试通过! ?\n\n";
    
    // ═══════════════════════════════════════════════════════════════════
    // 性能测试
    // ═══════════════════════════════════════════════════════════════════
    std::cout << "═══════════════════════ 性能测试 ═══════════════════════\n\n";
    std::cout << std::setw(22) << "类型" << " │ " 
              << std::setw(10) << "大小" << " │ "
              << std::setw(14) << "FYX" << " │ " 
              << std::setw(14) << "std" << " │ "
              << std::setw(9) << "加速比" << " │ 状态\n";
    std::cout << std::string(85, '─') << "\n";
    
    for (size_t n : {10000, 100000, 1000000, 10000000})
        bench<int>("int随机", n, [](auto& g) { return static_cast<int>(g()); });
    std::cout << std::string(85, '─') << "\n";
    
    for (size_t n : {10000, 100000, 1000000})
        bench<double>("double随机", n, [](auto& g) { 
            return std::uniform_real_distribution<>(-1e9, 1e9)(g); 
        });
    std::cout << std::string(85, '─') << "\n";
    
    for (size_t n : {10000, 100000, 1000000})
        bench<float>("float随机", n, [](auto& g) { 
            return std::uniform_real_distribution<float>(-1e6f, 1e6f)(g); 
        });
    std::cout << std::string(85, '─') << "\n";
    
    for (size_t n : {10000, 100000, 1000000})
        bench<int64_t>("int64随机", n, [](auto& g) { 
            return std::uniform_int_distribution<int64_t>(INT64_MIN, INT64_MAX)(g); 
        });
    std::cout << std::string(85, '─') << "\n";
    
    bench<int>("已排序", 10000000, [i=0](auto&) mutable { return i++; });
    bench<int>("逆序", 10000000, [i=10000000](auto&) mutable { return i--; });
    bench<int>("小范围%100", 1000000, [](auto& g) { return static_cast<int>(g() % 100); });
    bench<int>("小范围%1000", 1000000, [](auto& g) { return static_cast<int>(g() % 1000); });
    bench<int>("高重复%2", 1000000, [](auto& g) { return static_cast<int>(g() % 2); });
    std::cout << std::string(85, '─') << "\n";
    
    for (size_t n : {5000, 20000, 50000})
        bench<Large>("Large(256B)", n, [](auto& g) { Large l; l.key = static_cast<int>(g()); return l; });
    
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "                    测试完成！\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    return 0;
}

#endif // FYX_MAIN
