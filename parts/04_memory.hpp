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
