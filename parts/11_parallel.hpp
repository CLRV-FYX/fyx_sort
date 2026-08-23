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
