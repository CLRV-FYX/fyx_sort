// Component timings for the prototype vectorised quicksort: one partition
// pass, and the leaf network, measured separately per type.
//   g++ -std=c++17 -O3 -march=native -pthread -DNDEBUG tools/dev/vqs_parts.cpp -o /tmp/vqsp
#define main proto_main_unused
#include "vqs.cpp"
#undef main

#include <cstdio>

template <class P>
static void component(const char* name, std::size_t n) {
    using T = typename P::T;
    std::vector<T> src = make_random<T>(n, 7);
    std::vector<T> work(n);

    // one partition pass over the whole array
    double best_part = 1e30;
    for (int i = 0; i < 5; ++i) {
        work = src;
        const T pivot = proto::pick_pivot<P, 2>(work.data(), n);
        const double t0 = now();
        T lo, hi;
        volatile std::size_t s = proto::partition<P>(work.data(), n, pivot, lo, hi);
        (void)s;
        best_part = std::min(best_part, now() - t0);
    }

    // leaf network over 64-element blocks
    for (int base : {64, 128, 256}) {
        double best_leaf = 1e30;
        for (int i = 0; i < 5; ++i) {
            work = src;
            const double t0 = now();
            for (std::size_t off = 0; off + base <= n; off += base)
                proto::net_sort<P>(work.data() + off, static_cast<std::size_t>(base));
            best_leaf = std::min(best_leaf, now() - t0);
        }
        std::printf("%-4s n=%zu  partition %.4f (%.2f ns/elem)   leaf%-4d %.4f (%.2f ns/elem)\n",
                    name, n, best_part, best_part / n * 1e9, base,
                    best_leaf, best_leaf / n * 1e9);
    }
}

int main(int argc, char** argv) {
    const std::size_t n = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 8000000;
    component<proto::I32>("i32", n);
    component<proto::I64>("i64", n);
    component<proto::F32>("f32", n);
    component<proto::F64>("f64", n);
    return 0;
}
