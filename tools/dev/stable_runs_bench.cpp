// Focused, fair benchmark for the generic stable-sort path.
//
// Measures a reconstructed previous bottom-up path, current fyx::stable_sort,
// and std::stable_sort. Inputs cover orderedness exits, low-cardinality dispatch,
// natural-run boundaries, and high-entropy data. Each sample sorts the same
// reusable work buffer; restoring input and validating output are untimed.
// Algorithm order rotates so no implementation always runs first or last.
//
//   g++ -std=c++17 -O3 -DNDEBUG -pthread -Wall -Wextra -Werror -I. tools/dev/stable_runs_bench.cpp -o /tmp/stable_runs_bench
//   /tmp/stable_runs_bench [n=1000000] [reps=21] [seed=20261001]
#include "../../fyx_sort.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

struct Record {
    std::uint32_t key;
    std::uint32_t id;
};

inline bool operator==(const Record& a, const Record& b) noexcept {
    return a.key == b.key && a.id == b.id;
}

struct ByKey {
    bool operator()(const Record& a, const Record& b) const noexcept {
        return a.key < b.key;
    }
};

// Baseline reconstructed from the former public generic fallback.
template <class T, class Comp>
void previous_bottom_up_merge(T* p, std::size_t n, Comp comp) {
    fyx::detail::StableSortBuffer<T> src(n), dst(n);
    for (std::size_t i = 0; i < n; ++i) src.emplace_back(std::move(p[i]));
    for (std::size_t width = 1; width < n;) {
        dst.clear();
        for (std::size_t i = 0; i < n;) {
            const std::size_t mid = i + std::min(width, n - i);
            const std::size_t end = mid + std::min(width, n - mid);
            std::size_t left = i, right = mid;
            while (left < mid && right < end) {
                if (comp(src[right], src[left])) dst.emplace_back(std::move(src[right++]));
                else                              dst.emplace_back(std::move(src[left++]));
            }
            while (left < mid) dst.emplace_back(std::move(src[left++]));
            while (right < end) dst.emplace_back(std::move(src[right++]));
            i = end;
        }
        src.clear();
        src.swap(dst);
        if (width > n / 2) break;
        width *= 2;
    }
    for (std::size_t i = 0; i < n; ++i) p[i] = std::move(src.data()[i]);
}

template <class Comp>
void previous_public_path(std::vector<Record>& values, Comp comp) {
    auto first = values.begin();
    auto last = values.end();
    if (fyx::detail::try_monotonic_sort(first, last, comp, false)) return;
    if (fyx::detail::try_low_cardinality_count_sort(first, last, comp)) return;
    previous_bottom_up_merge(values.data(), values.size(), comp);
}

enum class Algorithm : std::size_t { Previous, Fyx, Standard, Count };
constexpr std::size_t kAlgorithmCount = static_cast<std::size_t>(Algorithm::Count);
constexpr std::array<std::array<Algorithm, kAlgorithmCount>, 3> kOrders{{
    {{Algorithm::Previous, Algorithm::Fyx, Algorithm::Standard}},
    {{Algorithm::Fyx, Algorithm::Standard, Algorithm::Previous}},
    {{Algorithm::Standard, Algorithm::Previous, Algorithm::Fyx}}
}};

const char* algorithm_name(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Previous: return "old";
        case Algorithm::Fyx: return "fyx";
        case Algorithm::Standard: return "std";
        default: return "?";
    }
}

double run_once(Algorithm algorithm, std::vector<Record>& work,
                const std::vector<Record>& input, ByKey comp) {
    // Keep data-copy/cache setup equal for every implementation, but outside
    // the timed interval as in the usual sorting benchmark convention.
    std::copy(input.begin(), input.end(), work.begin());
    const auto begin = std::chrono::steady_clock::now();
    switch (algorithm) {
        case Algorithm::Previous:
            previous_public_path(work, comp);
            break;
        case Algorithm::Fyx:
            fyx::stable_sort(work.begin(), work.end(), comp);
            break;
        case Algorithm::Standard:
            std::stable_sort(work.begin(), work.end(), comp);
            break;
        default:
            std::abort();
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool output_matches(const std::vector<Record>& output,
                    const std::vector<Record>& reference) {
    return output == reference;
}

double quantile(const std::vector<double>& sorted, std::size_t numerator,
                std::size_t denominator) {
    const std::size_t index = ((sorted.size() - 1) * numerator) / denominator;
    return sorted[index];
}

struct Summary {
    double q1;
    double median;
    double q3;
};

Summary summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return {quantile(samples, 1, 4), quantile(samples, 1, 2),
            quantile(samples, 3, 4)};
}

struct Shape {
    const char* name;
    std::vector<Record> values;
};

std::vector<Record> make_natural_runs(std::size_t n, std::size_t run_count) {
    std::vector<Record> values(n);
    const std::size_t quotient = n / run_count;
    const std::size_t remainder = n % run_count;
    for (std::size_t run = 0; run < run_count; ++run) {
        const std::size_t begin = run * quotient + std::min(run, remainder);
        const std::size_t length = quotient + (run < remainder ? 1 : 0);
        const std::size_t end = begin + length;
        const std::size_t key_base = run * quotient / 2;
        for (std::size_t i = begin; i < end; ++i)
            values[i] = {static_cast<std::uint32_t>(key_base + i - begin),
                         static_cast<std::uint32_t>(i)};
    }
    return values;
}

std::vector<Shape> make_shapes(std::size_t n, std::uint64_t seed) {
    std::vector<Shape> shapes;
    shapes.reserve(8);

    std::vector<Record> concat(n);
    const std::size_t half = n / 2;
    for (std::size_t i = 0; i < half; ++i)
        concat[i] = {static_cast<std::uint32_t>(i + half / 2),
                     static_cast<std::uint32_t>(i)};
    for (std::size_t i = half; i < n; ++i)
        concat[i] = {static_cast<std::uint32_t>(i - half),
                     static_cast<std::uint32_t>(i)};
    shapes.push_back({"concat2", std::move(concat)});
    shapes.push_back({"runs64", make_natural_runs(n, 64)});
    shapes.push_back({"runs65", make_natural_runs(n, 65)});

    std::vector<Record> sorted(n), reverse(n), equal(n), random(n), random8(n);
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < n; ++i) {
        const auto id = static_cast<std::uint32_t>(i);
        sorted[i] = {static_cast<std::uint32_t>(i), id};
        reverse[i] = {static_cast<std::uint32_t>(n - i), id};
        equal[i] = {0u, id};
        random[i] = {static_cast<std::uint32_t>(rng()), id};
        random8[i] = {static_cast<std::uint32_t>(rng() & 7u), id};
    }
    shapes.push_back({"sorted", std::move(sorted)});
    shapes.push_back({"reverse", std::move(reverse)});
    shapes.push_back({"equal", std::move(equal)});
    shapes.push_back({"random8", std::move(random8)});
    shapes.push_back({"random", std::move(random)});
    return shapes;
}

void benchmark(const Shape& shape, int reps) {
    const ByKey comp{};
    std::vector<Record> work(shape.values.size());
    std::vector<Record> reference = shape.values;
    std::stable_sort(reference.begin(), reference.end(), comp);

    std::array<std::vector<double>, kAlgorithmCount> samples;
    for (auto& values : samples) values.reserve(static_cast<std::size_t>(reps));

    // One untimed warmup per implementation reduces first-use and cold-page
    // effects. Use the same rotating schedule as the measured repetitions.
    for (const Algorithm algorithm : kOrders[0]) {
        run_once(algorithm, work, shape.values, comp);
        if (!output_matches(work, reference)) {
            std::fprintf(stderr, "%s: %s warmup mismatch\n", shape.name,
                         algorithm_name(algorithm));
            std::exit(2);
        }
    }

    for (int repetition = 0; repetition < reps; ++repetition) {
        const auto& order = kOrders[static_cast<std::size_t>(repetition) % kOrders.size()];
        for (const Algorithm algorithm : order) {
            const double elapsed = run_once(algorithm, work, shape.values, comp);
            samples[static_cast<std::size_t>(algorithm)].push_back(elapsed);
            if (!output_matches(work, reference)) {
                std::fprintf(stderr, "%s: %s output mismatch at repetition %d\n",
                             shape.name, algorithm_name(algorithm), repetition);
                std::exit(2);
            }
        }
    }

    std::array<Summary, kAlgorithmCount> summary;
    for (std::size_t i = 0; i < kAlgorithmCount; ++i)
        summary[i] = summarize(std::move(samples[i]));
    const auto& old = summary[static_cast<std::size_t>(Algorithm::Previous)];
    const auto& fyx = summary[static_cast<std::size_t>(Algorithm::Fyx)];
    const auto& standard = summary[static_cast<std::size_t>(Algorithm::Standard)];
    std::printf("%-8s n=%zu reps=%d | old %.3f [%.3f,%.3f] | fyx %.3f [%.3f,%.3f]"
                " | std %.3f [%.3f,%.3f] | old/fyx %.3fx std/fyx %.3fx\n",
                shape.name, shape.values.size(), reps,
                old.median, old.q1, old.q3, fyx.median, fyx.q1, fyx.q3,
                standard.median, standard.q1, standard.q3,
                old.median / fyx.median, standard.median / fyx.median);
}

bool parse_u64(const char* text, std::uint64_t& value) {
    if (!text || !*text || *text == '-') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return false;
    value = static_cast<std::uint64_t>(parsed);
    return true;
}

int main(int argc, char** argv) {
    std::uint64_t parsed_n = 1000000;
    std::uint64_t parsed_reps = 21;
    std::uint64_t seed = 20261001;
    if ((argc > 1 && !parse_u64(argv[1], parsed_n)) ||
        (argc > 2 && !parse_u64(argv[2], parsed_reps)) ||
        (argc > 3 && !parse_u64(argv[3], seed)) || argc > 4 ||
        parsed_n < 8192 || parsed_n > std::numeric_limits<std::uint32_t>::max() ||
        parsed_reps < 3 || parsed_reps > 10000) {
        std::fprintf(stderr,
                     "usage: stable_runs_bench [n=8192..UINT32_MAX] [reps=3..10000] [seed]\n");
        return 2;
    }

    const auto n = static_cast<std::size_t>(parsed_n);
    const auto reps = static_cast<int>(parsed_reps);
    const auto shapes = make_shapes(n, seed);
    std::printf("stable_sort benchmark: n=%zu reps=%d seed=%llu, copies/validation untimed; "
                "schedule=rotating; warmups=1 per algorithm and shape\n",
                n, reps, static_cast<unsigned long long>(seed));
    for (const auto& shape : shapes) benchmark(shape, reps);
}
