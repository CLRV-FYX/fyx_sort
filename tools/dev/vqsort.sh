#!/usr/bin/env bash
# Rebuild the comparison matrix with google/highway vqsort in it.
#
# bench/bench_matrix.cpp already knows how to call vqsort (and ips4o, and
# orlp/pdqsort); it just needs the headers, which are not in the repository.
# Everything this script fetches goes under build/ (gitignored, and not
# persisted by the sandbox), so re-run it after a re-provision.
set -e
cd "$(dirname "$0")/../.."            # repo root

mkdir -p third_party build
[ -d third_party/ips4o/.git ]   || git clone --depth 1 https://github.com/ips4o/ips4o.git third_party/ips4o
[ -d third_party/pdqsort/.git ] || git clone --depth 1 https://github.com/orlp/pdqsort.git third_party/pdqsort
[ -d build/highway/.git ]       || git clone --depth 1 --branch 1.4.0 https://github.com/google/highway.git build/highway

if ! command -v cmake >/dev/null 2>&1; then
    python3 -m pip install --quiet --break-system-packages cmake
    export PATH="$HOME/.local/bin:$PATH"
fi

cmake -S build/highway -B build/highway/build -DCMAKE_BUILD_TYPE=Release \
      -DHWY_ENABLE_TESTS=OFF -DHWY_ENABLE_EXAMPLES=OFF -DHWY_ENABLE_CONTRIB=ON
cmake --build build/highway/build -j"$(nproc)"

g++ -std=c++17 -O3 -march=native -DNDEBUG -pthread \
    -DFYX_MATRIX_HAVE_VQSORT -DFYX_MATRIX_HAVE_IPS4O -DFYX_MATRIX_HAVE_PDQ \
    -I. -Ibuild/highway -Ithird_party/ips4o -Ithird_party/pdqsort \
    bench/bench_matrix.cpp -o build/bench_matrix \
    -Lbuild/highway/build -lhwy_contrib -lhwy -lpthread -ldl

N=${1:-1000000}
./build/bench_matrix --size="$N" --reps=3 | tee "build/vqsort_${N}.txt"
