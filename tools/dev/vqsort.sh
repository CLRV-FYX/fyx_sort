#!/usr/bin/env bash
# Rebuild the comparison matrix with every opponent wired in.
#
# bench/bench_matrix.cpp already knows how to call std::sort, orlp/pdqsort,
# google/highway vqsort, IPS4o (serial and through oneTBB) and
# intel/x86-simd-sort; it just needs the sources, which are not in the
# repository.  Everything fetched here goes under third_party/ (gitignored --
# clone it again after a re-provision) and the binaries/results under build/.
#
#   bash tools/dev/vqsort.sh 1000000
#   bash tools/dev/vqsort.sh 8000000
#
# Without oneTBB the IPS4o parallel column is dropped rather than faked: the
# script builds a reduced matrix instead of a wrong one.
set -e
cd "$(dirname "$0")/../.."            # repo root

mkdir -p third_party build
PIP_TARGET=""
if ! command -v cmake >/dev/null 2>&1; then
    python3 -m pip install --quiet --break-system-packages cmake
    export PATH="$HOME/.local/bin:$PATH"
fi

clone() {                              # clone <url> <dir> [branch]
    [ -d "$2/.git" ] && return 0
    if [ -n "${3:-}" ]; then git clone --depth 1 --branch "$3" "$1" "$2"
    else                     git clone --depth 1 "$1" "$2"; fi
}

clone https://github.com/ips4o/ips4o.git                 third_party/ips4o
clone https://github.com/orlp/pdqsort.git                third_party/pdqsort
clone https://github.com/google/highway.git              third_party/highway 1.4.0
clone https://github.com/intel/x86-simd-sort.git         third_party/x86-simd-sort
clone https://github.com/uxlfoundation/oneTBB.git        third_party/oneTBB

# --- google/highway (vqsort lives in the contrib library) -------------------
if [ ! -f third_party/highway/build/libhwy_contrib.a ]; then
    cmake -S third_party/highway -B third_party/highway/build \
          -DCMAKE_BUILD_TYPE=Release -DHWY_ENABLE_TESTS=OFF \
          -DHWY_ENABLE_EXAMPLES=OFF -DBUILD_TESTING=OFF -DHWY_ENABLE_CONTRIB=ON
    cmake --build third_party/highway/build -j"$(nproc)"
fi

# --- oneTBB (IPS4o's parallel scheduler) -----------------------------------
TBB_PREFIX="third_party/tbb-install"
if [ ! -f "$TBB_PREFIX/lib/libtbb.so" ]; then
    cmake -S third_party/oneTBB -B third_party/oneTBB/build \
          -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DTBB_EXAMPLES=OFF \
          -DTBB_STRICT=OFF -DCMAKE_INSTALL_PREFIX="$PWD/$TBB_PREFIX"
    cmake --build third_party/oneTBB/build -j"$(nproc)"
    cmake --install third_party/oneTBB/build
fi

# ips4o keeps its header in include/ nowadays, older checkouts at the root
IPS4O_INC=third_party/ips4o/include
[ -f "$IPS4O_INC/ips4o.hpp" ] || IPS4O_INC=third_party/ips4o

COMMON="-std=c++17 -O3 -march=native -DNDEBUG -pthread -I. \
        -Ithird_party/pdqsort -Ithird_party/highway -Ithird_party/x86-simd-sort/src \
        -I$IPS4O_INC"
# x86-simd-sort is used through its header-only "static-incl" translation unit,
# so no library build is needed for it.
TBB_FLAGS="-Ithird_party/tbb-install/include"
TBB_LINK="-Lthird_party/tbb-install/lib -Wl,-rpath,$PWD/$TBB_PREFIX/lib -ltbb -latomic"
ALWAYS="-DFYX_MATRIX_HAVE_PDQ -DFYX_MATRIX_HAVE_VQSORT -DFYX_MATRIX_HAVE_XSS"
HWY_LINK="-Lthird_party/highway/build -lhwy_contrib -lhwy"
IPS4O_SEQ="-DFYX_MATRIX_HAVE_IPS4O"

build_matrix() {                       # build_matrix <extra defines...>
    g++ $COMMON $ALWAYS "$@" $TBB_FLAGS bench/bench_matrix.cpp \
        -o build/bench_matrix $HWY_LINK $TBB_LINK -lpthread -ldl
}

if build_matrix $IPS4O_SEQ -DFYX_MATRIX_HAVE_IPS4O_PARALLEL 2>/dev/null; then
    echo "  matrix: std::sort + pdqsort + vqsort + IPS4o (serial/parallel) + x86-simd-sort"
else
    echo "  (IPS4o parallel needs oneTBB here -- IPS4o stays serial-only)"
    build_matrix $IPS4O_SEQ
fi

N=${1:-1000000}
REPS=${2:-}
[ -n "$REPS" ] || { [ "$N" -ge 4000000 ] && REPS=5 || REPS=7; }
./build/bench_matrix --size="$N" --reps="$REPS" | tee "build/vqsort_${N}.txt"
